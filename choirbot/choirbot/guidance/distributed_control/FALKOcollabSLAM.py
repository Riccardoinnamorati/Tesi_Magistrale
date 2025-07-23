import numpy as np
from ..guidance import Guidance
from sensor_msgs.msg import Image, LaserScan
import cv2
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

from choirbot_examples.FALKO.FALKO_algorithm import FALKOKeypointDetector
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import math
from choirbot_interfaces.msg import KeypointArray, KeypointMatch
import matplotlib.pyplot as plt
import casadi as ca

from ...communicator import TimeVaryingCommunicator

import time

class FALKOCollabSLAMGuidance(Guidance):
    """
    Collaborative Mapping and Localization using FALKO algorithm.
    This class extends the Guidance class to implement a distributed SLAM system
    using the FALKO algorithm. It handles the collection of odometry data,
    laser scans, and keypoint detection, and communicates with other agents
    in the network to share localization information.
    """

    def __init__(self, pose_handler: str=None, pose_topic: str=None, sampling_freq: float=0.5, receive_freq: float=10.0, keypoints_frequency: float=1.0, num_agent: int=5):
        super().__init__(pose_handler, pose_topic)


        self.sampling_freq = sampling_freq
        self.receive_freq = receive_freq
        self.keypoints_freq = keypoints_frequency # Frequency for keypoint detection
        self.num_agent = num_agent

        self.sampling_timer = self.create_timer(1.0/self.sampling_freq, self.sample_odom)
        self.keypoint_timer = self.create_timer(1.0/self.keypoints_freq, self.detect_keypoints)
        self.get_logger().info('CollabSLAMGuidance {} started'.format(self.agent_id))

        self.falko = FALKOKeypointDetector()

        # Noisy measurement
        self.noisy = True
        self.theta = 0.0

        # Subscribe to laser scan topic
        self.laser_sub = self.create_subscription(
            LaserScan, f'/agent_{self.agent_id}/agent_{self.agent_id}/LDS_01', self.laser_callback, 10)
        
        for i in self.in_neighbors:
            self.match_sub = self.create_subscription(
                KeypointMatch, 
                f'/matches_agent_{i}',
                self.falko_match_callback,
                10
            )

        self.marker_pub = self.create_publisher(
            MarkerArray, f'agent_{self.agent_id}/keypoint_markers', 10)
        self.latest_scan = None

        # Aruco data
        self.aruco_data = []    # Sampled aruco (if available)
        self.falko_matches = {}  # Store FALKO matches from other agents

        # Localization data
        self.vertices = []      # List of sampled location -> [[x_0, y_0, theta_0], [], ..., []]
        self.edges = []          # List of sampled edge  between vertices -> [[dx, dy, dtheta, covariance], [], ..., []]
        self.previous_point = 0

        # Optimization data
        self.separators = {agent:{} for agent in self.in_neighbors}
        self.separator_counters = [0 for _ in self.in_neighbors]

        # Set Casadi options
        self.casadi_max_iters = 10
        self.casadi_opt = {
            "ipopt.max_iter": 3,     # Set the maximum number of iterations to 100
            "print_time": False,     # Optional: To prevent printing time statistics'print_level': 0
            "ipopt.print_level": 0,  # Optional: To reduce the verbosity
        }


        # Create publisher to centralized plotter
        self.plot_results = False
        self.publisher_plotter = self.create_publisher(Float64MultiArray, f"agent{self.agent_id}_plot", 10)
        
        # Publisher for keypoints
        self.keypoint_pub = self.create_publisher(
            KeypointArray, f'/agent_{self.agent_id}/keypoints', 10)

    def compute_measurements(self, this_location, that_location):

        this_x, this_y, this_theta = this_location
        that_x, that_y, that_theta = that_location

        dx = np.cos(-this_theta) * (that_x - this_x) - np.sin(-this_theta) * (that_y - this_y)
        dy = np.sin(-this_theta) * (that_x - this_x) + np.cos(-this_theta) * (that_y - this_y)
        dtheta = that_theta - this_theta

        # Normalize dtheta to [-pi, pi]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        # Add edge with a simple example covariance (triangular matrix values)
        # Values comes from https://github.com/itzsid/admm-slam
        covariance = [44.72135955, 0, 0, 44.72135955, 0, 44.72135955]

        return [dx, dy, dtheta, covariance]


    def falko_match_callback(self, msg):
        """
        Callback function for FALKO keypoint matches.
        
        This function processes incoming FALKO matches and stores them for
        later use in separator creation during ADMM optimization.
        
        Args:
            msg: KeypointMatch message containing match data
        """
        print(f'Received FALKO match from agent {msg.agent_id} to agent {msg.neigh_id}')
        # Extract match information

        if msg.agent_id == self.agent_id:
            neighbor_id = msg.neigh_id
            dx, dy, dz = msg.translation  # dz is typically 0 for 2D case
            dtheta = msg.theta
            
            # Store the match information
            self.falko_matches[neighbor_id] = {
                'translation': [dx, dy, dz],
                'rotation': dtheta,
                'agent_keypoints': msg.agent_keypoints,
                'neigh_keypoints': msg.neigh_keypoints,
                'timestamp': time.time()
            }
        elif msg.neigh_id == self.agent_id:
            neighbor_id = msg.agent_id

            dx = -(math.cos(msg.theta) * msg.translation[0] + math.sin(msg.theta) * msg.translation[1])
            dy = -(-math.sin(msg.theta) * msg.translation[0] + math.cos(msg.theta) * msg.translation[1])
            dz = msg.translation[2]  # dz is typically 0 for 2D case
            dtheta = -msg.theta

            # Store the match information
            self.falko_matches[neighbor_id] = {
                'translation': [dx, dy, dz],
                'rotation': dtheta,
                'agent_keypoints': msg.neigh_keypoints,
                'neigh_keypoints': msg.agent_keypoints,
                'timestamp': time.time()
            }

    def create_separators_from_falko(self):
        """
        Create separators for ADMM optimization using FALKO matches.
        
        This function replaces the ArUco-based separator creation logic.
        It uses FALKO matches to establish connections between agents and
        creates the necessary separator structures for distributed optimization.
        """
        for neighbor_id, match_data in self.falko_matches.items():
            # Get the relative transformation from FALKO match
            dx, dy, dz = match_data['translation']
            dtheta = match_data['rotation']
            
            # Create measurement data in the format expected by ADMM
            covariance = [44.72135955, 0, 0, 44.72135955, 0, 44.72135955]
        
            measurements = [dx, dy, dtheta, covariance]
            
            # Current vertex ID (the vertex where this match was observed)
            current_vertex_id = match_data['agent_keypoints'].vertex_index
            
            neighbor_pose = [match_data['neigh_keypoints'].agent_position.x,
                            match_data['neigh_keypoints'].agent_position.y,
                            match_data['neigh_keypoints'].agent_position.theta] # [x, y, theta]
            # print(f'Creating separator for neighbor {neighbor_id} in {neighbor_pose} and me in {match_data["agent_keypoints"].agent_position}')

            # Create separator entry
            # Using neighbor_id as the separator key (replaces ArUco ID)
            separator_key = self.separator_counters[neighbor_id]
            self.separator_counters[neighbor_id] += 1
    
            
            self.separators[neighbor_id][separator_key] = (
                measurements,           # [dx, dy, dtheta, covariance]
                current_vertex_id,      # This agent's vertex ID
                neighbor_pose          # Neighbor's pose [x, y, theta]
            )

    def sample_odom(self):
        # skip if position is not available yet
        if self.current_pose.position is None:
            return

        _, _, self.theta = R.from_quat(self.current_pose.orientation).as_euler('xyz')

        if self.noisy and (self.agent_id == 0 or self.agent_id == 1):
            self.current_pose.position[0] += 0.5
            # self.theta += 0.1
        
        self.vertices.append([self.current_pose.position[0], self.current_pose.position[1], self.theta])

        if len(self.vertices) == 1:
            # Save first measurement (then used only for agent_0)
            self.prior = self.vertices[0]

            # Plot settings
            if self.plot_results:
                plt.figure(figsize=(2, 2))


        if len(self.vertices) > 1:
            self.edges.append(self.compute_measurements(self.previous_point, self.vertices[-1]))
        self.previous_point = self.vertices[-1]

        self.optimize_graph()

        print(f'Agent {self.agent_id} first vertex: {self.vertices[0]}')

    def optimize_graph(self):

        if self.falko_matches:
            # self.get_logger().info(f'Agent {self.agent_id} has {len(self.falko_matches)} FALKO matches')
            self.create_separators_from_falko()

        if len(self.vertices) == 1:
            return

        admm = ADMM(self.casadi_opt, self.casadi_max_iters, self.edges, self.vertices, self.prior, self.separators, self.agent_id)
        
        
        self.vertices = admm.optimize()


        # Send to centralized plotter
        array = []
        array.append(float(self.agent_id))
        msg = Float64MultiArray()
        msg.data = array + [item for sublist in self.vertices for item in sublist]
        self.publisher_plotter.publish(msg)

        if self.plot_results:
            self.plotter()


    def plotter(self):

        x_vals = []
        y_vals = []

        # Plot trajectory
        for i in range(len(self.vertices)):
            pose_data = self.vertices[i]

            # Extract x, y, and theta values
            x = pose_data[0]
            x_vals.append(pose_data[0])
            y = pose_data[1]
            y_vals.append(pose_data[1])
            theta = pose_data[2]

            # Add arrows to show direction (pose orientation)
            dx = np.cos(theta) * 0.1  # Arrow length for orientation
            dy = np.sin(theta) * 0.1
            plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1, fc='red', ec='red')
        
        plt.plot(x_vals, y_vals, 'o-', label="Trajectory", markersize=5)

        #plt.margins(x=10, y=10)
        plt.title(f"Agent {self.agent_id}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling for both axes

        plt.pause(0.001)
        plt.clf()

    def keypoints_msg_to_markers(self, keypoints_msg):
        """Converte KeypointArray in MarkerArray per RViz"""
        marker_array = MarkerArray()
        
        # clear previous markers
        marker_array.markers.clear()

        # Use keypoints_msg header to determine the target frame
        target_frame = keypoints_msg.header.frame_id
        
        for i, keypoint in enumerate(keypoints_msg.keypoints):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = target_frame  # LaserScan frame
            
            marker.ns = f'keypoints_agent_{self.agent_id}'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position = keypoint.position
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Colore distintivo per agente
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            color = colors[self.agent_id % len(colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)

            # Arrow per orientazione
            arrow = Marker()
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.header.frame_id = target_frame
            arrow.ns = f'orientations_agent_{self.agent_id}'
            arrow.id = i + 1000  # ID diverso per evitare conflitti
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            
            start_point = keypoint.position
            end_point = Point()
            end_point.x = start_point.x + 0.3 * math.cos(keypoint.orientation)
            end_point.y = start_point.y + 0.3 * math.sin(keypoint.orientation)
            end_point.z = 0.0
            
            arrow.points = [start_point, end_point]
            arrow.scale.x = 0.02
            arrow.scale.y = 0.05
            arrow.scale.z = 0.05
            arrow.color = marker.color
            arrow.color.a = 0.7
            
            marker_array.markers.append(arrow)
    
        return marker_array

    def detect_keypoints(self):
        if self.latest_scan is None:
            return
        
        keypoints, _ = self.falko.extract_keypoints(self.latest_scan)
        # print(f'Agent {self.agent_id} detected {len(keypoints.keypoints)} keypoints')
        # add agent_id to keypoints to identify the source later in the matching process
        keypoints.agent_id = self.agent_id
        
        # Add current robot position to keypoints message
        keypoints.agent_position.x = float(self.current_pose.position[0]) if self.current_pose.position is not None else 0.0
        keypoints.agent_position.y = float(self.current_pose.position[1]) if self.current_pose.position is not None else 0.0
        keypoints.agent_position.theta = self.theta
        if len(self.vertices) > 0:
            # Set the vertex index to the last vertex
            keypoints.vertex_index = len(self.vertices) - 1
        else:
            keypoints.vertex_index = 0
        self.keypoint_pub.publish(keypoints)
    
        markers_msg = self.keypoints_msg_to_markers(keypoints)
        self.marker_pub.publish(markers_msg)

    def laser_callback(self, msg):
        self.latest_scan = msg

class ADMM:
    def __init__(self, opt, max_iters, edges, vertex, prior, separators, graph):

        self.max_iters = max_iters
        self.opt = opt

        self.vertex = vertex
        self.edges = edges

        self.v_index = [i for i in range(len(self.vertex))]
        self.e_index = [(i,i+1) for i in range((len(self.vertex) - 1))]
        
        self.prior = prior
        self.separators = separators

        # Vector stacking lagrangian multipliers
        self.y = {sep: {aurco_id: [0, 0, 0] for aurco_id in separators[sep].keys()} for sep in separators.keys()}
        
        self.graph = graph

        # Penality parameter
        self.p = 0.1

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
    
    def rotation_matrix_ca(self, theta):
        return ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta)),
            ca.horzcat(ca.sin(theta), ca.cos(theta))
        )

    def log_map_SE2_ca(self, T):
        R = T[:2, :2]
        t = T[:2, 2]

        theta = ca.atan2(R[1, 0], R[0, 0])

        # Use ca.if_else to handle the small theta case
        v_x = ca.if_else(ca.fabs(theta) < 1e-5, t[0], (t[0] * ca.sin(theta) - t[1] * (1 - ca.cos(theta))) / theta)
        v_y = ca.if_else(ca.fabs(theta) < 1e-5, t[1], (t[0] * (1 - ca.cos(theta)) + t[1] * ca.sin(theta)) / theta)

        return ca.vertcat(v_x, v_y, theta)
    
    # Function to compute the logarithmic map of SE(2)
    def log_map_SE2(self, T):

        # Extract the rotation part R and the translation part t from the matrix T
        R = T[:2, :2]
        t = T[:2, 2]
        
        # Compute the rotation angle theta
        theta = np.arctan2(R[1, 0], R[0, 0])

        if np.abs(theta) < 1e-5:
            # Near zero rotation, translation is simply t
            v_x, v_y = t
            
        else:
            # Compute the inverse of the Jacobian J(theta)
            sin_theta = R[1, 0]
            cos_theta = R[0, 0]
            J_inv = (1 / theta) * np.array([[sin_theta, -(1 - cos_theta)], [1 - cos_theta, sin_theta]])
            
            # Compute the translation part v
            v = np.dot(J_inv, t)
            v_x, v_y = v[0], v[1]
        
        # Return the twist vector
        return np.array([v_x, v_y, theta])

    # Function to create an SE(2) matrix from a rotation (theta) and translation (t)
    def create_SE2(self, theta, t):
        T = ca.DM.eye(3)
        T[:2, :2] = self.rotation_matrix(theta)
        T[:2, 2] = t
        return T
    
    def create_SE2_ca(self, theta, t):
        # Ensure the rotation matrix R is 2x2
        R = self.rotation_matrix_ca(theta)
        
        # Ensure t is a column vector (2x1)
        t = ca.reshape(t, 2, 1)
        
        # Horizontally concatenate R (2x2) and t (2x1) to get a 2x3 matrix
        upper_part = ca.horzcat(R, t)
        
        # Add the homogeneous row [0, 0, 1] (ensure it's 1x3)
        lower_part = ca.DM([[0, 0, 1]])  # Explicitly define as a 1x3 matrix
        
        # Vertically concatenate to form the full 3x3 SE(2) transformation matrix
        T = ca.vertcat(upper_part, lower_part)
    
        return T

    def cost_function(self, x, v_index, edges, e_index, graph):

        # x: optimization variable for vertex

        cost = ca.DM(0.0)
        
        for (i, j) in e_index:
            relative_pose = self.create_SE2(edges[e_index.index((i,j)) * 4 + 2], edges[e_index.index((i,j)) * 4:e_index.index((i,j)) * 4 + 2])
            
            x_i = self.create_SE2_ca(x[v_index.index(i) * 3 + 2], x[v_index.index(i) * 3:v_index.index(i) * 3 + 2])
            x_j = self.create_SE2_ca(x[v_index.index(j) * 3 + 2], x[v_index.index(j) * 3:v_index.index(j) * 3 + 2])
            
            Omega_ij = ca.DM.eye(3)
            cov = edges[e_index.index((i,j)) * 4 + 3]
            Omega_ij[0, :] = cov[0:3]
            Omega_ij[1, 1:] = cov[3:5]
            Omega_ij[2, 2] = cov[5]
            
            log_term = self.log_map_SE2_ca(ca.mtimes(ca.mtimes(ca.inv(relative_pose), ca.inv(x_i)), x_j))
            cost += ca.mtimes(ca.mtimes(log_term.T, Omega_ij), log_term)
        
        if graph == 0:
            prior_se = self.create_SE2(self.prior[2], self.prior[:2])
            x_p = self.create_SE2_ca(x[2], x[:2])
            log_term = self.log_map_SE2_ca(ca.mtimes(ca.inv(prior_se), x_p))
            cost += ca.mtimes(ca.mtimes(log_term.T, ca.DM.eye(3)), log_term)
        
        for agent in self.separators.keys():
            if agent is not self.graph:
                for sep in self.separators[agent].keys():
                    mesurements, xi, xj = self.separators[agent][sep]
                    x_i = self.create_SE2_ca(x[v_index.index(xi) * 3 + 2], x[v_index.index(xi) * 3:v_index.index(xi) * 3 + 2])
                    x_j = self.create_SE2_ca(xj[2], xj[:2])

                    relative_pose = self.create_SE2(mesurements[2], mesurements[:2])
                    Omega_ij = ca.DM.eye(3)
                    cov = mesurements[3]
                    Omega_ij[0, :] = cov[0:3]
                    Omega_ij[1, 1:] = cov[3:5]
                    Omega_ij[2, 2] = cov[5]
                        
                    log_term = self.log_map_SE2_ca(ca.mtimes(ca.mtimes(ca.inv(relative_pose), ca.inv(x_i)), x_j))

                    y_vector = ca.vertcat(
                        self.y[agent][sep][0] / self.p,
                        self.y[agent][sep][1] / self.p,
                        self.y[agent][sep][2] / self.p
                    )
                    term = log_term + y_vector
                    cost += (self.p / 2) * ca.mtimes(ca.mtimes(term.T, Omega_ij), term)
        
        return cost

    def solve(self):
        x = ca.SX.sym('x', len(self.vertex)*3)
        
        self.edges = np.array(self.edges, dtype=object).flatten()

        objective = self.cost_function(x, self.v_index, self.edges, self.e_index, self.graph)
        nlp = {'x': x, 'f': objective}
        solver = ca.nlpsol('solver', 'ipopt', nlp, self.opt)

        initial_guess = np.array(self.vertex).flatten()

        sol = solver(x0=initial_guess)
        
        optimal_poses = [[float(sol['x'][i*3]),float(sol['x'][i*3+1]),float(sol['x'][i*3+2])] for i in range(len(np.array(sol['x']))//3)]

        return optimal_poses

    def update(self, sep):
        mesurements, xi, xj = sep
        relative_pose = self.create_SE2(mesurements[2], mesurements[:2])
        x_i = self.create_SE2_ca(self.vertex[self.v_index.index(xi)][2], self.vertex[self.v_index.index(xi)][:2])
        x_j = self.create_SE2(xj[2], xj[:2])
        return self.log_map_SE2_ca(ca.mtimes(ca.mtimes(ca.inv(relative_pose), ca.inv(x_i)), x_j))

    def optimize(self):

        for i in range(self.max_iters):
            if i > 0:
                self.p = 1
            self.vertex = self.solve()
            for agent in self.separators.keys():
                if agent is not self.graph:
                    for sep in self.separators[agent].keys():          
                        self.y[agent][sep] += self.update(self.separators[agent][sep])
        return self.vertex
