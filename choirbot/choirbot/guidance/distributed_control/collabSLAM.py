import numpy as np
from ..guidance import Guidance
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import casadi as ca

from ...communicator import TimeVaryingCommunicator

import time

class CollabSLAMGuidance(Guidance):
    """
    Collaborative Mapping

    Implements a formation control law for systems....
    """

    def __init__(self, pose_handler: str=None, pose_topic: str=None, sampling_freq: float=0.5, receive_freq: float=10.0):
        super().__init__(pose_handler, pose_topic)


        self.sampling_freq = sampling_freq
        self.receive_freq = receive_freq
        self.sampling_timer = self.create_timer(1.0/self.sampling_freq, self.sample_odom)
        self.receiving_timer = self.create_timer(1.0/self.receive_freq, self.check_receiving)
        self.get_logger().info('CollabSLAMGuidance {} started'.format(self.agent_id))


        # Noisy measurement
        self.noisy = True

        # Subscribe to image topic
        image_topic = f"/agent_{self.agent_id}/agent_{self.agent_id}/camera"
        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Aruco data
        self.aruco_data = []    # Sampled aruco (if available)
        
        # Localization data
        self.vertices = []      # List of sampled location -> [[x_0, y_0, theta_0], [], ..., []]
        self.edges = []          # List of sampled edge  between vertices -> [[dx, dy, dtheta, covariance], [], ..., []]
        self.previous_point = 0

        # Data structure for communication
        self.S = {agent:{} for agent in self.in_neighbors+[self.agent_id]}

        # Optimization data
        self.separators = {agent:{} for agent in self.in_neighbors}

        # Approximate camera intrinsic parameters
        self.frame_width  = 640  # Camera resolution width
        self.frame_height = 480  # Camera resolution height

        # Focal length estimation (rough approximation, in pixels)
        self.focal_length = self.frame_width  # Assume focal length ~ width of the image
        self.camera_matrix = np.array([ [self.focal_length, 0, self.frame_width / 2],
                                        [0, self.focal_length, self.frame_height / 2],
                                        [0, 0, 1]])
        # No distortion
        self.dist_coeffs = np.zeros(5)

        # Define the size of the ArUco marker (in meters)
        self.marker_length = 0.05  

        # Set Casadi options
        self.casadi_max_iters = 10
        self.casadi_opt = {
            "ipopt.max_iter": 3,     # Set the maximum number of iterations to 100
            "print_time": False,     # Optional: To prevent printing time statistics'print_level': 0
            "ipopt.print_level": 0,  # Optional: To reduce the verbosity
        }


        # Create publisher to centralized plotter
        self.plot_results = True
        self.publisher_plotter = self.create_publisher(Float64MultiArray, f"agent{self.agent_id}_plot", 10)


    def _instantiate_communicator(self):
        return TimeVaryingCommunicator(self.agent_id, self.n_agents, self.in_neighbors,
            out_neighbors=self.out_neighbors, synchronous_mode = False)

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


    def check_receiving(self):
        # self.get_logger().info('Checking receiving data...')
        data_neighbors = self.communicator.neighbors_receive_asynchronous(self.in_neighbors)
        # print(f'data_neighbors: {data_neighbors}')
        # data = [id_aruco, self.aruco_data[1], self.aruco_data[2], self.aruco_data[3], self.vertices[-1][0], self.vertices[-1][1], self.vertices[-1][2]]

        if data_neighbors:
            self.get_logger().info(f'Agent {self.agent_id} received data from neighbors: {data_neighbors.keys()}')
            for id_neigh in data_neighbors.keys():
                data = data_neighbors[id_neigh]
                if data[0] not in self.S[id_neigh].keys():
                    self.S[id_neigh][data[0]] = data[1:]


    def sample_odom(self):
        # self.get_logger().info('Sampling odometry data...')

        # skip if position is not available yet
        if self.current_pose.position is None:
            return

        _, _, theta = R.from_quat(self.current_pose.orientation).as_euler('xyz')
        
        if self.noisy and (self.agent_id == 0 or self.agent_id == 3):
            self.current_pose.position[0] += 0.3
            # theta += 1
        self.vertices.append([self.current_pose.position[0], self.current_pose.position[1], theta])

        if len(self.vertices) == 1:
            # Save first measurement (then used only for agent_0)
            self.prior = self.vertices[0]

            # Plot settings
            if self.plot_results:
                plt.figure(figsize=(2, 2))


        if len(self.vertices) > 1:
            self.edges.append(self.compute_measurements(self.previous_point, self.vertices[-1]))
        self.previous_point = self.vertices[-1]

        if self.aruco_data:
            # print(f'Agent {self.agent_id} found ArUco marker: {self.aruco_data}')
            id_aruco = self.aruco_data[0]
            if id_aruco not in self.S[self.agent_id].keys():
                
                data = [id_aruco, self.aruco_data[1], self.aruco_data[2], self.aruco_data[3], self.vertices[-1][0], self.vertices[-1][1], self.vertices[-1][2], len(self.vertices) - 1]
                
                self.S[self.agent_id][id_aruco] = data[1:]

                self.communicator.neighbors_send(data[:-1],self.out_neighbors)
                f = open("Comunication.txt", "a")
                f.write(f"{time.time()} {self.agent_id}\n")
                f.close()

                # Flush aruco data
                self.aruco_data = []

        self.optimize_graph()

    def check_angle_deg(self, alpha):
        if alpha > 180:
            alpha -= 360
        elif alpha < -180:
            alpha += 360
        return alpha

    def compute_agent_location(self, a, b, alpha, agent_gamma, neigh_gamma, neigh_theta):

        a *= 1000
        b *= 1000
        c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(np.radians(alpha)))

        beta = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))

        neigh_theta_deg = np.degrees(neigh_theta)

        # Compute angle (delta) wrt neigh's x-axis and distance vec between agent and neigh 
        if alpha < 0:
            delta = neigh_theta_deg - agent_gamma - beta
        else:
            delta = neigh_theta_deg + agent_gamma + beta

        agent_theta = agent_gamma - alpha - neigh_gamma

        c /= 1000
        agent_x = c * np.cos(np.radians(self.check_angle_deg(delta))) 
        agent_y = c * np.sin(np.radians(self.check_angle_deg(delta)))

        return agent_x, agent_y, np.radians(self.check_angle_deg(agent_theta))

    def compute_separator(self, neigh_data, agent_data):

        """
            Return the agent_id location wrt the neigh's framework
        """

        # Compute Triangle among agent, neigh and aruco
        a = neigh_data[0]        
        b = agent_data[0] 
        alpha = self.check_angle_deg(agent_data[2] - neigh_data[2])

        agent_gamma = agent_data[1]
        neigh_gamma = neigh_data[1]

        neigh_x, neigh_y, neigh_theta = neigh_data[3:6]
        agent_x, agent_y, agent_theta = self.compute_agent_location(a, b, alpha, neigh_gamma, agent_gamma, neigh_theta)

        # Convert agent location in original framework
        agent_x += neigh_x
        agent_y += neigh_y
        agent_theta += neigh_theta

        sep = self.compute_measurements((agent_x, agent_y, agent_theta), (neigh_x, neigh_y, neigh_theta))
        print(f'Agent {self.agent_id} computed separator: {sep}')
        agent_vertex_id = agent_data[6]
        return (sep, agent_vertex_id, neigh_data[3:6])

    def optimize_graph(self):

        for neigh_id in self.in_neighbors:
            if neigh_id != self.agent_id:
                id_aruco_list = list(set(self.S[neigh_id].keys()) & set(self.S[self.agent_id].keys()))
                #self.get_logger().info(f'Agent {self.agent_id} found {len(id_aruco_list)} common ArUco markers with agent {neigh_id}')
                if id_aruco_list:
                    for id_aruco in id_aruco_list:
                        self.separators[neigh_id][id_aruco] = self.compute_separator(self.S[neigh_id][id_aruco], self.S[self.agent_id][id_aruco])

        if len(self.vertices) == 1:
            return

        admm = ADMM(self.casadi_opt, self.casadi_max_iters, self.edges, self.vertices, self.prior, self.separators, self.agent_id)
        
        
        self.vertices = admm.optimize()

        if self.plot_results:
            self.plotter()


    def plotter(self):

        # Send to centralized plotter
        array = []
        array.append(float(self.agent_id))
        msg = Float64MultiArray()
        msg.data = array + [item for sublist in self.vertices for item in sublist]
        self.publisher_plotter.publish(msg)

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

    def image_callback(self, msg):
        # self.get_logger().info('Receiving image data...')

        try:
            # Convert ROS Image message to OpenCV image
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.waitKey(1)  # This keeps the image window responsive
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Clean aruco data
            self.aruco_data = []
    
            # Detect ArUco markers if available
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
            
            if ids is not None:
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                
                # Extract distance from image -> distance [m]
                for i, tvec in enumerate(tvecs):
                    # Draw the detected marker
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Draw the axis for the marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.01)
                    
                    # Calculate distance (Euclidean norm of the translation vector)
                    distance = np.linalg.norm(tvec) + 0.05

                    # Display the distance on the image
                    cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Extract the agent rotation wrt aruco image -> gamma [°]
                for rvec, tvec in zip(rvecs, tvecs):
                    # Calculate angles
                    t_x, _, t_z = tvec[0][0], tvec[0][1], tvec[0][2]

                    # Calculate horizontal angles (in radians)
                    gamma = np.arctan2(t_x, t_z)

                    # Convert to degrees
                    gamma_deg = np.degrees(gamma)

                    cv2.putText(frame, f"Gamma: {gamma_deg:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Draw marker and axes
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.01)

                # Extract aruco angle wrt agent direction -> Aruco Angle, angle [°]
                for i, rvec in enumerate(rvecs):
                    # Draw the detected marker
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Draw the axis for the marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvecs[i], 0.01)
                    
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Compute yaw, pitch, roll from the rotation matrix
                    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                    singular = sy < 1e-5
                    
                    if not singular:
                        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        yaw = 0
                    
                    # Convert from radians to degrees
                    yaw_deg = np.degrees(yaw)
                    
                    # Display the angles on the image
                    angle = self.check_angle_deg(180 - yaw_deg + gamma_deg)

                    cv2.putText(frame, f"Phi: {angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if distance < 0.30:
                    id = tuple(ids.flatten())[0]
                    self.aruco_data = [id, distance, gamma_deg, angle]
                        
            # Show the frame
            #cv2.imshow("Frame", frame)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

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

                    y_vector = ca.vertcat(self.y[agent][sep][0] / self.p,
                                          self.y[agent][sep][1] / self.p,
                                          self.y[agent][sep][2] / self.p)
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
