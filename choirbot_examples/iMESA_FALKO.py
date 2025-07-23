#!/usr/bin/env python3
"""
iMESA with FALKO Integration - Distributed Multi-Robot SLAM

This file implements the iMESA (distributed SLAM) algorithm using FALKO keypoint 
matching instead of ArUco markers for inter-robot pose estimation.

MIGRATION FROM ARUCO TO FALKO:
=============================

Original ArUco Logic:
- Used visual markers for robot detection
- Calculated relative poses using camera geometry
- Required line-of-sight between robots
- Limited to detection range and angle

New FALKO Logic:
- Uses keypoint matching between robot observations
- Leverages place recognition for robust matching
- Works with partial overlaps and different viewpoints
- More robust to environmental conditions

Key Changes:
1. Removed: camera subscriber, image processing, ArUco detection
2. Added: FALKO match subscriber, keypoint-based separators
3. Replaced: marker-based pose estimation with FALKO transformation data
4. Enhanced: separator creation using keypoint match confidence

ADMM ALGORITHM OVERVIEW:
=======================

The Alternating Direction Method of Multipliers (ADMM) is used for distributed
optimization of robot trajectories. Each robot maintains:

- Vertices: Local trajectory points (x, y, theta)
- Edges: Odometry constraints between consecutive vertices  
- Separators: Inter-robot constraints from FALKO matches
- Dual variables (y): Lagrange multipliers for consensus

The optimization alternates between:
1. Local trajectory optimization (minimize cost with separators)
2. Dual variable updates (enforce consensus between robots)

FALKO INTEGRATION:
=================

FALKO provides:
- KeypointMatch messages with relative transformations
- Translation: [dx, dy, dz] between matched keypoint sets
- Rotation: theta angle between robot poses
- Confidence: implicit in successful matching

These are converted to separator constraints for ADMM optimization.

Author: Riccardo (migrated from ArUco implementation)
Date: 2024
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from choirbot_interfaces.msg import KeypointMatch
import numpy as np
import math
import time  # For timestamp handling
import matplotlib.pyplot as plt
import casadi as ca
import sys

opt = {
    "ipopt.max_iter": 3,  # Set the maximum number of iterations to 100
    "print_time": False,     # Optional: To prevent printing time statistics'print_level': 0
    "ipopt.print_level": 0,  # Optional: To reduce the verbosity
}

class ADMM:
    """
    Alternating Direction Method of Multipliers for Distributed SLAM Optimization.
    
    This class implements the ADMM algorithm for distributed pose graph optimization.
    Each robot runs its own ADMM instance to optimize its local trajectory while
    maintaining consensus with other robots through separator constraints.
    
    Key Components:
    - Local Variables: Robot's own trajectory vertices
    - Separators: Inter-robot constraints from FALKO matches  
    - Dual Variables: Lagrange multipliers for consensus enforcement
    - Penalty Parameter: Controls convergence vs. consensus trade-off
    
    ADMM Steps:
    1. Minimize local cost + separator terms + dual variable terms
    2. Update dual variables based on constraint violations
    3. Repeat until convergence
    
    Separator Structure (FALKO-based):
    Each separator contains:
    - measurements: [dx, dy, dtheta, covariance] from FALKO match
    - local_vertex_id: This robot's vertex where match occurred
    - neighbor_pose: Other robot's pose [x, y, theta] at match time
    """

    def __init__(self, edges, eindex, vertex, index, prior, separators, graph):
        self.edges = edges
        self.eindex = eindex
        self.vertex = vertex
        self.index = index
        self.prior = prior
        self.separators = separators
        #self.y = [[[0,0,0] for s in agent.keys()] for agent in separators]
        self.y = [{} for agent in separators]
        for i in range(len(separators)):
            agent = separators[i]
            for s in agent.keys():
                self.y[i][s] = [0,0,0]
        self.graph = graph
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

        '''
        v_x = (t[0] * ca.sin(theta) - t[1] * (1 - ca.cos(theta))) / theta
        v_y = (t[0] * (1 - ca.cos(theta)) + t[1] * ca.sin(theta)) / theta
        '''

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
        T = np.eye(3)
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

    def cost_function(self, x, index, edges, eindex, graph):
        cost = ca.DM(0.0)
        
        for (i, j) in eindex:
            relative_pose = self.create_SE2(edges[eindex.index((i,j)) * 4 + 2], edges[eindex.index((i,j)) * 4:eindex.index((i,j)) * 4 + 2])
            
            x_i = self.create_SE2_ca(x[index.index(i) * 3 + 2], x[index.index(i) * 3:index.index(i) * 3 + 2])
            x_j = self.create_SE2_ca(x[index.index(j) * 3 + 2], x[index.index(j) * 3:index.index(j) * 3 + 2])
            
            Omega_ij = np.eye(3)
            cov = edges[eindex.index((i,j)) * 4 + 3]
            Omega_ij[0, :] = cov[0:3]
            Omega_ij[1, 1:] = cov[3:5]
            Omega_ij[2, 2] = cov[5]
            
            log_term = self.log_map_SE2_ca(ca.mtimes(ca.mtimes(np.linalg.inv(relative_pose), ca.inv(x_i)), x_j))
            cost += ca.mtimes(ca.mtimes(log_term.T, Omega_ij), log_term)
        
        if graph == 0:
            prior_se = self.create_SE2(self.prior[2], self.prior[:2])
            x_p = self.create_SE2_ca(x[2], x[:2])
            log_term = self.log_map_SE2_ca(ca.mtimes(np.linalg.inv(prior_se), x_p))
            cost += ca.mtimes(ca.mtimes(log_term.T, ca.DM.eye(3)), log_term)
        
        for agent in range(len(self.separators)):
            if agent is not self.graph:
                for sep in self.separators[agent].keys():
                    mesurements, xi, xj = self.separators[agent][sep]
                    x_i = self.create_SE2_ca(x[index.index(xi) * 3 + 2], x[index.index(xi) * 3:index.index(xi) * 3 + 2])
                    x_j = self.create_SE2_ca(xj[2], xj[:2])

                    #x_i = x[index.index(xi) * 3:index.index(xi) * 3 + 2]
                    #x_j = xj[:2]
                        
                    relative_pose = self.create_SE2(mesurements[2], mesurements[:2])
                    Omega_ij = np.eye(3)
                    cov = mesurements[3]
                    Omega_ij[0, :] = cov[0:3]
                    Omega_ij[1, 1:] = cov[3:5]
                    Omega_ij[2, 2] = cov[5]
                        
                    log_term = self.log_map_SE2_ca(ca.mtimes(ca.mtimes(np.linalg.inv(relative_pose), ca.inv(x_i)), x_j))
                    #log_term = ca.norm_2(x_i - x_j)

                    term = log_term + np.array([self.y[agent][sep][0] / self.p, self.y[agent][sep][1] / self.p, self.y[agent][sep][2] / self.p])
                    cost += (self.p / 2) * ca.mtimes(ca.mtimes(term.T, Omega_ij), term)
        
        return cost

    def solve(self):
        x = ca.SX.sym('x', self.vertex.size)
        
        self.edges = np.array(self.edges).flatten()

        objective = self.cost_function(x, self.index, self.edges, self.eindex, self.graph)
        nlp = {'x': x, 'f': objective}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opt)

        initial_guess = self.vertex.flatten()

        sol = solver(x0=initial_guess)
        
        optimal_poses = np.array(sol['x']).reshape((-1, 3)).flatten()

        return optimal_poses

    def update(self, sep):
        mesurements, xi, xj = sep
        relative_pose = self.create_SE2(mesurements[2], mesurements[:2])
        x_i = self.create_SE2_ca(self.vertex[self.index.index(xi) * 3 + 2], self.vertex[self.index.index(xi) * 3:self.index.index(xi) * 3 + 2])
        x_j = self.create_SE2(xj[2], xj[:2])
        return self.log_map_SE2_ca(ca.mtimes(ca.mtimes(np.linalg.inv(relative_pose), ca.inv(x_i)), x_j))
        #return self.log_map_SE2(np.linalg.inv(x_i) @ x_j)

    def optimize(self):
        for i in range(10):
            if i > 0:
                self.p = 1
            self.vertex = self.solve()     
            for agent in range(len(self.separators)):
                if agent is not self.graph:
                    for sep in self.separators[agent].keys():          
                        self.y[agent][sep] += self.update(self.separators[agent][sep])
        return self.vertex

class Agent_iMESA(Node):
    """
    FALKO-based iMESA Agent for Distributed Multi-Robot SLAM.
    
    This class implements a distributed SLAM agent that uses FALKO keypoint 
    matching for inter-robot pose estimation instead of ArUco markers.
    
    Key Features:
    - Subscribes to odometry for trajectory building
    - Subscribes to FALKO matches for inter-robot constraints
    - Publishes optimized trajectory for other agents
    - Runs ADMM optimization for distributed consensus
    
    Data Flow:
    1. Odometry -> New vertices in local trajectory
    2. FALKO matches -> Separator constraints between robots
    3. ADMM optimization -> Optimized trajectory
    4. Publish results -> Share with other agents
    
    Major Changes from ArUco Version:
    - Removed: Camera subscriber, image processing, ArUco detection
    - Added: FALKO match processing, keypoint-based separators
    - Enhanced: Robust separator creation from keypoint matches
    - Improved: Error handling and logging
    """

    def __init__(self, agent_id, num_agent):
        super().__init__(f"agent{agent_id}_subscriber")

        self.publisher_ = self.create_publisher(Float64MultiArray, f"agent{agent_id}_pub", 10)
        self.publisherFinal = self.create_publisher(Float64MultiArray, f"agent{agent_id}_pub_final", 10)

        self.agent_id = agent_id
        self.num_agent = num_agent

        self.subscription = self.create_subscription(
            Odometry,
            f"/agent_{agent_id}/odom",  # Topic name
            self.odom_callback,
            10  # QoS Depth
        )
        
        # Subscribe to FALKO keypoint matches
        self.subscription = self.create_subscription(
            KeypointMatch, 
            f'/matches_agent_{self.agent_id}',
            self.falko_match_callback,
            10
        )
        
        # Subscribe to other agents' pose data
        for agent in range(num_agent):
            if agent != agent_id:
                self.subscription = self.create_subscription(
                    Float64MultiArray,
                    f"agent{agent}_pub",
                    lambda msg, agent_id=agent: self.other_agent_callback(msg, agent_id),
                    10
                )
        
        self.subscription  # prevent unused variable warning

        # Data storage for vertices and edges
        self.vertices = []  # Each entry is (id, x, y, theta)
        self.edges = []     # Each entry is (id1, id2, dx, dy, dtheta, covariance)
        self.index = []
        self.eindex = []

        self.prev_pose = None
        self.prev_id = None
        self.vertex_id = 0  # Vertex counter
        self.last_time = time.time()  # Timestamp for the last processed message

        # FALKO-based data structures (replacing ArUco logic)
        self.falko_matches = {}  # Dictionary to store FALKO matches by neighbor ID
        self.separators = [{} for _ in range(num_agent)]  # Separators for ADMM
        self.separator_counters = [0 for _ in range(num_agent)]
        
        # Store information about other agents for separator creation
        self.other_agents_data = {}  # {agent_id: latest_pose_data}

    def falko_match_callback(self, msg):
        """
        Callback function for FALKO keypoint matches.
        
        This function processes incoming FALKO matches and stores them for
        later use in separator creation during ADMM optimization.
        
        Args:
            msg: KeypointMatch message containing match data
        """
        # Extract match information
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
        
        # Log the match for debugging
        self.get_logger().info(f"Agent {self.agent_id} received FALKO match from Agent {neighbor_id}: "
                              f"dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}")

    def other_agent_callback(self, msg, agent_id):
        """
        Callback function for other agents' pose data.
        
        This stores the latest pose information from other agents,
        which is needed for separator creation in ADMM.
        
        Args:
            msg: Float64MultiArray containing pose data
            agent_id: ID of the agent sending the data
        """
        # Store the latest pose data from other agents
        # msg.data format: [agent_id, x1, y1, theta1, x2, y2, theta2, ...]
        if len(msg.data) > 1:
            self.other_agents_data[agent_id] = msg.data[1:]  # Skip the first element (agent_id)

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
            current_vertex_id = self.vertex_id - 1  # Most recent vertex
            
            neighbor_pose = [match_data.neigh_position.agent_position.x,
                            match_data.neigh_position.agent_position.y,
                            match_data.neigh_position.agent_position.theta] # [x, y, theta]
            
            # Create separator entry
            # Using neighbor_id as the separator key (replaces ArUco ID)
            separator_key = self.separator_counters[neighbor_id]
            self.separator_counters[neighbor_id] += 1
    
            
            self.separators[neighbor_id][separator_key] = (
                measurements,           # [dx, dy, dtheta, covariance]
                current_vertex_id,      # This agent's vertex ID
                neighbor_pose          # Neighbor's pose [x, y, theta]
            )
            
            self.get_logger().info(f"Created separator for Agent {neighbor_id} at vertex {current_vertex_id}")

    def compute_measurements(self, this_x, this_y, this_theta, that_x, that_y, that_theta):
        """
        Compute relative measurements between two poses.
        
        This function calculates the relative transformation between two poses
        in the format expected by the ADMM optimization algorithm.
        
        Args:
            this_x, this_y, this_theta: Current agent's pose
            that_x, that_y, that_theta: Other agent's pose
            
        Returns:
            List containing [dx, dy, dtheta, covariance]
        """
        # Transform the other agent's position to the current agent's local frame
        dx = math.cos(-this_theta) * (that_x - this_x) - math.sin(-this_theta) * (that_y - this_y)
        dy = math.sin(-this_theta) * (that_x - this_x) + math.cos(-this_theta) * (that_y - this_y)
        dtheta = that_theta - this_theta
        
        # Normalize dtheta to [-pi, pi]
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        # Add edge with a simple example covariance (diagonal matrix values)
        covariance = [44.72135955, 0, 0, 44.72135955, 0, 44.72135955]

        return [dx, dy, dtheta, covariance]

    @staticmethod
    def get_yaw_from_quaternion(quat):
        """
        Convert a quaternion to a yaw angle (theta).
        :param quat: geometry_msgs.msg.Quaternion
        :return: yaw angle in radians
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return yaw

    def odom_callback(self, msg):
        """
        Odometry callback - processes robot pose and triggers ADMM optimization.
        
        This is the main callback that:
        1. Creates new vertices from odometry data
        2. Creates edges between consecutive vertices
        3. Creates separators from FALKO matches
        4. Runs ADMM optimization
        5. Publishes results and updates visualization
        """
        current_time = time.time()

        # Process data only every 2 seconds (configurable)
        if current_time - self.last_time < 2.0 and self.vertex_id != 0:
            return  # Skip this message

        self.last_time = current_time  # Update the timestamp

        # Extract pose from odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        theta = self.get_yaw_from_quaternion(orientation)

        # Create current pose tuple
        current_pose = (position.x, position.y, theta)
        
        # Small adjustment for specific agents (if needed for calibration)
        if self.agent_id == 1 or self.agent_id == 3:
            current_pose = (position.x, position.y, theta + 0.01)

        # Add new vertex to the trajectory
        self.vertices = np.append(self.vertices, [position.x, position.y, theta])
        self.index.append(self.vertex_id)

        # Initialize prior for the first vertex
        if self.vertex_id == 0:
            self.prior = self.vertices
            # Initialize plotting
            plt.figure(figsize=(8, 8))

        # Create edge between consecutive vertices (odometry edges)
        if self.prev_pose is not None:
            edge_measurements = self.compute_measurements(
                self.prev_pose[0], self.prev_pose[1], self.prev_pose[2], 
                position.x, position.y, theta
            )
            self.edges = np.append(self.edges, edge_measurements)
            self.eindex.append((self.prev_id, self.vertex_id))

        # Create separators from FALKO matches
        # This replaces the ArUco-based separator creation
        if self.falko_matches:
            self.create_separators_from_falko()
            self.get_logger().info(f"Agent {self.agent_id}: Created separators from {len(self.falko_matches)} FALKO matches")

        # Run ADMM optimization if we have more than one vertex
        if self.vertex_id > 0:
            try:
                # Create and run ADMM optimizer
                admm = ADMM(self.edges, self.eindex, self.vertices, self.index, self.prior, self.separators, self.agent_id)
                optimized_vertices = admm.optimize()
                
                # Update vertices with optimized values
                self.vertices = optimized_vertices
                
                # Publish optimized trajectory
                msg_array = Float64MultiArray()
                msg_array.data = [float(self.agent_id)] + self.vertices.tolist()
                self.publisherFinal.publish(msg_array)
                
                # Update visualization
                self.update_visualization()
                
                self.get_logger().info(f"Agent {self.agent_id}: ADMM optimization completed at vertex {self.vertex_id}")
                
            except Exception as e:
                self.get_logger().error(f"Agent {self.agent_id}: ADMM optimization failed: {e}")

        # Update state for next iteration
        self.prev_pose = current_pose
        self.prev_id = self.vertex_id
        self.vertex_id += 1

    def update_visualization(self):
        """
        Update the trajectory visualization plot.
        
        This function creates a real-time plot showing:
        - The optimized trajectory as connected points
        - Orientation arrows at each vertex
        - Agent ID and grid for reference
        """
        x_vals = []
        y_vals = []

        # Extract trajectory points for plotting
        for i in range(len(self.vertices)//3):
            pose_data = self.vertices[i*3:i*3+3]
            
            # Extract x, y, and theta values
            x = pose_data[0]
            y = pose_data[1]
            theta = pose_data[2]
            
            x_vals.append(x)
            y_vals.append(y)

            # Add arrows to show direction (pose orientation)
            dx = np.cos(theta) * 0.1  # Arrow length for orientation
            dy = np.sin(theta) * 0.1
            plt.arrow(x, y, dx, dy, head_width=0.02, head_length=0.01, fc='red', ec='red')
        
        # Plot trajectory
        plt.plot(x_vals, y_vals, 'o-', label=f"Agent {self.agent_id} Trajectory", markersize=5)

        # Configure plot
        plt.title(f"Agent {self.agent_id} - FALKO-based iMESA")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling for both axes

        # Update plot
        plt.pause(0.001)
        plt.clf()

def main(args=None):
    """
    Main function to initialize and run the FALKO-based iMESA agent.
    
    Args:
        args: Command line arguments (unused)
    """
    # Get agent configuration from command line arguments
    agent_id = int(sys.argv[1])
    num_agent = int(sys.argv[2])
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create agent instance
    agent = Agent_iMESA(agent_id, num_agent)
    
    # Initialize interactive plotting
    plt.ion()
    plt.show()

    try:
        print(f"Starting FALKO-based iMESA Agent {agent_id}")
        print(f"Total number of agents: {num_agent}")
        print("Waiting for FALKO matches and odometry data...")
        
        # Run the agent
        rclpy.spin(agent)
        
    except KeyboardInterrupt:
        agent.get_logger().info(f'Agent {agent_id} shutting down...')
    finally:
        # Cleanup
        agent.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()