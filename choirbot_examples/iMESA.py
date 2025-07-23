import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
import time  # For timestamp handling
import matplotlib.pyplot as plt
import casadi as ca
import sys

# Approximate camera intrinsic parameters
frame_width = 640  # Camera resolution width
frame_height = 480  # Camera resolution height

# Focal length estimation (rough approximation, in pixels)
focal_length = frame_width  # Assume focal length ~ width of the image
camera_matrix = np.array([[focal_length, 0, frame_width / 2],
                          [0, focal_length, frame_height / 2],
                          [0, 0, 1]])

# No distortion
dist_coeffs = np.zeros(5)

# Define the size of the ArUco marker (in meters)
marker_length = 0.05  # Example: 5 cm

opt = {
    "ipopt.max_iter": 3,  # Set the maximum number of iterations to 100
    "print_time": False,     # Optional: To prevent printing time statistics'print_level': 0
    "ipopt.print_level": 0,  # Optional: To reduce the verbosity
}

class ADMM:
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
        
        # Subscribe to camera image topic
        self.subscription = self.create_subscription(
            Image,
            f"/agent_{agent_id}/agent_{agent_id}/camera",
            self.image_callback,
            10  # QoS (Quality of Service) Depth
        )
        
        for agent in range(num_agent):
            if agent is not agent_id:
                self.subscription = self.create_subscription(
                    Float64MultiArray,
                    f"agent{agent}_pub",
                    self.listener_callback,
                    10
                )
        
        self.subscription  # prevent unused variable warning

        self.bridge = CvBridge()
        
        # Data storage for vertices and edges
        self.vertices = []  # Each entry is (id, x, y, theta)
        self.edges = []     # Each entry is (id1, id2, dx, dy, dtheta, covariance)
        self.index = []
        self.eindex = []

        self.prev_pose = None
        self.prev_id = None
        self.vertex_id = 0  # Vertex counter
        self.last_time = time.time()  # Timestamp for the last processed message

        self.temp_aruco = []
        self.S1 = [[] for _ in range(num_agent)] 
        self.S = [{} for _ in range(num_agent)]

        self.separators = [{} for _ in range(num_agent)]

    # Funzione per calcolare il terzo lato usando il teorema del coseno
    def calcola_lato(self, a, b, gamma):
        # gamma in gradi -> converti in radianti
        gamma_rad = math.radians(gamma)
        c = math.sqrt(a**2 + b**2 - 2 * a * b * math.cos(gamma_rad))
        return c

    # Funzione per calcolare un angolo usando il teorema del coseno
    def calcola_angolo(self, opposto, lato1, lato2):
        cos_angolo = (lato1**2 + lato2**2 - opposto**2) / (2 * lato1 * lato2)
        angolo = math.degrees(math.acos(cos_angolo))  # Converti in gradi
        return angolo

    def compute_separators(self, a, b, alpha, this_theta, that_theta, og_theta):
        c = self.calcola_lato(a*100, b*100, abs(alpha))/100 # Calcolo del terzo lato
        print("Distance between: ", c)
        beta = self.calcola_angolo(b*100, a*100, c*100)   # Angolo opposto a lato b
        og_deg = np.degrees(og_theta)
        print("Alpha: ", alpha)
        print("Og theta: ", og_deg)
        if alpha < 0:
            angle = og_deg - this_theta - beta
        else:
            angle = og_deg + this_theta + beta
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        print("Angolo rispetto gli assi: ", angle)
        theta = this_theta - alpha - that_theta
        if theta > 180:
            theta -= 360
        elif theta < -180:
            theta += 360

        # that cartesian coordinates
        x = c * math.cos(np.radians(angle)) 
        y = c * math.sin(np.radians(angle))

        return x, y, np.radians(theta)

    def listener_callback(self, msg):
        arucoID = int(msg.data[0])
        agent_ID = int(msg.data[7])
        if arucoID not in self.S1[agent_ID]:
            self.S1[agent_ID].append(arucoID)
            self.S[agent_ID][arucoID] = [msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5], msg.data[6]]

    def compute_mesurments(self, this_x, this_y, this_theta, that_x, that_y, that_theta):
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
        current_time = time.time()

        # Process data only every 2 seconds
        if current_time - self.last_time < 2.0 and self.vertex_id != 0:
            return  # Skip this message

        self.last_time = current_time  # Update the timestamp

        # Extract pose
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        theta = self.get_yaw_from_quaternion(orientation)

        current_pose = (position.x, position.y, theta)
        if self.agent_id == 1 or self.agent_id == 3:
            current_pose = (position.x, position.y, theta + 0.01)

        self.vertices = np.append(self.vertices, [position.x, position.y, theta])
        self.index.append(self.vertex_id)

        if self.vertex_id == 0:
            self.prior = self.vertices
            # Plot settings
            plt.figure(figsize=(2, 2))
            # Add large margins

        # If we have a previous pose, calculate the relative transformation (edge)
        if self.prev_pose is not None:
            self.edges = np.append(self.edges, self.compute_mesurments(self.prev_pose[0], self.prev_pose[1], self.prev_pose[2], position.x, position.y, theta))
            self.eindex.append((self.prev_id, self.vertex_id))

        if self.temp_aruco:
            id = self.temp_aruco[0]
            if id not in self.S1[self.agent_id]:
                self.S1[self.agent_id].append(id)
                self.S[self.agent_id][id] = [self.temp_aruco[1], self.temp_aruco[2], self.temp_aruco[3], position.x, position.y, theta, self.vertex_id]
                print("Point is: ", position.x, position.y)
                msg = Float64MultiArray()
                msg.data = [float(id), self.temp_aruco[1], self.temp_aruco[2], self.temp_aruco[3], position.x, position.y, theta, float(self.agent_id)]
                self.publisher_.publish(msg)
                self.temp_aruco = []

        for agent in range(self.num_agent):
            if agent is not self.agent_id:
                if bool(set(self.S[agent].keys()) & set(self.S[self.agent_id].keys())):
                    arucoID_list = list(set(self.S[agent].keys()) & set(self.S[self.agent_id].keys()))
                    for arucoID in arucoID_list:
                        print("\n\n\n\n\n\n\n")
                        print("===================================================================")
                        print("Id: ", arucoID)
                        b = self.S[self.agent_id][arucoID][0]  # Primo lato
                        a = self.S[agent][arucoID][0]  # Secondo lato
                        print("Distance: ", a)
                        alpha = self.S[self.agent_id][arucoID][2] - self.S[agent][arucoID][2]  # Angolo compreso tra a e b in gradi
                        if alpha > 180:
                            alpha -= 360
                        elif alpha < -180:
                            alpha += 360
                        that_angle = self.S[self.agent_id][arucoID][1]
                        this_angle = self.S[agent][arucoID][1]
                        that_x, that_y, that_theta = self.S[agent][arucoID][3:7]
                        this_x, this_y, this_theta = self.compute_separators(a, b, alpha, this_angle, that_angle, that_theta)
                        print("this point: ", this_x, this_y)
                    
                        this_x += that_x
                        this_y += that_y
                        this_theta += that_theta

                        print("Point should be: ", this_x, this_y)

                        sep = self.compute_mesurments(this_x, this_y, this_theta, that_x, that_y, that_theta)
                        vertexID = self.S[self.agent_id][arucoID][6]
                        this_point = vertexID
                        that_point = self.S[agent][arucoID][3:7]
                        self.separators[agent][arucoID] = (sep, this_point, that_point)
                        print("===================================================================")
                        print("\n\n\n\n\n\n\n")

        if self.vertex_id != 0:
            admm = ADMM(self.edges, self.eindex, self.vertices, self.index, self.prior, self.separators, self.agent_id)
            self.vertices = admm.optimize()

            array = []
            array.append(float(self.agent_id))
            msg = Float64MultiArray()
            msg.data = array + self.vertices.tolist()
            self.publisherFinal.publish(msg)

            x_vals = []
            y_vals = []

            # Plot trajectory
            for i in range(len(self.vertices)//3):
                pose_data = self.vertices[i*3:i*3+3]

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

        # Update previous pose and vertex ID
        self.prev_pose = current_pose
        self.prev_id = self.vertex_id
        self.vertex_id += 1

    def image_callback(self, msg):
        #self.get_logger().info('Receiving image data...')
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.waitKey(1)  # This keeps the image window responsive
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters =  cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            #detector = cv2.aruco.ArucoDetector(dictionary)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.temp_aruco= {}
    
            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
            
            if ids is not None:
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
                
                for i, tvec in enumerate(tvecs):
                    # Draw the detected marker
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Draw the axis for the marker
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
                    
                    # Calculate distance (Euclidean norm of the translation vector)
                    distance = np.linalg.norm(tvec) + 0.05

                    #distance = math.sqrt(((distance*100)**2) - ((0.16*100)**2))/100
                    
                    # Display the distance on the image
                    cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                for rvec, tvec in zip(rvecs, tvecs):
                    # Calculate angles
                    t_x, _, t_z = tvec[0][0], tvec[0][1], tvec[0][2]

                    # Calculate horizontal angles (in radians)
                    horizontal_angle = np.arctan2(t_x, t_z)

                    # Convert to degrees
                    horizontal_angle_deg = np.degrees(horizontal_angle)

                    cv2.putText(frame, f"Theta: {horizontal_angle_deg:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Draw marker and axes
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                for i, rvec in enumerate(rvecs):
                    # Draw the detected marker
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Draw the axis for the marker
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvecs[i], 0.1)
                    
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
                    angle = 180 - yaw_deg + horizontal_angle_deg
                    if angle > 360:
                        angle -= 360
                    if angle < 0:
                        angle += 360
                    cv2.putText(frame, f"Aruco Angle: {angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if distance < 0.30:
                    id = tuple(ids.flatten())[0]
                    self.temp_aruco = [id, distance, horizontal_angle_deg, angle]
                        
            # Show the frame
            cv2.imshow("Frame", frame)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    agent_id = int(sys.argv[1])
    num_agent = int(sys.argv[2])
    rclpy.init(args=args)
    agent = Agent_iMESA(agent_id, num_agent)
    plt.ion()
    plt.show()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Shutting down...')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()