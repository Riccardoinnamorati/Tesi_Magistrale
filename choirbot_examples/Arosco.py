import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import math

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

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/agent_0/agent_0/camera',
            self.image_callback,
            10  # QoS (Quality of Service) Depth
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

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
                    distance = np.linalg.norm(tvec)
                    
                    # Display the distance on the image
                    cv2.putText(frame, f"Distance: {math.sqrt(((distance*100)**2) - ((0.135*100)**2))/100:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                for rvec, tvec in zip(rvecs, tvecs):
                    # Calculate angles
                    t_x, t_y, t_z = tvec[0][0], tvec[0][1], tvec[0][2]

                    # Calculate horizontal and vertical angles (in radians)
                    horizontal_angle = np.arctan2(t_x, t_z)
                    #vertical_angle = np.arctan2(t_y, t_z)

                    # Convert to degrees
                    horizontal_angle_deg = np.degrees(horizontal_angle)
                    #vertical_angle_deg = np.degrees(vertical_angle)

                    cv2.putText(frame, f"Theta: {horizontal_angle_deg:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    #cv2.putText(frame, f"Vertical: {vertical_angle_deg:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
                    singular = sy < 1e-6
                    
                    if not singular:
                        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        roll = np.arctan2(-rotation_matrix[2, 0], sy)
                        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                        roll = np.arctan2(-rotation_matrix[2, 0], sy)
                        yaw = 0
                    
                    # Convert from radians to degrees
                    pitch_deg = np.degrees(pitch)
                    roll_deg = np.degrees(roll)
                    yaw_deg = np.degrees(yaw)
                    
                    # Display the angles on the image
                    #cv2.putText(frame, f"Pitch: {pitch_deg:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #cv2.putText(frame, f"Roll: {roll_deg:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    angle = 180 - yaw_deg + horizontal_angle_deg
                    if angle > 360:
                        angle -= 360
                    if angle < 0:
                        angle += 360
                    cv2.putText(frame, f"Aruca Angle: {angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        
            # Show the frame
            cv2.imshow("Frame", frame)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info('Shutting down...')
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()