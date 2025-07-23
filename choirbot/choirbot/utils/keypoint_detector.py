import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from choirbot.utils.keypoint_detector import FALKOKeypointDetector
from choirbot_interfaces.msg import KeypointArray
import math
import time
# Aggiungi questi import
import matplotlib.pyplot as plt
import threading
import imageio
import os
plt.ion()  # Modalità interattiva per aggiornamento in tempo reale

class KeypointDetector(Node):
    def __init__(self):
        super().__init__('keypoint_detector')
        
        # Parameters
        self.declare_parameter('freq', 20.0)
        self.declare_parameter('agent_id', 0)
        self.declare_parameter('N', 5)  # Number of agents
        self.declare_parameter('initial_position', [0.0, 0.0])  # Initial position [x, y]
        
        self.agent_id = self.get_parameter('agent_id').value
        self.freq = self.get_parameter('freq').value
        self.N = self.get_parameter('N').value
        initial_pos = self.get_parameter('initial_position').value
        
                # Configurazione plot matplotlib
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle(f'Analisi Keypoint per Agent {self.agent_id}', fontsize=16)
        
        # Configurazione subplot di sinistra (punti laser + keypoint)
        self.scan_plot = self.axs[0]
        self.scan_plot.set_title('Scansione Laser e Keypoint')
        self.scan_plot.set_xlabel('X [m]')
        self.scan_plot.set_ylabel('Y [m]')
        self.scan_plot.grid(True)
        self.scan_plot.axis('equal')  # Stesso rapporto tra X e Y
        
        # Configurazione subplot di destra (score keypoint)
        self.score_plot = self.axs[1]
        self.score_plot.set_title('Score points')
        self.score_plot.set_xlabel('Points')
        self.score_plot.set_ylabel('Score')
        self.score_plot.grid(True)
        
        # Variabili per il plot
        self.laser_line = None
        self.keypoint_scatter = None
        self.score_bars = None
        
        # Lock per thread safety
        self.plot_lock = threading.Lock()
        
        # Flag per attivare/disattivare plotting
        self.enable_plotting = False
        self.save = False  # Flag to save scan data
        # Initialize FALKO algorithm
        self.falko = FALKOKeypointDetector()

        # Initialize robot position tracking
        self.current_position = np.array(initial_pos if len(initial_pos) >= 2 else [0.0, 0.0])  # [x, y]
        self.current_orientation = 0.0  # yaw angle
        self.current_velocity = np.array([0.0, 0.0])  # [vx, vy]
        self.last_pose_time = None
        self.last_position_update = time.time()

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, f'/agent_{self.agent_id}/agent_{self.agent_id}/LDS_01', self.laser_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, f'/agent_{self.agent_id}/odom', self.odom_callback, 10)
        
        # self.cmd_vel_sub = self.create_subscription(
        #     Twist, f'/agent_{self.agent_id}/cmd_vel', self.cmd_vel_callback, 10)
        
        # Publishers
        self.keypoint_pub = self.create_publisher(
            KeypointArray, f'/agent_{self.agent_id}/keypoints', 10)
        
        self.marker_pub = self.create_publisher(
            MarkerArray, f'agent_{self.agent_id}/keypoint_markers', 10)
        
        # Timer
        self.timer = self.create_timer(1.0/self.freq, self.detect_keypoints)
        
        # Timer per aggiornare posizione basata su velocità
        # self.position_timer = self.create_timer(0.1, self.update_position_from_velocity)
        
        self.all_scans = []
        self.latest_scan = None
        
        # Variabili per salvataggio frame e video

        self.frames = 780  # Cambia questo valore come preferisci
        self.frame_dir = f"frames_agent_{self.agent_id}"
        self.frame_idx = 0
        self.video_idx = 0
        if self.save and not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)

        self.get_logger().info(f'Keypoint Detector initialized for agent {self.agent_id}')

    def laser_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        """Callback per aggiornare la posizione del robot"""
        # Estrai posizione
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y
        
        # Estrai orientazione (conversione da quaternion a yaw)
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)
        
        # Aggiorna timestamp dell'ultima ricezione pose
        self.last_pose_time = time.time()

    # def cmd_vel_callback(self, msg):
    #     """Callback per aggiornare la velocità del robot"""
    #     self.current_velocity[0] = msg.linear.x
    #     self.current_velocity[1] = msg.linear.y

    # def update_position_from_velocity(self):
    #     """Aggiorna posizione basata su velocità se non ci sono dati di odometry recenti"""
    #     current_time = time.time()
    #     dt = current_time - self.last_position_update
        
    #     # Se non abbiamo ricevuto dati di odometry recenti, integra la velocità
    #     if (self.last_pose_time is None or 
    #         current_time - self.last_pose_time > 1.0):
            
    #         # Aggiorna posizione basata su velocità
    #         self.current_position[0] += self.current_velocity[0] * dt
    #         self.current_position[1] += self.current_velocity[1] * dt
        
    #     self.last_position_update = current_time


    def detect_keypoints(self):
        if self.latest_scan is None:
            return
        
        self.scan_data = {
                'ranges': np.array(self.latest_scan.ranges),
                'angle_min': self.latest_scan.angle_min,
                'angle_max': self.latest_scan.angle_max,
                'angle_increment': self.latest_scan.angle_increment,
                'range_min': self.latest_scan.range_min,
                'range_max': self.latest_scan.range_max,
            }

        self.get_logger().info(f'Collecting scan data for agent {self.agent_id}...')
        self.all_scans.append(self.scan_data)
        
        if len(self.all_scans) == self.frames and self.save:
            self.get_logger().info(f'Saving scan data for agent {self.agent_id}...')
            np.save(f'scan_data_agent_{self.agent_id}.npy', self.all_scans)
            self.get_logger().info(f'length of all_scans: {len(self.all_scans)}')
            self.all_scans = []  # Reset after saving

        # Implement your keypoint detection algorithm here
        keypoints, scores = self.falko.extract_keypoints(self.latest_scan)

        # add agent_id to keypoints to identify the source later in the matching process
        keypoints.agent_id = self.agent_id
        
        # Add current robot position to keypoints message
        keypoints.agent_position.x = float(self.current_position[0])
        keypoints.agent_position.y = float(self.current_position[1])
        keypoints.agent_position.theta = float(self.current_orientation)
        
        self.get_logger().info(f'Found {len(keypoints.keypoints)} keypoints for agent {self.agent_id} at position ({self.current_position[0]:.2f}, {self.current_position[1]:.2f})')
        self.keypoint_pub.publish(keypoints)

        markers_msg = self.keypoints_msg_to_markers(keypoints)
        self.marker_pub.publish(markers_msg)

        self.update_plots(keypoints, scores)

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

    def update_plots(self, keypoints_msg, scores):
        """Aggiorna i plot con i dati correnti"""
        if not self.enable_plotting or self.latest_scan is None:
            return
            
        with self.plot_lock:
            try:
                # Clear plots
                self.scan_plot.clear()
                self.score_plot.clear()
                
                # Setup assi e griglie
                self.scan_plot.set_title('Scansione Laser e Keypoint')
                self.scan_plot.set_xlabel('X [m]')
                self.scan_plot.set_ylabel('Y [m]')
                self.scan_plot.grid(True)
                self.scan_plot.axis('equal')
                
                self.score_plot.set_title('Score Keypoint')
                self.score_plot.set_xlabel('Keypoint ID')
                self.score_plot.set_ylabel('Score')
                
                self.score_plot.plot(range(len(scores)), scores, color='orange', label='Score')

                # Plot scansione laser
                scan = self.latest_scan
                angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
                x_points = np.array(scan.ranges) * np.cos(angles)
                y_points = np.array(scan.ranges) * np.sin(angles)
                self.scan_plot.scatter(x_points, y_points, s=1, c='blue', alpha=0.5, label='Laser')
                
                # Plot keypoint
                if keypoints_msg and keypoints_msg.keypoints:
                    kp_x = []
                    kp_y = []
                    scores = []
                    
                    for i, kp in enumerate(keypoints_msg.keypoints):
                        kp_x.append(kp.position.x)
                        kp_y.append(kp.position.y)
                        scores.append(kp.score)
                    
                    # Plot keypoint su scansione
                    self.scan_plot.scatter(kp_x, kp_y, s=50, c='red', marker='*', label='Keypoints')
                    
                    # Numera i keypoint
                    for i, (x, y) in enumerate(zip(kp_x, kp_y)):
                        self.scan_plot.annotate(str(i), (x, y), fontsize=8)
                # Write the number of frame
                self.scan_plot.text(0.05, 0.95, f'Frame: {self.frame_idx}', 
                                    transform=self.scan_plot.transAxes, fontsize=12, verticalalignment='top')    
                # Plot score for each points as normal plot plt

                self.scan_plot.legend()
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)  # Necessario per aggiornare
                # Salva ogni frame come immagine
                if self.save:
                    frame_path = os.path.join(self.frame_dir, f"frame_{self.frame_idx:05d}.png")
                    self.fig.savefig(frame_path)
                    self.frame_idx += 1
                    # Quando raggiungi il numero desiderato di frame, crea il video
                    if self.frame_idx % self.frames == 0:
                        self.create_video_from_frames()
            except Exception as e:
                self.get_logger().error(f"Errore nell'aggiornamento plot: {e}")

    def create_video_from_frames(self):
        try:
            video_path = f"keypoints_video_agent_{self.agent_id}.mp4"
            images = []
            file_list = sorted([f for f in os.listdir(self.frame_dir) if f.endswith('.png')])
            if not file_list:
                self.get_logger().warn("Nessun frame trovato per creare il video.")
                return
            for filename in file_list:
                img = imageio.imread(os.path.join(self.frame_dir, filename))
                images.append(img)
            imageio.mimsave(video_path, images, fps=10)
            self.get_logger().info(f"✅ Video dei plot salvato in {video_path}")
            # Cancella i frame dopo aver creato il video
            for filename in file_list:
                os.remove(os.path.join(self.frame_dir, filename))

            self.frame_idx = 0
        except Exception as e:
            self.get_logger().error(f"Errore nel salvataggio del video: {e}")

    def destroy_node(self):
        """Sovrascrivi per chiudere correttamente matplotlib e salvare video residuo"""
        try:
            plt.close('all')
        except Exception:
            pass
        # Salva eventuali frame residui come video
        if self.save and self.frame_idx > 0:
            self.create_video_from_frames()
        super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    node = KeypointDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()