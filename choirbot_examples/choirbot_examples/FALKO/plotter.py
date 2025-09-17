import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
import os
from choirbot_interfaces.msg import KeypointArray
from functools import partial
import imageio
import datetime

class Plotter(Node):

    def __init__(self):
        super().__init__(f"plotter_subscriber")

        self.N = 4  # Number of agents
        self.pose = {}
        self.keypoint_subs = []
        self.robot_keypoints = {i: None for i in range(self.N)}  # Store latest keypoints for each robot
        self.robot_positions = {}  # Store latest positions for each robot

        self.plot_timer = self.create_timer(0.1, self.listener_callback)  # Timer to trigger plot updates

        for agent in range(self.N):
            self.subscription = self.create_subscription(
                Float64MultiArray,
                f"/agent_{agent}/agent{agent}_plot",
                self.plot_callback,
                10
            )     
            keypoint_sub = self.create_subscription(
                KeypointArray, f'/agent_{agent}/keypoints',
                partial(self.keypoint_callback, agent_id=agent), 10)

            self.robot_pos_sub = self.create_subscription(
                Float64MultiArray, f'/agent_{agent}/pose',
                partial(self.robot_pose_callback, agent_id=agent), 10)

        self.keypoint_subs.append(keypoint_sub)
        
        # Imposta la directory padre src come base per salvataggio video
        self.script_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
        self.script_dir_img = os.path.dirname(os.path.realpath(__file__))
        # Variabili per salvataggio frame e video
        self.save_frames = True  # Attiva/disattiva salvataggio frame
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_dir = os.path.join(self.script_dir, "frames_plot", f"run_{run_id}")
        self.frame_idx = 0
        self.frames_per_video = 300  # Cambia a piacere
        if self.save_frames and not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)

    def robot_pose_callback(self, msg, agent_id):

        msg.data.pop(0)

        if agent_id not in self.robot_positions:
            self.robot_positions[agent_id] = []

        self.robot_positions[agent_id].append(msg.data)

        self.get_logger().info(f"self.robot_positions[{agent_id}]: {self.robot_positions[agent_id]}")

    def plot_callback(self, msg):
        id = int(msg.data[0])
        msg.data.pop(0)

        self.pose[id] = msg.data
        

    def listener_callback(self):
        #plt.margins(x=10, y=10)
        plt.title(f"Full Plot")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        plt.axis('equal')  # Ensure equal scaling for both axes
        img_path = os.path.join(self.script_dir_img, "bg.png")
        img = plt.imread(img_path)
        plt.imshow(img, extent=[-7.2,7.2,-3.53,3.53], aspect='auto')

        for id in self.pose.keys():
            x_vals = []
            y_vals = []

            # plot keypoints for each robot
            if self.robot_keypoints[id] is not None and id ==9:
                keypoints_msg = self.robot_keypoints[id]
                robot_pos = self.pose[id][-3:]
                
                # Extract keypoint positions relative to robot
                kp_x_local = []
                kp_y_local = []
                
                for kp in keypoints_msg.keypoints:
                    kp_x_local.append(kp.position.x)
                    kp_y_local.append(kp.position.y)
                
                if kp_x_local and kp_y_local:
                    # Translate keypoints to global coordinates (robot position + keypoint relative position) + rotation
                    kp_x_local = np.array(kp_x_local)
                    kp_y_local = np.array(kp_y_local)
                    kp_x_local = kp_x_local * np.cos(-robot_pos[2]) - kp_y_local * np.sin(-robot_pos[2])
                    kp_y_local = kp_x_local * np.sin(-robot_pos[2]) + kp_y_local * np.cos(-robot_pos[2])
                    kp_x_global = [x + robot_pos[0] for x in kp_x_local]
                    kp_y_global = [y + robot_pos[1] for y in kp_y_local]

                    
                    # Plot keypoints as small dots
                    plt.scatter(kp_x_global, kp_y_global, 
                                        s=15, marker='.', alpha=0.6)
                        

            # Plot trajectory
            for i in range(len(self.pose[id])//3):
                pose_data = self.pose[id][i*3:i*3+3]

                # Extract x, y, and theta values
                x = pose_data[0]
                x_vals.append(pose_data[0])
                y = pose_data[1]
                y_vals.append(pose_data[1])
                theta = pose_data[2]

                # Add arrows to show direction (pose orientation)
                dx = np.cos(theta) * 0.1  # Arrow length for orientation
                dy = np.sin(theta) * 0.1
                plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
            
            plt.plot(x_vals, y_vals, 'o-', label="Trajectory", markersize=5)
        for id in range(self.N):
            if id in self.robot_positions:
                trajectory = self.robot_positions[id]
                if len(trajectory) > 0:
                    x_real = [pose[0] for pose in trajectory]
                    y_real = [pose[1] for pose in trajectory]
                    plt.plot(x_real, y_real, '--', markersize=3, label=f'Traiettoria agente {id}')

        #plt.legend()
            # Salva il frame come immagine
            if self.save_frames:
                frame_path = os.path.join(self.frame_dir, f"frame_{self.frame_idx:05d}.png")
                plt.savefig(frame_path)
                self.frame_idx += 1
                # Quando raggiungi il numero desiderato di frame, crea il video
                if self.frame_idx % self.frames_per_video == 0:
                    self.create_video_from_frames()
        plt.pause(0.001)
        plt.clf()

    def create_video_from_frames(self, fps=10):
        try:
            # Trova l'ultima cartella creata in frames_plot
            frames_root = os.path.join(self.script_dir, "frames_plot")
            run_dirs = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
            if not run_dirs:
                self.get_logger().error("Nessuna cartella run trovata in frames_plot.")
                return
            last_run = sorted(run_dirs)[-1]
            last_run_dir = os.path.join(frames_root, last_run)
            video_dir = os.path.join(self.script_dir, "videos")
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            video_path = os.path.join(video_dir, f"plot_video_{last_run}.mp4")
            images = []
            file_list = sorted([f for f in os.listdir(last_run_dir) if f.endswith('.png')])
            if not file_list:
                self.get_logger().warn("Nessun frame trovato per creare il video.")
                return
            for filename in file_list:
                img = imageio.imread(os.path.join(last_run_dir, filename))
                images.append(img)
            imageio.mimsave(video_path, images, fps=fps)
            self.get_logger().info(f"✅ Video dei plot salvato in {video_path}")
            # Cancella i frame dopo aver creato il video
            for filename in file_list:
                os.remove(os.path.join(last_run_dir, filename))
            self.frame_idx = 0
        except Exception as e:
            self.get_logger().error(f"Errore nel salvataggio del video: {e}")
    
    def keypoint_callback(self, msg, agent_id):
        """Callback for keypoint messages"""
        if len(msg.keypoints) >= 3:  # Only store if there are enough keypoints
            self.robot_keypoints[agent_id] = msg

    def plot_robot_keypoints(self, robot_id):
        """Plot keypoints for a specific robot translated to global coordinates"""

        keypoints_msg = self.robot_keypoints[robot_id]
        robot_pos = self.robot_positions[robot_id]
        
        # Extract keypoint positions relative to robot
        kp_x_local = []
        kp_y_local = []
        
        for kp in keypoints_msg.keypoints:
            kp_x_local.append(kp.position.x)
            kp_y_local.append(kp.position.y)
        
        if kp_x_local and kp_y_local:
            # Translate keypoints to global coordinates (robot position + keypoint relative position)
            kp_x_global = [x + robot_pos[0] for x in kp_x_local]
            kp_y_global = [y + robot_pos[1] for y in kp_y_local]
            
            # Plot keypoints as small dots
            self.ax_traj.scatter(kp_x_global, kp_y_global, 
                                s=15, marker='.', alpha=0.6, 
                                #label=f'Robot {robot_id} keypoints' if robot_id == 0 else ""
                                )
                

def main(args=None):
    rclpy.init(args=args)
    agent = Plotter()
    plt.ion()
    plt.show()
    plt.figure(figsize=(6, 3))


    def handle_signal(signum, frame):
        agent.get_logger().info(f'Received signal {signum}, saving video and shutting down...')
        agent.create_video_from_frames()
        agent.destroy_node()
        rclpy.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Shutting down...')
        agent.create_video_from_frames()
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()