import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
import os
from choirbot_interfaces.msg import KeypointArray
from functools import partial

class Plotter(Node):
    def __init__(self):
        super().__init__(f"plotter_subscriber")

        self.N = 4  # Number of agents
        self.pose = {}
        self.keypoint_subs = []
        self.robot_keypoints = {i: None for i in range(self.N)}  # Store latest keypoints for each robot

        for agent in range(self.N):
            self.subscription = self.create_subscription(
                Float64MultiArray,
                f"/agent_{agent}/agent{agent}_plot",
                self.listener_callback,
                10
            )     
            keypoint_sub = self.create_subscription(
                KeypointArray, f'/agent_{agent}/keypoints',
                partial(self.keypoint_callback, agent_id=agent), 10)
            
        self.keypoint_subs.append(keypoint_sub)
        self.script_dir = os.path.dirname(os.path.realpath(__file__))

    def listener_callback(self, msg):
        #plt.margins(x=10, y=10)
        plt.title(f"Full Plot")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling for both axes
        img_path = os.path.join(self.script_dir, "bg.png")
        img = plt.imread(img_path)
        plt.imshow(img, extent=[-7.2,7.2,-3.53,3.53], aspect='auto')

        id = int(msg.data[0])
        msg.data.pop(0)

        self.pose[id] = msg.data

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
                plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1, fc='red', ec='red')
            
            plt.plot(x_vals, y_vals, 'o-', label="Trajectory", markersize=5)

        plt.pause(0.001)
        plt.clf()
    
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
                                label=f'Robot {robot_id} keypoints' if robot_id == 0 else "")
                

def main(args=None):
    rclpy.init(args=args)
    agent = Plotter()
    plt.ion()
    plt.show()
    plt.figure(figsize=(6, 3))

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Shutting down...')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()