#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from choirbot_interfaces.msg import KeypointMatch, KeypointArray
import matplotlib.pyplot as plt
import matplotlib
# Use interactive backend for live plotting
try:
    matplotlib.use('Qt5Agg')  # Interactive backend
except ImportError:
    try:
        matplotlib.use('TkAgg')  # Fallback to TkAgg
    except ImportError:
        try:
            matplotlib.use('GTK4Agg')  # Another fallback
        except ImportError:
            matplotlib.use('Agg')  # Last resort - non-interactive
            print("Warning: Using non-interactive backend for plotting")
plt.ion()  # Enable interactive mode
import numpy as np
from matplotlib.animation import FuncAnimation
import threading
import time
from functools import partial

class FALKOPlotter(Node):
    def __init__(self):
        super().__init__('falko_plotter')
        
        # Parameters
        self.declare_parameter('N', 5)
        self.declare_parameter('freq', 10.0)
        self.declare_parameter('initial_positions', [])
        
        self.N = self.get_parameter('N').value
        self.freq = self.get_parameter('freq').value
        self.initial_positions = self.get_parameter('initial_positions').value
        
        # Data storage
        self.robot_trajectories = {i: {'x': [], 'y': [], 'time': []} for i in range(self.N)}
        self.robot_positions = {i: [0.0, 0.0] for i in range(self.N)}
        self.robot_velocities = {i: [0.0, 0.0] for i in range(self.N)}
        self.robot_colors = {i: 'blue' for i in range(self.N)}
        self.robot_keypoints = {i: None for i in range(self.N)}  # Store latest keypoints for each robot
        self.match_events = []
        self.current_matches = {}
        self.latest_match = None  # Store the most recent match for visualization
        
        # Set initial positions if provided
        if len(self.initial_positions) >= self.N * 2:
            for i in range(self.N):
                self.robot_positions[i] = [self.initial_positions[i*2], self.initial_positions[i*2+1]]
                self.robot_trajectories[i]['x'].append(self.initial_positions[i*2])
                self.robot_trajectories[i]['y'].append(self.initial_positions[i*2+1])
                self.robot_trajectories[i]['time'].append(time.time())
        
        # Subscribers for robot poses/odometry
        self.pose_subs = []
        self.odom_subs = []
        self.cmd_vel_subs = []
        self.keypoint_subs = []
        
        for i in range(self.N):
            # Try to subscribe to different possible topics for Webots
            pose_sub = self.create_subscription(
                PoseStamped, f'/agent_{i}/pose',
                partial(self.pose_callback, agent_id=i), 10)
            self.pose_subs.append(pose_sub)
            
            # Webots odometry topic
            odom_sub = self.create_subscription(
                Odometry, f'/agent_{i}/odom',
                partial(self.odom_callback, agent_id=i), 10)
            self.odom_subs.append(odom_sub)
            
            # Webots GPS topic (if available)
            # gps_sub = self.create_subscription(
            #     PoseStamped, f'/agent_{i}/gps',
            #     partial(self.pose_callback, agent_id=i), 10)
            # self.pose_subs.append(gps_sub)
            
            # cmd_vel_sub = self.create_subscription(
            #     Twist, f'/agent_{i}/cmd_vel',
            #     partial(self.cmd_vel_callback, agent_id=i), 10)
            # self.cmd_vel_subs.append(cmd_vel_sub)
            
            # Keypoints subscription
            keypoint_sub = self.create_subscription(
                KeypointArray, f'/agent_{i}/keypoints',
                partial(self.keypoint_callback, agent_id=i), 10)
            self.keypoint_subs.append(keypoint_sub)
        
        # Subscribers for matches
        self.match_subs = []
        for i in range(self.N):
            match_sub = self.create_subscription(
                KeypointMatch, f'/matches_agent_{i}',
                partial(self.match_callback, agent_id=i), 10)
            self.match_subs.append(match_sub)
        
        # Timer for updating positions based on velocities
        self.timer = self.create_timer(1.0/self.freq, self.update_positions)
        
        # Timer for updating plots (più frequente per aggiornamento live)
        self.plot_timer = self.create_timer(0.1, self.update_live_plot)
        
        # Plot setup
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('FALKO Multi-Robot Plotter', fontsize=16)
        self.ax_traj = self.axes[0]
        self.ax_matches = self.axes[1]
        
        # Initialize trajectory plot
        self.ax_traj.set_title('Robot Trajectories')
        self.ax_traj.set_xlabel('X [m]')
        self.ax_traj.set_ylabel('Y [m]')
        self.ax_traj.grid(True)
        self.ax_traj.set_aspect('equal')
        
        # Initialize match plot
        self.ax_matches.set_title('Keypoint Matches')
        self.ax_matches.set_xlabel('X [m]')
        self.ax_matches.set_ylabel('Y [m]')
        self.ax_matches.grid(True)
        self.ax_matches.set_aspect('equal')
        
        # Robot trajectory lines and points
        self.traj_lines = {}
        self.robot_points = {}
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i in range(self.N):
            color = colors[i % len(colors)]
            self.robot_colors[i] = color
            line, = self.ax_traj.plot([], [], color=color, linewidth=2, label=f'Robot {i}')
            point, = self.ax_traj.plot([], [], 'o', color=color, markersize=8)
            self.traj_lines[i] = line
            self.robot_points[i] = point
        
        self.ax_traj.legend()
        
        # Lock per thread safety
        self.plot_lock = threading.Lock()
        
        # Show plot
        plt.show(block=False)
        
        self.get_logger().info(f'FALKO Plotter initialized for {self.N} robots')
    
    def pose_callback(self, msg, agent_id):
        """Callback for pose messages"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        self.robot_positions[agent_id] = [x, y]
        self.robot_trajectories[agent_id]['x'].append(x)
        self.robot_trajectories[agent_id]['y'].append(y)
        self.robot_trajectories[agent_id]['time'].append(time.time())
        
        # Keep only last 1000 points to avoid memory issues
        if len(self.robot_trajectories[agent_id]['x']) > 1000:
            self.robot_trajectories[agent_id]['x'].pop(0)
            self.robot_trajectories[agent_id]['y'].pop(0)
            self.robot_trajectories[agent_id]['time'].pop(0)
    
    def odom_callback(self, msg, agent_id):
        """Callback for odometry messages"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        self.robot_positions[agent_id] = [x, y]
        self.robot_trajectories[agent_id]['x'].append(x)
        self.robot_trajectories[agent_id]['y'].append(y)
        self.robot_trajectories[agent_id]['time'].append(time.time())
        
        # Keep only last 1000 points to avoid memory issues
        if len(self.robot_trajectories[agent_id]['x']) > 1000:
            self.robot_trajectories[agent_id]['x'].pop(0)
            self.robot_trajectories[agent_id]['y'].pop(0)
            self.robot_trajectories[agent_id]['time'].pop(0)
    
    def cmd_vel_callback(self, msg, agent_id):
        """Callback for velocity commands"""
        self.robot_velocities[agent_id] = [msg.linear.x, msg.linear.y]
    
    def keypoint_callback(self, msg, agent_id):
        """Callback for keypoint messages"""
        if len(msg.keypoints) >= 3:  # Only store if there are enough keypoints
            self.robot_keypoints[agent_id] = msg
    
    def match_callback(self, msg, agent_id):
        """Callback for keypoint matches"""
        current_time = time.time()
        
        if len(msg.matches) > 0:
            # Extract neighbor position from the KeypointArray
            neigh_position = [msg.neigh_keypoints.agent_position.x, msg.neigh_keypoints.agent_position.y]
            
            match_info = {
                'agent_id': msg.agent_id,
                'neigh_id': msg.neigh_id,
                'time': current_time,
                'num_matches': len(msg.matches),
                'matches': msg.matches,
                'agent_keypoints': msg.agent_keypoints,  # Complete KeypointArray of agent who found match
                'neigh_keypoints': msg.neigh_keypoints,  # Complete KeypointArray of neighbor
                'neigh_position_at_match': neigh_position  # Position when neighbor detected those keypoints
            }
            self.match_events.append(match_info)
            
            # Store the latest match for detailed visualization
            self.latest_match = match_info
            
            # Highlight only the agent that found the match
            self.current_matches[msg.agent_id] = current_time
            
            # Keep only last 50 match events
            if len(self.match_events) > 50:
                self.match_events.pop(0)
            
            self.get_logger().info(f'Match detected: Agent {msg.agent_id} <-> Agent {msg.neigh_id}, {len(msg.matches)} matches')
    
    def update_positions(self):
        """Update robot positions based on velocities if no pose data available"""
        current_time = time.time()
        dt = 1.0 / self.freq
        
        for i in range(self.N):
            # If we haven't received pose data recently, integrate velocity
            if (len(self.robot_trajectories[i]['time']) == 0 or 
                current_time - self.robot_trajectories[i]['time'][-1] > 1.0):
                
                # Update position based on velocity
                self.robot_positions[i][0] += self.robot_velocities[i][0] * dt
                self.robot_positions[i][1] += self.robot_velocities[i][1] * dt
                
                # Add to trajectory
                self.robot_trajectories[i]['x'].append(self.robot_positions[i][0])
                self.robot_trajectories[i]['y'].append(self.robot_positions[i][1])
                self.robot_trajectories[i]['time'].append(current_time)
    
    def update_live_plot(self):
        """Aggiorna il plot live in tempo reale"""
        if not plt.fignum_exists(self.fig.number):
            # La figura è stata chiusa, non fare nulla
            return
            
        with self.plot_lock:
            try:
                current_time = time.time()
                
                # Clear both axes
                self.ax_traj.clear()
                self.ax_matches.clear()
                
                # Setup trajectory plot
                self.ax_traj.set_title('Robot Trajectories')
                self.ax_traj.set_xlabel('X [m]')
                self.ax_traj.set_ylabel('Y [m]')
                self.ax_traj.grid(True)
                self.ax_traj.set_aspect('equal')
                
                # Setup match plot
                self.ax_matches.set_title('Latest Keypoint Matches')
                self.ax_matches.set_xlabel('X [m]')
                self.ax_matches.set_ylabel('Y [m]')
                self.ax_matches.grid(True)
                self.ax_matches.set_aspect('equal')
                
                # Plot trajectories and current positions
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                all_x = []
                all_y = []
                
                for i in range(self.N):
                    color = colors[i % len(colors)]
                    
                    if len(self.robot_trajectories[i]['x']) > 0:
                        # Plot trajectory
                        self.ax_traj.plot(self.robot_trajectories[i]['x'], 
                                         self.robot_trajectories[i]['y'], 
                                         color=color, linewidth=2, label=f'Robot {i}')
                        
                        # Plot current position
                        current_color = color
                        current_size = 8
                        
                        # Highlight only the robot that found the match (not both)
                        if i in self.current_matches and current_time - self.current_matches[i] < 3.0:
                            current_color = 'red'
                            current_size = 12
                        
                        self.ax_traj.plot(self.robot_positions[i][0], 
                                         self.robot_positions[i][1], 
                                         'o', color=current_color, markersize=current_size)
                        
                        # Plot keypoints for this robot (translated to global coordinates)
                        if self.robot_keypoints[i] is not None:
                            self.plot_robot_keypoints(i, color)
                        
                        all_x.extend(self.robot_trajectories[i]['x'])
                        all_y.extend(self.robot_trajectories[i]['y'])
                
                # Plot neighbor position when it detected the matched keypoints
                if self.latest_match and current_time - self.latest_match['time'] < 3.0:
                    neigh_pos = self.latest_match['neigh_position_at_match']
                    neigh_id = self.latest_match['neigh_id']
                    neigh_color = colors[neigh_id % len(colors)]
                    
                    # Plot the neighbor's position when the keypoints were detected (cross marker)
                    self.ax_traj.plot(neigh_pos[0], neigh_pos[1], 
                                     'x', color=neigh_color, markersize=15, markeredgewidth=3,
                                     label=f'Robot {neigh_id} pos at keypoint detection')
                    
                    # Add text annotation
                    self.ax_traj.annotate(f'Robot {neigh_id}\n@keypoint time', 
                                        (neigh_pos[0], neigh_pos[1]),
                                        xytext=(10, 10), textcoords='offset points',
                                        fontsize=8, 
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    
                    # Update axis limits to include the historical position
                    all_x.append(neigh_pos[0])
                    all_y.append(neigh_pos[1])
                
                # Set axis limits
                # if all_x and all_y:
                #     margin = 1.0
                #     self.ax_traj.set_xlim(min(all_x) - margin, max(all_x) + margin)
                #     self.ax_traj.set_ylim(min(all_y) - margin, max(all_y) + margin)
                
                self.ax_traj.legend()
                
                # Plot the latest keypoint matches
                if self.latest_match and current_time - self.latest_match['time'] < 10.0:
                    self.plot_keypoint_matches(self.latest_match)
                else:
                    # Show message when no recent matches
                    self.ax_matches.text(0.5, 0.5, 'No recent matches\n(waiting for keypoint matches)', 
                                       transform=self.ax_matches.transAxes, 
                                       ha='center', va='center', fontsize=12,
                                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
                # Update the plot
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)  # Necessario per aggiornare
                
            except Exception as e:
                self.get_logger().error(f"Errore nell'aggiornamento plot live: {e}")
    
    def plot_robot_keypoints(self, robot_id, robot_color):
        """Plot keypoints for a specific robot translated to global coordinates"""
        try:
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
                                   s=15, c=robot_color, marker='.', alpha=0.6, 
                                   label=f'Robot {robot_id} keypoints' if robot_id == 0 else "")
                
        except Exception as e:
            self.get_logger().error(f"Errore nel plottare keypoints robot {robot_id}: {e}")
    
    def plot_keypoint_matches(self, match_info):
        """Plotta i keypoint matchati nel subplot di destra"""
        try:
            agent_id = match_info['agent_id']
            neigh_id = match_info['neigh_id']
            matches = match_info['matches']
            
            # Colori per i due robot
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            color_a = colors[agent_id % len(colors)]
            color_b = colors[neigh_id % len(colors)]
            
            # Estrai le posizioni dei keypoint
            keypoints_a_x = []
            keypoints_a_y = []
            keypoints_b_x = []
            keypoints_b_y = []
            
            for match_pair in matches:
                # Keypoint del primo robot (keypoint_a)
                kp_a = match_pair.keypoint_a
                keypoints_a_x.append(kp_a.position.x)
                keypoints_a_y.append(kp_a.position.y)
                
                # Keypoint del secondo robot (keypoint_b)
                kp_b = match_pair.keypoint_b
                keypoints_b_x.append(kp_b.position.x)
                keypoints_b_y.append(kp_b.position.y)
            
            # Plotta i keypoint del primo robot
            self.ax_matches.scatter(keypoints_a_x, keypoints_a_y, 
                                  c=color_a, s=80, marker='o', 
                                  label=f'Robot {agent_id} keypoints', alpha=0.8)
            
            # Plotta i keypoint del secondo robot
            self.ax_matches.scatter(keypoints_b_x, keypoints_b_y, 
                                  c=color_b, s=80, marker='^', 
                                  label=f'Robot {neigh_id} keypoints', alpha=0.8)
            
            # Connetti i keypoint matchati con linee
            for i, (xa, ya, xb, yb) in enumerate(zip(keypoints_a_x, keypoints_a_y, 
                                                    keypoints_b_x, keypoints_b_y)):
                # Linea di connessione
                self.ax_matches.plot([xa, xb], [ya, yb], 'k--', alpha=0.5, linewidth=1)
                
                # Numera i match
                mid_x = (xa + xb) / 2
                mid_y = (ya + yb) / 2
                self.ax_matches.annotate(str(i), (mid_x, mid_y), 
                                       fontsize=8, ha='center', va='center',
                                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
            
            # Aggiungi informazioni sul match
            info_text = f'Match: Agent {agent_id} ↔ Agent {neigh_id}\n{len(matches)} keypoint pairs'
            self.ax_matches.text(0.02, 0.98, info_text, 
                               transform=self.ax_matches.transAxes, 
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Imposta i limiti degli assi
            if keypoints_a_x or keypoints_b_x:
                all_x = keypoints_a_x + keypoints_b_x
                all_y = keypoints_a_y + keypoints_b_y
                margin = 0.5
                self.ax_matches.set_xlim(min(all_x) - margin, max(all_x) + margin)
                self.ax_matches.set_ylim(min(all_y) - margin, max(all_y) + margin)
            
            self.ax_matches.legend(loc='upper right')
            
        except Exception as e:
            self.get_logger().error(f"Errore nel plottare i match: {e}")
            # Mostra messaggio di errore
            self.ax_matches.text(0.5, 0.5, f'Error plotting matches:\n{str(e)}', 
                               transform=self.ax_matches.transAxes, 
                               ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def run_plot(self):
        """Deprecated - no longer used"""
        pass
    
    def destroy_node(self):
        """Clean up matplotlib resources"""
        try:
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FALKOPlotter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
