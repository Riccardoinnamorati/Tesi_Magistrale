import rclpy
from rclpy.node import Node
import numpy as np
from choirbot_interfaces.msg import KeypointArray, KeypointMatch, KeypointPair
from choirbot_examples.FALKO.Matcher import MixedMatcher
from functools import partial

class KeypointMatcher(Node):
    def __init__(self):
        super().__init__('keypoint_matcher')
        
        # Parameters
        self.declare_parameter('agent_id', 0)
        self.declare_parameter('in_neighbors', 5)
        self.declare_parameter('freq', 10.0)
        self.declare_parameter('save_path', './data')
        
        self.agent_id = self.get_parameter('agent_id').value
        self.in_neighbors = self.get_parameter('in_neighbors').value
        self.freq = self.get_parameter('freq').value
        self.save_path = self.get_parameter('save_path').value
        
        self.my_last_keypoints = KeypointArray()  # Last keypoints of the current agent

        # Subscribers for keypoints from all agents
        # REMEMBER: Will be saved also my keypoints 
        self.keypoint_subs = []
        for i in self.in_neighbors:
            sub = self.create_subscription(
                KeypointArray, f'/agent_{i}/keypoints',
                partial(self.keypoint_callback, agent_id=i), 10)
            self.keypoint_subs.append(sub)

        # Publisher for matches
        # This will publish matches for the current agent to a global topic
        # so that other agents can receive it
        self.match_global_pub = self.create_publisher(
            KeypointMatch, f'/matches_agent_{self.agent_id}', 10)
        
        # Timer
        self.timer = self.create_timer(1.0/self.freq, self.match_keypoints)
        
        # Data storage
        self.keypoints_data = {i: [] for i in self.in_neighbors}  # {agent_id: KeypointArray}
        
        self.get_logger().info(f'Keypoint Matcher initialized for agent {self.agent_id}')

    def keypoint_callback(self, msg, agent_id):
        if len(msg.keypoints) >=3: 
            last_keypoints = msg
            self.keypoints_data[agent_id].append(last_keypoints)

        if agent_id == self.agent_id:
            self.my_last_keypoints = msg
        

    def match_keypoints(self):
        if len(self.keypoints_data) < 2 :
            return
        
        if len(self.my_last_keypoints.keypoints) < 3:
            return

        my_data = self.keypoints_data.get(self.agent_id, [])
        if not my_data:  # Lista vuota
            self.get_logger().warn(f'No keypoints available for agent {self.agent_id}')
            return

        my_keypoints = my_data[-1]
        
        for neigh_id, data in self.keypoints_data.items():
            for data in data:
                if neigh_id == self.agent_id or my_keypoints is None or data is None:
                    continue
                matcher = MixedMatcher(dist_tol = 0.01)
                matches = []
                _, H = matcher.match(my_keypoints.keypoints, data.keypoints, matches)
                H_inv = np.linalg.inv(H)  # Invert the transformation matrix
                # Assuming matches is a list of tuples (index_in_my_keypoints, index_in_neigh_keypoints)
                if len(matches) >= 3 :    
                    self.get_logger().info( f'Agent {self.agent_id} found {len(matches)} matches with agent {neigh_id}')
                    if matches:
                        match_msg = KeypointMatch()
                        match_msg.agent_id = self.agent_id
                        match_msg.neigh_id = neigh_id
                        match_msg.translation = [H_inv[0, 2],H_inv[1, 2], 0.0]  # Assuming translation is in 2D
                        match_msg.theta = np.arctan2(H_inv[1, 0], H_inv[0, 0])
                        # Add complete KeypointArray for both agents
                        match_msg.agent_keypoints = my_keypoints
                        match_msg.neigh_keypoints = data 
                        
                        for i1, i2 in matches:
                            if i1 >= 0 and i2 >= 0:
                                pair = KeypointPair()
                                pair.keypoint_a = my_keypoints.keypoints[i1]
                                pair.keypoint_b = data.keypoints[i2]
                                match_msg.matches.append(pair)

                    self.match_global_pub.publish(match_msg)
                else:
                    continue
            

def main(args=None):
    rclpy.init(args=args)
    node = KeypointMatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()