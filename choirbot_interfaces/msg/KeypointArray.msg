# Keypoint array structure
std_msgs/Header header
geometry_msgs/Pose2D agent_position     # 3D position of agent for this scan
uint32 agent_id                 # agent id 
Keypoint[] keypoints            # Keypoint list
uint32 total_scan_points        # Total number of scan points
uint32 vertex_index             # graph vertex index