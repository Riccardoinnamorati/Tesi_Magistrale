# Keypoint match structure 
uint32 agent_id                 # agent id 
uint32 neigh_id                 # neighborhood id
KeypointPair[] matches          # list of couple of matches keypoints
KeypointArray agent_keypoints   # complete keypoint array of agent_id
KeypointArray neigh_keypoints   # complete keypoint array of neigh_id
float64[3] translation          # [dx, dy, dz]
float64 theta                   # rotation angle in radians