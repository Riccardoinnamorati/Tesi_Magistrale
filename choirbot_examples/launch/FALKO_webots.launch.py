# copiato e modificato da pasi
import os
import numpy as np
import pathlib
import sys

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher

freq_guidance = 50  # [Hz]
freq_closest_getter = 100   # [Hz]
safe_distance = 0.3    # [m]
freq_collision_tb = 50         # [Hz]

freq_keypoint_detection = 0.5  # [Hz]
freq_keypoint_matching = 1   # [Hz]

def get_webots_driver_tb(agent_id):
    package_dir_driver = get_package_share_directory('choirbot_examples')
    robot_description = pathlib.Path(os.path.join(package_dir_driver, 'turtlebot_unicycle.urdf')).read_text()

    turtlebot_driver = Node(
        package='webots_ros2_driver',
        executable='driver',
        namespace=f'agent_{agent_id}',
        output='screen',
        additional_env={
            'WEBOTS_ROBOT_NAME':f'agent_{agent_id}',
            },
        parameters=[
            {'robot_description': robot_description},
        ]
    )

    return turtlebot_driver


def generate_webots_world_file(robots, source_filename, target_filename):
    with open(source_filename, 'r') as source_file:
        contents = source_file.read()

    with open(target_filename, 'w') as target_file:
        target_file.write(contents)

        for robot in robots:
            template_filename = os.path.join(os.path.dirname(source_filename), f'obj_{robot["type"]}.wbt')
            with open(template_filename, 'r') as template_file:
                template = template_file.read()
                template = template.replace('$NAME', robot["name"])
                template = template.replace('$X', str(robot["position"][0]))
                template = template.replace('$Y', str(robot["position"][1]))
                template = template.replace('$Z', str(robot["position"][2]))
                if robot["type"] == 'turtlebot':
                    template = template.replace('$A', str(robot["rotation"][0]))
                    template = template.replace('$B', str(robot["rotation"][1]))
                    template = template.replace('$C', str(robot["rotation"][2]))
                    template = template.replace('$D', str(robot["rotation"][3]))
                target_file.write(template)

def generate_launch_description():

    N_tb = 2
    seed = 3
    save_path = './src/crazychoir_aggregative_feedback/crazychoir_aggregative_feedback/test_tb/data/sim'


    # Get arguments from command line
    for arg in sys.argv:
        if arg.startswith("N_tb:="):
            N_tb = int(arg.split(":=")[1])
        if arg.startswith("seed:="):
            seed = int(arg.split(":=")[1])
        if arg.startswith("save_path:="):
            save_path = arg.split(":=")[1]

    np.random.seed(seed)

    N = N_tb
    # generate communication graph (this function also sets the seed)
    robots = []

    P = [
        # [-6.48743, -2.50161, 0.0], 
        #  [-2.65, 2.32, 0.0], 
        #  #[-6.48743, -2.50161, 0.0], 
        #  [1.8, -2.99, 0.0], 
         [4.2, -0.03, 0.0], 
         [6.51681, 1.26283, 0.0]
         ]
    R = [[0, 0, 1, 0], [0, 0, -1, 1.57], [0, 0, 1, 3.14], [0, 0, 1, 0], [0, 0, 1, 3.14]]

    #P = [[4.2, -0.03, 0.0], [6.51681, 1.26283, 0.0]]
    #R = [[0, 0, 1, 0], [0, 0, 1, 3.14]]

    # Flatten initial positions for plotter
    initial_positions_flat = [coord for pos in P for coord in pos[:2]]

    robots = [{
                'name': f'agent_{i}',
                'type': 'turtlebot', 
                'position': P[i],
                'rotation': R[i] 
            } for i in range(N_tb) ]

    # Target
    targets = []
    # targets.append([[-5.31,-2.50161,0.0], [-5.31,-1.04,0.0], [-6.35,-1.04,0.0], [-6.35,0.27,0.0], [-5.1,0.27,0.0], [-3.83,0.27,0.0], [-3.83,1.83,0.0], [-4.86,1.83,0.0], [-4.86,2.6,0.0], [-0.53,3,0.0], [-0.53,1.97,0.0]])
    # #targets.append([[-4.31,-1.50161,0.0], [-4.31,-0.04,0.0], [-6.35,-1.04,0.0], [-6.35,0.27,0.0], [-5.1,0.27,0.0], [-3.83,0.27,0.0], [-3.83,1.83,0.0], [-4.86,1.83,0.0], [-4.86,2.6,0.0], [-0.53,3,0.0], [-0.53,1.97,0.0]])
    
    # targets.append([[-2.65,1.97,0.0], [-2.65,0.27,0.0], [-2.65,-2.88,0.0], [-4.03,-2.88,0.0], [-4.03,-1.04,0.0], [-5.31,-1.04,0.0], [-6.35,-1.04,0.0], [-6.35,0.27,0.0], [-5.1,0.27,0.0], [-3.83,0.27,0.0], [-3.83,1.83,0.0], [-4.86,1.83,0.0], [-6.25,1.57,0.0], [-6.25,2.76,0.0]])
    # targets.append([[0.62,-2.99,0.0], [-1.15,-2.99,0.0], [-1.15,0.88,0.0], [0.62,0.88,0.0], [0.62,1.97,0.0], [-0.53,1.97,0.0], [-2.65,1.97,0.0], [-2.65,0.27,0.0], [-3.83,0.27,0.0], [-3.83,1.83,0.0]])
    targets.append([[4.66,-0.03,0.0], [6.44,-0.03,0.0], [6.44,-2.74,0.0], [4.66,-2.74,0.0], [3.41,-2.74,0.0], [3.41,-1.56,0.0], [2.05,-1.56,0.0], [2.05,-2.99,0.0], [0.62,-2.99,0.0], [0.62,0.88,0.0], [0.62,1.97,0.0], [0.62,2.77,0.0], [3.41,2.77,0.0], [6.52,2.77,0.0]])
    targets.append([[4.66,1.26,0.0], [4.66,-0.03,0.0], [4.66,-2.74,0.0], [3.41,-2.74,0.0], [3.41,-1.56,0.0], [3.41,-0.35,0.0], [2.05,-0.35,0.0], [2.05,1.17,0.0], [3.41,1.17,0.0], [3.41,2.77,0.0]])

    for i in range(len(targets)):
        robots +=[{
                    'name': f'target_{i}{j}',
                    'type': 'target', 
                    'position': targets[i][j], 
                } for j in range(len(targets[i]))]

    launch_description = []

    world_package_dir = get_package_share_directory('choirbot_examples')
    source_filename = os.path.join(world_package_dir, 'worlds', 'Test_world.wbt')
    target_filename = os.path.join(world_package_dir, 'worlds', 'aggregative_feedback.wbt')
    generate_webots_world_file(robots, source_filename, target_filename)            
    webots = WebotsLauncher(world=os.path.join(world_package_dir, 'worlds', 'aggregative_feedback.wbt'))
    launch_description.append(webots)

    # FALKO Plotter Node
    launch_description.append(Node(
        package='choirbot_examples',
        executable='choirbot_FALKO_plotter',
        output='screen',
        prefix='xterm -title "FALKO_plotter" -geometry 160x10+2460+0 -hold -e ',
        parameters=[{
            'N': N,
            'freq': 10.0,
            'initial_positions': initial_positions_flat,
        }]))

    
    # neighbors getter tb
    # launch_description.append(Node(
    #     package='crazychoir_aggregative_feedback', 
    #     executable='crazychoir_test_tb_vicon_closest_robot_getter', 
    #     output='screen', 
    #     # prefix='xterm  -title "closest_getter_tb" -geometry 160x7+1980+0 -hold -e ',
    #     remappings=[('/closest_robots_getter', '/closest_robots_getter_tb')],
    #     parameters=[{'first_id': 0, 'last_id':N_tb, 'freq': freq_closest_getter}]))


    # # Rviz2 node
    # launch_description.append(Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     output='screen',
    #     parameters=[{'use_sim_time': True}],
    #     arguments=['-d', world_package_dir + '/config.rviz'],
    # ))

    # add executables for each robot
    for i in range(N):
        in_neighbors  = [False]*N
        out_neighbors  = [False]*N
        initial_position = P[i]
        targets_list = np.array(targets[i]).flatten().tolist()

        # webots exec
        launch_description.append(get_webots_driver_tb(i))
            
        launch_description.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            additional_env={'WEBOTS_ROBOT_NAME':f'agent_{i}'},
            namespace=f'agent_{i}',
            output='screen',
            parameters=[{
                'robot_description': '<robot name=""><link name=""/></robot>',
                }]))

        # Keypoint Detection Node
        launch_description.append(Node(
            package='choirbot_examples',
            executable='choirbot_FALKO_keypoint_detector',
            output='screen',
            prefix=f'xterm -title "keypoint_detector_{i}" -geometry 160x7+2140+{int(135*i)} -hold -e ',
            namespace=f'agent_{i}',
            parameters=[{
                'agent_id': i,
                'freq': freq_keypoint_detection,
                'N': N,
                'initial_position': initial_position[:2],  # Pass only [x, y]
            }]))

        # Keypoint Matcher Node  
        launch_description.append(Node(
            package='choirbot_examples',
            executable='choirbot_FALKO_keypoint_matcher',
            output='screen',
            prefix=f'xterm -title "keypoint_matcher_{i}" -geometry 160x7+2300+{int(135*i)} -hold -e ',
            namespace=f'agent_{i}',
            parameters=[{
                'agent_id': i,
                'in_neighbors': in_neighbors,
                'freq': freq_keypoint_matching,
            }]))

        launch_description.append(Node(
            package='choirbot_examples',
            executable='choirbot_webots_guidance', 
            output='screen',
            prefix=f'xterm  -title "guidance_{i}" -geometry 160x7+1980+{int(135*(i))} -hold -e ',# if i == 0 else None,
            namespace=f'agent_{i}',
            parameters=[{
                'freq': freq_guidance, 
                'agent_id': i, 
                'N': N, 
                'in_neigh': in_neighbors, 
                'out_neigh': out_neighbors, 
                'init_pos': initial_position,
                'save_path': save_path,
                'targets': targets_list,
                }]))
            
        # controller
        launch_description.append(Node(
            package='choirbot_examples', executable='choirbot_webots_controller', output='screen',
            namespace=f'agent_{i}',
            # prefix=f'xterm  -title "controller_{i}" -geometry 160x7+1980+{int(135*(i+N))} -hold -e ',# if i == 0 else None,
            # remappings=[('cmd_vel', 'cmd_vel_des')],
            parameters=[{'agent_id': i, 'save_path': save_path, }]))
        
        # # collision avoidance
        # launch_description.append(Node(
        #     package='crazychoir_aggregative_feedback', executable='crazychoir_test_tb_vicon_collision', output='screen', 
        #     namespace=f'agent_{i}',
        #     parameters=[{
        #         'agent_id': i, 
        #         'freq': freq_collision_tb, 
        #         'safe_distance': safe_distance,
        #         'auto_trigger': True,
        #         'save_path': save_path,
        #         }]))
        

    return LaunchDescription(launch_description)