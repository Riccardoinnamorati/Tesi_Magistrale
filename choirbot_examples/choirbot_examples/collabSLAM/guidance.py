import rclpy
from choirbot.guidance.distributed_control import CollabSLAMGuidance
from rclpy.node import Node

from geometry_msgs.msg import Vector3

def main():
    rclpy.init()

    node = Node('guid_tb', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

    sampling_freq = node.get_parameter('sampling_freq').value
    receive_freq = node.get_parameter('receive_freq').value
    guidance = CollabSLAMGuidance('pubsub', 'odom', sampling_freq, receive_freq)
    

    rclpy.spin(guidance)
    rclpy.shutdown()
