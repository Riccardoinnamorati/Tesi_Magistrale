import rclpy
from choirbot.guidance.distributed_control import SimpleGuidance
from rclpy.node import Node

from geometry_msgs.msg import Vector3

def main():
    rclpy.init()

    node = Node('simpl_guid_tb', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

    freq_guidance = node.get_parameter('freq').value
    guidance = SimpleGuidance(freq_guidance, 'pubsub', 'odom')

    rclpy.spin(guidance)
    rclpy.shutdown()
