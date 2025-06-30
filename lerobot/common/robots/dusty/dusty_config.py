from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.ros2 import ROS2CameraConfig
from lerobot.common.robots import RobotConfig

from typing import Any, Optional
@RobotConfig.register_subclass("dusty")
@dataclass
class DustyConfig(RobotConfig):
    # ROS2 Topic Names
    joint_states_topic: str = "/joint_states"
    action_topic: str = "/joint_state_goal"
    camera_topic: str = "/camera/image_raw"
    detections_topic: str = "/object_detections"
    # Add other topics as needed
    htof_topic: str = "/htop_cam/image_raw"
    # overcurrent_topic: str = "/motor_currents"

    # LeRobot camera configs (can be empty if handled by ROS)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)