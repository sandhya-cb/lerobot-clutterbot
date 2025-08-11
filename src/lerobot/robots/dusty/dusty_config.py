from dataclasses import dataclass, field

from lerobot.cameras.utils import CameraConfig
from lerobot.cameras.ros2 import ROS2CameraConfig
from .. import RobotConfig

from typing import Any, Optional
@RobotConfig.register_subclass("dusty")
@dataclass
class DustyConfig(RobotConfig):
    # ROS2 Topic Names
    name: str = "dusty"
    joint_states_topic: str = "/actuator_feedback"
    action_topic: str = "/leader/actuator_diag"
    camera_segmented_topic: str = "/camera_0/detection/image_overlay"
    camera_raw_topic: str = "/camera_0/image_raw"
    detections_topic: str = "/object_detections"
    # Add other topics as needed
    htof_topic: str = "/depth_cloud"
    # overcurrent_topic: str = "/motor_currents"

    # LeRobot camera configs (can be empty if handled by ROS)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
