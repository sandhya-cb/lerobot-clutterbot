import rclpy
from rclpy.node import Node
from threading import Thread, Lock
import time
from typing import Any, Optional

# --- STANDARD IMPORTS ONLY ---
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Int32
from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.ros2 import ROS2CameraConfig
from ... import Robot
from ..dusty_config import DustyConfig
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class dusty_isaac(Robot):
    """
    A robot class that interfaces with Isaac Sim via standard ROS2 JointState messages.
    """
    config_class = DustyConfig
    name = "dusty_isaac"

    def __init__(self, config: DustyConfig):
        super().__init__(config)
        self.config = config
        
        # --- ISAAC SIM MAPPING ---
        # Map: { "Isaac_Sim_USD_Joint_Name": "Your_Internal_Key" }
        self.isaac_joint_map = {
            "arm_left_joint": "arm_left_angle",
            "arm_right_joint": "arm_right_angle",
            "palm_left_joint": "palm_left_angle",
            "palm_right_joint": "palm_right_angle",
            # "scoop_lift_joint": "scoop_lift_angle",
        }
        
        # Create a reverse map for sending actions efficiently: { "Internal_Key": "Isaac_Name" }
        self.internal_to_isaac_map = {v: k for k, v in self.isaac_joint_map.items()}

        self.cameras = make_cameras_from_configs({
            "camera_raw": ROS2CameraConfig(
                fps=10,
                topic=config.camera_raw_topic,
            ), 
            "camera_segmented": ROS2CameraConfig(
                fps=10,
                topic=config.camera_segmented_topic,
                channels=3,
            ),
            "rs_camera_depth": ROS2CameraConfig(
                topic=config.depth_img_topic,
                fps=10,
                width=109,
                height=224,
                channels=1,
            ), 
        })
                
        # Initialize ROS2 infrastructure
        self.node = None
        self.executor_thread = None
        self.lock = Lock()
        self.bridge = CvBridge()

        # --- Internal state variables ---
        self._latest_joint_states: Optional[dict] = None
        self._latest_detections: Optional[int] = None
        self._latest_htof: Optional[Any] = None
        
        # --- ROS2 Publishers and Subscribers ---
        self.action_publisher = None
        self.joint_state_subscriber = None
        self.detection_subscriber = None

    @property
    def joint_states(self) -> dict[str, type]:
        return {
            "arm_left_angle": float,
            "arm_right_angle": float,
            # "scoop_lift_angle": float,
            "palm_left_angle": float,
            "palm_right_angle": float,
            # "scoop_tilt_angle": float
        }

    @property   
    def camera_states(self) -> dict[str, tuple]:
        return {"camera_raw": (640, 852, 3), "camera_segmented": (320, 320, 3), "camera_depth": (109, 224, 3)}
    
    @property 
    def detections(self) -> dict[str, type]:
        return {"detections": int}
    
    @property
    def htof(self) -> dict[str, tuple]:
        return {"htof": (320, 320, 4)}

    @property
    def observation_features(self) -> dict:
        return {**self.joint_states, **self.camera_states}
    
    @property
    def action_features(self) -> dict:
        return self.joint_states.copy()
    
    def get_action(self) -> dict[str, float]:
        return self._latest_joint_states

    def connect(self):
        try:
            rclpy.init()
        except Exception:
            pass
        self.node = Node(f"{self.name}_lerobot_interface")
        self.configure() 
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        self.executor_thread = Thread(target=self.executor.spin, daemon=False)
        self.executor_thread.start()
        
        self.node.get_logger().info("Waiting for initial msg.")
        while self._latest_joint_states is None:
            time.sleep(0.1)
            
        self.node.get_logger().info("ROS2 node is up and spinning.")

    def configure(self):
        # --- Publisher (ACTIONS) ---
        # We now publish standard JointState messages to control the robot
        self.action_publisher = self.node.create_publisher(
            JointState, self.config.action_topic, 10
        )

        # --- Subscriber (OBSERVATIONS) ---
        # We listen to standard JointState messages from Isaac Sim
        self.joint_state_subscriber = self.node.create_subscription(
            JointState, 
            self.config.joint_states_topic, 
            self._joint_state_callback, 
            10
        )
        
        for key, _ in self.cameras.items():
            self.cameras[key].connect()
            
        self.detection_subscriber = self.node.create_subscription(
            Int32, self.config.detections_topic, self._detections_callback, 10
        )
        self.htof_group = ReentrantCallbackGroup()
        self.htof_subscriber = self.node.create_subscription(
            PointCloud2, self.config.htof_topic, self._htof_callback, 10, callback_group=self.htof_group
        )

    def _joint_state_callback(self, msg: JointState):
        """
        Receives standard JointState from Isaac Sim and maps to internal keys.
        """
        temp_states = {}
        for i, joint_name in enumerate(msg.name):
            if joint_name in self.isaac_joint_map:
                internal_name = self.isaac_joint_map[joint_name]
                temp_states[internal_name] = float(msg.position[i])

        with self.lock:
            if not self._latest_joint_states:
                self._latest_joint_states = {}
            self._latest_joint_states.update(temp_states)

    def _htof_callback(self, msg: PointCloud2):
        with self.lock:
            pc_array = point_cloud2.read_points_numpy(msg)
            self._latest_htof = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in pc_array]

    def _detections_callback(self, msg: Int32):
        with self.lock:
            self._latest_detections = msg.data
    
    def get_observation(self) -> dict[str, Any]:
        with self.lock:
            obs_dict = {
                **(self._latest_joint_states or {}),
                "camera_raw": self.cameras["camera_raw"].async_read(),
                "camera_segmented": self.cameras["camera_segmented"].async_read(),
                "camera_depth": self.cameras["rs_camera_depth"].async_read()
            }
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Converts internal action dict to a standard ROS2 JointState message.
        """
        if self.action_publisher is None:
            return action
            
        msg = JointState()
        # Optional: Add timestamp if Isaac Sim requires it, usually it's fine without for basic teleop
        # msg.header.stamp = self.node.get_clock().now().to_msg()
        
        with self.lock:
            for internal_name, joint_value in action.items():
                # Find the Isaac Sim name for this internal key
                if internal_name in self.internal_to_isaac_map:
                    isaac_name = self.internal_to_isaac_map[internal_name]
                    
                    msg.name.append(isaac_name)
                    msg.position.append(float(joint_value))
        
        # Only publish if we have data
        if msg.name:
            self.action_publisher.publish(msg)

        return action
    
    # Removed get_joint_id() as it is no longer needed

    def disconnect(self):
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.executor_thread:
            self.executor_thread.join()
    
    def calibrate(self)->None:
        pass
    
    def is_calibrated(self)->bool:
        return True
    
    def is_connected(self)->bool:
        return True13.230