import rclpy
from rclpy.node import Node
from threading import Thread, Lock
import time
from typing import Any, Optional

# Assuming ROS2 message types. Make sure they match your robot's publishers.
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Int32 # Example for detections, adjust as needed

# You'll need cv_bridge to convert ROS Image messages to numpy arrays
from cv_bridge import CvBridge

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.cameras.ros2 import ROS2CameraConfig
from lerobot.common.robots import Robot
from ..dusty_config import DustyConfig
from dataclasses import dataclass


class dusty(Robot):
    """
    A robot class that interfaces with ROS2 to get observations and send actions.
    """
    config_class = DustyConfig
    name = "dusty"

    def __init__(self, config: DustyConfig):
        super().__init__(config)
        self.config = config
        # This is for LeRobot's internal camera handling. If you get images
        # from ROS, you might not need it, but it's good to keep.
        # self.cameras = make_cameras_from_configs(config.cameras)
        self.cameras = make_cameras_from_configs({"camera": ROS2CameraConfig(
        topic="/camera/image_raw",
        fps=30,
        width=640,
        height=480,
        rotation=90  # Optional rotation
        )})
                

        # Initialize ROS2 infrastructure
        self.node = None
        self.executor_thread = None
        self.lock = Lock() # Crucial for thread-safe access to state variables
        self.bridge = CvBridge() # For converting ROS images

        # --- Internal state variables to hold the latest data from ROS topics ---
        self._latest_joint_states: Optional[dict] = None
        self._latest_camera_image: Optional[Any] = None
        self._latest_detections: Optional[int] = None
        # self._latest_htof: Optional[Any] = None
        # self._latest_overcurrent: Optional[dict] = None
        
        # --- ROS2 Publishers and Subscribers ---
        self.action_publisher = None
        self.joint_state_subscriber = None
        self.camera_subscriber = None
        self.detection_subscriber = None

    # Your property definitions are great, no changes needed here.
    @property
    def joint_states(self) -> dict[str, type]:
        return {"joint_1": float, "joint_2": float, "joint_3": float, "joint_4": float, "joint_5": float, "joint_6": float}

    @property 
    def camera_states(self) -> dict[str, tuple]:
        return {"cam": (640, 640, 3)}
    
    @property 
    def detections(self) -> dict[str, type]:
        return {"detections": int}
    
    # ... your other properties are fine ...
    @property
    def observation_features(self) -> dict:
        # Assuming you will add the other properties back in
        return {**self.joint_states, **self.camera_states, **self.detections}
    
    @property
    def action_features(self) -> dict:
        return self.joint_states


    def connect(self):
        """Initializes the ROS2 node and spins it in a background thread."""
        try:
            rclpy.init()
        except Exception:
             # Already initialized
            pass
        self.node = Node(f"{self.name}_lerobot_interface")
        self.configure() # Set up subscribers/publishers

        # Start spinning the node in a background thread to receive messages
        self.executor_thread = Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.executor_thread.start()
        self.node.get_logger().info(f"ROS2 node for '{self.name}' is up and spinning.")

    def configure(self):
        """This function sets up all ROS2 subscribers and publishers."""
        # --- Publishers ---
        self.action_publisher = self.node.create_publisher(
            JointState, self.config.action_topic, 10
        )

        # --- Subscribers ---
        self.joint_state_subscriber = self.node.create_subscription(
            JointState, self.config.joint_states_topic, self._joint_state_callback, 10
        )
        self.camera_subscriber = self.node.create_subscription(
            Image, self.config.camera_topic, self._camera_callback, 10
        )
        self.detection_subscriber = self.node.create_subscription(
            Int32, self.config.detections_topic, self._detections_callback, 10
        )
        self.node.get_logger().info("ROS2 publishers and subscribers configured.")

    def _joint_state_callback(self, msg: JointState):
        """Callback to update the latest joint states."""
        with self.lock:
            # Assuming msg.name is ['joint_1', 'joint_2', ...] and msg.position has corresponding values
            self._latest_joint_states = {name: pos for name, pos in zip(msg.name, msg.position)}

    def _camera_callback(self, msg: Image):
        """Callback to update the latest camera image."""
        with self.lock:
            # Convert ROS Image message to a NumPy array
            self._latest_camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _detections_callback(self, msg: Int32):
        """Callback to update the latest detections."""
        with self.lock:
            self._latest_detections = msg.data

    
    def get_observation(self) -> dict[str, Any]:
        """
        Reads the latest data from internal state variables.
        This is thread-safe thanks to the lock.
        """
        with self.lock:
            # Check if all required data has been received at least once
            if self._latest_joint_states is None or self._latest_camera_image is None: #or self._latest_detections is None:
                # You might want to wait a bit here on the first call
                # or raise an error if data isn't arriving.
                raise RuntimeError("Waiting for first ROS2 messages to arrive. Check topics.")

            # Create a copy of the data to avoid issues outside the lock
            obs_dict = {
                **self._latest_joint_states,
                "cam": self._latest_camera_image.copy(),
                "detections": self._latest_detections,
            }
        return obs_dict

    #TODO: Make it a goal for low level
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Publishes an action dictionary to the appropriate ROS2 topic."""
        if self.action_publisher is None:
            self.node.get_logger().warn("Action publisher is not initialized.")
            return action

        # Create a ROS2 JointState message from the action dictionary
        # The action dictionary keys must match the joint names
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(action.keys())
        msg.position = list(action.values())
        # Set velocity/effort if your controller needs them, otherwise they can be empty
        msg.velocity = []
        msg.effort = []

        self.action_publisher.publish(msg)
        return action

    def disconnect(self):
        """Shuts down the ROS2 node cleanly."""
        if self.node:
            self.node.get_logger().info("Shutting down ROS2 node.")
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.executor_thread:
            self.executor_thread.join()
    
    def calibrate(self)->None:
        print(self.is_calibrated())
    
    def is_calibrated(self)->bool:
        return True
    
    def is_connected(self)->bool:
        return True