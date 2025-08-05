import rclpy
from rclpy.node import Node
from threading import Thread, Lock
import time
from typing import Any, Optional

# Assuming ROS2 message types. Make sure they match your robot's publishers.
from sensor_msgs.msg import JointState, Image, PointCloud2
from ll_msgs.msg import ActuatorFeedback, ActuatorDiag
from std_msgs.msg import Int32 # Example for detections, adjust as needed
from sensor_msgs_py import point_cloud2
# You'll need cv_bridge to convert ROS Image messages to numpy arrays
from cv_bridge import CvBridge
import time
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.ros2 import ROS2CameraConfig
from ... import Robot
from ..dusty_config import DustyConfig
from dataclasses import dataclass
from rclpy.callback_groups import ReentrantCallbackGroup

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
        topic=config.camera_topic,
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
        self._latest_htof: Optional[Any] = None
        # self._latest_overcurrent: Optional[dict] = None
        
        # --- ROS2 Publishers and Subscribers ---
        self.action_publisher = None
        self.joint_state_subscriber = None
        self.camera_subscriber = None
        self.detection_subscriber = None

    # Your property definitions are great, no changes needed here.
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
        return {"cam": (320, 320, 3)}
    
    @property 
    def detections(self) -> dict[str, type]:
        return {"detections": int}
    
    @property
    def htof(self) -> dict[str, tuple]:
        return {"htof": (320, 320, 4)}

    @property
    def observation_features(self) -> dict:
        # Assuming you will add the other properties back in
        return {**self.joint_states, **self.camera_states, **self.detections}
    
    @property
    def action_features(self) -> dict:
        return self.joint_states.copy()
    
    def get_action(self) -> dict[str, float]:
        return self._latest_joint_states


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
        self.executor_thread = Thread(target=rclpy.spin, args=(self.node,), daemon=False)
        self.executor_thread.start()
          # Wait for innitial msg
        self.node.get_logger().info("Waiting for initiasl msg.")
        while self._latest_joint_states is None or self._latest_camera_image is None or self._latest_htof is None:
            time.sleep(0.1)
            
        self.node.get_logger().info("ROS2 publishers and subscribers configured.")
        self.node.get_logger().info(f"ROS2 node for '{self.name}' is up and spinning.")

    def configure(self):
        """This function sets up all ROS2 subscribers and publishers."""
        # --- Publishers ---
        self.action_publisher = self.node.create_publisher(
            ActuatorDiag, self.config.action_topic, 10
        )

        # --- Subscribers ---
        self.joint_state_subscriber = self.node.create_subscription(
            ActuatorFeedback, self.config.joint_states_topic, self._joint_state_callback, 10
        )
        self.camera_subscriber = self.node.create_subscription(
            Image, self.config.camera_topic, self._camera_callback, 10
        )
        self.detection_subscriber = self.node.create_subscription(
            Int32, self.config.detections_topic, self._detections_callback, 10
        )
        self.htof_group = ReentrantCallbackGroup()
        self.htof_subscriber = self.node.create_subscription(
            PointCloud2, self.config.htof_topic, self._htof_callback, 10, callback_group=self.htof_group
        )
      

    def _joint_state_callback(self, msg: ActuatorFeedback):

        with self.lock:
            self._latest_joint_states = {
                "arm_left_angle": msg.arm_left_angle,
                "arm_right_angle": msg.arm_right_angle,
                # "scoop_lift_angle": msg.scoop_lift_angle,
                "palm_left_angle": msg.palm_left_angle,
                "palm_right_angle": msg.palm_right_angle,
                # "scoop_tilt_angle": msg.scoop_tilt_angle,
            }
        
    def _htof_callback(self, msg: PointCloud2):
        with self.lock:
            pc_array = point_cloud2.read_points_numpy(msg)
            self._latest_htof = [
                {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
                for p in pc_array]


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
                "htof": self._latest_htof.copy(),
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
        with self.lock:
            for joint_name, joint_value in action.items():
                msg = ActuatorDiag()
                msg.actuator_id = self.get_joint_id(joint_name)
                # print(joint_value.item())
                msg.current_position = joint_value

                self.action_publisher.publish(msg)

        return action
    
    def get_joint_id(self, joint_name):
        joint_id_map = {
            "drive_left": ActuatorDiag.DRIVE_LEFT,
            "drive_right": ActuatorDiag.DRIVE_RIGHT,
            "arm_left_angle": ActuatorDiag.ARM_LEFT,
            "arm_right_angle": ActuatorDiag.ARM_RIGHT,
            "scoop_lift_angle": ActuatorDiag.SCOOP_LIFT,
            "palm_left_angle": ActuatorDiag.PALM_LEFT,
            "palm_right_angle": ActuatorDiag.PALM_RIGHT,
            "scoop_tilt_angle": ActuatorDiag.SCOOP_TILT,
        }
        if joint_name not in joint_id_map:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return joint_id_map[joint_name]
        

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