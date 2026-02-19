from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras import ColorMode, Cv2Rotation
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("Dual_xArm7")
@dataclass
class Dual_xArm7Config(RobotConfig):
    id: str
    robot_ip_right: str = "10.2.134.152"
    robot_ip_left: str = "10.2.134.151"


    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top_down": RealSenseCameraConfig(
                serial_number_or_name="352122273665",
                fps=30,
                width=640,
                height=480),
            "wurm_eye": RealSenseCameraConfig(
                serial_number_or_name="409122274688",
                fps=30,
                width=640,
                height=480),
            "wrist_right": RealSenseCameraConfig(
                serial_number_or_name="352122273091",
                fps=30,
                width=640,
                height=480),
            "wrist_left": RealSenseCameraConfig(
                serial_number_or_name="409122271056",
                fps=30,
                width=640,
                height=480),
                
        }
    )