from typing import Any
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot

from .config_dual_xarm7 import Dual_xArm7Config
from .xarm_class_joint_space import xArm7

from xarm.wrapper import XArmAPI
import numpy as np

class Dual_xArm7(Robot):
    config_class = Dual_xArm7Config
    name = "Dual_xArm7"
    def __init__(self, config: Dual_xArm7Config):
        super().__init__(config)
        
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self._ip_right = config.robot_ip_right
        self._ip_left = config.robot_ip_left
        self.robot_right: xArm7
        self.robot_left: xArm7
        self._is_connected = False
    
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.robot_right = xArm7(ip=self._ip_right,
                                 gripper_g2=True)
        self.robot_left = xArm7(ip=self._ip_left,
                                 gripper_g2=False)

        self.robot_right.start_up()
        self.robot_left.start_up()
        self._is_connected = True  

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def reset(self):
        self.robot_right.reset()
        self.robot_left.reset()

    def manual_mode(self):
        self.robot_right.manual_mode()
        self.robot_left.manual_mode()

    def servo_mode(self):
        self.robot_right.servo_mode()
        self.robot_left.servo_mode()

    def is_error(self):
        if self.robot_right.is_error() or self.robot_left.is_error():
            return True
        else:
            return False
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Read arm position
        joint_angles, joint_velocity, joint_efforts = self.robot_right.get_joints_radian()
        obs_dict = {}
        for i, angle in enumerate(joint_angles):  # store angle,velocity and effort
            obs_dict[f"right_joint_{i+1}.pos"] = angle
            obs_dict[f"right_joint_{i+1}.effort"] = joint_efforts[i]
            obs_dict[f"right_joint_{i+1}.vel"] = joint_velocity[i]
        joint_angles, joint_velocity, joint_efforts = self.robot_left.get_joints_radian()
        for i, angle in enumerate(joint_angles):  # store angle,velocity and effort
            obs_dict[f"left_joint_{i+1}.pos"] = angle
            obs_dict[f"left_joint_{i+1}.effort"] = joint_efforts[i]
            obs_dict[f"left_joint_{i+1}.vel"] = joint_velocity[i]
        gripper_pos = self.robot_right.get_gripper_pos()
        obs_dict["right_gripper.pos"] = gripper_pos
        gripper_pos = self.robot_left.get_gripper_pos()
        obs_dict["left_gripper.pos"] = gripper_pos
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        joints_right = [goal_pos[f"right_joint_{i}"] for i in range(1,8)]
        joints_left = [goal_pos[f"left_joint_{i}"] for i in range(1,8)]

        gripper_right = goal_pos["right_gripper"]
        gripper_left = goal_pos["left_gripper"]

        self.robot_right.set_joints_radian(joints_right)
        self.robot_left.set_joints_radian(joints_left)       
        self.robot_right.set_gripper_pos(gripper_right)
        self.robot_left.set_gripper_pos(gripper_left)  

        action = {**{f"right_joint_{i}.pos": joints_right[i-1] for i in range(1,8)},
                  "right_gripper.pos": gripper_right,
                  **{f"left_joint_{i}.pos": joints_left[i-1] for i in range(1,8)},
                  "left_gripper.pos": gripper_left}      

        return action


    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        self.robot_right.destroy()
        self.robot_left.destroy()
        self._is_connected = False
        for cam in self.cameras.values():
            cam.disconnect()


    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "right_joint_1.pos": float,
            "right_joint_2.pos": float,
            "right_joint_3.pos": float,
            "right_joint_4.pos": float,
            "right_joint_5.pos": float,
            "right_joint_6.pos": float,
            "right_joint_7.pos": float,
            "right_gripper.pos": float,
            "left_joint_1.pos": float,
            "left_joint_2.pos": float,
            "left_joint_3.pos": float,
            "left_joint_4.pos": float,
            "left_joint_5.pos": float,
            "left_joint_6.pos": float,
            "left_joint_7.pos": float,
            "left_gripper.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict:
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:        
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass
    
    def configure(self) -> None:
        return
    

