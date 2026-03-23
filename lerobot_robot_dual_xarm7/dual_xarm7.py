from typing import Any
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot

from .config_dual_xarm7 import Dual_xArm7Config

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
        self.robot_right = None
        self.robot_left = None
        self._is_connected = False
    
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.robot_right = XArmAPI(self._ip_right)
        self.robot_left = XArmAPI(self._ip_left)
        self._is_connected = True
        self.robot_right.motion_enable(enable=True)
        self.robot_left.motion_enable(enable=True)
        self.robot_right.set_collision_tool_model(tool_type=1)
        self.robot_left.set_collision_tool_model(tool_type=1)
        self.robot_right.set_gripper_enable(enable=True)
        self.robot_left.set_gripper_enable(enable=True)
        self.robot_right.set_tcp_offset([0, 0, 0, 0, 0, 0], wait=True)
        self.robot_left.set_tcp_offset([0, 0, 0, 0, 0, 0], wait=True)
        self.robot_right.set_mode(1)
        self.robot_left.set_mode(1)
        self.robot_right.set_state(0)    
        self.robot_left.set_state(0)    

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Read arm position
        code, [joint_angles, joint_velocity, joint_efforts] = self.robot_right.get_joint_states(is_radian=True)
        obs_dict = {}
        for i, angle in enumerate(joint_angles):  # store angle,velocity and effort
            obs_dict[f"right_joint_{i+1}.pos"] = angle
            obs_dict[f"right_joint_{i+1}.effort"] = joint_efforts[i]
            obs_dict[f"right_joint_{i+1}.vel"] = joint_velocity[i]
        code, [joint_angles, joint_velocity, joint_efforts] = self.robot_left.get_joint_states(is_radian=True)
        for i, angle in enumerate(joint_angles):  # store angle,velocity and effort
            obs_dict[f"left_joint_{i+1}.pos"] = angle
            obs_dict[f"left_joint_{i+1}.effort"] = joint_efforts[i]
            obs_dict[f"left_joint_{i+1}.vel"] = joint_velocity[i]
        code, gripper_pos = self.robot_right.get_gripper_position()
        obs_dict["right_gripper.pos"] = gripper_pos
        code, gripper_pos = self.robot_left.get_gripper_position()
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
        gripper_right_G2 = gripper_right/10 # griper G2
        gripper_left = goal_pos["left_gripper"]

        self.robot_right.set_servo_angle_j(joints_right, is_radian=True)
        self.robot_left.set_servo_angle_j(joints_left, is_radian=True)       
        self.robot_right.set_gripper_g2_position(gripper_right_G2,speed=225)
        self.robot_left.set_gripper_position(gripper_left, speed=5000)  

        action = {**{f"right_joint_{i}.pos": joints_right[i-1] for i in range(1,8)},
                  "right_gripper.pos": gripper_right,
                  **{f"left_joint_{i}.pos": joints_left[i-1] for i in range(1,8)},
                  "left_gripper.pos": gripper_left}      

        return action


    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        self.robot_right.disconnect()
        self.robot_left.disconnect()
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
    

