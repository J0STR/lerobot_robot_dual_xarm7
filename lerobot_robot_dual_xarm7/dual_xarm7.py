from concurrent.futures import ThreadPoolExecutor
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


def limit_joint_step(goal: list[float], present: list[float], max_step: float) -> list[float]:
    """Move from present towards goal by at most max_step on any joint, keeping the joint-space direction."""
    delta = np.asarray(goal, dtype=float) - np.asarray(present, dtype=float)
    largest = np.abs(delta).max()
    if largest <= max_step:
        return list(goal)
    return (np.asarray(present, dtype=float) + delta * (max_step / largest)).tolist()


class Dual_xArm7(Robot):
    config_class = Dual_xArm7Config
    name = "Dual_xArm7"
    def __init__(self, config: Dual_xArm7Config):
        super().__init__(config)

        self.cameras = make_cameras_from_configs(config.cameras)

        self._ip_right = config.robot_ip_right
        self._ip_left = config.robot_ip_left
        self._max_step_rad = config.max_step_rad
        self.robot_right: xArm7
        self.robot_left: xArm7
        self._is_connected = False

        self.current_states_right = None
        self.current_states_left = None

        self._executor = ThreadPoolExecutor(max_workers=2)
    
    
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

        # Read both arms in parallel
        future_right = self._executor.submit(self._read_arm, self.robot_right, "right")
        future_left = self._executor.submit(self._read_arm, self.robot_left, "left")
        joints_right, obs_right = future_right.result()
        joints_left, obs_left = future_left.result()
        self.current_states_right = joints_right
        self.current_states_left = joints_left
        obs_dict = {**obs_right, **obs_left}
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    @staticmethod
    def _read_arm(robot: xArm7, side: str) -> tuple[list[float], dict[str, Any]]:
        joint_angles, joint_velocity, joint_efforts = robot.get_joints_radian()
        obs = {}
        for i, angle in enumerate(joint_angles):  # store angle,velocity and effort
            obs[f"{side}_joint_{i+1}.pos"] = angle
            obs[f"{side}_joint_{i+1}.effort"] = joint_efforts[i]
            obs[f"{side}_joint_{i+1}.vel"] = joint_velocity[i]
        obs[f"{side}_gripper.pos"] = robot.get_gripper_pos()
        return joint_angles, obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        joints_right = [goal_pos[f"right_joint_{i}"] for i in range(1,8)]
        joints_left = [goal_pos[f"left_joint_{i}"] for i in range(1,8)]

        gripper_right = goal_pos["right_gripper"]
        gripper_left = goal_pos["left_gripper"]

        if self._max_step_rad is not None:
            joints_right = self._limit_step(self.robot_right, joints_right)
            joints_left = self._limit_step(self.robot_left, joints_left)

        # Send to both arms in parallel
        future_right = self._executor.submit(self._send_arm, self.robot_right, joints_right, gripper_right)
        future_left = self._executor.submit(self._send_arm, self.robot_left, joints_left, gripper_left)
        future_right.result()
        future_left.result()

        action = {**{f"right_joint_{i}.pos": joints_right[i-1] for i in range(1,8)},
                  "right_gripper.pos": gripper_right,
                  **{f"left_joint_{i}.pos": joints_left[i-1] for i in range(1,8)},
                  "left_gripper.pos": gripper_left}      

        return action

    @staticmethod
    def _send_arm(robot: xArm7, joints: list[float], gripper: float) -> None:
        robot.set_joints_radian(joints)
        robot.set_gripper_pos(gripper)


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

    def _limit_step(self, robot: xArm7, goal: list[float]) -> list[float]:
        # Read fresh joint states, clamping against a failed read would command a jump
        if robot.ip == self._ip_left:
            states = self.current_states_left
        else:
            states = self.current_states_right
        if states is None:
            raise ConnectionError(f"[{robot.ip}] Joint read failed, cannot limit step size")
        return limit_joint_step(goal, states, self._max_step_rad)

    def calibrate(self) -> None:
        pass
    
    def configure(self) -> None:
        return
    

