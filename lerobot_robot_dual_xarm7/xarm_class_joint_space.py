from xarm.wrapper import XArmAPI
from .xarm_errors import controller_error_codes,gripper_error_codes
# from myLibs.kinematic.ik_solver import IK_Solver
import numpy as np

init_pose = np.array([-0.020695317536592484,
                       -0.979644238948822,
                         -0.008580705150961876,
                           0.6497616767883301,
                             0.018501725047826767,
                               1.630096197128296-(np.pi/18),
                                 -0.0007919175550341606])


class xArm7:
    def __init__(self, ip, gripper_g2 = False):
        """Create a xArm7 class to control the Robot in Joint-Space.

        Args:
            ip (str): IP of Robot-Arm
        """
        # Robo vars
        self.gripper_g2 = gripper_g2
        self.arm = XArmAPI(ip)
        self.ip = ip
        self.arm.motion_enable(enable=True)
        self.arm.set_collision_tool_model(tool_type=1)
        self.arm.set_gripper_enable(enable=True)        
        self.status_code, self.position = self.arm.get_position()
        self.gripper_pos = self.get_gripper_pos()
        self.previous_gripper_pos = self.get_gripper_pos()
        self.gripper_timer = 0
        self.gripper_moving = True
        # Solver vars
        # self.IK_Solver = IK_Solver()

    def start_up(self):
        """
        Run after connection. Sets the robot to an initial pose and opens the gripper.
        """
        self.is_error()
        self.position_mode()
        self.arm.set_servo_angle(angle=init_pose,speed=1,mvacc=100,wait=True,is_radian=True)
        self.servo_mode()       
        self.set_gripper_pos(840)

    def get_states(self)->np.ndarray:
        """Return the current position in [x,y,z,roll,pitch,yaw].
        Units are mm and degree.

        Returns:
            position: Current position in [x,y,z,roll,pitch,yaw]
        """
        self.status, position = self.arm.get_position()
        self.position = position        
        return np.array(position)

    def get_joints_radian(self) ->np.ndarray:
        """Returns current joint positions in radian.

        Returns:
            joints: Current joints
        """
        code, [position, velocity, effort] = self.arm.get_joint_states(is_radian=True)
        return position, velocity, effort

    def set_joints_radian(self,joints:np.ndarray)->int:
        """Set joints to the Robot.

        Args:
            joints (np.ndarray): Target joint position
        """
        code = self.arm.set_servo_angle_j(angles=joints, is_radian=True)
        return code      

    def set_gripper_pos(self,pos: float)-> None:
        """Set the Grippers positon.

        Args:
            pos (float): Target position.
        """
        if self.gripper_g2:
            pos_g2 = pos/10
            self.arm.set_gripper_g2_position(pos=pos_g2, speed=200)
        else:
            self.arm.set_gripper_position(pos=pos,speed=1000)

    def get_gripper_pos(self)->float:
        """Returns current Gripper position.

        Returns:
            float: Current gripper position
        """
        gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos[1]
        return gripper_pos[1]

    def is_gripper_moving(self, current_time)->bool:
        """Checks whether the gripper is moving or not.

        Args:
            current_time (_type_): Current loop time for timer.

        Returns:
            Moving: True or False
        """
        pose = self.get_gripper_pos()
        time_diff = current_time - self.gripper_timer

        if pose is None or self.previous_gripper_pos is None:
            self.gripper_moving = False
            return self.gripper_moving

        if abs(pose-self.previous_gripper_pos) < 2:
            if time_diff > 0.5:
                self.previous_gripper_pos = pose
                self.gripper_moving = False
                return self.gripper_moving
            else:
                self.previous_gripper_pos = pose
                self.gripper_moving = True
                return self.gripper_moving
        else:
            self.gripper_timer = current_time
            self.previous_gripper_pos = pose
            self.gripper_moving = True
            return self.gripper_moving        
        

    def manual_mode(self)-> None:
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_mode(2)
        self.arm.set_state(0)

    def position_mode(self)-> None:
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
    
    def servo_mode(self)-> None:
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)


    def reset(self)-> None:
        if self.is_error():
            return
        self.start_up()

    def is_error(self)->bool:
        code_robo, [error_code_robo, warn_code]= self.arm.get_err_warn_code()
        if code_robo == 0 and error_code_robo!=0:
            print(controller_error_codes[error_code_robo])

        code_gripper, error_code_gripper = self.arm.get_gripper_err_code()
        if code_gripper == 0 and error_code_gripper!=0:
            print(gripper_error_codes[error_code_gripper])

        if (code_gripper and code_robo)==0 and (error_code_gripper or error_code_robo) !=0:
            return True
        
        return False

    def get_arm_err(self):
        code_robo, [error_code_robo, warn_code]= self.arm.get_err_warn_code()
        return error_code_robo
    
    def get_gripper_err(self):
        code_gripper, error_code_gripper = self.arm.get_gripper_err_code()
        return error_code_gripper   

    def destroy(self)->None:
        self.arm.set_state(state=4)
        self.arm.disconnect()
        # self.IK_Solver.destroy()

    
