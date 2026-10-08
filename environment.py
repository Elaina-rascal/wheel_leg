import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from caculation import *

class LegWheelRobot:
    """腿轮机器人仿真类"""
    
    def __init__(self, model_path: str = 'legwheel_robot1.xml', fix_legs=False):
        # 加载模型
        spec = mujoco.MjSpec.from_file(model_path)
        if fix_legs:
            # 固定四个腿驱动关节在 XML 初始角度，机身和轮子仍然自由运动。
            for name in ('jAB', 'jAG', 'jIJ', 'jIO'):
                spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=name,
                                  data=[0.0] * 11, solref=[0.002, 1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        base_free_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'base_free'
        )
        if base_free_id < 0:
            raise ValueError("车体速度读取需要名为 base_free 的自由关节")
        self.base_qpos_adr = self.model.jnt_qposadr[base_free_id]
        self.base_dof_adr = self.model.jnt_dofadr[base_free_id]
        self.ik_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ('jAB', 'jAG', 'jIJ', 'jIO')
        ]
        if any(joint_id < 0 for joint_id in self.ik_joint_ids):
            raise ValueError("IK control requires joints jAB, jAG, jIJ and jIO")

        self.sensor_T = 0.001
        self.sensor_f = 1/self.sensor_T 
        self.wheel_r = 0.077

        self.gyro = []
        self.accel = []
        self.orien = []
        self.euler = []

        self.joint_pos = []
        self.wheel_vel = [0,0]

        self.x = 0 #整车位移
        self.d_x = 0 #整车速度

        self.sensor_data = []


        self.left_wheel_pos = 0
        self.right_wheel_pos = 0
        
        self.last_left_wheel_pos = 0
        self.last_right_wheel_pos = 0

        self.wheel_torque: list[float] = [0.0, 0.0]  # 顺序：右、左
        self.joint_position_target = np.array([
            self.model.qpos0[self.model.jnt_qposadr[joint_id]]
            for joint_id in self.ik_joint_ids
        ])
        # MuJoCo 的 position actuator 以 ctrl 为角度目标，启动时先对齐初始姿态。
        self.data.ctrl[:4] = self.joint_position_target



        # 启动可视化界面
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("MuJoCo界面已启动！按ESC退出")
    
    def sensor_read_data(self):
        """读取传感器数据"""
        # 更新传感器数据
        mujoco.mj_forward(self.model, self.data)
        
        # 四元数+欧拉角
        self.orien = self.data.sensor('orientation').data.copy()
        self.euler = orientation2euler(self.orien)
        # 陀螺仪（角速度）
        self.gyro = self.data.sensor('gyro').data.copy()
        # 轮速（官方给的轮速好像有问题，这边直接用当前位置与上一次位置做差，实车还是要用LK电机的轮速数据）
        self.right_wheel_pos = self.data.sensor('Right_Wheel_pos').data.copy()[0]
        self.left_wheel_pos =  self.data.sensor('Left_Wheel_pos').data.copy()[0]
        self.wheel_vel[0] = round((float)(self.right_wheel_pos - self.last_right_wheel_pos) * self.sensor_f,3)
        self.wheel_vel[1] = -round((float)(self.left_wheel_pos - self.last_left_wheel_pos  ) * self.sensor_f,3)
        self.last_right_wheel_pos = self.right_wheel_pos
        self.last_left_wheel_pos = self.left_wheel_pos
        
        # 仿真中读取车体自由关节的世界坐标位置和线速度；轮速仅保留作参考。
        self.x = float(self.data.qpos[self.base_qpos_adr])
        self.d_x = float(self.data.qvel[self.base_dof_adr])

        # 右前关节位置
        right_front_pos = self.data.sensor('Right_front_joint_pos').data.copy()[0]+0.027  #AB
        # 右后关节位置
        right_rear_pos = self.data.sensor('Right_rear_joint_pos').data.copy()[0]+1.3     #AG
        # 左前关节位置
        left_front_pos = self.data.sensor('Left_front_joint_pos').data.copy()[0]+0.003   #IJ
        # 左后关节位置
        left_rear_pos = self.data.sensor('Left_rear_joint_pos').data.copy()[0]-1.3       #IO
        self.joint_pos = np.array([right_front_pos, right_rear_pos, left_front_pos, left_rear_pos])
        

    def actuator_set_control(self):
        """写入腿部位置目标和轮端力矩控制量。"""
        self.data.ctrl[:4] = self.joint_position_target
        
        # 设置轮子力矩
        self.data.ctrl[4] = self.wheel_torque[0]  # 右轮
        self.data.ctrl[5] = self.wheel_torque[1]  # 左轮（注意gainprm为-1）

    @staticmethod
    def _vmc_angles_to_qpos(phi1_right, phi4_right, phi1_left, phi4_left):
        return np.array([
            phi1_right - math.pi - 0.027,  # jAB
            phi4_right - 1.3,             # jAG
            phi4_left - 0.003,            # jIJ
            phi1_left - math.pi + 1.3,    # jIO
        ])

    def set_vmc_ik_targets(self, phi1_right, phi4_right,
                           phi1_left, phi4_left):
        """Map VMC link-angle targets to the MuJoCo position-servo setpoints."""
        joint_ids = self.ik_joint_ids
        qpos_target = self._vmc_angles_to_qpos(
            phi1_right, phi4_right, phi1_left, phi4_left
        )
        qpos_target = np.clip(
            qpos_target,
            self.model.jnt_range[joint_ids, 0],
            self.model.jnt_range[joint_ids, 1],
        )
        self.joint_position_target = qpos_target

    def set_joint_positions(self, joint_angles):

        # 获取关节索引
        joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jAG'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jGH'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jIO'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jOP')
        ]
        
        # 设置关节位置和速度
        for i, idx in enumerate(joint_indices):
            if idx != -1 and i < len(joint_angles):
                self.data.qpos[idx] = joint_angles[i]
                self.data.qvel[idx] = 0.0  # 只重置关节速度
                # print(idx)
        
        # 更新模型状态
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
    
    def reset(self):
        """重置机器人状态"""
        mujoco.mj_resetData(self.model, self.data)
        # self.motor_set_torque(0.0, 0.0)
