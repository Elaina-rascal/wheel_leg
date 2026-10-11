"""MuJoCo 后端：模型加载、传感器、执行器和 viewer。"""

import math
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .rotations import orientation2euler
from .visual import show_center_of_mass


class LegWheelRobot:
    joint_names = ('jAB', 'jAG', 'jIJ', 'jIO')
    joint_sensors = ('Right_front_joint_pos', 'Right_rear_joint_pos',
                     'Left_front_joint_pos', 'Left_rear_joint_pos')
    joint_offsets = np.array([0.027, 1.3, 0.003, -1.3])

    def __init__(self, model_path=None, render_mode=None, fix_legs=False):
        self.viewer = None
        if model_path is None:
            # __file__ 是当前模块文件；parents[0] 是 simulation，parents[1] 是项目根目录。
            # Path 的 / 运算符拼接路径，让模型定位不依赖终端当前工作目录。
            model_path = str(Path(__file__).resolve().parents[1] / 'MJCF' / 'env.xml')
        spec = mujoco.MjSpec.from_file(model_path)  # type: ignore
        if fix_legs:
            for name in self.joint_names:
                spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=name,  # type: ignore
                                  data=[0.0] * 11, solref=[0.002, 1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)  # type: ignore

        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'base_free')  # type: ignore
        # [表达式 for name in ...] 是列表推导式，按 joint_names 顺序建立关节 ID 列表。
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
                          for name in self.joint_names]
        if base_id < 0 or min(self.joint_ids) < 0:
            raise ValueError('XML 缺少 base_free 或腿驱动关节')
        self.base_body_id = self.model.jnt_bodyid[base_id]
        self.base_dof_adr = self.model.jnt_dofadr[base_id]
        self.sensor_T = float(self.model.opt.timestep)
        self.reset()
        if render_mode == 'human':
            # passive viewer 只负责交互和显示，物理步进由本类 step() 主动调用。
            # sync() 将界面操作同步回 data，也让窗口显示最新物理状态。
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            show_center_of_mass(self.viewer)
            print(f'MuJoCo 仿真进程 PID={os.getpid()}，按 ESC 退出', flush=True)

    def _wheel_positions(self):
        return np.array([self.data.sensor(name).data[0]
                         for name in ('Right_Wheel_pos', 'Left_Wheel_pos')])

    def read_sensors(self):
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        self.euler = orientation2euler(self.data.sensor('orientation').data)
        self.gyro = self.data.sensor('gyro').data.copy()
        wheel_positions = self._wheel_positions()
        # 左轮关节轴方向相反；差分周期为一个物理步，单位 rad/s。
        self.wheel_vel = np.round((wheel_positions - self.last_wheel_positions)
                                  * np.array([1, -1]) / self.sensor_T, 3)
        self.last_wheel_positions = wheel_positions
        self.joint_pos = np.array([self.data.sensor(name).data[0]
                                   for name in self.joint_sensors]) + self.joint_offsets
        # 自由关节的线速度是世界坐标；R.T 将它转换到上层车体局部坐标。
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        world_velocity = self.data.qvel[self.base_dof_adr:self.base_dof_adr + 3]
        # .T 为矩阵转置，@ 为矩阵乘法；这里是 3×3 矩阵乘长度为 3 的向量。
        self.d_x = float((rotation.T @ world_velocity)[0])

    def set_control(self, leg_angles, wheel_torques):
        """将 VMC 角度映射为四关节位置目标，写入归一化的左右轮力矩。"""
        phi1_right, phi4_right, phi1_left, phi4_left = leg_angles
        targets = np.array([phi1_right - math.pi - 0.027, phi4_right - 1.3,
                            phi4_left - 0.003, phi1_left - math.pi + 1.3])
        limits = self.model.jnt_range[self.joint_ids]
        # 二维切片 [:,0]/[:,1] 取所有关节的下界/上界；np.clip 逐项限制目标角。
        self.data.ctrl[:4] = np.clip(targets, limits[:, 0], limits[:, 1])
        self.data.ctrl[4:6] = wheel_torques  # 左轮 gainprm=-1 在 XML 中归一化。

    def step(self):
        """推进一个物理步，并按同样的周期读取传感器。"""
        mujoco.mj_step(self.model, self.data)  # type: ignore
        self.read_sensors()
        # 每个物理步积分一次，重复读取传感器或同步 viewer 不累加位移。
        # x 是沿车体瞬时局部 x 轴累计的位移，不是世界坐标位置。
        self.x += self.d_x * self.sensor_T

    def reset(self):
        """恢复 XML 初始姿态，位置执行器目标对齐初始关节角。"""
        mujoco.mj_resetData(self.model, self.data)  # type: ignore
        self.x = 0.0
        # jnt_qposadr 把“关节 ID”映射到 qpos 中的起始索引；自由关节占多项，
        # 因而关节 ID 不能直接用作 qpos 索引。列表索引一次取出四个驱动关节的位置。
        self.data.ctrl[:4] = self.model.qpos0[self.model.jnt_qposadr[self.joint_ids]]
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        self.last_wheel_positions = self._wheel_positions()
        self.read_sensors()
        self.render()

    def read_info(self):
        return {'sim_time': float(self.data.time), 'physics_dt': self.sensor_T,
                'simulation_pid': os.getpid(), 'actuator_ctrl': self.data.ctrl.copy(),
                'viewer_running': self.is_running()}

    def is_running(self):
        return self.viewer is None or self.viewer.is_running()

    def render(self):
        """同步 UI，不推进物理时间。"""
        if self.viewer is not None and self.viewer.is_running():
            previous_time = float(self.data.time)
            self.viewer.sync()
            # 界面 Reset 在 sync 内生效；时间回退时同步清理后端状态。
            # 这里只清理仿真后端，主进程中的控制器状态不会通过 viewer 自动重置。
            if self.data.time < previous_time:
                self.reset()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
