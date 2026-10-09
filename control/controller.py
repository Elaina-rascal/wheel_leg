import math

import numpy as np

from .vmc import leg_VMC
from .mpc import BalanceMPC


class LegWheelController:
    """仅处理状态快照和动作，不持有 MuJoCo 对象。"""

    def __init__(self, control_period=0.004, leg_length_target=0.285):
        self.control_period = control_period
        self.leg_length_target = leg_length_target
        self.left_leg=leg_VMC()
        self.right_leg=leg_VMC()
        self.balance = BalanceMPC(
            mb=13.902, ma=0.9469, J=0.2, r=0.077,
            dt=control_period, n_horizon=100, l_nominal=leg_length_target,
            torque_limit=8,
        )
        self.last_leg_target = None
        self.last_info = {}

    def update(self, observation):
        joints = observation['joint_pos']
        left_angles = np.array([joints[3] + math.pi, joints[2]])
        right_angles = np.array([joints[0] + math.pi, joints[1]])

        self.left_leg.vmc_calc_pos(dt=self.control_period, phi1=left_angles[0], phi4=left_angles[1])
        self.right_leg.vmc_calc_pos(dt=self.control_period, phi1=right_angles[0], phi4=right_angles[1])

        if self.last_leg_target is None:
            self.last_leg_target = np.array([right_angles, left_angles]).ravel()

        right_target = self.right_leg.inverse_kinematics(
            self.leg_length_target, target_theta=math.pi / 2-0.14)
        left_target = self.left_leg.inverse_kinematics(
            self.leg_length_target, target_theta=math.pi / 2+0.14)
        self.last_leg_target = np.array([right_target, left_target]).ravel()

        leg_length = (self.left_leg.L0 + self.right_leg.L0) / 2
        torque = float(self.balance.update(observation['balance'], leg_length)[0])
        self.last_info = {'status': self.balance.last_status,
                          'leg_length': leg_length, 'total_wheel_torque': torque}
        # np.r_ 把四个腿目标角和两个轮力矩拼成长度为 6 的一维 action：
        # [phi1_right, phi4_right, phi1_left, phi4_left, tau_right, tau_left]。
        # 前四项单位 rad，采用 VMC 坐标；后两项单位 Nm。
        # MPC 的 torque 是左右轮总力矩，直行时均分，每个轮子给 torque / 2。
        return np.r_[self.last_leg_target, torque / 2, torque / 2]
