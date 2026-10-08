import math

import numpy as np

from .vmc import leg_VMC
from .mpc import BalanceMPC


class LegWheelController:
    """仅处理状态快照和动作，不持有 MuJoCo 对象。"""

    def __init__(self, control_period=0.004, leg_length_target=0.285):
        self.control_period = control_period
        self.leg_length_target = leg_length_target
        self.legs = (leg_VMC(), leg_VMC())  # 右、左
        self.balance = BalanceMPC(
            mb=13.902, ma=0.9469, J=0.2, r=0.077,
            dt=control_period, n_horizon=100, l_nominal=leg_length_target,
            torque_limit=8,
        )
        self.last_leg_target = None
        self.last_info = {}

    def update(self, observation):
        joints = observation['joint_pos']
        angles = ((joints[0] + math.pi, joints[1]),
                  (joints[3] + math.pi, joints[2]))
        for leg, (phi1, phi4) in zip(self.legs, angles):
            leg.vmc_calc_pos(dt=self.control_period, phi1=phi1, phi4=phi4)
        if self.last_leg_target is None:
            self.last_leg_target = np.array(angles).ravel()
        ik_error = None
        try:
            # 逆解默认使用刚计算的关节角作为分支种子，目标相对车身为 90 度。
            targets = [leg.inverse_kinematics(self.leg_length_target, target_theta=0.0)
                       for leg in self.legs]
            self.last_leg_target = np.array(targets).ravel()
        except ValueError as exc:
            ik_error = str(exc)

        leg_length = sum(leg.L0 for leg in self.legs) / 2
        torque = float(self.balance.update(observation['balance'], leg_length)[0])
        self.last_info = {'status': self.balance.last_status, 'ik_error': ik_error,
                          'leg_length': leg_length, 'total_wheel_torque': torque}
        # np.r_ 把四个腿目标角和两个轮力矩拼成长度为 6 的一维 action：
        # [phi1_right, phi4_right, phi1_left, phi4_left, tau_right, tau_left]。
        # 前四项单位 rad，采用 VMC 坐标；后两项单位 Nm。
        # MPC 的 torque 是左右轮总力矩，直行时均分，每个轮子给 torque / 2。
        return np.r_[self.last_leg_target, torque / 2, torque / 2]
