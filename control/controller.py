import math

import numpy as np
from .pid import PID
from .vmc import leg_VMC
from .mpc import BalanceMPC


class LegWheelController:
    """仅处理状态快照和动作，不持有 MuJoCo 对象。"""

    def __init__(self, control_period=0.004, leg_length_target=0.25):
        self.control_period = control_period
        self.leg_length_target = leg_length_target
        self.speed_target = 0.0       # m/s，正值前进；键盘在控制主进程更新。
        self.yaw_rate_target = 0.0    # rad/s，正值左转。
        self.yaw_pid = PID(p=0.5,i=0,d=0,dt=control_period)
        self.roll_target = 0.0  # rad，0 表示车身保持水平。
        # 输出为单腿支撑力修正量 N；P: N/rad，I: N/(rad*s)，D: N*s/rad。
        self.roll_pid = PID(p=150, i=0, d=10, dt=control_period)
        self.roll_force_limit = 50.0  # 每条腿的 roll 修正量限幅，单位 N。
        self.wheel_torque_limit = 4.0
        self.left_leg=LegController(dt=control_period)
        self.right_leg=LegController(dt=control_period)
        self.balance = BalanceMPC(
            mb=13.902, ma=0.9469, J=0.2, r=0.077,
            dt=control_period, n_horizon=100, l_nominal=leg_length_target,
            torque_limit=8,
            # 速度控制将位移权重置零，避免速度目标与回原点冲突。
            q_diag=[50.0, 20.0, 0.0, 5.0],
        )
        self.last_leg_torque = None
        self.last_info = {}

    def update(self, observation):
        joints = observation['joint_pos']
        left_angles = np.array([joints[3] + math.pi, joints[2]])
        right_angles = np.array([joints[0] + math.pi, joints[1]])

        # self.left_leg.vmc_calc_pos(dt=self.control_period, phi1=left_angles[0], phi4=left_angles[1])
        # self.right_leg.vmc_calc_pos(dt=self.control_period, phi1=right_angles[0], phi4=right_angles[1])

        roll = float(observation['roll'][0])
        roll_rate = float(observation['roll'][1])
        roll_force = self.roll_pid.calc(roll, self.roll_target, current_rate=roll_rate)
        roll_force = float(np.clip(roll_force, -self.roll_force_limit, self.roll_force_limit))
        # 车身 x 向前、y 向左、z 向上；正 roll 是左侧抬高、右侧降低。
        # 正 roll 时 PID 输出负值，因此右腿加力、左腿减力，产生恢复侧倾的力矩。
        # 两腿分别叠加相反的轴向力，不改变原有腿长和腿角度目标。
        right_torque = self.right_leg.vmc_control(
            0.0314, self.leg_length_target, right_angles[0], right_angles[1],
            force_offset=-roll_force)
        left_torque = self.left_leg.vmc_control(
            -0.0314, self.leg_length_target, left_angles[0], left_angles[1],
            force_offset=roll_force)
        self.last_leg_torque = np.array([right_torque[0], right_torque[1],
                                         left_torque[0], left_torque[1]])

        leg_length = (self.left_leg.L0 + self.right_leg.L0) / 2
        mass_center_height=leg_length-0.05 
        # MPC update 返回 shape=(1,) 的数组，[0] 取总轮力矩，float 转为标量。
        torque = float(self.balance.update(observation['balance'], mass_center_height,
                                          velocity_target=self.speed_target)[0])
        yaw_rate = float(observation['yaw_rate'][0])
        yaw_torque = self.yaw_pid.calc(yaw_rate,self.yaw_rate_target)
        # 平衡力矩优先；剩余单轮力矩余量用于左右差动转向。
        yaw_limit = max(0.0,self.wheel_torque_limit - abs(torque) / 2)
        yaw_torque = float(np.clip(yaw_torque,-yaw_limit,yaw_limit))
        right_wheel_torque = torque / 2 + yaw_torque
        left_wheel_torque = torque / 2 - yaw_torque
        self.last_info = {'status': self.balance.last_status,
                          'leg_length': leg_length, 'total_wheel_torque': torque,
                          'speed_target': self.speed_target, 'speed': float(observation['balance'][3]),
                          'yaw_rate_target': self.yaw_rate_target, 'yaw_rate': yaw_rate,
                          'roll_target': self.roll_target, 'roll': roll,
                          'roll_rate': roll_rate, 'roll_force': roll_force}
        # action = [tau_phi1_R, tau_phi4_R, tau_phi1_L, tau_phi4_L, tau_R, tau_L]。
        # 六项全部为 Nm；后端负责交换左腿两项以匹配 XML 顺序。
        # 总轮力矩均分，再叠加符号相反的转向力矩；正转向让右轮力矩更大。
        return np.r_[self.last_leg_torque, right_wheel_torque, left_wheel_torque]
#直角坐标转换成极坐标

class LegController(leg_VMC):
    '''腿部控制器'''
    def __init__(self, dt=0.004):
        #先调用父类初始化
        super().__init__()
        self.height_pid = PID(p=800,i=20,d=10,dt=dt)
        self.theta_pid = PID(p=200,i=10,d=10,dt=dt)
        self.dt=dt
    def kinematic_control(self,target_x,target_y,phi1,phi4):
        # 先运动学正解算出当前的腿长和腿角度
        self.vmc_calc_pos(dt=self.dt, phi1=phi1, phi4=phi4)
        target_vector=LegController.cartesian_to_polar(target_x,target_y)
        return self.inverse_kinematics(*target_vector)
    def vmc_control(self,target_x,target_y,phi1,phi4,g_force=14.8*9.81,force_offset=0.0):
        # 正解得到当前腿长、方向；PID 输出虚拟力 F0 和虚拟力矩 Tp。
        self.vmc_calc_pos(dt=self.dt, phi1=phi1, phi4=phi4)
        target_vector=LegController.cartesian_to_polar(target_x,target_y)
        #F0是沿着轴向的力，Tp是绕着轴的力矩
        #将直角坐标力转化成极坐标力
        # force_offset 是额外的轴向支撑力 N，roll 控制在两条腿上分别加减。
        F0=self.height_pid.calc(self.L0,target_vector[0])+g_force/2+force_offset
        Tp=self.theta_pid.calc(self.theta,target_vector[1])
        Tp=20
        # self.F0,self.Tp=self.cartesian_to_polar_force(0,-10)
        self.F0,self.Tp=F0,Tp
        # tau = J.T @ [F0, Tp]，返回 [phi1 力矩, phi4 力矩]。
        self.vmc_calc_torque()
        return np.array(self.torque_set)

    @staticmethod
    def cartesian_to_polar(x, y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
    # def cartesian_to_polar_force(self,Fx, Fy):
    #     F0 = Fx*math.cos(self.theta) + Fy*math.sin(self.theta)
    #     Tp = Fy*math.cos(self.theta) - Fx*math.sin(self.theta)
    #     return F0, Tp
