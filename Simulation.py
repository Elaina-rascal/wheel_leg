import mujoco
import mujoco.viewer
import numpy as np
import time
from environment import *
from VMC import *
from keyboard import *
import math
from Controller import *
def main():
    
    TORQUE = 1  #为1时给力矩，为0是无力矩
    GBC486 = LegWheelRobot('MJCF/env.xml')
    i = 0
    t1 = 1
    t2 = 4
    # WASD 轮子力矩指令，先用较小值避免突然冲出。
    DRIVE_TORQUE = 1.0
    TURN_TORQUE = 0.5
    vmc_r = leg_VMC()
    vmc_l = leg_VMC()
    keyboard = KeyboardController()


    while True:
        i = i + 1
        
        # 执行仿真步
        GBC486.step()  # 仿真的timestep是1ms，意味着每执行一次step仿真世界时间过去1ms
        #传感器数据获取
        if i % t1 == 0: 
            GBC486.sensor_read_data()
        #vmc计算、观测器计算、LQR计算
        if i % t2 == 0:
            #正向运动学计算，获取状态
            vmc_r.vmc_calc_pos(phi1=GBC486.joint_pos[0]+math.pi,phi4=GBC486.joint_pos[1],pitch= GBC486.euler[1],gyro=GBC486.gyro[1])
            vmc_l.vmc_calc_pos(phi1=GBC486.joint_pos[3]+math.pi,phi4=GBC486.joint_pos[2],pitch=-GBC486.euler[1],gyro=-GBC486.gyro[1])
            vmc_r.F0 = 0
            vmc_l.F0 = 0
            vmc_r.Tp = 0
            vmc_l.Tp = 0
            
            vmc_l.vmc_calc_torque()
            vmc_r.vmc_calc_torque()
            # vmc.vmc_calc()
            # W/S 前进/后退，A/D 左/右转。
            cmd = keyboard.get_command()
            forward = cmd[0] * DRIVE_TORQUE
            turn = cmd[1] * TURN_TORQUE
            w_r = float(np.clip(forward - turn, -4.0, 4.0))
            w_l = float(np.clip(forward + turn, -4.0, 4.0))
            GBC486.wheel_torque = [w_r,w_l]
            GBC486.joint_torque = [vmc_r.torque_set[1],vmc_r.torque_set[0],vmc_l.torque_set[0],vmc_l.torque_set[1]]
            GBC486.actuator_set_torque()


if __name__ == '__main__':
    main()
