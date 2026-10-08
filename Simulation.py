import math

from environment import LegWheelRobot
from VMC import leg_VMC
from observ import BalanceMPC


def main():
    robot = LegWheelRobot('MJCF/env.xml')
    right_leg = leg_VMC()
    left_leg = leg_VMC()

    control_period = 0.004
    control_steps = round(control_period / robot.sensor_T)
    leg_length_target = 0.285
    wheel_torque_enabled = True

    # 轮端控制车身直立；腿部位置环维持目标长度和相对车身的 90 度姿态。
    balance = BalanceMPC(
        mb=13.902, ma=0.9469, J=0.2, r=0.077,
        dt=control_period, n_horizon=100, l_nominal=leg_length_target,
        torque_limit=8,
    )

    step_count = 0
    while robot.viewer.is_running():
        robot.step()
        step_count += 1
        robot.sensor_read_data()

        if step_count % control_steps != 0:
            continue

        # 逆解在车身坐标系中求解，不补偿车身俯仰。
        right_leg.vmc_calc_pos(
            dt=control_period,
            phi1=robot.joint_pos[0] + math.pi,
            phi4=robot.joint_pos[1],
        )
        left_leg.vmc_calc_pos(
            dt=control_period,
            phi1=robot.joint_pos[3] + math.pi,
            phi4=robot.joint_pos[2],
        )

        try:
            right_target = right_leg.inverse_kinematics(
                leg_length_target, target_theta=0.0,
                seed_phi1=right_leg.phi1, seed_phi4=right_leg.phi4,
            )
            left_target = left_leg.inverse_kinematics(
                leg_length_target, target_theta=0.0,
                seed_phi1=left_leg.phi1, seed_phi4=left_leg.phi4,
            )
            robot.set_vmc_ik_targets(
                right_target[0], right_target[1],
                left_target[0], left_target[1],
            )
        except ValueError as exc:
            # 目标暂时不可达时保留上一周期的位置目标。
            if step_count % (control_steps * 250) == 0:
                print(f"腿部逆解失败，保持上一目标: {exc}")

        state = [robot.euler[1], robot.gyro[1], robot.x, robot.d_x]
        leg_length = (right_leg.L0 + left_leg.L0) / 2
        wheel_torque = float(balance.update(state, leg_length)[0])
        robot.wheel_torque = (
            [wheel_torque / 2, wheel_torque / 2]
            if wheel_torque_enabled else [0.0, 0.0]
        )
        robot.actuator_set_control()


if __name__ == '__main__':
    main()
