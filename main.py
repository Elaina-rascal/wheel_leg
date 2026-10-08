"""双进程启动入口：主进程控制，子进程仿真。"""

import os
import time

from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

from simulation.gym_env import WheelLegEnv, viewer_worker


CONTROL_PERIOD = 0.004


def make_env():
    return WheelLegEnv(render_mode='human', control_period=CONTROL_PERIOD)


def main():
    # 子进程只导入仿真模块，MPC 及 acados 在控制主进程加载。
    from control.controller import LegWheelController

    controller = LegWheelController(control_period=CONTROL_PERIOD)
    envs = AsyncVectorEnv([make_env], context='spawn', worker=viewer_worker,
                          autoreset_mode=AutoresetMode.DISABLED)
    try:
        observations, infos = envs.reset()
        print(f"控制进程 PID={os.getpid()}，"
              f"仿真进程 PID={int(infos['simulation_pid'][0])}", flush=True)
        actions = create_empty_array(envs.single_action_space, n=1)
        last_log_time = -1.0
        while True:
            cycle_start = time.monotonic()
            observation = next(iterate(envs.observation_space, observations))
            action = controller.update(observation)
            concatenate(envs.single_action_space, [action], actions)
            observations, _, terminated, truncated, infos = envs.step(actions)
            sim_time = float(infos['sim_time'][0])
            if sim_time - last_log_time >= 1.0:
                state = next(iterate(envs.observation_space, observations))
                print(f'[control] t={sim_time:.3f}s state={state} '
                      f'control={controller.last_info}', flush=True)
                last_log_time = sim_time
            if terminated[0] or truncated[0]:
                break
            # 控制超时则直接继续，按实际控制周期限速，不追赶积压时间。
            time.sleep(max(0.0, CONTROL_PERIOD - (time.monotonic() - cycle_start)))
    except KeyboardInterrupt:
        pass
    finally:
        envs.close(timeout=5)


if __name__ == '__main__':
    main()
