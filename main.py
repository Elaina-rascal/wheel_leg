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
    # AsyncVectorEnv 接收“创建环境的函数”列表，不是已创建的环境实例。
    # 这里只放一个 make_env，也有一层批量维度：balance 从 (4,) 变为 (1,4)。
    # spawn 新建独立 Python 进程；worker 指定子进程如何收发控制消息。
    # DISABLED 表示环境结束后不自动 reset，由上层决定下一步如何处理。
    envs = AsyncVectorEnv([make_env], context='spawn', worker=viewer_worker,
                          autoreset_mode=AutoresetMode.DISABLED)
    try:
        observations, infos = envs.reset()
        print(f"控制进程 PID={os.getpid()}，"
              f"仿真进程 PID={int(infos['simulation_pid'][0])}", flush=True)
        # single_action_space 是单个环境的动作描述 (6,)；
        # create_empty_array 按 space 分配批量动作缓冲区，这里形状为 (1,6)。
        actions = create_empty_array(envs.single_action_space, n=1)
        last_log_time = -1.0
        while True:
            cycle_start = time.monotonic()
            # iterate 按批量 space 将观测拆成逐个环境的观测；
            # next 取第一个环境：字典结构保留，各数组去掉最外层批量维度。
            observation = next(iterate(envs.observation_space, observations))
            action = controller.update(observation)
            # 将单环境动作列表 [action] 写进批量缓冲区 actions，供 vector env 接收。
            concatenate(envs.single_action_space, [action], actions)
            # 左侧按顺序解包五项返回值；_ 接收这里不使用的 reward（普通变量名）。
            # infos 也按环境批量组织，因此后面用 infos['sim_time'][0] 取第一个环境。
            observations, _, terminated, truncated, infos = envs.step(actions)
            sim_time = float(infos['sim_time'][0])
            if sim_time - last_log_time >= 1.0:
                state = next(iterate(envs.observation_space, observations))
                # %.3f 要逐个格式化数组元素，不能直接用于整个数组。
                # join 把字符串连接起来；后面的 for ... 是生成器表达式，逐项提供字符串。
                # state.items() 提供每一组“键、值”，依次解包给 name、values。
                state_text = ', '.join(
                    '%s=[%s]' % (name, ', '.join('%.3f' % value for value in values))
                    for name, values in state.items()
                )
                print('[control] t=%.3fs state={%s} control=%s'
                      % (sim_time, state_text, controller.last_info), flush=True)
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
