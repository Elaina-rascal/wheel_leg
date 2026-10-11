"""Gymnasium 环境与等待控制消息时同步 UI 的 worker。"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# .backend 是同一个 Python 包 simulation 下的模块（相对导入）。
from .backend import LegWheelRobot


class WheelLegEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None, control_period=0.004, **backend_options):
        # **backend_options 收集额外的关键字参数为字典，如 fix_legs=True。
        self.render_mode = render_mode
        self.control_period = control_period
        self.backend_options = backend_options
        self.robot = None
        self.frame_skip = None
        self.step_id = 0
        # space 描述数据的结构、形状、类型和允许范围，不保存实时观测值。
        # Dict 表示观测是字典，必须与 _observation() 的键和各项数组形状对应。
        # Box 表示实数数组；±np.inf 表示不限制范围，dtype 规定元素的数据类型。
        # shape=(4,) 是长度为 4 的一维数组；逗号让 (4,) 成为单元素元组。
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            'joint_pos': spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            'wheel_vel': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
        })
        # 前四项为 VMC 角度，后两项为单轮力矩；只声明范围，不在此处 clip。
        # low/high 各有 6 项，Box 从它们推断动作 shape=(6,)，逐项定义上下界。
        # 这些界限使用控制器的 VMC 角度坐标，执行时由后端转换到 XML 关节坐标。
        self.action_space = spaces.Box(
            low=np.array([np.pi - 1.57 + .027, -1.57 + 1.3,
                          np.pi - .3 - 1.3, -.5 + .003, -4., -4.]),
            high=np.array([np.pi + .5 + .027, .3 + 1.3,
                           np.pi + 1.57 - 1.3, 1.57 + .003, 4., 4.]),
            dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        # 参数中的 * 表示其后必须按名字传入，如 reset(seed=1)，不能 reset(1)。
        # super() 调用父类 Gym Env 的实现，此处初始化/更新随机数生成器。
        super().reset(seed=seed)
        if self.robot is None:
            # 父进程只检查空间，模型/viewer 在子进程 reset 时创建。
            # 调用处 **字典 将各键值展开为关键字参数，等价于逐个写 fix_legs=...。
            self.robot = LegWheelRobot(render_mode=self.render_mode, **self.backend_options)
            self.frame_skip = round(self.control_period / self.robot.sensor_T)
            if self.frame_skip < 1 or not np.isclose(
                self.frame_skip * self.robot.sensor_T, self.control_period,
                rtol=0, atol=1e-12,
            ):
                self.close()
                raise ValueError('控制周期必须是 MuJoCo 步长的正整数倍')
        else:
            self.robot.reset()
        self.step_id = 0
        # Gym reset 固定返回 (observation, info)；info 是额外诊断信息。
        return self._observation(), self._info()

    def step(self, action):
        if self.robot.is_running():  # type: ignore
            # action = [phi1_R, phi4_R, phi1_L, phi4_L, tau_R, tau_L]。
            self.robot.set_control(action[:4], action[4:])  # type: ignore
            for _ in range(self.frame_skip):  # type: ignore
                self.robot.step()  # type: ignore
            self.step_id += 1
            self.render()
        # 当前用于 MPC 调试；训练前在这里定义奖励和任务终止条件。
        return self._observation(), 0.0, False, not self.robot.is_running(), self._info()  # type: ignore

    def _observation(self):
        robot = self.robot
        # .copy() 保存当前数值快照，避免后续仿真更新影响已经返回的数组。
        return {'balance': np.array([robot.euler[1], robot.gyro[1], robot.x, robot.d_x]),  # type: ignore
                'joint_pos': robot.joint_pos.copy(), 'wheel_vel': robot.wheel_vel.copy()}  # type: ignore

    def _info(self):
        # {**原字典, '新键': 值} 新建合并字典；同名键以后写的值为准。
        return {**self.robot.read_info(), 'step_id': self.step_id,  # type: ignore
                'control_period': self.control_period}

    def render(self):
        if self.robot is not None:
            self.robot.render()

    def close(self):
        if self.robot is not None:
            self.robot.close()
            self.robot = None


class _ViewerConnection:
    """只扩展管道等待，消息、共享内存和错误处理沿用 Gymnasium。"""

    def __init__(self, pipe, environment):
        self.pipe = pipe
        self.environment = environment

    def recv(self):
        # poll(timeout) 等待管道有消息，recv() 再读取；等待期间继续同步 viewer，
        # 所以主进程停在控制器断点时，仿真不步进但窗口仍可操作。
        while not self.pipe.poll(1 / 60):
            self.environment.render()
        return self.pipe.recv()

    def send(self, message):
        self.pipe.send(message)


def viewer_worker(index, env_fn, pipe, parent_pipe, shared_memory,
                  error_queue, autoreset_mode, semaphore=None):
    """Gymnasium 1.4 worker 适配：控制器停在断点时 UI 仍同步。"""
    from gymnasium.vector.async_vector_env import _async_worker

    environment = env_fn()
    # lambda: environment 是一个无参数函数，调用时返回已经创建的 environment。
    # 用包装后的连接替换原 pipe，让 Gym 原 worker 等动作时也能刷新窗口。
    # _async_worker 是 Gym 的内部接口，版本升级时需核对参数，不能当稳定公共 API。
    _async_worker(index, lambda: environment,
                  _ViewerConnection(pipe, environment), parent_pipe,  # type: ignore
                  shared_memory, error_queue, autoreset_mode, semaphore)
