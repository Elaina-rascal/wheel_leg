"""Gymnasium 环境与等待控制消息时同步 UI 的 worker。"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backend import LegWheelRobot


class WheelLegEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None, control_period=0.004, **backend_options):
        self.render_mode = render_mode
        self.control_period = control_period
        self.backend_options = backend_options
        self.robot = None
        self.frame_skip = None
        self.step_id = 0
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            'joint_pos': spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            'wheel_vel': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
        })
        # 前四项为 VMC 角度，后两项为单轮力矩；只声明范围，不在此处 clip。
        self.action_space = spaces.Box(
            low=np.array([np.pi - 1.57 + .027, -1.57 + 1.3,
                          np.pi - .3 - 1.3, -.5 + .003, -4., -4.]),
            high=np.array([np.pi + .5 + .027, .3 + 1.3,
                           np.pi + 1.57 - 1.3, 1.57 + .003, 4., 4.]),
            dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.robot is None:
            # 父进程只检查空间，模型/viewer 在子进程 reset 时创建。
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
        return {'balance': np.array([robot.euler[1], robot.gyro[1], robot.x, robot.d_x]),  # type: ignore
                'joint_pos': robot.joint_pos.copy(), 'wheel_vel': robot.wheel_vel.copy()}  # type: ignore

    def _info(self):
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
    _async_worker(index, lambda: environment,
                  _ViewerConnection(pipe, environment), parent_pipe,  # type: ignore
                  shared_memory, error_queue, autoreset_mode, semaphore)
