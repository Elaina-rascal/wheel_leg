"""控制主进程的键盘输入，直接修改控制器目标。"""

from pynput import keyboard


class KeyboardInput:
    def __init__(self, controller):
        self.controller = controller
        self.speed_step = 0.3
        self.yaw_rate_step = 0.5
        self.speed_limit = 3.0
        self.yaw_rate_limit = 5.0
        # on_press 传入函数本身；监听线程收到按键后调用它，不经 MuJoCo。
        self.listener = keyboard.Listener(on_press=self.on_press)

    def start(self):
        self.listener.start()
        print('控制键盘：↑/↓ 增减速度，←/→ 增减左转角速度，空格清零', flush=True)

    def on_press(self, key):
        if key == keyboard.Key.up:
            self.controller.speed_target += self.speed_step
        elif key == keyboard.Key.down:
            self.controller.speed_target -= self.speed_step
        elif key == keyboard.Key.left:
            self.controller.yaw_rate_target += self.yaw_rate_step
        elif key == keyboard.Key.right:
            self.controller.yaw_rate_target -= self.yaw_rate_step
        elif key == keyboard.Key.space:
            self.reset()
        else:
            return
        self.controller.speed_target = max(-self.speed_limit, min(
            self.speed_limit, self.controller.speed_target))
        self.controller.yaw_rate_target = max(-self.yaw_rate_limit, min(
            self.yaw_rate_limit, self.controller.yaw_rate_target))
        print(f'[keyboard] v={self.controller.speed_target:.2f} m/s, '
              f'yaw_rate={self.controller.yaw_rate_target:.2f} rad/s', flush=True)

    def reset(self):
        self.controller.speed_target = 0.0
        self.controller.yaw_rate_target = 0.0

    def stop(self):
        self.listener.stop()
