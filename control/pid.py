"""PID 控制工具。"""


class PID:
    def __init__(self, p,i,d,dt=0.004):
        # self.kp, self.ki, self.kd = pid_params
        self.kp = p
        self.ki = i
        self.kd = d
        self.integral = 0
        self.prev_error = 0
        self.dt=dt
    def calc(self, current, target, current_rate=None):
        """current_rate 可选：提供测量变化率时，D 项直接使用它作为阻尼。"""
        error = target - current
        
        p_term = self.kp * error
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        if current_rate is None:
            d_term = self.kd * (error - self.prev_error)/self.dt
        else:
            # 对测量值微分：固定目标时 de/dt=-d(current)/dt。
            # 姿态控制可直接传角速度，避免角度差分噪声和目标跳变的微分冲击。
            d_term = -self.kd * current_rate
        self.prev_error = error
        
        return p_term + i_term + d_term
