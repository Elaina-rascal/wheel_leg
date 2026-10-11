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
    def calc(self, current, target):
        error = target - current
        
        p_term = self.kp * error
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error)/self.dt
        self.prev_error = error
        
        return p_term + i_term + d_term
