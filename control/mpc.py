"""四状态平衡车 MPC，使用 CasADi + acados，l 在每次控制更新时刷新。

状态: [theta, theta_dot, position, velocity]，theta=0 表示直立，前倾为正。
输入: [tau]，左右轮总驱动力矩，正值使车向前运动，单位 Nm。
参数: [l]。
固定平衡目标: [0, 0, 0, 0]，position 应相对希望停留的位置测量。

l 是轮轴到等效摆体质心的距离，不能直接把腿长当成质心距离。
每次 update 读取当前 l，在该次整个预测时域内保持相同值。
下一次 update 可传入新的 l；模型不包含 l_dot、l_ddot 项。
这是纵向平衡模型；高度、roll 和左右轮差动控制由外部控制器处理。
"""

from pathlib import Path
import tempfile

import numpy as np
from casadi import SX, cos, sin, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class BalanceMPC:
    """可直接实例化的平衡车控制器，无抽象基类或虚函数接口。

    mb: 等效摆体质量，kg。
    ma: 轮组/底座等效平移质量，kg；可计入轮转动惯量 / r**2。
    J: 摆体绕质心的俯仰转动惯量，kg*m**2，不是绕轮轴的惯量。
    r: 轮子滚动半径，m。
    l_nominal: 求解器初始化用的质心距离，实际长度在 update 中传入。
    torque_limit: 总轮力矩上限；两个轮各 4 Nm 时，总上限为 8 Nm。

    基本调用：
        mpc = BalanceMPC(mb=mb, ma=ma, J=J, r=wheel_radius)
        tau = mpc.update([pitch, pitch_rate, position, velocity], l=current_l)[0]
        wheel_torque = [tau / 2, tau / 2]

    wheel_torque 的方向按 simulation/backend.py 的左右轮指令约定。
    """

    def __init__(self, mb, ma, J, r, dt=0.01, n_horizon=20,
                 l_nominal=0.30, torque_limit=8.0, q_diag=None,
                 r_weight=0.1, terminal_scale=5.0, export_directory=None):
        self.mb, self.ma, self.J, self.r = map(float, (mb, ma, J, r))
        self.dt = float(dt)
        self.l_nominal = float(l_nominal)
        self.torque_limit = float(torque_limit)
        self.r_weight = float(r_weight)
        self.terminal_scale = float(terminal_scale)
        positive = [self.mb, self.ma, self.r, self.dt, self.l_nominal,
                    self.torque_limit, self.r_weight, self.terminal_scale]
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("质量、半径、步长、长度、力矩上限和代价系数必须为正有限数")
        if not np.isfinite(self.J) or self.J < 0:
            raise ValueError("J 必须为非负有限数")
        if not isinstance(n_horizon, (int, np.integer)) or n_horizon < 1:
            raise ValueError("n_horizon 必须为正整数")
        self.n_horizon = int(n_horizon)
        self.nx, self.nu, self.np_p = 4, 1, 1
        self.g = 9.81
        self.q_diag = np.asarray(
            [50.0, 20.0, 20.0, 5.0] if q_diag is None else q_diag,
            dtype=float)
        if (self.q_diag.shape != (4,) or not np.all(np.isfinite(self.q_diag))
                or np.any(self.q_diag < 0)):
            raise ValueError("q_diag 必须包含四个非负有限权重")
        self.last_status = None
        # 每个实例使用独立生成目录，避免不同参数的共享库互相覆盖。
        self.export_directory = (
            Path(tempfile.mkdtemp(prefix="balance_mpc_"))
            if export_directory is None else Path(export_directory).resolve())
        self.export_directory.mkdir(parents=True, exist_ok=True)
        self.solver = self._init_solver()

    def _define_model(self):
        model = AcadosModel()
        model.name = "balance_mpc"
        self.x_ = SX.sym("x", self.nx)  # type: ignore
        #分别为 theta, theta_dot, x, x_dot
        self.u_ = SX.sym("u", self.nu)  # type: ignore
        self.p_ = SX.sym("p", self.np_p)  # type: ignore
        x_dot = SX.sym("x_dot", self.nx)  # type: ignore
        theta, theta_dot = self.x_[0], self.x_[1]
        velocity, tau, l = self.x_[3], self.u_[0], self.p_[0]

        # 本次预测时域内长度固定，由两条耦合方程求解加速度：
        # (ma+mb)*position_ddot + mb*l*cos(theta)*theta_ddot
        #     = tau/r + mb*l*sin(theta)*theta_dot**2
        # mb*l*cos(theta)*position_ddot + (J+mb*l**2)*theta_ddot
        #     = mb*g*l*sin(theta) - tau
        # -tau 是轮电机对摆体的反作用力矩。
        mass = self.ma + self.mb
        inertia = self.J + self.mb * l**2
        coupling = self.mb * l * cos(theta)
        force_rhs = tau / self.r + self.mb * l * sin(theta) * theta_dot**2
        moment_rhs = self.mb * self.g * l * sin(theta) - tau
        determinant = mass * inertia - coupling**2
        theta_ddot = (mass * moment_rhs - coupling * force_rhs) / determinant
        position_ddot = (inertia * force_rhs - coupling * moment_rhs) / determinant

        model.x, model.xdot = self.x_, x_dot
        model.u, model.p = self.u_, self.p_
        model.f_expl_expr = vertcat(theta_dot, theta_ddot, velocity, position_ddot)
        model.f_impl_expr = x_dot - model.f_expl_expr
        return model

    def _setup_cost_and_constraints(self, ocp):
        # 只做零状态平衡，直接惩罚状态和轮力矩，不设置目标轨迹。
        ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_0 = ocp.model.cost_y_expr
        ocp.model.cost_y_expr_e = ocp.model.x
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_0 = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.diag(np.r_[self.q_diag, self.r_weight])
        ocp.cost.W_0 = ocp.cost.W.copy()
        ocp.cost.W_e = np.diag(self.terminal_scale * self.q_diag)
        # acados 最小二乘代价所需的固定零向量，运行时不更新。
        ocp.cost.yref = np.zeros(self.nx + self.nu)
        ocp.cost.yref_0 = ocp.cost.yref.copy()
        ocp.cost.yref_e = np.zeros(self.nx)
        ocp.constraints.idxbu = np.array([0], dtype=int)
        ocp.constraints.lbu = np.array([-self.torque_limit])
        ocp.constraints.ubu = np.array([self.torque_limit])
        ocp.constraints.x0 = np.zeros(self.nx)

    def _init_solver(self):
        ocp = AcadosOcp()
        ocp.model = self._define_model()
        ocp.solver_options.N_horizon = self.n_horizon
        ocp.solver_options.tf = self.n_horizon * self.dt
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        self._setup_cost_and_constraints(ocp)
        # l 不能初始化为零；它进入倒立摆惯量和质量矩阵。
        ocp.parameter_values = np.array([self.l_nominal])
        ocp.code_export_directory = str(self.export_directory / "code")
        self.ocp = ocp
        return AcadosOcpSolver(
            ocp, json_file=str(self.export_directory / "balance_mpc_ocp.json"),
            verbose=False)

    def update(self, x_current, l: float):
        """使用当前正长度 l 预测，返回总轮力矩，shape=(1,)。"""
        current = np.asarray(x_current, dtype=float).reshape(self.nx)
        parameter = np.array([l], dtype=float)
        for k in range(self.n_horizon + 1):
            self.solver.set(k, "p", parameter)
        self.solver.set(0, "lbx", current)
        self.solver.set(0, "ubx", current)
        try:
            self.last_status = self.solver.solve()
        except Exception as e:
            print(f"BalanceMPC 求解异常: {e}")
        return self.solver.get(0, "u")
