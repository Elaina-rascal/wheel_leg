from casadi import SX, vertcat, cos, sin, atan2
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
#倒立摆状态空间
class InvertedPendulum:
    def __init__(self):
        self.n_horizon = 20
        self.dt=0.01

    def _init_solver(self):
        ocp = AcadosOcp()
        ocp.model = self._define_model()
        ocp.solver_options.N_horizon = self.n_horizon
        ocp.solver_options.tf = self.n_horizon * self.dt
        
        # 通用求解器设置
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
    def _define_model(self,mb,ma,J):
        '''
        定义倒立摆的动力学模型
        mb为
        '''
        self.p_=SX.sym('l',1) #type: ignore
        model=AcadosModel()
        self.x_=SX.sym('x',4) #type: ignore #theta,theta_dot,x,x_dot
        self.u_=SX.sym('u',1) #type: ignore #力矩
        return model