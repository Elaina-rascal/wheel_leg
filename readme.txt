MuJoCo 轮腿机器人仿真与控制

当前开发容器 wheel_leg：Python 3.12、MuJoCo 3.7.0、Gymnasium 1.4.0。
MPC 使用容器内已有的 CasADi 和 acados_template。
安装 Gymnasium：pip install gymnasium==1.4.0

运行（项目根目录）：
    python main.py
宿主机启动容器内仿真：
    docker exec -it -w /home/Elaina/ros2_ws wheel_leg python main.py

文件职责：
    main.py                    双进程启动、控制循环、日志和实时限速
    simulation/backend.py      MuJoCo 模型、传感器、执行器和 viewer
    simulation/gym_env.py      Gymnasium 环境和等待控制消息时同步 UI 的 worker
    simulation/rotations.py    四元数与欧拉角转换
    MJCF/                      XML、网格和模型资源（根目录）
    control/controller.py     MPC 与逆解组合，只读取状态快照
    control/mpc.py             平衡模型与 acados 求解器
    control/vmc.py             机构运动学和 VMC
    control/pid.py             PID 工具

控制周期在 main.py 中设置，当前为 4 ms。
每次 env.step 保持动作并执行 4 个 1 ms 物理步。
主进程计算控制，仿真子进程执行动作，通信由 AsyncVectorEnv 管理。
控制器停在断点时仿真等待动作，worker 每约 16 ms 同步 UI。
按 ESC 关闭 viewer 或 Ctrl+C 退出。

初始姿态在 MJCF/robot.xml 中定义。
控制目标为相对车身 90 度、腿长 0.285 m，目标位置为 0。

Gymnasium 接口在 simulation/gym_env.py 中定义：
    observation 为 Dict：balance=[theta, theta_dot, x, dx]，
    joint_pos=四关节角，wheel_vel=右/左轮角速度。
    action=[phi1_right, phi4_right, phi1_left, phi4_left, tau_right, tau_left]。
    前四维单位 rad，采用 VMC 坐标；后两维单位 Nm。
    Gym 不做重复动作检查或 clip，执行器限位由后端和 XML 处理。
    reward 当前为 0；训练 RL 前在此定义奖励和任务终止条件。

修改 Gym 空间、观测组装和动作解析：simulation/gym_env.py。
修改 MuJoCo 传感器或执行器映射：simulation/backend.py。
修改控制算法：control/；MPC 模型参数在 control/controller.py 中配置。
worker 适配 Gymnasium 1.4 的内部接口，升级版本时需检查签名。
