MuJoCo 轮腿机器人仿真与控制

当前开发容器 wheel_leg：Python 3.12、MuJoCo 3.7.0、Gymnasium 1.4.0。
MPC 使用容器内已有的 CasADi 和 acados_template。
安装 Gymnasium：pip install gymnasium==1.4.0

运行（项目根目录）：
    python main.py --scene terrain   真实地形（不指定 --scene 时也是这个场景）
    python main.py --scene flat      原来的棋盘平地
宿主机启动容器内仿真：
    docker exec -it -w /home/Elaina/ros2_ws wheel_leg python main.py --scene terrain
    docker exec -it -w /home/Elaina/ros2_ws wheel_leg python main.py --scene flat

文件职责：
    main.py                    双进程启动、控制循环、日志和实时限速
    simulation/backend.py      MuJoCo 模型、传感器、执行器和 viewer
    simulation/gym_env.py      Gymnasium 环境和等待控制消息时同步 UI 的 worker
    simulation/rotations.py    四元数与欧拉角转换
    MJCF/                      XML、网格和模型资源（根目录）
    control/controller.py     MPC 与 VMC 力矩控制，只读取状态快照
    control/mpc.py             平衡模型与 acados 求解器
    control/vmc.py             机构运动学和 VMC
    control/pid.py             PID 工具
    control/keyboard.py        控制主进程键盘监听，直接修改控制器速度目标

控制周期在 main.py 中设置，当前为 4 ms。
每次 env.step 保持动作并执行 4 个 1 ms 物理步。
主进程计算控制，仿真子进程执行动作，通信由 AsyncVectorEnv 管理。
控制器停在断点时仿真等待动作，worker 每约 16 ms 同步 UI。
按 ESC 关闭 viewer 或 Ctrl+C 退出。

初始姿态在 MJCF/robot.xml 中定义，机身 z=0.2545 m，直接接触地面。
MJCF/scene.xml 共用机器人和物理设置，suspend_base 默认关闭，active=true 可固定机身。
MJCF/env.xml 是原来的无限棋盘平地；MJCF/env_terrain.xml 是土色坑洼地形，不使用棋盘纹理。
真实地形为 40×30 m 的可碰撞高度场，范围 x=[-12, 28] m、y=[-15, 15] m；超出范围没有地面。
出生点半径 0.55 m 内保持 z=0，前方有 7.5~12 cm 深的坑、10~18 cm 高的土包和两条车辙。
其他区域随机分布 120 处坑包，重叠区域的起伏可能更大；固定种子保证场景可重复。
修改 MJCF/generate_terrain.py 中 TERRAIN 和 FEATURES 后，在项目根目录运行：
    python MJCF/generate_terrain.py
脚本同时更新 terrain.bin 和 terrain.xml，随后重启仿真加载；不需要额外安装包。
高度场方法参考：https://mujoco.readthedocs.io/en/3.7.0/XMLreference.html#asset-hfield
腿部目标在 controller.py 两侧 vmc_control 调用处修改。
键盘由控制主进程的 pynput 监听，目标直接写入控制器，不经过 MuJoCo 回调或 info。
    ↑/↓：每次增加/减少 0.1 m/s，范围 ±1 m/s，正值前进。
    ←/→：每次增加/减少 0.2 rad/s，范围 ±2 rad/s，正值左转。
    空格：两个目标清零；松开后保持。仿真时间回退时也清零键盘目标。
键盘步长、范围在 control/keyboard.py 修改；yaw_pid 增益在 controller.py 修改。
容器已安装 pynput；其他环境需 pip install pynput，并连接桌面的 DISPLAY。
速度由 MPC 跟踪（位移权重为零），转向由 PID 控制左右轮差动力矩。
侧倾由 controller.py 的 roll_pid 控制，roll_target 默认 0 rad。
P=150、I=0、D=10，输出为单腿轴向支撑力修正，限幅 ±50 N。
左腿加 roll_force，右腿减 roll_force；正 roll（左侧抬高）时输出负值，右腿增加支撑力。
D 项直接使用 IMU x 轴角速度作为阻尼；roll_pid、roll_force_limit 可在控制器初始化处调整。
逆解 target_theta 与 VMC theta 从车身局部 x 正轴逆时针量取，单位 rad；
垂直目标为 pi/2（90 度），与正运动学 phi0 一致。

Gymnasium 接口在 simulation/gym_env.py 中定义：
    observation 为 Dict：balance=[theta, theta_dot, x, dx]，
    joint_pos=四关节角，wheel_vel=右/左轮角速度，yaw_rate=[车体 z 轴角速度]，
    roll=[侧倾角 rad，车体 x 轴角速度 rad/s]。
    action=[tau_phi1_right, tau_phi4_right, tau_phi1_left, tau_phi4_left, tau_right, tau_left]。
    六项全部为 Nm；腿 motor 限幅 ±20 Nm，轮执行器限幅 ±4 Nm。
    Gym 不做重复动作检查或 clip，执行器限位由后端和 XML 处理。
    reward 当前为 0；训练 RL 前在此定义奖励和任务终止条件。

修改 Gym 空间、观测组装和动作解析：simulation/gym_env.py。
修改 MuJoCo 传感器或执行器映射：simulation/backend.py。
修改控制算法：control/；MPC 模型参数在 control/controller.py 中配置。
worker 适配 Gymnasium 1.4 的内部接口，升级版本时需检查签名。
