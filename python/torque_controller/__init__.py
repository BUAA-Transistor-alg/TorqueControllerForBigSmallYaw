"""torque_controller — RoboMaster 双级 yaw 云台控制（Python ctypes 绑定）

C API: ``include/tcbs/c_api/RobotCommunicationC.h`` → ``build/libtcbs_robot_comm_c.so``
桥接实现: :mod:`torque_controller._bridge`

对外提供两个类::

    from torque_controller import TcbsRobotCommunication   # 低层: 串口 + 映射 + 状态估计
    from torque_controller import TcbsRobotController      # 高层: 估计 + 耦合 MPC + 后台发送线程

低层 ``TcbsRobotCommunication``::

    with TcbsRobotCommunication() as comm:
        data = comm.get_latest_data()     # mcu / imu 原始反馈（valid 标志）
        est  = comm.get_estimate()        # 完整状态估计
        comm.send_to_mcu(auto_aim_enable=1, yaw_big_mode=1, yaw_big_target_angle=0.0)

高层 ``TcbsRobotController``（目标为**世界方位角**语义，rad、多圈连续）::

    with TcbsRobotController() as ctrl:
        st = ctrl.get_state()             # mcu / imu / est / mpc 四组（序列为 list）
        ctrl.set(big_yaw_azimuth=st.est.platform_azimuth,
                 small_yaw_azimuth=st.est.platform_azimuth + 0.2)
        ctrl.set_joint_angles(big_joint_angle=0.0, small_joint_angle=0.0)

动力学模型为**平面 2 自由度 8 参模型**（`include/tcbs/mpc/planar_yaw_model.h`）::

    from torque_controller import default_model_params, model_params_to_vector

    p = default_model_params()            # C++ 默认值（占位值，需标定）
    vec = model_params_to_vector(p)       # 8 维: Jbig_eff, Js, Px, Py, fc/fv_big, fc/fv_small
    ctrl.set_model_params(Jbig_eff=0.024, Js=0.013, Px=1e-3)
    back = ctrl.get_model_params()        # 读回当前参数（含几何 dx/dy/gravity/m_u_known）

无硬件时两个类都能正常创建，``get_state()``/``get_latest_data()`` 的 ``valid`` 全为 0
（可离线跑通整条调用链）。

时间语义（协议修订）: MCU 端**不提供时钟**，且大 yaw 与底盘 IMU 同源（MCU2 一次送来），
因此两者共用**一个**新样本序号 ``mcu.mcu2_seq``（每次取到新数据 +1；值被保持时不变，
判定新样本: 首帧到达 || 序号变化 || 值变化，无 0 哨兵约定）；
值的**年龄 age**、**新样本间隔 interval**、**采样过旧 stale** 由上位机计时，
见 ``est.big_enc_age`` / ``est.big_sample_interval`` / ``est.chassis_imu_age`` 与
``est.prov.big_enc.{new_samples, rejected, stale, big_enc_interval_s, big_enc_sample_age_s}``。
注意 ``est.gravity_a[3]`` 是**重力在 A 系（大 yaw 转子系）** 的投影（v4 起，旧名 gravity_c）。

运行环境:
  * 先构建动态库: ``cd build && cmake .. && make -j8 tcbs_robot_comm_c``
  * 让 Python 找到包: ``PYTHONPATH=python python3 your_script.py``
  * 库不在默认位置时: ``export TORQUE_BS_LIB=/abs/path/libtcbs_robot_comm_c.so``
    （旧名 ``TORQUE_CONTROLLER_LIB`` 仍兼容）
"""

from ._bridge import (
    API_VERSION,
    ERR_ABI_MISMATCH,
    ERR_ALLOC,
    ERR_INVALID_ARG,
    ERR_MODE_MISMATCH,
    ERR_NULL_HANDLE,
    ERR_RUNTIME,
    ERR_UNKNOWN,
    IMU_ON_BIG_YAW,
    IMU_ON_HEAD,
    MODEL_PARAM_NAMES,
    MODE_SEQUENCE,
    MODE_SINGLE,
    TcbsControllerConfig,
    TcbsDualYawJointLimits,
    TcbsDualYawModelParams,
    TcbsDualYawMpcConfig,
    TcbsEstimatorConfig,
    TcbsLinearParams,
    TcbsRobotCommAbiInfo,
    TcbsRobotCommunication,
    TcbsRobotController,
    TcbsRobotControllerState,
    TcbsRobotEstimate,
    TcbsRobotImuData,
    TcbsRobotLatestData,
    TcbsRobotMcuData,
    TcbsRobotMpcData,
    TcbsRobotProvenance,
    TcbsRobotSourceInfo,
    TorqueControllerError,
    check_abi,
    default_controller_config,
    default_estimator_config,
    default_linear_params,
    default_model_params,
    default_mpc_config,
    library_path,
    load_library,
    model_params_to_vector,
    strerror,
    vector_to_model_params,
)

__version__ = "1.3.0"   # 1.3: 跟随 C API v4（平面 2 自由度 8 参模型）

__all__ = [
    # 高层 / 低层主要入口
    "TcbsRobotController",
    "TcbsRobotCommunication",
    # 异常与工具
    "TorqueControllerError",
    "strerror",
    "check_abi",
    "load_library",
    "library_path",
    "API_VERSION",
    # 配置结构体（ctypes）
    "TcbsDualYawModelParams",
    "TcbsDualYawMpcConfig",
    "TcbsDualYawJointLimits",
    "TcbsEstimatorConfig",
    "TcbsLinearParams",
    "TcbsControllerConfig",
    # 数据/状态结构体（ctypes）
    "TcbsRobotMcuData",
    "TcbsRobotImuData",
    "TcbsRobotLatestData",
    "TcbsRobotSourceInfo",
    "TcbsRobotProvenance",
    "TcbsRobotEstimate",
    "TcbsRobotMpcData",
    "TcbsRobotControllerState",
    "TcbsRobotCommAbiInfo",
    # 默认配置取值
    "default_model_params",
    "default_mpc_config",
    "default_estimator_config",
    "default_linear_params",
    "default_controller_config",
    # 平面 8 参模型的参数向量工具（顺序同 C++ paramsToVector / regressor）
    "MODEL_PARAM_NAMES",
    "model_params_to_vector",
    "vector_to_model_params",
    # 常量
    "MODE_SINGLE",
    "MODE_SEQUENCE",
    "IMU_ON_BIG_YAW",
    "IMU_ON_HEAD",
    "ERR_NULL_HANDLE",
    "ERR_INVALID_ARG",
    "ERR_MODE_MISMATCH",
    "ERR_RUNTIME",
    "ERR_UNKNOWN",
    "ERR_ALLOC",
    "ERR_ABI_MISMATCH",
]
