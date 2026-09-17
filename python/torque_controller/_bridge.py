"""_bridge.py — 双级 yaw 云台 C API 的 ctypes 桥接（底层，不对外承诺稳定）

对应 C 头文件: ``include/tcbs/c_api/RobotCommunicationC.h``（纯 C ABI）。

分层:
  * 低层 ``TcbsRobotCommunication`` : MCU/IMU 串口 + 编码器映射 + 状态估计
        get_latest_data() / get_estimate() / send_to_mcu() / send_to_imu() / stop()
  * 高层 ``TcbsRobotController``    : 状态估计 + 耦合 MPC + 后台发送线程
        set() / set_joint_angles() / set_sequence() / get_state()
        配置: set_model_params / set_mpc_config / set_estimator_config /
              set_linear_params / set_controller_config

约定:
  * 返回的结构体都转换成 ``types.SimpleNamespace``（字段名与 C 结构体一致），
    定长数组（如 ``base_omega[3]``）转成 Python list。
  * C 侧返回 ``int`` 的接口: >= 0 表示成功（部分接口返回长度），< 0 是错误码；
    本模块统一转成 ``TorqueControllerError`` 抛出。
  * 库文件查找顺序: 环境变量 ``TORQUE_BS_LIB``（兼容旧名 ``TORQUE_CONTROLLER_LIB``）
    → 仓库 ``build/libtcbs_robot_comm_c.so``
    → 系统库搜索路径。导入时若库可用，会自动做一次结构体布局自检（ABI check）。
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_double,
    c_float,
    c_int32,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
    sizeof,
)
from types import SimpleNamespace

__all__ = [
    "API_VERSION",
    "TorqueControllerError",
    "TcbsRobotCommunication",
    "TcbsRobotController",
    "TcbsRobotMcuData",
    "TcbsRobotImuData",
    "TcbsRobotLatestData",
    "TcbsRobotSourceInfo",
    "TcbsRobotProvenance",
    "TcbsRobotEstimate",
    "TcbsDualYawJointLimits",
    "TcbsDualYawModelParams",
    "TcbsDualYawMpcConfig",
    "TcbsEstimatorConfig",
    "TcbsLinearParams",
    "TcbsControllerConfig",
    "TcbsRobotMpcData",
    "TcbsRobotControllerState",
    "TcbsRobotCommAbiInfo",
    "load_library",
    "check_abi",
    "library_path",
    "MODE_SINGLE",
    "MODE_SEQUENCE",
    "IMU_ON_BIG_YAW",
    "IMU_ON_HEAD",
    "MODEL_PARAM_NAMES",
    "model_params_to_vector",
    "vector_to_model_params",
    "ERR_NULL_HANDLE",
    "ERR_INVALID_ARG",
    "ERR_MODE_MISMATCH",
    "ERR_RUNTIME",
    "ERR_UNKNOWN",
    "ERR_ALLOC",
    "ERR_ABI_MISMATCH",
]

API_VERSION = 4   # v4: 平面 2 自由度 8 参模型（ModelParams/MpcConfig 重写；gravity_c → gravity_a）

# ── 错误码（与 TcbsRobotCommStatus 一致）──
ERR_OK = 0
ERR_NULL_HANDLE = -1
ERR_INVALID_ARG = -2
ERR_MODE_MISMATCH = -3
ERR_RUNTIME = -4
ERR_UNKNOWN = -5
ERR_ALLOC = -6
ERR_ABI_MISMATCH = -7

MODE_SINGLE = 0
MODE_SEQUENCE = 1

# IMU 安装位置（对应 YawStateEstimator::Config::ImuLocation）
IMU_ON_BIG_YAW = 0   # IMU 固定在大 yaw 转子 A 上（现状；用 mount_* 标定）
IMU_ON_HEAD = 1      # IMU 装在头上（pitch 之后，H 系；用 head_mount_* 标定）

# ── ★8 个待辨识模型参数（顺序与 dual_yaw::paramsToVector / regressor 列一致）──
#   注: 前 4 个与 ctypes 结构体字段同名；摩擦四个在 C++/辨识脚本里叫 fc_big 等，
#       C 结构体里叫 fcBig 等 —— 两套名字的对应关系见下面的 _MODEL_PARAM_FIELDS。
MODEL_PARAM_NAMES = ("Jbig_eff", "Js", "Px", "Py",
                     "fc_big", "fv_big", "fc_small", "fv_small")
_MODEL_PARAM_FIELDS = ("Jbig_eff", "Js", "Px", "Py",
                       "fcBig", "fvBig", "fcSmall", "fvSmall")


class TorqueControllerError(RuntimeError):
    """C API 返回负错误码时抛出（含错误码与可读描述）。"""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# ============================================================================
# 库加载
# ============================================================================
# 库路径**可被环境变量覆盖**（按优先级从高到低）:
#   TORQUE_BS_LIB        —— 本仓库（模块标识 tcbs）专用
#   TORQUE_CONTROLLER_LIB —— 历史名，保留兼容
_LIB_ENVS = ("TORQUE_BS_LIB", "TORQUE_CONTROLLER_LIB")
_LIB_ENV = _LIB_ENVS[0]          # 文档/报错信息里展示的首选变量名
_LIB_NAME = "libtcbs_robot_comm_c.so"


def _candidate_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    for env_name in _LIB_ENVS:
        env = os.environ.get(env_name)
        if env:
            yield env
    yield os.path.join(repo_root, "build", _LIB_NAME)
    yield os.path.join(repo_root, "build", "lib", _LIB_NAME)
    yield _LIB_NAME  # 交给系统加载器


def load_library(path: str | None = None):
    """加载 libtcbs_robot_comm_c.so（path 为空时按默认顺序查找）。"""
    candidates = [path] if path else list(_candidate_paths())
    errors = []
    for cand in candidates:
        if cand is None:
            continue
        if os.path.sep in cand and not os.path.isfile(cand):
            errors.append(f"{cand}: 文件不存在")
            continue
        try:
            return ctypes.CDLL(cand)
        except OSError as exc:  # pragma: no cover - 依赖环境
            errors.append(f"{cand}: {exc}")
    raise TorqueControllerError(
        "无法加载 " + _LIB_NAME + "，请先构建（cd build && cmake .. && make -j8 tcbs_robot_comm_c）"
        "或用环境变量 " + " / ".join(_LIB_ENVS) + " 指定路径。尝试过:\n  " + "\n  ".join(errors)
    )


# ============================================================================
# 结构体定义（字段顺序必须与 RobotCommunicationC.h 完全一致）
# ============================================================================


class TcbsRobotMcuData(Structure):
    """MCU 反馈（已按映射参数换算）。"""

    _fields_ = [
        ("valid", c_uint8),                 # 是否收到过有效 MCU 帧
        ("bullet_velocity", c_float),       # m/s
        ("pitch_angle", c_float),           # rad（已映射）
        ("yaw_big_angle", c_double),        # rad，多圈连续（有延迟/误差）
        ("yaw_big_omega", c_float),         # rad/s
        ("yaw_small_angle", c_float),       # rad（相对大 yaw，可信）
        ("yaw_small_omega", c_float),       # rad/s
        ("chassis_imu_yaw", c_float),       # rad，0~2π
        ("chassis_imu_omega", c_float),     # rad/s
        ("mark", c_uint8),                  # 电控递增标志位
        ("color", c_uint8),                 # 颜色
        ("auto_aim_switch", c_uint8),       # 电控自瞄开关
        ("yaw_big_temperature", c_uint8),   # ℃
        ("yaw_small_temperature", c_uint8), # ℃
        # MCU2 数据的新样本序号（大 yaw 与底盘 IMU **同源**，共用同一个序号）:
        # MCU1 每从 MCU2 取到一次新数据 +1；两次更新之间重复发送旧值时序号不变。
        # 新样本判定: 首帧到达 || 序号变化 || 值变化（无"0 = 从未收到"哨兵约定）。
        ("mcu2_seq", c_uint8),              # MCU2 新样本序号（每次取到新数据 +1；值保持时不变）
    ]


class TcbsRobotImuData(Structure):
    """大 yaw 转子上 IMU 的反馈。"""

    _fields_ = [
        ("valid", c_uint8),
        ("gx", c_float), ("gy", c_float), ("gz", c_float),   # rad/s
        ("ax", c_float), ("ay", c_float), ("az", c_float),   # m/s²
        ("euler_yaw", c_double),                             # rad（ZXY）
        ("euler_pitch", c_double),
        ("euler_roll", c_double),
        ("dt_one_tenth_ms", c_uint32),                       # 0.1ms 单位
    ]


class TcbsRobotLatestData(Structure):
    """最新一帧原始反馈。"""

    _fields_ = [
        ("mcu", TcbsRobotMcuData),
        ("imu", TcbsRobotImuData),
    ]


class TcbsRobotSourceInfo(Structure):
    """单个数据源的更新情况（年龄/间隔均由上位机计时；MCU 侧无时钟）。"""

    _fields_ = [
        ("valid", c_uint8),        # 是否有可用数据（"值保持"通道 = 已收到过）
        ("age_s", c_double),       # 值年龄 s（-1 = 从未收到；值保持期间持续增大）
        ("count", c_uint32),       # 收到帧数（含值被保持的帧）
        ("new_samples", c_uint32), # 真正的新样本数（序号变化）
        ("rejected", c_uint32),    # 收到但值被保持（非新数据）的帧数
        ("stale", c_uint8),        # 年龄超过 stale_age_s
    ]


class TcbsRobotProvenance(Structure):
    """本次估计所用的数据来源。"""

    _fields_ = [
        ("imu", TcbsRobotSourceInfo),
        ("big_enc", TcbsRobotSourceInfo),
        ("small_enc", TcbsRobotSourceInfo),
        ("pitch_enc", TcbsRobotSourceInfo),
        ("chassis_imu", TcbsRobotSourceInfo),
        ("big_rate_from_imu", c_uint8),
        ("reverse_from_trusted", c_uint8),
        ("big_enc_delay_used", c_double),      # 本帧大 yaw 值的实测年龄 s（上位机计时）
        ("big_enc_innovation", c_double),
        ("big_enc_interval_s", c_double),      # 最近两次大 yaw 新样本间隔 s
        ("big_enc_sample_age_s", c_double),    # 最近一次大 yaw 新样本的实测年龄 s
        ("used_mask", c_uint64),   # bit0 IMU, bit1 大yaw编码器, bit2 小yaw, bit3 pitch, bit4 底盘IMU
    ]


class TcbsRobotEstimate(Structure):
    """完整状态估计: 可信实时量 + 大 yaw 延迟补偿 + 反解真实位姿 + 外生量 + 来源。"""

    _fields_ = [
        ("valid", c_uint8),
        # 1) 可信实时量
        ("imu_yaw", c_double),
        ("imu_pitch", c_double),
        ("imu_roll", c_double),
        ("platform_azimuth", c_double),   # ψ_big（解卷绕）
        ("platform_rate", c_double),
        ("small_joint_angle", c_double),  # θ_small
        ("small_joint_rate", c_double),
        ("pitch_joint_angle", c_double),
        ("pitch_joint_rate", c_double),
        # 2) 大 yaw（延迟编码器 + IMU 速率 → 延迟补偿）
        ("big_joint_angle_meas", c_double),
        ("big_joint_angle", c_double),
        ("big_joint_rate", c_double),
        ("big_enc_age", c_double),            # 大 yaw 值的年龄 s（上位机计时，-1 = 从未收到）
        ("big_sample_interval", c_double),    # 最近两次新样本间隔 s（上位机计时）
        ("chassis_imu_age", c_double),        # 底盘 IMU 值的年龄 s（上位机计时，-1 = 从未收到）
        ("big_enc_innovation", c_double),
        ("big_has_encoder", c_uint8),
        # 3) 反解真实位姿
        ("head_world_yaw", c_double),
        ("head_world_pitch", c_double),
        ("head_world_roll", c_double),
        ("small_output_azimuth", c_double),  # ψ_small（解卷绕）
        ("los_azimuth", c_double),
        ("los_elevation", c_double),
        ("chassis_azimuth", c_double),
        ("chassis_yaw_rate", c_double),
        # 4) 模型外生量（平面 8 参模型；见 mpc/planar_yaw_model.h 的 ModelExo）
        ("base_omega", c_double * 3),      # 底盘角速度（关节参考系 C）
        ("gravity_a", c_double * 3),       # ★ 重力矢量（**A 系 = 大 yaw 转子系**，指向下）
        ("pitch_acc", c_double),           # pitch 关节角加速度（平面模型不含 pitch，仅供显示）
        # 数据来源
        ("prov", TcbsRobotProvenance),
    ]


class TcbsDualYawJointLimits(Structure):
    """单关节约束。"""

    _fields_ = [
        ("max_torque", c_double),
        ("max_torque_rate", c_double),
        ("min_angle", c_double),
        ("max_angle", c_double),
    ]


class TcbsDualYawModelParams(Structure):
    """模型参数（对应 dual_yaw::ModelParams — **平面 2 自由度 8 参模型**）。

    字段顺序与 C++ 结构体 / C 头文件完全一致:
      * 实测几何: ``dx, dy``（小 yaw 轴相对大 yaw 轴的平面偏置）、``gravity``、
        ``m_u_known``（可选上装质量，不称重保持 0）；
      * ★8 参待辨识: ``Jbig_eff, Js, Px, Py, fcBig, fvBig, fcSmall, fvSmall``
        （顺序同 ``dual_yaw::paramsToVector`` / 回归矩阵列序）；
      * 固定/可选: ``frictionLambda, tau_offset_big, tau_offset_small``。

    用 :func:`model_params_to_vector` / :func:`vector_to_model_params` 与 8 维参数向量互转。
    """

    _fields_ = [
        # ── 实测几何（不辨识）──
        ("dx", c_double), ("dy", c_double),       # 非共轴平面偏置 d = (dx, dy)，m
        ("gravity", c_double),                    # 重力加速度，m/s²
        ("m_u_known", c_double),                  # 可选上装质量 kg（仅倾斜时 m_u·d·g⊥ 项）
        # ── ★8 参待辨识 ──
        ("Jbig_eff", c_double),                   # [0] 大 yaw 侧惯量（含 m_u|d|²），kg·m²
        ("Js", c_double),                         # [1] 上装绕小 yaw 轴总惯量，kg·m²
        ("Px", c_double),                         # [2] 上装一阶矩 m_u·ρ_x，kg·m
        ("Py", c_double),                         # [3] 上装一阶矩 m_u·ρ_y，kg·m
        ("fcBig", c_double), ("fvBig", c_double),       # [4][5] 大 yaw 库仑/粘滞摩擦
        ("fcSmall", c_double), ("fvSmall", c_double),   # [6][7] 小 yaw 库仑/粘滞摩擦
        # ── 固定 / 可选 ──
        ("frictionLambda", c_double),             # tanh 软符号陡度 λ（默认 10）
        ("tau_offset_big", c_double),             # 可选常数负载（默认 0 = 关闭），N·m
        ("tau_offset_small", c_double),
    ]


class TcbsDualYawMpcConfig(Structure):
    """MPC 配置（对应 dual_yaw::TcbsDualYawMpcConfig）。

    注 1: v4 起**没有** ``extrapolate_pitch`` —— 平面模型不含 pitch 自由度。
    注 2: 本 C ABI 结构体布局已冻结, **没有** ``small_center_angle`` 字段：
          C++ 侧的小 yaw 回中目标角由行程中心派生
          （``0.5·(small.min_angle + small.max_angle)``，默认行程 ±30° ⇒ 0），
          与 ``defaultMpcConfig()`` 的默认值一致。需要自定中心请用 C++ 的
          ``DualYawMpc::setConfig()``。
    注 3: ``small.min_angle/max_angle`` 是机械行程（默认 ±30°；**允许非对称**）；
          软限位由 ``small_limit_soft_ratio`` 从**两侧各自**向内推：
          ``soft_min = min + (1−ratio)·(max−min)``、``soft_max = max − (1−ratio)·(max−min)``
          （默认 ratio=0.75 ⇒ 软限位区 [−13.75°, +8.75°]）。
    """

    _fields_ = [
        ("dt_control", c_double),
        ("N", c_int32),
        ("substeps", c_int32),
        ("use_rk4", c_uint8),
        ("max_iter", c_int32),
        ("w_big_azimuth", c_double),
        ("w_small_azimuth", c_double),
        ("w_small_center", c_double),
        ("w_small_limit", c_double),
        ("small_limit_soft_ratio", c_double),
        ("r_big_torque", c_double),
        ("r_small_torque", c_double),
        ("rd_big_rate", c_double),
        ("rd_small_rate", c_double),
        ("smooth_eps", c_double),
        ("big", TcbsDualYawJointLimits),
        ("small", TcbsDualYawJointLimits),
        ("ref_delay_steps", c_int32),
    ]


class TcbsEstimatorConfig(Structure):
    """状态估计配置（对应 YawStateEstimator::Config）。"""

    _fields_ = [
        # IMU 安装位置（运行时切换；0 = ON_BIG_YAW，1 = ON_HEAD）→ 用 IMU_ON_BIG_YAW / IMU_ON_HEAD
        ("imu_location", c_int32),
        ("mount_yaw", c_double),             # R_A_IMU（IMU → 大 yaw 转子 A 系），ZXY，rad
        ("mount_pitch", c_double),           #   —— 仅 ON_BIG_YAW 使用
        ("mount_roll", c_double),
        ("head_mount_yaw", c_double),        # R_H_IMU（IMU → 头 H 系），ZXY，rad
        ("head_mount_pitch", c_double),      #   —— 仅 ON_HEAD 使用
        ("head_mount_roll", c_double),
        ("transport_delay_s", c_double),     # 链路传输时延 s（MCU 无时钟；值年龄另由上位机计时）
        ("big_enc_max_jump", c_double),
        ("stale_age_s", c_double),           # 采样年龄超过该值 → prov.*.stale
        ("chassis_imu_timeout_s", c_double), # 底盘 IMU 可用超时 s（零阶保持）
        ("max_extrap_s", c_double),
        ("rate_lpf_alpha", c_double),
        ("pitch_rate_lpf_alpha", c_double),
        ("pitch_acc_lpf_alpha", c_double),
        ("bore", c_double * 3),
        ("gravity", c_double),
        ("use_chassis_imu", c_uint8),
        ("source_timeout_s", c_double),
    ]


class TcbsLinearParams(Structure):
    """编码器/指令线性映射（对应 McuDataPreprocessor::TcbsLinearParams）。"""

    _fields_ = [
        ("send_pitch_scale", c_double),
        ("send_pitch_offset", c_double),
        ("recv_pitch_scale", c_double),
        ("recv_pitch_offset", c_double),
        ("recv_big_yaw_scale", c_double),
        ("recv_big_yaw_offset", c_double),
        ("recv_big_omega_scale", c_double),
        ("send_big_yaw_scale", c_double),
        ("send_big_yaw_offset", c_double),
        ("send_big_velocity_scale", c_double),
        ("send_big_torque_scale", c_double),
        ("recv_small_yaw_scale", c_double),
        ("recv_small_yaw_offset", c_double),
        ("recv_small_omega_scale", c_double),
        ("send_small_yaw_scale", c_double),
        ("send_small_yaw_offset", c_double),
        ("send_small_velocity_scale", c_double),
        ("send_small_torque_scale", c_double),
    ]


class TcbsControllerConfig(Structure):
    """控制器配置（对应 McuMpcController::Config）。"""

    _fields_ = [
        ("loop_period", c_double),
        ("big_torque_only", c_uint8),
        ("small_torque_only", c_uint8),
        ("ref_delay_steps", c_int32),
        ("integral_gain", c_double * 2),
        ("integral_limit", c_double * 2),
        ("integral_on_big", c_uint8),
    ]


class TcbsRobotMpcData(Structure):
    """MPC / 控制输出（序列长度在此；内容由 get_*_sequence 取）。"""

    _fields_ = [
        ("torque", c_double * 2),
        ("torque_mpc", c_double * 2),
        ("integral", c_double * 2),
        ("target_joint", c_double * 2),
        ("target_joint_rate", c_double * 2),
        ("big_torque_only", c_uint8),
        ("small_torque_only", c_uint8),
        ("ref_azimuth", c_double * 2),
        ("delayed_ref_azimuth", c_double * 2),
        ("small_ref_over_limit", c_uint8),
        ("solve_ms", c_double),
        ("loop_fps", c_double),
        ("ticks_since_set", c_uint64),
        ("solve_count", c_uint32),
        ("solve_fail_count", c_uint32),
        ("estimator_valid", c_uint8),
        ("sent_ok", c_uint8),
        ("ref_azimuth_seq_len", c_int32 * 2),
        ("pred_azimuth_seq_len", c_int32 * 2),
        ("pred_joint_seq_len", c_int32 * 2),
    ]


class TcbsRobotControllerState(Structure):
    """完整状态: mcu / imu / est / mpc 四组。"""

    _fields_ = [
        ("mcu", TcbsRobotMcuData),
        ("imu", TcbsRobotImuData),
        ("est", TcbsRobotEstimate),
        ("mpc", TcbsRobotMpcData),
    ]


class TcbsRobotCommAbiInfo(Structure):
    """布局自检信息。"""

    _fields_ = [
        ("api_version", c_uint32),
        ("sizeof_pointer", c_uint32),
        ("sizeof_latest_data", c_uint32),
        ("sizeof_estimate", c_uint32),
        ("sizeof_source_info", c_uint32),
        ("sizeof_provenance", c_uint32),
        ("sizeof_controller_state", c_uint32),
        ("sizeof_mpc_data", c_uint32),
        ("sizeof_model_params", c_uint32),
        ("sizeof_mpc_config", c_uint32),
        ("sizeof_estimator_config", c_uint32),
        ("sizeof_linear_params", c_uint32),
        ("sizeof_controller_config", c_uint32),
    ]


# ============================================================================
# 函数签名
# ============================================================================
_SIGNATURES = [
    ("tcbs_robot_comm_strerror", [ctypes.c_int], ctypes.c_char_p),
    ("tcbs_robot_comm_abi_info", [POINTER(TcbsRobotCommAbiInfo)], ctypes.c_int),
    ("tcbs_robot_comm_check_abi", [c_uint32, POINTER(TcbsRobotCommAbiInfo)], ctypes.c_int),

    ("tcbs_robot_comm_create", [], c_void_p),
    ("tcbs_robot_comm_destroy", [c_void_p], None),
    ("tcbs_robot_comm_stop", [c_void_p], None),
    ("tcbs_robot_comm_get_latest_data", [c_void_p, POINTER(TcbsRobotLatestData)], ctypes.c_int),
    ("tcbs_robot_comm_get_estimate", [c_void_p, POINTER(TcbsRobotEstimate)], ctypes.c_int),
    ("tcbs_robot_comm_send_to_mcu",
     [c_void_p, c_uint8, c_uint8, c_float, c_uint8, c_double, c_float, c_float,
      c_uint8, c_float, c_float, c_float], ctypes.c_int),
    ("tcbs_robot_comm_send_to_imu", [c_void_p], ctypes.c_int),

    ("tcbs_robot_controller_create", [], c_void_p),
    ("tcbs_robot_controller_destroy", [c_void_p], None),
    ("tcbs_robot_controller_mode", [c_void_p], ctypes.c_int),
    ("tcbs_robot_controller_default_model_params", [POINTER(TcbsDualYawModelParams)], ctypes.c_int),
    ("tcbs_robot_controller_default_mpc_config", [POINTER(TcbsDualYawMpcConfig)], ctypes.c_int),
    ("tcbs_robot_controller_default_estimator_config", [POINTER(TcbsEstimatorConfig)], ctypes.c_int),
    ("tcbs_robot_controller_default_linear_params", [POINTER(TcbsLinearParams)], ctypes.c_int),
    ("tcbs_robot_controller_default_controller_config", [POINTER(TcbsControllerConfig)], ctypes.c_int),
    ("tcbs_robot_controller_get_model_params", [c_void_p, POINTER(TcbsDualYawModelParams)], ctypes.c_int),
    ("tcbs_robot_controller_get_mpc_config", [c_void_p, POINTER(TcbsDualYawMpcConfig)], ctypes.c_int),
    ("tcbs_robot_controller_get_estimator_config", [c_void_p, POINTER(TcbsEstimatorConfig)], ctypes.c_int),
    ("tcbs_robot_controller_set_model_params", [c_void_p, POINTER(TcbsDualYawModelParams)], ctypes.c_int),
    ("tcbs_robot_controller_set_mpc_config", [c_void_p, POINTER(TcbsDualYawMpcConfig)], ctypes.c_int),
    ("tcbs_robot_controller_set_estimator_config", [c_void_p, POINTER(TcbsEstimatorConfig)], ctypes.c_int),
    ("tcbs_robot_controller_set_linear_params", [c_void_p, POINTER(TcbsLinearParams)], ctypes.c_int),
    ("tcbs_robot_controller_set_controller_config", [c_void_p, POINTER(TcbsControllerConfig)], ctypes.c_int),
    ("tcbs_robot_controller_set",
     [c_void_p, c_uint8, c_uint8, c_uint8, c_double, c_double, c_double, c_uint8, c_uint8],
     ctypes.c_int),
    ("tcbs_robot_controller_set_joint_angles",
     [c_void_p, c_uint8, c_uint8, c_uint8, c_double, c_double, c_double, c_uint8, c_uint8],
     ctypes.c_int),
    ("tcbs_robot_controller_set_sequence",
     [c_void_p, c_uint8, c_uint8, c_uint8,
      POINTER(c_double), c_int32, POINTER(c_double), c_int32, POINTER(c_double), c_int32,
      POINTER(c_uint8), c_int32, c_uint8], ctypes.c_int),
    ("tcbs_robot_controller_set_sequence_unchecked",
     [c_void_p, c_uint8, c_uint8, c_uint8,
      POINTER(c_double), c_int32, POINTER(c_double), c_int32, POINTER(c_double), c_int32,
      POINTER(c_uint8), c_int32, c_uint8], ctypes.c_int),
    ("tcbs_robot_controller_get_state", [c_void_p, POINTER(TcbsRobotControllerState)], ctypes.c_int),
    ("tcbs_robot_controller_get_ref_sequence", [c_void_p, c_int32, POINTER(c_double), c_int32], ctypes.c_int),
    ("tcbs_robot_controller_get_pred_sequence", [c_void_p, c_int32, POINTER(c_double), c_int32], ctypes.c_int),
    ("tcbs_robot_controller_get_pred_joint_sequence",
     [c_void_p, c_int32, POINTER(c_double), c_int32], ctypes.c_int),
]

_ABI_STRUCTS = [
    ("sizeof_latest_data", TcbsRobotLatestData),
    ("sizeof_estimate", TcbsRobotEstimate),
    ("sizeof_source_info", TcbsRobotSourceInfo),
    ("sizeof_provenance", TcbsRobotProvenance),
    ("sizeof_controller_state", TcbsRobotControllerState),
    ("sizeof_mpc_data", TcbsRobotMpcData),
    ("sizeof_model_params", TcbsDualYawModelParams),
    ("sizeof_mpc_config", TcbsDualYawMpcConfig),
    ("sizeof_estimator_config", TcbsEstimatorConfig),
    ("sizeof_linear_params", TcbsLinearParams),
    ("sizeof_controller_config", TcbsControllerConfig),
]


def _configure(lib):
    for name, argtypes, restype in _SIGNATURES:
        fn = getattr(lib, name, None)
        if fn is None:
            raise TorqueControllerError(f"库中缺少符号 {name}（版本不匹配？）")
        fn.argtypes = argtypes
        fn.restype = restype


# 导入时尝试加载（失败不致命，首次调用 API 时再报错）
_lib = None
_load_error = None
try:
    _lib = load_library()
    _configure(_lib)
except TorqueControllerError as _exc:  # pragma: no cover - 未构建时
    _load_error = _exc


def library_path() -> str | None:
    """当前加载的库路径（未加载返回 None）。"""
    if _lib is None:
        return None
    try:
        return _lib._name
    except AttributeError:  # pragma: no cover
        return None


def _require_lib():
    if _lib is None:
        raise TorqueControllerError(f"C API 库不可用: {_load_error}")
    return _lib


def strerror(status: int) -> str:
    """错误码 → 可读字符串。"""
    lib = _require_lib()
    raw = lib.tcbs_robot_comm_strerror(int(status))
    return raw.decode("utf-8", "replace") if raw else f"status {status}"


def _check(status: int, what: str) -> int:
    if status < 0:
        raise TorqueControllerError(f"{what} 失败: {status} ({strerror(status)})", status)
    return status


def _python_abi_sizes() -> TcbsRobotCommAbiInfo:
    """按 Python 侧 ctypes 结构体算出各结构体 sizeof（供 C 侧 tcbs_robot_comm_check_abi 比对）。"""
    info = TcbsRobotCommAbiInfo()
    info.api_version = API_VERSION
    info.sizeof_pointer = sizeof(c_void_p)
    for field_name, cls in _ABI_STRUCTS:
        setattr(info, field_name, sizeof(cls))
    return info


def check_abi() -> SimpleNamespace:
    """校验 ctypes 结构体布局与 C 库一致；不一致直接抛异常。

    流程: 先让 **C 库** 用 ``tcbs_robot_comm_check_abi(版本号, Python 侧 sizeof 表)`` 判定
    （不匹配时库返回 ``ERR_ABI_MISMATCH`` 并打印是哪一项漂了），再用 Python 侧比对
    给出精确的字段级诊断。
    """
    lib = _require_lib()
    info = TcbsRobotCommAbiInfo()
    _check(lib.tcbs_robot_comm_abi_info(byref(info)), "tcbs_robot_comm_abi_info")
    if info.api_version != API_VERSION:
        raise TorqueControllerError(
            f"C API 版本不匹配: 库={info.api_version}, Python 绑定={API_VERSION}")
    if info.sizeof_pointer != sizeof(c_void_p):
        raise TorqueControllerError(
            f"指针宽度不匹配: 库={info.sizeof_pointer}, Python={sizeof(c_void_p)}")
    for field_name, cls in _ABI_STRUCTS:
        lib_size = getattr(info, field_name)
        py_size = sizeof(cls)
        if lib_size != py_size:
            raise TorqueControllerError(
                f"结构体布局不匹配 {cls.__name__}: 库={lib_size}, Python={py_size}"
                f"（C 头文件与 _bridge.py 不同步？）")
    # C 侧的权威判定（版本号 + 布局）: 正常情况下必然 OK；异常时返回负错误码
    _check(lib.tcbs_robot_comm_check_abi(API_VERSION, byref(_python_abi_sizes())),
           "tcbs_robot_comm_check_abi")
    return info


def _to_obj(value):
    """ctypes 结构体/数组 → SimpleNamespace / list（递归）。"""
    if isinstance(value, ctypes.Array):
        return [_to_obj(v) for v in value]
    if isinstance(value, Structure):
        obj = SimpleNamespace()
        for name, *_rest in value._fields_:
            setattr(obj, name, _to_obj(getattr(value, name)))
        return obj
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _apply_fields(struct, kwargs):
    """把 kwargs 套用到 ctypes 结构体上（数组字段支持传序列）。"""
    for key, value in kwargs.items():
        field = getattr(struct, key, None)
        if field is None:
            raise TorqueControllerError(f"未知配置字段: {key}")
        if isinstance(field, ctypes.Array):
            if not hasattr(value, "__len__"):
                raise TorqueControllerError(f"字段 {key} 是数组，需要序列（如 list）")
            if len(value) != len(field):
                raise TorqueControllerError(
                    f"字段 {key} 长度不匹配: 需要 {len(field)}，给了 {len(value)}")
            for i, v in enumerate(value):
                field[i] = float(v)
        else:
            setattr(struct, key, value)
    return struct


def _dbl_array(seq):
    """Python 序列 → (c_double 数组或 None, 长度)（空序列 → NULL/0，表示该通道保持当前值）"""
    if not seq:
        return None, 0
    vals = [float(v) for v in seq]
    return (c_double * len(vals))(*vals), len(vals)


def _u8_array(seq):
    if not seq:
        return None, 0
    vals = [1 if v else 0 for v in seq]
    return (c_uint8 * len(vals))(*vals), len(vals)


# ============================================================================
# 默认配置
# ============================================================================
def _default_of(fn_name: str, struct_cls):
    lib = _require_lib()
    s = struct_cls()
    _check(getattr(lib, fn_name)(byref(s)), fn_name)
    return s


def default_model_params() -> TcbsDualYawModelParams:
    """C++ 默认模型参数（占位值，需标定）。"""
    return _default_of("tcbs_robot_controller_default_model_params", TcbsDualYawModelParams)


def default_mpc_config() -> TcbsDualYawMpcConfig:
    """C++ 默认 MPC 配置。"""
    return _default_of("tcbs_robot_controller_default_mpc_config", TcbsDualYawMpcConfig)


def default_estimator_config() -> TcbsEstimatorConfig:
    """C++ 默认状态估计配置。"""
    return _default_of("tcbs_robot_controller_default_estimator_config", TcbsEstimatorConfig)


def default_linear_params() -> TcbsLinearParams:
    """C++ 默认编码器/指令映射。"""
    return _default_of("tcbs_robot_controller_default_linear_params", TcbsLinearParams)


def default_controller_config() -> TcbsControllerConfig:
    """C++ 默认控制器配置。"""
    return _default_of("tcbs_robot_controller_default_controller_config", TcbsControllerConfig)


# ── ★8 参模型参数 ↔ 向量（与 dual_yaw::paramsToVector / vectorToParams 同序）──
def model_params_to_vector(params: TcbsDualYawModelParams) -> list:
    """取 8 个待辨识参数为 list（顺序: Jbig_eff, Js, Px, Py, fc_big, fv_big, fc_small, fv_small）。"""
    return [float(getattr(params, field)) for field in _MODEL_PARAM_FIELDS]


def vector_to_model_params(vector, base: TcbsDualYawModelParams | None = None) -> TcbsDualYawModelParams:
    """把 8 维向量写回模型参数结构体（未给的字段取 ``base``，默认取 C++ 默认值）。

    几何量（dx/dy/gravity/m_u_known）、frictionLambda 与 tau_offset_* 不在 8 参里，
    因此会保留 ``base`` 的值。
    """
    params = base if base is not None else default_model_params()
    vec = [float(v) for v in vector]
    if len(vec) != len(_MODEL_PARAM_FIELDS):
        raise TorqueControllerError(
            f"参数向量长度应为 {len(_MODEL_PARAM_FIELDS)}（{', '.join(MODEL_PARAM_NAMES)}），"
            f"实际 {len(vec)}")
    for field, value in zip(_MODEL_PARAM_FIELDS, vec):
        setattr(params, field, value)
    return params


# ============================================================================
# 低层: TcbsRobotCommunication
# ============================================================================
class TcbsRobotCommunication:
    """TcbsRobotCommunication 的 Python 接口（低层: 串口 + 映射 + 状态估计）。

    无硬件时构造同样成功（串口后台重连），此时 get_latest_data()/get_estimate()
    返回的对象里 ``valid`` 均为 0。
    """

    def __init__(self):
        lib = _require_lib()
        check_abi()
        self._handle = lib.tcbs_robot_comm_create()
        if not self._handle:
            raise TorqueControllerError("tcbs_robot_comm_create() 返回 NULL")

    # ── 数据 ──
    def get_latest_data(self) -> SimpleNamespace:
        """最新一帧原始反馈（mcu/imu 两组，字段名同 C 结构体）。"""
        lib = _require_lib()
        data = TcbsRobotLatestData()
        _check(lib.tcbs_robot_comm_get_latest_data(self._handle, byref(data)),
               "tcbs_robot_comm_get_latest_data")
        return _to_obj(data)

    def get_estimate(self) -> SimpleNamespace:
        """完整状态估计（可信量 / 大 yaw 补偿 / 反解位姿 / 外生量 / 数据来源）。"""
        lib = _require_lib()
        est = TcbsRobotEstimate()
        _check(lib.tcbs_robot_comm_get_estimate(self._handle, byref(est)),
               "tcbs_robot_comm_get_estimate")
        return _to_obj(est)

    # ── 发送 ──
    def send_to_mcu(self, auto_aim_enable=0, fire=0, pitch_target_angle=0.0,
                    yaw_big_mode=0, yaw_big_target_angle=0.0,
                    yaw_big_target_velocity=0.0, yaw_big_torque=0.0,
                    yaw_small_mode=0, yaw_small_target_angle=0.0,
                    yaw_small_target_velocity=0.0, yaw_small_torque=0.0) -> bool:
        """发送 MCU 指令包（内部按映射参数预处理 + 计算 CRC）。

        yaw_*_mode: 0 = 仅力矩，1 = 力矩 + 电控位置/速度内环。
        返回 True = 写串口成功；False = 串口未打开/写失败。
        """
        lib = _require_lib()
        ret = lib.tcbs_robot_comm_send_to_mcu(
            self._handle, int(auto_aim_enable), int(fire), float(pitch_target_angle),
            int(yaw_big_mode), float(yaw_big_target_angle),
            float(yaw_big_target_velocity), float(yaw_big_torque),
            int(yaw_small_mode), float(yaw_small_target_angle),
            float(yaw_small_target_velocity), float(yaw_small_torque))
        _check(ret, "tcbs_robot_comm_send_to_mcu")   # <0 = 错误码；0 = 串口未打开
        return bool(ret)

    def send_to_imu(self) -> bool:
        """发送 IMU 心跳帧（无载荷）。"""
        lib = _require_lib()
        ret = lib.tcbs_robot_comm_send_to_imu(self._handle)
        _check(ret, "tcbs_robot_comm_send_to_imu")
        return bool(ret)

    def stop(self):
        """停止串口后台线程（句柄仍可用，但数据不再更新）。"""
        lib = _require_lib()
        if self._handle:
            lib.tcbs_robot_comm_stop(self._handle)

    def close(self):
        """销毁句柄（停止线程并释放）；可重复调用。"""
        lib = _require_lib()
        if self._handle:
            lib.tcbs_robot_comm_destroy(self._handle)
            self._handle = None

    # ── 生命周期 ──
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# 高层: TcbsRobotController
# ============================================================================
class TcbsRobotController:
    """TcbsRobotController 的 Python 接口（高层: 估计 + 耦合 MPC + 后台发送线程）。

    用法::

        from torque_controller import TcbsRobotController

        with TcbsRobotController() as ctrl:
            # 平面 8 参模型: 标定后替换（未给的字段保持 C++ 默认值）
            ctrl.set_model_params(Jbig_eff=0.024, Js=0.013, Px=1e-3, Py=0.0)
            ctrl.set_mpc_config(N=12, max_iter=8)     # 可选: 覆盖部分 MPC 参数
            st = ctrl.get_state()
            ctrl.set(big_yaw_azimuth=st.est.platform_azimuth,
                     small_yaw_azimuth=st.est.platform_azimuth + 0.2)

    目标语义为**世界方位角**（rad，多圈连续）:
      big_yaw_azimuth   = ψ_big   大 yaw 平台 x 轴世界方位角
      small_yaw_azimuth = ψ_small 小 yaw 输出 x 轴世界方位角（= ψ_big + θ_small）
    无硬件时构造同样成功，get_state() 里 valid 全为 0。
    """

    def __init__(self, lib_path: str | None = None):
        global _lib, _load_error
        if lib_path is not None:                      # 允许显式指定 .so
            _lib = load_library(lib_path)
            _configure(_lib)
            _load_error = None
        lib = _require_lib()
        check_abi()
        self._handle = lib.tcbs_robot_controller_create()
        if not self._handle:
            raise TorqueControllerError("tcbs_robot_controller_create() 返回 NULL")

    # ── 配置 ──
    def get_model_params(self) -> TcbsDualYawModelParams:
        """读取当前模型参数（平面 8 参模型；含几何与可选常数负载）。"""
        lib = _require_lib()
        s = TcbsDualYawModelParams()
        _check(lib.tcbs_robot_controller_get_model_params(self._handle, byref(s)),
               "tcbs_robot_controller_get_model_params")
        return s

    def get_mpc_config(self) -> TcbsDualYawMpcConfig:
        """读取当前 MPC 配置（N / 权重 / 限位 / 积分器等）。"""
        lib = _require_lib()
        s = TcbsDualYawMpcConfig()
        _check(lib.tcbs_robot_controller_get_mpc_config(self._handle, byref(s)),
               "tcbs_robot_controller_get_mpc_config")
        return s

    def get_estimator_config(self) -> TcbsEstimatorConfig:
        """读取当前状态估计配置（含 IMU 安装位置 imu_location）。"""
        lib = _require_lib()
        s = TcbsEstimatorConfig()
        _check(lib.tcbs_robot_controller_get_estimator_config(self._handle, byref(s)),
               "tcbs_robot_controller_get_estimator_config")
        return s

    def set_model_params(self, config: TcbsDualYawModelParams | None = None, **kwargs) -> None:
        """设置模型参数（**平面 8 参模型**；未给的字段用 C++ 默认值，也可直接传结构体）。

        例: ``ctrl.set_model_params(Jbig_eff=0.024, Js=0.013, Px=1e-3, fc_big=0.09)``
        （8 参字段名: Jbig_eff, Js, Px, Py, fcBig, fvBig, fcSmall, fvSmall；
        另有几何 dx/dy/gravity/m_u_known 与 frictionLambda/tau_offset_*）。
        调用会复位 MPC 热启动。
        """
        lib = _require_lib()
        s = _apply_fields(config if config is not None else default_model_params(), kwargs)
        _check(lib.tcbs_robot_controller_set_model_params(self._handle, byref(s)),
               "tcbs_robot_controller_set_model_params")

    def set_mpc_config(self, config: TcbsDualYawMpcConfig | None = None, **kwargs) -> None:
        """设置 MPC 配置（如 ``set_mpc_config(N=12, max_iter=8, use_rk4=1)``）。

        关节约束用嵌套字段: ``ctrl.set_mpc_config(**{"small.max_torque": ...})`` 不支持，
        请先取 ``default_mpc_config()``、改 ``cfg.small.max_torque`` 后再传入。
        """
        lib = _require_lib()
        s = _apply_fields(config if config is not None else default_mpc_config(), kwargs)
        _check(lib.tcbs_robot_controller_set_mpc_config(self._handle, byref(s)),
               "tcbs_robot_controller_set_mpc_config")

    def set_estimator_config(self, config: TcbsEstimatorConfig | None = None, **kwargs) -> None:
        """设置状态估计配置（在线标定: IMU 安装位置/安装角、链路时延、滤波系数等）。

        IMU 位置用 ``imu_location=IMU_ON_BIG_YAW``（0，现状）或 ``IMU_ON_HEAD``（1）；
        选择 ``IMU_ON_HEAD`` 时必须同时给出 ``head_mount_yaw/pitch/roll``（R_H_IMU）。
        """
        lib = _require_lib()
        s = _apply_fields(config if config is not None else default_estimator_config(), kwargs)
        _check(lib.tcbs_robot_controller_set_estimator_config(self._handle, byref(s)),
               "tcbs_robot_controller_set_estimator_config")

    def set_linear_params(self, config: TcbsLinearParams | None = None, **kwargs) -> None:
        """设置编码器/指令线性映射。"""
        lib = _require_lib()
        s = _apply_fields(config if config is not None else default_linear_params(), kwargs)
        _check(lib.tcbs_robot_controller_set_linear_params(self._handle, byref(s)),
               "tcbs_robot_controller_set_linear_params")

    def set_controller_config(self, config: TcbsControllerConfig | None = None, **kwargs) -> None:
        """设置控制器配置（后台线程周期、力矩模式位、积分补偿）。

        注意: McuMpcController 的配置只能在构造时注入，因此本调用会**重建**内部
        TcbsRobotController（后台线程重启、MPC 热启动与积分状态清零）。请在开始控制前调用。
        """
        lib = _require_lib()
        s = _apply_fields(config if config is not None else default_controller_config(), kwargs)
        _check(lib.tcbs_robot_controller_set_controller_config(self._handle, byref(s)),
               "tcbs_robot_controller_set_controller_config")

    # ── 目标设置 ──
    def set(self, big_yaw_azimuth: float, small_yaw_azimuth: float,
            pitch_target_angle: float = 0.0, auto_aim_enable: bool = True,
            big_torque_only: bool = False, small_torque_only: bool = False,
            fire: bool = False, integral_enable: bool = False) -> None:
        """按**世界方位角**设置目标（rad，多圈连续）。"""
        lib = _require_lib()
        _check(lib.tcbs_robot_controller_set(
            self._handle, int(bool(auto_aim_enable)), int(bool(big_torque_only)),
            int(bool(small_torque_only)), float(big_yaw_azimuth), float(small_yaw_azimuth),
            float(pitch_target_angle), int(bool(fire)), int(bool(integral_enable))),
            "tcbs_robot_controller_set")

    def set_joint_angles(self, big_joint_angle: float, small_joint_angle: float,
                         pitch_target_angle: float = 0.0, auto_aim_enable: bool = True,
                         big_torque_only: bool = False, small_torque_only: bool = False,
                         fire: bool = False, integral_enable: bool = False) -> None:
        """按**关节系**角度设置目标（内部按当前底盘方位角估计换算为世界方位角）。

        big_joint_angle: 大 yaw 关节角（相对底盘，多圈，rad）
        small_joint_angle: 小 yaw 关节角（相对大 yaw，行程 ±30°，rad）
        """
        lib = _require_lib()
        _check(lib.tcbs_robot_controller_set_joint_angles(
            self._handle, int(bool(auto_aim_enable)), int(bool(big_torque_only)),
            int(bool(small_torque_only)), float(big_joint_angle), float(small_joint_angle),
            float(pitch_target_angle), int(bool(fire)), int(bool(integral_enable))),
            "tcbs_robot_controller_set_joint_angles")

    def set_sequence(self, big_yaw_azimuth_seq=(), small_yaw_azimuth_seq=(),
                     pitch_seq=(), fire_seq=(), auto_aim_enable: bool = True,
                     big_torque_only: bool = False, small_torque_only: bool = False,
                     integral_enable: bool = False, unchecked: bool = False) -> None:
        """按序列设置目标（各通道独立，不截断；空序列 = 该通道保持当前值）。

        ``unchecked=False``（默认）走 ``TcbsRobotController::set(序列)``，会检查模式位；
        当前 C++ 实现没有对外切换 SEQUENCE 模式的接口，故通常抛
        ``ERR_MODE_MISMATCH``。``unchecked=True`` 直接调用 ``McuMpcController::set``，
        绕过模式保护（语义等价于 SEQUENCE 模式，但 mode 仍报告 SINGLE）。
        """
        lib = _require_lib()
        big, big_n = _dbl_array(big_yaw_azimuth_seq)
        small, small_n = _dbl_array(small_yaw_azimuth_seq)
        pitch, pitch_n = _dbl_array(pitch_seq)
        fire, fire_n = _u8_array(fire_seq)
        fn = (lib.tcbs_robot_controller_set_sequence_unchecked if unchecked
              else lib.tcbs_robot_controller_set_sequence)
        status = fn(self._handle, int(bool(auto_aim_enable)), int(bool(big_torque_only)),
                    int(bool(small_torque_only)), big, big_n, small, small_n,
                    pitch, pitch_n, fire, fire_n, int(bool(integral_enable)))
        if status == ERR_MODE_MISMATCH and not unchecked:
            raise TorqueControllerError(
                "序列 set 被模式检查拒绝（当前为 SINGLE 模式；TcbsRobotController 未提供切换 "
                "SEQUENCE 模式的接口）。如需强制下发序列请使用 set_sequence(..., unchecked=True)。",
                status)
        _check(status, "tcbs_robot_controller_set_sequence")

    # ── 状态 ──
    def get_state(self) -> SimpleNamespace:
        """取完整状态（mcu/imu/est/mpc 四组；预测/参考序列为 Python list）。"""
        lib = _require_lib()
        state = TcbsRobotControllerState()
        _check(lib.tcbs_robot_controller_get_state(self._handle, byref(state)),
               "tcbs_robot_controller_get_state")
        obj = _to_obj(state)
        mpc = obj.mpc
        mpc.ref_azimuth_seq = [self._get_sequence(lib.tcbs_robot_controller_get_ref_sequence, i)
                               for i in (0, 1)]
        mpc.pred_azimuth_seq = [self._get_sequence(lib.tcbs_robot_controller_get_pred_sequence, i)
                                for i in (0, 1)]
        mpc.pred_joint_seq = [self._get_sequence(lib.tcbs_robot_controller_get_pred_joint_sequence, i)
                              for i in (0, 1)]
        return obj

    def _get_sequence(self, fn, which: int):
        n = fn(self._handle, int(which), None, 0)      # 先查询长度
        if n <= 0:
            return []
        buf = (c_double * n)()
        n = fn(self._handle, int(which), buf, n)
        return list(buf[:n]) if n > 0 else []

    def get_ref_sequence(self, which: int = 0):
        """参考世界方位角序列（which: 0 = 大 yaw, 1 = 小 yaw），返回 list。"""
        lib = _require_lib()
        return self._get_sequence(lib.tcbs_robot_controller_get_ref_sequence, which)

    def get_pred_sequence(self, which: int = 0):
        """预测世界方位角序列（which: 0 = 大 yaw, 1 = 小 yaw），返回 list。"""
        lib = _require_lib()
        return self._get_sequence(lib.tcbs_robot_controller_get_pred_sequence, which)

    def get_pred_joint_sequence(self, which: int = 0):
        """预测关节角序列（which: 0 = 大 yaw, 1 = 小 yaw），返回 list。"""
        lib = _require_lib()
        return self._get_sequence(lib.tcbs_robot_controller_get_pred_joint_sequence, which)

    # ── 模式 ──
    @property
    def mode_code(self) -> int:
        """0 = SINGLE, 1 = SEQUENCE。"""
        lib = _require_lib()
        return _check(lib.tcbs_robot_controller_mode(self._handle), "tcbs_robot_controller_mode")

    @property
    def mode(self) -> str:
        """'SINGLE' / 'SEQUENCE'。"""
        return "SEQUENCE" if self.mode_code == MODE_SEQUENCE else "SINGLE"

    # ── 生命周期 ──
    def close(self):
        """销毁句柄（停后台线程 + 关串口）；可重复调用。"""
        lib = _require_lib()
        if self._handle:
            lib.tcbs_robot_controller_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
