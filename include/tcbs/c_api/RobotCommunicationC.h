#ifndef TCBS_ROBOT_COMMUNICATION_C_H
#define TCBS_ROBOT_COMMUNICATION_C_H

// ============================================================================
// RobotCommunicationC.h — 双级 yaw 云台 C API（纯 C ABI）
//
// 目的: 把 C++ 侧的 RobotCommunication（低层: 串口 + 映射 + 状态估计）与
//       RobotController（高层: 估计 + 耦合 MPC + 后台发送线程）暴露成**稳定的 C 接口**，
//       供 Python ctypes（python/torque_controller）或任意 C 程序调用。
//
// 约定:
//   - 本头文件是**纯 C**（只用 stdint.h/stddef.h 类型），不跨越 C ABI 传递任何 STL 类型。
//     所有结构体都是固定宽度标量/定长数组（double/float/uint8_t/uint32_t/uint64_t/int32_t）。
//   - 布尔量一律用 uint8_t（0 = false，1 = true）。
//   - 所有可能抛异常的 C++ 路径都在实现内 catch 并转成错误码（见 TcbsRobotCommStatus）；
//     返回 int 的函数: >= 0 表示成功（部分函数返回长度），< 0 表示错误码。
//   - 结构体按平台默认对齐（不 pack）；调用方（ctypes）必须使用相同的默认对齐。
//     版本/布局自检: tcbs_robot_comm_abi_info() 返回本库各结构的 sizeof，Python 侧导入时比对。
//   - 线程安全: 句柄内部自带互斥；可从多线程调用，但不要对同一句柄并发 destroy。
//
// 关于"无硬件也能创建":
//   tcbs_robot_comm_create() / tcbs_robot_controller_create() 在**没有串口硬件**时同样成功返回
//   （串口打开失败不会抛异常，只会在后台线程里周期性重连），此时
//   get_latest_data()/get_state() 的 valid 标志全为 0（mcu.valid = imu.valid = est.valid = 0），
//   发送函数返回失败（send_to_mcu 的返回值 / state.mpc.sent_ok = 0），但 CPU 上的
//   MPC 后台线程、状态估计与接口调用链路完全可用。这正是"离线调试/纯 Python 演示"的前提。
//
// 关于"序列模式"（重要，来自 C++ 侧现状）:
//   RobotController::set(序列) 在内部 mode() 为 SINGLE 时会抛 std::runtime_error；
//   而 RobotController 当前**没有**对外切换 SEQUENCE 模式的接口（sequence_mode_ 私有且
//   默认 false）。因此 tcbs_robot_controller_set_sequence() 在现有 C++ 实现下会返回
//   TCBS_ROBOT_COMM_ERR_MODE_MISMATCH。若确需序列控制，见
//   tcbs_robot_controller_set_sequence_unchecked()（绕过模式位、经 McuMpcController 直接下发）。
// ============================================================================

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── API 版本（结构体布局或语义变化时 +1）──
// v2: MCU 端不提供时钟 → 删除 mcu_tick_ms，改用序号；SourceInfo 增 new_samples/stale；
//     Provenance 增 big_enc_interval_s/big_enc_sample_age_s；Estimate 增
//     big_sample_interval/chassis_imu_age；Config 的 big_enc_delay_s → transport_delay_s，
//     增 stale_age_s/chassis_imu_timeout_s。年龄/间隔/时延全部由**上位机计时**。
// v3: 大 yaw 与底盘 IMU 同源（MCU2 一次送来）→ 两个序号 yaw_big_seq/chassis_imu_seq
//     合并为单个 mcu2_seq（TcbsRobotMcuData_C 布局随之变化）；取消"0 = 从未收到"哨兵约定
//     （新样本判定规则: 首帧到达 || 序号变化 || 值变化）。
//     ※ v2 的绑定与 v3 的库**不可混用**（sizeof/偏移不同），故升版本号而非仅改注释。
// v4: 动力学模型由"40 维刚体参数的三刚体链"更换为**平面 2 自由度 8 参模型**
//     （mpc/planar_yaw_model.h）:
//       - TcbsDualYawModelParams_C 重写为 8 个待辨识参数 + 实测几何 + 可选常数负载；
//       - TcbsDualYawMpcConfig_C 删除 extrapolate_pitch（模型已不含 pitch 自由度）；
//       - TcbsEstimatorConfig_C 增加 imu_location（IMU 装在**大 yaw 转子**还是**头**上）
//         与 head_mount_* 安装角；
//       - TcbsRobotEstimate_C 的 gravity_c[3] 改为 gravity_a[3]（★ **语义变了**: 重力在
//         **A 系（大 yaw 转子系）** 的投影，不再是 C 系）；
//       - 新增 tcbs_robot_controller_get_* 配置读取与 tcbs_robot_comm_check_abi（版本/布局自检）。
//     ※ 旧绑定的 sizeof/偏移与 v4 不同，**不可混用**。
#define TCBS_ROBOT_COMM_C_API_VERSION 4u

// ============================================================================
// 错误码
// ============================================================================
typedef enum TcbsRobotCommStatus {
    TCBS_ROBOT_COMM_OK                =  0,   // 成功
    TCBS_ROBOT_COMM_ERR_NULL_HANDLE   = -1,   // 句柄为 NULL
    TCBS_ROBOT_COMM_ERR_INVALID_ARG   = -2,   // 参数非法（NULL 指针/负长度/未知 which 等）
    TCBS_ROBOT_COMM_ERR_MODE_MISMATCH = -3,   // RobotController 模式不匹配（单点 set ↔ 序列 set）
    TCBS_ROBOT_COMM_ERR_RUNTIME       = -4,   // C++ 侧 std::exception（消息见 strerror 与 stderr）
    TCBS_ROBOT_COMM_ERR_UNKNOWN       = -5,   // 非 std::exception 的未知异常
    TCBS_ROBOT_COMM_ERR_ALLOC         = -6,   // 句柄分配/构造失败
    TCBS_ROBOT_COMM_ERR_ABI_MISMATCH  = -7    // ABI 版本或结构体布局不匹配（见 tcbs_robot_comm_check_abi）
} TcbsRobotCommStatus;

// 错误码 → 静态字符串（不释放；未知码返回 "unknown status"）
const char* tcbs_robot_comm_strerror(int status);

// ============================================================================
// 不透明句柄
// ============================================================================
// 低层句柄: 内部持有 RobotCommunication（MCU/IMU 串口 + 映射 + YawStateEstimator）
typedef struct TcbsRobotCommHandle TcbsRobotCommHandle;

// 高层句柄: 内部持有 RobotController（含 McuMpcController 后台线程）
typedef struct TcbsRobotController_C TcbsRobotController_C;

// ============================================================================
// 一、低层数据结构（RobotCommunication）
// ============================================================================

// MCU 反馈（已按 McuDataPreprocessor::LinearParams 映射）
typedef struct TcbsRobotMcuData_C {
    uint8_t  valid;                 // 是否收到过有效 MCU 帧（0 = 从未收到）
    float    bullet_velocity;       // 弹速，m/s
    float    pitch_angle;           // pitch 关节角，rad（已映射）
    double   yaw_big_angle;         // 大 yaw 关节角，rad，多圈连续（链路有延迟/误差）
    float    yaw_big_omega;         // 大 yaw 关节角速度，rad/s
    float    yaw_small_angle;       // 小 yaw 关节角，rad（相对大 yaw；可信、实时）
    float    yaw_small_omega;       // 小 yaw 关节角速度，rad/s
    float    chassis_imu_yaw;       // 底盘 IMU 偏航角，rad（0 ~ 2π）
    float    chassis_imu_omega;     // 底盘 yaw 角速度，rad/s
    uint8_t  mark;                  // 电控递增标志位
    uint8_t  color;                 // 颜色
    uint8_t  auto_aim_switch;       // 电控自瞄开关
    uint8_t  yaw_big_temperature;   // 大 yaw 电机温度，℃
    uint8_t  yaw_small_temperature; // 小 yaw 电机温度，℃
    // MCU2 数据的新样本序号（大 yaw 与底盘 IMU **同源**、同一次取数一起刷新，故共用一个序号）:
    //   MCU1 每从 MCU2 取到一次新数据 +1；两次更新之间重复发送旧值时序号不变。
    //   判定"新样本": 首帧到达 || 序号变化 || 值变化（无"0 = 从未收到"哨兵约定）。
    //   值的年龄由上位机自己的时钟测量（从首次看到该序号起算）+ 传输时延。
    uint8_t  mcu2_seq;              // MCU2 新样本序号（每次取到新数据 +1；值保持时不变）
} TcbsRobotMcuData_C;

// 大 yaw 转子上的 IMU 反馈（可信、高频）
typedef struct TcbsRobotImuData_C {
    uint8_t  valid;              // 是否收到过有效 IMU 帧
    float    gx;                 // IMU 本体系角速度，rad/s
    float    gy;
    float    gz;
    float    ax;                 // IMU 本体系加速度，m/s²
    float    ay;
    float    az;
    double   euler_yaw;          // 世界系欧拉角，rad（ZXY 约定）
    double   euler_pitch;
    double   euler_roll;
    uint32_t dt_one_tenth_ms;    // 本帧间隔，0.1 ms 单位
} TcbsRobotImuData_C;

// 最新一帧原始反馈（MCU 已映射；MCU2 新样本序号在 mcu.mcu2_seq，
// 语义同 TcbsRobotMcuData_C 的注释）
typedef struct TcbsRobotLatestData_C {
    TcbsRobotMcuData_C mcu;   // mcu.valid=0 → 未收到
    TcbsRobotImuData_C imu;   // imu.valid=0 → 未收到
} TcbsRobotLatestData_C;

// 单个数据源的更新情况（年龄/间隔均由**上位机计时**；MCU 侧没有时钟）
typedef struct TcbsRobotSourceInfo_C {
    uint8_t  valid;        // 是否有可用数据（对"值保持"通道 = 已收到过；1 = 可用）
    double   age_s;        // **值年龄**（s，上位机计时）: 从首次看到该新样本起算 + 传输时延；
                           // 值保持期间持续增大；-1 = 从未收到
    uint32_t count;        // 累计收到的帧数（含值被保持的帧）
    uint32_t new_samples;  // 真正的新样本数（序号发生变化）
    uint32_t rejected;     // 收到但值被保持（非新数据）的帧数
    uint8_t  stale;        // 采样年龄超过 stale_age_s（1 = 过旧；估计仍继续用 IMU 速率积分）
} TcbsRobotSourceInfo_C;

// 数据来源（"本次估计所用的数据"）
typedef struct TcbsRobotProvenance_C {
    TcbsRobotSourceInfo_C imu;          // 大 yaw 上的 IMU
    TcbsRobotSourceInfo_C big_enc;      // 大 yaw 编码器（经 MCU/串口，有传输时延）
    TcbsRobotSourceInfo_C small_enc;    // 小 yaw 编码器
    TcbsRobotSourceInfo_C pitch_enc;    // pitch 编码器
    TcbsRobotSourceInfo_C chassis_imu;  // 底盘 IMU（经 MCU，零阶保持）
    uint8_t  big_rate_from_imu;     // 大 yaw 角速度是否来自 IMU（1 = 是）
    uint8_t  reverse_from_trusted;  // 反解是否全由可信量完成（无延迟源参与）
    double   big_enc_delay_used;    // 本帧大 yaw 值的实测年龄（s，上位机计时）
    double   big_enc_innovation;    // 编码器观测 − 预测，rad（诊断用）
    double   big_enc_interval_s;    // 最近两次大 yaw 新样本的间隔，s（反映更新率）
    double   big_enc_sample_age_s;  // 最近一次大 yaw 新样本的实测年龄，s
    uint64_t used_mask;             // 各源使用的位掩码（bit0 IMU, bit1 大yaw编码器,
                                    // bit2 小yaw编码器, bit3 pitch编码器, bit4 底盘IMU）
} TcbsRobotProvenance_C;

// ============================================================================
// 完整状态估计（对应 YawStateEstimator::Estimate，含 1)可信量 2)大yaw延迟补偿
// 3)反解真实位姿 4)模型外生量 + Provenance）
// ============================================================================
typedef struct TcbsRobotEstimate_C {
    uint8_t  valid;                 // 大 yaw 已有绝对基准后为 1

    // ── 1) 可信实时量 ──
    double   imu_yaw;               // 平台世界系偏航角，rad
    double   imu_pitch;             // 平台世界系俯仰角，rad
    double   imu_roll;              // 平台世界系横滚角，rad
    double   platform_azimuth;      // ψ_big: 大 yaw 平台 x 轴世界方位角，rad（解卷绕）
    double   platform_rate;         // ψ̇_big，rad/s
    double   small_joint_angle;     // θ_small，rad（可信）
    double   small_joint_rate;      // θ̇_small，rad/s
    double   pitch_joint_angle;     // θ_pitch，rad（可信）
    double   pitch_joint_rate;      // θ̇_pitch，rad/s

    // ── 2) 大 yaw（延迟编码器 + IMU 速率 → 延迟补偿估计）──
    double   big_joint_angle_meas;  // 原始测量值（滞后），rad
    double   big_joint_angle;       // 延迟补偿后的估计（控制用），rad
    double   big_joint_rate;        // 关节角速度估计，rad/s
    double   big_enc_age;           // 大 yaw 值的年龄，s（上位机计时；-1 = 从未收到）
    double   big_sample_interval;   // 最近两次大 yaw 新样本的间隔，s（上位机计时）
    double   chassis_imu_age;       // 底盘 IMU 值的年龄，s（上位机计时；-1 = 从未收到）
    double   big_enc_innovation;    // 观测残差，rad（诊断/延迟标定）
    uint8_t  big_has_encoder;       // 是否已收到大 yaw 编码器数据

    // ── 3) 反解真实位姿（可信量 + 标定参数，严格）──
    double   head_world_yaw;        // 头部世界系偏航角，rad
    double   head_world_pitch;      // 头部世界系俯仰角，rad
    double   head_world_roll;       // 头部世界系横滚角，rad
    double   small_output_azimuth;  // ψ_small: 小 yaw 输出 x 轴世界方位角，rad（解卷绕）
    double   los_azimuth;           // 视轴（bore）世界方位角，rad
    double   los_elevation;         // 视轴世界俯仰角，rad
    double   chassis_azimuth;       // ψ_chassis = ψ_big − θ_big，rad
    double   chassis_yaw_rate;      // 底盘 yaw 角速度，rad/s

    // ── 4) 模型外生量（平面 8 参模型; 见 mpc/planar_yaw_model.h 的 ModelExo）──
    double   base_omega[3];         // 底盘角速度，关节参考系 C，rad/s（模型用绕关节轴分量）
    // ★ 重力矢量，**A 系（大 yaw 转子系）**，不再随小 yaw 关节角转动（指向下）。
    //   语义变更（v4）: 旧版是 C 系重力；现在供平面模型的 e.gravity_a[0..1] 使用，
    //   取 A 系才能让"r 与 g 在同一旋转系"（绕轴力矩公式对 θ_b 无关）。
    //   底盘水平时其平面分量 ≈ (0, 0) → 模型重力项全为 0。
    double   gravity_a[3];
    double   pitch_acc;             // pitch 关节角加速度估计，rad/s²
                                    //（平面模型不含 pitch 自由度，本字段仅供上层/显示）

    // ── 数据来源 ──
    TcbsRobotProvenance_C prov;
} TcbsRobotEstimate_C;

// ============================================================================
// 二、配置结构（与 C++ 结构体字段一一对应；默认值见 tcbs_robot_controller_default_*）
// ============================================================================

// 单关节约束（对应 dual_yaw::JointLimits）
typedef struct TcbsDualYawJointLimits_C {
    double max_torque;       // N·m，硬限幅
    double max_torque_rate;  // N·m/s，硬约束
    double min_angle;        // rad，机械行程下界（小 yaw 默认 −30°；**允许非对称**）
    double max_angle;        // rad，机械行程上界（小 yaw 默认 +30°）
} TcbsDualYawJointLimits_C;

// 模型参数（对应 dual_yaw::ModelParams — **平面 2 自由度 8 参模型**，字段顺序与 C++ 一致）
//
// 记号（详见 include/mpc/planar_yaw_model.h 顶部推导）:
//   d  = (dx, dy)  小 yaw 轴相对大 yaw 轴的平面偏置（实测几何，不辨识）
//   P  = (Px, Py)  上装一阶矩 m_u·ρ（ρ = 上装质心相对小 yaw 轴的位移）
//   Js             上装绕小 yaw 轴的总惯量（含 m_u|ρ|²）
//   Jbig_eff       大 yaw 侧惯量 = J_big + m_u|d|²
typedef struct TcbsDualYawModelParams_C {
    // ── 实测几何（不辨识；★ 请按实际机械填写）──
    double dx, dy;           // 小 yaw 轴相对大 yaw 轴的平面偏置，m
    double gravity;          // 重力加速度，m/s²
    double m_u_known;        // 可选: 上装质量，kg（**仅**用于倾斜时 m_u·d·g⊥ 项；不称重保持 0）

    // ── ★8 参待辨识（顺序与 dual_yaw::paramsToVector / regressor 列一致）──
    double Jbig_eff;         // [0] 大 yaw 侧惯量（含 m_u|d|²），kg·m²
    double Js;               // [1] 上装绕小 yaw 轴总惯量，kg·m²
    double Px;               // [2] 上装一阶矩 m_u·ρ_x，kg·m
    double Py;               // [3] 上装一阶矩 m_u·ρ_y，kg·m
    double fcBig, fvBig;     // [4][5] 大 yaw 库仑/粘滞摩擦，N·m, N·m·s/rad
    double fcSmall, fvSmall; // [6][7] 小 yaw 库仑/粘滞摩擦

    // ── 固定 / 可选（不辨识）──
    double frictionLambda;   // 摩擦软符号系数 λ（tanh(λ·ω)；默认 10，见 recommendedFrictionLambda）
    double tau_offset_big;   // 可选常数负载（默认 0 = 关闭），N·m
    double tau_offset_small;
} TcbsDualYawModelParams_C;

// MPC 配置（对应 dual_yaw::DualYawMpcConfig，字段顺序与 C++ 一致）
typedef struct TcbsDualYawMpcConfig_C {
    double  dt_control;             // 控制周期，s
    int32_t N;                      // 预测步数（决策变量 = 2N）
    int32_t substeps;               // 每步 RK4 子步
    uint8_t use_rk4;                // 1 = RK4，0 = 半隐式欧拉
    int32_t max_iter;               // Ceres 迭代上限
    double  w_big_azimuth;          // 大 yaw 世界方位角跟踪权重
    double  w_small_azimuth;        // 小 yaw 世界方位角跟踪权重
    double  w_small_center;         // 小 yaw 回中权重
    double  w_small_limit;          // 小 yaw 软限位权重
    double  small_limit_soft_ratio; // 软限位区宽度比例：距任一侧限位 (1−ratio)·总行程 内开始惩罚
    double  r_big_torque;           // 大 yaw 力矩惩罚
    double  r_small_torque;         // 小 yaw 力矩惩罚
    double  rd_big_rate;            // 大 yaw 力矩变化率惩罚
    double  rd_small_rate;          // 小 yaw 力矩变化率惩罚
    double  smooth_eps;             // 位置误差平滑常数 a
    TcbsDualYawJointLimits_C big;       // 大 yaw 约束
    TcbsDualYawJointLimits_C small;     // 小 yaw 约束
    int32_t ref_delay_steps;        // 参考延迟步数（0 = 不延迟）
    // 注意: v4 起**没有** extrapolate_pitch —— 平面模型不含 pitch 自由度，
    //       pitch 只作为下发给电控的目标角，不进入动力学/预测窗。
    // 注意: **没有** small_center_angle 字段（C ABI 布局冻结，不新增字段/不升版本号）：
    //       小 yaw 的回中目标角由 C 侧按行程中心 0.5·(min_angle+max_angle) 派生
    //       （当前行程对称 ±30° ⇒ 0），与 C++ defaultMpcConfig() 的默认值一致；
    //       需要自定中心时请用 C++ 的 DualYawMpc::setConfig()。
} TcbsDualYawMpcConfig_C;

// 状态估计配置（对应 YawStateEstimator::Config，字段顺序与 C++ 一致）
typedef struct TcbsEstimatorConfig_C {
    // IMU 安装位置（**运行时**可切换；一份二进制支持两种构型）:
    //   0 = ON_BIG_YAW: IMU 固定在大 yaw 转子 A 上（备选）→ 用 mount_* 标定
    //   1 = ON_HEAD   : IMU 装在头上（pitch 之后，H 系）→ 用 head_mount_* 标定
    // 切换只改变"反解/重力/关节轴/角速度投影"的分支，所有对外字段语义不变。
    int32_t imu_location;          // 0 = ON_BIG_YAW, 1 = ON_HEAD（其它值按 0 处理）
    double  mount_yaw;             // IMU 安装旋转 R_A_IMU（IMU → A 系），ZXY 欧拉角，rad
    double  mount_pitch;           //   —— 仅 ON_BIG_YAW 使用
    double  mount_roll;
    double  head_mount_yaw;        // IMU 安装旋转 R_H_IMU（IMU → 头 H 系），ZXY 欧拉角，rad
    double  head_mount_pitch;      //   —— 仅 ON_HEAD 使用
    double  head_mount_roll;
    double  transport_delay_s;     // 链路**传输**时延，s（MCU 打包 → 上位机收到；MCU 端无时钟，
                                   // 值的"年龄"另由上位机按新样本序号计时）
    double  big_enc_max_jump;      // 单次测量可修正的最大幅度，rad（抗编码器跳变/坏帧）
    double  stale_age_s;           // 采样年龄超过该值视为"过旧"（见 prov.*.stale），s
    double  chassis_imu_timeout_s; // 底盘 IMU"可用"超时，s（零阶保持，比 stale_age_s 宽松）
    double  max_extrap_s;          // 可信量（小 yaw / pitch 编码器）外推上限，s
    double  rate_lpf_alpha;        // 角速度低通系数
    double  pitch_rate_lpf_alpha;  // pitch 角速度低通系数
    double  pitch_acc_lpf_alpha;   // pitch 角加速度低通系数（0 = 不使用角加速度）
    double  bore[3];               // 视轴方向（head 系单位矢量）
    double  gravity;               // 重力加速度，m/s²
    uint8_t use_chassis_imu;       // 1 = 用底盘 IMU yaw 角速度分离大 yaw 关节角速度
    double  source_timeout_s;      // 数据源超时，s
} TcbsEstimatorConfig_C;

// 编码器/指令线性映射（对应 McuDataPreprocessor::LinearParams）
typedef struct TcbsLinearParams_C {
    double send_pitch_scale;          // 发送: value = scale * value + offset
    double send_pitch_offset;
    double recv_pitch_scale;          // 接收: angle = scale * raw + offset，rad
    double recv_pitch_offset;
    double recv_big_yaw_scale;
    double recv_big_yaw_offset;
    double recv_big_omega_scale;      // 接收: omega = scale * raw_omega，rad/s
    double send_big_yaw_scale;
    double send_big_yaw_offset;
    double send_big_velocity_scale;
    double send_big_torque_scale;     // 发送: torque = scale * torque（N·m → 电控单位）
    double recv_small_yaw_scale;
    double recv_small_yaw_offset;
    double recv_small_omega_scale;
    double send_small_yaw_scale;
    double send_small_yaw_offset;
    double send_small_velocity_scale;
    double send_small_torque_scale;
} TcbsLinearParams_C;

// 控制器配置（对应 McuMpcController::Config）
typedef struct TcbsControllerConfig_C {
    double  loop_period;            // 后台线程周期，s
    uint8_t big_torque_only;        // 1 = 大 yaw 仅力矩（false → 力矩 + 电控位置/速度内环）
    uint8_t small_torque_only;      // 1 = 小 yaw 仅力矩
    int32_t ref_delay_steps;        // 参考延迟步数
    double  integral_gain[2];       // {大 yaw, 小 yaw} 积分增益（0 = 关闭）
    double  integral_limit[2];      // {大 yaw, 小 yaw} 积分限幅
    uint8_t integral_on_big;        // 1 = 允许大 yaw 积分补偿
} TcbsControllerConfig_C;

// ============================================================================
// 三、高层状态结构（RobotController::getState 的 C 版本）
// ============================================================================

// MPC / 控制输出（对应 RobotController::MpcData，序列单独用缓冲区接口取）
typedef struct TcbsRobotMpcData_C {
    double   torque[2];               // {大 yaw, 小 yaw} 实际发送力矩，N·m
    double   torque_mpc[2];           // MPC 解（未加积分补偿），N·m
    double   integral[2];             // 积分补偿量，N·m
    double   target_joint[2];         // 发送给 MCU 的 θ*，rad
    double   target_joint_rate[2];    // 发送给 MCU 的 ω*，rad/s
    uint8_t  big_torque_only;         // 本拍大 yaw 模式位
    uint8_t  small_torque_only;       // 本拍小 yaw 模式位
    double   ref_azimuth[2];          // 本拍使用的大/小 yaw 世界方位角参考，rad
    double   delayed_ref_azimuth[2];  // 延迟缓冲后的参考，rad
    uint8_t  small_ref_over_limit;    // 小 yaw 参考本身超出限位
    double   solve_ms;                // 最近一次 MPC 求解耗时，ms
    double   loop_fps;                // 后台 loop 实测频率，Hz
    uint64_t ticks_since_set;         // 距上次 set 的拍数
    uint32_t solve_count;             // 累计求解次数
    uint32_t solve_fail_count;        // 累计求解失败次数
    uint8_t  estimator_valid;         // 本拍状态估计是否有效
    uint8_t  sent_ok;                 // 最近一次发送是否成功
    // 序列长度（内容由 tcbs_robot_controller_get_*_sequence 取；数组下标 0 = 大 yaw, 1 = 小 yaw）
    int32_t  ref_azimuth_seq_len[2];  // 参考世界方位角序列长度
    int32_t  pred_azimuth_seq_len[2]; // 预测世界方位角序列长度
    int32_t  pred_joint_seq_len[2];   // 预测关节角序列长度
} TcbsRobotMpcData_C;

// 完整状态（四组: mcu / imu / est / mpc）
typedef struct TcbsRobotControllerState_C {
    TcbsRobotMcuData_C  mcu;   // MCU 原始反馈（已映射）
    TcbsRobotImuData_C  imu;   // 大 yaw 上 IMU 原始数据
    TcbsRobotEstimate_C est;   // 状态估计（可信量 + 大 yaw 补偿 + 反解位姿 + 来源）
    TcbsRobotMpcData_C  mpc;   // 控制输出 / 参考 / 预测长度 / 性能
} TcbsRobotControllerState_C;

// ============================================================================
// 四、布局自检
// ============================================================================
typedef struct TcbsRobotCommAbiInfo_C {
    uint32_t api_version;              // = TCBS_ROBOT_COMM_C_API_VERSION
    uint32_t sizeof_pointer;           // 指针宽度（8 = 64 位）
    uint32_t sizeof_latest_data;       // sizeof(TcbsRobotLatestData_C)
    uint32_t sizeof_estimate;          // sizeof(TcbsRobotEstimate_C)
    uint32_t sizeof_source_info;       // sizeof(TcbsRobotSourceInfo_C)
    uint32_t sizeof_provenance;        // sizeof(TcbsRobotProvenance_C)
    uint32_t sizeof_controller_state;  // sizeof(TcbsRobotControllerState_C)
    uint32_t sizeof_mpc_data;          // sizeof(TcbsRobotMpcData_C)
    uint32_t sizeof_model_params;      // sizeof(TcbsDualYawModelParams_C)
    uint32_t sizeof_mpc_config;        // sizeof(TcbsDualYawMpcConfig_C)
    uint32_t sizeof_estimator_config;  // sizeof(TcbsEstimatorConfig_C)
    uint32_t sizeof_linear_params;     // sizeof(TcbsLinearParams_C)
    uint32_t sizeof_controller_config; // sizeof(TcbsControllerConfig_C)
} TcbsRobotCommAbiInfo_C;

// 填充 ABI 信息（供 ctypes 侧校验结构体布局）；成功返回 TCBS_ROBOT_COMM_OK
int tcbs_robot_comm_abi_info(TcbsRobotCommAbiInfo_C* out);

// ABI 自检（**调用方主动校验**，推荐在加载库之后、创建句柄之前调用一次）:
//   expected_version: 调用方编译时用的 TCBS_ROBOT_COMM_C_API_VERSION
//   expected_sizes  : 调用方按自己的头文件/绑定算出的各结构体 sizeof（字段同 TcbsRobotCommAbiInfo_C，
//                     但只需填 sizeof_* 字段；api_version/sizeof_pointer 会被忽略）
//   expected_sizes 为 NULL 时只校验版本号。
// 返回: TCBS_ROBOT_COMM_OK；版本号不一致或任一 sizeof 不一致 → TCBS_ROBOT_COMM_ERR_ABI_MISMATCH
//       （不一致的项会打印到 stderr，便于定位是哪个结构体漂了）。
int tcbs_robot_comm_check_abi(uint32_t expected_version, const TcbsRobotCommAbiInfo_C* expected_sizes);

// ============================================================================
// 五、低层接口 — RobotCommunication（MCU/IMU 串口 + 映射 + 状态估计）
// ============================================================================

// 创建低层句柄（使用默认编码器映射与默认估计器配置；无硬件时也能成功）
// 失败返回 NULL（例如线程/内存分配失败）。
TcbsRobotCommHandle* tcbs_robot_comm_create(void);

// 销毁句柄（内部先 stop 再释放；NULL 安全）
void tcbs_robot_comm_destroy(TcbsRobotCommHandle* handle);

// 停止串口后台线程（保留句柄；之后仍可 get_latest_data/get_estimate，只是不再更新）
void tcbs_robot_comm_stop(TcbsRobotCommHandle* handle);

// 取最新一帧原始反馈（MCU 已按映射换算）；out 必须非空
int tcbs_robot_comm_get_latest_data(TcbsRobotCommHandle* handle, TcbsRobotLatestData_C* out);

// 取完整状态估计（可信量 + 大 yaw 延迟补偿 + 反解真实位姿 + 外生量 + 数据来源）
int tcbs_robot_comm_get_estimate(TcbsRobotCommHandle* handle, TcbsRobotEstimate_C* out);

// 发送 MCU 指令包（内部按映射参数预处理 + 计算 CRC；返回 1 = 发送成功，0 = 失败）
// yaw_*_mode: 0 = 仅力矩，1 = 力矩 + 电控位置/速度内环（见 mcu::YawMode）
int tcbs_robot_comm_send_to_mcu(TcbsRobotCommHandle* handle,
                           uint8_t auto_aim_enable,          // 自瞄总开关
                           uint8_t fire,                     // 火控
                           float   pitch_target_angle,       // pitch 目标角（映射前语义）
                           uint8_t yaw_big_mode,             // 大 yaw 模式位
                           double  yaw_big_target_angle,     // 大 yaw 关节目标角，rad（多圈连续）
                           float   yaw_big_target_velocity,  // 大 yaw 目标角速度，rad/s
                           float   yaw_big_torque,           // 大 yaw 力矩，N·m
                           uint8_t yaw_small_mode,           // 小 yaw 模式位
                           float   yaw_small_target_angle,   // 小 yaw 关节目标角，rad（行程 ±30°，由电控侧再限位）
                           float   yaw_small_target_velocity,// 小 yaw 目标角速度，rad/s
                           float   yaw_small_torque);        // 小 yaw 力矩，N·m

// 发送 IMU 心跳帧（无载荷）；返回 1 = 发送成功，0 = 失败
int tcbs_robot_comm_send_to_imu(TcbsRobotCommHandle* handle);

// ============================================================================
// 六、高层接口 — RobotController（估计 + 耦合 MPC + 后台发送线程）
// ============================================================================

// 创建高层句柄（等价于 C++ 侧 RobotController(RobotController::Config{}) 全默认参数）
//   - 会启动后台 MPC 线程与两条串口线程；**无硬件时同样成功返回**（串口后台重连）
//   - 失败返回 NULL
TcbsRobotController_C* tcbs_robot_controller_create(void);

// 销毁句柄（析构 RobotController: 停后台线程 + 关串口；NULL 安全）
void tcbs_robot_controller_destroy(TcbsRobotController_C* handle);

// 当前模式: 0 = SINGLE, 1 = SEQUENCE；句柄为 NULL 返回 TCBS_ROBOT_COMM_ERR_NULL_HANDLE
int tcbs_robot_controller_mode(TcbsRobotController_C* handle);

// ── 默认配置取值（与 C++ 默认值一致，便于调用方"改一项、其余用默认"）──
int tcbs_robot_controller_default_model_params(TcbsDualYawModelParams_C* out);
int tcbs_robot_controller_default_mpc_config(TcbsDualYawMpcConfig_C* out);
int tcbs_robot_controller_default_estimator_config(TcbsEstimatorConfig_C* out);
int tcbs_robot_controller_default_linear_params(TcbsLinearParams_C* out);
int tcbs_robot_controller_default_controller_config(TcbsControllerConfig_C* out);

// ── 配置读取（与 setter 对称: 读的是句柄内**累积配置**，即默认值与所有 setter 的结果）──
int tcbs_robot_controller_get_model_params(TcbsRobotController_C* handle, TcbsDualYawModelParams_C* out);
int tcbs_robot_controller_get_mpc_config(TcbsRobotController_C* handle, TcbsDualYawMpcConfig_C* out);
int tcbs_robot_controller_get_estimator_config(TcbsRobotController_C* handle, TcbsEstimatorConfig_C* out);

// ── 配置设置（各一个 setter，字段与 C++ 结构体一一对应）──
// 模型参数: 转发 RobotController::setModelParams（同时复位 MPC 热启动）
int tcbs_robot_controller_set_model_params(TcbsRobotController_C* handle, const TcbsDualYawModelParams_C* params);
// MPC 配置: 转发 RobotController::setMpcConfig
int tcbs_robot_controller_set_mpc_config(TcbsRobotController_C* handle, const TcbsDualYawMpcConfig_C* config);
// 估计器配置: 转发 RobotController::setEstimatorConfig（在线标定）
int tcbs_robot_controller_set_estimator_config(TcbsRobotController_C* handle, const TcbsEstimatorConfig_C* config);
// 编码器映射: 转发 RobotController::setLinearParams
int tcbs_robot_controller_set_linear_params(TcbsRobotController_C* handle, const TcbsLinearParams_C* params);
// 控制器配置: RobotController/McuMpcController **没有**运行时 setter，本函数通过
// "用累积配置重建 RobotController" 生效 → 副作用: 后台线程重启、MPC 热启动/积分状态清零、
// 目标保持上次 set 的值（不保留）。请在启动控制前一次性调用。
int tcbs_robot_controller_set_controller_config(TcbsRobotController_C* handle, const TcbsControllerConfig_C* config);

// ── 目标设置（世界方位角语义；参数与 RobotController::set 一一对应）──
// big_yaw_azimuth / small_yaw_azimuth: 世界系方位角，rad，多圈连续（解卷绕）
// 返回 TCBS_ROBOT_COMM_ERR_MODE_MISMATCH 表示当前为 SEQUENCE 模式（应改用 set_sequence）
int tcbs_robot_controller_set(TcbsRobotController_C* handle,
                         uint8_t auto_aim_enable,        // 自瞄总开关
                         uint8_t big_torque_only,        // 大 yaw 仅力矩
                         uint8_t small_torque_only,      // 小 yaw 仅力矩
                         double  big_yaw_azimuth,        // 大 yaw 世界方位角，rad
                         double  small_yaw_azimuth,      // 小 yaw 世界方位角，rad
                         double  pitch_target_angle,     // pitch 目标角，rad
                         uint8_t fire,                   // 火控
                         uint8_t integral_enable);       // 是否启用积分补偿

// 关节系便捷接口（内部按当前底盘方位角估计换算为世界方位角）
// big_joint_angle: 大 yaw 关节角（相对底盘，多圈）；small_joint_angle: 小 yaw 关节角（行程 ±30°）
int tcbs_robot_controller_set_joint_angles(TcbsRobotController_C* handle,
                                      uint8_t auto_aim_enable,
                                      uint8_t big_torque_only,
                                      uint8_t small_torque_only,
                                      double  big_joint_angle,      // rad
                                      double  small_joint_angle,    // rad
                                      double  pitch_target_angle,   // rad
                                      uint8_t fire,
                                      uint8_t integral_enable);

// 序列设置（各通道独立，不截断；长度 <= 0 或指针为 NULL 的通道保持当前值）
// 注意: 现有 C++ 实现下大概率返回 TCBS_ROBOT_COMM_ERR_MODE_MISMATCH（见文件头说明）
int tcbs_robot_controller_set_sequence(TcbsRobotController_C* handle,
                                  uint8_t auto_aim_enable,
                                  uint8_t big_torque_only,
                                  uint8_t small_torque_only,
                                  const double*  big_yaw_azimuth_seq, int32_t big_len,
                                  const double*  small_yaw_azimuth_seq, int32_t small_len,
                                  const double*  pitch_seq, int32_t pitch_len,
                                  const uint8_t* fire_seq, int32_t fire_len,
                                  uint8_t integral_enable);

// 序列设置（不检查模式位）: 直接调用 McuMpcController::set(序列)，绕过
// RobotController 的 SINGLE/SEQUENCE 模式保护。语义与 SEQUENCE 模式一致
// （后台 loop 每拍消费一个元素），但 tcbs_robot_controller_mode() 仍会报告 SINGLE。
// 仅在明确知道自己需要绕过模式保护时使用。
int tcbs_robot_controller_set_sequence_unchecked(TcbsRobotController_C* handle,
                                            uint8_t auto_aim_enable,
                                            uint8_t big_torque_only,
                                            uint8_t small_torque_only,
                                            const double*  big_yaw_azimuth_seq, int32_t big_len,
                                            const double*  small_yaw_azimuth_seq, int32_t small_len,
                                            const double*  pitch_seq, int32_t pitch_len,
                                            const uint8_t* fire_seq, int32_t fire_len,
                                            uint8_t integral_enable);

// ── 状态获取 ──
// 取完整状态（含 mcu/imu/est/mpc 四组；序列长度在 out->mpc.*_seq_len 中，
// 序列内容用下面的 get_*_sequence 取）
int tcbs_robot_controller_get_state(TcbsRobotController_C* handle, TcbsRobotControllerState_C* out);

// 取参考世界方位角序列（which: 0 = 大 yaw, 1 = 小 yaw）
// 返回序列实际长度（>= 0，可能大于 max_len 表示被截断）；out 为 NULL 或 max_len <= 0
// 时只查询长度；参数非法或句柄为空返回负错误码。
int tcbs_robot_controller_get_ref_sequence(TcbsRobotController_C* handle, int32_t which,
                                      double* out, int32_t max_len);

// 取预测世界方位角序列（which: 0 = 大 yaw, 1 = 小 yaw）
int tcbs_robot_controller_get_pred_sequence(TcbsRobotController_C* handle, int32_t which,
                                       double* out, int32_t max_len);

// 取预测关节角序列（which: 0 = 大 yaw, 1 = 小 yaw）
int tcbs_robot_controller_get_pred_joint_sequence(TcbsRobotController_C* handle, int32_t which,
                                             double* out, int32_t max_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TCBS_ROBOT_COMMUNICATION_C_H
