#ifndef TCBS_YAW_STATE_ESTIMATOR_H
#define TCBS_YAW_STATE_ESTIMATOR_H

#include <cstdint>
#include <mutex>

#include "tcbs/common/RotationUtils.h"
#include "tcbs/common/StrictPose.h"

namespace tcbs {

// ============================================================================
// YawStateEstimator — 双级 yaw 云台状态估计（IMU 安装位置可配置: 大 yaw 转子 A / 头 H）
//
// 传感器可信度约定（按实际系统给定）:
//   - IMU（装在 A 系上，或装在头上——由 Config::imu_location 选择）: **可信、实时**。
//     给出其所在刚体的世界姿态与角速度 → 大 yaw 角度的"高频真值分量"
//   - 小 yaw 编码器、pitch 编码器: **可信、实时**（直接当关节角真值使用）
//   - 大 yaw 编码器（经 MCU1↔MCU2，约 10Hz 量级、间隔不规则）: 两次更新之间电控重复
//     发送旧值（值被保持），到达值本身可视为准确但**延迟不确定** → 靠**新样本序号**
//     识别新样本（序号变化即新样本），年龄由**上位机自己的时钟**测量，再用 IMU
//     角速度做**一阶（速度）外推**补齐
//   - 底盘 IMU（同样经该链路，与大 yaw 编码器**共用同一个新样本序号**）: 延迟 + 值保持
//     与上一条相同 ⇒ 同样做**一阶（速度）延时补偿**（底盘没有第二个实时传感器，只能用
//     被保持的底盘角速度外推: ψ_c(T) ≈ ψ_c(样本) + ω_c(样本)·年龄；外推时间上限 = stale_age_s）。
//     ★ 这一步是必须的: θ_p = ψ_platform − ψ_chassis，而 ψ_platform 来自**实时** IMU；
//     底盘转动时若直接拿被保持的 ψ_chassis 相减，θ_p 会有 ω_c·年龄 的误差（最长一个保持周期），
//     它同时污染 MPC 的关节初值、背隙 Δ 的在线估计与反解底盘方位角。
//     其年龄同样由上位机计时
//   - **不使用电控侧时钟**（MCU 端计时不可用）；判定新样本只用: 首帧 / 序号变化 / 值变化，
//     不做"可用性"检测（上电第一帧即视为一次更新）
//
// 关键运动学（两 yaw 轴平行、非共轴）—— **IMU 安装位置可运行时切换（一份二进制两种构型）**:
//   记号: R_X = 把 X 系向量转到世界系的旋转；关节轴恒为 A/B/C 系的 z；pitch 轴 = B 系 x；H = 头
//     R_world_B = R_world_A · Rz(θ_s)
//     R_world_H = R_world_B · Rx(θ_p)              （H 相对 B 只绕 x 转 pitch）
//
//   ★ 构型 **ON_HEAD（现状/默认，IMU 固定在头上、pitch 之后；见 Config::imu_location）**
//     —— 备选构型 ON_BIG_YAW（IMU 固定在大 yaw 转子 A 系上）仍然支持，切 `imu_location` 即可
//     R_world_imu = R_world_A · R_mount            （R_mount = R_A_IMU，由 mount_yaw/pitch/roll 给出）
//     反解: R_world_A = R_world_imu · R_mountᵀ
//           R_world_H = R_world_A · Rz(θ_s) · Rx(θ_p)
//     重力(A 系): g_A = R_mount · g_imu ,  g_imu = R_world_imuᵀ·(0,0,−g)     → 不需要 θ_b
//     关节轴在 IMU 系: a_imu = R_mountᵀ·ẑ（常量）
//     θ̇_b = gyro·a_imu − ω_chassis
//           （陀螺投影 = 大 yaw 平台的世界角速度 = ω_chassis + θ̇_b）
//
//   ★ 构型 ON_HEAD（新增，IMU 装在头上＝pitch 之后）
//     R_world_imu = R_world_H · R_mount_head       （R_mount_head = R_H_IMU，由 head_mount_* 给出）
//     反解: R_world_H = R_world_imu · R_mount_headᵀ （**头姿态由 IMU 直接给出**）
//           R_world_A = R_world_H · (Rz(θ_s)·Rx(θ_p))ᵀ = R_world_H·Rx(θ_p)ᵀ·Rz(θ_s)ᵀ
//                       （用可信编码器 θ_s、θ_p 反推大 yaw 平台）
//     小 yaw 输出方位角 = **头 x 轴的世界方位角**（Rx(p)·x̂ = x̂ ⇒ pitch 不影响它）
//     重力(A 系): g_H = R_mount_head·g_imu ,  g_A = Rz(θ_s)·Rx(θ_p)·g_H       → 不需要 θ_b
//     关节轴在 IMU 系: a_imu(p) = R_mount_headᵀ·Rx(θ_p)ᵀ·ẑ（**随 pitch 变化**）
//     θ̇_b = gyro·a_imu(p) − ω_chassis − θ̇_s
//           （陀螺投影 = **头**的世界角速度在关节轴上的分量 = ω_chassis + θ̇_b + θ̇_s；
//             俯仰角速度 ṗ 与关节轴垂直，不进入投影）
//
//   ⚠ ON_HEAD 的固有代价（两种构型的**本质差异**，务必知悉）:
//     大 yaw 关节角速度估计里**多了一项 θ̇_s**（小 yaw 编码器角速度）⇒ 该项的一阶低通滞后
//     （α=0.35 @100Hz ⇒ τ≈19ms，实测误差 ≈0.34 rad/s）会经"延迟补偿 + 两帧间外推"传入
//     big_joint_angle / big_joint_rate: 实测关节角速度误差比 ON_BIG_YAW 大 20 倍
//     （0.50 vs 0.025 rad/s），关节角误差大 2.5 倍（0.029 vs 0.012 rad）；
//     在"值保持"链路下更放大为 ≈0.12 rad（≈0.34 rad/s × 最长保持 0.33s）。
//     base_omega 也会多出 θ̇_s·ẑ_A + ṗ·x̂ 的残余（见 recompute 内注释）。
//     但**真正用于瞄准的世界方位角**（platform_azimuth / small_output_azimuth /
//     head_world_* / LOS）在两种构型下都只由 IMU + 可信编码器严格反解，不受该噪声影响
//     （实测仍为 mrad 量级；重力同理，不依赖 θ_b 与延迟链路）。
//     **ON_BIG_YAW 在『大 yaw 关节角/角速度』这一项上精度仍是最优**；ON_HEAD 的代价是
//     θ̇_big 要用「平台角速度 − 编码器小 yaw 速率」算（多一次差分+LPF ⇒ 该通道噪声更大），
//     换来的是 IMU 离开大 yaw 转子的机械/走线便利（且 pitch 可直接由 IMU 读出 ⇒ pitch 标定可行）。
//
// 输出分四组:
//   1) Trusted: IMU 欧拉角/平台方位角/平台角速度 + 小 yaw/pitch 关节角与角速度
//   2) BigYaw : 大 yaw 编码器原始值 + 延迟补偿后的关节角/角速度估计 + 诊断量
//   3) Truth  : 用可信值+标定参数反解的真实位姿（head 欧拉角、LOS 方位/俯仰）
//   4) Exo    : 供 MPC 使用的模型外生量（底盘角速度、重力方向、pitch 角加速度）
//   另附 Provenance: 每个数据源的时延/有效性/是否被使用（“所用数据”）
// ============================================================================
class YawStateEstimator {
public:
    // ── 标定/配置参数 ──
    struct Config {
        // ── IMU 安装位置（**运行时**可切换；一份二进制支持两种构型）──
        //   ON_HEAD   : IMU 装在头上（pitch 之后，H 系）→ 用 head_mount_* 标定【默认/现状】
        //   ON_BIG_YAW: IMU 固定在大 yaw 转子 A 上（备选）→ 用 mount_* 标定
        // 说明: 切换构型只改变"反解/重力/关节轴/角速度投影"的分支，**所有对外字段
        //       的语义不变**（见 Estimate 各字段注释）。
        enum class ImuLocation { ON_BIG_YAW = 0, ON_HEAD = 1 };
        ImuLocation imu_location = ImuLocation::ON_HEAD;

        // IMU 安装旋转 R_A_IMU（IMU 系 → 大 yaw 转子 A 系），ZXY 欧拉角
        // ——仅 ON_BIG_YAW（备选构型）使用
        double mount_yaw   = 0.0;
        double mount_pitch = 0.0;
        double mount_roll  = 0.0;

        // IMU 安装旋转 R_H_IMU（IMU 系 → 头 H 系），ZXY 欧拉角
        // ——仅 ON_HEAD 使用（H 相对 B 只绕 x 转 pitch，故 H 系 = 欠 pitch 的 B 系）
        double head_mount_yaw   = 0.0;
        double head_mount_pitch = 0.0;
        double head_mount_roll  = 0.0;

        // 链路传输时延（s）: 从"值被 MCU1 打包"到上位机收到的时延（串口+转发+调度）。
        // 注意: MCU 侧不提供时钟，因此这里只标**传输**时延（很小，十几 ms 量级）；
        // 值的**年龄**由上位机计时（从首次看到该新样本序号起算）后上报。
        double transport_delay_s = 0.0;
        // 单次测量可修正的最大幅度（rad）: 抗编码器跳变/坏帧
        double big_enc_max_jump = 0.30;
        // 大于该采样年龄视为"过旧"（在 Provenance.stale 中反映；估计仍继续用 IMU 速率积分）
        double stale_age_s = 0.30;
        // 底盘 IMU 值即使较旧也**继续使用**（零阶保持；比当作"底盘不转"好得多，
        // 且底盘角速度本身变化缓慢），因此其"可用"超时取得比 stale_age_s 宽松得多
        double chassis_imu_timeout_s = 1.0;
        // 可信量（小 yaw / pitch 编码器）外推上限（s）: 防 MCU 停流时发散
        double max_extrap_s = 0.30;
        // ── 角速度低通系数（**分轴**，两者来源不同，噪声特性也不同）──
        //   小 yaw: 来源 = MCU 每帧发来的 `yaw_small_omega`（不差分）；
        //   大 yaw: 来源 = IMU 陀螺在关节轴上的投影 − 底盘 ω（ON_HEAD 再 − θ̇_s）。
        // α = 1.0 ⇒ **直通（不做任何滤波）**；α 越小越平滑、滞后越大。
        // ★ 要"角速度完全取自 MCU、不滤波"就是 `small_rate_lpf_alpha = 1.0`
        //   （大 yaw 那个是陀螺投影，与本项无关）。
        double small_rate_lpf_alpha = 0.35;   // 小 yaw 关节角速度
        double big_rate_lpf_alpha   = 0.35;   // 大 yaw 平台/关节角速度（IMU 支路的高频低通）
        // ── 大 yaw **电机侧**角速度的低通 ──
        //   来源 = MCU 的 `yaw_big_omega`（电控按**编码器**算出的电机角速度）。
        //   它只用于"电机侧"状态 θ̇_m；**不再**用它去修正任何云台侧的量
        //   （曾经加过"用编码器角速度校正 IMU 支路直流"的互补滤波/偏置校正，已按用户
        //    要求**完全删除** —— 编码器量的物理含义是电机侧，与云台侧之间隔着背隙，
        //    拿它修正云台角速度在原理上就是错的）。
        double big_motor_rate_tau_s = 0.30;   // 低通时间常数 (s)，按 MCU 新样本间隔换算
        double big_motor_rate_alpha = 0.25;   // 拿不到采样间隔时的兜底系数
        // ── 背隙中心（β）在线估计的遗忘时间常数（s）──
        //   越短越跟得上 IMU 漂移，但会被"单侧贴住"的运动带偏；
        //   3 s 是"包含一次换向"的折中。≤0 关闭估计（β 恒 0）。
        double backlash_center_tau_s = 3.0;
        double pitch_rate_lpf_alpha = 0.25;
        // pitch 角加速度估计低通（0 = 不使用角加速度，置 0）
        double pitch_acc_lpf_alpha = 0.15;

        // 视轴方向（head 系单位矢量）。
        // ★ 本工程坐标系约定（与父工程 UnifiedAutoAimPipeline 一致）:
        //     x = 右, y = 前, z = 上；yaw 绕 z（从上方看逆时针, x→y）；pitch 绕 x（+ = 抬头, y→z）。
        //   因此 pitch 轴就是 head 系的 x 轴 ⇒ **光轴/枪管在 y-z 平面内、默认沿 +y**。
        //   若实际视轴与默认方向不一致（装配偏角），按 docs/calibration.md §3.4 标定后填入。
        double bore[3] = {0.0, 1.0, 0.0};
        double gravity = 9.81;

        // 是否用底盘 IMU 的 yaw 角速度分离"大 yaw 关节角速度"
        //   ON_BIG_YAW: θ̇_big = ω_platform·ẑ − ω_chassis,z
        //   ON_HEAD   : θ̇_big = ω_head·ẑ − ω_chassis,z − θ̇_small（多减一项小 yaw 编码器角速度）
        bool   use_chassis_imu = true;

        // 数据源超时（s）: 超过该时间未更新则标记为无效
        double source_timeout_s = 0.5;
    };

    // ── 数据来源信息（“所用数据”上报）──
    struct SourceInfo {
        bool   valid = false;      // 是否有可用数据（对"值保持"通道=已收到过）
        double age_s = -1.0;       // **值年龄**（s，上位机计时）: 从上位机首次看到该
                                   // 新样本算到现在 + 传输时延；值保持期间持续增大；
                                   // -1 = 从未收到
        uint32_t count = 0;        // 收到的帧数
        uint32_t new_samples = 0;  // 真正的新样本数（时刻戳变化）
        uint32_t rejected = 0;     // 收到但值被保持（非新数据）的帧数
        bool   stale = false;      // 采样年龄超过 stale_age_s
    };

    struct Provenance {
        SourceInfo imu;            // IMU（大yaw 上）
        SourceInfo big_enc;        // 大 yaw 编码器（经 MCU，有延迟）
        SourceInfo small_enc;      // 小 yaw 编码器
        SourceInfo pitch_enc;      // pitch 编码器
        SourceInfo chassis_imu;    // 底盘 IMU（经 MCU）

        bool big_rate_from_imu = false;    // 大 yaw 角速度的高频是否来自 IMU
        bool big_rate_from_encoder = false;  // （已废弃：编码器不再参与云台角速度；恒 false，仅为 ABI 兼容保留）
        bool reverse_from_trusted = false; // 反解是否全部由可信量完成（无延迟源参与）
        double big_enc_delay_used = 0.0;   // 本帧大 yaw 值的实测年龄（s）
        double big_enc_innovation = 0.0;   // 编码器观测 − 预测（rad），诊断用
        double big_enc_interval_s = 0.0;   // 最近两次大 yaw 新样本的间隔（s），反映更新率
        double big_enc_sample_age_s = 0.0; // 最近一次大 yaw 新样本的实测年龄（s）
        uint64_t used_mask = 0;            // 各源使用的位掩码（见 UsedBit）
    };

    // 使用位掩码
    enum UsedBit : uint64_t {
        USED_IMU        = 1ull << 0,
        USED_BIG_ENC    = 1ull << 1,
        USED_SMALL_ENC  = 1ull << 2,
        USED_PITCH_ENC  = 1ull << 3,
        USED_CHASSIS_IMU= 1ull << 4,
    };

    struct Estimate {
        bool valid = false;        // 大 yaw 有绝对基准（编码器或 IMU+编码器）后为 true

        // ── 1) 可信实时量 ──
        double imu_yaw = 0.0, imu_pitch = 0.0, imu_roll = 0.0;   // IMU 自身姿态（世界系，ZXY）
        double platform_azimuth = 0.0;   // ψ_big: 大yaw平台 x 轴世界方位角（解卷绕）
                                         //   ON_BIG_YAW: 由 IMU 直接给出；
                                         //   ON_HEAD: 由 头IMU 减去可信 θ_s/θ_p 反推
        double platform_rate = 0.0;      // ψ̇_big (rad/s)，陀螺在关节轴上投影
                                         //   （ON_HEAD 下已扣除 θ̇_small，见头文件顶部注释）

        double small_joint_angle = 0.0;  // θ_small（可信）
        double small_joint_rate = 0.0;   // θ̇_small
        double pitch_joint_angle = 0.0;  // θ_pitch（可信）
        double pitch_joint_rate = 0.0;   // θ̇_pitch

        // ── 2) 大 yaw（延迟/带误差编码器 + IMU 速率 → 延迟补偿估计）──
        double big_joint_angle_meas = 0.0;  // 原始测量（滞后）
        double big_joint_angle = 0.0;       // 延迟补偿后的估计（控制用）
        double big_joint_rate = 0.0;        // 关节角速度估计（= 云台侧，见下）
        // ── ★ 大 yaw 电机侧 / 云台侧 显式分离 ──
        //   `big_joint_angle` / `_meas` 是**电机侧**（MCU 编码器；带链路延迟与保持）；
        //   `big_joint_rate` 是**云台侧**关节角速度（IMU 陀螺投影）—— 两者不同源，
        //   差异就是传动形变 Δ（背隙）。旧字段名保留以兼容，新代码请用下面四个。
        double big_motor_angle = 0.0;       // 电机侧关节角（= big_joint_angle）
        double big_motor_rate = 0.0;        // 电机侧角速度（MCU 编码器，低通后）
        double big_platform_angle = 0.0;    // 云台侧关节角 θ_p = platform_azimuth − ψ_chassis
                                            //   （ψ_chassis 已做一阶延时补偿，见文件头）
        double big_platform_rate = 0.0;     // 云台侧角速度（= big_joint_rate）
        // ── ★ 背隙中心的**在线**估计（β）──
        //   Δ_raw = θ_motor − θ_platform；死区中心随电机/云台共同旋转而移动、
        //   且云台角由 IMU 推出会漂移 ⇒ **只有宽度 δ 是静态标定量**，中心必须在用中确定。
        //   做法: 对 Δ_raw 维护带遗忘的滑动 min/max（τ≈window τ），
        //         center = (max+min)/2,  width_obs = max−min。
        //   模型里用 Δ = θ_motor − (θ_platform + β)，β = −center ⇒ 死区关于 0 对称。
        double backlash_center = 0.0;       // β（rad）—— 加在云台角上
        double backlash_width_obs = 0.0;    // 观测到的 Δ_raw 极差（≈ δ；诊断用）
        double big_enc_age = -1.0;          // ★ 大 yaw 值的年龄（上位机计时，s）
        double big_sample_interval = 0.0;   // 最近两次新样本间隔（s，上位机计时）
        double chassis_imu_age = -1.0;      // 底盘 IMU 值的年龄（上位机计时，s）
        double big_enc_innovation = 0.0;    // 观测残差（诊断/延迟标定）
        bool   big_has_encoder = false;

        // ── 3) 反解真实位姿（可信量 + 标定参数，严格；两构型同义）──
        //   ON_BIG_YAW: R_world_A = R_world_imu·R_mountᵀ ; R_world_H = R_world_A·Rz(θ_s)·Rx(θ_p)
        //   ON_HEAD   : R_world_H = R_world_imu·R_mount_headᵀ ; R_world_A = R_world_H·Rx(θ_p)ᵀ·Rz(θ_s)ᵀ
        double head_world_yaw = 0.0, head_world_pitch = 0.0, head_world_roll = 0.0;
        double small_output_azimuth = 0.0;  // ψ_small: 小yaw输出 x 轴世界方位角（解卷绕）
                                            //   = B 系 x 轴方位角 = 头 x 轴方位角（Rx(p)x̂=x̂）
        double los_azimuth = 0.0;           // 视轴（bore）世界方位角
        double los_elevation = 0.0;         // 视轴世界俯仰角
        double chassis_azimuth = 0.0;       // ψ_chassis = ψ_big − θ_big（估计；已做一阶延时补偿）
        double chassis_yaw_rate = 0.0;      // 底盘 yaw 角速度（底盘 IMU，零阶保持，rad/s）

        // ── 4) 模型外生量 ──
        double base_omega[3] = {0.0, 0.0, 0.0};   // 底盘角速度，关节参考系 C
        // 重力矢量，**A 系（大 yaw 转子系）**，指向下。
        // 用 A 系而非 C 系: ①绕轴重力力矩需要 r 与 g 在同一旋转系，取 A 系时对 θ_b 无关；
        // ②两种构型下 A 系重力都**不需要 θ_b**:
        //    ON_BIG_YAW: g_A = R_A_IMU·g_imu
        //    ON_HEAD   : g_A = Rz(θ_s)·Rx(θ_p)·R_H_IMU·g_imu（只用可信编码器 θ_s/θ_p）
        // ★ 严格反解数据包: 以 IMU 为准确值反解底盘姿态，并把反解用到的**全部数据**
        //   打包（IMU 欧拉角、θ_b/θ_s/θ_p、安装矩阵参数、构型），使外部可只凭这一包
        //   复算整车姿态；忽略浮点误差时重构出的 IMU 姿态严格等于 IMU 实际数据
        //   （`strict_pose.recon_err_rot` 就是该校核残差，量级 ~1e-16）。
        dual_yaw::StrictPose strict_pose;

        double gravity_a[3] = {0.0, 0.0, -9.81};
        double pitch_acc = 0.0;                   // pitch 关节角加速度估计

        Provenance prov;
    };

    static Config defaultConfig() { return Config{}; }
    explicit YawStateEstimator(const Config& cfg = defaultConfig());

    // 高频路径: 每个 IMU 包调用（IMU 位置由 Config::imu_location 决定）
    void onImu(double euler_yaw, double euler_pitch, double euler_roll,
               double gx, double gy, double gz);

    // 低频路径: 每个 MCU 包调用（角度已由 McuDataPreprocessor 映射）
    // mcu2_seq: MCU2 数据的**新样本序号**（大 yaw 与底盘 IMU 同源共用；值保持时不变）。
    // 新样本判定 = 首帧到达 或 序号变化 或 值变化（后两者任一成立即更新）；
    // 之后用上位机自身时钟计时该值的年龄。不需要任何"可用性/从未收到"约定。
    void onMcu(double yaw_big_angle, double yaw_big_omega,
               double yaw_small_angle, double yaw_small_omega,
               double pitch_angle,
               double chassis_imu_yaw, double chassis_imu_omega,
               uint8_t mcu2_seq = 0);

    // 读取估计结果（线程安全）
    Estimate estimate() const;

    // 更新配置（标定参数在线修改；线程安全）
    void setConfig(const Config& cfg);
    Config config() const;

    // 复位（重上电/重连）
    void reset();

    // 当前 IMU 安装矩阵（IMU 系 → A 系，仅 ON_BIG_YAW 有意义）
    const rot::Mat3& mountRotation() const { return R_A_IMU_; }
    // 当前头安装矩阵（IMU 系 → H 系，仅 ON_HEAD 有意义）
    const rot::Mat3& headMountRotation() const { return R_H_IMU_; }

private:
    static double nowSeconds();
    void propagate(double now);          // 把大 yaw 估计外推到 now
    void recompute(double now);          // 重算全部输出

    // 由 cfg_ 重建两个安装矩阵与常量关节轴（不加锁；调用方持锁）
    void rebuildMountRotations();
    // 可信关节角外推到 now（小 yaw / pitch 编码器；与 recompute 内原逻辑一致）
    void trustedJointAngles(double now, double& theta_s, double& theta_p) const;
    // 关节轴（A/B 系 z）在 IMU 系中的方向:
    //   ON_BIG_YAW: a = R_A_IMUᵀ·ẑ（常量）;  ON_HEAD: a = R_H_IMUᵀ·Rx(θ_p)ᵀ·ẑ（随 pitch 变化）
    void axisInImu(double theta_p, double out[3]) const;

    mutable std::mutex mtx_;
    Config cfg_;
    rot::Mat3 R_A_IMU_;                          // IMU 系 → A 系（ON_BIG_YAW）
    rot::Mat3 R_H_IMU_;                          // IMU 系 → H 系（ON_HEAD）
    double axis_in_imu_[3] = {0.0, 0.0, 1.0};   // ON_BIG_YAW: 关节轴在 IMU 系中的方向

    Estimate out_;

    // ── IMU 状态 ──
    bool   imu_seen_ = false;
    double imu_t_ = -1.0;
    double imu_yaw_raw_ = 0.0, imu_pitch_raw_ = 0.0, imu_roll_raw_ = 0.0;
    double imu_yaw_unwrapped_ = 0.0;
    double imu_yaw_corr_ = 0.0;
    double platform_azimuth_ = 0.0;
    double platform_azimuth_corr_ = 0.0;
    double small_azimuth_ = 0.0;
    double small_azimuth_corr_ = 0.0;
    double gyro_[3] = {0.0, 0.0, 0.0};
    rot::Mat3 R_world_imu_ = rot::identity();

    // ── 编码器状态 ──
    bool   big_have_ = false;
    double big_meas_ = 0.0;        // 最近一次大 yaw 测量
    double big_meas_t_ = -1.0;     // 到达时刻
    double big_angle_ = 0.0;       // 延迟补偿估计
    double big_rate_ = 0.0;        // 关节角速度估计（互补滤波后）
    double big_rate_lpf_ = 0.0;    // IMU 支路低通（提供高频分量）
    double big_motor_rate_lpf_ = 0.0;   // 电机侧角速度低通
    bool   big_motor_rate_seen_ = false;
    double motor_rate_t_last_ = -1.0;   // 上次电机侧新样本时刻（用于 dt 相关系数）
    // ── 背隙中心（β）在线估计: 带遗忘的滑动 min/max ──
    //   这些量在 `estimate()`（const）里更新 —— estimate() 是"读估计"，
    //   但 β 的滑动窗口是随读带更新的在线状态，故显式 mutable。
    mutable double bl_win_min_ = 0.0;
    mutable double bl_win_max_ = 0.0;
    mutable bool   bl_win_seen_ = false;
    mutable double bl_t_last_ = -1.0;
    mutable double chassis_azimuth_ = 0.0;   // 上一拍底盘方位角（算 θ_p / 解卷绕参考用）
    double big_innovation_ = 0.0;
    uint8_t mcu2_seq_ = 0;
    bool   mcu2_seq_seen_ = false;        // 是否已收到过第一帧（首帧即视为一次更新）
    bool   big_first_ = true;
    double big_anchor_t_ = -1.0;         // 最近一次"学到新值"的上位机时刻
    double big_last_sample_t_ = -1.0;    // 同 big_anchor_t_（用于计算间隔）
    double big_sample_interval_ = 0.0;   // 新样本间隔（上位机计时）

    bool   small_seen_ = false;
    double small_t_ = -1.0;
    double small_angle_ = 0.0;
    double small_rate_ = 0.0;
    double small_rate_lpf_ = 0.0;

    bool   pitch_seen_ = false;
    double pitch_t_ = -1.0;
    double pitch_angle_ = 0.0;
    double pitch_rate_ = 0.0;
    double pitch_rate_lpf_ = 0.0;
    double pitch_acc_ = 0.0;
    double pitch_tick_ = -1.0;

    bool   chassis_seen_ = false;
    double chassis_t_ = -1.0;
    double chassis_anchor_t_ = -1.0;     // 最近一次"学到新值"的上位机时刻
    double chassis_yaw_ = 0.0;
    double chassis_yaw_unwrapped_ = 0.0;
    double chassis_yaw_corr_ = 0.0;
    double chassis_rate_ = 0.0;            // 零阶保持（不做加速度级外推）

    double last_update_t_ = -1.0;

    // 源统计
    Provenance prov_;
};

} // namespace tcbs

#endif // TCBS_YAW_STATE_ESTIMATOR_H
