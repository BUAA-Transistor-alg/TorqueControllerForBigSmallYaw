#include "tcbs/communication/YawStateEstimator.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tcbs {

using rot::Mat3;

namespace {
constexpr double TWO_PI = 2.0 * M_PI;
}

YawStateEstimator::YawStateEstimator(const Config& cfg) : cfg_(cfg) {
    rebuildMountRotations();
}

// 由 cfg_ 重建两个安装矩阵（**两个都建**，用哪个由 imu_location 决定，便于在线切换构型）
//   R_A_IMU_ = R_A_IMU（IMU → A，ON_BIG_YAW 用）
//   R_H_IMU_ = R_H_IMU（IMU → H，ON_HEAD   用；H 相对 B 只绕 x 转 pitch）
//   axis_in_imu_ = R_A_IMUᵀ·ẑ（ON_BIG_YAW 的关节轴常量方向；ON_HEAD 需按 pitch 实时算）
void YawStateEstimator::rebuildMountRotations() {
    R_A_IMU_ = rot::eulerZXY(cfg_.mount_yaw, cfg_.mount_pitch, cfg_.mount_roll);
    R_H_IMU_ = rot::eulerZXY(cfg_.head_mount_yaw, cfg_.head_mount_pitch,
                             cfg_.head_mount_roll);
    const double z[3] = {0.0, 0.0, 1.0};
    rot::mulVec(rot::transpose(R_A_IMU_), z, axis_in_imu_);
}

// 关节轴（A/B 系 z，两 yaw 轴平行）在 IMU 系中的单位方向:
//   ON_BIG_YAW: IMU 与 A 固连 ⇒ 常量 a = R_A_IMUᵀ·ẑ
//   ON_HEAD   : 头相对 B 只绕 x 转 θ_p ⇒ ẑ_A 在 H 系 = Rx(θ_p)ᵀ·ẑ = (0, sin θ_p, cos θ_p)，
//               再转到 IMU 系: a(θ_p) = R_H_IMUᵀ·Rx(θ_p)ᵀ·ẑ  （**随 pitch 变化**）
void YawStateEstimator::axisInImu(double theta_p, double out[3]) const {
    if (cfg_.imu_location == Config::ImuLocation::ON_BIG_YAW) {
        out[0] = axis_in_imu_[0];
        out[1] = axis_in_imu_[1];
        out[2] = axis_in_imu_[2];
        return;
    }
    const double z_in_head[3] = {0.0, std::sin(theta_p), std::cos(theta_p)};
    rot::mulVec(rot::transpose(R_H_IMU_), z_in_head, out);
}

// 可信关节角（小 yaw / pitch 编码器）外推到 now；与 recompute 内的原逻辑完全一致
// （外推上限 max_extrap_s 防 MCU 停流时发散；源超时则退化为 0）
void YawStateEstimator::trustedJointAngles(double now, double& theta_s, double& theta_p) const {
    const bool small_ok = small_seen_ && (now - small_t_) < cfg_.source_timeout_s;
    const bool pitch_ok = pitch_seen_ && (now - pitch_t_) < cfg_.source_timeout_s;
    double dt_s = 0.0, dt_p = 0.0;
    if (small_ok) dt_s = std::clamp(now - small_t_, 0.0, cfg_.max_extrap_s);
    if (pitch_ok) dt_p = std::clamp(now - pitch_t_, 0.0, cfg_.max_extrap_s);
    theta_s = small_ok ? (small_angle_ + small_rate_ * dt_s) : 0.0;
    theta_p = pitch_ok ? (pitch_angle_ + pitch_rate_ * dt_p) : 0.0;
}

void YawStateEstimator::setConfig(const Config& cfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    cfg_ = cfg;
    // 两个安装矩阵都重建（构型切换时不必重新构造估计器）
    rebuildMountRotations();
}

YawStateEstimator::Config YawStateEstimator::config() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return cfg_;
}

double YawStateEstimator::nowSeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void YawStateEstimator::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    out_ = Estimate{};
    imu_seen_ = false;      imu_t_ = -1.0;
    imu_yaw_raw_ = imu_pitch_raw_ = imu_roll_raw_ = 0.0;
    imu_yaw_unwrapped_ = 0.0; imu_yaw_corr_ = 0.0;
    platform_azimuth_ = 0.0;  platform_azimuth_corr_ = 0.0;
    small_azimuth_ = 0.0;     small_azimuth_corr_ = 0.0;
    for (int i = 0; i < 3; ++i) gyro_[i] = 0.0;
    R_world_imu_ = rot::identity();

    big_have_ = false;      big_meas_ = 0.0;   big_meas_t_ = -1.0;
    big_angle_ = 0.0;       big_rate_ = 0.0;   big_rate_lpf_ = 0.0;
    big_rate_enc_lpf_ = 0.0; big_rate_enc_seen_ = false;
    big_rate_bias_ = 0.0;    enc_t_last_ = -1.0;
    big_innovation_ = 0.0;
    big_anchor_t_ = -1.0; big_last_sample_t_ = -1.0; big_sample_interval_ = 0.0;
    mcu2_seq_ = 0; mcu2_seq_seen_ = false;
    chassis_anchor_t_ = -1.0;
    big_first_ = true;

    small_seen_ = false;    small_t_ = -1.0;   small_angle_ = 0.0;
    small_rate_ = 0.0;      small_rate_lpf_ = 0.0;

    pitch_seen_ = false;    pitch_t_ = -1.0;   pitch_angle_ = 0.0;
    pitch_rate_ = 0.0;      pitch_rate_lpf_ = 0.0; pitch_acc_ = 0.0; pitch_tick_ = -1.0;

    chassis_seen_ = false;  chassis_t_ = -1.0;
    chassis_yaw_ = 0.0;     chassis_yaw_unwrapped_ = 0.0;
    chassis_yaw_corr_ = 0.0; chassis_rate_ = 0.0;

    last_update_t_ = -1.0;
    prov_ = Provenance{};
}

// ============================================================================
// 高频路径: IMU（位置由 Config::imu_location 决定: 大 yaw 转子 A / 头 H）
// ============================================================================
void YawStateEstimator::onImu(double euler_yaw, double euler_pitch, double euler_roll,
                              double gx, double gy, double gz) {
    std::lock_guard<std::mutex> lock(mtx_);
    const double now = nowSeconds();

    imu_yaw_raw_ = euler_yaw;
    imu_pitch_raw_ = euler_pitch;
    imu_roll_raw_ = euler_roll;
    gyro_[0] = gx; gyro_[1] = gy; gyro_[2] = gz;
    R_world_imu_ = rot::eulerZXY(euler_yaw, euler_pitch, euler_roll);

    // IMU 世界 yaw 解卷绕（连续化，供外部显示/控制）
    imu_yaw_unwrapped_ = rot::unwrapTo(euler_yaw, imu_yaw_unwrapped_, imu_yaw_corr_);

    // 可信关节角（ON_HEAD 需要 θ_p 定关节轴方向、并扣除 θ̇_s；ON_BIG_YAW 不需要，
    // 故 theta_s_now 仅由 helper 一并给出、此处不参与投影）
    const bool on_head = cfg_.imu_location == Config::ImuLocation::ON_HEAD;
    double theta_s_now = 0.0, theta_p_now = 0.0;
    trustedJointAngles(now, theta_s_now, theta_p_now);
    const bool small_ok_now = small_seen_ && (now - small_t_) < cfg_.source_timeout_s;
    (void)theta_s_now;

    // 陀螺在关节轴上的投影:
    //   ON_BIG_YAW: = 大 yaw 平台的世界角速度在关节轴上的分量 = ω_chassis + θ̇_b
    //   ON_HEAD   : = **头**的世界角速度在关节轴上的分量 = ω_chassis + θ̇_b + θ̇_s
    //               （a_imu 随 θ_p 变化；俯仰角速度 ṗ 与关节轴垂直，不进入投影）
    double a_imu[3];
    axisInImu(theta_p_now, a_imu);
    const double axis_proj = gx * a_imu[0] + gy * a_imu[1] + gz * a_imu[2];

    // 先把上一拍的估计外推到当前时刻，再更新角速度
    propagate(now);

    // 大 yaw 关节角速度 = 平台（或头）世界角速度 − 底盘角速度（− 小 yaw 关节角速度）
    //   （底盘 IMU 的 yaw 角速度更新率低、值被保持 → 用其采样年龄判断可用性）
    const double chassis_age_now = (chassis_seen_ && chassis_anchor_t_ > 0.0)
        ? (now - chassis_anchor_t_ + cfg_.transport_delay_s) : 1e9;
    // 底盘 IMU 值较旧时继续沿用（零阶保持: 底盘角速度变化缓慢，比当作"底盘不转"准得多；
    // 刻意不做加速度级外推），超时才退化
    const bool chassis_ok = cfg_.use_chassis_imu && chassis_seen_ &&
                            chassis_age_now < cfg_.chassis_imu_timeout_s;
    // ⚠ ON_HEAD: 这里减去的是**小 yaw 编码器角速度**，它带低通滞后与量化噪声
    //   （小 yaw 编码器只按 MCU 包率更新）⇒ 大 yaw 关节角速度估计的噪声比
    //   ON_BIG_YAW 明显更大，并会经延迟补偿/外推进入 big_joint_angle。
    //   这是构型切换的**固有代价**（IMU 不再与大 yaw 关节同体），非 bug。
    const double small_rate_term = (on_head && small_ok_now) ? small_rate_ : 0.0;
    const double rate_target = axis_proj - (chassis_ok ? chassis_rate_ : 0.0)
                             - small_rate_term;
    if (!imu_seen_) {
        big_rate_lpf_ = rate_target;
    } else {
        big_rate_lpf_ += cfg_.big_rate_lpf_alpha * (rate_target - big_rate_lpf_);
    }
    // ── 用编码器值**慢速校正 IMU 支路的直流误差**（不把延迟的编码器值当 DC 用）──
    //   ★ 为什么不是"互补滤波（低频取编码器 + 高频取 IMU）":
    //     编码器值走 MCU1↔MCU2 低速链路（约 3~10 Hz）**且带传输延迟**，
    //     它的直流分量本身就是"延迟后的真值"。而互补滤波只能补相位、补不了延迟
    //     ⇒ 拿它当 DC 会给 0.3 Hz 量级的运动引入 (延迟 × θ̈) 的系统误差。
    //     实测（tests/test_yaw_state_estimator）: 那种写法把 ON_BIG_YAW 的
    //     大 yaw 角速度误差从 0.25 rad/s 放大到 ~1.07 rad/s，`base_omega` 也偏 0.21 rad/s。
    //   ✅ 正确用法: 编码器角速度**无积分漂移**，所以只用它把 IMU 支路的**直流**慢慢拉回来
    //     （陀螺偏置 / 底盘角速度残差 / ON_HEAD 的 θ̇_s 低通残差都是直流型的误差）。
    //     高频路径完全不动 ⇒ 不引入链路延迟。
    prov_.big_rate_from_imu = imu_seen_;
    if (cfg_.big_rate_use_encoder && big_rate_enc_seen_ && cfg_.big_rate_bias_tau_s > 1e-9) {
        const double dc_err = big_rate_lpf_ - big_rate_enc_lpf_;   // IMU 低频 − 编码器低频
        // 按 IMU 实际间隔给系数 ⇒ 与采样率无关（时间常数 = big_rate_bias_tau_s）
        const double dt_imu = (imu_t_ > 0.0) ? (now - imu_t_) : 0.0;
        const double a_bias = (dt_imu > 1e-9)
                                  ? (1.0 - std::exp(-dt_imu / cfg_.big_rate_bias_tau_s)) : 0.0;
        big_rate_bias_ += a_bias * (dc_err - big_rate_bias_);
        big_rate_ = big_rate_lpf_ - big_rate_bias_;
        prov_.big_rate_from_encoder = true;
    } else {
        big_rate_ = big_rate_lpf_;
        prov_.big_rate_from_encoder = false;
    }

    imu_seen_ = true;
    imu_t_ = now;
    ++prov_.imu.count;
    prov_.imu.valid = true;
    prov_.imu.age_s = 0.0;

    recompute(now);
}

// ============================================================================
// 低频路径: MCU（大/小 yaw 与 pitch 编码器 + 底盘 IMU）
// ============================================================================
void YawStateEstimator::onMcu(double yaw_big_angle, double yaw_big_omega,
                              double yaw_small_angle, double yaw_small_omega,
                              double pitch_angle,
                              double chassis_imu_yaw, double chassis_imu_omega,
                              uint8_t mcu2_seq) {
    std::lock_guard<std::mutex> lock(mtx_);
    const double now = nowSeconds();

    // 新样本判定（大 yaw 与底盘 IMU 同源、共用同一个序号）:
    //   首帧到达（上电第一帧即视为一次更新，不做可用性检测）
    //   或 序号变化（即使值恰好相同也能识别）
    //   或 值变化（兼容序号未递增/未提供序号的实现）
    const bool mcu2_seq_changed = mcu2_seq_seen_ && (mcu2_seq != mcu2_seq_);
    const bool first_packet = !mcu2_seq_seen_;
    mcu2_seq_ = mcu2_seq;
    mcu2_seq_seen_ = true;


    // ── 底盘 IMU（经 MCU2: 更新率低、间隔不规则、值被保持；仅零阶保持）──
    //      序号变化 = 新样本；序号未变化 = 值被保持（此时不刷新锚点，年龄继续增长）
    {
        const bool value_changed = !chassis_seen_ ||
            std::fabs(chassis_imu_omega - chassis_rate_) > 1e-9 ||
            std::fabs(chassis_imu_yaw - chassis_yaw_) > 1e-12;
        const bool new_sample = first_packet || mcu2_seq_changed || value_changed;
        ++prov_.chassis_imu.count;
        if (new_sample) {
            chassis_yaw_ = chassis_imu_yaw;
            chassis_yaw_unwrapped_ = rot::unwrapTo(chassis_imu_yaw, chassis_yaw_unwrapped_,
                                                   chassis_yaw_corr_);
            // 底盘角速度**零阶保持**（不做加速度级外推: 底盘角速度变化缓慢，
            // 外推反而可能发散 —— 这是刻意的设计约束）
            chassis_rate_ = chassis_imu_omega;
            chassis_seen_ = true;
            chassis_anchor_t_ = now;
            ++prov_.chassis_imu.new_samples;
        } else {
            ++prov_.chassis_imu.rejected;   // 值被保持（不是新数据）
        }
    }

    // ── 小 yaw 编码器（可信、实时）──
    {
        const bool new_sample = !small_seen_ ||
                                std::fabs(yaw_small_angle - small_angle_) > 1e-12 ||
                                std::fabs(yaw_small_omega - small_rate_lpf_) > 1e-9;
        small_angle_ = yaw_small_angle;
        if (!small_seen_) {
            small_rate_lpf_ = yaw_small_omega;
        } else {
            small_rate_lpf_ += cfg_.small_rate_lpf_alpha * (yaw_small_omega - small_rate_lpf_);
        }
        small_rate_ = small_rate_lpf_;
        small_seen_ = true;
        small_t_ = now;
        ++prov_.small_enc.count;
        if (new_sample) ++prov_.small_enc.new_samples; else ++prov_.small_enc.rejected;
        prov_.small_enc.valid = true;
        prov_.small_enc.age_s = 0.0;
    }

    // ── pitch 编码器（可信、实时）──
    {
        const bool new_sample = !pitch_seen_ || std::fabs(pitch_angle - pitch_angle_) > 1e-12;
        if (pitch_seen_ && pitch_t_ > 0.0 && new_sample) {
            const double dt = now - pitch_t_;
            if (dt > 1e-4 && dt < 0.5) {
                const double rate_raw = (pitch_angle - pitch_angle_) / dt;
                const double rate_lp = pitch_rate_lpf_ +
                                       cfg_.pitch_rate_lpf_alpha * (rate_raw - pitch_rate_lpf_);
                if (cfg_.pitch_rate_lpf_alpha > 1e-9 && dt > 1e-4) {
                    const double acc_raw = (rate_lp - pitch_rate_lpf_) / dt;
                    pitch_acc_ += cfg_.pitch_acc_lpf_alpha * (acc_raw - pitch_acc_);
                }
                pitch_rate_lpf_ = rate_lp;
                pitch_rate_ = rate_lp;
            }
        }
        pitch_angle_ = pitch_angle;
        pitch_seen_ = true;
        pitch_t_ = now;
        ++prov_.pitch_enc.count;
        if (new_sample) ++prov_.pitch_enc.new_samples; else ++prov_.pitch_enc.rejected;
        prov_.pitch_enc.valid = true;
        prov_.pitch_enc.age_s = 0.0;
    }

    // ── 大 yaw 反馈（**经 MCU2: 更新率低、间隔不规则、值被保持**）──
    {
        // 新样本判定: 首帧 / 序号变化 / 值变化
        //   ★ 编码器角速度 `yaw_big_omega` 与角度**同源、同一次 MCU2 取数一起刷新**，
        //     所以直接复用这一个判据即可（**不要**把"omega 变了"并进来 ——
        //     那会把"值被保持"误判成新样本，破坏 value-hold 检测）。
        const bool new_sample = first_packet || mcu2_seq_changed ||
                                !big_have_ || std::fabs(yaw_big_angle - big_meas_) > 1e-12;

        ++prov_.big_enc.count;
        prov_.big_enc.valid = true;

        // ── 大 yaw 角速度的**编码器支路**（互补滤波的低频/直流部分）──
        //   `yaw_big_omega` 是电控按编码器算出的关节角速度（已过映射，符号/比例一致）。
        //   它走 MCU1↔MCU2 低速链路、值被保持 ⇒ 只在**新样本**时推动一次低通，
        //   避免"值被保持"期间把同一个数反复灌进滤波器（那会伪造出额外的平滑）。
        //   ⚠ 判据必须与上面**同一个**（含"角度值变化"那一项）：只按序号判断的话，
        //     遇到序号不递增（或恒为 0）的实现会**永远不更新**，低频支路被冻结在首帧值。
        if (cfg_.big_rate_use_encoder && new_sample) {
            if (!big_rate_enc_seen_) {
                big_rate_enc_lpf_ = yaw_big_omega;      // 首帧直接锚定，避免从 0 慢慢爬
                big_rate_enc_seen_ = true;
            } else {
                // 按**实际间隔**给系数：链路率在 3~100 Hz 之间变化，固定 α 会让时间常数飘
                const double d_enc = (enc_t_last_ > 0.0) ? (now - enc_t_last_) : 0.0;
                const double a = (d_enc > 1e-9 && cfg_.big_rate_enc_tau_s > 1e-9)
                                     ? (1.0 - std::exp(-d_enc / cfg_.big_rate_enc_tau_s))
                                     : cfg_.big_rate_enc_alpha;
                big_rate_enc_lpf_ += a * (yaw_big_omega - big_rate_enc_lpf_);
            }
            enc_t_last_ = now;
        }

        if (new_sample) {
            if (big_anchor_t_ > 0.0) big_sample_interval_ = now - big_last_sample_t_;
            big_last_sample_t_ = now;
            big_anchor_t_ = now;
            ++prov_.big_enc.new_samples;
            big_meas_ = yaw_big_angle;
            big_meas_t_ = now;

            // 先把当前估计外推到 now（用 IMU 提供的高频速率）
            propagate(now);

            // ── 延迟补偿（一阶: 只用速度外推，无反馈环、无条件稳定）──
            // 值的年龄 = 传输时延（上位机无法知道电控侧采样时刻，因为 MCU 无时钟可用），
            //   θ̂(now) = m + θ̇_big·transport_delay
            // 不做"与历史估计比较再乘增益"的闭环校正（回路增益受延迟限制，易振荡/发散）；
            // 修正量另加幅值上限（抗坏帧/跳变）。
            if (big_first_ || !big_have_) {
                big_angle_ = yaw_big_angle;
                big_first_ = false;
                big_innovation_ = 0.0;
                if (!imu_seen_) big_rate_lpf_ = yaw_big_omega;   // IMU 还没来时用编码器角速度兜底
            } else {
                const double target = yaw_big_angle + big_rate_lpf_ * cfg_.transport_delay_s;
                const double corr = std::clamp(target - big_angle_,
                                               -cfg_.big_enc_max_jump, cfg_.big_enc_max_jump);
                big_innovation_ = target - big_angle_;
                big_angle_ += corr;
            }
            big_have_ = true;
        } else {
            ++prov_.big_enc.rejected;   // 值被保持（不是新数据）
        }
    }

    propagate(now);
    recompute(now);
}

// ============================================================================
// 外推 + 历史
// ============================================================================
void YawStateEstimator::propagate(double now) {
    if (!big_have_) {
        last_update_t_ = now;
        return;
    }
    if (last_update_t_ > 0.0) {
        const double dt = now - last_update_t_;
        if (dt > 0.0 && dt < 0.5) {
            big_angle_ += big_rate_ * dt;
        }
    }
    last_update_t_ = now;
}

// ============================================================================
// 汇总输出
// ============================================================================
void YawStateEstimator::recompute(double now) {
    Estimate o;

    const bool imu_ok = imu_seen_ && (now - imu_t_) < cfg_.source_timeout_s;
    // 值的年龄: 从上位机"首次看到该新样本"算起（+ 传输时延），保持期间持续增大
    const double big_sample_age_now = (big_have_ && big_anchor_t_ > 0.0)
        ? (now - big_anchor_t_ + cfg_.transport_delay_s) : -1.0;
    const double chassis_sample_age_now = (chassis_seen_ && chassis_anchor_t_ > 0.0)
        ? (now - chassis_anchor_t_ + cfg_.transport_delay_s) : -1.0;
    const bool big_ok = big_have_;   // 已获得绝对基准（值可能偏旧，用年龄上报反映）
    const bool small_ok = small_seen_ && (now - small_t_) < cfg_.source_timeout_s;
    const bool pitch_ok = pitch_seen_ && (now - pitch_t_) < cfg_.source_timeout_s;
    const bool chassis_ok = chassis_seen_ &&
                            chassis_sample_age_now < cfg_.chassis_imu_timeout_s;

    prov_.imu.age_s = imu_seen_ ? (now - imu_t_) : -1.0;
    prov_.big_enc.age_s = big_sample_age_now;
    prov_.chassis_imu.age_s = chassis_sample_age_now;
    prov_.big_enc_sample_age_s = big_sample_age_now;
    prov_.small_enc.age_s = small_seen_ ? (now - small_t_) : -1.0;
    prov_.pitch_enc.age_s = pitch_seen_ ? (now - pitch_t_) : -1.0;

    prov_.imu.valid = imu_ok;
    prov_.big_enc.valid = big_ok;
    prov_.small_enc.valid = small_ok;
    prov_.pitch_enc.valid = pitch_ok;
    prov_.chassis_imu.valid = chassis_ok;
    // "过旧"标记: 值被长时间保持时置位（供上层决定是否降权/告警）
    prov_.imu.stale = imu_seen_ && (now - imu_t_) > cfg_.stale_age_s;
    prov_.big_enc.stale = big_have_ && big_sample_age_now > cfg_.stale_age_s;
    prov_.small_enc.stale = small_seen_ && (now - small_t_) > cfg_.stale_age_s;
    prov_.pitch_enc.stale = pitch_seen_ && (now - pitch_t_) > cfg_.stale_age_s;
    prov_.chassis_imu.stale = chassis_seen_ && chassis_sample_age_now > cfg_.stale_age_s;

    uint64_t mask = 0;
    if (imu_ok) mask |= USED_IMU;
    if (big_ok) mask |= USED_BIG_ENC;
    if (small_ok) mask |= USED_SMALL_ENC;
    if (pitch_ok) mask |= USED_PITCH_ENC;
    if (chassis_ok) mask |= USED_CHASSIS_IMU;
    prov_.used_mask = mask;
    prov_.big_enc_innovation = big_innovation_;

    // ── 1) 可信量（小 yaw / pitch 编码器为实时可信量；按其角速度外推到当前时刻，
    //        消除 MCU 包周期内的采样滞后，外推上限 max_extrap_s 防止 MCU 停流时发散）──
    double theta_s = 0.0, theta_p = 0.0;
    trustedJointAngles(now, theta_s, theta_p);

    o.valid = big_ok;
    o.imu_yaw = imu_yaw_raw_;
    o.imu_pitch = imu_pitch_raw_;
    o.imu_roll = imu_roll_raw_;
    o.platform_rate = big_rate_lpf_;
    o.small_joint_angle = theta_s;
    o.small_joint_rate = small_rate_;
    o.pitch_joint_angle = theta_p;
    o.pitch_joint_rate = pitch_rate_;

    // ── 2) 大 yaw 估计 ──
    o.big_joint_angle_meas = big_meas_;
    o.big_joint_angle = big_angle_;
    o.big_joint_rate = big_rate_;
    o.big_enc_age = big_sample_age_now;
    o.big_sample_interval = big_sample_interval_;
    o.chassis_imu_age = chassis_sample_age_now;
    o.big_enc_innovation = big_innovation_;
    o.big_has_encoder = big_have_;

    // ── 3) 反解真实位姿（仅用可信量 + 标定参数；两种构型分支，对外语义完全一致）──
    //   ON_BIG_YAW: R_world_A    = R_world_imu · R_mountᵀ
    //               R_world_head = R_world_A · Rz(θ_s) · Rx(θ_p)
    //   ON_HEAD   : R_world_head = R_world_imu · R_mount_headᵀ   （头姿态由 IMU 直接给出）
    //               R_world_A    = R_world_head · Rx(θ_p)ᵀ · Rz(θ_s)ᵀ
    //   R_world_B = R_world_A · Rz(θ_s)（= 欠 pitch 的头系）: 其 x 轴方位角
    //               在两种构型下都等于**头 x 轴的世界方位角**（Rx(p)·x̂ = x̂）
    const bool on_head = cfg_.imu_location == Config::ImuLocation::ON_HEAD;
    Mat3 R_world_A, R_world_head;
    if (!on_head) {
        R_world_A = rot::mul(R_world_imu_, rot::transpose(R_A_IMU_));
        R_world_head = rot::mul(rot::mul(R_world_A, rot::rotZ(theta_s)), rot::rotX(theta_p));
    } else {
        R_world_head = rot::mul(R_world_imu_, rot::transpose(R_H_IMU_));
        R_world_A = rot::mul(rot::mul(R_world_head, rot::transpose(rot::rotX(theta_p))),
                             rot::transpose(rot::rotZ(theta_s)));
    }
    const Mat3 R_world_B = rot::mul(R_world_A, rot::rotZ(theta_s));

    // ── ★ 严格反解数据包（IMU 为准确值 ⇒ 反解底盘；并打包反解用到的全部数据）──
    o.strict_pose = dual_yaw::makeStrictPose(imu_yaw_raw_, imu_pitch_raw_, imu_roll_raw_,
                                   on_head ? static_cast<int>(dual_yaw::StrictPoseImuLocation::ON_HEAD)
                                           : static_cast<int>(dual_yaw::StrictPoseImuLocation::ON_BIG_YAW),
                                   big_angle_, theta_s, theta_p,
                                   cfg_.mount_yaw, cfg_.mount_pitch, cfg_.mount_roll,
                                   cfg_.head_mount_yaw, cfg_.head_mount_pitch,
                                   cfg_.head_mount_roll,
                                   big_sample_age_now);


    {
        const double ax[3] = {R_world_A.m[0][0], R_world_A.m[1][0], R_world_A.m[2][0]};
        const double az_raw = std::atan2(ax[1], ax[0]);
        platform_azimuth_ = rot::unwrapTo(az_raw, platform_azimuth_, platform_azimuth_corr_);
        o.platform_azimuth = platform_azimuth_;
    }
    {
        const double ax[3] = {R_world_B.m[0][0], R_world_B.m[1][0], R_world_B.m[2][0]};
        const double az_raw = std::atan2(ax[1], ax[0]);
        small_azimuth_ = rot::unwrapTo(az_raw, small_azimuth_, small_azimuth_corr_);
        o.small_output_azimuth = small_azimuth_;
    }
    rot::matToEulerZXY(R_world_head, o.head_world_yaw, o.head_world_pitch, o.head_world_roll);

    {
        double bore[3] = {cfg_.bore[0], cfg_.bore[1], cfg_.bore[2]};
        const double n = std::sqrt(bore[0] * bore[0] + bore[1] * bore[1] + bore[2] * bore[2]);
        if (n > 1e-9) { bore[0] /= n; bore[1] /= n; bore[2] /= n; }
        double los[3];
        rot::mulVec(R_world_head, bore, los);
        o.los_azimuth = std::atan2(los[1], los[0]);
        o.los_elevation = std::asin(std::max(-1.0, std::min(1.0, los[2])));
    }

    // 底盘方位角: ★ 用运动学链矩阵的精确值（R_world_chassis 的 x 轴世界方位角），
    //   而不是 "ψ_platform − θ_b" 的标量近似 —— 底盘有俯仰/横滚倾斜时两者差约 pitch·roll
    //   （本仓库测试里 8°/5° 倾斜下差 ~0.01 rad）；矩阵值在倾斜下严格正确。
    //   解卷绕: 取与标量近似值最近的圈，保持与 platform_azimuth_ 同源的连续性。
    {
        const double az_wrapped = o.strict_pose.chassis_azimuth;
        const double ref = platform_azimuth_ - big_angle_;
        o.chassis_azimuth = std::remainder(az_wrapped - ref, 2.0 * M_PI) + ref;
    }
    o.chassis_yaw_rate = chassis_seen_ ? chassis_rate_ : 0.0;   // 零阶保持

    // ── 4) 模型外生量 ──
    // 重力: 先转到 IMU 系 g_imu = R_world_imuᵀ·(0,0,−g)，再转到 **A 系（大 yaw 转子系）**。
    //   ON_BIG_YAW（IMU 就在 A 系上）: g_A = R_A_IMU·g_imu —— 与 θ_b 无关，最稳。
    //   ON_HEAD  （IMU 在头上）       : g_H = R_H_IMU·g_imu，再
    //                                   g_A = Rz(θ_s)·Rx(θ_p)·g_H
    //                                   —— 只用**可信编码器** θ_s/θ_p，同样与 θ_b 无关。
    //   （两种构型都不需要延迟/带误差的大 yaw 编码器，这正是"重力必须给 A 系"的原因之一）
    {
        const double g_world[3] = {0.0, 0.0, -cfg_.gravity};
        double g_imu[3];
        rot::mulVec(rot::transpose(R_world_imu_), g_world, g_imu);
        double g_A[3];
        if (!on_head) {
            rot::mulVec(R_A_IMU_, g_imu, g_A);
        } else {
            double g_H[3];
            rot::mulVec(R_H_IMU_, g_imu, g_H);
            rot::mulVec(rot::mul(rot::rotZ(theta_s), rot::rotX(theta_p)), g_H, g_A);
        }
        for (int k = 0; k < 3; ++k) o.gravity_a[k] = g_A[k];
    }
    // 底盘角速度在 C 系: ω_c^A = ω_platform^A − θ̇_b·ẑ，再转到 C 系
    //   ON_BIG_YAW: ω_platform^A = R_A_IMU·gyro（IMU 与大 yaw 平台同体，**就是**平台角速度）
    //   ON_HEAD   : IMU 在头上 ⇒ 先得到头的角速度 ω_H = R_H_IMU·gyro，再反推到 A 系
    //               ω_platform^A = Rz(θ_s)·Rx(θ_p)·ω_H
    //   ⚠ ON_HEAD 的残余: 头的角速度里还含 θ̇_s·ẑ 与 ṗ·x̂（俯仰角速度），本式只减去了 θ̇_b·ẑ
    //     ⇒ base_omega 会多出 θ̇_s·ẑ_A + ṗ·x̂ 两项（ṗ 与 θ̇_s 均是可信量，但它们在
    //     "关节参考系角速度"里并无对应关节）。ON_BIG_YAW 下无此残余（真值即 ω_chassis）。
    {
        double w_imu[3] = {gyro_[0], gyro_[1], gyro_[2]};
        double w_A[3];
        if (!on_head) {
            rot::mulVec(R_A_IMU_, w_imu, w_A);
        } else {
            double w_H[3];
            rot::mulVec(R_H_IMU_, w_imu, w_H);
            rot::mulVec(rot::mul(rot::rotZ(theta_s), rot::rotX(theta_p)), w_H, w_A);
        }
        w_A[2] -= big_rate_;
        const Mat3 Rz_minus = rot::rotZ(-big_angle_);
        rot::mulVec(Rz_minus, w_A, o.base_omega);
    }
    o.pitch_acc = pitch_acc_;

    // ── 反解来源判定 ──
    prov_.reverse_from_trusted = imu_ok && small_ok && pitch_ok;
    o.prov = prov_;
    out_ = o;
}

YawStateEstimator::Estimate YawStateEstimator::estimate() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return out_;
}

} // namespace tcbs
