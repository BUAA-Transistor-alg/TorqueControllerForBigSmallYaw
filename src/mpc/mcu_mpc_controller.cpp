#include "tcbs/mpc/mcu_mpc_controller.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tcbs {

using dual_yaw::DualYawMpc;
using dual_yaw::ModelExo;
using dual_yaw::ModelParams;

namespace {
template <typename T>
inline T clampv(const T& x, double lo, double hi) {
    return (x < lo) ? static_cast<T>(lo) : ((x > hi) ? static_cast<T>(hi) : x);
}
} // namespace

McuMpcController::McuMpcController(RobotCommunication* comm,
                                   const ModelParams& model,
                                   const dual_yaw::DualYawMpcConfig& mpc_cfg,
                                   const Config& cfg)
    : comm_(comm), mpc_(model, mpc_cfg), cfg_(cfg) {
    big_torque_only_ = cfg_.big_torque_only;
    small_torque_only_ = cfg_.small_torque_only;
    if (!cfg_.integral_on_big) cfg_.integral_gain[0] = 0.0;
}

McuMpcController::~McuMpcController() {
    stop();
}

void McuMpcController::start() {
    if (!running_.exchange(true)) {
        thread_ = std::thread(&McuMpcController::loop, this);
    }
}

void McuMpcController::stop() {
    if (running_.exchange(false)) {
        if (thread_.joinable()) thread_.join();
    }
}

// ============================================================================
// set（单目标）
// ============================================================================
void McuMpcController::set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
                           double big_yaw_azimuth, double small_yaw_azimuth,
                           double pitch_target_angle, bool fire, bool integral_enable) {
    std::lock_guard<std::mutex> lock(set_mtx_);
    auto_aim_enable_ = auto_aim_enable;
    big_torque_only_ = big_torque_only;
    small_torque_only_ = small_torque_only;
    integral_enable_ = integral_enable;
    target_big_azimuth_ = big_yaw_azimuth;
    target_small_azimuth_ = small_yaw_azimuth;
    pitch_target_angle_ = pitch_target_angle;
    fire_ = fire;

    big_azimuth_seq_.clear();
    small_azimuth_seq_.clear();
    pitch_seq_.clear();
    fire_seq_.clear();

    ticks_since_set_ = 0;
}

// ============================================================================
// set（序列）
// ============================================================================
void McuMpcController::set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
                           const std::vector<double>& big_yaw_azimuth_seq,
                           const std::vector<double>& small_yaw_azimuth_seq,
                           const std::vector<double>& pitch_seq,
                           const std::vector<bool>& fire_seq, bool integral_enable) {
    std::lock_guard<std::mutex> lock(set_mtx_);
    auto_aim_enable_ = auto_aim_enable;
    big_torque_only_ = big_torque_only;
    small_torque_only_ = small_torque_only;
    integral_enable_ = integral_enable;

    big_azimuth_seq_.assign(big_yaw_azimuth_seq.begin(), big_yaw_azimuth_seq.end());
    small_azimuth_seq_.assign(small_yaw_azimuth_seq.begin(), small_yaw_azimuth_seq.end());
    pitch_seq_.assign(pitch_seq.begin(), pitch_seq.end());
    fire_seq_.assign(fire_seq.begin(), fire_seq.end());

    ticks_since_set_ = 0;
}

McuMpcController::State McuMpcController::state() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return last_state_;
}

void McuMpcController::setConfig(const Config& c) {
    std::lock_guard<std::mutex> lock(cfg_mtx_);
    Config cc = c;
    if (!cc.integral_on_big) cc.integral_gain[0] = 0.0;   // 与构造语义一致
    cfg_ = cc;
    // 模式位与积分状态立即同步（模式位由 loop 每拍读取，无需重启线程）
    std::lock_guard<std::mutex> slock(set_mtx_);
    big_torque_only_ = cc.big_torque_only;
    small_torque_only_ = cc.small_torque_only;
    if (!integral_enable_) { integral_[0] = 0.0; integral_[1] = 0.0; }
}

McuMpcController::Config McuMpcController::config() const {
    std::lock_guard<std::mutex> lock(cfg_mtx_);
    return cfg_;
}

// ============================================================================
// 后台主循环
// ============================================================================
void McuMpcController::loop() {
    const int N = mpc_.config().N;
    McuMpcController::Config cfg_local;
    {
        std::lock_guard<std::mutex> lock(cfg_mtx_);
        cfg_local = cfg_;
    }
    const int ref_delay = std::max(0, cfg_local.ref_delay_steps);

    while (running_) {
        const auto loop_start = std::chrono::steady_clock::now();

        // ── 1. 取设置 + 序列消费 ──
        bool aa = true, big_to = false, small_to = false, fire = false, integral_en = false;
        double pitch_target = 0.0;
        std::vector<double> big_seq, small_seq;
        bool use_big_seq = false;
        {
            std::lock_guard<std::mutex> lock(set_mtx_);
            aa = auto_aim_enable_;
            big_to = big_torque_only_;
            small_to = small_torque_only_;
            fire = fire_;
            integral_en = integral_enable_;
            pitch_target = pitch_target_angle_;

            if (!big_azimuth_seq_.empty()) {
                big_seq.assign(big_azimuth_seq_.begin(), big_azimuth_seq_.end());
                target_big_azimuth_ = big_azimuth_seq_.front();
                big_azimuth_seq_.pop_front();
                use_big_seq = true;
            }
            if (!small_azimuth_seq_.empty()) {
                small_seq.assign(small_azimuth_seq_.begin(), small_azimuth_seq_.end());
                target_small_azimuth_ = small_azimuth_seq_.front();
                small_azimuth_seq_.pop_front();
            } else {
                small_seq = big_seq;   // 小 yaw 序列为空时与大 yaw 同步（保持关节角）
            }
            if (!pitch_seq_.empty()) {
                pitch_target_angle_ = pitch_seq_.front();
                pitch_seq_.pop_front();
                pitch_target = pitch_target_angle_;
            }
            if (!fire_seq_.empty()) {
                fire_ = fire_seq_.front();
                fire_seq_.pop_front();
                fire = fire_;
            }

            // 参考延迟缓冲
            const size_t want = static_cast<size_t>(ref_delay + 1);
            if (ref_delay > 0) {
                ref_buf_big_.push_back(target_big_azimuth_);
                ref_buf_small_.push_back(target_small_azimuth_);
                while (ref_buf_big_.size() > want) ref_buf_big_.pop_front();
                while (ref_buf_small_.size() > want) ref_buf_small_.pop_front();
            } else {
                ref_buf_big_.assign(1, target_big_azimuth_);
                ref_buf_small_.assign(1, target_small_azimuth_);
            }
        }
        const double target_big = ref_buf_big_.front();
        const double target_small = ref_buf_small_.front();

        // ── 2. 状态估计 ──
        YawStateEstimator::Estimate est;
        if (comm_) est = comm_->getEstimate();

        // ── 3. 组装 MPC 输入 ──
        DualYawMpc::Input in;
        // ★ 大 yaw 的模型状态 = **云台侧**关节角（θ_p）:
        //   模型（无论 2-DOF 还是 3-DOF 的云台行）描述的是云台，且参考是世界方位角，
        //   二者必须同源。以前这里填的是**电机侧**编码器角（big_joint_angle），
        //   在背隙内电机与云台解耦 ⇒ 状态与参考不同源，是抖动的模型根源之一。
        // 3-DOF 状态 {θ_motor, θ_platform, θ_small}: 电机侧有独立惯量，与小 yaw 一起填
        in.q[0] = est.big_motor_angle;
        in.q[1] = est.big_platform_angle;
        in.q[2] = est.small_joint_angle;
        in.qd[0] = est.big_joint_rate;
        in.qd[1] = est.small_joint_rate;
        // 平面模型外生量: 重力在关节参考系的平面分量 + 底盘绕关节轴的 ω/α
        //   （底盘倾角通过 gravity_a 进入 = 重力在 **A 系** 的投影；α_c 默认 0，视为慢变扰动）
        in.exo.gravity_a[0] = est.gravity_a[0];
        in.exo.gravity_a[1] = est.gravity_a[1];
        in.exo.base_omega = est.base_omega[2];   // 绕关节轴分量
        in.exo.base_alpha = 0.0;
        in.platform_azimuth = est.platform_azimuth;
        // 背隙死区中心 β（在线估计）作为外生量送进模型
        in.exo.backlash_beta = est.backlash_center;
        in.chassis_azimuth = est.chassis_azimuth;
        in.chassis_rate = est.chassis_yaw_rate;

        // 参考序列（世界方位角）
        if (use_big_seq && !big_seq.empty()) {
            // 整序列：直接提供（不足 N 用末值补齐，DualYawMpc 内部处理）
            in.ref_big_azimuth = big_seq;
            in.ref_small_azimuth = small_seq.empty() ? big_seq : small_seq;
            // 小 yaw 序列为空 → 与大 yaw 相同 → 关节角保持不变（回中倾向）
        } else {
            in.ref_big_azimuth.assign(N, target_big);
            in.ref_small_azimuth.assign(N, target_small);
        }
        in.prev_torque[0] = prev_torque_[0];
        in.prev_torque[1] = prev_torque_[1];

        // ── 4. 求解 ──
        DualYawMpc::Output res;
        bool solved = false;
        if (est.valid) {
            res = mpc_.solve(in);
            solved = true;
        }

        // ── 5. 积分补偿 ──
        double tau_out[2] = {0.0, 0.0};
        if (solved) {
            if (integral_en) {
                if (have_prev_pred_) {
                    integral_[0] += cfg_local.integral_gain[0] * (prev_pred_joint_[0] - est.big_platform_angle);
                    integral_[1] += cfg_local.integral_gain[1] * (prev_pred_joint_[1] - est.small_joint_angle);
                }
                integral_[0] = clampv(integral_[0], -cfg_local.integral_limit[0], cfg_local.integral_limit[0]);
                integral_[1] = clampv(integral_[1], -cfg_local.integral_limit[1], cfg_local.integral_limit[1]);
            } else {
                integral_[0] = integral_[1] = 0.0;
            }
            prev_pred_joint_[0] = res.pred_q[1];   // 云台侧
            prev_pred_joint_[1] = res.pred_q[2];   // 小 yaw
            have_prev_pred_ = true;

            const auto& mc = mpc_.config();
            tau_out[0] = clampv(res.torque[0] + integral_[0],
                                -mc.big.max_torque, mc.big.max_torque);
            tau_out[1] = clampv(res.torque[1] + integral_[1],
                                -mc.small.max_torque, mc.small.max_torque);
        }

        // ── 6. 组包（两关节: 模式位 + θ*/ω* + τ）──
        mcu::SendPacket pkt;
        pkt.auto_aim_enable = aa ? 1 : 0;
        pkt.fire = fire ? 1 : 0;
        pkt.pitch_target_angle = static_cast<float>(pitch_target);
        // 估计未就绪（从未拿到大 yaw 绝对基准）时退化为"仅力矩 + 零力矩"，
        // 避免电控内环去追一个无意义的目标角
        const bool ctrl_ready = est.valid;
        pkt.yaw_big_mode = (big_to || !ctrl_ready)
                               ? mcu::YAW_MODE_TORQUE_ONLY
                               : mcu::YAW_MODE_TORQUE_PLUS_PID;
        pkt.yaw_small_mode = (small_to || !ctrl_ready)
                                 ? mcu::YAW_MODE_TORQUE_ONLY
                                 : mcu::YAW_MODE_TORQUE_PLUS_PID;
        double theta_star[2] = {0.0, 0.0};
        double omega_star[2] = {0.0, 0.0};
        if (solved) {
            theta_star[0] = res.pred_q[1];     // ★ 云台侧（theta_star 语义 = 云台目标）
            theta_star[1] = res.pred_q[2];
            omega_star[0] = res.pred_qd[1];
            omega_star[1] = res.pred_qd[2];
        } else {
            theta_star[0] = est.big_platform_angle;
            theta_star[1] = est.small_joint_angle;
        }
        // ★ 下发的 yaw_big_target_angle/velocity **控制的是电机**，而 theta_star 是云台角
        //   ⇒ mode=1（电控位置环）时必须换算到**电机**坐标系: 保持当前传动形变
        //   Δ_meas = θ_motor − θ_platform，令电机目标 = 云台目标 + Δ_meas。
        //   （mode=0 仅力矩时电控不使用该字段，换算也无害。）
        const double transmission_offset = est.big_motor_angle - est.big_platform_angle;
        pkt.yaw_big_target_angle = theta_star[0] + transmission_offset;
        pkt.yaw_big_target_velocity = static_cast<float>(omega_star[0]);
        pkt.yaw_big_torque = static_cast<float>(tau_out[0]);
        pkt.yaw_small_target_angle = static_cast<float>(theta_star[1]);
        pkt.yaw_small_target_velocity = static_cast<float>(omega_star[1]);
        pkt.yaw_small_torque = static_cast<float>(tau_out[1]);

        bool sent = false;
        if (comm_) sent = comm_->sendToMcu(pkt);
        prev_torque_[0] = tau_out[0];
        prev_torque_[1] = tau_out[1];

        // ── 7. 缓存状态 ──
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            last_state_.torque[0] = tau_out[0];
            last_state_.torque[1] = tau_out[1];
            last_state_.torque_mpc[0] = solved ? res.torque[0] : 0.0;
            last_state_.torque_mpc[1] = solved ? res.torque[1] : 0.0;
            last_state_.integral[0] = integral_[0];
            last_state_.integral[1] = integral_[1];
            last_state_.target_joint[0] = theta_star[0];
            last_state_.target_joint[1] = theta_star[1];
            last_state_.target_joint_rate[0] = omega_star[0];
            last_state_.target_joint_rate[1] = omega_star[1];
            last_state_.big_torque_only = big_to;
            last_state_.small_torque_only = small_to;
            last_state_.ref_azimuth[0] = in.ref_big_azimuth.empty() ? 0.0 : in.ref_big_azimuth[0];
            last_state_.ref_azimuth[1] = in.ref_small_azimuth.empty() ? 0.0 : in.ref_small_azimuth[0];
            last_state_.delayed_ref_azimuth[0] = target_big;
            last_state_.delayed_ref_azimuth[1] = target_small;
            last_state_.ref_azimuth_seq[0] = in.ref_big_azimuth;
            last_state_.ref_azimuth_seq[1] = in.ref_small_azimuth;
            last_state_.pred_azimuth_seq[0] = res.pred_azimuth[0];
            last_state_.pred_azimuth_seq[1] = res.pred_azimuth[1];
            last_state_.pred_joint_seq[0] = res.pred_joint[0];
            last_state_.pred_joint_seq[1] = res.pred_joint[1];
            last_state_.small_ref_over_limit = solved && res.small_ref_over_limit;
            last_state_.solve_ms = res.solve_ms;
            last_state_.loop_fps = fps_counter_.fps();
            last_state_.ticks_since_set = ticks_since_set_.load();
            if (solved) {
                ++last_state_.solve_count;
                if (!res.usable) ++last_state_.solve_fail_count;
            }
            last_state_.estimator_valid = est.valid;
            last_state_.sent_ok = sent;
            last_state_.est = est;
        }

        // ── 8. 周期等待 ──
        std::this_thread::sleep_until(loop_start +
                                      std::chrono::duration<double>(cfg_local.loop_period));
        fps_counter_.tick();
        ++ticks_since_set_;
    }
}

} // namespace tcbs
