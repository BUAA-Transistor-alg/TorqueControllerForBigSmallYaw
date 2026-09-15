#ifndef TCBS_MCU_MPC_CONTROLLER_H
#define TCBS_MCU_MPC_CONTROLLER_H

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "tcbs/communication/Communications.hpp"
#include "tcbs/common/FrameRateCounter.h"
#include "tcbs/mpc/dual_yaw_mpc.h"

namespace tcbs {

// ============================================================================
// McuMpcController — 实车双级 yaw 控制封装（状态估计 + 耦合 MPC + 组包发送 + 后台线程）
//
// 后台线程（周期 = loop_period）每拍:
//   1) 取状态估计（YawStateEstimator::Estimate）: 可信实时量 + 大yaw延迟补偿估计
//   2) 组装 MPC 输入（含 pitch 外生量、底盘 ω_c、IMU 实测重力方向 → 任意倾斜）
//   3) 求解双 yaw 耦合非线性 MPC（参考 = 世界方位角序列）
//   4) 可选积分补偿（逐关节，补偿模型误差/未建模扰动）
//   5) 组包（两关节: 模式位 + θ*/ω*/τ_ff）并发给 MCU
//
// 目标语义: 大 yaw 与小 yaw 的**世界系方位角**（解卷绕连续），分别给出。
//   大 yaw 方位角 ψ_big  = 大 yaw 平台 x 轴的世界方位角（IMU 直测）
//   小 yaw 方位角 ψ_small = 小 yaw 输出 x 轴的世界方位角 = ψ_big + θ_small（编码器直测）
//   两者之差即小 yaw 的关节角指令；调用方负责"快慢分配"（小 yaw 承担快速分量、
//   大 yaw 承担展开/回中），控制器按给定分配精确跟踪并自动考虑耦合与限位。
// ============================================================================
class McuMpcController {
public:
    struct Config {
        double loop_period = 0.01;      // 后台 loop 周期 (s)
        // 关节控制模式（发送给电控的模式位）
        bool big_torque_only = false;   // false → 力矩 + 电控位置/速度内环
        bool small_torque_only = false;
        // 参考延迟步数（0 = 直接使用最新目标；>0 = 与旧实现相同的"目标前瞻"缓冲）
        int ref_delay_steps = 0;
        // 积分补偿（逐关节；0 = 关闭）
        double integral_gain[2]  = {0.0, 0.01};
        double integral_limit[2] = {0.0, 0.30};
        bool   integral_on_big = false; // 是否也允许大 yaw 积分补偿
    };

    struct State {
        // ── 控制输出 ──
        double torque[2] = {0.0, 0.0};          // 实际发送的两关节力矩 (N·m)
        double torque_mpc[2] = {0.0, 0.0};      // MPC 解（未加积分）
        double integral[2] = {0.0, 0.0};
        double target_joint[2] = {0.0, 0.0};    // 发送的 θ*（大yaw关节系/小yaw关节系）
        double target_joint_rate[2] = {0.0, 0.0};
        bool   big_torque_only = false, small_torque_only = false;

        // ── 参考与预测（世界方位角 & 关节角）──
        double ref_azimuth[2] = {0.0, 0.0};         // 本拍使用的大/小 yaw 世界方位角参考
        double delayed_ref_azimuth[2] = {0.0, 0.0}; // 延迟缓冲后的参考（等价语义）
        std::vector<double> ref_azimuth_seq[2];
        std::vector<double> pred_azimuth_seq[2];
        std::vector<double> pred_joint_seq[2];
        bool   small_ref_over_limit = false;

        // ── 性能/诊断 ──
        double solve_ms = 0.0;
        double loop_fps = 0.0;
        uint64_t ticks_since_set = 0;
        uint32_t solve_count = 0;
        uint32_t solve_fail_count = 0;
        bool   estimator_valid = false;
        bool   sent_ok = false;

        // ── 状态估计快照（含"可信值+滤波值反解"结果与数据来源）──
        YawStateEstimator::Estimate est;
    };

    static Config defaultConfig() { return Config{}; }
    McuMpcController(RobotCommunication* comm,
                     const dual_yaw::ModelParams& model,
                     const dual_yaw::DualYawMpcConfig& mpc_cfg,
                     const Config& cfg = defaultConfig());
    ~McuMpcController();

    void start();
    void stop();

    // 单目标 set: 大/小 yaw 世界方位角 + pitch 目标 + 火控
    //   big_torque_only / small_torque_only: 逐关节模式位
    //   integral_enable: 本步是否启用积分补偿（false 时积分清零）
    void set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
             double big_yaw_azimuth, double small_yaw_azimuth,
             double pitch_target_angle, bool fire, bool integral_enable = false);

    // 序列 set: 三个通道各自独立序列（不截断；某序列为空则该通道保持当前值）
    void set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
             const std::vector<double>& big_yaw_azimuth_seq,
             const std::vector<double>& small_yaw_azimuth_seq,
             const std::vector<double>& pitch_seq,
             const std::vector<bool>& fire_seq,
             bool integral_enable = false);

    State state() const;

    // 运行时更新控制器配置（不重启后台线程；loop_period 变更下一拍生效）
    void setConfig(const Config& c);
    Config config() const;

    dual_yaw::DualYawMpc& mpc() { return mpc_; }
    const dual_yaw::DualYawMpc& mpc() const { return mpc_; }

private:
    void loop();

    RobotCommunication* comm_;
    dual_yaw::DualYawMpc mpc_;

    mutable std::mutex set_mtx_;
    bool   auto_aim_enable_ = true;
    bool   big_torque_only_ = false;
    bool   small_torque_only_ = false;
    bool   integral_enable_ = false;
    double target_big_azimuth_ = 0.0;
    double target_small_azimuth_ = 0.0;
    double pitch_target_angle_ = 0.0;
    bool   fire_ = false;

    // 序列模式成员（非空时 loop 优先消费）
    std::deque<double> big_azimuth_seq_;
    std::deque<double> small_azimuth_seq_;
    std::deque<double> pitch_seq_;
    std::deque<bool>   fire_seq_;

    // 参考延迟缓冲
    std::deque<double> ref_buf_big_;
    std::deque<double> ref_buf_small_;

    // 上一拍实际施加的力矩（仅 loop 线程读写；作为力矩变化率约束的起点）
    double prev_torque_[2] = {0.0, 0.0};

    // 积分补偿状态
    double integral_[2] = {0.0, 0.0};
    double prev_pred_joint_[2] = {0.0, 0.0};
    bool   have_prev_pred_ = false;

    mutable std::mutex state_mtx_;
    State last_state_;

    mutable std::mutex cfg_mtx_;
    Config cfg_;
    FrameRateCounter fps_counter_;
    std::atomic<uint64_t> ticks_since_set_{0};
    std::thread thread_;
    std::atomic<bool> running_{false};
};

} // namespace tcbs

#endif // TCBS_MCU_MPC_CONTROLLER_H