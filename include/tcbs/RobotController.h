#ifndef TCBS_ROBOT_CONTROLLER_H
#define TCBS_ROBOT_CONTROLLER_H

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tcbs/communication/Communications.hpp"
#include "tcbs/mpc/planar_yaw_params.h"
#include "tcbs/mpc/mcu_mpc_controller.h"

namespace tcbs {

// ============================================================================
// RobotController — 双级 yaw 云台一体化控制封装（对外主接口）
//
// 内部:
//   - RobotCommunication: MCU/IMU 串口 + 编码器映射（McuDataPreprocessor）
//     + 状态估计（YawStateEstimator: IMU 在大 yaw 转子上）
//   - McuMpcController: 耦合非线性 MPC + 后台发送线程（周期 = loop_period）
//
// 对外接口语义（**世界系方位角**，两轴分别设置）:
//   big_yaw_azimuth   : 大 yaw 平台 x 轴的世界方位角 ψ_big（多圈连续，IMU 直测）
//   small_yaw_azimuth : 小 yaw 输出 x 轴的世界方位角 ψ_small = ψ_big + θ_small
//   两者之差即小 yaw 关节角指令；调用方决定快慢分配与展开策略。
//   另有 setJointAngles() 便捷接口（关节系输入，内部按当前底盘方位角换算）。
//
// getState() 返回按来源分组的完整状态:
//   mcu  : MCU 原始反馈（已按映射参数换算）
//   imu  : 大 yaw 上 IMU 的原始数据
//   est  : 状态估计（可信实时量 / 大 yaw 延迟补偿估计 / 反解真实位姿 / 数据来源）
//   mpc  : 控制输出、参考与预测序列、求解性能
// ============================================================================
class RobotController {
public:
    // 构造参数集合（避免长参数列表；标定后逐项替换）
    struct Config {
        dual_yaw::ModelParams              model      = dual_yaw::defaultModelParams();
        dual_yaw::DualYawMpcConfig         mpc        = dual_yaw::defaultMpcConfig();
        McuDataPreprocessor::LinearParams  mcu_linear = McuDataPreprocessor::LinearParams{};
        YawStateEstimator::Config          estimator  = YawStateEstimator::Config{};
        McuMpcController::Config           controller = McuMpcController::Config{};
        // 序列模式: 构造时选定；单目标模式下调用序列 set() 会抛 std::runtime_error
        bool                               sequence_mode = false;
    };

    // ── MCU 原始反馈（已映射）──
    struct McuData {
        bool    valid = false;
        float   bullet_velocity = 0.0f;
        float   pitch_angle = 0.0f;      // 已映射
        double  yaw_big_angle = 0.0;     // 延迟/带误差
        float   yaw_big_omega = 0.0f;
        float   yaw_small_angle = 0.0f;  // 可信
        float   yaw_small_omega = 0.0f;
        float   chassis_imu_yaw = 0.0f;
        float   chassis_imu_omega = 0.0f;
        uint8_t mark = 0, color = 0, auto_aim_switch = 0;
        uint8_t yaw_big_temperature = 0, yaw_small_temperature = 0;
        // MCU2 数据的新样本序号（大 yaw 与底盘 IMU 同源共用；值被保持时不变）
        uint8_t mcu2_seq = 0;
    };

    // ── IMU 原始数据（大 yaw 上）──
    struct ImuData {
        bool  valid = false;
        float gx = 0.0f, gy = 0.0f, gz = 0.0f;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        double euler_yaw = 0.0, euler_pitch = 0.0, euler_roll = 0.0;
        uint32_t dt_one_tenth_ms = 0;
    };

    // ── 状态估计（可信量 + 延迟补偿估计 + 反解真实位姿 + 数据来源）──
    struct EstData {
        bool   valid = false;
        // 可信实时量
        double imu_yaw = 0.0, imu_pitch = 0.0, imu_roll = 0.0;
        double platform_azimuth = 0.0;      // ψ_big（IMU 反解）
        double platform_rate = 0.0;
        double small_joint_angle = 0.0;
        double small_joint_rate = 0.0;
        double pitch_joint_angle = 0.0;
        double pitch_joint_rate = 0.0;
        // 大 yaw（延迟编码器 + IMU 速率 → 延迟补偿）
        double big_joint_angle_meas = 0.0;
        double big_joint_angle = 0.0;
        double big_joint_rate = 0.0;
        // ── ★ 大 yaw 电机侧 / 云台侧 分离（背隙建模用）──
        double big_motor_angle = 0.0;      // 电机侧关节角（MCU 编码器）
        double big_motor_rate = 0.0;       // 电机侧角速度
        double big_platform_angle = 0.0;   // 云台侧关节角 θ_p
        double big_platform_rate = 0.0;    // 云台侧角速度
        double backlash_center = 0.0;      // β（在线估计，加在云台角上）
        double backlash_width_obs = 0.0;   // 观测到的 Δ 极差
        double big_enc_age = -1.0;          // 大 yaw 值的年龄（上位机计时）
        double big_sample_interval = 0.0;   // 最近两次新样本间隔（s）→ MCU1↔MCU2 链路状况
        double chassis_imu_age = -1.0;      // 底盘 IMU 值的年龄（上位机计时）
        double big_enc_innovation = 0.0;
        bool   big_has_encoder = false;
        // 反解真实位姿（可信量 + 标定参数）
        double head_world_yaw = 0.0, head_world_pitch = 0.0, head_world_roll = 0.0;
        double small_output_azimuth = 0.0;  // ψ_small
        double los_azimuth = 0.0, los_elevation = 0.0;
        double chassis_azimuth = 0.0;
        double chassis_yaw_rate = 0.0;
        // 模型外生量
        double base_omega[3] = {0.0, 0.0, 0.0};
        double gravity_a[3] = {0.0, 0.0, -9.81};  // 重力矢量（大 yaw 转子 A 系）
        double pitch_acc = 0.0;
        // 数据来源（“所用数据”）
        YawStateEstimator::Provenance prov;
    };

    // ── MPC / 控制输出 ──
    struct MpcData {
        double torque[2] = {0.0, 0.0};
        double torque_mpc[2] = {0.0, 0.0};
        double integral[2] = {0.0, 0.0};
        double target_joint[2] = {0.0, 0.0};        // 发送给 MCU 的 θ*
        double target_joint_rate[2] = {0.0, 0.0};   // 发送给 MCU 的 ω*
        bool   big_torque_only = false, small_torque_only = false;
        double ref_azimuth[2] = {0.0, 0.0};
        double delayed_ref_azimuth[2] = {0.0, 0.0};
        std::vector<double> ref_azimuth_seq[2];
        std::vector<double> pred_azimuth_seq[2];
        std::vector<double> pred_joint_seq[2];
        bool   small_ref_over_limit = false;
        double solve_ms = 0.0;
        double loop_fps = 0.0;
        uint64_t ticks_since_set = 0;
        uint32_t solve_count = 0;
        uint32_t solve_fail_count = 0;
        bool   estimator_valid = false;
        bool   sent_ok = false;
    };

    struct State {
        McuData mcu;
        ImuData imu;
        EstData est;
        // ★ 严格反解数据包（与 mcu/imu/est/mpc **并列**，独立于 EstData）:
        //   以 IMU 数据为准确值反解**底盘**姿态，并打包反解用到的全部数据，
        //   外部只凭这一包即可重构整车姿态；忽略浮点误差时重构出的 IMU 姿态 == IMU 实际数据。
        //   与原仓库一致: **没有 valid 标志、始终解算**，缺失值用历史值或 0 参与，重构关系恒成立。
        dual_yaw::StrictPose strict_pose;
        MpcData mpc;
    };

    enum class Mode { SINGLE = 0, SEQUENCE = 1 };

    static Config defaultConfig() { return Config{}; }
    explicit RobotController(const Config& cfg = defaultConfig());
    ~RobotController();

    // ── 目标设置（世界方位角语义）──
    // big_yaw_azimuth / small_yaw_azimuth: 世界系方位角（rad，多圈连续，解卷绕语义）
    // big_torque_only / small_torque_only: 逐关节模式位（true = 电控仅施加力矩）
    // integral_enable: 是否启用积分补偿（false 时积分清零）
    void set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
             double big_yaw_azimuth, double small_yaw_azimuth,
             double pitch_target_angle, bool fire, bool integral_enable = false);

    // 序列版（各通道独立，不截断）
    void set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
             const std::vector<double>& big_yaw_azimuth_seq,
             const std::vector<double>& small_yaw_azimuth_seq,
             const std::vector<double>& pitch_seq,
             const std::vector<bool>& fire_seq,
             bool integral_enable = false);

    // 便捷接口: 以**关节系**角度设置（内部按当前底盘方位角估计换算为世界方位角）
    //   big_joint_angle : 大 yaw 关节角（相对底盘，多圈）
    //   small_joint_angle: 小 yaw 关节角（相对大 yaw，行程 ±30°，见 defaultMpcConfig()）
    void setJointAngles(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
                        double big_joint_angle, double small_joint_angle,
                        double pitch_target_angle, bool fire, bool integral_enable = false);

    // 统一状态获取（线程安全）
    State getState();

    Mode mode() const { return sequence_mode_ ? Mode::SEQUENCE : Mode::SINGLE; }

    // 子系统访问（标定/调试用）
    RobotCommunication& communication() { return comm_; }
    McuMpcController& mcuMpc() { return mcu_mpc_; }
    YawStateEstimator& estimator() { return comm_.estimator(); }
    const dual_yaw::ModelParams& modelParams() const { return cfg_.model; }
    void setModelParams(const dual_yaw::ModelParams& m) { cfg_.model = m; mcu_mpc_.mpc().setModel(m); }
    void setMpcConfig(const dual_yaw::DualYawMpcConfig& c) { cfg_.mpc = c; mcu_mpc_.mpc().setConfig(c); }
    // 控制器配置（模式位/积分增益/loop 周期）: 运行时生效，不重启后台线程
    void setControllerConfig(const McuMpcController::Config& c) { cfg_.controller = c; mcu_mpc_.setConfig(c); }
    void setLinearParams(const McuDataPreprocessor::LinearParams& p) { comm_.setLinearParams(p); }
    void setEstimatorConfig(const YawStateEstimator::Config& c) { comm_.estimator().setConfig(c); }

private:
    Config cfg_;
    bool sequence_mode_ = false;
    RobotCommunication comm_;
    McuMpcController   mcu_mpc_;
};

} // namespace tcbs

#endif // TCBS_ROBOT_CONTROLLER_H