#include "tcbs/RobotController.h"

#include <cmath>

namespace tcbs {

RobotController::RobotController(const Config& cfg)
    : cfg_(cfg),
      sequence_mode_(cfg.sequence_mode),
      comm_(cfg.mcu_linear, cfg.estimator),
      mcu_mpc_(&comm_, cfg.model, cfg.mpc, cfg.controller) {
    mcu_mpc_.start();   // 启动后台线程（周期 = controller.loop_period）
}

RobotController::~RobotController() {
    // mcu_mpc_ 析构 stop + join；comm_ 析构停止串口线程
}

RobotController::State RobotController::getState() {
    State st;

    // ── MCU 原始反馈（已映射）──
    auto raw = comm_.getLatestData();
    if (raw.mcu_valid) {
        const auto& m = raw.mcu_packet;
        st.mcu.valid = true;
        st.mcu.bullet_velocity = m.bullet_velocity;
        st.mcu.pitch_angle = m.pitch_angle;
        st.mcu.yaw_big_angle = m.yaw_big_angle;
        st.mcu.yaw_big_omega = m.yaw_big_omega;
        st.mcu.yaw_small_angle = m.yaw_small_angle;
        st.mcu.yaw_small_omega = m.yaw_small_omega;
        st.mcu.chassis_imu_yaw = m.chassis_imu_yaw;
        st.mcu.chassis_imu_omega = m.chassis_imu_omega;
        st.mcu.mark = m.mark;
        st.mcu.color = m.color;
        st.mcu.auto_aim_switch = m.auto_aim_switch;
        st.mcu.yaw_big_temperature = m.yaw_big_temperature;
        st.mcu.yaw_small_temperature = m.yaw_small_temperature;
        st.mcu.mcu2_seq = raw.mcu2_seq;
    }
    if (raw.imu_valid) {
        const auto& im = raw.imu_packet;
        st.imu.valid = true;
        st.imu.gx = im.gx; st.imu.gy = im.gy; st.imu.gz = im.gz;
        st.imu.ax = im.ax; st.imu.ay = im.ay; st.imu.az = im.az;
        st.imu.euler_yaw = im.euler_yaw;
        st.imu.euler_pitch = im.euler_pitch;
        st.imu.euler_roll = im.euler_roll;
        st.imu.dt_one_tenth_ms = im.dt_one_tenth_ms;
    }

    // ── 状态估计 ──
    auto e = comm_.getEstimate();
    st.est.valid = e.valid;
    st.est.imu_yaw = e.imu_yaw;
    st.est.imu_pitch = e.imu_pitch;
    st.est.imu_roll = e.imu_roll;
    st.est.platform_azimuth = e.platform_azimuth;
    st.est.platform_rate = e.platform_rate;
    st.est.small_joint_angle = e.small_joint_angle;
    st.est.small_joint_rate = e.small_joint_rate;
    st.est.pitch_joint_angle = e.pitch_joint_angle;
    st.est.pitch_joint_rate = e.pitch_joint_rate;
    st.est.big_joint_angle_meas = e.big_joint_angle_meas;
    st.est.big_joint_angle = e.big_joint_angle;
    st.est.big_joint_rate = e.big_joint_rate;
    st.est.big_enc_age = e.big_enc_age;
    st.est.big_sample_interval = e.big_sample_interval;
    st.est.chassis_imu_age = e.chassis_imu_age;
    st.est.big_enc_innovation = e.big_enc_innovation;
    st.est.big_has_encoder = e.big_has_encoder;
    st.strict_pose = e.strict_pose;      // ★ 顶层并列的严格反解数据包（IMU 为准确值 → 反解底盘）
    st.est.head_world_yaw = e.head_world_yaw;
    st.est.head_world_pitch = e.head_world_pitch;
    st.est.head_world_roll = e.head_world_roll;
    st.est.small_output_azimuth = e.small_output_azimuth;
    st.est.los_azimuth = e.los_azimuth;
    st.est.los_elevation = e.los_elevation;
    st.est.chassis_azimuth = e.chassis_azimuth;
    st.est.chassis_yaw_rate = e.chassis_yaw_rate;
    for (int i = 0; i < 3; ++i) {
        st.est.base_omega[i] = e.base_omega[i];
        st.est.gravity_a[i] = e.gravity_a[i];
    }
    st.est.pitch_acc = e.pitch_acc;
    st.est.prov = e.prov;

    // ── MPC / 控制输出 ──
    auto m = mcu_mpc_.state();
    st.mpc.torque[0] = m.torque[0];
    st.mpc.torque[1] = m.torque[1];
    st.mpc.torque_mpc[0] = m.torque_mpc[0];
    st.mpc.torque_mpc[1] = m.torque_mpc[1];
    st.mpc.integral[0] = m.integral[0];
    st.mpc.integral[1] = m.integral[1];
    st.mpc.target_joint[0] = m.target_joint[0];
    st.mpc.target_joint[1] = m.target_joint[1];
    st.mpc.target_joint_rate[0] = m.target_joint_rate[0];
    st.mpc.target_joint_rate[1] = m.target_joint_rate[1];
    st.mpc.big_torque_only = m.big_torque_only;
    st.mpc.small_torque_only = m.small_torque_only;
    st.mpc.ref_azimuth[0] = m.ref_azimuth[0];
    st.mpc.ref_azimuth[1] = m.ref_azimuth[1];
    st.mpc.delayed_ref_azimuth[0] = m.delayed_ref_azimuth[0];
    st.mpc.delayed_ref_azimuth[1] = m.delayed_ref_azimuth[1];
    st.mpc.ref_azimuth_seq[0] = m.ref_azimuth_seq[0];
    st.mpc.ref_azimuth_seq[1] = m.ref_azimuth_seq[1];
    st.mpc.pred_azimuth_seq[0] = m.pred_azimuth_seq[0];
    st.mpc.pred_azimuth_seq[1] = m.pred_azimuth_seq[1];
    st.mpc.pred_joint_seq[0] = m.pred_joint_seq[0];
    st.mpc.pred_joint_seq[1] = m.pred_joint_seq[1];
    st.mpc.small_ref_over_limit = m.small_ref_over_limit;
    st.mpc.solve_ms = m.solve_ms;
    st.mpc.loop_fps = m.loop_fps;
    st.mpc.ticks_since_set = m.ticks_since_set;
    st.mpc.solve_count = m.solve_count;
    st.mpc.solve_fail_count = m.solve_fail_count;
    st.mpc.estimator_valid = m.estimator_valid;
    st.mpc.sent_ok = m.sent_ok;

    return st;
}

void RobotController::set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
                          double big_yaw_azimuth, double small_yaw_azimuth,
                          double pitch_target_angle, bool fire, bool integral_enable) {
    if (sequence_mode_) {
        throw std::runtime_error("RobotController: SEQUENCE mode selected, "
                                 "use sequence set() instead of single set()");
    }
    mcu_mpc_.set(auto_aim_enable, big_torque_only, small_torque_only,
                 big_yaw_azimuth, small_yaw_azimuth,
                 pitch_target_angle, fire, integral_enable);
}

void RobotController::set(bool auto_aim_enable, bool big_torque_only, bool small_torque_only,
                          const std::vector<double>& big_yaw_azimuth_seq,
                          const std::vector<double>& small_yaw_azimuth_seq,
                          const std::vector<double>& pitch_seq,
                          const std::vector<bool>& fire_seq, bool integral_enable) {
    if (!sequence_mode_) {
        throw std::runtime_error("RobotController: SINGLE mode selected, "
                                 "use single set() instead of sequence set()");
    }
    mcu_mpc_.set(auto_aim_enable, big_torque_only, small_torque_only,
                 big_yaw_azimuth_seq, small_yaw_azimuth_seq, pitch_seq, fire_seq,
                 integral_enable);
}

void RobotController::setJointAngles(bool auto_aim_enable, bool big_torque_only,
                                     bool small_torque_only,
                                     double big_joint_angle, double small_joint_angle,
                                     double pitch_target_angle, bool fire,
                                     bool integral_enable) {
    // 关节系 → 世界方位角: ψ_big = ψ_chassis + θ_big, ψ_small = ψ_big + θ_small
    const auto e = comm_.getEstimate();
    const double psi_big = e.chassis_azimuth + big_joint_angle;
    const double psi_small = psi_big + small_joint_angle;
    set(auto_aim_enable, big_torque_only, small_torque_only,
        psi_big, psi_small, pitch_target_angle, fire, integral_enable);
}

} // namespace tcbs
