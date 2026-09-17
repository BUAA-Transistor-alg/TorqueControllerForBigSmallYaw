// ============================================================================
// config_propagation.cpp — 验证「全部参数都能在 RobotController 构造时传入」
//
// 这不是 ctest 用例（不需要硬件），而是给"我把标定值填进去了，真的生效了吗"
// 这个问题的一个**可执行证据**: 用一个**逐项都改成非默认值**的 Config 构造，
// 再从各子系统的 getter 读回来逐项比对。
//
// 它是 ctest 用例（`tcbs_test_config_propagation`，无硬件）: 若将来有人往 Config
// 里加了字段却忘了往子系统里传，这里会直接失败。
// ============================================================================
#include "tcbs/RobotController.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace tcbs;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
    std::printf("  [%s] %-58s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    if (!ok) ++g_fail;
}

bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

}  // namespace

int main() {
    // ── 构造一个"每一项都不是默认值"的 Config ──
    RobotController::Config cfg;

    // ① 模型（实测几何 + 8 个辨识参数 + λ + 可选常数负载）
    cfg.model.dx = 0.0;
    cfg.model.dy = 0.07;
    cfg.model.gravity = 9.80665;
    cfg.model.m_u_known = 0.123;
    cfg.model.Jbig_eff = 0.0501;
    cfg.model.Js = 0.0204;
    cfg.model.Px = 0.00991;
    cfg.model.Py = 0.00402;
    cfg.model.fcBig = 0.2209;
    cfg.model.fvBig = 0.0525;
    cfg.model.fcSmall = 0.1010;
    cfg.model.fvSmall = 0.0180;
    cfg.model.frictionLambda = 100.0;
    cfg.model.tau_offset_big = 0.011;
    cfg.model.tau_offset_small = -0.007;

    // ② MPC（步长/N/子步/积分器/权重/力矩与限位）
    cfg.mpc.dt_control = 0.01;
    cfg.mpc.N = 16;
    cfg.mpc.substeps = 6;
    cfg.mpc.use_rk4 = false;
    cfg.mpc.max_iter = 11;
    cfg.mpc.w_big_azimuth = 2.0;
    cfg.mpc.w_small_azimuth = 3.0;
    cfg.mpc.w_small_center = 0.07;
    cfg.mpc.w_small_limit = 2e4;
    cfg.mpc.small_limit_soft_ratio = 0.7;
    cfg.mpc.r_big_torque = 0.02;
    cfg.mpc.r_small_torque = 0.03;
    cfg.mpc.rd_big_rate = 0.2;
    cfg.mpc.rd_small_rate = 0.3;
    cfg.mpc.smooth_eps = 1e-5;
    cfg.mpc.ref_delay_steps = 2;
    cfg.mpc.big.max_torque = 1.7;
    cfg.mpc.big.max_torque_rate = 55.0;
    cfg.mpc.small.max_torque = 0.8;
    cfg.mpc.small.max_torque_rate = 90.0;
    cfg.mpc.small.min_angle = -30.0 * M_PI / 180.0;
    cfg.mpc.small.max_angle =  30.0 * M_PI / 180.0;
    cfg.mpc.small_center_angle = 0.0;

    // ③ 编码器/指令映射（标定结果）
    cfg.mcu_linear.recv_pitch_scale = 1.005207;
    cfg.mcu_linear.recv_pitch_offset = -0.021541;
    cfg.mcu_linear.send_pitch_scale = 1.023322;
    cfg.mcu_linear.send_pitch_offset = 0.050772;
    cfg.mcu_linear.recv_small_yaw_scale = 1.0011;
    cfg.mcu_linear.recv_small_yaw_offset = -0.087393;
    cfg.mcu_linear.recv_big_yaw_scale = 1.0007;
    cfg.mcu_linear.recv_big_yaw_offset = 0.0033;
    cfg.mcu_linear.recv_big_omega_scale = 1.002;
    cfg.mcu_linear.recv_small_omega_scale = 0.998;
    cfg.mcu_linear.send_big_yaw_scale = 1.0;
    cfg.mcu_linear.send_big_yaw_offset = 0.0;
    cfg.mcu_linear.send_big_velocity_scale = 1.0;
    cfg.mcu_linear.send_big_torque_scale = 1.13;
    cfg.mcu_linear.send_small_yaw_scale = 1.0;
    cfg.mcu_linear.send_small_yaw_offset = 0.0;
    cfg.mcu_linear.send_small_velocity_scale = 1.0;
    cfg.mcu_linear.send_small_torque_scale = 0.97;

    // ④ 状态估计（IMU 位置/安装/延迟/视轴/滤波/门限）
    cfg.estimator.imu_location = YawStateEstimator::Config::ImuLocation::ON_HEAD;
    cfg.estimator.mount_yaw = 0.011;
    cfg.estimator.mount_pitch = -0.021;
    cfg.estimator.mount_roll = 0.032;
    cfg.estimator.head_mount_yaw = 0.041;
    cfg.estimator.head_mount_pitch = -0.052;
    cfg.estimator.head_mount_roll = 0.063;
    cfg.estimator.transport_delay_s = 0.017;
    cfg.estimator.big_enc_max_jump = 0.25;
    cfg.estimator.stale_age_s = 0.40;
    cfg.estimator.chassis_imu_timeout_s = 1.5;
    cfg.estimator.max_extrap_s = 0.06;
    cfg.estimator.rate_lpf_alpha = 0.30;
    cfg.estimator.pitch_rate_lpf_alpha = 0.20;
    cfg.estimator.pitch_acc_lpf_alpha = 0.10;
    cfg.estimator.bore[0] = 0.02; cfg.estimator.bore[1] = 0.999; cfg.estimator.bore[2] = 0.01;
    cfg.estimator.gravity = 9.80665;
    cfg.estimator.use_chassis_imu = false;
    cfg.estimator.source_timeout_s = 0.6;

    // ⑤ 控制器（后台 loop + 模式位 + 积分增益）
    cfg.controller.loop_period = 0.02;
    cfg.controller.big_torque_only = true;
    cfg.controller.small_torque_only = true;
    cfg.controller.ref_delay_steps = 1;
    cfg.controller.integral_gain[0] = 0.02;
    cfg.controller.integral_gain[1] = 0.03;
    cfg.controller.integral_limit[0] = 0.25;
    cfg.controller.integral_limit[1] = 0.35;
    // ★ 大 yaw 积分被 `integral_on_big` 门控（默认 false ⇒ 构造时把 gain[0] 清零）:
    //   要真正启用大 yaw 积分必须同时置 true，否则上面那行 gain[0] 会被静默清零。
    cfg.controller.integral_on_big = true;

    // ⑥ 序列模式开关
    cfg.sequence_mode = true;

    std::printf("=== 用逐项非默认的 Config 构造 RobotController（无硬件）===\n\n");
    RobotController rc(cfg);

    const auto m = rc.modelParams();
    const auto mp = rc.mcuMpc().mpc().config();
    const auto lin = rc.communication().preprocessor().params();
    const auto est = rc.estimator().config();
    const auto ctl = rc.mcuMpc().config();

    std::printf("[① 模型 model]\n");
    check(near(m.dx, 0.0) && near(m.dy, 0.07), "几何 d = (0, 0.07)");
    check(near(m.gravity, 9.80665), "gravity");
    check(near(m.m_u_known, 0.123), "m_u_known");
    check(near(m.Jbig_eff, 0.0501) && near(m.Js, 0.0204), "Jbig_eff / Js");
    check(near(m.Px, 0.00991) && near(m.Py, 0.00402), "Px / Py");
    check(near(m.fcBig, 0.2209) && near(m.fvBig, 0.0525), "fc_big / fv_big");
    check(near(m.fcSmall, 0.1010) && near(m.fvSmall, 0.0180), "fc_small / fv_small");
    check(near(m.frictionLambda, 100.0), "frictionLambda");
    check(near(m.tau_offset_big, 0.011) && near(m.tau_offset_small, -0.007), "tau_offset_big/small");

    std::printf("\n[② MPC 配置]\n");
    check(near(mp.dt_control, 0.01) && mp.N == 16 && mp.substeps == 6, "dt_control / N=16 / substeps=6");
    check(mp.use_rk4 == false && mp.max_iter == 11, "use_rk4=false / max_iter=11");
    check(near(mp.w_big_azimuth, 2.0) && near(mp.w_small_azimuth, 3.0), "w_big/w_small_azimuth");
    check(near(mp.w_small_center, 0.07) && near(mp.w_small_limit, 2e4), "w_small_center / w_small_limit");
    check(near(mp.small_limit_soft_ratio, 0.7), "small_limit_soft_ratio");
    check(near(mp.r_big_torque, 0.02) && near(mp.r_small_torque, 0.03), "r_*_torque");
    check(near(mp.rd_big_rate, 0.2) && near(mp.rd_small_rate, 0.3), "rd_*_rate");
    check(near(mp.smooth_eps, 1e-5) && mp.ref_delay_steps == 2, "smooth_eps / ref_delay_steps");
    check(near(mp.big.max_torque, 1.7) && near(mp.big.max_torque_rate, 55.0), "big 力矩与变化率");
    check(near(mp.small.max_torque, 0.8) && near(mp.small.max_torque_rate, 90.0), "small 力矩与变化率");
    check(near(mp.small.min_angle, -30.0 * M_PI / 180.0) &&
          near(mp.small.max_angle, 30.0 * M_PI / 180.0), "小 yaw 行程 [-30°, +30°]");
    check(near(mp.small_center_angle, 0.0), "small_center_angle = 0");

    std::printf("\n[③ 编码器/指令映射 mcu_linear]\n");
    check(near(lin.recv_pitch_scale, 1.005207) && near(lin.recv_pitch_offset, -0.021541), "recv_pitch_*");
    check(near(lin.send_pitch_scale, 1.023322) && near(lin.send_pitch_offset, 0.050772), "send_pitch_*");
    check(near(lin.recv_small_yaw_scale, 1.0011) && near(lin.recv_small_yaw_offset, -0.087393),
          "recv_small_yaw_*（零点标定结果）");
    check(near(lin.recv_big_yaw_scale, 1.0007) && near(lin.recv_big_yaw_offset, 0.0033), "recv_big_yaw_*");
    check(near(lin.recv_big_omega_scale, 1.002) && near(lin.recv_small_omega_scale, 0.998), "recv_*_omega_scale");
    check(near(lin.send_big_torque_scale, 1.13) && near(lin.send_small_torque_scale, 0.97), "send_*_torque_scale");

    std::printf("\n[④ 状态估计 estimator]\n");
    check(est.imu_location == YawStateEstimator::Config::ImuLocation::ON_HEAD, "imu_location = ON_HEAD");
    check(near(est.mount_yaw, 0.011) && near(est.mount_pitch, -0.021) && near(est.mount_roll, 0.032), "mount_*");
    check(near(est.head_mount_yaw, 0.041) && near(est.head_mount_pitch, -0.052) &&
          near(est.head_mount_roll, 0.063), "head_mount_*");
    check(near(est.transport_delay_s, 0.017), "transport_delay_s");
    check(near(est.big_enc_max_jump, 0.25) && near(est.stale_age_s, 0.40), "big_enc_max_jump / stale_age_s");
    check(near(est.chassis_imu_timeout_s, 1.5) && near(est.max_extrap_s, 0.06), "chassis_imu_timeout / max_extrap");
    check(near(est.rate_lpf_alpha, 0.30) && near(est.pitch_rate_lpf_alpha, 0.20) &&
          near(est.pitch_acc_lpf_alpha, 0.10), "三个 LPF alpha");
    check(near(est.bore[0], 0.02) && near(est.bore[1], 0.999) && near(est.bore[2], 0.01), "bore[3]");
    check(near(est.gravity, 9.80665), "重力");
    check(est.use_chassis_imu == false && near(est.source_timeout_s, 0.6), "use_chassis_imu / source_timeout_s");

    std::printf("\n[⑤ 控制器 controller]\n");
    check(near(ctl.loop_period, 0.02), "loop_period");
    check(ctl.big_torque_only == true && ctl.small_torque_only == true, "模式位（仅力矩）");
    check(ctl.ref_delay_steps == 1, "ref_delay_steps");
    check(near(ctl.integral_gain[0], 0.02) && near(ctl.integral_gain[1], 0.03), "integral_gain[2]");
    check(near(ctl.integral_limit[0], 0.25) && near(ctl.integral_limit[1], 0.35), "integral_limit[2]");
    check(ctl.integral_on_big == true, "integral_on_big = true");

    std::printf("\n[⑥ 序列模式]\n");
    check(rc.mode() == RobotController::Mode::SEQUENCE, "sequence_mode = true ⇒ Mode::SEQUENCE");

    std::printf("\n[⑦ integral_on_big 门控语义（构造与运行时设置必须一致）]\n");
    {
        RobotController::Config c2;
        c2.controller.integral_gain[0] = 0.05;        // 故意开大 yaw 积分但**不**开闸
        c2.controller.integral_gain[1] = 0.03;
        RobotController rc2(c2);
        const auto k2 = rc2.mcuMpc().config();
        check(near(k2.integral_gain[0], 0.0),
              "构造: integral_on_big=false ⇒ gain[0] 被清零",
              "gain[0]=" + std::to_string(k2.integral_gain[0]));
        check(near(k2.integral_gain[1], 0.03), "构造: 小 yaw 积分不受门控", "gain[1]=" + std::to_string(k2.integral_gain[1]));

        // 运行时路径必须给出同样的结果（否则"构造时传"和"以后 setControllerConfig"语义不一致）
        RobotController::Config c3 = c2;
        c3.controller.integral_on_big = true;
        RobotController rc3;
        rc3.setControllerConfig(c3.controller);
        check(near(rc3.mcuMpc().config().integral_gain[0], 0.05),
              "setControllerConfig: integral_on_big=true ⇒ gain[0] 保留",
              "gain[0]=" + std::to_string(rc3.mcuMpc().config().integral_gain[0]));
        RobotController rc4;
        rc4.setControllerConfig(c2.controller);   // 不设 true 的那份
        check(near(rc4.mcuMpc().config().integral_gain[0], 0.0),
              "setControllerConfig: integral_on_big=false ⇒ gain[0] 清零（与构造一致）",
              "gain[0]=" + std::to_string(rc4.mcuMpc().config().integral_gain[0]));
    }

    std::printf("\n[⑧ 默认构造 = 各 default*() 的值]\n");
    {
        RobotController rcd;   // 默认 Config
        const auto dm = rcd.modelParams();
        const auto dm_ref = dual_yaw::defaultModelParams();
        check(near(dm.dx, dm_ref.dx) && near(dm.dy, dm_ref.dy) && near(dm.Jbig_eff, dm_ref.Jbig_eff) &&
              near(dm.fcSmall, dm_ref.fcSmall), "默认 = defaultModelParams()");
        check(near(dm.dx, 0.0) && near(dm.dy, 0.07), "默认几何 = 实测 (0, 0.07)");
        check(rcd.mode() == RobotController::Mode::SINGLE, "默认 = Mode::SINGLE");
    }

    std::printf("\n=== %s（%d 项失败）===\n", g_fail == 0 ? "全部通过 ✔" : "有失败 ✘", g_fail);
    return g_fail == 0 ? 0 : 1;
}
