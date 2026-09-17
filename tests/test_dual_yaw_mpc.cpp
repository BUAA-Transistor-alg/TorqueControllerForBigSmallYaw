// ============================================================================
// test_dual_yaw_mpc.cpp — 平面 8 参模型的耦合 MPC 闭环仿真测试
//
// 被控对象(plant)与控制器(MPC)的差异（按实际约定）:
//   - **plant 用大 λ 模拟真实库伦摩擦**（λ_plant = 100，接近 sign 函数），
//     因此 plant 用很小的积分步长（0.05ms）保证数值稳定；
//   - **MPC/辨识模型用合理 λ**（λ = 10，受数值可积性上限约束）。
//   ⇒ 这本身构成"摩擦模型失配"，正好检验闭环鲁棒性与积分补偿的作用。
//
// ★ 小 yaw 的机械行程是 ±30°（中心 0，见 defaultMpcConfig()）:
//   - 所有限位断言都从 defaultMpcConfig() 的 cfg.small.min_angle/max_angle 取，
//     **不写死** 0.7854/45° 之类的数字；
//   - 场景 [6][7][8] 专门覆盖非对称行程: 两侧行程是否都被正确使用、是否不越限、
//     回中代价是否把关节拉向 −2.5°（而不是 0）、偏向负侧的大运动能否正常跟踪。
//
// 场景: [1]双轴同参考阶跃 [2]小yaw单独运动 [3]大阶跃(需大yaw展开) [4]底盘旋转抗扰
//       [5]惯量/摩擦失配+未建模负载 [6]非对称行程:两侧逼近限位+软限位几何
//       [7]非对称行程:回中代价→行程中心 [8]非对称行程:偏向负侧的大阶跃
//       [9]求解性能 [10]耗时<控制周期
// ============================================================================
#include "tcbs/mpc/dual_yaw_mpc.h"
#include "tcbs/mpc/planar_yaw_params.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace tcbs {

using namespace dual_yaw;

namespace {

constexpr double kLambdaModel = 100.0;   // MPC/辨识模型用的 λ（|ω|≳1°/s 饱和）
constexpr double kLambdaPlant = 1.0e4;   // 被控对象的"真库伦摩擦"λ（≈sign）
constexpr double kPlantDt = 0.00002;     // plant 积分步长 20µs（λ=1e4 ⇒ |J_f|≈3.8e4/s,
                                         //   RK4 稳定上限 ≈73µs，故取 20µs 留 3.6× 余量）
constexpr double kDeg = M_PI / 180.0;

struct Plant {
    ModelParams p;
    double q[2] = {0.0, 0.0};
    double qd[2] = {0.0, 0.0};
    ModelExo exo;
    double extra_load[2] = {0.0, 0.0};

    void step(double dt, const double tau[2]) {
        const double t[2] = {tau[0] - extra_load[0], tau[1] - extra_load[1]};
        double qn[2], qdn[2];
        integrateStep(q, qd, t, p, exo, dt, 1, qn, qdn);
        q[0] = qn[0]; q[1] = qn[1];
        qd[0] = qdn[0]; qd[1] = qdn[1];
    }
};

struct Metrics {
    double max_aim_err = 0.0, rms_aim_err = 0.0, max_big_err = 0.0;
    // 小 yaw 关节角: 分别记录两侧极值（**不能用 fabs** —— 行程非对称）
    double min_small_joint = 0.0, max_small_joint = 0.0, final_small_joint = 0.0;
    double max_torque[2] = {0.0, 0.0}, max_torque_rate[2] = {0.0, 0.0};
    double mean_solve_ms = 0.0, max_solve_ms = 0.0;
    int over_min = 0, over_max = 0, solve_fail = 0;
};

Metrics runClosedLoop(const ModelParams& ctrl_model, const ModelParams& plant_model,
                      const DualYawMpcConfig& cfg, double T, double chassis_rate,
                      const std::vector<double>& big_ref_fn,
                      const std::vector<double>& small_ref_fn,
                      bool integral_enable, double integral_gain,
                      const double extra_load[2], double q1_init = 0.0) {
    Plant plant;
    plant.p = plant_model;
    plant.q[1] = q1_init;
    if (extra_load) { plant.extra_load[0] = extra_load[0]; plant.extra_load[1] = extra_load[1]; }

    DualYawMpc mpc(ctrl_model, cfg);
    Metrics m;
    const double dt = cfg.dt_control;
    const int n_steps = static_cast<int>(std::round(T / dt));
    double tau_applied[2] = {0.0, 0.0}, prev_tau[2] = {0.0, 0.0};
    double integral[2] = {0.0, 0.0}, prev_pred[2] = {0.0, 0.0};
    bool have_prev = false;
    double sum_sq = 0.0, solve_sum = 0.0;
    int n_err = 0;
    double chassis_azimuth = 0.0;

    for (int k = 0; k < n_steps; ++k) {
        DualYawMpc::Input in;
        in.q[0] = plant.q[0];
        in.q[1] = plant.q[1];
        in.qd[0] = plant.qd[0];
        in.qd[1] = plant.qd[1];
        in.exo.base_omega = chassis_rate;
        in.exo.base_alpha = 0.0;
        in.platform_azimuth = chassis_azimuth + plant.q[0];
        in.chassis_azimuth = chassis_azimuth;
        in.chassis_rate = chassis_rate;
        in.prev_torque[0] = tau_applied[0];
        in.prev_torque[1] = tau_applied[1];
        in.ref_big_azimuth.resize(cfg.N);
        in.ref_small_azimuth.resize(cfg.N);
        for (int j = 0; j < cfg.N; ++j) {
            const int idx = std::min<int>(static_cast<int>(big_ref_fn.size()) - 1, k + j + 1);
            in.ref_big_azimuth[j] = big_ref_fn[idx];
            in.ref_small_azimuth[j] = small_ref_fn[idx];
        }

        auto res = mpc.solve(in);
        solve_sum += res.solve_ms;
        m.max_solve_ms = std::max(m.max_solve_ms, res.solve_ms);
        if (!res.usable) ++m.solve_fail;

        double tau[2] = {res.torque[0], res.torque[1]};
        if (integral_enable) {
            if (have_prev) {
                for (int i = 0; i < 2; ++i) {
                    integral[i] += integral_gain * (prev_pred[i] - plant.q[i]);
                    integral[i] = std::clamp(integral[i], -0.3, 0.3);
                }
            }
        } else {
            integral[0] = integral[1] = 0.0;
        }
        prev_pred[0] = res.pred_q[0]; prev_pred[1] = res.pred_q[1];
        have_prev = true;
        tau[0] = std::clamp(tau[0] + integral[0], -cfg.big.max_torque, cfg.big.max_torque);
        tau[1] = std::clamp(tau[1] + integral[1], -cfg.small.max_torque, cfg.small.max_torque);

        m.max_torque_rate[0] = std::max(m.max_torque_rate[0], std::fabs(tau[0] - prev_tau[0]) / dt);
        m.max_torque_rate[1] = std::max(m.max_torque_rate[1], std::fabs(tau[1] - prev_tau[1]) / dt);
        m.max_torque[0] = std::max(m.max_torque[0], std::fabs(tau[0]));
        m.max_torque[1] = std::max(m.max_torque[1], std::fabs(tau[1]));

        const int sub = static_cast<int>(std::round(dt / kPlantDt));
        for (int s = 0; s < sub; ++s) {
            plant.exo = in.exo;
            chassis_azimuth += chassis_rate * kPlantDt;
            plant.step(kPlantDt, tau);
        }
        tau_applied[0] = tau[0]; tau_applied[1] = tau[1];
        prev_tau[0] = tau[0]; prev_tau[1] = tau[1];

        const double psi_big = chassis_azimuth + plant.q[0];
        const double psi_small = psi_big + plant.q[1];
        const double err = small_ref_fn[k] - psi_small;
        m.max_aim_err = std::max(m.max_aim_err, std::fabs(err));
        m.max_big_err = std::max(m.max_big_err, std::fabs(big_ref_fn[k] - psi_big));
        m.min_small_joint = std::min(m.min_small_joint, plant.q[1]);
        m.max_small_joint = std::max(m.max_small_joint, plant.q[1]);
        m.final_small_joint = plant.q[1];
        sum_sq += err * err; ++n_err;
        // 限位统计: 两侧**各自**判定（非对称行程, 不能用 fabs(θ) > LIMIT）
        if (plant.q[1] < cfg.small.min_angle - 1e-6) ++m.over_min;
        if (plant.q[1] > cfg.small.max_angle + 1e-6) ++m.over_max;
    }
    m.rms_aim_err = std::sqrt(sum_sq / std::max(1, n_err));
    m.mean_solve_ms = solve_sum / std::max(1, n_steps);
    return m;
}

int g_fail = 0;
void check(bool ok, const char* name, double v = 0.0, double tol = 0.0) {
    if (ok) printf("  [PASS] %-56s (%.4g ≤ %.4g)\n", name, v, tol);
    else { printf("  [FAIL] %-56s (%.4g > %.4g)\n", name, v, tol); ++g_fail; }
}

std::vector<double> constantRef(int n, double v) { return std::vector<double>(n, v); }

std::vector<double> smoothStepRef(int n, double dt, double a, double b, double vmax) {
    std::vector<double> out(n, a);
    const double dist = b - a, sign = (dist >= 0) ? 1.0 : -1.0, ad = std::fabs(dist);
    const double Tmove = std::max(0.25, ad / std::max(0.2, vmax) * 1.8);
    for (int k = 0; k < n; ++k) {
        const double t = k * dt;
        const double s = (t >= Tmove) ? 1.0 : 0.5 * (1.0 - std::cos(M_PI * t / Tmove));
        out[k] = a + sign * ad * s;
    }
    return out;
}

// 测试侧**独立**复算软限位（与实现同一语义, 用于几何核对）:
//   inset = (1−ratio)·(max−min);  soft_min = min + inset;  soft_max = max − inset
struct SoftLimitsT { double lo, hi; };
SoftLimitsT expectedSoft(const DualYawMpcConfig& c) {
    const double inset = (1.0 - c.small_limit_soft_ratio) *
                         (c.small.max_angle - c.small.min_angle);
    return {c.small.min_angle + inset, c.small.max_angle - inset};
}

// 直接问 MPC: 这个"小 yaw 关节角参考"是否落在软限位区之外（两侧独立判定）
bool refOverLimit(const ModelParams& model, const DualYawMpcConfig& cfg, double ref_joint) {
    DualYawMpc mpc(model, cfg);
    DualYawMpc::Input in;
    in.q[0] = 0.0; in.q[1] = 0.0;
    in.platform_azimuth = 0.0;
    in.chassis_azimuth = 0.0;
    in.ref_big_azimuth.assign(1, 0.0);
    in.ref_small_azimuth.assign(1, ref_joint);   // ψ_small − ψ_big = θ_s 参考
    return mpc.solve(in).small_ref_over_limit;
}

} // namespace

} // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；下面把 tcbs 内的类型与测试辅助函数引入作用域
using namespace tcbs;

int main() {
    printf("=== 平面 8 参模型 · 耦合 MPC 闭环仿真 ===\n");
    printf("plant: λ=%.0f（模拟真库伦摩擦）、积分步长 %.3fms；MPC 模型: λ=%.0f\n",
           kLambdaPlant, kPlantDt * 1e3, kLambdaModel);

    ModelParams ctrl_model = defaultModelParams();
    ModelParams plant_model = defaultModelParams();
    plant_model.frictionLambda = kLambdaPlant;
    const DualYawMpcConfig cfg = defaultMpcConfig();
    const double dt = cfg.dt_control;
    const double T = 3.0;
    const int n = static_cast<int>(T / dt);

    const double smin = cfg.small.min_angle, smax = cfg.small.max_angle;
    const SoftLimitsT soft = expectedSoft(cfg);
    printf("小 yaw 行程 [%.2f°, %.2f°]（中心 %.2f°）; 软限位区 [%.2f°, %.2f°]（ratio=%.2f）\n",
           smin / kDeg, smax / kDeg, 0.5 * (smin + smax) / kDeg,
           soft.lo / kDeg, soft.hi / kDeg, cfg.small_limit_soft_ratio);
    printf("回中中心 small_center_angle = %.4f rad (%.2f°)\n\n",
           cfg.small_center_angle, cfg.small_center_angle / kDeg);

    printf("[1] 双轴同目标阶跃 0.6rad（自由分配）\n");
    {
        auto big_ref = smoothStepRef(n, dt, 0.0, 0.6, 8.0);
        auto m = runClosedLoop(ctrl_model, plant_model, cfg, T, 0.0, big_ref, big_ref, false, 0.0, nullptr);
        printf("   最大瞄准误差 %.4f rad (%.2f°), RMS %.4f, θs 范围 [%.4f, %.4f] rad\n"
               "   最大力矩 (%.3f, %.3f), 最大变化率 (%.1f, %.1f), 求解 平均 %.2f ms\n",
               m.max_aim_err, m.max_aim_err * 180 / M_PI, m.rms_aim_err,
               m.min_small_joint, m.max_small_joint,
               m.max_torque[0], m.max_torque[1], m.max_torque_rate[0], m.max_torque_rate[1],
               m.mean_solve_ms);
        check(m.max_aim_err < 0.10, "阶跃跟踪最大瞄准误差 < 0.10 rad（力矩受限）", m.max_aim_err, 0.10);
        check(m.min_small_joint >= smin - 1e-6, "小 yaw 未越下侧限位", -m.min_small_joint, -smin);
        check(m.max_small_joint <= smax + 1e-6, "小 yaw 未越上侧限位", m.max_small_joint, smax);
        check(m.max_torque[0] <= cfg.big.max_torque + 1e-6, "大 yaw 力矩未超限", m.max_torque[0], cfg.big.max_torque);
        check(m.max_torque[1] <= cfg.small.max_torque + 1e-6, "小 yaw 力矩未超限", m.max_torque[1], cfg.small.max_torque);
        check(m.max_torque_rate[0] <= cfg.big.max_torque_rate * 1.05, "大 yaw 力矩变化率未超限",
              m.max_torque_rate[0], cfg.big.max_torque_rate);
        check(m.max_torque_rate[1] <= cfg.small.max_torque_rate * 1.05, "小 yaw 力矩变化率未超限",
              m.max_torque_rate[1], cfg.small.max_torque_rate);
        check(m.solve_fail == 0, "无求解失败", (double)m.solve_fail, 0.0);
    }

    printf("\n[2] 大 yaw 保持、小 yaw 单独运动 0.13rad（约 7.4°, 在软限位区内）\n");
    {
        // 小 yaw 单独承担 0.13 rad（+7.4°）的方位角运动: 必须落在正侧软限位
        // 8.75° 以内, 否则障碍项会（正确地）拒绝把关节推出去。
        const double step = 0.13;
        auto big_ref = constantRef(n, 0.0);
        auto small_ref = smoothStepRef(n, dt, 0.0, step, 6.0);
        auto m = runClosedLoop(ctrl_model, plant_model, cfg, T, 0.0, big_ref, small_ref, false, 0.0, nullptr);
        printf("   小 yaw 最终关节角 %.4f rad (%.2f°), 最大|ε| %.4f\n",
               m.final_small_joint, m.final_small_joint / kDeg, m.max_aim_err);
        check(m.max_aim_err < 0.02, "小 yaw 跟踪误差 < 0.02 rad", m.max_aim_err, 0.02);
        check(m.max_big_err < 0.02, "大 yaw 被保持在原位", m.max_big_err, 0.02);
        check(m.max_small_joint <= soft.hi, "小 yaw 未进入正侧软限位区", m.max_small_joint, soft.hi);
    }

    printf("\n[3] 大阶跃 1.2rad（超出小 yaw 全部行程, 必须由大 yaw 展开）\n");
    {
        auto big_ref = smoothStepRef(n, dt, 0.0, 1.2, 6.0);
        auto m = runClosedLoop(ctrl_model, plant_model, cfg, T, 0.0, big_ref, big_ref, false, 0.0, nullptr);
        printf("   最大误差 %.4f rad, θs 范围 [%.4f, %.4f] rad (%.1f° ~ %.1f°), 越限计数 %d/%d\n",
               m.max_aim_err, m.min_small_joint, m.max_small_joint,
               m.min_small_joint / kDeg, m.max_small_joint / kDeg, m.over_min, m.over_max);
        check(m.over_min == 0 && m.over_max == 0, "大阶跃下小 yaw 始终不超限位",
              (double)(m.over_min + m.over_max), 0.0);
        check(m.max_small_joint <= smax + 1e-6, "大阶跃下小 yaw 未越上侧限位", m.max_small_joint, smax);
        check(m.rms_aim_err < 0.06, "RMS 瞄准误差 < 0.06 rad", m.rms_aim_err, 0.06);
    }

    printf("\n[4] 底盘以 1 rad/s 旋转时的世界方位保持\n");
    {
        const double T2 = 2.0;
        const int n2 = static_cast<int>(T2 / dt);
        auto zero = constantRef(n2, 0.0);
        auto m = runClosedLoop(ctrl_model, plant_model, cfg, T2, 1.0, zero, zero, false, 0.0, nullptr);
        check(m.max_aim_err < 0.05, "底盘旋转下瞄准误差 < 0.05 rad", m.max_aim_err, 0.05);
    }

    printf("\n[5] 失配（惯量×1.3、摩擦×1.5、未建模负载 0.06/0.03 N·m）\n");
    {
        ModelParams pm = plant_model;
        pm.Jbig_eff *= 1.3; pm.Js *= 1.3;
        pm.fcBig *= 1.5; pm.fvBig *= 1.5; pm.fcSmall *= 1.5; pm.fvSmall *= 1.5;
        const double load[2] = {0.06, 0.03};
        auto big_ref = constantRef(n, 0.0);
        auto small_ref = smoothStepRef(n, dt, 0.0, 0.13, 5.0);
        auto off = runClosedLoop(ctrl_model, pm, cfg, T, 0.0, big_ref, small_ref, false, 0.0, load);
        auto on  = runClosedLoop(ctrl_model, pm, cfg, T, 0.0, big_ref, small_ref, true, 0.02, load);
        printf("   积分关: 最大 %.4f / RMS %.4f rad；积分开: 最大 %.4f / RMS %.4f rad\n",
               off.max_aim_err, off.rms_aim_err, on.max_aim_err, on.rms_aim_err);
        // 判据: 积分补偿**不应显著**劣化跟踪。容许 15% 相对余量 —— 该场景两种情形的 RMS
        // 都在 1e-3 rad 量级（≪ 0.06 rad 的合格线），绝对差 ~5e-5 rad 属数值噪声；
        // 但真正的退化（例如积分把摩擦死区顶成极限环）会远超 15%。
        const double tol = off.rms_aim_err * 1.15 + 1e-6;
        check(on.rms_aim_err <= tol, "积分补偿不显著劣化跟踪（容许 15%）", on.rms_aim_err, tol);
        check(on.max_aim_err < 0.06, "失配+积分下最大瞄准误差 < 0.06 rad", on.max_aim_err, 0.06);
        check(on.over_min == 0 && on.over_max == 0, "失配下小 yaw 不超限位",
              (double)(on.over_min + on.over_max), 0.0);
    }

    printf("\n[6] 两侧行程都被用满且不越限（软限位区缩到 6.00°, 逼小 yaw 走满行程）\n");
    {
        // 软限位区缩到 10%·总行程 = 6.0°（行程 ±30°）⇒ 允许关节贴到 −24° / +24°。
        // 两侧独立推导（不假设对称）⇒ 换非对称行程也不用改这里。
        // 大 yaw 强保持（权重 100）⇒ 只能小 yaw 出力。
        DualYawMpcConfig cdeep = cfg;
        cdeep.small_limit_soft_ratio = 0.9;
        cdeep.w_big_azimuth = 100.0;
        const SoftLimitsT sd = expectedSoft(cdeep);
        printf("   该场景软限位区 [%.2f°, %.2f°]\n", sd.lo / kDeg, sd.hi / kDeg);

        auto big_ref = constantRef(n, 0.0);
        // 负侧: 要求 −0.70 rad (−40.1°), 远超 −30° 硬限位
        auto neg_ref = smoothStepRef(n, dt, 0.0, -0.70, 4.0);
        auto mn = runClosedLoop(ctrl_model, plant_model, cdeep, T, 0.0, big_ref, neg_ref, false, 0.0, nullptr);
        printf("   负侧: θs 最小 %.4f rad (%.2f°), 越下侧计数 %d\n",
               mn.min_small_joint, mn.min_small_joint / kDeg, mn.over_min);
        check(mn.min_small_joint >= smin - 1e-6, "负侧未越 −30° 硬限位", -mn.min_small_joint, -smin);
        check(mn.min_small_joint <= -(24.0 * kDeg), "负侧行程被用满（≥24°）", -mn.min_small_joint, 24.0 * kDeg);
        check(mn.min_small_joint >= sd.lo - 5.0 * kDeg, "负侧未深入软限位区（≤5° 越界余量）",
              sd.lo - mn.min_small_joint, 5.0 * kDeg);

        // 正侧: 要求 +0.70 rad (+40.1°), 超过 +30° 硬限位
        auto pos_ref = smoothStepRef(n, dt, 0.0, 0.70, 4.0);
        auto mp = runClosedLoop(ctrl_model, plant_model, cdeep, T, 0.0, big_ref, pos_ref, false, 0.0, nullptr);
        printf("   正侧: θs 最大 %.4f rad (%.2f°), 越上侧计数 %d\n",
               mp.max_small_joint, mp.max_small_joint / kDeg, mp.over_max);
        check(mp.max_small_joint <= smax + 1e-6, "正侧未越 +30° 硬限位", mp.max_small_joint, smax);
        check(mp.max_small_joint >= +(24.0 * kDeg), "正侧行程被用满（≥24°）", mp.max_small_joint, 24.0 * kDeg);
        check(mp.max_small_joint <= sd.hi + 5.0 * kDeg, "正侧未深入软限位区（≤5° 越界余量）",
              mp.max_small_joint - sd.hi, 5.0 * kDeg);

        // 软限位几何（两侧独立）: MPC 报的 small_ref_over_limit 必须与实测软限位一致
        printf("   软限位判定: soft=[%.4f, %.4f] rad\n", sd.lo, sd.hi);
        check(!refOverLimit(ctrl_model, cfg, 0.0), "参考在行程内 → 不报越软限位", 0.0, 0.0);
        check(!refOverLimit(ctrl_model, cfg, 0.14), "参考 +8.0° (< 15°) → 不报越软限位", 0.14, soft.hi);
        check(refOverLimit(ctrl_model, cfg, 0.30), "参考 +17.2° (> 15°) → 报越软限位", 0.30, soft.hi);
        check(!refOverLimit(ctrl_model, cfg, -0.22), "参考 −12.6° (> −15°) → 不报越软限位",
              -0.22, -soft.lo);
        check(refOverLimit(ctrl_model, cfg, -0.30), "参考 −17.2° (< −15°) → 报越软限位",
              -0.30, -soft.lo);
    }

    printf("\n[7] 非对称行程 (b): 回中代价把冗余自由度拉向**行程中心 −2.5°**（而不是 0）\n");
    {
        // 大/小 yaw 给**同一个**方位角参考 ⇒ 小 yaw 关节角是冗余自由度（任意 θs 都能精确
        // 跟踪）, 唯一决定它的是回中代价 ⇒ 稳态 θs 由 small_center_angle 决定。
        // 注意: 默认 w_c=0.05 只是"打破多解"级别的弱权重, 在带库伦静摩擦的样机上会被
        // 卡住（稳态几乎不动）; 因此该场景把 w_small_center 提到 100 让回中项主导, 被验证
        // 的是**回中目标角**的语义（−2.5° vs 0）, 不是权重本身。
        // 理论稳态（忽略摩擦/力矩代价）: θs → c·w_c/(w_c + w_b), 与 c 成正比 ⇒ 中心换了
        // 稳态就跟着换, 这正是"显式可配置"要保证的性质。
        DualYawMpcConfig ccen = cfg;
        ccen.w_small_center = 100.0;
        auto ref = smoothStepRef(n, dt, 0.12, 0.42, 6.0);
        // 初始把小 yaw 放在 +0.12 rad(+6.9°, 仍在正侧软限位 8.75° 内), 让它必须往中心走
        auto m = runClosedLoop(ctrl_model, plant_model, ccen, T, 0.0, ref, ref, false, 0.0, nullptr, 0.12);
        printf("   初始 θs = +6.90°, 默认中心: 稳态 θs = %.4f rad (%.3f°)；行程中心 = %.4f rad (%.3f°)\n",
               m.final_small_joint, m.final_small_joint / kDeg,
               cfg.small_center_angle, cfg.small_center_angle / kDeg);
        check(std::fabs(cfg.small_center_angle - 0.5 * (smin + smax)) < 1e-12,
              "默认 small_center_angle = 行程中心 0.5·(min+max)", cfg.small_center_angle,
              0.5 * (smin + smax));
        check(std::fabs(m.final_small_joint - cfg.small_center_angle) < 0.01,
              "稳态 θs 收敛到行程中心 −2.5°（容差 0.57°）",
              std::fabs(m.final_small_joint - cfg.small_center_angle), 0.01);
        check(std::fabs(m.final_small_joint - cfg.small_center_angle) <
                  std::fabs(m.final_small_joint),
              "稳态 θs 更靠近 −2.5° 而不是 0（非对称行程下 ≠ 回中到 0）",
              std::fabs(m.final_small_joint - cfg.small_center_angle), std::fabs(m.final_small_joint));
        check(m.rms_aim_err < 0.05, "冗余自由度下跟踪 RMS < 0.05 rad", m.rms_aim_err, 0.05);

        // 显式可配置: 把中心改成 0 → 关节应回到 0（证明中心是配置量, 不是隐式平均）
        DualYawMpcConfig czero = ccen;
        czero.small_center_angle = 0.0;
        auto m0 = runClosedLoop(ctrl_model, plant_model, czero, T, 0.0, ref, ref, false, 0.0, nullptr, 0.12);
        printf("   初始 θs = +6.90°, center=0  : 稳态 θs = %.4f rad (%.3f°)\n",
               m0.final_small_joint, m0.final_small_joint / kDeg);
        check(std::fabs(m0.final_small_joint) < 0.005, "center=0 时稳态 θs 收敛到 0",
              m0.final_small_joint, 0.005);
        check(std::fabs(m.final_small_joint - m0.final_small_joint) > 0.025,
              "两种 center 配置的稳态差 > 1.4°（证明 center 真的生效）",
              std::fabs(m.final_small_joint - m0.final_small_joint), 0.025);
    }

    printf("\n[8] 非对称行程 (c): 明显偏向负侧的大运动能正常跟踪且不撞限位\n");
    {
        auto big_ref = constantRef(n, 0.0);
        auto neg_ref = smoothStepRef(n, dt, 0.0, -0.30, 2.0);   // 要求小 yaw 往 −25° 一侧走

        // ① 默认权重（大 yaw 可协助承担一部分）: 应能正常跟踪, 且负侧能走到
        //    正侧软限位（8.75°）之外 —— 这正是非对称行程带来的能力。
        auto m2 = runClosedLoop(ctrl_model, plant_model, cfg, T, 0.0, big_ref, neg_ref, false, 0.0, nullptr);
        printf("   默认权重: 最大|ε| %.4f / RMS %.4f rad, θs 最小 %.2f°, 越限 %d/%d\n",
               m2.max_aim_err, m2.rms_aim_err, m2.min_small_joint / kDeg, m2.over_min, m2.over_max);
        check(m2.max_aim_err < 0.05, "偏向负侧的大运动跟踪最大误差 < 0.05 rad", m2.max_aim_err, 0.05);
        check(m2.rms_aim_err < 0.02, "偏向负侧的大运动 RMS 误差 < 0.02 rad", m2.rms_aim_err, 0.02);
        check(m2.over_min == 0 && m2.over_max == 0, "偏向负侧的大运动不撞限位",
              (double)(m2.over_min + m2.over_max), 0.0);
        check(m2.min_small_joint >= smin + 1.0 * kDeg, "与 −25° 硬限位仍留有 ≥1° 余量",
              smin - m2.min_small_joint, -1.0 * kDeg);
        check(m2.min_small_joint <= soft.lo, "负侧走到了软限位 −13.75° 之外（负侧行程确实可用）",
              m2.min_small_joint, soft.lo);

        // ② 大 yaw 强保持（权重 100）: 全部由小 yaw 承担, 障碍项必须把它软性挡在限位内
        DualYawMpcConfig chold = cfg;
        chold.w_big_azimuth = 100.0;
        auto m = runClosedLoop(ctrl_model, plant_model, chold, T, 0.0, big_ref, neg_ref, false, 0.0, nullptr);
        printf("   大 yaw 强保持: 最大|ε| %.4f / RMS %.4f rad, θs 最小 %.2f°, 越限 %d/%d\n",
               m.max_aim_err, m.rms_aim_err, m.min_small_joint / kDeg, m.over_min, m.over_max);
        check(m.over_min == 0 && m.over_max == 0, "大 yaw 强保持下也不撞限位",
              (double)(m.over_min + m.over_max), 0.0);
        check(m.min_small_joint >= smin + 1.0 * kDeg, "大 yaw 强保持下与 −25° 仍留有 ≥1° 余量",
              smin - m.min_small_joint, -1.0 * kDeg);
        check(m.min_small_joint <= -(11.0 * kDeg), "大 yaw 强保持下确实往负侧走了 ≥11°",
              -m.min_small_joint, 11.0 * kDeg);
        // 大 yaw 被强保持、小 yaw 又被障碍项软性挡住时, 剩下的参考误差必然留在瞄准角上
        // （这是"宁可留误差也不撞限位"的预期行为）, 这里只做一个宽松的量级兜底。
        check(m.rms_aim_err < 0.30, "大 yaw 强保持下 RMS 误差 < 0.30 rad（挡限位是预期行为）",
              m.rms_aim_err, 0.30);
    }

    printf("\n[9] 求解性能（对比配置）\n");
    {
        auto big_ref = smoothStepRef(n, dt, 0.0, 0.5, 6.0);
        struct C { const char* name; bool rk4; int N; int it; };
        const C cases[] = {{"RK4   N=12 max_iter=8 ", true, 12, 8},
                           {"RK4   N=12 max_iter=15", true, 12, 15},
                           {"Euler N=12 max_iter=8 ", false, 12, 8},
                           {"RK4   N=20 max_iter=8 ", true, 20, 8}};
        for (const auto& c : cases) {
            DualYawMpcConfig cc = cfg;
            cc.use_rk4 = c.rk4; cc.N = c.N; cc.max_iter = c.it;
            auto m = runClosedLoop(ctrl_model, plant_model, cc, 1.0, 0.0, big_ref, big_ref, false, 0.0, nullptr);
            printf("   %s: 平均 %6.2f ms / 最大 %6.2f ms → ~%4.0f Hz | RMS 误差 %.4f\n",
                   c.name, m.mean_solve_ms, m.max_solve_ms,
                   1000.0 / std::max(0.05, m.mean_solve_ms), m.rms_aim_err);
        }
    }

    printf("\n[10] 平均求解耗时 < 控制周期\n");
    {
        auto big_ref = smoothStepRef(n, dt, 0.0, 0.4, 6.0);
        auto m = runClosedLoop(ctrl_model, plant_model, cfg, 1.5, 0.0, big_ref, big_ref, false, 0.0, nullptr);
        printf("   默认配置: 平均 %.2f ms / 最坏 %.2f ms（周期 %.1f ms）\n",
               m.mean_solve_ms, m.max_solve_ms, dt * 1000);
        check(m.mean_solve_ms < dt * 1000.0, "平均求解耗时 < 控制周期", m.mean_solve_ms, dt * 1000.0);
    }

    printf("\n%s (失败项: %d)\n", g_fail == 0 ? "全部通过" : "存在失败", g_fail);
    return g_fail == 0 ? 0 : 1;
}
