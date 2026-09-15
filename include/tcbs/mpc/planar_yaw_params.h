#ifndef TCBS_PLANAR_YAW_PARAMS_H
#define TCBS_PLANAR_YAW_PARAMS_H

#include "tcbs/mpc/planar_yaw_model.h"
#include "tcbs/mpc/dual_yaw_mpc.h"

namespace tcbs {

// ============================================================================
// planar_yaw_params.h — 默认参数集中定义
//
// ★ **8 个待辨识参数全是占位值**，必须由 python/scripts/collect_sysid.py 采集、
//   再经 LS（tools/identify_params）或 torch（python/scripts/identify_params_torch.py）
//   辨识后替换。几何量（d）是实测值，请按实际机械填写。
// ============================================================================
namespace dual_yaw {

inline ModelParams defaultModelParams() {
    ModelParams p;
    // ── 实测几何（★ 请按实际机械填写）──
    p.dx = 0.030;         // 小 yaw 轴相对大 yaw 轴的平面偏置
    p.dy = 0.0;
    p.gravity = 9.81;
    p.m_u_known = 0.0;    // 不称重 ⇒ 保持 0（仅倾斜时 m_u·d·g⊥ 项受影响）

    // ── 8 个待辨识参数（占位值，量级取自典型 RM 云台）──
    p.Jbig_eff = 0.0240;  // 大 yaw 侧惯量（含 m_u|d|²）
    p.Js       = 0.0130;  // 上装绕小 yaw 轴总惯量
    p.Px       = 0.0000;  // 上装一阶矩（kg·m）；占位 0 = 假定质心在小 yaw 轴上
    p.Py       = 0.0000;
    p.fcBig    = 0.090;  p.fvBig   = 0.030;
    p.fcSmall  = 0.030;  p.fvSmall = 0.008;

    // ── 固定/可选 ──
    // ★ λ = 100（用户确认）: 软符号在 |ω| ≳ 1°/s ≈ 0.0175 rad/s 即饱和，足以逼近真实库仑摩擦。
    //   数值稳定性: 摩擦模态 Jacobian = fc·λ/J_eff；λ=100、fc≈0.22/0.097、J≈0.02~0.05
    //   ⇒ |J_f| ≈ 490~1100 /s ⇒ 显式 RK4 稳定上限 dt ≲ 2.78/|J_f| ≈ 2.5~5.7 ms。
    //   控制步 dt=10 ms 时必须用积分子步（`DualYawMpcConfig::substeps ≥ 2`，建议 4~8），
    //   否则 ω≈0 附近梯度符号翻转 → MPC 输出零力矩（历史上踩过）。
    //   用 recommendedFrictionLambda() 检查当前参数下的上限。
    p.frictionLambda = 100.0;
    p.tau_offset_big = 0.0;      // 可选常数负载（默认关）
    p.tau_offset_small = 0.0;
    return p;
}

// 默认 MPC 配置（面向 ~100Hz 后台 loop；实测 loop_fps 后可调 N / 积分器）
inline DualYawMpcConfig defaultMpcConfig() {
    DualYawMpcConfig c;
    c.dt_control = 0.01;
    // 平面 8 参模型的每次求值远小于原 40 参版本 → 预测步数可以取更长
    c.N = 12;
    // ★ λ=100 时 dt=10 ms 的显式 RK4 不稳定（|J_f|≈360~700/s ⇒ 上限 ≈4~8 ms）
    //   ⇒ 每个控制步内细分 4 个子步（2.5 ms），兼顾稳定与求解耗时（实测 ~4×）
    c.substeps = 4;
    c.use_rk4 = true;
    c.max_iter = 8;

    c.w_big_azimuth = 1.0;
    c.w_small_azimuth = 1.0;
    c.w_small_center = 0.05;
    c.w_small_limit = 1e4;
    c.small_limit_soft_ratio = 0.75;
    c.r_big_torque = 0.01;
    c.r_small_torque = 0.01;
    c.rd_big_rate = 0.1;
    c.rd_small_rate = 0.1;
    c.smooth_eps = 1e-6;
    c.ref_delay_steps = 0;

    c.big.max_torque = 1.0;          // N·m（沿用旧系统量级；★ 按实测替换）
    c.big.max_torque_rate = 40.0;    // N·m/s
    c.big.min_angle = -1e9;
    c.big.max_angle = 1e9;

    c.small.max_torque = 0.5;        // N·m（★ 占位：小 yaw 电机力矩能力）
    c.small.max_torque_rate = 80.0;  // N·m/s
    // ── 小 yaw 机械行程（**非对称**: −25° ~ +20°）──
    // 大 yaw 可多圈自由转（min/max = ∓1e9 = 不限位）; 小 yaw 只能在这个区间内转动。
    c.small.min_angle = -25.0 * M_PI / 180.0;  // −25°
    c.small.max_angle =  20.0 * M_PI / 180.0;  // +20°
    // 回中（冗余自由度分配）目标角 = 行程中心。**显式给出**, 而不是让 MPC 内部按
    // 0.5·(min+max) 隐式推算 —— 换机械后必须在这里改（0 = 回中到关节零位）。
    // 例: [−25°, +20°] ⇒ −2.5°（注意 0 并不是行程中心, 回中到 0 会把冗余自由度
    // 分配偏向正侧、白白吃掉负侧行程）。
    c.small_center_angle = 0.5 * (c.small.min_angle + c.small.max_angle);  // −2.5°
    return c;
}

} // namespace dual_yaw

} // namespace tcbs

#endif // TCBS_PLANAR_YAW_PARAMS_H