#ifndef TCBS_PLANAR_YAW_PARAMS_H
#define TCBS_PLANAR_YAW_PARAMS_H

#include "tcbs/mpc/planar_yaw_model.h"
#include "tcbs/mpc/dual_yaw_mpc.h"

namespace tcbs {

// ============================================================================
// planar_yaw_params.h — 默认参数集中定义
//
// ★ **8 个平面参数**由 python/scripts/collect_sysid.py 采集、再用
//   python/scripts/identify_params_torch.py（**唯一辨识路径**，torch 输出误差法）辨识后替换。
//   几何量（d = dx/dy）是实测值，请按实际机械填写（当前 (0, 0.07)）。
// ★ **大 yaw 背隙那 8 个参数**（δ/k/c/γ/J_motor/电机摩擦/β）由**同一个** torch 辨识
//   作为 3-DOF 模型参数**一起拟合**（共 16 参，不再"占位 + 手猜"）—— 见
//   docs/backlash_model.md。注意 β 虽然也参与离线拟合，但**运行期必须用估计器的在线值**
//   （`Estimate::backlash_center`），因为它随电机/云台共同旋转而漂移。
// ============================================================================
namespace dual_yaw {

inline ModelParams defaultModelParams() {
    ModelParams p;
    // ── 实测几何（★ 请按实际机械填写）──
    // d = 小 yaw 轴相对大 yaw 轴的平面偏置（大 yaw 系下，m）。它只以「耦合项」形式进入模型:
    //   M12 = Js + d·Q(θs)、M11 = Jbig_eff + Js + 2·d·Q、μ = ∂M11/∂θs = 2(dy·Qx − dx·Qy)
    //   ⇒ d 错 k 倍, 同数据辨识出的 |P| 就会错 1/k 倍（**必须实测, 不要用占位值**）。
    // ★ 本构型实测: **两轴在 x（右）方向无偏置，只有 y（前）方向偏 0.07 m**
    //   ⇒ d = (0, 0.07)，|d| = 0.07 m，方向 = +y（底盘前向）。
    //   注意与 dx≠0 时相比耦合项形态不同: μ = 2(dy·Qx − dx·Qy) = 2·dy·Qx（dx=0 时只剩 Qx 支路）。
    p.dx = 0.0;           // x（右）方向偏置 = 0（实测两轴在横向共面）
    p.dy = 0.07;          // ★ y（前）方向偏置 0.07 m（卡尺实测；符号: 小 yaw 轴在大 yaw 轴**前方**）
    p.gravity = 9.81;
    p.m_u_known = 0.0;    // 不称重 ⇒ 保持 0（仅倾斜时 m_u·d·g⊥ 项受影响）

    // ── 8 个待辨识参数（★ 已由实车数据辨识；来源见下方注释）──
    // 数据: data/cars/Sentry1/ 287 段（95 大 + 93 小 + 99 保持段）
    // 方法: python/scripts/identify_params_torch.py --epochs=1000（λ=100，输出误差法）
    // 结果: data/cars/Sentry1/ident/lam100.txt（用时 3022 s，loss 0.130 → 0.060）
    // ⚠ **可信度分级**（同一批数据用不同 seed/epochs 复跑仍会漂的参数不要当真）:
    //   · 较可信: Jbig_eff、fc_big、fc_small
    //   · 可疑:   Js（−30%）、fv_big / fv_small（训练末尾仍在单调上升，且 fv_big=0.209
    //            是 fc_big=0.103 的 **2.0 倍** —— 粘滞>库仑在云台上不寻常，通常说明
    //            有未建模的"速度比例项"被 fv 吸收）
    //   · **不可信: Px/Py**（本批数据底盘只倾斜 ~1.4°（|g_A| 中位 0.238 m/s²），
    //            而辨识 P 需要固定 ~10°（1.70 m/s²）；训练中 Px 在 ±0.005 之间来回跳，
    //            幅值 ~0.002 远小于预期的 |P|≈0.01 ⇒ 这就是噪声。
    //            要定 P 必须补一批**固定 ~10° 倾角**的数据（`--tilted`）。）
    p.Jbig_eff = 0.051893;  // 大 yaw 侧惯量（含 m_u|d|²）
    p.Js       = 0.009162;  // 上装绕小 yaw 轴总惯量
    p.Px       = 0.001897;  // 上装一阶矩（kg·m）★ 不可信（见上）
    p.Py       = -0.001017; //                    ★ 不可信（见上）
    p.fcBig    = 0.103360;  p.fvBig   = 0.209044;
    p.fcSmall  = 0.030582;  p.fvSmall = 0.048735;

    // ── 固定/可选 ──
    // ★ λ = 100（用户确认）: 软符号在 |ω| ≳ 1°/s ≈ 0.0175 rad/s 即饱和，足以逼近真实库仑摩擦。
    //   数值稳定性: 摩擦模态 Jacobian = fc·λ·(M⁻¹)_kk；λ=100、当前辨识参数
    //   （fc_big=0.1034/fc_small=0.0306、Jbig_eff=0.0519/Js=0.00916）下
    //   det M = Jbig_eff·Js = 4.75e-4，(M⁻¹)_bb = 19.3、(M⁻¹)_ss = 128
    //   ⇒ |J_f| ≈ fc_big·λ·19.3 ≈ 200 /s（大 yaw）、fc_small·λ·128 ≈ 392 /s（小 yaw）
    //   ⇒ 显式 RK4 稳定上限 dt ≲ 2.78/392 ≈ **7.1 ms**（小 yaw 侧是瓶颈，因为 Js 小）。
    //   用 recommendedFrictionLambda() 复核；换参数后必须重算。
    //   控制步 dt=10 ms 时必须用积分子步（`DualYawMpcConfig::substeps ≥ 2`，建议 4~8），
    //   否则 ω≈0 附近梯度符号翻转 → MPC 输出零力矩（历史上踩过）。
    //   用 recommendedFrictionLambda() 检查当前参数下的上限。
    p.frictionLambda = 100.0;
    p.tau_offset_big = 0.0;      // 可选常数负载（默认关）
    p.tau_offset_small = 0.0;

    // ── ★ 大 yaw 传动背隙（3-DOF 模型 `eomBacklash()` 用；2-DOF 的 eom() 不受影响）──
    //   τ_t = k·dz(Δ) + c·Δ̇,  Δ = θ_motor − θ_platform − β
    //   **全是占位初值，必须用（重采的）数据标定**:
    //     · δ: 用户实测"大约 5°" ⇒ 0.0873 rad。它同时是**唯一静态可标定**的背隙量；
    //     · k / c: 同步带+啮合的接触刚度/阻尼（弹性范围很小 ⇒ k 较大）。若 k 与 c
    //       在辨识里共线，按 c = 2ζ√(k·J_motor)（ζ≈0.05）固定 c —— 用户已同意；
    //     · Jmotor / fcMotor / fvMotor: 电机侧（折算到关节侧）惯量与摩擦 —— 背隙内
    //       电机几乎空载，所以它与云台侧那组**必须分开**（这正是 fv_big 被高估的原因）。
    //   ⚠ β（死区中心）**不是参数**: 它随电机/云台共同旋转而移动，且云台角由 IMU 推出
    //     会有漂移 ⇒ 由估计器在线给出（`YawStateEstimator::Estimate::backlash_center`）。
    //   起标定作用的是"全量数据"里的 `theta_big_motor` 与 `theta_big_platform` 两列。
    p.backlash_delta   = 0.0873;   // δ ≈ 5°
    p.backlash_k       = 200.0;    // 占位：接触刚度（**必须实测**；k 给太大会让 MPC 的
                                   //   Jacobian 求值失败 —— 试过 2000 会出现 trust_region 报错）
    p.backlash_c       = 2.0;      // 占位：接触阻尼
    p.backlash_smooth_eps = 1.0e-4;
    p.backlash_through    = 0.002; // 直通线性项 γ（死区内的微弱梯度引导；物理严格=0）
    p.Jmotor           = 0.006;    // 占位：电机侧惯量（关节侧）
    p.fcMotor          = 0.030;    // 占位：电机侧库仑摩擦
    p.fvMotor          = 0.010;    // 占位：电机侧粘滞摩擦
    p.tau_offset_motor = 0.0;
    // 稳定性自检: 接触刚度引入的快模态上限（k=200 在子步 2.5ms 下余量 ≈30×）
    //   上限 = recommendedBacklashStiffness(p, dt/substeps)；
    //   实际瓶颈仍是摩擦 λ（见上面的注释），背隙刚度不是限制项。
    return p;
}

// 默认 MPC 配置（面向 ~100Hz 后台 loop；实测 loop_fps 后可调 N / 积分器）
inline DualYawMpcConfig defaultMpcConfig() {
    DualYawMpcConfig c;
    c.dt_control = 0.01;
    // 平面 8 参模型的每次求值远小于原 40 参版本 → 预测步数可以取更长
    c.N = 24;
    // ★ λ=100 时 dt=10 ms 的显式 RK4 不稳定（|J_f|≈360~700/s ⇒ 上限 ≈4~8 ms）
    //   ⇒ 每个控制步内细分 4 个子步（2.5 ms），兼顾稳定与求解耗时（实测 ~4×）
    c.substeps = 4;
    c.use_rk4 = true;
    c.max_iter = 8;

    c.w_big_azimuth = 1.0;
    c.w_small_azimuth = 1.0;
    c.w_small_center = 0.0;
    c.w_small_limit = 1e4;
    c.small_limit_soft_ratio = 0.75;
    c.r_big_torque = 0.01;
    c.r_small_torque = 0.01;
    c.rd_big_rate = 10.0;
    c.rd_small_rate = 10.0;
    c.smooth_eps = 1e-6;
    c.ref_delay_steps = 0;

    c.big.max_torque = 1.0;          // N·m（沿用旧系统量级；★ 按实测替换）
    c.big.max_torque_rate = 40.0;    // N·m/s
    c.big.min_angle = -1e9;
    c.big.max_angle = 1e9;

    c.small.max_torque = 1.0;        // N·m（★ 占位：小 yaw 电机力矩能力）
    c.small.max_torque_rate = 40.0;  // N·m/s
    // ── 小 yaw 机械行程（**对称** ±30°）──
    // 大 yaw 可多圈自由转（min/max = ∓1e9 = 不限位）; 小 yaw 只能在这个区间内转动。
    //   ★ 该值必须与电控侧硬限位宏（mcu_code_demo 的 YAW_SMALL_MIN_RAD/MAX_RAD）
    //     以及采集脚本的 SMALL_TRAVEL_MIN/MAX 保持一致；改行程要三处一起改。
    //   ★ 软限位区: 两侧各自从硬限位向行程内缩 (1−small_limit_soft_ratio)·总行程
    //     = 0.25·60° = 15° ⇒ 软限位区 [−15°, +15°]（见 dual_yaw_mpc.cpp 的 smallSoftLimits）。
    c.small.min_angle = -30.0 * M_PI / 180.0;  // −30°
    c.small.max_angle =  30.0 * M_PI / 180.0;  // +30°
    // 回中（冗余自由度分配）目标角 = 行程中心。**显式给出**, 而不是让 MPC 内部按
    // 0.5·(min+max) 隐式推算 —— 换机械后必须在这里改（0 = 回中到关节零位）。
    // 当前行程对称 ⇒ 中心 = 0。**若哪天行程又变成非对称**（例如 [−25°,+20°] ⇒ −2.5°），
    // 0 就不再是行程中心，回中到 0 会把冗余自由度分配偏向一侧、白白吃掉另一侧行程。
    c.small_center_angle = 0.5 * (c.small.min_angle + c.small.max_angle);  // 0.0
    return c;
}

} // namespace dual_yaw

} // namespace tcbs

#endif // TCBS_PLANAR_YAW_PARAMS_H
