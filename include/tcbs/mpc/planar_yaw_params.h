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

    // ── 16 个待辨识参数（★ 已由实车数据辨识；来源见下方注释）──
    // 数据: data/cars/Sentry1/sysid/ 的 **279 训练段**（140 激励 + 139 保持段，保持段只取前 3 s）
    //       + 80 留出段（全局段号 ≥140，不参与拟合；大 90 + 小 90 激励段 + 179 保持段）
    // 方法: python/scripts/identify_params_torch.py --epochs=10000 --batch-segments
    //       --substeps=2 --threads=4 --eval-every=100 --beta-mode=auto --state-mode=est
    //       （λ=100，输出误差法，16 参一起拟合、γ 冻结在 0.002）
    // 结果: data/cars/Sentry1/ident/params.txt（用时 2604.8 s；loss 前 5 均值 0.1063 → 后 5 均值 0.0934）
    //       图: ident_convergence.png / ident_traj.png / ident_convergence_learning.png
    //       留出集窗口(0.1 s) RMSE: 角度 1.325/0.476/0.583°、角速度 0.440/0.193/0.230 rad/s
    //       （对照初值 1.332/0.455/0.591°、0.479/0.190/0.232）
    //       整段 3 s 开环 RMSE 13.1/13.0/16.9°（换向接触事件时序误差累积，属预期，不要按它判好坏）
    // ⚠ **可信度分级**（同一批数据换 seed/epochs 复跑仍会漂的参数不要当真）:
    //   · 较可信: Jbig_eff、Js、fc_small、fv_small、backlash_delta、Jmotor
    //   · 可疑:   fv_big = 0.2374 是 fc_big = 0.0962 的 **2.47 倍** —— "粘滞 > 库仑" 在云台上
    //            不寻常，通常说明有未建模的速度比例项被 fv 吸收（上一批数据同一个毛病）；
    //            fc_motor / fv_motor = 0.0041 / 0.0303，与上一次的 0.030 / 0.010 几乎"互换"
    //            ⇒ 电机侧库仑/粘滞不可分，只有两者之和近似可信；
    //            backlash_k / backlash_c（与 δ 共线，δ 才是有物理含义的那个）
    //   · **不可信: Px/Py**（本批数据 `tilted=0`，A 系重力只剩 ~0.28 m/s² 的残余倾斜，
    //            而辨识 P 需要固定 ~10°（1.70 m/s²）⇒ 拟合出的 |P|=0.0226 是噪声/垃圾桶。
    //            要定 P 必须补一批**固定 ~10° 倾角**的数据（`--tilted`/`--tilt-rolling`）一起拟合。）
    p.Jbig_eff = 0.045614;  // 大 yaw 侧惯量（含 m_u|d|²）
    p.Js       = 0.008116;  // 上装绕小 yaw 轴总惯量
    p.Px       = 0.021348;  // 上装一阶矩（kg·m）★ 不可信（见上）
    p.Py       = -0.007430; //                    ★ 不可信（见上）
    p.fcBig    = 0.096245;  p.fvBig   = 0.237374;
    p.fcSmall  = 0.033434;  p.fvSmall = 0.048466;

    // ── 固定/可选 ──
    // ★ λ = 100（用户确认）: 软符号在 |ω| ≳ 1°/s ≈ 0.0175 rad/s 即饱和，足以逼近真实库仑摩擦。
    //   数值稳定性: 摩擦模态 Jacobian = fc·λ·(M⁻¹)_kk；λ=100、当前辨识参数
    //   （fc_big=0.096245/fc_small=0.033434、Jbig_eff=0.045614/Js=0.008116）下
    //   det M = Jbig_eff·Js = 3.70e-4，(M⁻¹)_bb = 21.9、(M⁻¹)_ss = 145.1
    //   ⇒ |J_f| ≈ fc_big·λ·21.9 ≈ 211 /s（大 yaw）、fc_small·λ·145.1 ≈ 485 /s（小 yaw）
    //   ⇒ 显式 RK4 稳定上限 dt ≲ 2.78/485 ≈ **5.7 ms**（小 yaw 侧是瓶颈，因为 Js 小）。
    //   ⚠ 换这组参数后 **substeps=2（dt_eff=5.0 ms）只剩 1.15× 余量、已在边界上**；
    //     dt=10 ms 的控制步请用 `substeps ≥ 4`（dt_eff=2.5 ms，余量 2.29×）—— 见 defaultMpcConfig()。
    //   用 recommendedFrictionLambda() 复核；换参数后必须重算。
    //   否则 ω≈0 附近梯度符号翻转 → MPC 输出零力矩（历史上踩过）。
    p.frictionLambda = 100.0;
    p.tau_offset_big = 0.0;      // 可选常数负载（默认关）
    p.tau_offset_small = 0.0;

    // ── ★ 大 yaw 传动背隙（3-DOF 模型 `eomBacklash()` 用；2-DOF 的 eom() 不受影响）──
    //   τ_t = k·[dz(Δ) + γ·Δ] + c·Δ̇,  Δ = θ_motor − θ_platform − β
    //   ★ 这 5 个（δ/k/c/J_motor/电机摩擦）**已与平面 8 参由同一次 torch 拟合一起给出**，
    //     不再是占位值；γ 仍默认冻结在 0.002（只作死区内的梯度引导，不是被辨识对象）。
    //     · δ: 实车辨识 0.096463 rad ≈ 5.53°（与用户实测"大约 5°"一致）；它是**唯一
    //       有物理含义**的背隙量（k/c 与它共线 ⇒ 只有 δ 可当真）；
    //     · k / c: 同步带+啮合的接触刚度/阻尼（弹性范围很小 ⇒ k 较大）。若 k 与 c
    //       在辨识里共线，按 c = 2ζ√(k·J_motor)（ζ≈0.05）固定 c —— 用户已同意；
    //     · Jmotor / fcMotor / fvMotor: 电机侧（折算到关节侧）惯量与摩擦 —— 背隙内
    //       电机几乎空载，所以它与云台侧那组**必须分开**（这正是 fv_big 被高估的原因）。
    //   ⚠ β（死区中心）**不是参数**: 实机上它就是"电机多圈角 − 云台角（±π 环绕）"的圈数差
    //     + 死区中心 —— 本批数据里在线 β 走遍 +0.45…+6.34 rad，而 Δ−β 全程只在 ±0.082 rad 内
    //     （口径自洽）。所以运行期必须用估计器的在线值
    //     （`YawStateEstimator::Estimate::backlash_center`），离线拟合出的全局 β=0 无意义。
    //   起标定作用的是"全量数据"里的 `theta_big_motor` 与 `theta_big_platform` 两列。
    p.backlash_delta   = 0.096463; // δ ≈ 5.53°（辨识值；上一批 0.0873）
    p.backlash_k       = 157.8279; // 接触刚度（★ 与 c/δ 共线 ⇒ 数值可信度低；k 给太大会让 MPC 的
                                   //   Jacobian 求值失败 —— 试过 2000 会出现 trust_region 报错）
    p.backlash_c       = 2.591145; // 接触阻尼（★ 同上，共线）
    p.backlash_smooth_eps = 1.0e-4;
    p.backlash_through    = 0.002; // 直通线性项 γ（死区内的微弱梯度引导；物理严格=0；默认冻结不辨识）
    p.Jmotor           = 0.005455; // 电机侧惯量（关节侧）
    p.fcMotor          = 0.004139; // 电机侧库仑摩擦（★ 与 fvMotor 严重互换，只有和值可信）
    p.fvMotor          = 0.030332; // 电机侧粘滞摩擦（★ 同上）
    p.tau_offset_motor = 0.0;
    // 稳定性自检: 接触刚度引入的快模态上限（k=158 在子步 2.5 ms 下余量 ≈390×）
    //   上限 = recommendedBacklashStiffness(p, dt/substeps) ≈ 6.1e4；
    //   实际瓶颈仍是摩擦 λ（见上面的注释），背隙刚度不是限制项。
    return p;
}

// 默认 MPC 配置（面向 ~100Hz 后台 loop；实测 loop_fps 后可调 N / 积分器）
inline DualYawMpcConfig defaultMpcConfig() {
    DualYawMpcConfig c;
    c.dt_control = 0.01;
    // 平面 8 参模型的每次求值远小于原 40 参版本 → 预测步数可以取更长
    c.N = 24;
    // ★ λ=100 时 dt=10 ms 的显式 RK4 不稳定（当前辨识参数下 |J_f|≈211/485 /s
    //   ⇒ 上限 ≈5.7 ms，瓶颈在小 yaw 侧；substeps=2 只剩 1.15× 余量，已在边界上）
    //   ⇒ 每个控制步内细分 4 个子步（2.5 ms，余量 2.29×），兼顾稳定与求解耗时（实测 ~4×）
    c.substeps = 4;
    c.use_rk4 = true;
    c.max_iter = 8;

    c.w_big_azimuth = 1.0;
    c.w_small_azimuth = 1.0;
    c.w_small_center = 0.0;
    c.w_small_limit = 1e4;
    c.small_limit_soft_ratio = 0.9;
    c.r_big_torque = 1.0;
    c.r_small_torque = 0.01;
    c.rd_big_rate = 100.0;
    c.rd_small_rate = 100.0;
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
    c.small.min_angle = -40.0 * M_PI / 180.0;  // −30°
    c.small.max_angle =  40.0 * M_PI / 180.0;  // +30°
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
