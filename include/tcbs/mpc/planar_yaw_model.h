// ============================================================================
// planar_yaw_model.h — 双级 yaw 云台的**平面二维**模型（8 参可辨识形式）
//
// 化简前提（用户确认）:
//   1) 两 yaw 轴平行；小 yaw 轴相对大 yaw 轴只有平面偏置 d（几何可实测）
//   2) 大 yaw 转轴不发生位移（底盘绕该轴转动时 O1 不动）
//   3) 忽略 pitch 转动对上装质心的影响 ⇒ **pitch 不进动力学**（只作为下发给电控的目标角）
//   4) 质心不单独测量：用"上装一阶矩" P = m_u·ρ 表征（ρ = 上装质心相对小 yaw 轴的位移）
//   5) 大 yaw 侧不含小 yaw 部分的质量/惯量整体并入 J_big
//
// 记号:
//   θ_b = q0（大 yaw 关节角）, θ_s = q1（小 yaw 关节角）
//   d   = (dx, dy)          小 yaw 轴相对大 yaw 轴的平面偏置（实测几何）
//   P   = (Px, Py)          ★ 上装一阶矩 m_u·ρ（待辨识）
//   J_s                     ★ 上装绕小 yaw 轴的总惯量（含 m_u|ρ|²，待辨识）
//   Jbig_eff = J_big + m_u|d|²  ★ 大 yaw 侧惯量（含小 yaw 整体被偏置的平行轴项，待辨识）
//   Q(θ_s) = R(θ_s)·P
//   g_A = 重力在 **A 系（大 yaw 转子系）** 的平面分量 —— 必须是 A 系，不能是 C 系!
//         理由: 绕轴的力矩 = ẑ·(r × m g)，其中 r、g 必须表达在**同一**旋转系里；
//         取 A 系时 r_u^A = d + R(θ_s)ρ 不含 θ_b ⇒ 公式对 θ_b 无关（且实测直接可得:
//         g_A = R_A_IMU·g_imu，不需要 θ_b 估计）。底盘水平时 g_A 的平面分量 = 0 ⇒ 重力项全为 0。
//
// ── 推导（拉格朗日，可直接审计）────────────────────────────────────────────────
// 上装质心在 C 系: r_u = R(θ_b)·[d + R(θ_s)·ρ]
//   ∂r_u/∂θ_b = ẑ×r_u ,  ∂r_u/∂θ_s = ẑ×(R(θ_b)R(θ_s)ρ)
//   M_ij = Σ m(∂r/∂q_i)·(∂r/∂q_j) + Σ J  ⇒
//     M22 = J_s
//     M12 = J_s + d·Q(θ_s)
//     M11 = Jbig_eff + J_s + 2·d·Q(θ_s)
//   μ ≜ ∂M11/∂θ_s = 2·(d_y·Q_x − d_x·Q_y)      （ν ≜ ∂M12/∂θ_s = μ/2）
//   重力势 V = −m_u g·r_u ⇒
//     G_b ≜ −∂V/∂θ_b = m_u(d_x g_y − d_y g_x) + (Q_x g_y − Q_y g_x)
//     G_s ≜ −∂V/∂θ_s = (Q_x g_y − Q_y g_x)
//   ⇒ τ_b = M11 θ̈_b + M12 θ̈_s + μ θ̇_b θ̇_s + ν θ̇_s² − G_b + fric_b + base_b
//      τ_s = M12 θ̈_b + M22 θ̈_s − ½μ θ̇_b²        − G_s + fric_s + base_s
//
// ── 底盘绕关节轴转动 (ω_c, α_c) 的耦合（无需任何额外参数）────────────────────
// 系统对关节轴的角动量 L = ω_c·M11 + Σ_j M_1j θ̇_j  ⇒ 交叉项 N_k = ω_c·M_k1
//   令 h 为偏置项（τ = M q̈ + h），则
//     base_k = ω_c·[ Σ_i (∂M_k1/∂q_i)θ̇_i − Σ_i (∂M_i1/∂q_k)θ̇_i ] − ∂T0/∂q_k + M_k1·α_c
//   其中 T0 = ½ω_c²M11 ⇒ ∂T0/∂θ_b = 0, ∂T0/∂θ_s = ½μω_c²
//   ⇒ base_b = μ ω_c θ̇_s + M11 α_c
//      base_s = ω_c (ν θ̇_s − μ θ̇_b) − ½μ ω_c² + M12 α_c
//   （**完全由 M 及其对 q 的偏导构成，不需要 m_u 或任何新参数**）
//
// ── 摩擦 ───────────────────────────────────────────────────────────────────
//   fric_k = fc_k·tanh(λ·θ̇_k) + fv_k·θ̇_k ，λ 固定 10（受数值可积性上限约束）
//
// ── 8 个待辨识参数（顺序与 paramsToVector / regressor 列一致）────────────────
//   0 Jbig_eff   1 J_s   2 Px   3 Py   4 fc_big   5 fv_big   6 fc_small   7 fv_small
//   实测几何 d（2 个分量）与重力不属于辨识参数；可选 τ_offset（默认 0 = 关闭）
// ============================================================================
#ifndef TCBS_PLANAR_YAW_MODEL_H
#define TCBS_PLANAR_YAW_MODEL_H

#include <cmath>
#include <vector>

namespace tcbs {

namespace dual_yaw {

// 参数向量长度（可辨识参数个数）
constexpr int kNumParams = 8;

struct ModelParams {
    // ── 实测几何（★ 必须与 dual_yaw::defaultModelParams() 保持一致，实际使用请改那里）──
    //   注意: d 只以「耦合项」进入模型（M12 = Js + d·Q、M11 = Jbig_eff + Js + 2·d·Q、
    //   μ = 2(dy·Qx − dx·Qy)）⇒ d 错 k 倍, 同数据辨识出的 |P| 就错 1/k 倍。
    double dx = 0.0, dy = 0.07;      // 小 yaw 轴相对大 yaw 轴的平面偏置 (m)，默认=实测 (0, 0.07)
    double gravity = 9.81;           // 重力加速度 (m/s²)
    // 可选: 上装质量（**仅**用于倾斜时 m_u·d·g⊥ 那一项；不称重时保持 0 即可）
    double m_u_known = 0.0;
    // ── ★ 8 个待辨识参数（标定值；**与 dual_yaw::defaultModelParams() 保持一致**，
    //    那里是权威来源、并记录了可信度分级）──
    double Jbig_eff = 0.051893;      // 大 yaw 侧惯量（含 m_u|d|²）      kg·m²
    double Js       = 0.009162;      // 上装绕小 yaw 轴总惯量            kg·m²
    double Px       = 0.001897;      // 上装一阶矩 m_u·ρ_x               kg·m（★ 本批数据不可辨识）
    double Py       = -0.001017;     // 上装一阶矩 m_u·ρ_y               kg·m（★ 同上）
    double fcBig    = 0.103360, fvBig   = 0.209044;   // 大 yaw 库仑/粘滞摩擦（fv 可疑）
    double fcSmall  = 0.030582, fvSmall = 0.048735;   // 小 yaw 库仑/粘滞摩擦（fv 可疑）
    // ── 固定 / 可选 ──
    double frictionLambda = 100.0;   // tanh 软符号陡度（固定，不辨识）
                                     // ★ 100: |ω| ≳ 1°/s 即饱和（逼近真库仑）;
                                     //   代价是显式积分需要子步 ≤ ~2.5 ms
    double tau_offset_big = 0.0;     // 可选常数负载（默认 0 = 关闭）
    double tau_offset_small = 0.0;
};

// 外生量（每次求解刷新）
struct ModelExo {
    double gravity_a[2] = {0.0, 0.0};  // ★ 重力在 **A 系（大 yaw 转子系）** 的平面分量（水平时 (0,0)）
    double base_omega = 0.0;           // 底盘绕关节轴的角速度 (rad/s)
    double base_alpha = 0.0;           // 底盘绕关节轴的角加速度 (rad/s²)，未知时置 0
};

// ── 派生量（供组装、测试与文档引用）──
template <typename T>
inline void planarDerived(const T qs, const ModelParams& p, T& Qx, T& Qy, T& M11, T& M12,
                          T& mu) {
    using std::cos;
    using std::sin;
    const T cs = cos(qs), sn = sin(qs);
    Qx = T(p.Px) * cs - T(p.Py) * sn;
    Qy = T(p.Px) * sn + T(p.Py) * cs;
    const T dQ = T(p.dx) * Qx + T(p.dy) * Qy;          // d·Q
    mu = T(2.0) * (T(p.dy) * Qx - T(p.dx) * Qy);       // ∂M11/∂θ_s
    M11 = T(p.Jbig_eff) + T(p.Js) + T(2.0) * dQ;
    M12 = T(p.Js) + dQ;
}

// 摩擦（τ 侧的耗散项；Jet 走 ADL 的 ceres::tanh）
template <typename T>
inline T frictionTorque(const T& omega, double fc, double fv, double lambda) {
    using std::tanh;
    return fc * tanh(lambda * omega) + fv * omega;
}

// ── M(2×2) 与 h(2)（含摩擦、重力、底盘耦合；不含 q̈）──
template <typename T>
inline void eom(const T q[2], const T qd[2], const ModelParams& p, const ModelExo& e,
                T M[2][2], T h[2]) {
    T Qx, Qy, M11, M12, mu;
    planarDerived(q[1], p, Qx, Qy, M11, M12, mu);
    const T M22 = T(p.Js);
    M[0][0] = M11;  M[0][1] = M12;
    M[1][0] = M12;  M[1][1] = M22;

    // 重力项（底盘水平时为 0）
    const T gx = T(e.gravity_a[0]), gy = T(e.gravity_a[1]);
    const T Gs = Qx * gy - Qy * gx;
    const T Gb = T(p.m_u_known) * (T(p.dx) * gy - T(p.dy) * gx) + Gs;

    // 科氏/离心 + 重力 + 底盘耦合 + 摩擦 + 可选常数负载
    const T wc = T(e.base_omega), ac = T(e.base_alpha);
    const T tb = qd[0], ts = qd[1];
    //   base_b = μ ω_c θ̇_s + M11 α_c
    //   base_s = ω_c(½μ θ̇_s − μ θ̇_b) − ½μ ω_c² + M12 α_c
    //   （α_c 项**不**乘 ω_c；−½μω_c² 来自 T0 = ½ω_c²M11 的 ∂/∂θ_s）
    h[0] = mu * tb * ts + T(0.5) * mu * ts * ts - Gb
         + mu * ts * wc + M11 * ac
         + frictionTorque(tb, p.fcBig, p.fvBig, p.frictionLambda)
         + T(p.tau_offset_big);
    // 注意: Σ_i(∂N_k/∂q_i)θ̇_i 里的 +½μθ̇_sω_c 与 −Σ_i(∂N_i/∂q_k)θ̇_i 里的 −½μθ̇_sω_c
    //       正好相消 ⇒ 只剩 −μ θ̇_b ω_c（少留一半会带来 ≈½μθ̇_sω_c 的系统偏差）
    h[1] = -T(0.5) * mu * tb * tb - Gs
         + (-mu * tb) * wc - T(0.5) * mu * wc * wc + M12 * ac
         + frictionTorque(ts, p.fcSmall, p.fvSmall, p.frictionLambda)
         + T(p.tau_offset_small);
}

// ── 前向动力学: q̈ = M⁻¹(τ − h) ──
template <typename T>
inline bool forwardAccel(const T q[2], const T qd[2], const T tau[2], const ModelParams& p,
                         const ModelExo& e, T qdd[2]) {
    using std::abs;
    T M[2][2], h[2];
    eom(q, qd, p, e, M, h);
    const T det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (!(abs(det) > T(1e-12))) { qdd[0] = T(0.0); qdd[1] = T(0.0); return false; }
    const T inv = T(1.0) / det;
    const T r0 = tau[0] - h[0], r1 = tau[1] - h[1];
    qdd[0] = ( M[1][1] * r0 - M[0][1] * r1) * inv;
    qdd[1] = (-M[1][0] * r0 + M[0][0] * r1) * inv;
    return true;
}

// ── RK4 单步（力矩零阶保持）──
template <typename T>
inline void integrateStep(const T q[2], const T qd[2], const T tau[2], const ModelParams& p,
                          const ModelExo& e, double dt, int substeps, T q_next[2],
                          T qd_next[2]) {
    if (substeps < 1) substeps = 1;
    const double hh = dt / static_cast<double>(substeps);
    T qa[2] = {q[0], q[1]}, qda[2] = {qd[0], qd[1]};
    for (int s = 0; s < substeps; ++s) {
        T k1v[2], k2v[2], k3v[2], k4v[2];
        T t1[2] = {qa[0], qa[1]}, td1[2] = {qda[0], qda[1]};
        forwardAccel(t1, td1, tau, p, e, k1v);
        T t2[2], td2[2];
        for (int i = 0; i < 2; ++i) {
            t2[i] = qa[i] + T(0.5 * hh) * td1[i];
            td2[i] = qda[i] + T(0.5 * hh) * k1v[i];
        }
        forwardAccel(t2, td2, tau, p, e, k2v);
        T t3[2], td3[2];
        for (int i = 0; i < 2; ++i) {
            t3[i] = qa[i] + T(0.5 * hh) * td2[i];
            td3[i] = qda[i] + T(0.5 * hh) * k2v[i];
        }
        forwardAccel(t3, td3, tau, p, e, k3v);
        T t4[2], td4[2];
        for (int i = 0; i < 2; ++i) {
            t4[i] = qa[i] + T(hh) * td3[i];
            td4[i] = qda[i] + T(hh) * k3v[i];
        }
        forwardAccel(t4, td4, tau, p, e, k4v);
        const double h6 = hh / 6.0;
        for (int i = 0; i < 2; ++i) {
            q_next[i] = qa[i] + T(h6) * (td1[i] + T(2.0) * td2[i] + T(2.0) * td3[i] + td4[i]);
            qd_next[i] = qda[i] + T(h6) * (k1v[i] + T(2.0) * k2v[i] + T(2.0) * k3v[i] + k4v[i]);
        }
        qa[0] = q_next[0]; qa[1] = q_next[1];
        qda[0] = qd_next[0]; qda[1] = qd_next[1];
    }
}

// ── 逆动力学: τ = M q̈ + h（辨识/前馈用）──
inline void inverseDynamics(const double q[2], const double qd[2], const double qdd[2],
                            const ModelParams& p, const ModelExo& e, double tau[2]) {
    double M[2][2], h[2];
    eom(q, qd, p, e, M, h);
    tau[0] = M[0][0] * qdd[0] + M[0][1] * qdd[1] + h[0];
    tau[1] = M[1][0] * qdd[0] + M[1][1] * qdd[1] + h[1];
}

// ============================================================================
// 参数向量化 + 解析回归矩阵
//   ★ 关键性质: 方程对 8 个参数**线性** ⇒ Y 与参数无关，可直接解析给出
//     τ_b = ... + Jbig_eff·θ̈_b + Js·(θ̈_b+θ̈_s) + [d·Q 的导数项] + P·(重力/cos-sin 项) ...
//   下面按"逐参数求 Y 的列"用**解析式**写出（避免数值差分误差）。
//   列顺序: 0 Jbig_eff, 1 Js, 2 Px, 3 Py, 4 fc_big, 5 fv_big, 6 fc_small, 7 fv_small
// ============================================================================
inline void paramsToVector(const ModelParams& p, double out[kNumParams]) {
    out[0] = p.Jbig_eff; out[1] = p.Js; out[2] = p.Px; out[3] = p.Py;
    out[4] = p.fcBig; out[5] = p.fvBig; out[6] = p.fcSmall; out[7] = p.fvSmall;
}

inline void vectorToParams(const double in[kNumParams], ModelParams& p) {
    p.Jbig_eff = in[0]; p.Js = in[1]; p.Px = in[2]; p.Py = in[3];
    p.fcBig = in[4]; p.fvBig = in[5]; p.fcSmall = in[6]; p.fvSmall = in[7];
}

inline const char* const* paramNames() {
    static const char* names[kNumParams] = {
        "Jbig_eff", "Js", "Px", "Py", "fc_big", "fv_big", "fc_small", "fv_small"};
    return names;
}

// Y(2×8) = ∂τ/∂φ（解析）
inline void regressor(const double q[2], const double qd[2], const double qdd[2],
                      const ModelParams& p, const ModelExo& e,
                      double Y[2][kNumParams]) {
    const double ts = q[1], tb = qd[0], vs = qd[1];
    const double cs = std::cos(ts), sn = std::sin(ts);
    const double gx = e.gravity_a[0], gy = e.gravity_a[1];
    const double wc = e.base_omega;

    for (int j = 0; j < kNumParams; ++j) { Y[0][j] = 0.0; Y[1][j] = 0.0; }

    // 注意: 惯性参数不仅通过 M·q̈ 进入，还通过 **M_k1·α_c** 这一项进入
    //（基座角加速度的惯性反作用），因此每个惯性参数列都要加 α_c 的贡献。
    // ── 0: Jbig_eff —— 只出现在 M11 ──
    Y[0][0] = qdd[0] + e.base_alpha;
    // ── 1: Js —— 出现在 M11、M12、M22 ──
    Y[0][1] = qdd[0] + qdd[1] + e.base_alpha;   // M11 含 J_s ⇒ α_c 项
    Y[1][1] = qdd[0] + qdd[1] + e.base_alpha;   // M12/M22 含 J_s ⇒ α_c 项
    // ── 2,3: Px, Py —— 通过 Q=R(θs)P、μ=2(dyQx−dxQy) 进入（M 与 h 都含 P）──
    //   ∂Q/∂Px = (cs, sn),  ∂Q/∂Py = (−sn, cs)
    const double dQdPx = p.dx * cs + p.dy * sn;        // ∂(d·Q)/∂Px
    const double dQdPy = -p.dx * sn + p.dy * cs;       // ∂(d·Q)/∂Py
    const double mudPx = 2.0 * (p.dy * cs - p.dx * sn);   // ∂μ/∂Px
    const double mudPy = -2.0 * (p.dy * sn + p.dx * cs);  // ∂μ/∂Py
    //   ∂M11/∂P = 2∂(d·Q)/∂P ; ∂M12/∂P = ∂(d·Q)/∂P
    const double dM11dPx = 2.0 * dQdPx, dM12dPx = dQdPx;
    const double dM11dPy = 2.0 * dQdPy, dM12dPy = dQdPy;
    //   ∂G_s/∂Px = cs·gy − sn·gx ; ∂G_s/∂Py = −sn·gy − cs·gx
    const double dGsdPx = cs * gy - sn * gx;
    const double dGsdPy = -sn * gy - cs * gx;
    auto pTerm = [&](double dM11, double dM12, double dmu, double dGs, double out[2]) {
        // τ_b = M11θ̈b + M12θ̈s + μ tb ts + ½μ ts² − G_b + μ ts ωc + M11 αc
        out[0] = dM11 * qdd[0] + dM12 * qdd[1] + dmu * tb * vs + 0.5 * dmu * vs * vs
               - dGs + dmu * vs * wc + dM11 * e.base_alpha;
        // τ_s = M12θ̈b + M22θ̈s − ½μ tb² − G_s + (½μ vs − μ tb)ωc − ½μ ωc² + M12 αc
        out[1] = dM12 * qdd[0] - 0.5 * dmu * tb * tb
               - dGs + (-dmu * tb) * wc - 0.5 * dmu * wc * wc
               + dM12 * e.base_alpha;
    };
    double col[2];
    pTerm(dM11dPx, dM12dPx, mudPx, dGsdPx, col);
    Y[0][2] = col[0]; Y[1][2] = col[1];
    pTerm(dM11dPy, dM12dPy, mudPy, dGsdPy, col);
    Y[0][3] = col[0]; Y[1][3] = col[1];
    // ── 4..7: 摩擦（∂/∂fc = tanh(λθ̇)，∂/∂fv = θ̇）──
    Y[0][4] = std::tanh(p.frictionLambda * tb);
    Y[0][5] = tb;
    Y[1][6] = std::tanh(p.frictionLambda * vs);
    Y[1][7] = vs;
    // 摩擦只作用于本轴 ⇒ 其它两列为 0（已初始化为 0）
}

// ── 数值可积性检查（摩擦模态时间常数必须显著大于预测步长）──
inline double recommendedFrictionLambda(const ModelParams& p, double dt, bool rk4 = true) {
    const double det = (p.Jbig_eff + p.Js) * p.Js - p.Js * p.Js;   // θ_s=0 处的 det
    if (std::fabs(det) < 1e-12) return 0.0;
    const double inv_bb = p.Js / det;                    // 大 yaw 方向等效惯量倒数
    const double inv_ss = (p.Jbig_eff + p.Js) / det;      // 小 yaw 方向等效惯量倒数
    const double lim = rk4 ? 2.78 : 2.0;
    const double lam_b = (p.fcBig > 1e-9) ? lim / (dt * p.fcBig * inv_bb) : 1e9;
    const double lam_s = (p.fcSmall > 1e-9) ? lim / (dt * p.fcSmall * inv_ss) : 1e9;
    return lam_b < lam_s ? lam_b : lam_s;
}

} // namespace dual_yaw

} // namespace tcbs

#endif // TCBS_PLANAR_YAW_MODEL_H
