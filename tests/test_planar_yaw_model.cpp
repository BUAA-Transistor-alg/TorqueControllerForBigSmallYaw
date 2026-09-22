// ============================================================================
// test_planar_yaw_model.cpp — 平面 8 参模型验证（手写推导 vs 独立数值参考）
//
// 四类验证:
//  A. 与**独立参考实现**比对: 用"物理参数化"(J_big, m_u, ρ, J_u) 从零写出
//     动能 T（含底盘 ω_c 的牵连速度）与势能 V，再用数值微分构造拉格朗日方程，
//     与模型的 M/h 逐点比较（模型用 8 参形式: Jbig_eff, Js, P=(Px,Py)）。
//  B. 解析特例: M 与 Q 的关系、小 yaw 的离心项 −½μω_b²、底盘自转项 −½μω_c²、
//     重力项（A 系重力）、摩擦项符号。
//  C. 回归矩阵: 解析 regressor() 与 inverseDynamics() 的数值偏导逐列比对。
//  D. 能量一致性: ω_c=0、τ=0 时 dE/dt = 0。
//
// 运行: ./build/test_planar_yaw_model
// ============================================================================
#include "tcbs/mpc/planar_yaw_model.h"
#include "tcbs/mpc/planar_yaw_params.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace tcbs {

using namespace dual_yaw;

namespace {

// ── 参考实现: 物理参数化 ──
struct Phys {
    double J_big = 0.0;   // 大 yaw 侧（不含上装）绕轴惯量
    double m_u = 1.0;     // 上装质量（任意取值：8 参形式对其不敏感，用于验证不变量）
    double rho[2] = {0, 0};   // 上装质心相对小 yaw 轴的位移
    double J_u = 0.0;     // 上装绕自身质心的惯量（绕轴）
    double d[2] = {0, 0};     // 小 yaw 轴偏置
};

// 由 8 参 + 选定 m_u 还原物理参数
Phys physFromParams(const ModelParams& p, double m_u) {
    Phys ph;
    ph.m_u = m_u;
    ph.d[0] = p.dx; ph.d[1] = p.dy;
    ph.J_big = p.Jbig_eff - m_u * (p.dx * p.dx + p.dy * p.dy);
    if (m_u > 1e-9) { ph.rho[0] = p.Px / m_u; ph.rho[1] = p.Py / m_u; }
    ph.J_u = p.Js - m_u * (ph.rho[0] * ph.rho[0] + ph.rho[1] * ph.rho[1]);
    return ph;
}

// 上装质心在 C 系（原点 O1）的位置
void rU_C(const Phys& ph, double tb, double ts, double out[2]) {
    const double c = std::cos(ts), s = std::sin(ts);
    const double x = ph.d[0] + c * ph.rho[0] - s * ph.rho[1];
    const double y = ph.d[1] + s * ph.rho[0] + c * ph.rho[1];
    const double cb = std::cos(tb), sb = std::sin(tb);
    out[0] = cb * x - sb * y;
    out[1] = sb * x + cb * y;
}

// 动能（C 系分量表达；牵连速度 = ω_c×r，牵连角速度 = ω_c）
double refT(const Phys& ph, const double q[2], const double qd[2], double wc) {
    double r[2];
    rU_C(ph, q[0], q[1], r);
    // 相对速度: θ̇_b·(ẑ×r) + θ̇_s·(ẑ×(R(θb)R(θs)ρ))
    const double c = std::cos(q[1]), s = std::sin(q[1]);
    const double px = c * ph.rho[0] - s * ph.rho[1];
    const double py = s * ph.rho[0] + c * ph.rho[1];
    const double cb = std::cos(q[0]), sb = std::sin(q[0]);
    const double rx = cb * px - sb * py, ry = sb * px + cb * py;   // R(θb)ρ
    const double vr[2] = {-qd[0] * r[1] - qd[1] * ry, qd[0] * r[0] + qd[1] * rx};
    // 牵连速度 ω_c ẑ×r
    const double vt[2] = {-wc * r[1], wc * r[0]};
    const double vx = vt[0] + vr[0], vy = vt[1] + vr[1];
    const double wu = wc + qd[0] + qd[1];   // 上装绕轴角速度
    // 注意: 大 yaw 侧的转子**也随底盘转动** ⇒ 绝对角速度 = ω_c + θ̇_b，
    //       漏掉这一项会丢掉 J_big·ω_c·θ̇_b（→ 基座 α_c 项）与 ½J_bigω_c²（常数）
    return 0.5 * ph.J_big * (wc + qd[0]) * (wc + qd[0])
         + 0.5 * ph.m_u * (vx * vx + vy * vy)
         + 0.5 * ph.J_u * wu * wu;
}

// 势能: V = −m_u·g_C·r_u^C （g_C 为 C 系中的重力矢量，指向下）
double refV(const Phys& ph, const double q[2], const double gC[2]) {
    double r[2];
    rU_C(ph, q[0], q[1], r);
    return -ph.m_u * (gC[0] * r[0] + gC[1] * r[1]);
}

// 数值微分构造拉格朗日方程；返回 M(2×2)、h(2)
void refEom(const Phys& ph, const double q[2], const double qd[2], double wc, double alpha,
            const double gC[2], double M[2][2], double h[2], bool with_friction,
            const ModelParams& p) {
    const double hs = 1e-4;   // 差分步长（过大/过小都会放大噪声）
    auto dTdqd = [&](int k, const double qq[2], const double qdd[2], double w) {
        double dp[2] = {qdd[0], qdd[1]}, dm[2] = {qdd[0], qdd[1]};
        dp[k] += hs; dm[k] -= hs;
        return (refT(ph, qq, dp, w) - refT(ph, qq, dm, w)) / (2 * hs);
    };
    for (int k = 0; k < 2; ++k) {
        double acc = 0.0;
        for (int j = 0; j < 2; ++j) {
            double qp[2] = {qd[0], qd[1]}, qm[2] = {qd[0], qd[1]};
            qp[j] += hs; qm[j] -= hs;
            M[k][j] = (dTdqd(k, q, qp, wc) - dTdqd(k, q, qm, wc)) / (2 * hs);
        }
        // Σ_j ∂²T/∂q̇_k∂q_j · θ̇_j （科氏/离心项；**容易漏**，漏了就会得出"模型错"的假结论）
        for (int j = 0; j < 2; ++j) {
            double qp[2] = {q[0], q[1]}, qm[2] = {q[0], q[1]};
            qp[j] += hs; qm[j] -= hs;
            acc += (dTdqd(k, qp, qd, wc) - dTdqd(k, qm, qd, wc)) / (2 * hs) * qd[j];
        }
        acc += (dTdqd(k, q, qd, wc + hs) - dTdqd(k, q, qd, wc - hs)) / (2 * hs) * alpha;
        for (int j = 0; j < 2; ++j) {   // −∂T/∂q_j + ∂V/∂q_j （仅 k==j 项进入该行）
            double qp[2] = {q[0], q[1]}, qm[2] = {q[0], q[1]};
            qp[j] += hs; qm[j] -= hs;
            if (j == k) {
                acc += -(refT(ph, qp, qd, wc) - refT(ph, qm, qd, wc)) / (2 * hs);
                acc += (refV(ph, qp, gC) - refV(ph, qm, gC)) / (2 * hs);
            }
        }
        h[k] = acc;
    }
    if (with_friction) {
        h[0] += frictionTorque(qd[0], p.fcBig, p.fvBig, p.frictionLambda);
        h[1] += frictionTorque(qd[1], p.fcSmall, p.fvSmall, p.frictionLambda);
    }
}

int g_fail = 0;
void check(bool ok, const char* name, double v = 0.0, double tol = 0.0) {
    if (ok) printf("  [PASS] %-58s (%.4g ≤ %.4g)\n", name, v, tol);
    else { printf("  [FAIL] %-58s (%.4g > %.4g)\n", name, v, tol); ++g_fail; }
}

// A. 随机状态与参考实现比对（模型侧把 g_C 转成 A 系: g_A = Rz(−θ_b)·g_C）
void testAgainstReference() {
    printf("\n[A] 与独立数值拉格朗日参考实现比对（含底盘 ω_c/α_c 与倾斜重力）\n");
    ModelParams p = defaultModelParams();
    p.fcBig = 0.0; p.fvBig = 0.0; p.fcSmall = 0.0; p.fvSmall = 0.0;   // 摩擦单独测
    p.Px = 0.021; p.Py = -0.013;      // 非零一阶矩
    // ★ 参考实现是"2-DOF 物理参数化"(J_big, m_u, ρ, J_u, 载荷质心 r=d+R(θs)ρ)，
    //   **不含**大 yaw 侧一阶矩 Pb（转子/支架自身偏心）。比对公式时把 Pb 置 0；
    //   Pb 那一项由 [E] 的 Gb 检查单独覆盖。
    p.Pbx = 0.0; p.Pby = 0.0;
    const double m_u = 1.0;           // 参考用任意 m_u（8 参形式应与之无关）
    // 注意: 模型里 m_u(d×g) 那一项需要 m_u（默认 0 = 不建模，见头文件说明）。
    // 为逐项验证公式，这里把同一个 m_u 也告诉模型。
    p.m_u_known = m_u;
    const Phys ph = physFromParams(p, m_u);

    std::mt19937 rng(20240914);
    std::uniform_real_distribution<double> uq(-3.0, 3.0), uw(-3.0, 3.0), up(-1.5, 1.5);
    double maxRelM = 0.0, maxRelH = 0.0, scaleM = 1e-9, scaleH = 1e-9;
    for (int it = 0; it < 200; ++it) {
        double q[2] = {uq(rng), up(rng)};
        double qd[2] = {uw(rng), uw(rng)};
        const double wc = uw(rng), alpha = uw(rng);
        const double gC[2] = {0.6 * uw(rng) / 3.0, -0.9 * uw(rng) / 3.0};  // 倾斜时的平面分量

        ModelExo e;
        e.base_omega = wc;
        e.base_alpha = alpha;
        // g_A = Rz(−θ_b)·g_C
        const double cb = std::cos(-q[0]), sb = std::sin(-q[0]);
        e.gravity_a[0] = cb * gC[0] - sb * gC[1];
        e.gravity_a[1] = sb * gC[0] + cb * gC[1];

        double M[2][2], h[2];
        eom(q, qd, p, e, M, h);
        double Mr[2][2], hr[2];
        refEom(ph, q, qd, wc, alpha, gC, Mr, hr, false, p);

        // 归一化用**全局量级**（逐状态归一会在 h≈0 的状态上放大噪声，产生假失败）
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) scaleM = std::max(scaleM, std::fabs(Mr[a][b]));
            scaleH = std::max(scaleH, std::fabs(hr[a]));
        }
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b)
                maxRelM = std::max(maxRelM, std::fabs(M[a][b] - Mr[a][b]));
            maxRelH = std::max(maxRelH, std::fabs(h[a] - hr[a]));
        }
        // 诊断: 若出现较大绝对偏差则打印该状态（便于定位遗漏项）
        for (int a = 0; a < 2; ++a) {
            if (std::fabs(h[a] - hr[a]) > 0.05 * std::max(1e-6, std::fabs(hr[a])) + 1e-3) {
                printf("    [诊断] q=(%.10g,%.10g) qd=(%.10g,%.10g) wc=%.10g al=%.10g gC=(%.10g,%.10g)\n"
                       "           model h=(%.10g,%.10g) ref h=(%.10g,%.10g) M00=%.10g Mr00=%.10g\n",
                       q[0], q[1], qd[0], qd[1], wc, alpha, gC[0], gC[1],
                       h[0], h[1], hr[0], hr[1], M[0][0], Mr[0][0]);
                break;
            }
        }
    }
    maxRelM /= scaleM; maxRelH /= scaleH;
    check(maxRelM < 1e-5, "M(2×2) 与参考一致（全局归一）", maxRelM, 1e-5);
    check(maxRelH < 1e-4, "h(2) 与参考一致（科氏/离心/底盘/重力，全局归一）", maxRelH, 1e-4);

    // 8 参形式对 m_u 的取值不敏感（换个 m_u 参考仍一致）
    {
        ModelParams p2 = p;
        const double m_u2 = 3.7;
        p2.m_u_known = m_u2;              // 与参考一致，验证"参数化与 m_u 拆分无关"
        const Phys ph2 = physFromParams(p2, m_u2);
        double q[2] = {0.4, -0.7}, qd[2] = {1.1, -0.6};
        ModelExo e; e.base_omega = 0.8; e.base_alpha = 0.0;
        e.gravity_a[0] = 0.3; e.gravity_a[1] = -0.2;
        double M[2][2], h[2];
        eom(q, qd, p2, e, M, h);
        const double cb = std::cos(-q[0]), sb = std::sin(-q[0]);
        const double gC[2] = {cb * e.gravity_a[0] + sb * e.gravity_a[1],
                              -sb * e.gravity_a[0] + cb * e.gravity_a[1]};
        double Mr[2][2], hr[2];
        refEom(ph2, q, qd, 0.8, 0.0, gC, Mr, hr, false, p2);
        double d = 0.0;
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) d = std::max(d, std::fabs(M[a][b] - Mr[a][b]));
            d = std::max(d, std::fabs(h[a] - hr[a]));
        }
        check(d < 1e-9, "模型对参考实现所选的 m_u 不敏感（参数化良定）", d, 1e-9);
    }
}

// B. 解析特例
void testAnalyticCases() {
    printf("\n[B] 解析特例\n");
    ModelParams p = defaultModelParams();
    p.Px = 0.03; p.Py = -0.02;
    p.fcBig = p.fvBig = p.fcSmall = p.fvSmall = 0.0;

    // B1: M 与 Q 的关系
    {
        const double ts = 0.7;
        double q[2] = {0.3, ts}, qd[2] = {0, 0};
        ModelExo e;
        double M[2][2], h[2];
        eom(q, qd, p, e, M, h);
        const double Qx = p.Px * std::cos(ts) - p.Py * std::sin(ts);
        const double Qy = p.Px * std::sin(ts) + p.Py * std::cos(ts);
        const double dQ = p.dx * Qx + p.dy * Qy;
        check(std::fabs(M[1][1] - p.Js) < 1e-15, "M22 = J_s", std::fabs(M[1][1] - p.Js), 1e-15);
        check(std::fabs(M[0][1] - (p.Js + dQ)) < 1e-15, "M12 = J_s + d·Q(θ_s)",
              std::fabs(M[0][1] - (p.Js + dQ)), 1e-15);
        check(std::fabs(M[0][0] - (p.Jbig_eff + p.Js + 2 * dQ)) < 1e-15,
              "M11 = Jbig_eff + J_s + 2·d·Q(θ_s)",
              std::fabs(M[0][0] - (p.Jbig_eff + p.Js + 2 * dQ)), 1e-15);
    }
    // B2: 小 yaw 的离心项 −½μω_b²（大 yaw 以 ω_b 转动、小 yaw 静止）
    {
        const double ts = 0.5, wb = 4.0;
        double q[2] = {0.2, ts}, qd[2] = {wb, 0.0};
        ModelExo e;
        double M[2][2], h[2];
        eom(q, qd, p, e, M, h);
        const double Qx = p.Px * std::cos(ts) - p.Py * std::sin(ts);
        const double Qy = p.Px * std::sin(ts) + p.Py * std::cos(ts);
        const double mu = 2.0 * (p.dy * Qx - p.dx * Qy);
        check(std::fabs(h[1] + 0.5 * mu * wb * wb) < 1e-12,
              "大 yaw 自转对小 yaw 的离心力矩 = −½μω_b²",
              std::fabs(h[1] + 0.5 * mu * wb * wb), 1e-12);
    }
    // B3: 底盘自转项 −½μω_c²（静止、仅底盘以 ω_c 转动）
    {
        const double ts = -0.3, wc = 3.0;
        double q[2] = {0.1, ts}, qd[2] = {0.0, 0.0};
        ModelExo e; e.base_omega = wc;
        double M[2][2], h[2];
        eom(q, qd, p, e, M, h);
        const double Qx = p.Px * std::cos(ts) - p.Py * std::sin(ts);
        const double Qy = p.Px * std::sin(ts) + p.Py * std::cos(ts);
        const double mu = 2.0 * (p.dy * Qx - p.dx * Qy);
        check(std::fabs(h[1] + 0.5 * mu * wc * wc) < 1e-12,
              "底盘自转对小 yaw 的离心力矩 = −½μω_c²",
              std::fabs(h[1] + 0.5 * mu * wc * wc), 1e-12);
    }
    // B4: 重力项（A 系重力）
    {
        const double ts = 0.8;
        double q[2] = {0.25, ts}, qd[2] = {0, 0};
        ModelExo e; e.gravity_a[0] = 0.7; e.gravity_a[1] = -0.4;
        double M[2][2], h[2];
        eom(q, qd, p, e, M, h);
        const double Qx = p.Px * std::cos(ts) - p.Py * std::sin(ts);
        const double Qy = p.Px * std::sin(ts) + p.Py * std::cos(ts);
        const double Gs = Qx * e.gravity_a[1] - Qy * e.gravity_a[0];
        const double Gb = (p.Pbx + p.m_u_known * p.dx) * e.gravity_a[1]
                        - (p.Pby + p.m_u_known * p.dy) * e.gravity_a[0] + Gs;
        check(std::fabs(h[0] + Gb) < 1e-12 && std::fabs(h[1] + Gs) < 1e-12,
              "重力项（A 系）: h_b=−G_b, h_s=−G_s",
              std::max(std::fabs(h[0] + Gb), std::fabs(h[1] + Gs)), 1e-12);
        // 水平底盘（g_A 平面分量 = 0）⇒ 重力项恒为 0
        ModelExo e0;
        eom(q, qd, p, e0, M, h);
        check(std::fabs(h[0]) < 1e-15 && std::fabs(h[1]) < 1e-15, "水平底盘 ⇒ 重力项为 0",
              std::max(std::fabs(h[0]), std::fabs(h[1])), 1e-15);
    }
    // B5: 摩擦项符号
    {
        ModelParams pf = p; pf.fcBig = 0.1; pf.fvBig = 0.02; pf.fcSmall = 0.05; pf.fvSmall = 0.01;
        double q[2] = {0, 0}, qd[2] = {1.5, 1.0};
        ModelExo e;
        double M1[2][2], h1[2], M0[2][2], h0[2];
        eom(q, qd, pf, e, M1, h1);
        eom(q, qd, p, e, M0, h0);
        check(h1[0] > h0[0] && h1[1] > h0[1], "正速度时摩擦项为正（需电机补上）",
              std::min(h1[0] - h0[0], h1[1] - h0[1]), 0.0);
    }
}

// C. 回归矩阵 vs 数值偏导
void testRegressor() {
    printf("\n[C] 解析回归矩阵 vs 逆动力学数值偏导\n");
    ModelParams p = defaultModelParams();
    p.Px = 0.017; p.Py = 0.009;
    p.tau_offset_big = 0.013; p.tau_offset_small = -0.007;   // 也验证 offset 不影响回归（常数项）
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    double maxErr = 0.0;
    for (int it = 0; it < 60; ++it) {
        double q[2] = {u(rng), u(rng)}, qd[2] = {u(rng), u(rng)}, qdd[2] = {u(rng), u(rng)};
        ModelExo e;
        e.base_omega = u(rng); e.base_alpha = u(rng);
        e.gravity_a[0] = 0.3 * u(rng); e.gravity_a[1] = 0.3 * u(rng);
        double Y[2][kNumParams];
        regressor(q, qd, qdd, p, e, Y);
        double phi[kNumParams];
        paramsToVector(p, phi);
        for (int j = 0; j < kNumParams; ++j) {
            const double h = 1e-6 * std::max(1.0, std::fabs(phi[j]));
            double pp[kNumParams], pm[kNumParams];
            for (int k = 0; k < kNumParams; ++k) { pp[k] = phi[k]; pm[k] = phi[k]; }
            pp[j] += h; pm[j] -= h;
            ModelParams pa, pb;
            vectorToParams(pp, pa); vectorToParams(pm, pb);
            // ★ vectorToParams 只写 8 个**待辨识**参数 ⇒ 必须把非辨识量（几何 d、重力、
            //   m_u、λ、τ_offset）从 p 拷回来, 否则有限差分用的是结构体默认几何、
            //   而解析 regressor 用的是 p 的几何 ⇒ 二者不可比（曾因两处默认值恰好相同而掩盖）。
            for (ModelParams* q : {&pa, &pb}) {
                q->dx = p.dx; q->dy = p.dy;
                q->gravity = p.gravity; q->m_u_known = p.m_u_known;
                q->frictionLambda = p.frictionLambda;
                q->tau_offset_big = p.tau_offset_big;
                q->tau_offset_small = p.tau_offset_small;
            }
            double ta[2], tb[2];
            inverseDynamics(q, qd, qdd, pa, e, ta);
            inverseDynamics(q, qd, qdd, pb, e, tb);
            for (int r = 0; r < 2; ++r) {
                const double fd = (ta[r] - tb[r]) / (2 * h);
                maxErr = std::max(maxErr, std::fabs(fd - Y[r][j]));
            }
        }
    }
    check(maxErr < 1e-6, "regressor 与 ∂τ/∂φ 数值偏导一致", maxErr, 1e-6);
}

// E. 大 yaw 传动**背隙**（3-DOF: q = (θ_motor, θ_platform, θ_small)）
//    τ_t = k·dz(Δ) + c·Δ̇,  Δ = θ_m − θ_p − β
void testBacklash() {
    printf("\n[E] 背隙传动（3-DOF）\n");
    ModelParams p = defaultModelParams();

    // ① 死区: |Δ| ≤ δ/2 ⇒ τ_t ≈ 0（中间"几乎完全自由"）
    double max_free = 0.0, max_err = 0.0;
    for (double D = -0.49 * p.backlash_delta; D <= 0.49 * p.backlash_delta; D += 0.01 * p.backlash_delta) {
        double q[3] = {D, 0.0, 0.0}, qd[3] = {0, 0, 0}, M[3][3], h[3];
        ModelExo e;
        eomBacklash(q, qd, p, e, M, h);
        max_free = std::max(max_free, std::fabs(h[0]));      // h[0] 含 τ_t + 电机摩擦(Δ̇=0 ⇒ 0)
    }
    // 容差: 直通项 k·γ·|Δ|（用户授权的梯度引导，Δ≤δ/2）+ **平滑残余**（∝ k·ε）。
    //   ★ 必须显式带上 ε: 当前默认 δ≈0.018、smooth_eps=1e-4 ⇒ ε/(δ/2)≈1.1%，
    //     在 |Δ|→δ/2 附近的平滑残余可达 k·ε/2 量级，比 γ 项还大（旧默认 δ≈0.087 时不明显）。
    const double free_tol = p.backlash_k * (p.backlash_through * 0.5 * p.backlash_delta
                                            + 2.0 * p.backlash_smooth_eps);
    check(max_free < free_tol, "死区内 τ_t ≈ 0（自由段, 仅剩直通项）", max_free, free_tol);

    // ② 接触区: τ_t ≈ k·(|Δ| − δ/2)，两侧反号对称
    double max_contact_err = 0.0, asym = 0.0;
    for (double D = 0.6 * p.backlash_delta; D <= 3.0 * p.backlash_delta; D += 0.1 * p.backlash_delta) {
        double qp[3] = {D, 0, 0}, qm[3] = {-D, 0, 0}, qd[3] = {0, 0, 0}, M[3][3], hp[3], hm[3];
        ModelExo e;
        eomBacklash(qp, qd, p, e, M, hp);
        eomBacklash(qm, qd, p, e, M, hm);
        // 含直通项: τ_t = k·[(Δ−δ/2) + γ·Δ]
        const double theory = p.backlash_k * ((D - 0.5 * p.backlash_delta)
                                              + p.backlash_through * D);
        max_contact_err = std::max(max_contact_err, std::fabs(hp[0] - theory));
        asym = std::max(asym, std::fabs(hp[0] + hm[0]));
    }
    check(max_contact_err < 1e-2, "接触区 τ_t ≈ k(Δ−δ/2)", max_contact_err, 1e-2);
    check(asym < 1e-9, "正负 Δ 的 τ_t 严格反号（对称）", asym, 1e-9);

    // ③ 阻尼项: Δ=0、Δ̇=1、去掉电机摩擦 ⇒ τ_t = c
    {
        ModelParams pn = p; pn.fcMotor = 0.0; pn.fvMotor = 0.0;
        double q[3] = {0, 0, 0}, qd[3] = {1, 0, 0}, M[3][3], h[3];
        ModelExo e;
        eomBacklash(q, qd, pn, e, M, h);
        check(std::fabs(h[0] - pn.backlash_c) < 1e-9, "接触阻尼项 = c·Δ̇", std::fabs(h[0] - pn.backlash_c), 1e-9);
    }

    // ④ β（死区中心）平移: Δ = θ_m − θ_p − β ⇒ 把 β 加到 θ_p 上等价于整体平移
    {
        // Δ = θ_m − θ_p − β 在 (θ_p → θ_p+δ, β → β−δ) 下不变
        const double beta = 0.021;
        double q1[3] = {0.05, 0.0, 0.0}, q2[3] = {0.05, beta, 0.0}, qd[3] = {0, 0, 0};
        ModelExo e2fix;
        double M[3][3], h1[3], h2[3];
        ModelExo e1; ModelExo e2; e2.backlash_beta = -beta;   // θ_p 加 β ⇒ β 减 β 才不变
        eomBacklash(q1, qd, p, e1, M, h1);
        eomBacklash(q2, qd, p, e2, M, h2);
        check(std::fabs(h1[0] - h2[0]) < 1e-12, "β 把死区整体平移（Δ = θ_m−θ_p−β）",
              std::fabs(h1[0] - h2[0]), 1e-12);
    }

    // ⑤ 电机侧与云台侧解耦: 死区内给电机力矩 ⇒ 云台加速度 ≈ 0
    {
        double q[3] = {0.0, 0.0, 0.0}, qd[3] = {0, 0, 0}, u[3] = {0.5, 0.0, 0.0};
        double qdd[3];
        ModelExo e;
        forwardAccelBacklash(q, qd, u, p, e, qdd);
        check(std::fabs(qdd[1]) < 1e-3, "死区内电机出力不驱动云台（解耦）", std::fabs(qdd[1]), 1e-3);
        check(qdd[0] > 1.0, "同一力矩全部作用在电机上（q̈_m = u/J_m）", qdd[0], 1.0);
    }

    // ⑥ 接触刚度余量
    {
        const double k_ok = recommendedBacklashStiffness(p, 0.0025, true);
        check(p.backlash_k <= k_ok, "接触刚度在子步 2.5ms 下稳定（有子步余量）", p.backlash_k, k_ok);
    }
}

// D. 能量一致性: ω_c = 0、无摩擦、τ = 0 ⇒ dE/dt = 0
void testEnergy() {
    printf("\n[D] 能量一致性（dE/dt = q̇ᵀτ）\n");
    ModelParams p = defaultModelParams();
    p.fcBig = p.fvBig = p.fcSmall = p.fvSmall = 0.0;
    ModelExo e;
    // ★ 重力必须置 0: 本检查只取动能 T，而含重力时守恒量是 T + V ⇒ dT/dt = q̇ᵀ·G ≠ 0，
    //   "dT/dt = 0" 这个断言本身不成立。
    //   历史上这条一直是"空过的": 那时 Px=Py=0（占位值）⇒ G≡0 且 μ≡0 ⇒ h≡0，检查退化成常数。
    //   换成实车辨识参数后 P≠0（重力通道打开）才暴露出来。
    //   置重力为 0 后，h 只剩科氏/离心项（∝ μ ∝ |P|）⇒ 这条检查反而**变成真的**:
    //   它验证 T 与 q̈=M⁻¹(τ−h) 在 μ≠0 时仍然自洽。重力项本身由 [B] 的解析特例覆盖。
    e.gravity_a[0] = 0.0; e.gravity_a[1] = 0.0;
    auto energy = [&](const double q[2], const double qd[2]) {
        double M[2][2], h[2];
        eom(q, qd, p, e, M, h);
        return 0.5 * (M[0][0] * qd[0] * qd[0] + 2 * M[0][1] * qd[0] * qd[1] + M[1][1] * qd[1] * qd[1])
             - (-1.0) * 0.0;   // 势能项由 eom 的重力项体现；这里用 dE/dt 检验一致性
    };
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    double maxRel = 0.0;
    for (int it = 0; it < 100; ++it) {
        double q[2] = {u(rng), u(rng)}, qd[2] = {u(rng), u(rng)};
        double tau[2] = {0.0, 0.0};
        double qdd[2];
        forwardAccel(q, qd, tau, p, e, qdd);
        const double hs = 1e-4;   // 差分步长（过大/过小都会放大噪声）
        double dE = 0.0;
        for (int k = 0; k < 2; ++k) {
            double qp[2] = {q[0], q[1]}, qm[2] = {q[0], q[1]};
            qp[k] += hs; qm[k] -= hs;
            dE += (energy(qp, qd) - energy(qm, qd)) / (2 * hs) * qd[k];
            double dp[2] = {qd[0], qd[1]}, dm[2] = {qd[0], qd[1]};
            dp[k] += hs; dm[k] -= hs;
            dE += (energy(q, dp) - energy(q, dm)) / (2 * hs) * qdd[k];
        }
        maxRel = std::max(maxRel, std::fabs(dE) / std::max(1.0, std::fabs(energy(q, qd))));
    }
    // 注: 含重力项时 E = T + V，此处 energy() 只取 T，因此该检验的是"T 与动力学的自洽性"
    check(maxRel < 1e-6, "T 与 q̈=M⁻¹(τ−h) 自洽（dT/dt 与功率匹配）", maxRel, 1e-6);
}

} // namespace

} // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；下面把 tcbs 内的类型与测试辅助函数引入作用域
using namespace tcbs;

int main() {
    printf("=== 平面 8 参双级 yaw 模型验证 ===\n");
    testAgainstReference();
    testAnalyticCases();
    testRegressor();
    testEnergy();
    testBacklash();
    printf("\n%s (失败项: %d)\n", g_fail == 0 ? "全部通过" : "存在失败", g_fail);
    return g_fail == 0 ? 0 : 1;
}
