// ============================================================================
// fast_sim.cpp —— 手写的高效率 3-DOF（含大 yaw 背隙）前向仿真，供 Python 侧 ctypes 调用
// ============================================================================
// ★ 与 python/scripts/identify_params/model.py 的 **numpy 参考实现逐项同序**
//   （planar_derived_np / eom_np / backlash_torque_np / eom_backlash_np /
//     forward_accel_backlash_np / rk4_step_backlash_np / simulate_backlash_np），
//   因此结果与 `simulate_backlash_np` 一致到 ~1e-14 —— 残差**只**来自 tanh 的实现差异
//   （numpy 的向量化 tanh 与 libm tanh 可差 1 ULP；sqrt 两边都是 IEEE 精确）。
//   ★ 同一组输入下，**不同线程数/分块宽度的结果彼此逐位一致**（线程只按 batch 切分，
//   没有跨条目的归约，所以并行不改变任何浮点结果）。
//
// 性能手段:
//   ① **分块向量化**: 把 batch 切成 block（默认 64），块内参数/状态全部按 SoA 存放
//      （`double a[blk]`），内层循环连续 ⇒ 编译器可自动向量化（sqrt/tanh 走标量 libm）。
//   ② **std::thread**: batch 各条目彼此独立 ⇒ 按块把 batch 切给多个线程，无同步。
//   ③ 数据预取: 每个块进入时把该块的 [T,5] 输入 gather 成"时间优先"的连续缓冲，
//      避免内层按 item-major 跨步读取。
//
// ★ 本文件与主工程（include/tcbs/...、CMakeLists.txt）**没有任何关系**:
//   不 include 任何项目头文件、不参与项目的 CMake 构建，只用 C++17 标准库 + libm。
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int NP = 18;      // 参数个数（顺序 = PARAM_NAMES）
constexpr int NC = 8;       // seq_const 通道: q0(3) qd0(3) base_omega base_alpha
constexpr int NV = 5;       // seq_var  通道: tau_big tau_small grav_x grav_y beta
constexpr int MAX_BLK = 256;

struct Fixed {
    double dx, dy, m_u_known, lambda, eps;
    double off_b, off_s, off_m;
};

// 块内参数（SoA；长度 = block）
struct BlockParams {
    double Jbig[MAX_BLK], Js[MAX_BLK], Px[MAX_BLK], Py[MAX_BLK];
    double fcb[MAX_BLK], fvb[MAX_BLK], fcs[MAX_BLK], fvs[MAX_BLK];
    double bd[MAX_BLK], bk[MAX_BLK], bc[MAX_BLK], bt[MAX_BLK];
    double Jm[MAX_BLK], fcm[MAX_BLK], fvm[MAX_BLK];
    double Pbx[MAX_BLK], Pby[MAX_BLK];
};

struct Vec3 {
    double x[MAX_BLK], y[MAX_BLK], z[MAX_BLK];
};

inline double smooth_relu(double x, double e) {
    return 0.5 * (x + std::sqrt(x * x + e * e));       // = smooth_relu_np
}

// ── 块内加速度 q̈ = M⁻¹(u − h)（3-DOF；与 forward_accel_backlash_np 同序）──
void accel_block(const BlockParams &p, const Fixed &f, int n,
                 const Vec3 &q, const Vec3 &d,
                 const double *tau_b, const double *tau_s,
                 const double *gx, const double *gy, const double *beta,
                 const double *wc, const double *ac, Vec3 &a) {
#pragma GCC ivdep
    for (int i = 0; i < n; ++i) {
        // ── planar_derived_np(q[...,1])：子块的 q[...,1] = **小 yaw 角** ──
        const double cs = std::cos(q.z[i]);
        const double sn = std::sin(q.z[i]);
        const double Qx = p.Px[i] * cs - p.Py[i] * sn;
        const double Qy = p.Px[i] * sn + p.Py[i] * cs;
        const double dQ = f.dx * Qx + f.dy * Qy;
        const double mu = 2.0 * (f.dy * Qx - f.dx * Qy);
        const double M11 = p.Jbig[i] + p.Js[i] + 2.0 * dQ;
        const double M12 = p.Js[i] + dQ;

        // ── eom_np（云台/小 yaw 子块；重力在最后减，与 numpy 同序）──
        //   子块里 tb = **云台角速度**（走 fc_big/fv_big），ts = **小 yaw 角速度**。
        const double tb = d.y[i];
        const double ts = d.z[i];
        double hp = mu * tb * ts + 0.5 * mu * ts * ts
                    + mu * ts * wc[i] + M11 * ac[i]
                    + (p.fcb[i] * std::tanh(f.lambda * tb) + p.fvb[i] * tb)
                    + f.off_b;
        double hs = -0.5 * mu * tb * tb
                    + (-mu * tb) * wc[i] - 0.5 * mu * wc[i] * wc[i] + M12 * ac[i]
                    + (p.fcs[i] * std::tanh(f.lambda * ts) + p.fvs[i] * ts)
                    + f.off_s;
        const double Gs = Qx * gy[i] - Qy * gx[i];
        const double Gb = ((p.Pbx[i] + f.m_u_known * f.dx) * gy[i]
                           - (p.Pby[i] + f.m_u_known * f.dy) * gx[i]) + Gs;
        hp = hp - Gb;
        hs = hs - Gs;

        // ── backlash_torque_np ──
        const double D = q.x[i] - q.y[i] - beta[i];
        const double Dd = d.x[i] - d.y[i];
        const double hh = 0.5 * p.bd[i];
        const double dz = smooth_relu(D - hh, f.eps) - smooth_relu(-D - hh, f.eps);
        const double tt = p.bk[i] * (dz + p.bt[i] * D) + p.bc[i] * Dd;

        // ── eom_backlash_np: h = [tt + fric_motor + off, hp − tt, hs] ──
        const double H0 = tt + (p.fcm[i] * std::tanh(f.lambda * d.x[i]) + p.fvm[i] * d.x[i])
                          + f.off_m;
        const double H1 = hp - tt;
        const double H2 = hs;

        // ── forward_accel_backlash_np: r = [τ_b, 0, τ_s] − h ──
        const double r0 = tau_b[i] - H0;
        const double r1 = 0.0 - H1;
        const double r2 = tau_s[i] - H2;
        const double det2 = M11 * p.Js[i] - M12 * M12;
        a.x[i] = r0 / p.Jm[i];
        a.y[i] = (p.Js[i] * r1 - M12 * r2) / det2;
        a.z[i] = (-M12 * r1 + M11 * r2) / det2;
    }
}

// ── 一个块（n ≤ MAX_BLK 条）从 t=0 前向 T 步；输出 [T,3] 写到 out ──
void rollout_block(int i0, int n, int T, int substeps, double dt, const Fixed &f,
                   const double *params, const double *seq_const, const double *seq_var,
                   double *theta_out, double *dtheta_out, std::vector<double> &scratch) {
    const double hh = dt / static_cast<double>(substeps);
    BlockParams p;
    Vec3 q, d, k1, k2, k3, k4, ta, tb_, tc;

    // 参数 → SoA（顺序 = PARAM_NAMES）
    for (int j = 0; j < n; ++j) {
        const double *pp = params + static_cast<std::size_t>(i0 + j) * NP;
        p.Jbig[j] = pp[0];  p.Js[j] = pp[1];   p.Px[j] = pp[2];   p.Py[j] = pp[3];
        p.fcb[j] = pp[4];   p.fvb[j] = pp[5];  p.fcs[j] = pp[6];  p.fvs[j] = pp[7];
        p.bd[j] = pp[8];    p.bk[j] = pp[9];   p.bc[j] = pp[10];  p.bt[j] = pp[11];
        p.Jm[j] = pp[12];   p.fcm[j] = pp[13]; p.fvm[j] = pp[14];
        p.Pbx[j] = pp[16];  p.Pby[j] = pp[17];             // pp[15]=β 由 seq_var 逐点给
        const double *sc = seq_const + static_cast<std::size_t>(i0 + j) * NC;
        q.x[j] = sc[0]; q.y[j] = sc[1]; q.z[j] = sc[2];
        d.x[j] = sc[3]; d.y[j] = sc[4]; d.z[j] = sc[5];
    }

    // 该块的 [T,5] 输入 gather 成"时间优先 + 块内连续"（tau_b/tau_s/gx/gy/beta/wc/ac）
    double *buf = scratch.data();
    double *b_tb = buf;                 // T*n
    double *b_ts = b_tb + static_cast<std::size_t>(T) * n;
    double *b_gx = b_ts + static_cast<std::size_t>(T) * n;
    double *b_gy = b_gx + static_cast<std::size_t>(T) * n;
    double *b_be = b_gy + static_cast<std::size_t>(T) * n;
    double *b_wc = b_be + static_cast<std::size_t>(T) * n;
    double *b_ac = b_wc + static_cast<std::size_t>(T) * n;
    for (int j = 0; j < n; ++j) {
        const std::size_t item = static_cast<std::size_t>(i0 + j);
        const double *sv = seq_var + item * static_cast<std::size_t>(T) * NV;
        const double wc = seq_const[item * NC + 6];
        const double ac = seq_const[item * NC + 7];
        for (int t = 0; t < T; ++t) {
            const std::size_t k = static_cast<std::size_t>(t) * n + j;
            const double *s = sv + static_cast<std::size_t>(t) * NV;
            b_tb[k] = s[0];
            b_ts[k] = s[1];
            b_gx[k] = s[2];
            b_gy[k] = s[3];
            b_be[k] = s[4];
            b_wc[k] = wc;
            b_ac[k] = ac;
        }
    }

    for (int t = 0; t < T; ++t) {
        // 记录**积分前**的状态（与 simulate_backlash_np 一致）
        for (int j = 0; j < n; ++j) {
            const std::size_t o = (static_cast<std::size_t>(i0 + j) * T + t) * 3;
            theta_out[o + 0] = q.x[j];
            theta_out[o + 1] = q.y[j];
            theta_out[o + 2] = q.z[j];
            dtheta_out[o + 0] = d.x[j];
            dtheta_out[o + 1] = d.y[j];
            dtheta_out[o + 2] = d.z[j];
        }
        const double *tb_t = b_tb + static_cast<std::size_t>(t) * n;
        const double *ts_t = b_ts + static_cast<std::size_t>(t) * n;
        const double *gx_t = b_gx + static_cast<std::size_t>(t) * n;
        const double *gy_t = b_gy + static_cast<std::size_t>(t) * n;
        const double *be_t = b_be + static_cast<std::size_t>(t) * n;
        const double *wc_t = b_wc + static_cast<std::size_t>(t) * n;
        const double *ac_t = b_ac + static_cast<std::size_t>(t) * n;

        for (int s = 0; s < substeps; ++s) {
            // k1
            accel_block(p, f, n, q, d, tb_t, ts_t, gx_t, gy_t, be_t, wc_t, ac_t, k1);
            // 中间量 a = q + 0.5·hh·d ; b = d + 0.5·hh·k1
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                ta.x[j] = q.x[j] + 0.5 * hh * d.x[j];
                ta.y[j] = q.y[j] + 0.5 * hh * d.y[j];
                ta.z[j] = q.z[j] + 0.5 * hh * d.z[j];
                tb_.x[j] = d.x[j] + 0.5 * hh * k1.x[j];
                tb_.y[j] = d.y[j] + 0.5 * hh * k1.y[j];
                tb_.z[j] = d.z[j] + 0.5 * hh * k1.z[j];
            }
            // k2 = accel(a, b)
            accel_block(p, f, n, ta, tb_, tb_t, ts_t, gx_t, gy_t, be_t, wc_t, ac_t, k2);
            // k3: a = q + 0.5·hh·(d + 0.5·hh·k1) ; b = d + 0.5·hh·k2
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                ta.x[j] = q.x[j] + 0.5 * hh * (d.x[j] + 0.5 * hh * k1.x[j]);
                ta.y[j] = q.y[j] + 0.5 * hh * (d.y[j] + 0.5 * hh * k1.y[j]);
                ta.z[j] = q.z[j] + 0.5 * hh * (d.z[j] + 0.5 * hh * k1.z[j]);
                tb_.x[j] = d.x[j] + 0.5 * hh * k2.x[j];
                tb_.y[j] = d.y[j] + 0.5 * hh * k2.y[j];
                tb_.z[j] = d.z[j] + 0.5 * hh * k2.z[j];
            }
            accel_block(p, f, n, ta, tb_, tb_t, ts_t, gx_t, gy_t, be_t, wc_t, ac_t, k3);
            // k4: a = q + hh·(d + 0.5·hh·k2) ; b = d + hh·k3
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                ta.x[j] = q.x[j] + hh * (d.x[j] + 0.5 * hh * k2.x[j]);
                ta.y[j] = q.y[j] + hh * (d.y[j] + 0.5 * hh * k2.y[j]);
                ta.z[j] = q.z[j] + hh * (d.z[j] + 0.5 * hh * k2.z[j]);
                tb_.x[j] = d.x[j] + hh * k3.x[j];
                tb_.y[j] = d.y[j] + hh * k3.y[j];
                tb_.z[j] = d.z[j] + hh * k3.z[j];
            }
            accel_block(p, f, n, ta, tb_, tb_t, ts_t, gx_t, gy_t, be_t, wc_t, ac_t, k4);
            // 状态更新（与 rk4_step_backlash_np 同序；两个更新都用**旧**状态）
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                const double c = hh / 6.0;
                tc.x[j] = q.x[j] + c * (d.x[j] + 2.0 * (d.x[j] + 0.5 * hh * k1.x[j])
                                        + 2.0 * (d.x[j] + 0.5 * hh * k2.x[j])
                                        + (d.x[j] + hh * k3.x[j]));
                tc.y[j] = q.y[j] + c * (d.y[j] + 2.0 * (d.y[j] + 0.5 * hh * k1.y[j])
                                        + 2.0 * (d.y[j] + 0.5 * hh * k2.y[j])
                                        + (d.y[j] + hh * k3.y[j]));
                tc.z[j] = q.z[j] + c * (d.z[j] + 2.0 * (d.z[j] + 0.5 * hh * k1.z[j])
                                        + 2.0 * (d.z[j] + 0.5 * hh * k2.z[j])
                                        + (d.z[j] + hh * k3.z[j]));
                k1.x[j] = d.x[j] + c * (k1.x[j] + 2.0 * k2.x[j] + 2.0 * k3.x[j] + k4.x[j]);
                k1.y[j] = d.y[j] + c * (k1.y[j] + 2.0 * k2.y[j] + 2.0 * k3.y[j] + k4.y[j]);
                k1.z[j] = d.z[j] + c * (k1.z[j] + 2.0 * k2.z[j] + 2.0 * k3.z[j] + k4.z[j]);
            }
            for (int j = 0; j < n; ++j) {      // 上一步把新角速度暂存在 k1 里
                q.x[j] = tc.x[j]; q.y[j] = tc.y[j]; q.z[j] = tc.z[j];
                d.x[j] = k1.x[j]; d.y[j] = k1.y[j]; d.z[j] = k1.z[j];
            }
        }
    }
}

}  // namespace

// ============================================================================
// C ABI（ctypes 用）
// ============================================================================
extern "C" {

/// 版本/构建信息（Python 侧打印用）。
const char *fast_sim_build_info(void) {
    return "fast_sim 1.0 (rk4 only; block-SoA vectorized; std::thread; -ffp-contract=off)";
}

/// 硬件并发数（nthreads=0 时的默认值）。
int fast_sim_hardware_threads(void) {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? static_cast<int>(hc) : 1;
}

/// 一次批量前向仿真。返回 0 成功；非 0 = 参数非法。
///
///   params     [B*18]   参数（顺序 = PARAM_NAMES；params[15]=β 忽略，β 由 seq_var 逐点给）
///   seq_const  [B*8]    q0(3) qd0(3) base_omega base_alpha
///   seq_var    [B*T*5]  tau_big tau_small grav_x grav_y beta
///   theta_out  [B*T*3]  θ（电机/云台/小 yaw），记录**积分前**的状态
///   dtheta_out [B*T*3]  θ̇
///   nthreads   ≤0 ⇒ 用 hardware_concurrency；block ≤0 ⇒ 默认 64
int fast_sim_rollout(int B, int T, int substeps, double dt,
                     const double *params, const double *seq_const, const double *seq_var,
                     double dx, double dy, double m_u_known, double friction_lambda,
                     double backlash_smooth_eps, double tau_offset_big,
                     double tau_offset_small, double tau_offset_motor,
                     double *theta_out, double *dtheta_out, int nthreads, int block) {
    if (B <= 0 || T <= 0 || substeps <= 0 || dt <= 0.0) return 1;
    if (!params || !seq_const || !seq_var || !theta_out || !dtheta_out) return 2;

    Fixed f;
    f.dx = dx;
    f.dy = dy;
    f.m_u_known = m_u_known;
    f.lambda = friction_lambda;
    f.eps = backlash_smooth_eps;
    f.off_b = tau_offset_big;
    f.off_s = tau_offset_small;
    f.off_m = tau_offset_motor;

    int blk = (block > 0) ? block : 64;
    blk = std::min(blk, MAX_BLK);
    int nth = (nthreads > 0) ? nthreads : fast_sim_hardware_threads();
    nth = std::max(1, std::min(nth, B));

    // 每个线程处理一段连续的 item 区间（区间内再按 block 切）
    const int per = (B + nth - 1) / nth;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(nth));
    for (int k = 0; k < nth; ++k) {
        const int lo = k * per;
        const int hi = std::min(B, lo + per);
        if (lo >= hi) break;
        pool.emplace_back([&, lo, hi, blk]() {
            std::vector<double> scratch(static_cast<std::size_t>(T) * blk * 7);
            for (int i0 = lo; i0 < hi; i0 += blk) {
                const int n = std::min(blk, hi - i0);
                rollout_block(i0, n, T, substeps, dt, f, params, seq_const, seq_var,
                              theta_out, dtheta_out, scratch);
            }
        });
    }
    for (auto &th : pool) th.join();
    return 0;
}

}  // extern "C"
