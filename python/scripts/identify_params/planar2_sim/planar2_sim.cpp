// ============================================================================
// planar2_sim.cpp —— 手写的高效率 **2 自由度**（大/小 yaw，A 系重力）前向仿真
// ============================================================================
// ★ 与 python/scripts/identify_params/planar2.py 的 numpy 参考实现逐项同序:
//     derived_np / gravity_from_Aframe / accel_np / rk4_step_np / rollout_np
//   因此结果与 rollout_np 一致到 ~1e-13（残差只来自 libm 与 numpy 的
//   sin/cos/tanh 实现差异；sqrt 两边都是 IEEE 精确）。**唯一**的"翻译"是:
//   plan2 的 accel_np 吃的是**世界系**重力 (g_x,g_y)+psi_off，本内核吃的是
//   **A 系**逐样本 (g_ax,g_ay)，用的是 gravity_from_Aframe 的 A 系写法
//   （plan2 文件头已证明两者严格等价: g_w = R(ψ_b)·g_A）。二者之间的换算见 selftest。
// ★ 同一组输入下，**不同线程数 / 分块宽度**的结果彼此逐位一致: 线程只按 batch
//   切连续区间、块内按 item 独立计算，没有任何跨条目归约 ⇒ 并行不改变浮点结果。
//
// 参数（每个 batch 元素 11 个 double，**物理量、非 log/raw 空间**；顺序 = PARAM_NAMES2）:
//     0 X_b  1 Y_b  2 X_s  3 Y_s  4 I_b  5 I_s  6 mu
//     7 f_bc 8 f_bv 9 f_sc 10 f_sv
//
// 已知常量（编译期，不是参数）: dx = 0.0, dy = 0.07（D 向量）, lambda = 100.0
//
// 性能手段（照抄 fast_sim 的工程风格）:
//   ① 分块向量化: batch 切成 block（默认 32），块内参数/状态按 SoA 存放
//      （`double a[MAX_BLK]`），内层连续 ⇒ 编译器可自动向量化（sin/cos/tanh 走 libm）。
//   ② std::thread: batch 各条目彼此独立 ⇒ 按块把 batch 切给多个线程，无同步。
//   ③ 数据预取: 每个块进入时把该块的 [T,4] 输入 gather 成"时间优先 + 块内连续"。
//
// ★ 本文件与主工程（include/tcbs/...、CMakeLists.txt）**没有任何关系**:
//   不 include 任何项目头文件、不参与项目的 CMake 构建，只用 C++17 标准库 + libm。
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int NP = 11;          // 参数个数（顺序 = PARAM_NAMES2）
constexpr int NC = 5;           // seq_const 通道: q0_b q0_s qd0_b qd0_s base_omega
constexpr int NV = 4;           // seq_var  通道: tau_b tau_s g_ax g_ay
constexpr int MAX_BLK = 256;

// ── 已知几何 / 摩擦陡度（编译期常量，与 planar2.py 的 Planar2Params 默认一致）──
constexpr double KDX = 0.0;         // D_x
constexpr double KDY = 0.07;        // D_y
constexpr double KLAMBDA = 100.0;   // tanh 软符号陡度

// 块内**参数派生量**（SoA；长度 = block）。
//   只留 accel 真正用到的量: A, B=J_s, X_s, Y_s, X_b+mu·dx, Y_b+mu·dy, 4 个摩擦。
//   J_b / J_s / A / B 只依赖参数，与状态无关 ⇒ 每个 item 只算一次。
struct BlockParams {
    double A[MAX_BLK], B[MAX_BLK];
    double Xs[MAX_BLK], Ys[MAX_BLK];
    double Xbe[MAX_BLK], Ybe[MAX_BLK];
    double fbc[MAX_BLK], fbv[MAX_BLK], fsc[MAX_BLK], fsv[MAX_BLK];
};

// 块内状态（SoA）。★ A 系重力的 G_b 不含 q_b ⇒ accel 只需要 q_s/v_b/v_s；
//   q_b 仍然积分（它的导数就是 v_b），只是不进右端。
struct State {
    double qb[MAX_BLK], qs[MAX_BLK], vb[MAX_BLK], vs[MAX_BLK];
};

struct Acc {
    double b[MAX_BLK], s[MAX_BLK];
};

// 参数 → BlockParams（顺序 = PARAM_NAMES2；与 planar2.py 的 J_b/J_s/A/B 同序）
void load_params(const double *params, int i0, int n, BlockParams &p) {
    for (int j = 0; j < n; ++j) {
        const double *pp = params + static_cast<std::size_t>(i0 + j) * NP;
        const double Xb = pp[0], Yb = pp[1], Xs = pp[2], Ys = pp[3];
        const double Ib = pp[4], Is = pp[5], mu = pp[6];
        const double Jb = Ib + Xb * Xb + Yb * Yb;                    // = I_b + |P_b|²
        const double Js = Is + (Xs * Xs + Ys * Ys) / mu;             // = I_s + |P_s|²/mu
        p.A[j] = Jb + mu * (KDX * KDX + KDY * KDY);                  // = J_b + mu·|D|²
        p.B[j] = Js;                                                 // = J_s
        p.Xs[j] = Xs;
        p.Ys[j] = Ys;
        p.Xbe[j] = Xb + mu * KDX;                                    // X_b + mu·D_x
        p.Ybe[j] = Yb + mu * KDY;                                    // Y_b + mu·D_y
        p.fbc[j] = pp[7];
        p.fbv[j] = pp[8];
        p.fsc[j] = pp[9];
        p.fsv[j] = pp[10];
    }
}

// ── 块内加速度 q̈ = M⁻¹R（2-DOF；与 derived_np+gravity_from_Aframe+accel_np 同序）──
//
//   g_ax/g_ay 是 **A 系**（随 b 转）逐样本重力分量，wc = base_omega。
//   `#pragma GCC ivdep` = 各 lane 独立（无归约），允许编译器放心向量化。
void accel_block(const BlockParams &p, int n, const State &x,
                 const double *tau_b, const double *tau_s,
                 const double *g_ax, const double *g_ay, const double *wc,
                 Acc &out) {
#pragma GCC ivdep
    for (int i = 0; i < n; ++i) {
        // ── derived_np(q_s) ──
        const double qs = x.qs[i];
        const double cs = std::cos(qs);
        const double sn = std::sin(qs);
        const double Qx = p.Xs[i] * cs - p.Ys[i] * sn;      // Q = R(q_s)·(X_s,Y_s)
        const double Qy = p.Xs[i] * sn + p.Ys[i] * cs;
        const double muK = KDX * Qx + KDY * Qy;             // D·Q
        const double muKp = KDY * Qx - KDX * Qy;            // d(D·Q)/d q_s
        const double Delta = p.A[i] * p.B[i] - muK * muK;   // > 0（I_b,I_s,mu>0）

        // ── gravity_from_Aframe（A 系写法；与 ψ 写法严格等价）──
        const double gx = g_ax[i], gy = g_ay[i];
        const double Gs = p.Xs[i] * (gx * sn - gy * cs) + p.Ys[i] * (gx * cs + gy * sn);
        const double Gb = -p.Xbe[i] * gy + p.Ybe[i] * gx + Gs;

        // ── accel_np: 广义力 / 右端 / M⁻¹ ──
        const double vb = x.vb[i], vs = x.vs[i], w = wc[i];
        const double Qb = tau_b[i] - p.fbc[i] * std::tanh(KLAMBDA * vb) - p.fbv[i] * vb;
        const double Qsj = tau_s[i] - p.fsc[i] * std::tanh(KLAMBDA * vs) - p.fsv[i] * vs;
        const double Rb = Qb - Gb - muKp * vs * (2.0 * (w + vb) + vs);
        const double wp = w + vb;
        const double Rs = Qsj - Gs + muKp * (wp * wp);
        const double BpK = p.B[i] + muK;
        out.b[i] = (p.B[i] * Rb - BpK * Rs) / Delta;
        out.s[i] = ((p.A[i] + p.B[i] + 2.0 * muK) * Rs - BpK * Rb) / Delta;
    }
}

// ── 一个块（n ≤ MAX_BLK 条）从 t=0 前向 T 步；记录**积分前**状态 ──
void rollout_block(int i0, int n, int T, int substeps, double dt,
                   const double *params, const double *seq_const, const double *seq_var,
                   double *theta_out, double *dtheta_out, std::vector<double> &scratch) {
    const double hh = dt / static_cast<double>(substeps);
    BlockParams p;
    State x, t;
    Acc k1, k2, k3, k4;
    double wcbuf[MAX_BLK];

    load_params(params, i0, n, p);
    for (int j = 0; j < n; ++j) {
        const double *sc = seq_const + static_cast<std::size_t>(i0 + j) * NC;
        x.qb[j] = sc[0];
        x.qs[j] = sc[1];
        x.vb[j] = sc[2];
        x.vs[j] = sc[3];
        wcbuf[j] = sc[4];
    }

    // 该块的 [T,4] 输入 gather 成"时间优先 + 块内连续"（tau_b/tau_s/g_ax/g_ay）
    double *buf = scratch.data();
    double *b_tb = buf;                       // T*n
    double *b_ts = b_tb + static_cast<std::size_t>(T) * n;
    double *b_gx = b_ts + static_cast<std::size_t>(T) * n;
    double *b_gy = b_gx + static_cast<std::size_t>(T) * n;
    for (int j = 0; j < n; ++j) {
        const std::size_t item = static_cast<std::size_t>(i0 + j);
        const double *sv = seq_var + item * static_cast<std::size_t>(T) * NV;
        for (int tg = 0; tg < T; ++tg) {
            const std::size_t k = static_cast<std::size_t>(tg) * n + j;
            const double *s = sv + static_cast<std::size_t>(tg) * NV;
            b_tb[k] = s[0];
            b_ts[k] = s[1];
            b_gx[k] = s[2];
            b_gy[k] = s[3];
        }
    }

    for (int tt = 0; tt < T; ++tt) {
        // 记录**积分前**的状态（与 rollout_np 一致）
        for (int j = 0; j < n; ++j) {
            const std::size_t o = (static_cast<std::size_t>(i0 + j) * T + tt) * 2;
            theta_out[o + 0] = x.qb[j];
            theta_out[o + 1] = x.qs[j];
            dtheta_out[o + 0] = x.vb[j];
            dtheta_out[o + 1] = x.vs[j];
        }
        const double *tb_t = b_tb + static_cast<std::size_t>(tt) * n;
        const double *ts_t = b_ts + static_cast<std::size_t>(tt) * n;
        const double *gx_t = b_gx + static_cast<std::size_t>(tt) * n;
        const double *gy_t = b_gy + static_cast<std::size_t>(tt) * n;

        // 力矩在子步内零阶保持 ⇒ 显式 RK4（与 rk4_step_np 逐式同序）
        for (int s = 0; s < substeps; ++s) {
            // k1 = accel(q, v)
            accel_block(p, n, x, tb_t, ts_t, gx_t, gy_t, wcbuf, k1);
            // k2 = accel(q + 0.5h·v, v + 0.5h·k1)
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                t.qs[j] = x.qs[j] + 0.5 * hh * x.vs[j];
                t.vb[j] = x.vb[j] + 0.5 * hh * k1.b[j];
                t.vs[j] = x.vs[j] + 0.5 * hh * k1.s[j];
            }
            accel_block(p, n, t, tb_t, ts_t, gx_t, gy_t, wcbuf, k2);
            // k3 = accel(q + 0.5h·(v + 0.5h·k1), v + 0.5h·k2)
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                t.qs[j] = x.qs[j] + 0.5 * hh * (x.vs[j] + 0.5 * hh * k1.s[j]);
                t.vb[j] = x.vb[j] + 0.5 * hh * k2.b[j];
                t.vs[j] = x.vs[j] + 0.5 * hh * k2.s[j];
            }
            accel_block(p, n, t, tb_t, ts_t, gx_t, gy_t, wcbuf, k3);
            // k4 = accel(q + h·(v + 0.5h·k2), v + h·k3)
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                t.qs[j] = x.qs[j] + hh * (x.vs[j] + 0.5 * hh * k2.s[j]);
                t.vb[j] = x.vb[j] + hh * k3.b[j];
                t.vs[j] = x.vs[j] + hh * k3.s[j];
            }
            accel_block(p, n, t, tb_t, ts_t, gx_t, gy_t, wcbuf, k4);
            // 状态更新（与 rk4_step_np 同序；q 与 v 的更新都用**旧**状态）
#pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                const double cq = hh * hh / 6.0;
                const double cv = hh / 6.0;
                const double qbn = x.qb[j] + hh * x.vb[j] + cq * (k1.b[j] + k2.b[j] + k3.b[j]);
                const double qsn = x.qs[j] + hh * x.vs[j] + cq * (k1.s[j] + k2.s[j] + k3.s[j]);
                const double vbn = x.vb[j] + cv * (k1.b[j] + 2.0 * k2.b[j] + 2.0 * k3.b[j] + k4.b[j]);
                const double vsn = x.vs[j] + cv * (k1.s[j] + 2.0 * k2.s[j] + 2.0 * k3.s[j] + k4.s[j]);
                x.qb[j] = qbn;
                x.qs[j] = qsn;
                x.vb[j] = vbn;
                x.vs[j] = vsn;
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
const char *planar2_build_info(void) {
    return "planar2_sim 1.0 (2-DOF A-frame gravity; rk4 only; block-SoA vectorized; "
           "std::thread; -ffp-contract=off; D=(0,0.07), lambda=100)";
}

/// 硬件并发数（nthreads=0 时的默认值）。
int planar2_hardware_threads(void) {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? static_cast<int>(hc) : 1;
}

/// 一次批量前向仿真。返回 0 成功；非 0 = 参数非法。
///
///   params     [B*11]   参数（顺序 = PARAM_NAMES2；物理量）
///   seq_const  [B*5]    q0_b q0_s qd0_b qd0_s base_omega
///   seq_var    [B*T*4]  tau_b tau_s g_ax g_ay（A 系逐样本重力）
///   theta      [B*T*2]  θ=(q_b,q_s)，记录**积分前**的状态
///   dtheta     [B*T*2]  θ̇=(v_b,v_s)
///   nthreads   ≤0 ⇒ 用 hardware_concurrency；block ≤0 ⇒ 默认 32，上限 MAX_BLK
int planar2_rollout(const double *params, int B, const double *seq_const,
                    const double *seq_var, int T, double dt, int substeps,
                    int nthreads, int block, double *theta, double *dtheta) {
    if (B <= 0 || T <= 0 || substeps <= 0 || dt <= 0.0) return 1;
    if (!params || !seq_const || !seq_var || !theta || !dtheta) return 2;

    int blk = (block > 0) ? block : 32;
    blk = std::min(blk, MAX_BLK);
    int nth = (nthreads > 0) ? nthreads : planar2_hardware_threads();
    nth = std::max(1, std::min(nth, B));

    // 每个线程处理一段连续 item 区间（区间内再按 block 切）⇒ 结果与线程数/块宽无关
    const int per = (B + nth - 1) / nth;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(nth));
    for (int k = 0; k < nth; ++k) {
        const int lo = k * per;
        const int hi = std::min(B, lo + per);
        if (lo >= hi) break;
        pool.emplace_back([&, lo, hi, blk]() {
            std::vector<double> scratch(static_cast<std::size_t>(T) * blk * NV);
            for (int i0 = lo; i0 < hi; i0 += blk) {
                const int n = std::min(blk, hi - i0);
                rollout_block(i0, n, T, substeps, dt, params, seq_const, seq_var,
                              theta, dtheta, scratch);
            }
        });
    }
    for (auto &th : pool) th.join();
    return 0;
}

/// 逐点加速度对拍用（batch，**只算一步 accel，不积分**）。返回 0 成功。
///
///   params     [B*11]  同上
///   state      [B*4]   q_b q_s v_b v_s
///   u          [B*4]   tau_b tau_s g_ax g_ay（A 系重力）
///   base_omega [B]     ω_c
///   qdd        [B*2]   输出 q̈=(q̈_b,q̈_s)
///
/// ★ 与 rollout 走**完全相同**的 accel_block ⇒ 对拍的是同一段代码。
int planar2_accel(const double *params, int B, const double *state, const double *u,
                  const double *base_omega, double *qdd) {
    if (B <= 0) return 1;
    if (!params || !state || !u || !base_omega || !qdd) return 2;

    const int blk = MAX_BLK;
    for (int i0 = 0; i0 < B; i0 += blk) {
        const int n = std::min(blk, B - i0);
        BlockParams p;
        State x;
        Acc out;
        double tb[MAX_BLK], ts[MAX_BLK], gx[MAX_BLK], gy[MAX_BLK], wc[MAX_BLK];
        load_params(params, i0, n, p);
        for (int j = 0; j < n; ++j) {
            const double *s = state + static_cast<std::size_t>(i0 + j) * 4;
            x.qb[j] = s[0];
            x.qs[j] = s[1];
            x.vb[j] = s[2];
            x.vs[j] = s[3];
            const double *uu = u + static_cast<std::size_t>(i0 + j) * 4;
            tb[j] = uu[0];
            ts[j] = uu[1];
            gx[j] = uu[2];
            gy[j] = uu[3];
            wc[j] = base_omega[i0 + j];
        }
        accel_block(p, n, x, tb, ts, gx, gy, wc, out);
        for (int j = 0; j < n; ++j) {
            double *o = qdd + static_cast<std::size_t>(i0 + j) * 2;
            o[0] = out.b[j];
            o[1] = out.s[j];
        }
    }
    return 0;
}

}  // extern "C"
