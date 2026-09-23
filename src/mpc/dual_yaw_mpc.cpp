#include "tcbs/mpc/dual_yaw_mpc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <atomic>

#include <ceres/ceres.h>

namespace tcbs {

namespace dual_yaw {

// 诊断用: 代价函数被求值次数（每次 Jacobian / 线搜索各计一次）
static std::atomic<uint64_t> g_cost_eval_count{0};
uint64_t dualYawMpcCostEvaluations() { return g_cost_eval_count.load(); }
void dualYawMpcResetCostEvaluations() { g_cost_eval_count = 0; }

namespace {

// 兼容 double / ceres::Jet 的 clamp（Jet 比较用标量部分）
template <typename T>
inline T clampT(const T& x, double lo, double hi) {
    if (x < lo) return T(lo);
    if (x > hi) return T(hi);
    return x;
}

// 平滑绝对值: sqrt(sqrt(err²+a))，平方后 = sqrt(err²+a)（与旧实现一致）
template <typename T>
inline T smoothAbs(const T& err, double eps) {
    using std::sqrt;
    return sqrt(sqrt(err * err + T(eps)));
}

// 单侧 4 次幂软限位形状（s = 超出软限位的相对量；s≤0 时为 0）
// s⁴ 在 s=0 处函数与一阶/二阶导都连续，便于优化收敛
template <typename T>
inline T limitPenaltyOneSide(const T& s) {
    if (!(s > T(0.0))) return T(0.0);
    return s * s * s * s;
}

// 小 yaw 的软限位（两侧各自从硬限位向行程内缩 (1−ratio)·总行程）:
//     inset    = (1 − ratio)·(max − min)
//     soft_min = min + inset          （负侧软限位，距离 min 侧 inset）
//     soft_max = max − inset          （正侧软限位，距离 max 侧 inset）
// 当前行程对称 ±30°、默认 ratio=0.75 ⇒ inset = 15°，
//     soft_min = −15°, soft_max = +15°。
// ★ 两侧**各自独立推导**（不假设行程对称）。旧实现是 `soft = ratio·max_angle` + `|θ|`
//   比较，隐含"行程对称 ±max_angle": 在非对称的 [−25°, +20°] 下负侧会被当成软限位
//   −0.75·20° = −15°、障碍宽度 hard−soft = 5°（真实余量是 11.25°），于是 θ 还没到 −20°
//   就已经吃到 1 倍惩罚、到 −25° 是 16 倍 —— 负侧行程基本用不满（且量纲随行程变化）。
//   现在的写法对对称/非对称行程都成立。
struct SmallSoftLimits {
    double lo = -1e9;      // 负侧软限位（≤ hard_lo）
    double hi = 1e9;       // 正侧软限位（≥ hard_hi 侧向内）
    double span_lo = 1e9;  // 负侧从软限位到硬限位的行程（>0）
    double span_hi = 1e9;  // 正侧从软限位到硬限位的行程（>0）
};

inline SmallSoftLimits smallSoftLimits(const DualYawMpcConfig& cfg) {
    SmallSoftLimits s;
    const double lo = cfg.small.min_angle;
    const double hi = cfg.small.max_angle;
    if (!(hi > lo)) {
        // 配置退化（min ≥ max）: 退化为"无软限位区"，由 checkLimitConfig() 告警
        s.lo = lo; s.hi = hi;
        s.span_lo = std::max(1e-9, 1.0);
        s.span_hi = std::max(1e-9, 1.0);
        return s;
    }
    const double travel = hi - lo;
    double ratio = cfg.small_limit_soft_ratio;
    if (!(ratio >= 0.0)) ratio = 0.0;                       // NaN/负数 → 退化为 ratio=0
    double inset = (1.0 - ratio) * travel;
    if (inset < 0.0) inset = 0.0;                            // ratio > 1 → 不回缩
    if (inset > 0.5 * travel) inset = 0.5 * travel;          // 两侧软限位不交叉
    s.lo = lo + inset;
    s.hi = hi - inset;
    s.span_lo = std::max(1e-9, inset);
    s.span_hi = std::max(1e-9, inset);
    return s;
}

// 双侧 4 次幂软限位障碍（保留原单侧 s⁴ 的形状与量纲, 只是两侧各自归一化并求和）:
//   s_hi = (θ − soft_hi)/span_hi （越出正侧软限位为正）
//   s_lo = (soft_lo − θ)/span_lo （越出负侧软限位为正）
// 由于 soft_lo ≤ soft_hi，两侧不会同时激活；s=0 处仍保持 C² 连续。
template <typename T>
inline T limitPenalty(const T& angle, const SmallSoftLimits& sl) {
    const T over_hi = (angle - T(sl.hi)) / T(sl.span_hi);
    const T over_lo = (T(sl.lo) - angle) / T(sl.span_lo);
    return limitPenaltyOneSide(over_hi) + limitPenaltyOneSide(over_lo);
}

} // namespace

// ============================================================================
// 代价函数
// ============================================================================
class DualYawMpcCost {
public:
    DualYawMpcCost(const ModelParams& model, const DualYawMpcConfig& cfg,
                   const DualYawMpc::Input& in,
                   const std::vector<double>& ref_big, const std::vector<double>& ref_small)
        : model_(model), cfg_(cfg), in_(in), ref_big_(ref_big), ref_small_(ref_small) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const;

private:
    const ModelParams& model_;
    const DualYawMpcConfig& cfg_;
    const DualYawMpc::Input& in_;
    const std::vector<double>& ref_big_;
    const std::vector<double>& ref_small_;
};

template <typename T>
bool DualYawMpcCost::operator()(T const* const* parameters, T* residuals) const {
    using std::sqrt;
    ++g_cost_eval_count;
    const T* d_raw = parameters[0];
    const int N = cfg_.N;
    // 力矩变化率硬约束: 在代价函数内部 clamp（等价于参数箱式边界，
    // 但允许上层使用无边界信任域求解器，显著减少函数求值次数）
    const double rs_b = cfg_.big.max_torque_rate * cfg_.dt_control;
    const double rs_s = cfg_.small.max_torque_rate * cfg_.dt_control;
    std::vector<T> d(2 * N);
    for (int k = 0; k < N; ++k) {
        d[k] = clampT(d_raw[k], -rs_b, rs_b);
        d[N + k] = clampT(d_raw[N + k], -rs_s, rs_s);
    }

    // ── 1. 由增量 d 重建两关节力矩序列（首步相对上一步实际力矩）──
    std::vector<T> ub(N), us(N);
    ub[0] = clampT(T(in_.prev_torque[0]) + d[0],
                   -cfg_.big.max_torque, cfg_.big.max_torque);
    for (int k = 1; k < N; ++k) {
        ub[k] = clampT(ub[k - 1] + d[k], -cfg_.big.max_torque, cfg_.big.max_torque);
    }
    us[0] = clampT(T(in_.prev_torque[1]) + d[N],
                   -cfg_.small.max_torque, cfg_.small.max_torque);
    for (int k = 1; k < N; ++k) {
        us[k] = clampT(us[k - 1] + d[N + k], -cfg_.small.max_torque, cfg_.small.max_torque);
    }

    // ── 2. 前向预测 ──
    T q[3] = {T(in_.q[0]), T(in_.q[1]), T(in_.q[2])};
    T qd[3] = {T(in_.qd[0]), T(in_.qd[1]), T(in_.qd[2])};

    const double sw_b = std::sqrt(cfg_.w_big_azimuth);
    const double sw_s = std::sqrt(cfg_.w_small_azimuth);
    const double sw_vb = std::sqrt(cfg_.w_big_rate);      // ★ 速度惩罚（云台侧 θ̇）
    const double sw_vs = std::sqrt(cfg_.w_small_rate);    // ★ 速度惩罚（小 yaw θ̇）
    const double sw_c = std::sqrt(cfg_.w_small_center);
    const double sw_l = std::sqrt(cfg_.w_small_limit);
    const double sr_b = std::sqrt(cfg_.r_big_torque);
    const double sr_s = std::sqrt(cfg_.r_small_torque);
    const double srd_b = std::sqrt(cfg_.rd_big_rate);
    const double srd_s = std::sqrt(cfg_.rd_small_rate);

    // 小 yaw 软限位（两侧各自从硬限位向内回缩 (1−ratio)·总行程 ⇒ 非对称行程也正确）
    const SmallSoftLimits sl = smallSoftLimits(cfg_);

    int idx = 0;
    for (int k = 0; k < N; ++k) {
        const ModelExo& e = in_.exo;   // 平面模型: pitch 不进动力学；含背隙中心 β
        // ★ 3-DOF: 力矩只有两个通道 —— 大 yaw 作用在**电机**（q[0]），小 yaw 直接驱动（q[2]）
        const T u3[3] = {ub[k], T(0.0), us[k]};
        T qn[3], qdn[3];
        if (cfg_.use_rk4) {
            integrateStepBacklash(q, qd, u3, model_, e, cfg_.dt_control, cfg_.substeps, qn, qdn);
        } else {
            // 半隐式（辛）欧拉: 先更新速度再更新位置
            T acc[3];
            forwardAccelBacklash(q, qd, u3, model_, e, acc);
            for (int i = 0; i < 3; ++i) {
                qdn[i] = qd[i] + T(cfg_.dt_control) * acc[i];
                qn[i] = q[i] + T(cfg_.dt_control) * qdn[i];
            }
        }
        for (int i = 0; i < 3; ++i) { q[i] = qn[i]; qd[i] = qdn[i]; }

        // ── 3. 世界方位角预测（底盘转动在窗内线性外推）──
        const T t = T((k + 1) * cfg_.dt_control);
        const T psi_c = T(in_.chassis_azimuth) + T(in_.chassis_rate) * t;
        // 世界方位角: 大 yaw 用**云台**侧 q[1]（跟踪代价作用在云台上），小 yaw 用 q[2]
        const T psi_b = psi_c + q[1];
        const T psi_s = psi_b + q[2];

        // ★ 大 yaw: **平方**误差（残差写成 √w·e ⇒ 代价 = w·e²）；小 yaw: 平滑绝对误差
        residuals[idx++] = T(sw_b) * (psi_b - T(ref_big_[k]));
        residuals[idx++] = T(sw_s) * smoothAbs(psi_s - T(ref_small_[k]), cfg_.smooth_eps);
        // ★ 速度惩罚: qd[1] = 云台角速度、qd[2] = 小 yaw 角速度（不是电机侧 qd[0]）
        residuals[idx++] = T(sw_vb) * qd[1];
        residuals[idx++] = T(sw_vs) * qd[2];
        residuals[idx++] = T(sr_b) * ub[k];
        residuals[idx++] = T(sr_s) * us[k];
        // 回中到**行程中心**（非对称行程下 ≠ 0, 由配置显式给出）
        residuals[idx++] = T(sw_c) * (q[2] - T(cfg_.small_center_angle));
        residuals[idx++] = T(sw_l) * limitPenalty(q[2], sl);
        if (k > 0) {
            residuals[idx++] = T(srd_b) * d[k];
            residuals[idx++] = T(srd_s) * d[N + k];
        }
    }
    return true;
}

// ============================================================================
// DualYawMpc
// ============================================================================
DualYawMpc::DualYawMpc(const ModelParams& model, const DualYawMpcConfig& cfg)
    : model_(model), cfg_(cfg) {
    if (cfg_.N < 1) throw std::invalid_argument("DualYawMpc: N must be >= 1");
    checkIntegrability();
    checkLimitConfig();
    reset();
}

void DualYawMpc::checkLimitConfig() const {
    const double lo = cfg_.small.min_angle;
    const double hi = cfg_.small.max_angle;
    if (!(hi > lo)) {
        printf("[DualYawMpc][警告] 小 yaw 行程配置退化: min_angle=%.6f rad 不小于 max_angle=%.6f rad"
               "（软限位/障碍项将退化为无软限位区, 只靠 max_torque 兜底）。\n", lo, hi);
    }
    // 回中中心必须落在行程内, 否则回中代价会把关节往行程外拉（非对称行程最易写错）
    if (cfg_.small_center_angle < lo || cfg_.small_center_angle > hi) {
        printf("[DualYawMpc][警告] small_center_angle=%.6f rad (=%.2f°) 不在小 yaw 行程 "
               "[%.6f, %.6f] rad (=[%.2f°, %.2f°]) 内：回中代价会把关节往行程外拉, 请检查配置。\n",
               cfg_.small_center_angle, cfg_.small_center_angle * 180.0 / M_PI,
               lo, hi, lo * 180.0 / M_PI, hi * 180.0 / M_PI);
    }
    if (!(cfg_.small_limit_soft_ratio >= 0.0 && cfg_.small_limit_soft_ratio < 1.0)) {
        printf("[DualYawMpc][警告] small_limit_soft_ratio=%.4f 不在 [0, 1) 内："
               "软限位区宽度 (1−ratio)·总行程 将被钳位（ratio≥1 时退化为无软限位区）。\n",
               cfg_.small_limit_soft_ratio);
    }
}

void DualYawMpc::checkIntegrability() const {
    // ★ 判据必须用**积分子步**的有效步长 dt/substeps（λ=100 时靠细分子步才稳定）
    const int sub = (cfg_.substeps > 0) ? cfg_.substeps : 1;
    const double dt_eff = cfg_.dt_control / static_cast<double>(sub);
    const double lam_max = recommendedFrictionLambda(model_, dt_eff, cfg_.use_rk4);
    if (model_.frictionLambda > lam_max && lam_max > 0.0) {
        printf("[DualYawMpc][警告] frictionLambda=%.1f 超过数值可积建议上限 %.1f "
               "(dt=%.4f, 子步=%d ⇒ 有效步长 %.4f ms, %s)：摩擦模态在 ω≈0 附近会失稳，"
               "建议降低 λ、增加 substeps 或减小 dt。\n",
               model_.frictionLambda, lam_max, cfg_.dt_control, sub, dt_eff * 1e3,
               cfg_.use_rk4 ? "RK4" : "semi-implicit Euler");
    }
}

void DualYawMpc::reset() {
    last_torque_seq_.assign(2 * cfg_.N, 0.0);
}

int DualYawMpc::residualSize() const {
    // 每步: 大 yaw 跟踪 + 小 yaw 跟踪 + **大 yaw 速度 + 小 yaw 速度** + u_b + u_s + 回中 + 软限位 = 8
    // 另加 k≥1 的 2 条力矩变化率 ⇒ 8N + 2(N−1)
    return cfg_.N * 8 + (cfg_.N - 1) * 2;
}

DualYawMpc::Output DualYawMpc::solve(const Input& in) {
    Output out;
    const int N = cfg_.N;
    const int n_res = residualSize();
    out.residual_size = n_res;

    // ── 参考序列补齐 ──
    std::vector<double> ref_big(N), ref_small(N);
    const double hold_big = in.platform_azimuth;
    const double hold_small = in.platform_azimuth + in.q[2];
    for (int k = 0; k < N; ++k) {
        if (k < static_cast<int>(in.ref_big_azimuth.size())) {
            ref_big[k] = in.ref_big_azimuth[k];
        } else if (!in.ref_big_azimuth.empty()) {
            ref_big[k] = in.ref_big_azimuth.back();
        } else {
            ref_big[k] = hold_big;
        }
        if (k < static_cast<int>(in.ref_small_azimuth.size())) {
            ref_small[k] = in.ref_small_azimuth[k];
        } else if (!in.ref_small_azimuth.empty()) {
            ref_small[k] = in.ref_small_azimuth.back();
        } else {
            ref_small[k] = hold_small;
        }
    }
    // 小 yaw 参考换到关节系（世界方位角之差）后, 两侧**独立**判定是否落在软限位区外
    {
        const SmallSoftLimits sl = smallSoftLimits(cfg_);
        const double ref_small_joint = ref_small[0] - ref_big[0];
        out.small_ref_over_limit = (ref_small_joint > sl.hi) || (ref_small_joint < sl.lo);
    }

    // ── 热启动初值（平移上一次最优力矩序列 → 增量）──
    const double rate_step_b = cfg_.big.max_torque_rate * cfg_.dt_control;
    const double rate_step_s = cfg_.small.max_torque_rate * cfg_.dt_control;

    std::vector<double> u_init(2 * N, 0.0);
    if (static_cast<int>(last_torque_seq_.size()) == 2 * N) {
        for (int k = 0; k < N - 1; ++k) {
            u_init[k] = last_torque_seq_[k + 1];
            u_init[N + k] = last_torque_seq_[N + k + 1];
        }
        u_init[N - 1] = last_torque_seq_[N - 1];
        u_init[2 * N - 1] = last_torque_seq_[2 * N - 1];
    }
    std::vector<double> d(2 * N, 0.0);
    d[0] = clampT(u_init[0] - in.prev_torque[0], -rate_step_b, rate_step_b);
    for (int k = 1; k < N; ++k) {
        d[k] = clampT(u_init[k] - u_init[k - 1], -rate_step_b, rate_step_b);
    }
    d[N] = clampT(u_init[N] - in.prev_torque[1], -rate_step_s, rate_step_s);
    for (int k = 1; k < N; ++k) {
        d[N + k] = clampT(u_init[N + k] - u_init[N + k - 1], -rate_step_s, rate_step_s);
    }

    auto t_start = std::chrono::steady_clock::now();

    ceres::Problem problem;
    auto* cost = new ceres::DynamicAutoDiffCostFunction<DualYawMpcCost>(
        new DualYawMpcCost(model_, cfg_, in, ref_big, ref_small), ceres::TAKE_OWNERSHIP);
    cost->SetNumResiduals(n_res);
    cost->AddParameterBlock(2 * N);
    problem.AddResidualBlock(cost, nullptr, d.data());

    // 说明: 力矩变化率硬约束已在代价函数内部通过 clamp 实现（见 DualYawMpcCost），
    // 因此这里不使用参数箱式边界（Ceres 的边界约束会强制切换到线搜索，
    // 每次迭代的函数求值次数大幅上升）。这样可以使用 LM 信任域，求值次数少一个量级。
    (void)rate_step_b; (void)rate_step_s;

    ceres::Solver::Options options;
    options.max_num_iterations = cfg_.max_iter;
    options.function_tolerance = 1e-8;
    options.parameter_tolerance = 1e-9;
    options.gradient_tolerance = 1e-12;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.update_state_every_iteration = false;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    auto t_end = std::chrono::steady_clock::now();
    out.solve_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    out.iterations = summary.iterations.size();
    out.initial_cost = summary.initial_cost;
    out.final_cost = summary.final_cost;
    out.usable = summary.IsSolutionUsable();

    // ── 由最终 d 重建力矩序列（增量先 clamp 到力矩变化率上限，与代价函数内部一致）──
    std::vector<double> ub(N), us(N);
    const double rs_b = cfg_.big.max_torque_rate * cfg_.dt_control;
    const double rs_s = cfg_.small.max_torque_rate * cfg_.dt_control;
    ub[0] = clampT(in.prev_torque[0] + clampT(d[0], -rs_b, rs_b),
                   -cfg_.big.max_torque, cfg_.big.max_torque);
    for (int k = 1; k < N; ++k) {
        ub[k] = clampT(ub[k - 1] + clampT(d[k], -rs_b, rs_b),
                       -cfg_.big.max_torque, cfg_.big.max_torque);
    }
    us[0] = clampT(in.prev_torque[1] + clampT(d[N], -rs_s, rs_s),
                   -cfg_.small.max_torque, cfg_.small.max_torque);
    for (int k = 1; k < N; ++k) {
        us[k] = clampT(us[k - 1] + clampT(d[N + k], -rs_s, rs_s),
                       -cfg_.small.max_torque, cfg_.small.max_torque);
    }
    if (!out.usable) {
        // 求解失败: 退化为零力矩（安全），并清空热启动
        std::fill(ub.begin(), ub.end(), 0.0);
        std::fill(us.begin(), us.end(), 0.0);
        last_torque_seq_.assign(2 * N, 0.0);
    } else {
        last_torque_seq_.resize(2 * N);
        for (int k = 0; k < N; ++k) {
            last_torque_seq_[k] = ub[k];
            last_torque_seq_[N + k] = us[k];
        }
    }

    // ── 用 double 重跑一遍预测，产出上报序列 ──
    out.ref_joint[0] = ref_big;    // 世界方位角（原始参考）
    out.ref_joint[1] = ref_small;
    out.pred_joint[0].resize(N);
    out.pred_joint[1].resize(N);
    out.pred_azimuth[0].resize(N);
    out.pred_azimuth[1].resize(N);
    {
        double q[3] = {in.q[0], in.q[1], in.q[2]};
        double qd[3] = {in.qd[0], in.qd[1], in.qd[2]};
        for (int k = 0; k < N; ++k) {
            const ModelExo& e = in.exo;
            const double u3[3] = {ub[k], 0.0, us[k]};
            double qn[3], qdn[3];
            if (cfg_.use_rk4) {
                integrateStepBacklash(q, qd, u3, model_, e, cfg_.dt_control, cfg_.substeps, qn, qdn);
            } else {
                double acc[3];
                forwardAccelBacklash(q, qd, u3, model_, e, acc);
                for (int i = 0; i < 3; ++i) {
                    qdn[i] = qd[i] + cfg_.dt_control * acc[i];
                    qn[i] = q[i] + cfg_.dt_control * qdn[i];
                }
            }
            for (int i = 0; i < 3; ++i) { q[i] = qn[i]; qd[i] = qdn[i]; }

            const double t = (k + 1) * cfg_.dt_control;
            const double psi_c = in.chassis_azimuth + in.chassis_rate * t;
            const double psi_b = psi_c + q[1];       // 大 yaw: 云台侧
            out.pred_joint[0][k] = q[1];             // pred_joint 沿用 {云台, 小yaw}
            out.pred_joint[1][k] = q[2];
            out.pred_azimuth[0][k] = psi_b;
            out.pred_azimuth[1][k] = psi_b + q[2];
        }
        out.pred_q[0] = q[0];                        // {θ_motor, θ_platform, θ_small}
        out.pred_q[1] = out.pred_joint[0][0];
        out.pred_q[2] = out.pred_joint[1][0];
        // 第一步预测速度（用解析动力学再算一次）
        const ModelExo& e0 = in.exo;
        const double u0[3] = {ub[0], 0.0, us[0]};
        double acc0[3];
        forwardAccelBacklash(in.q, in.qd, u0, model_, e0, acc0);
        for (int i = 0; i < 3; ++i) out.pred_qd[i] = in.qd[i] + cfg_.dt_control * acc0[i];
    }

    out.torque[0] = ub[0];
    out.torque[1] = us[0];
    return out;
}

} // namespace dual_yaw

} // namespace tcbs
