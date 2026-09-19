// ============================================================================
// mpc_param_eval.cpp — 用**辨识出来的参数**评估 MPC 闭环表现（λ 对比工具）
//
// 用途: 同一条闭环场景下比较不同 (模型 λ, MPC 积分子步, 参数集) 的表现，
//       回答"MPC 模型 λ 取 100 还是 1000 更合适、代价多大"。
//
// 被控对象(plant): λ=1e4（≈sign，模拟真库仑摩擦）+ 20µs 固定步长积分
// 控制器(MPC):     参数来自命令行（辨识结果），λ 与子步也可命令行给
//
// 用法:
//   ./mpc_param_eval --phi=J,Js,Px,Py,fcb,fvb,fcs,fvs --lambda=100 --substeps=4
//   [--plant-lambda=1e4] [--plant-dt=2e-5] [--T=3.0] [--integral=0.02] [--label=xxx]
//   ★ 大 yaw 背隙那 8 个参数（δ,k,c,γ,Jmotor,fcMotor,fvMotor,β）:
//     --phi2=δ,k,c,γ,Jmotor,fcMotor,fvMotor,β       （**控制器模型**用）
//     --plant-phi2=...                              （被控对象用；不给 = defaultModelParams()）
//     不给就都用 defaultModelParams() 的值（⇒ 参数准确时模型==被控对象）。
// ============================================================================
#include "tcbs/mpc/dual_yaw_mpc.h"
#include "tcbs/mpc/planar_yaw_params.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <random>
#include <string>
#include <vector>

namespace tcbs {

using namespace dual_yaw;

namespace {

struct Opt {
    double phi[8] = {0.024, 0.013, 0.0, 0.0, 0.090, 0.030, 0.030, 0.008};
    // ★ plant(被控对象)的参数: 默认 = 本次仿真的"真值"参数集（只有 λ 不同）
    double plant_phi[8] = {0.050, 0.020, 0.0087, 0.005, 0.220, 0.055, 0.0973, 0.028};
    // ★ 背隙/电机侧那 8 个（δ,k,c,γ,Jmotor,fcMotor,fvMotor,β）；NaN = 用 defaultModelParams()
    double phi2[8] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    double plant_phi2[8] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    double lambda = 100.0;
    double plant_lambda = 1.0e4;
    double plant_dt = 2.0e-5;
    int    substeps = 4;
    int    N = 12;                   // 预测窗步数
    int    max_iter = 8;             // Ceres 迭代上限
    bool   euler = false;            // true = 半隐式欧拉（更快）
    double T = 3.0;
    double integral_gain = 0.0;      // >0 时开启积分补偿（与 McuMpcController 同形式）
    std::string label = "";
    // ── ★ 被控对象: 背隙"接触面完全刚性 + 死区完全自由"（= 采集测试数据那个环境）──
    //   k = 1e5 N·m/rad（1 N·m 只压出 1e-5 rad 变形 ⇒ 实际完全刚性）、c = 0、γ = 0；
    //   死区内 τ_t ≡ 0（dz 在死区内为 0，且没有 γ 的直通项）。
    bool   plant_rigid = false;
    double plant_rigid_k = 1.0e5;
    // ── ★ β（死区中心）: 真值随机 + 微弱漂移；控制器侧用**在线估计**（滑动 min/max）──
    double plant_beta0 = 0.0;
    double plant_beta_drift = 0.0;    // 幅值 (rad)
    double plant_beta_period = 30.0;  // 周期 (s)
    double plant_beta_random = 0.0;   // >0 ⇒ β0 ~ U(−x, +x)（用下面的 rng seed 抽）
    unsigned plant_seed = 12345u;
    double beta_tau = 3.0;            // 在线估计的遗忘时间常数（= 估计器默认）
    int    ctrl_beta_mode = 0;        // 0=用在线估计 1=不用(β=0) 2=用真值(上限对照)
    std::string dump = "";            // 非空 ⇒ 把闭环轨迹写成 CSV（前缀，每场景一个 _0/_1/_2）
    std::string dump_file = "";       // 当前场景的实际文件名（runClosedLoop 内部用）
};

struct Plant {
    // ★ 3-DOF 被控对象（带真实大 yaw 背隙）: q = {θ_motor, θ_platform, θ_small}
    ModelParams p;
    double q[3] = {0.0, 0.0, 0.0}, qd[3] = {0.0, 0.0, 0.0};
    ModelExo exo;
};

// ★ 把长度 8 的背隙/电机侧参数写进 ModelParams（NaN ⇒ 保持默认值不动）
void applyExtra(ModelParams& p, const double v[8]) {
    if (!std::isnan(v[0])) p.backlash_delta = v[0];
    if (!std::isnan(v[1])) p.backlash_k = v[1];
    if (!std::isnan(v[2])) p.backlash_c = v[2];
    if (!std::isnan(v[3])) p.backlash_through = v[3];
    if (!std::isnan(v[4])) p.Jmotor = v[4];
    if (!std::isnan(v[5])) p.fcMotor = v[5];
    if (!std::isnan(v[6])) p.fvMotor = v[6];
    if (!std::isnan(v[7])) { /* β 由 exo.backlash_beta 给，不在 ModelParams 里 */ }
}

struct Metrics {
    double max_aim_err = 0.0, rms_aim_err = 0.0;
    double min_small = 0.0, max_small = 0.0;
    double max_tau[2] = {0.0, 0.0};
    double mean_solve_ms = 0.0, max_solve_ms = 0.0;
    int solve_fail = 0;
    bool diverged = false;
};

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

std::vector<double> sineRef(int n, double dt, double amp, double freq, double phase) {
    std::vector<double> out(n, 0.0);
    for (int k = 0; k < n; ++k)
        out[k] = amp * std::sin(2.0 * M_PI * freq * k * dt + phase);
    return out;
}

Metrics runClosedLoop(const ModelParams& ctrl_model, const Opt& o,
                      const std::vector<double>& big_ref, const std::vector<double>& small_ref) {
    DualYawMpcConfig cfg = defaultMpcConfig();
    cfg.substeps = o.substeps;
    cfg.N = o.N;
    cfg.max_iter = o.max_iter;
    cfg.use_rk4 = !o.euler;
    cfg.dt_control = 0.01;

    Plant plant;
    plant.p = defaultModelParams();
    plant.p.Jbig_eff = o.plant_phi[0]; plant.p.Js = o.plant_phi[1];
    plant.p.Px = o.plant_phi[2]; plant.p.Py = o.plant_phi[3];
    plant.p.fcBig = o.plant_phi[4]; plant.p.fvBig = o.plant_phi[5];
    plant.p.fcSmall = o.plant_phi[6]; plant.p.fvSmall = o.plant_phi[7];
    plant.p.frictionLambda = o.plant_lambda;
    applyExtra(plant.p, o.plant_phi2);          // ★ 被控对象的背隙/电机侧参数
    // ── ★ 刚性接触被控对象（接触面完全刚性 + 死区完全自由）──
    const double plant_beta0 = o.plant_beta0;      // 已在 main 里解析（含随机）
    if (o.plant_rigid) {
        plant.p.backlash_k = o.plant_rigid_k;   // 1 N·m 只压出 1e-5 rad ⇒ 实际完全刚性
        plant.p.backlash_c = 0.0;
        plant.p.backlash_through = 0.0;         // 死区内**完全自由**（τ_t ≡ 0）
    }
    // β 的在线估计（与 YawStateEstimator 同一套带遗忘滑动 min/max）
    double bl_min = 0.0, bl_max = 0.0, bl_t = 0.0;
    bool bl_seen = false;
    // 轨迹导出
    std::vector<std::array<double, 10>> traj;

    DualYawMpc mpc(ctrl_model, cfg);
    Metrics m;
    const double dt = cfg.dt_control;
    const double sub = std::round(dt / o.plant_dt);
    const int n_steps = static_cast<int>(std::round(o.T / dt));

    double tau_applied[2] = {0.0, 0.0};
    double integral[2] = {0.0, 0.0}, prev_pred[2] = {0.0, 0.0};
    bool have_prev = false;
    double sum_sq = 0.0, solve_sum = 0.0;
    int n_err = 0;

    for (int k = 0; k < n_steps; ++k) {
        const double t_now = k * dt;
        // ★ 被控对象的 β(t): 初始随机 + 微弱漂移
        const double beta_true = (o.plant_beta_drift > 0.0)
            ? plant_beta0 + o.plant_beta_drift * std::sin(2.0 * M_PI * t_now / o.plant_beta_period)
            : plant_beta0;
        plant.exo.backlash_beta = beta_true;
        // ★ β 的在线估计（控制器侧看到的就是它）: Δ_raw = θ_motor − θ_platform 的滑动极值中心
        const double draw = plant.q[0] - plant.q[1];
        if (!bl_seen) { bl_min = bl_max = draw; bl_seen = true; bl_t = t_now; }
        else {
            const double d = t_now - bl_t;
            if (d > 1e-9) {
                const double a = 1.0 - std::exp(-d / std::max(1e-3, o.beta_tau));
                bl_min = (draw < bl_min) ? draw : (bl_min + a * (draw - bl_min));
                bl_max = (draw > bl_max) ? draw : (bl_max + a * (draw - bl_max));
                bl_t = t_now;
            }
        }
        const double beta_hat = (o.ctrl_beta_mode == 1) ? 0.0
                              : (o.ctrl_beta_mode == 2) ? beta_true : 0.5 * (bl_max + bl_min);

        DualYawMpc::Input in;
        in.q[0] = plant.q[0]; in.q[1] = plant.q[1]; in.q[2] = plant.q[2];
        in.qd[0] = plant.qd[0]; in.qd[1] = plant.qd[1]; in.qd[2] = plant.qd[2];
        in.exo = plant.exo;
        in.exo.backlash_beta = beta_hat;        // ★ 控制器用在线估计的 β
        in.platform_azimuth = plant.q[1];
        in.chassis_azimuth = 0.0; in.chassis_rate = 0.0;
        in.prev_torque[0] = tau_applied[0];
        in.prev_torque[1] = tau_applied[1];
        in.ref_big_azimuth.resize(cfg.N);
        in.ref_small_azimuth.resize(cfg.N);
        for (int j = 0; j < cfg.N; ++j) {
            const size_t idx = static_cast<size_t>(std::min<int>(
                static_cast<int>(big_ref.size()) - 1, k + j + 1));
            in.ref_big_azimuth[j] = big_ref[idx];
            in.ref_small_azimuth[j] = small_ref[idx];
        }
        auto res = mpc.solve(in);
        solve_sum += res.solve_ms;
        m.max_solve_ms = std::max(m.max_solve_ms, res.solve_ms);
        if (!res.usable) ++m.solve_fail;

        double tau[2] = {res.torque[0], res.torque[1]};
        if (o.integral_gain > 0.0 && have_prev) {
            for (int i = 0; i < 2; ++i) {
                const double q_axis = (i == 0) ? plant.q[1] : plant.q[2];
                integral[i] += o.integral_gain * (prev_pred[i] - q_axis);
                integral[i] = std::clamp(integral[i], -0.3, 0.3);
            }
        }
        prev_pred[0] = res.pred_q[1]; prev_pred[1] = res.pred_q[2];
        have_prev = true;
        tau[0] = std::clamp(tau[0] + integral[0], -cfg.big.max_torque, cfg.big.max_torque);
        tau[1] = std::clamp(tau[1] + integral[1], -cfg.small.max_torque, cfg.small.max_torque);
        m.max_tau[0] = std::max(m.max_tau[0], std::fabs(tau[0]));
        m.max_tau[1] = std::max(m.max_tau[1], std::fabs(tau[1]));

        for (int s = 0; s < static_cast<int>(sub); ++s) {
            const double u3[3] = {tau[0], 0.0, tau[1]};
            double qn[3], qdn[3];
            integrateStepBacklash(plant.q, plant.qd, u3, plant.p, plant.exo, o.plant_dt, 1, qn, qdn);
            for (int i = 0; i < 3; ++i) { plant.q[i] = qn[i]; plant.qd[i] = qdn[i]; }
        }
        tau_applied[0] = tau[0]; tau_applied[1] = tau[1];

        if (!std::isfinite(plant.q[1]) || !std::isfinite(plant.q[2]) ||
            std::fabs(plant.q[1]) > 1e3 || std::fabs(plant.q[2]) > 1e3) {
            m.diverged = true;
            break;
        }
        const double err = small_ref[static_cast<size_t>(k)] - (plant.q[1] + plant.q[2]);
        m.max_aim_err = std::max(m.max_aim_err, std::fabs(err));
        if (!o.dump_file.empty())
            traj.push_back({t_now, big_ref[static_cast<size_t>(k)],
                            small_ref[static_cast<size_t>(k)],
                            plant.q[1], plant.q[1] + plant.q[2], plant.q[2],
                            tau[0], tau[1], beta_true, beta_hat});
        m.min_small = std::min(m.min_small, plant.q[2]);
        m.max_small = std::max(m.max_small, plant.q[2]);
        sum_sq += err * err; ++n_err;
    }
    m.rms_aim_err = std::sqrt(sum_sq / std::max(1, n_err));
    m.mean_solve_ms = solve_sum / std::max(1, n_steps);
    if (!o.dump.empty()) {
        FILE* fp = std::fopen(o.dump_file.c_str(), "w");
        if (fp) {
            std::fprintf(fp, "t,ref_big,ref_aim,platform,aim,small,tau_big,tau_small,"
                             "beta_true,beta_hat\n");
            for (const auto& r : traj)
                std::fprintf(fp, "%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                             r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]);
            std::fclose(fp);
            std::printf("[dump] 闭环轨迹已写入 %s（%zu 行）\n",
                        o.dump_file.c_str(), traj.size());
        }
    }
    return m;
}

} // namespace

} // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；下面把 tcbs 内的类型与测试辅助函数引入作用域
using namespace tcbs;

int main(int argc, char** argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* key, double& dst) {
            const std::string k = std::string(key) + "=";
            if (a.rfind(k, 0) == 0) { dst = std::atof(a.c_str() + k.size()); return true; }
            return false;
        };
        if (a.rfind("--plant-phi2=", 0) == 0) {
            std::string s2 = a.substr(13);
            for (auto& c : s2) if (c == ',') c = ' ';
            std::sscanf(s2.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                        &o.plant_phi2[0], &o.plant_phi2[1], &o.plant_phi2[2], &o.plant_phi2[3],
                        &o.plant_phi2[4], &o.plant_phi2[5], &o.plant_phi2[6], &o.plant_phi2[7]);
        }
        else if (a.rfind("--phi2=", 0) == 0) {
            std::string s = a.substr(7);
            for (auto& c : s) if (c == ',') c = ' ';
            std::sscanf(s.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                        &o.phi2[0], &o.phi2[1], &o.phi2[2], &o.phi2[3],
                        &o.phi2[4], &o.phi2[5], &o.phi2[6], &o.phi2[7]);
        }
        else if (a.rfind("--plant-phi=", 0) == 0) {
            std::string s2 = a.substr(12);
            for (auto& c : s2) if (c == ',') c = ' ';
            std::sscanf(s2.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                        &o.plant_phi[0], &o.plant_phi[1], &o.plant_phi[2], &o.plant_phi[3],
                        &o.plant_phi[4], &o.plant_phi[5], &o.plant_phi[6], &o.plant_phi[7]);
        }
        else if (a.rfind("--phi=", 0) == 0) {
            std::string s = a.substr(6);
            for (auto& c : s) if (c == ',') c = ' ';
            std::sscanf(s.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                        &o.phi[0], &o.phi[1], &o.phi[2], &o.phi[3],
                        &o.phi[4], &o.phi[5], &o.phi[6], &o.phi[7]);
        }
        else if (a.rfind("--label=", 0) == 0) o.label = a.substr(8);
        else if (a == "--plant-rigid") o.plant_rigid = true;
        else if (val("--plant-rigid-k", o.plant_rigid_k)) {}
        else if (val("--plant-beta0", o.plant_beta0)) {}
        else if (val("--plant-beta-drift", o.plant_beta_drift)) {}
        else if (val("--plant-beta-period", o.plant_beta_period)) {}
        else if (val("--plant-beta-random", o.plant_beta_random)) {}
        else if (a.rfind("--plant-seed=", 0) == 0)
            o.plant_seed = static_cast<unsigned>(std::atoi(a.c_str() + 13));
        else if (val("--beta-tau", o.beta_tau)) {}
        else if (a.rfind("--ctrl-beta=", 0) == 0) {
            const std::string v = a.substr(12);
            o.ctrl_beta_mode = (v == "off" || v == "none") ? 1 : (v == "true" ? 2 : 0);
        }
        else if (a.rfind("--dump=", 0) == 0) o.dump = a.substr(7);
        else if (val("--lambda", o.lambda)) {}
        else if (val("--plant-lambda", o.plant_lambda)) {}
        else if (val("--plant-dt", o.plant_dt)) {}
        else if (val("--T", o.T)) {}
        else if (val("--integral", o.integral_gain)) {
        }
        else if (a.rfind("--substeps=", 0) == 0) o.substeps = std::atoi(a.c_str() + 11);
        else if (a.rfind("--N=", 0) == 0) o.N = std::atoi(a.c_str() + 4);
        else if (a == "--euler") o.euler = true;
        else if (a.rfind("--max-iter=", 0) == 0) o.max_iter = std::atoi(a.c_str() + 11);
        else { std::printf("未知选项: %s\n", a.c_str()); return 2; }
    }

    // ★ β0 的随机抽样在 main 里做一次: 三个场景共用同一个 β0（可比），打印也是真值
    if (o.plant_beta_random > 0.0) {
        std::mt19937 rng(o.plant_seed);
        std::uniform_real_distribution<double> ur(-o.plant_beta_random, o.plant_beta_random);
        o.plant_beta0 = ur(rng);
    }

    ModelParams mp = defaultModelParams();
    // 被控对象与控制器都用默认参数（含 5° 背隙）⇒ 这是"参数准确"的闭环评估
    mp.Jbig_eff = o.phi[0]; mp.Js = o.phi[1]; mp.Px = o.phi[2]; mp.Py = o.phi[3];
    mp.fcBig = o.phi[4]; mp.fvBig = o.phi[5]; mp.fcSmall = o.phi[6]; mp.fvSmall = o.phi[7];
    mp.frictionLambda = o.lambda;
    applyExtra(mp, o.phi2);                     // ★ 控制器模型的背隙/电机侧参数

    const int n = static_cast<int>(std::round(o.T / 0.01));
    const double dt = 0.01;

    struct Sc { const char* name; std::vector<double> b, s; };
    std::vector<Sc> scs;
    scs.push_back({"阶跃 0.6rad（双轴同参考）", smoothStepRef(n, dt, 0.0, 0.6, 8.0),
                   smoothStepRef(n, dt, 0.0, 0.6, 8.0)});
    scs.push_back({"大阶跃 1.2rad（需大 yaw 展开）", smoothStepRef(n, dt, 0.0, 1.2, 8.0),
                   smoothStepRef(n, dt, 0.0, 1.2, 8.0)});
    scs.push_back({"正弦跟踪 0.3rad @0.5Hz", sineRef(n, dt, 0.3, 0.5, 0.0),
                   sineRef(n, dt, 0.3, 0.5, 0.0)});

    std::printf("=== MPC 参数评估 %s ===\n", o.label.c_str());
    std::printf("φ = [%.5f %.5f %.5f %.5f %.4f %.4f %.4f %.4f]\n",
                mp.Jbig_eff, mp.Js, mp.Px, mp.Py, mp.fcBig, mp.fvBig, mp.fcSmall, mp.fvSmall);
    std::printf("背隙/电机侧: δ=%.4f rad (%.2f°)  k=%.2f  c=%.3f  γ=%.5f  "
                "Jmotor=%.5f  fcMotor=%.4f  fvMotor=%.4f\n",
                mp.backlash_delta, mp.backlash_delta * 180.0 / M_PI, mp.backlash_k,
                mp.backlash_c, mp.backlash_through, mp.Jmotor, mp.fcMotor, mp.fvMotor);
    if (o.plant_rigid)
        std::printf("★ 被控对象 = **刚性接触 + 死区完全自由**: k=%.3g N·m/rad（1N·m ⇒ %.1e rad "
                    "变形）、c=0、γ=0\n", o.plant_rigid_k, 1.0 / o.plant_rigid_k);
    std::printf("★ β: 真值 β0=%+.4f rad%s%s；控制器侧 β = %s（τ=%.1fs 滑动极值）\n",
                o.plant_beta0,
                o.plant_beta_drift > 0.0 ? " + 漂移" : "",
                o.plant_beta_random > 0.0 ? "（**随机重抽**）" : "",
                o.ctrl_beta_mode == 1 ? "**不用（恒 0）**"
                                      : (o.ctrl_beta_mode == 2 ? "真值（上限对照）" : "在线估计"),
                o.beta_tau);
    std::printf("N=%d max_iter=%d\n", o.N, o.max_iter);
    std::printf("模型 λ=%.0f  子步=%d (有效步长 %.2f ms)  plant λ=%.0f (步长 %.1f µs)  T=%.1fs\n",
                o.lambda, o.substeps, 10.0 / std::max(1, o.substeps),
                o.plant_lambda, o.plant_dt * 1e6, o.T);
    std::printf("%-30s %10s %10s %12s %12s %10s %10s %6s\n",
                "场景", "最大误差", "RMS", "θs 最小(°)", "θs 最大(°)", "求解均ms", "求解最大", "失败");
    int sc_idx = 0;
    for (auto& sc : scs) {
        if (!o.dump.empty()) {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s_%d.csv", o.dump.c_str(), sc_idx);
            o.dump_file = buf;
        }
        ++sc_idx;
        Metrics m = runClosedLoop(mp, o, sc.b, sc.s);
        std::printf("%-30s %10.4f %10.4f %12.2f %12.2f %10.2f %10.2f %6d%s\n",
                    sc.name, m.max_aim_err, m.rms_aim_err,
                    m.min_small * 180.0 / M_PI, m.max_small * 180.0 / M_PI,
                    m.mean_solve_ms, m.max_solve_ms, m.solve_fail,
                    m.diverged ? "  ★发散" : "");
    }
    return 0;
}
