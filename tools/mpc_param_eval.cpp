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
// ============================================================================
#include "tcbs/mpc/dual_yaw_mpc.h"
#include "tcbs/mpc/planar_yaw_params.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace tcbs {

using namespace dual_yaw;

namespace {

struct Opt {
    double phi[8] = {0.024, 0.013, 0.0, 0.0, 0.090, 0.030, 0.030, 0.008};
    // ★ plant(被控对象)的参数: 默认 = 本次仿真的"真值"参数集（只有 λ 不同）
    double plant_phi[8] = {0.050, 0.020, 0.0087, 0.005, 0.220, 0.055, 0.0973, 0.028};
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
};

struct Plant {
    ModelParams p;
    double q[2] = {0.0, 0.0}, qd[2] = {0.0, 0.0};
    ModelExo exo;
};

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
        DualYawMpc::Input in;
        in.q[0] = plant.q[0]; in.q[1] = plant.q[1];
        in.qd[0] = plant.qd[0]; in.qd[1] = plant.qd[1];
        in.exo = plant.exo;
        in.platform_azimuth = plant.q[0];
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
                integral[i] += o.integral_gain * (prev_pred[i] - plant.q[i]);
                integral[i] = std::clamp(integral[i], -0.3, 0.3);
            }
        }
        prev_pred[0] = res.pred_q[0]; prev_pred[1] = res.pred_q[1];
        have_prev = true;
        tau[0] = std::clamp(tau[0] + integral[0], -cfg.big.max_torque, cfg.big.max_torque);
        tau[1] = std::clamp(tau[1] + integral[1], -cfg.small.max_torque, cfg.small.max_torque);
        m.max_tau[0] = std::max(m.max_tau[0], std::fabs(tau[0]));
        m.max_tau[1] = std::max(m.max_tau[1], std::fabs(tau[1]));

        for (int s = 0; s < static_cast<int>(sub); ++s) {
            double qn[2], qdn[2];
            integrateStep(plant.q, plant.qd, tau, plant.p, plant.exo, o.plant_dt, 1, qn, qdn);
            plant.q[0] = qn[0]; plant.q[1] = qn[1];
            plant.qd[0] = qdn[0]; plant.qd[1] = qdn[1];
        }
        tau_applied[0] = tau[0]; tau_applied[1] = tau[1];

        if (!std::isfinite(plant.q[0]) || !std::isfinite(plant.q[1]) ||
            std::fabs(plant.q[0]) > 1e3 || std::fabs(plant.q[1]) > 1e3) {
            m.diverged = true;
            break;
        }
        const double err = small_ref[static_cast<size_t>(k)] - (plant.q[0] + plant.q[1]);
        m.max_aim_err = std::max(m.max_aim_err, std::fabs(err));
        m.min_small = std::min(m.min_small, plant.q[1]);
        m.max_small = std::max(m.max_small, plant.q[1]);
        sum_sq += err * err; ++n_err;
    }
    m.rms_aim_err = std::sqrt(sum_sq / std::max(1, n_err));
    m.mean_solve_ms = solve_sum / std::max(1, n_steps);
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
        if (a.rfind("--plant-phi=", 0) == 0) {
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

    ModelParams mp = defaultModelParams();
    mp.Jbig_eff = o.phi[0]; mp.Js = o.phi[1]; mp.Px = o.phi[2]; mp.Py = o.phi[3];
    mp.fcBig = o.phi[4]; mp.fvBig = o.phi[5]; mp.fcSmall = o.phi[6]; mp.fvSmall = o.phi[7];
    mp.frictionLambda = o.lambda;

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
    std::printf("N=%d max_iter=%d\n", o.N, o.max_iter);
    std::printf("模型 λ=%.0f  子步=%d (有效步长 %.2f ms)  plant λ=%.0f (步长 %.1f µs)  T=%.1fs\n",
                o.lambda, o.substeps, 10.0 / std::max(1, o.substeps),
                o.plant_lambda, o.plant_dt * 1e6, o.T);
    std::printf("%-30s %10s %10s %12s %12s %10s %10s %6s\n",
                "场景", "最大误差", "RMS", "θs 最小(°)", "θs 最大(°)", "求解均ms", "求解最大", "失败");
    for (auto& sc : scs) {
        Metrics m = runClosedLoop(mp, o, sc.b, sc.s);
        std::printf("%-30s %10.4f %10.4f %12.2f %12.2f %10.2f %10.2f %6d%s\n",
                    sc.name, m.max_aim_err, m.rms_aim_err,
                    m.min_small * 180.0 / M_PI, m.max_small * 180.0 / M_PI,
                    m.mean_solve_ms, m.max_solve_ms, m.solve_fail,
                    m.diverged ? "  ★发散" : "");
    }
    return 0;
}
