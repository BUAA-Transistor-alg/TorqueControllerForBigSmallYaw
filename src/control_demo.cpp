// ============================================================================
// control_demo.cpp — RobotController 双级 yaw 控制演示（正弦方位角跟踪）
//
// 演示内容:
//   - 对外接口语义: 分别设置**大 yaw 世界方位角**与**小 yaw 世界方位角**
//     （两者之差即小 yaw 关节角指令；本示例让大 yaw 平台方位保持、小 yaw 做正弦）
//   - 单目标模式与序列模式（--sequence: 每拍给出未来 N 步参考轨迹，等价真实自瞄用法）
//   - 状态读取: getState() 的 est（可信量 / 大 yaw 延迟补偿估计 / 反解真实位姿 /
//     数据来源）与 mpc（控制输出、参考与预测序列、求解性能）
//
// 用法:
//   ./build/control_demo [--dur=10] [--sequence] [--amp=0.30] [--period=3.0]
//   无硬件时: 程序会等待估计就绪而超时退出（属预期行为）。
// ============================================================================
#include "tcbs/RobotController.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace tcbs {

namespace {

std::atomic<bool> g_running{true};
void onSignal(int) { g_running = false; }

constexpr double kDeg = M_PI / 180.0;

struct Options {
    double duration = 10.0;
    double period = 3.0;       // 正弦周期
    double amp = 0.12;         // 小 yaw 方位角正弦幅值 (rad)
                               // ★ 小 yaw 机械行程只有 −25° ~ +20°（非对称, 中心 −2.5°）, 且 C++ 侧
                               //   从两侧各向内 11.25° 起就加软限位代价（默认 ratio=0.75 ⇒
                               //   软限位区 [−13.75°, +8.75°]）; 所以演示幅值取 0.12 rad ≈ 6.9°,
                               //   留足余量。要更大摆幅请让大 yaw 分担（本 demo 两目标同相）。
    bool   sequence = false;   // 序列模式
    double loop_dt = 0.01;     // 主循环周期（与 mpc_loop_period 一致更自然）
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* key) -> const char* {
            const size_t n = std::strlen(key);
            return (a.rfind(key, 0) == 0 && a.size() > n) ? a.c_str() + n : nullptr;
        };
        if (const char* v = val("--dur=")) o.duration = std::atof(v);
        else if (const char* v = val("--period=")) o.period = std::atof(v);
        else if (const char* v = val("--amp=")) o.amp = std::atof(v);
        else if (const char* v = val("--loop-dt=")) o.loop_dt = std::atof(v);
        else if (a == "--sequence") o.sequence = true;
        else { printf("未知参数: %s\n", a.c_str()); std::exit(1); }
    }
    return o;
}

} // namespace

} // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；下面把 tcbs 内的类型与测试辅助函数引入作用域
using namespace tcbs;

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    printf("=== 双级 yaw control_demo ===\n");
    printf("模式: %s，小 yaw 方位角正弦 ±%.1f°，周期 %.1fs，时长 %.1fs\n",
           opt.sequence ? "序列模式（给出未来 N 步参考）" : "单目标模式",
           opt.amp / kDeg, opt.period, opt.duration);

    // ── 控制参数（默认值见 include/tcbs/mpc/planar_yaw_params.h；标定后替换）──
    RobotController::Config cfg;
    cfg.controller.loop_period = opt.loop_dt;
    cfg.controller.big_torque_only = false;     // 大 yaw: 力矩 + 电控内环
    cfg.controller.small_torque_only = false;   // 小 yaw: 力矩 + 电控内环
    cfg.controller.integral_gain[1] = 0.01;     // 小 yaw 积分补偿
    cfg.controller.integral_limit[1] = 0.20;
    cfg.sequence_mode = opt.sequence;     // 序列模式必须在构造时选定
    RobotController rc(cfg);
    const bool sequence_mode = opt.sequence;

    // ── 等待状态就绪 ──
    printf("等待状态估计就绪（IMU + MCU 编码器）...\n");
    const auto t_wait = std::chrono::steady_clock::now();
    while (g_running) {
        auto st = rc.getState();
        if (st.est.valid && st.est.prov.imu.valid) break;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait).count() > 10.0) {
            printf("超时: 未收到有效数据（无硬件时属预期）。\n");
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!g_running) return 0;
    {
        auto st = rc.getState();
        printf("就绪。初始状态: ψ_big=%.3f, θ_big=%.3f(测量 %.3f, 延迟 %.1fms), "
               "θ_small=%.3f, pitch=%.3f\n",
               st.est.platform_azimuth, st.est.big_joint_angle, st.est.big_joint_angle_meas,
               st.est.big_enc_age * 1000.0, st.est.small_joint_angle,
               st.est.pitch_joint_angle);
    }

    const double omega = 2.0 * M_PI / std::max(0.1, opt.period);
    const auto t0 = std::chrono::steady_clock::now();
    int loop = 0;
    double max_aim_err = 0.0, sum_sq_err = 0.0;
    int n_err = 0;
    double max_solve_ms = 0.0;

    while (g_running) {
        const auto loop_start = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(loop_start - t0).count();
        if (t > opt.duration) break;

        auto st = rc.getState();
        const double psi_big_now = st.est.platform_azimuth;
        const double psi_small_now = st.est.small_output_azimuth;

        // 目标: 大 yaw 平台方位保持当前值；小 yaw 方位做正弦（±amp）
        const double psi_big_target = psi_big_now;                 // 大 yaw 不动
        const auto smallTargetAt = [&](double tt) {
            return psi_big_now + opt.amp * std::sin(omega * tt);
        };
        const double psi_small_target = smallTargetAt(t);
        const double pitch_target = (5.0 + 15.0 * std::sin(omega * t - M_PI / 2.0)) * kDeg;

        if (sequence_mode) {
            // 序列模式: 给出未来 N 步参考（这里用同一解析轨迹，实际由自瞄预测给出）
            const int N = cfg.mpc.N;
            std::vector<double> big_seq(N), small_seq(N), pitch_seq(N);
            for (int k = 0; k < N; ++k) {
                const double tk = t + (k + 1) * cfg.mpc.dt_control;
                big_seq[k] = psi_big_now;
                small_seq[k] = smallTargetAt(tk);
                pitch_seq[k] = pitch_target;
            }
            std::vector<bool> fire_seq(N, false);
            rc.set(true, false, false, big_seq, small_seq, pitch_seq, fire_seq, true);
        } else {
            rc.set(/*auto_aim_enable=*/true,
                   /*big_torque_only=*/false, /*small_torque_only=*/false,
                   psi_big_target, psi_small_target,
                   pitch_target, /*fire=*/false, /*integral_enable=*/true);
        }

        const double aim_err = psi_small_target - psi_small_now;
        max_aim_err = std::max(max_aim_err, std::fabs(aim_err));
        sum_sq_err += aim_err * aim_err;
        ++n_err;
        max_solve_ms = std::max(max_solve_ms, st.mpc.solve_ms);

        if (++loop % static_cast<int>(std::max(1.0, 0.1 / opt.loop_dt)) == 0) {
            printf("[t=%5.2fs] 目标 ψs=%+.3f | 实际 ψs=%+.3f (误差 %+.4f rad = %+.2f°) "
                   "θs=%+.3f | τ=(%+.3f,%+.3f) | solve %.2fms fps %.0f "
                   "| big_enc age %.1fms innov %+.4f | est_valid=%d\n",
                   t, psi_small_target, psi_small_now, aim_err, aim_err / kDeg,
                   st.est.small_joint_angle, st.mpc.torque[0], st.mpc.torque[1],
                   st.mpc.solve_ms, st.mpc.loop_fps,
                   st.est.big_enc_age * 1000.0, st.est.big_enc_innovation,
                   (int)st.est.valid);
        }

        std::this_thread::sleep_until(loop_start + std::chrono::duration_cast<
                                          std::chrono::steady_clock::duration>(
                                          std::chrono::duration<double>(opt.loop_dt)));
    }

    // ── 收尾: 停止控制（后台线程继续发送，但目标保持最后值；退出前清零力矩）──
    printf("\n结束。统计: 最大瞄准误差 %.4f rad (%.2f°), RMS %.4f rad, "
           "最大求解耗时 %.2f ms\n",
           max_aim_err, max_aim_err / kDeg,
           std::sqrt(sum_sq_err / std::max(1, n_err)), max_solve_ms);
    return 0;
}
