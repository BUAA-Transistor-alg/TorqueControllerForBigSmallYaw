#ifndef TCBS_DUAL_YAW_MPC_H
#define TCBS_DUAL_YAW_MPC_H

#include <cstdint>
#include <vector>

#include "tcbs/mpc/planar_yaw_model.h"

namespace tcbs {

// ============================================================================
// DualYawMpc — 双级 yaw 耦合非线性 MPC
//
// 优化问题（N 步、控制周期 dt，决策量 = 两关节力矩增量序列，共 2N 个）:
//   状态: x = {θ_big, θ̇_big, θ_small, θ̇_small}（仅两个 yaw 关节进入优化；
//         pitch 角/角速度/角加速度作为外生量按步刷新）
//   动力学: 严格刚体模型 τ = M(q,pitch)·q̈ + h(q,q̇,pitch,底盘 ω_c/α_c, 重力)
//           —— 含两轴交叉惯量、非共轴偏置引起的离心/科氏项、大yaw加速对小yaw的
//              惯性反作用、任意底盘倾斜下的重力项、pitch 惯量调度与耦合
//   代价:
//     Σ_k  w_b·|ψ_big(k) − ψ_big*(k)|_smooth + w_s·|ψ_small(k) − ψ_small*(k)|_smooth
//        + w_c·(θ_small(k) − small_center_angle)²  ← 冗余自由度回中/打破多解
//                                  （中心默认 = 行程中心, 非对称行程下 ≠ 0）
//        + w_lim·ρ(θ_small(k))                 （小 yaw 软限位, **双侧** 4 次幂障碍）
//        + r_b·u_b(k)² + r_s·u_s(k)²           （力矩惩罚）
//        + rd_b·Δu_b(k)² + rd_s·Δu_s(k)²       （力矩变化率惩罚，软）
//     其中 ψ 为世界系方位角（与对外接口语义一致）:
//        ψ_big(k)   = ψ_chassis + ω_chassis·t_k + θ_big(k)
//        ψ_small(k) = ψ_big(k) + θ_small(k)
//     —— ψ_chassis 当前的旋转由底盘 IMU 提供，预测窗内按 ω_chassis 线性外推，
//        因此底盘转动造成的"参考漂移"被模型显式补偿（与旧实现的 ref 外推等价）。
//   约束:
//     |u| ≤ max_torque（硬限幅，参数化中 clamp）
//     |Δu| ≤ max_torque_rate·dt（参数箱式边界，硬约束）
//
// 求解器: Ceres + 动态自动微分（模板化动力学，Jet 传播梯度），增量重参数化。
// 性能: 每次求解耗时随 N、积分器（RK4/半隐式欧拉）与 max_iter 变化，
//       结果里的 solve_ms 与上层 loop_fps 用于实测标定；默认参数面向 ~100 Hz。
// ============================================================================
namespace dual_yaw {

// 诊断: 代价函数求值计数（用于分析求解耗时构成）
uint64_t dualYawMpcCostEvaluations();
void dualYawMpcResetCostEvaluations();

// 单关节约束
struct JointLimits {
    double max_torque = 1.0;         // N·m
    double max_torque_rate = 40.0;   // N·m/s
    // 机械行程（**硬限位**, 可以非对称; 仅用于代价中的双侧障碍项与健康检查,
    // 不做硬约束）。软限位由 small_limit_soft_ratio 从两侧各自向内推出。
    double min_angle = -1e9;
    double max_angle =  1e9;
};

struct DualYawMpcConfig {
    // ── 离散化 ──
    double dt_control = 0.01;   // 控制周期 (s)
    int    N = 12;              // 预测步数
    int    substeps = 1;        // 每步 RK4 子步
    bool   use_rk4 = true;      // true: RK4；false: 半隐式欧拉（更快，精度略低）
    int    max_iter = 15;       // Ceres 迭代上限

    // ── 权重 ──
    double w_big_azimuth = 1.0;     // 大 yaw 世界方位角跟踪
    double w_small_azimuth = 1.0;   // 小 yaw 世界方位角跟踪
    double w_small_center = 0.05;   // 小 yaw 回中（冗余自由度分配）
    // 回中目标角（rad）: 代价项 = w_c·(θ_small − small_center_angle)²。
    // ★ 行程**非对称**时 0 就不是行程中心（例如 [−25°,+20°] 的中心是 −2.5°），因此这里
    //   必须显式配置（`defaultMpcConfig()` 取 0.5·(min_angle+max_angle) = −2.5°），
    //   不能依赖"回中到 0"或内部的隐式平均；设 0 = 回中到关节零位。
    double small_center_angle = 0.0;
    double w_small_limit = 1e4;     // 小 yaw 软限位
    // 软限位区宽度: 距**任一侧**限位在 (1−ratio)·(max_angle−min_angle) 以内开始惩罚
    // ⇒ soft_min = min + (1−ratio)·(max−min), soft_max = max − (1−ratio)·(max−min)
    //    （两侧各自按自己的剩余行程取宽度, 对非对称行程同样正确）
    double small_limit_soft_ratio = 0.75;
    double r_big_torque = 0.01;
    double r_small_torque = 0.01;
    double rd_big_rate = 0.1;
    double rd_small_rate = 0.1;
    double smooth_eps = 1e-6;       // 位置误差平滑绝对值常数 a

    JointLimits big, small;

    // 参考延迟步数（"目标前瞻"语义；0 = 不延迟）
    int ref_delay_steps = 0;

};

class DualYawMpc {
public:
    struct Input {
        // 状态（关节系）: q = {θ_big, θ_small}, qd = {θ̇_big, θ̇_small}
        double q[2] = {0.0, 0.0};
        double qd[2] = {0.0, 0.0};
        ModelExo exo;                       // 含 pitch/pitch_rate/pitch_acc、底盘 ω/α、重力

        double platform_azimuth = 0.0;      // ψ_big（当前，解卷绕）
        double chassis_azimuth = 0.0;       // ψ_chassis = ψ_big − θ_big
        double chassis_rate = 0.0;          // 底盘 yaw 角速度 (rad/s)

        // 世界方位角参考序列（长度 N；不足则用最后一个值补齐；空则用当前方位角保持）
        std::vector<double> ref_big_azimuth;
        std::vector<double> ref_small_azimuth;

        double prev_torque[2] = {0.0, 0.0}; // 上一步实际施加的力矩（变化率约束起点）
    };

    struct Output {
        double torque[2] = {0.0, 0.0};          // 第一步力矩指令 (N·m)
        double pred_q[2] = {0.0, 0.0};          // 第一步预测关节角
        double pred_qd[2] = {0.0, 0.0};         // 第一步预测关节角速度
        std::vector<double> ref_joint[2];       // 参考换算到关节系（显示/诊断）
        std::vector<double> pred_joint[2];      // 预测关节角序列
        std::vector<double> pred_azimuth[2];    // 预测世界方位角序列 {big, small}
        bool   usable = false;
        int    iterations = 0;
        double solve_ms = 0.0;
        double initial_cost = 0.0;
        double final_cost = 0.0;
        bool   small_ref_over_limit = false;    // 参考（换算到关节系后）落在软限位区之外
                                                // （两侧**独立**判定: 高于 soft_max 或 低于 soft_min）
        int    residual_size = 0;
    };

    DualYawMpc(const ModelParams& model, const DualYawMpcConfig& cfg);

    Output solve(const Input& in);

    // 复位热启动状态
    void reset();

    const DualYawMpcConfig& config() const { return cfg_; }
    void setConfig(const DualYawMpcConfig& c) { cfg_ = c; }
    const ModelParams& model() const { return model_; }
    void setModel(const ModelParams& m) { model_ = m; reset(); }

    // 残差个数（供测试/调试）
    int residualSize() const;
    // 决策变量个数 = 2N
    int parameterSize() const { return 2 * cfg_.N; }

private:
    void checkIntegrability() const;
    // 限位/回中配置的健康检查（非对称行程下 min/max/center/ratio 写错会静默退化）
    void checkLimitConfig() const;

    ModelParams model_;
    DualYawMpcConfig cfg_;
    std::vector<double> last_torque_seq_;   // 2N（热启动）
};

} // namespace dual_yaw

} // namespace tcbs

#endif // TCBS_DUAL_YAW_MPC_H