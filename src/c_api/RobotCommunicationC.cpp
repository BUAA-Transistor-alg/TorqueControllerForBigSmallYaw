// ============================================================================
// RobotCommunicationC.cpp — 双级 yaw 云台 C API 实现
//
// 设计要点:
//   - 两个不透明句柄: TcbsRobotCommHandle（包 RobotCommunication）
//                     TcbsRobotController_C（包 RobotController + 累积配置）
//   - 所有对 C++ 的调用都包在 try/catch 里，异常 → 负错误码；消息打到 stderr
//   - 结构体填充集中在一组 fill*/toCpp 辅助函数里，字段一一对应
// ============================================================================

#include "tcbs/c_api/RobotCommunicationC.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "tcbs/RobotController.h"
#include "tcbs/communication/Communications.hpp"
#include "tcbs/communication/McuDataPreprocessor.h"
#include "tcbs/communication/Protocol.hpp"
#include "tcbs/communication/YawStateEstimator.h"
#include "tcbs/mpc/planar_yaw_model.h"
#include "tcbs/mpc/dual_yaw_mpc.h"
#include "tcbs/mpc/planar_yaw_params.h"
#include "tcbs/mpc/mcu_mpc_controller.h"

// ============================================================================
// 句柄定义
//
// 注意: 这两个不透明句柄是 **C ABI 类型**（在 c_api/RobotCommunicationC.h 里以
// `typedef struct ... ...` 声明），按统一封装约定它们属于 C 侧，因此留在
// **全局命名空间**并带 Tcbs 前缀；用到的 C++ 侧类型显式写 tcbs:: 限定。
// 其余实现（含下面那些 extern "C" 函数定义）全部包在 namespace tcbs 内 ——
// 带 C 语言链接的函数在命名空间里定义，与全局声明是**同一个实体**，导出符号
// 仍是裸 C 名 tcbs_*（见验收里的 nm 检查）。
// ============================================================================

struct TcbsRobotCommHandle {
    std::mutex mtx;
    std::unique_ptr<tcbs::RobotCommunication> comm;   // 构造即启动两条串口线程
};

struct TcbsRobotController_C {
    std::mutex mtx;
    // 累积配置: 每个 setter 同步更新，tcbs_robot_controller_set_controller_config 用它重建对象
    tcbs::RobotController::Config cfg;
    std::unique_ptr<tcbs::RobotController> rc;
};


namespace tcbs {

namespace {

// ── 小工具 ──
inline uint8_t b2u(bool b) { return b ? 1u : 0u; }
inline bool u2b(uint8_t v) { return v != 0u; }

inline void reportException(const char* fn, const std::exception& e) {
    std::fprintf(stderr, "[robot_c_api] %s: std::exception: %s\n", fn, e.what());
}

// 是否属于"模式不匹配"异常（RobotController 的 mode 保护）
inline bool isModeMismatch(const std::runtime_error& e) {
    return std::strstr(e.what(), "mode") != nullptr;
}

// 序列 → 调用方缓冲区；返回序列实际长度（out 为空或 max_len<=0 时仅查询长度）
int copySeqTo(const std::vector<double>& v, double* out, int32_t max_len) {
    const int32_t n = static_cast<int32_t>(v.size());
    if (out == nullptr || max_len <= 0) return n;
    const int32_t m = (n < max_len) ? n : max_len;
    for (int32_t i = 0; i < m; ++i) out[i] = v[i];
    return n;
}

// ============================================================================
// 填充: C++ → C
// ============================================================================

// 低层: RobotCommunication::LatestData（原始包）→ TcbsRobotMcuData_C / TcbsRobotImuData_C
void fillMcuFromRaw(const RobotCommunication::LatestData& raw, TcbsRobotMcuData_C& d) {
    d = TcbsRobotMcuData_C{};
    if (!raw.mcu_valid) return;
    const mcu::ReceivePacket& m = raw.mcu_packet;
    d.valid = 1;
    d.bullet_velocity = m.bullet_velocity;
    d.pitch_angle = m.pitch_angle;
    d.yaw_big_angle = m.yaw_big_angle;
    d.yaw_big_omega = m.yaw_big_omega;
    d.yaw_small_angle = m.yaw_small_angle;
    d.yaw_small_omega = m.yaw_small_omega;
    d.chassis_imu_yaw = m.chassis_imu_yaw;
    d.chassis_imu_omega = m.chassis_imu_omega;
    d.mark = m.mark;
    d.color = m.color;
    d.auto_aim_switch = m.auto_aim_switch;
    d.yaw_big_temperature = m.yaw_big_temperature;
    d.yaw_small_temperature = m.yaw_small_temperature;
    // MCU 端不提供时钟 → 只给"MCU2 新样本序号"（大 yaw 与底盘 IMU 同源共用）
    d.mcu2_seq = raw.mcu2_seq;                  // LatestData 级别（= mcu_packet.mcu2_seq）
}

void fillImuFromRaw(const RobotCommunication::LatestData& raw, TcbsRobotImuData_C& d) {
    d = TcbsRobotImuData_C{};
    if (!raw.imu_valid) return;
    const imu::ReceivePacket& im = raw.imu_packet;
    d.valid = 1;
    d.gx = im.gx;
    d.gy = im.gy;
    d.gz = im.gz;
    d.ax = im.ax;
    d.ay = im.ay;
    d.az = im.az;
    d.euler_yaw = im.euler_yaw;
    d.euler_pitch = im.euler_pitch;
    d.euler_roll = im.euler_roll;
    d.dt_one_tenth_ms = im.dt_one_tenth_ms;
}

// 高层: RobotController::McuData / ImuData → C
void fillMcu(const RobotController::McuData& s, TcbsRobotMcuData_C& d) {
    d = TcbsRobotMcuData_C{};
    d.valid = b2u(s.valid);
    d.bullet_velocity = s.bullet_velocity;
    d.pitch_angle = s.pitch_angle;
    d.yaw_big_angle = s.yaw_big_angle;
    d.yaw_big_omega = s.yaw_big_omega;
    d.yaw_small_angle = s.yaw_small_angle;
    d.yaw_small_omega = s.yaw_small_omega;
    d.chassis_imu_yaw = s.chassis_imu_yaw;
    d.chassis_imu_omega = s.chassis_imu_omega;
    d.mark = s.mark;
    d.color = s.color;
    d.auto_aim_switch = s.auto_aim_switch;
    d.yaw_big_temperature = s.yaw_big_temperature;
    d.yaw_small_temperature = s.yaw_small_temperature;
    d.mcu2_seq = s.mcu2_seq;                    // MCU2 新样本序号（大 yaw 与底盘 IMU 共用）
}

void fillImu(const RobotController::ImuData& s, TcbsRobotImuData_C& d) {
    d = TcbsRobotImuData_C{};
    d.valid = b2u(s.valid);
    d.gx = s.gx;
    d.gy = s.gy;
    d.gz = s.gz;
    d.ax = s.ax;
    d.ay = s.ay;
    d.az = s.az;
    d.euler_yaw = s.euler_yaw;
    d.euler_pitch = s.euler_pitch;
    d.euler_roll = s.euler_roll;
    d.dt_one_tenth_ms = s.dt_one_tenth_ms;
}

void fillSourceInfo(const YawStateEstimator::SourceInfo& s, TcbsRobotSourceInfo_C& d) {
    d.valid = b2u(s.valid);
    d.age_s = s.age_s;                  // 值年龄（上位机计时）
    d.count = s.count;                  // 收到帧数（含值保持帧）
    d.new_samples = s.new_samples;      // 真正的新样本数（序号变化）
    d.rejected = s.rejected;            // 值被保持（非新数据）的帧数
    d.stale = b2u(s.stale);             // 年龄超过 stale_age_s
}

void fillProvenance(const YawStateEstimator::Provenance& p, TcbsRobotProvenance_C& d) {
    d = TcbsRobotProvenance_C{};
    fillSourceInfo(p.imu, d.imu);
    fillSourceInfo(p.big_enc, d.big_enc);
    fillSourceInfo(p.small_enc, d.small_enc);
    fillSourceInfo(p.pitch_enc, d.pitch_enc);
    fillSourceInfo(p.chassis_imu, d.chassis_imu);
    d.big_rate_from_imu = b2u(p.big_rate_from_imu);
    d.big_rate_from_encoder = b2u(p.big_rate_from_encoder);
    d.reverse_from_trusted = b2u(p.reverse_from_trusted);
    d.big_enc_delay_used = p.big_enc_delay_used;      // 本帧大 yaw 值的实测年龄（上位机计时）
    d.big_enc_innovation = p.big_enc_innovation;
    d.big_enc_interval_s = p.big_enc_interval_s;      // 最近两次新样本间隔
    d.big_enc_sample_age_s = p.big_enc_sample_age_s;  // 最近一次新样本的实测年龄
    d.used_mask = p.used_mask;
}

// 低层路径给出 YawStateEstimator::Estimate，高层路径给出 RobotController::EstData；
// 两者字段一一对应（prov 同为 YawStateEstimator::Provenance），故用模板统一处理。
// 注: 若将来其中一个类型少了某字段，这里会编译报错（宁可编译期暴露，也不要静默填 0）。
template <typename EstT>
void fillEstimate(const EstT& e, TcbsRobotEstimate_C& d) {
    d = TcbsRobotEstimate_C{};
    d.valid = b2u(e.valid);
    // 1) 可信实时量
    d.imu_yaw = e.imu_yaw;
    d.imu_pitch = e.imu_pitch;
    d.imu_roll = e.imu_roll;
    d.platform_azimuth = e.platform_azimuth;
    d.platform_rate = e.platform_rate;
    d.small_joint_angle = e.small_joint_angle;
    d.small_joint_rate = e.small_joint_rate;
    d.pitch_joint_angle = e.pitch_joint_angle;
    d.pitch_joint_rate = e.pitch_joint_rate;
    // 2) 大 yaw
    d.big_joint_angle_meas = e.big_joint_angle_meas;
    d.big_joint_angle = e.big_joint_angle;
    d.big_joint_rate = e.big_joint_rate;
    d.big_motor_angle = e.big_motor_angle;
    d.big_motor_rate = e.big_motor_rate;
    d.big_platform_angle = e.big_platform_angle;
    d.big_platform_rate = e.big_platform_rate;
    d.backlash_center = e.backlash_center;
    d.backlash_width_obs = e.backlash_width_obs;
    d.big_enc_age = e.big_enc_age;              // 值年龄（上位机计时，-1 = 从未收到）
    d.big_sample_interval = e.big_sample_interval;  // 最近两次新样本间隔（上位机计时）
    d.chassis_imu_age = e.chassis_imu_age;      // 底盘 IMU 值的年龄（上位机计时）
    d.big_enc_innovation = e.big_enc_innovation;
    d.big_has_encoder = b2u(e.big_has_encoder);
    // 3) 反解真实位姿
    d.head_world_yaw = e.head_world_yaw;
    d.head_world_pitch = e.head_world_pitch;
    d.head_world_roll = e.head_world_roll;
    d.small_output_azimuth = e.small_output_azimuth;
    d.los_azimuth = e.los_azimuth;
    d.los_elevation = e.los_elevation;
    d.chassis_azimuth = e.chassis_azimuth;
    d.chassis_yaw_rate = e.chassis_yaw_rate;
    // 4) 模型外生量
    for (int i = 0; i < 3; ++i) {
        d.base_omega[i] = e.base_omega[i];
        d.gravity_a[i] = e.gravity_a[i];   // 重力矢量（★ A 系 = 大 yaw 转子系，指向下）
    }
    d.pitch_acc = e.pitch_acc;
    // 数据来源
    fillProvenance(e.prov, d.prov);
}

void fillMpc(const RobotController::MpcData& m, TcbsRobotMpcData_C& d) {
    d = TcbsRobotMpcData_C{};
    for (int i = 0; i < 2; ++i) {
        d.torque[i] = m.torque[i];
        d.torque_mpc[i] = m.torque_mpc[i];
        d.integral[i] = m.integral[i];
        d.target_joint[i] = m.target_joint[i];
        d.target_joint_rate[i] = m.target_joint_rate[i];
        d.ref_azimuth[i] = m.ref_azimuth[i];
        d.delayed_ref_azimuth[i] = m.delayed_ref_azimuth[i];
        d.ref_azimuth_seq_len[i] = static_cast<int32_t>(m.ref_azimuth_seq[i].size());
        d.pred_azimuth_seq_len[i] = static_cast<int32_t>(m.pred_azimuth_seq[i].size());
        d.pred_joint_seq_len[i] = static_cast<int32_t>(m.pred_joint_seq[i].size());
    }
    d.big_torque_only = b2u(m.big_torque_only);
    d.small_torque_only = b2u(m.small_torque_only);
    d.small_ref_over_limit = b2u(m.small_ref_over_limit);
    d.solve_ms = m.solve_ms;
    d.loop_fps = m.loop_fps;
    d.ticks_since_set = m.ticks_since_set;
    d.solve_count = m.solve_count;
    d.solve_fail_count = m.solve_fail_count;
    d.estimator_valid = b2u(m.estimator_valid);
    d.sent_ok = b2u(m.sent_ok);
}

// ============================================================================
// 转换: C → C++（字段一一对应）
// ============================================================================

dual_yaw::JointLimits toCpp(const TcbsDualYawJointLimits_C& c) {
    dual_yaw::JointLimits d;
    d.max_torque = c.max_torque;
    d.max_torque_rate = c.max_torque_rate;
    d.min_angle = c.min_angle;
    d.max_angle = c.max_angle;
    return d;
}

void toC(const dual_yaw::JointLimits& s, TcbsDualYawJointLimits_C& c) {
    c = TcbsDualYawJointLimits_C{};
    c.max_torque = s.max_torque;
    c.max_torque_rate = s.max_torque_rate;
    c.min_angle = s.min_angle;
    c.max_angle = s.max_angle;
}

dual_yaw::ModelParams toCpp(const TcbsDualYawModelParams_C& c) {
    dual_yaw::ModelParams p;
    // 实测几何
    p.dx = c.dx; p.dy = c.dy;
    p.gravity = c.gravity;
    p.m_u_known = c.m_u_known;
    // ★8 参待辨识（顺序与 paramsToVector / regressor 一致）
    p.Jbig_eff = c.Jbig_eff;
    p.Js = c.Js;
    p.Px = c.Px;
    p.Py = c.Py;
    p.fcBig = c.fcBig; p.fvBig = c.fvBig;
    p.fcSmall = c.fcSmall; p.fvSmall = c.fvSmall;
    // 固定 / 可选
    p.frictionLambda = c.frictionLambda;
    p.tau_offset_big = c.tau_offset_big;
    p.tau_offset_small = c.tau_offset_small;
    return p;
}

void toC(const dual_yaw::ModelParams& p, TcbsDualYawModelParams_C& c) {
    c = TcbsDualYawModelParams_C{};
    c.dx = p.dx; c.dy = p.dy;
    c.gravity = p.gravity;
    c.m_u_known = p.m_u_known;
    c.Jbig_eff = p.Jbig_eff;
    c.Js = p.Js;
    c.Px = p.Px;
    c.Py = p.Py;
    c.fcBig = p.fcBig; c.fvBig = p.fvBig;
    c.fcSmall = p.fcSmall; c.fvSmall = p.fvSmall;
    c.frictionLambda = p.frictionLambda;
    c.tau_offset_big = p.tau_offset_big;
    c.tau_offset_small = p.tau_offset_small;
}

dual_yaw::DualYawMpcConfig toCpp(const TcbsDualYawMpcConfig_C& c) {
    dual_yaw::DualYawMpcConfig d;
    d.dt_control = c.dt_control;
    d.N = c.N;
    d.substeps = c.substeps;
    d.use_rk4 = u2b(c.use_rk4);
    d.max_iter = c.max_iter;
    d.w_big_azimuth = c.w_big_azimuth;
    d.w_small_azimuth = c.w_small_azimuth;
    d.w_small_center = c.w_small_center;
    d.w_small_limit = c.w_small_limit;
    d.small_limit_soft_ratio = c.small_limit_soft_ratio;
    d.r_big_torque = c.r_big_torque;
    d.r_small_torque = c.r_small_torque;
    d.rd_big_rate = c.rd_big_rate;
    d.rd_small_rate = c.rd_small_rate;
    d.smooth_eps = c.smooth_eps;
    d.big = toCpp(c.big);
    d.small = toCpp(c.small);
    // 小 yaw 回中目标角（C++ 侧 DualYawMpcConfig::small_center_angle）:
    // **C ABI 结构体布局已冻结**（TcbsDualYawMpcConfig_C 不新增字段、不升版本号），
    // 因此这里按行程中心派生 —— 与本仓库 defaultMpcConfig() 的默认值
    // （0.5·(min_angle + max_angle)，对称行程 ±30° ⇒ 0）语义一致。
    // C++ 侧若要单独指定中心，请直接用 DualYawMpc::setConfig()。
    d.small_center_angle = 0.5 * (c.small.min_angle + c.small.max_angle);
    d.ref_delay_steps = c.ref_delay_steps;
    // 注: 平面 8 参模型的 DualYawMpcConfig 没有 extrapolate_pitch（pitch 不进动力学）
    return d;
}

void toC(const dual_yaw::DualYawMpcConfig& s, TcbsDualYawMpcConfig_C& c) {
    c = TcbsDualYawMpcConfig_C{};
    c.dt_control = s.dt_control;
    c.N = s.N;
    c.substeps = s.substeps;
    c.use_rk4 = b2u(s.use_rk4);
    c.max_iter = s.max_iter;
    c.w_big_azimuth = s.w_big_azimuth;
    c.w_small_azimuth = s.w_small_azimuth;
    c.w_small_center = s.w_small_center;
    c.w_small_limit = s.w_small_limit;
    c.small_limit_soft_ratio = s.small_limit_soft_ratio;
    c.r_big_torque = s.r_big_torque;
    c.r_small_torque = s.r_small_torque;
    c.rd_big_rate = s.rd_big_rate;
    c.rd_small_rate = s.rd_small_rate;
    c.smooth_eps = s.smooth_eps;
    toC(s.big, c.big);
    toC(s.small, c.small);
    c.ref_delay_steps = s.ref_delay_steps;
}

// IMU 安装位置（C int32 ↔ C++ enum class）; 非 0 一律视为 ON_HEAD
YawStateEstimator::Config::ImuLocation imuLocationFromC(int32_t v) {
    return (v == 1) ? YawStateEstimator::Config::ImuLocation::ON_HEAD
                    : YawStateEstimator::Config::ImuLocation::ON_BIG_YAW;
}

int32_t imuLocationToC(YawStateEstimator::Config::ImuLocation loc) {
    return (loc == YawStateEstimator::Config::ImuLocation::ON_HEAD) ? 1 : 0;
}

YawStateEstimator::Config toCpp(const TcbsEstimatorConfig_C& c) {
    YawStateEstimator::Config d;
    d.imu_location = imuLocationFromC(c.imu_location);
    d.mount_yaw = c.mount_yaw;
    d.mount_pitch = c.mount_pitch;
    d.mount_roll = c.mount_roll;
    d.head_mount_yaw = c.head_mount_yaw;
    d.head_mount_pitch = c.head_mount_pitch;
    d.head_mount_roll = c.head_mount_roll;
    d.transport_delay_s = c.transport_delay_s;
    d.big_enc_max_jump = c.big_enc_max_jump;
    d.stale_age_s = c.stale_age_s;
    d.chassis_imu_timeout_s = c.chassis_imu_timeout_s;
    d.max_extrap_s = c.max_extrap_s;
    d.small_rate_lpf_alpha = c.small_rate_lpf_alpha;
    d.big_rate_lpf_alpha = c.big_rate_lpf_alpha;
    d.big_motor_rate_tau_s = c.big_motor_rate_tau_s;
    d.big_motor_rate_alpha = c.big_motor_rate_alpha;
    d.backlash_center_tau_s = c.backlash_center_tau_s;
    d.pitch_rate_lpf_alpha = c.pitch_rate_lpf_alpha;
    d.pitch_acc_lpf_alpha = c.pitch_acc_lpf_alpha;
    for (int i = 0; i < 3; ++i) d.bore[i] = c.bore[i];
    d.gravity = c.gravity;
    d.use_chassis_imu = u2b(c.use_chassis_imu);
    d.source_timeout_s = c.source_timeout_s;
    return d;
}

void toC(const YawStateEstimator::Config& s, TcbsEstimatorConfig_C& c) {
    c = TcbsEstimatorConfig_C{};
    c.imu_location = imuLocationToC(s.imu_location);
    c.mount_yaw = s.mount_yaw;
    c.mount_pitch = s.mount_pitch;
    c.mount_roll = s.mount_roll;
    c.head_mount_yaw = s.head_mount_yaw;
    c.head_mount_pitch = s.head_mount_pitch;
    c.head_mount_roll = s.head_mount_roll;
    c.transport_delay_s = s.transport_delay_s;
    c.big_enc_max_jump = s.big_enc_max_jump;
    c.stale_age_s = s.stale_age_s;
    c.chassis_imu_timeout_s = s.chassis_imu_timeout_s;
    c.max_extrap_s = s.max_extrap_s;
    c.small_rate_lpf_alpha = s.small_rate_lpf_alpha;
    c.big_rate_lpf_alpha = s.big_rate_lpf_alpha;
    c.big_motor_rate_tau_s = s.big_motor_rate_tau_s;
    c.big_motor_rate_alpha = s.big_motor_rate_alpha;
    c.backlash_center_tau_s = s.backlash_center_tau_s;
    c.pitch_rate_lpf_alpha = s.pitch_rate_lpf_alpha;
    c.pitch_acc_lpf_alpha = s.pitch_acc_lpf_alpha;
    for (int i = 0; i < 3; ++i) c.bore[i] = s.bore[i];
    c.gravity = s.gravity;
    c.use_chassis_imu = b2u(s.use_chassis_imu);
    c.source_timeout_s = s.source_timeout_s;
}

McuDataPreprocessor::LinearParams toCpp(const TcbsLinearParams_C& c) {
    McuDataPreprocessor::LinearParams p;
    p.send_pitch_scale = c.send_pitch_scale;
    p.send_pitch_offset = c.send_pitch_offset;
    p.recv_pitch_scale = c.recv_pitch_scale;
    p.recv_pitch_offset = c.recv_pitch_offset;
    p.recv_big_yaw_scale = c.recv_big_yaw_scale;
    p.recv_big_yaw_offset = c.recv_big_yaw_offset;
    p.recv_big_omega_scale = c.recv_big_omega_scale;
    p.send_big_yaw_scale = c.send_big_yaw_scale;
    p.send_big_yaw_offset = c.send_big_yaw_offset;
    p.send_big_velocity_scale = c.send_big_velocity_scale;
    p.send_big_torque_scale = c.send_big_torque_scale;
    p.recv_small_yaw_scale = c.recv_small_yaw_scale;
    p.recv_small_yaw_offset = c.recv_small_yaw_offset;
    p.recv_small_omega_scale = c.recv_small_omega_scale;
    p.send_small_yaw_scale = c.send_small_yaw_scale;
    p.send_small_yaw_offset = c.send_small_yaw_offset;
    p.send_small_velocity_scale = c.send_small_velocity_scale;
    p.send_small_torque_scale = c.send_small_torque_scale;
    return p;
}

void toC(const McuDataPreprocessor::LinearParams& s, TcbsLinearParams_C& c) {
    c = TcbsLinearParams_C{};
    c.send_pitch_scale = s.send_pitch_scale;
    c.send_pitch_offset = s.send_pitch_offset;
    c.recv_pitch_scale = s.recv_pitch_scale;
    c.recv_pitch_offset = s.recv_pitch_offset;
    c.recv_big_yaw_scale = s.recv_big_yaw_scale;
    c.recv_big_yaw_offset = s.recv_big_yaw_offset;
    c.recv_big_omega_scale = s.recv_big_omega_scale;
    c.send_big_yaw_scale = s.send_big_yaw_scale;
    c.send_big_yaw_offset = s.send_big_yaw_offset;
    c.send_big_velocity_scale = s.send_big_velocity_scale;
    c.send_big_torque_scale = s.send_big_torque_scale;
    c.recv_small_yaw_scale = s.recv_small_yaw_scale;
    c.recv_small_yaw_offset = s.recv_small_yaw_offset;
    c.recv_small_omega_scale = s.recv_small_omega_scale;
    c.send_small_yaw_scale = s.send_small_yaw_scale;
    c.send_small_yaw_offset = s.send_small_yaw_offset;
    c.send_small_velocity_scale = s.send_small_velocity_scale;
    c.send_small_torque_scale = s.send_small_torque_scale;
}

McuMpcController::Config toCpp(const TcbsControllerConfig_C& c) {
    McuMpcController::Config d;
    d.loop_period = c.loop_period;
    d.big_torque_only = u2b(c.big_torque_only);
    d.small_torque_only = u2b(c.small_torque_only);
    d.ref_delay_steps = c.ref_delay_steps;
    d.integral_gain[0] = c.integral_gain[0];
    d.integral_gain[1] = c.integral_gain[1];
    d.integral_limit[0] = c.integral_limit[0];
    d.integral_limit[1] = c.integral_limit[1];
    d.integral_on_big = u2b(c.integral_on_big);
    return d;
}

void toC(const McuMpcController::Config& s, TcbsControllerConfig_C& c) {
    c = TcbsControllerConfig_C{};
    c.loop_period = s.loop_period;
    c.big_torque_only = b2u(s.big_torque_only);
    c.small_torque_only = b2u(s.small_torque_only);
    c.ref_delay_steps = s.ref_delay_steps;
    c.integral_gain[0] = s.integral_gain[0];
    c.integral_gain[1] = s.integral_gain[1];
    c.integral_limit[0] = s.integral_limit[0];
    c.integral_limit[1] = s.integral_limit[1];
    c.integral_on_big = b2u(s.integral_on_big);
}

// 把序列参数（指针 + 长度）转成 STL 容器
void toCppSeq(const double* src, int32_t len, std::vector<double>& dst) {
    dst.clear();
    if (src == nullptr || len <= 0) return;
    dst.assign(src, src + static_cast<size_t>(len));
}

void toCppSeq(const uint8_t* src, int32_t len, std::vector<bool>& dst) {
    dst.clear();
    if (src == nullptr || len <= 0) return;
    dst.reserve(static_cast<size_t>(len));
    for (int32_t i = 0; i < len; ++i) dst.push_back(src[i] != 0u);
}

// 用累积配置重建 RobotController（controller 配置无运行时 setter 时的唯一途径）
int rebuildLocked(TcbsRobotController_C* h) {
    h->rc.reset();   // 停后台线程 + 关串口
    try {
        h->rc = std::make_unique<RobotController>(h->cfg);
    } catch (const std::exception& e) {
        reportException("RobotController rebuild", e);
        return TCBS_ROBOT_COMM_ERR_ALLOC;
    } catch (...) {
        std::fprintf(stderr, "[robot_c_api] RobotController rebuild: unknown exception\n");
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
    return TCBS_ROBOT_COMM_OK;
}

} // namespace

// ============================================================================
// 错误码描述
// ============================================================================
extern "C" const char* tcbs_robot_comm_strerror(int status) {
    switch (status) {
        case TCBS_ROBOT_COMM_OK:                return "ok";
        case TCBS_ROBOT_COMM_ERR_NULL_HANDLE:   return "null handle";
        case TCBS_ROBOT_COMM_ERR_INVALID_ARG:   return "invalid argument";
        case TCBS_ROBOT_COMM_ERR_MODE_MISMATCH: return "controller mode mismatch (SINGLE vs SEQUENCE)";
        case TCBS_ROBOT_COMM_ERR_RUNTIME:       return "c++ runtime error";
        case TCBS_ROBOT_COMM_ERR_UNKNOWN:       return "unknown error";
        case TCBS_ROBOT_COMM_ERR_ALLOC:         return "allocation/construction failed";
        case TCBS_ROBOT_COMM_ERR_ABI_MISMATCH:  return "c_api abi mismatch (api version or struct layout)";
        default:                           return "unknown status";
    }
}

// ============================================================================
// 布局自检
// ============================================================================
extern "C" int tcbs_robot_comm_abi_info(TcbsRobotCommAbiInfo_C* out) {
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    *out = TcbsRobotCommAbiInfo_C{};
    out->api_version = TCBS_ROBOT_COMM_C_API_VERSION;
    out->sizeof_pointer = static_cast<uint32_t>(sizeof(void*));
    out->sizeof_latest_data = static_cast<uint32_t>(sizeof(TcbsRobotLatestData_C));
    out->sizeof_estimate = static_cast<uint32_t>(sizeof(TcbsRobotEstimate_C));
    out->sizeof_source_info = static_cast<uint32_t>(sizeof(TcbsRobotSourceInfo_C));
    out->sizeof_provenance = static_cast<uint32_t>(sizeof(TcbsRobotProvenance_C));
    out->sizeof_controller_state = static_cast<uint32_t>(sizeof(TcbsRobotControllerState_C));
    out->sizeof_mpc_data = static_cast<uint32_t>(sizeof(TcbsRobotMpcData_C));
    out->sizeof_model_params = static_cast<uint32_t>(sizeof(TcbsDualYawModelParams_C));
    out->sizeof_mpc_config = static_cast<uint32_t>(sizeof(TcbsDualYawMpcConfig_C));
    out->sizeof_estimator_config = static_cast<uint32_t>(sizeof(TcbsEstimatorConfig_C));
    out->sizeof_linear_params = static_cast<uint32_t>(sizeof(TcbsLinearParams_C));
    out->sizeof_controller_config = static_cast<uint32_t>(sizeof(TcbsControllerConfig_C));
    return TCBS_ROBOT_COMM_OK;
}

// ABI 自检: 版本号 + 逐个 sizeof 比对（调用方只需填 sizeof_* 字段）
extern "C" int tcbs_robot_comm_check_abi(uint32_t expected_version,
                                    const TcbsRobotCommAbiInfo_C* expected_sizes) {
    if (expected_version != TCBS_ROBOT_COMM_C_API_VERSION) {
        std::fprintf(stderr,
                     "[robot_c_api] ABI 版本不匹配: 调用方=%u, 库=%u（请重新构建绑定/库）\n",
                     static_cast<unsigned>(expected_version),
                     static_cast<unsigned>(TCBS_ROBOT_COMM_C_API_VERSION));
        return TCBS_ROBOT_COMM_ERR_ABI_MISMATCH;
    }
    if (expected_sizes == nullptr) return TCBS_ROBOT_COMM_OK;

    TcbsRobotCommAbiInfo_C actual{};
    tcbs_robot_comm_abi_info(&actual);
    // {调用方字段, 本库字段, 结构体名}
    struct SizeCheck {
        uint32_t caller;
        uint32_t lib;
        const char* name;
    };
    const SizeCheck checks[] = {
        {expected_sizes->sizeof_latest_data,     actual.sizeof_latest_data,     "TcbsRobotLatestData_C"},
        {expected_sizes->sizeof_estimate,        actual.sizeof_estimate,        "TcbsRobotEstimate_C"},
        {expected_sizes->sizeof_source_info,     actual.sizeof_source_info,     "TcbsRobotSourceInfo_C"},
        {expected_sizes->sizeof_provenance,      actual.sizeof_provenance,      "TcbsRobotProvenance_C"},
        {expected_sizes->sizeof_controller_state, actual.sizeof_controller_state, "TcbsRobotControllerState_C"},
        {expected_sizes->sizeof_mpc_data,        actual.sizeof_mpc_data,        "TcbsRobotMpcData_C"},
        {expected_sizes->sizeof_model_params,    actual.sizeof_model_params,    "TcbsDualYawModelParams_C"},
        {expected_sizes->sizeof_mpc_config,      actual.sizeof_mpc_config,      "TcbsDualYawMpcConfig_C"},
        {expected_sizes->sizeof_estimator_config, actual.sizeof_estimator_config, "TcbsEstimatorConfig_C"},
        {expected_sizes->sizeof_linear_params,   actual.sizeof_linear_params,   "TcbsLinearParams_C"},
        {expected_sizes->sizeof_controller_config, actual.sizeof_controller_config, "TcbsControllerConfig_C"},
    };
    int mismatches = 0;
    for (const SizeCheck& c : checks) {
        if (c.caller != c.lib) {
            ++mismatches;
            std::fprintf(stderr, "[robot_c_api] 结构体布局不匹配 %s: 调用方=%u, 库=%u\n",
                         c.name, static_cast<unsigned>(c.caller), static_cast<unsigned>(c.lib));
        }
    }
    return (mismatches == 0) ? TCBS_ROBOT_COMM_OK : TCBS_ROBOT_COMM_ERR_ABI_MISMATCH;
}

// ============================================================================
// 五、低层接口
// ============================================================================
extern "C" TcbsRobotCommHandle* tcbs_robot_comm_create(void) {
    try {
        std::unique_ptr<TcbsRobotCommHandle> h(new TcbsRobotCommHandle());
        h->comm = std::make_unique<RobotCommunication>();   // 默认映射 + 默认估计器配置
        return h.release();
    } catch (const std::exception& e) {
        reportException("tcbs_robot_comm_create", e);
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[robot_c_api] tcbs_robot_comm_create: unknown exception\n");
        return nullptr;
    }
}

extern "C" void tcbs_robot_comm_destroy(TcbsRobotCommHandle* handle) {
    if (handle == nullptr) return;
    try {
        handle->comm.reset();
    } catch (...) {
    }
    delete handle;
}

extern "C" void tcbs_robot_comm_stop(TcbsRobotCommHandle* handle) {
    if (handle == nullptr) return;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (handle->comm) handle->comm->stop();
    } catch (const std::exception& e) {
        reportException("tcbs_robot_comm_stop", e);
    } catch (...) {
    }
}

extern "C" int tcbs_robot_comm_get_latest_data(TcbsRobotCommHandle* handle, TcbsRobotLatestData_C* out) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->comm) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const RobotCommunication::LatestData raw = handle->comm->getLatestData();
        *out = TcbsRobotLatestData_C{};
        fillMcuFromRaw(raw, out->mcu);
        fillImuFromRaw(raw, out->imu);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_comm_get_latest_data", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_comm_get_estimate(TcbsRobotCommHandle* handle, TcbsRobotEstimate_C* out) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->comm) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        fillEstimate(handle->comm->getEstimate(), *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_comm_get_estimate", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_comm_send_to_mcu(TcbsRobotCommHandle* handle,
                                      uint8_t auto_aim_enable, uint8_t fire,
                                      float pitch_target_angle,
                                      uint8_t yaw_big_mode,
                                      double yaw_big_target_angle,
                                      float yaw_big_target_velocity,
                                      float yaw_big_torque,
                                      uint8_t yaw_small_mode,
                                      float yaw_small_target_angle,
                                      float yaw_small_target_velocity,
                                      float yaw_small_torque) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->comm) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        mcu::SendPacket p;   // 默认成员初始化器已填好前导/版本/data_size
        p.auto_aim_enable = auto_aim_enable;
        p.fire = fire;
        p.pitch_target_angle = pitch_target_angle;
        p.yaw_big_mode = yaw_big_mode;
        p.yaw_big_target_angle = yaw_big_target_angle;
        p.yaw_big_target_velocity = yaw_big_target_velocity;
        p.yaw_big_torque = yaw_big_torque;
        p.yaw_small_mode = yaw_small_mode;
        p.yaw_small_target_angle = yaw_small_target_angle;
        p.yaw_small_target_velocity = yaw_small_target_velocity;
        p.yaw_small_torque = yaw_small_torque;
        // 返回 1 = 成功，0 = 串口未打开/写失败（与 C++ bool 语义一致）
        return handle->comm->sendToMcu(p) ? 1 : 0;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_comm_send_to_mcu", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_comm_send_to_imu(TcbsRobotCommHandle* handle) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->comm) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        imu::SendPacket p;   // 心跳帧，无载荷
        return handle->comm->sendToImu(p) ? 1 : 0;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_comm_send_to_imu", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

// ============================================================================
// 六、高层接口
// ============================================================================
extern "C" TcbsRobotController_C* tcbs_robot_controller_create(void) {
    try {
        std::unique_ptr<TcbsRobotController_C> h(new TcbsRobotController_C());
        h->cfg = RobotController::Config{};                 // 全默认参数
        h->rc = std::make_unique<RobotController>(h->cfg);  // 启动后台线程（无硬件也可）
        return h.release();
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_create", e);
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[robot_c_api] tcbs_robot_controller_create: unknown exception\n");
        return nullptr;
    }
}

extern "C" void tcbs_robot_controller_destroy(TcbsRobotController_C* handle) {
    if (handle == nullptr) return;
    try {
        {
            std::lock_guard<std::mutex> lock(handle->mtx);
            handle->rc.reset();   // 停后台线程 + 关串口
        }
    } catch (...) {
    }
    delete handle;
}

extern "C" int tcbs_robot_controller_mode(TcbsRobotController_C* handle) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        return (handle->rc->mode() == RobotController::Mode::SEQUENCE) ? 1 : 0;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_mode", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

// ── 默认配置 ──
extern "C" int tcbs_robot_controller_default_model_params(TcbsDualYawModelParams_C* out) {
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        toC(dual_yaw::defaultModelParams(), *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_default_mpc_config(TcbsDualYawMpcConfig_C* out) {
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        toC(dual_yaw::defaultMpcConfig(), *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_default_estimator_config(TcbsEstimatorConfig_C* out) {
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        toC(YawStateEstimator::Config{}, *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_default_linear_params(TcbsLinearParams_C* out) {
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        toC(McuDataPreprocessor::LinearParams{}, *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_default_controller_config(TcbsControllerConfig_C* out) {
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        toC(McuMpcController::Config{}, *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

// ── 配置读取（读句柄内累积配置: 默认值 + 所有 setter 的结果）──
// 模型参数与估计器配置直接读 **C++ 对象**（getter 自身线程安全 / 只在 setter 与
// 后台线程之间用句柄互斥保护），因此读到的必然是与控制回路一致的值；
// MPC 配置读累积配置镜像（DualYawMpc::config() 没有互斥，不在 loop 运行期间跨线程读）。
extern "C" int tcbs_robot_controller_get_model_params(TcbsRobotController_C* handle,
                                                 TcbsDualYawModelParams_C* out) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (handle->rc) {
            toC(handle->rc->modelParams(), *out);   // 控制回路正在用的模型参数
        } else {
            toC(handle->cfg.model, *out);
        }
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_model_params", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_get_mpc_config(TcbsRobotController_C* handle,
                                               TcbsDualYawMpcConfig_C* out) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        toC(handle->cfg.mpc, *out);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_mpc_config", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_get_estimator_config(TcbsRobotController_C* handle,
                                                     TcbsEstimatorConfig_C* out) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (handle->rc) {
            toC(handle->rc->estimator().config(), *out);   // 估计器自身加锁，可跨线程读
        } else {
            toC(handle->cfg.estimator, *out);
        }
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_estimator_config", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

// ── 配置设置 ──
extern "C" int tcbs_robot_controller_set_model_params(TcbsRobotController_C* handle,
                                                const TcbsDualYawModelParams_C* params) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (params == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const dual_yaw::ModelParams m = toCpp(*params);
        handle->cfg.model = m;                 // 镜像（供 controller 配置重建时使用）
        handle->rc->setModelParams(m);         // 运行时生效（含 MPC 热启动复位）
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_model_params", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_mpc_config(TcbsRobotController_C* handle,
                                              const TcbsDualYawMpcConfig_C* config) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (config == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const dual_yaw::DualYawMpcConfig c = toCpp(*config);
        handle->cfg.mpc = c;
        handle->rc->setMpcConfig(c);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_mpc_config", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_estimator_config(TcbsRobotController_C* handle,
                                                    const TcbsEstimatorConfig_C* config) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (config == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const YawStateEstimator::Config c = toCpp(*config);
        handle->cfg.estimator = c;
        handle->rc->setEstimatorConfig(c);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_estimator_config", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_linear_params(TcbsRobotController_C* handle,
                                                 const TcbsLinearParams_C* params) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (params == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const McuDataPreprocessor::LinearParams p = toCpp(*params);
        handle->cfg.mcu_linear = p;
        handle->rc->setLinearParams(p);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_linear_params", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_controller_config(TcbsRobotController_C* handle,
                                                     const TcbsControllerConfig_C* config) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (config == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        handle->cfg.controller = toCpp(*config);
        // McuMpcController::Config 只能在构造时注入 → 用累积配置重建
        return rebuildLocked(handle);
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_controller_config", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

// ── 目标设置 ──
extern "C" int tcbs_robot_controller_set(TcbsRobotController_C* handle,
                                    uint8_t auto_aim_enable,
                                    uint8_t big_torque_only,
                                    uint8_t small_torque_only,
                                    double big_yaw_azimuth,
                                    double small_yaw_azimuth,
                                    double pitch_target_angle,
                                    uint8_t fire,
                                    uint8_t integral_enable) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        handle->rc->set(u2b(auto_aim_enable), u2b(big_torque_only), u2b(small_torque_only),
                        big_yaw_azimuth, small_yaw_azimuth, pitch_target_angle,
                        u2b(fire), u2b(integral_enable));
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::runtime_error& e) {
        reportException("tcbs_robot_controller_set", e);
        return isModeMismatch(e) ? TCBS_ROBOT_COMM_ERR_MODE_MISMATCH : TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_joint_angles(TcbsRobotController_C* handle,
                                                 uint8_t auto_aim_enable,
                                                 uint8_t big_torque_only,
                                                 uint8_t small_torque_only,
                                                 double big_joint_angle,
                                                 double small_joint_angle,
                                                 double pitch_target_angle,
                                                 uint8_t fire,
                                                 uint8_t integral_enable) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        handle->rc->setJointAngles(u2b(auto_aim_enable), u2b(big_torque_only),
                                   u2b(small_torque_only),
                                   big_joint_angle, small_joint_angle,
                                   pitch_target_angle, u2b(fire), u2b(integral_enable));
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::runtime_error& e) {
        reportException("tcbs_robot_controller_set_joint_angles", e);
        return isModeMismatch(e) ? TCBS_ROBOT_COMM_ERR_MODE_MISMATCH : TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_joint_angles", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_sequence(TcbsRobotController_C* handle,
                                             uint8_t auto_aim_enable,
                                             uint8_t big_torque_only,
                                             uint8_t small_torque_only,
                                             const double* big_yaw_azimuth_seq, int32_t big_len,
                                             const double* small_yaw_azimuth_seq, int32_t small_len,
                                             const double* pitch_seq, int32_t pitch_len,
                                             const uint8_t* fire_seq, int32_t fire_len,
                                             uint8_t integral_enable) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        std::vector<double> big, small, pitch;
        std::vector<bool> fire;
        toCppSeq(big_yaw_azimuth_seq, big_len, big);
        toCppSeq(small_yaw_azimuth_seq, small_len, small);
        toCppSeq(pitch_seq, pitch_len, pitch);
        toCppSeq(fire_seq, fire_len, fire);
        // 序列 set 在 SINGLE 模式下会抛 std::runtime_error → 转成错误码
        handle->rc->set(u2b(auto_aim_enable), u2b(big_torque_only), u2b(small_torque_only),
                        big, small, pitch, fire, u2b(integral_enable));
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::runtime_error& e) {
        reportException("tcbs_robot_controller_set_sequence", e);
        return isModeMismatch(e) ? TCBS_ROBOT_COMM_ERR_MODE_MISMATCH : TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_sequence", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_set_sequence_unchecked(TcbsRobotController_C* handle,
                                                       uint8_t auto_aim_enable,
                                                       uint8_t big_torque_only,
                                                       uint8_t small_torque_only,
                                                       const double* big_yaw_azimuth_seq, int32_t big_len,
                                                       const double* small_yaw_azimuth_seq, int32_t small_len,
                                                       const double* pitch_seq, int32_t pitch_len,
                                                       const uint8_t* fire_seq, int32_t fire_len,
                                                       uint8_t integral_enable) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        std::vector<double> big, small, pitch;
        std::vector<bool> fire;
        toCppSeq(big_yaw_azimuth_seq, big_len, big);
        toCppSeq(small_yaw_azimuth_seq, small_len, small);
        toCppSeq(pitch_seq, pitch_len, pitch);
        toCppSeq(fire_seq, fire_len, fire);
        // 直接走 McuMpcController（无模式标志），绕过 RobotController 的 SINGLE/SEQUENCE 保护
        handle->rc->mcuMpc().set(u2b(auto_aim_enable), u2b(big_torque_only),
                                 u2b(small_torque_only), big, small, pitch, fire,
                                 u2b(integral_enable));
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_set_sequence_unchecked", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

// ── 状态获取 ──
extern "C" int tcbs_robot_controller_get_state(TcbsRobotController_C* handle, TcbsRobotControllerState_C* out) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (out == nullptr) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const RobotController::State st = handle->rc->getState();
        *out = TcbsRobotControllerState_C{};
        fillMcu(st.mcu, out->mcu);
        fillImu(st.imu, out->imu);
        fillEstimate(st.est, out->est);
        fillMpc(st.mpc, out->mpc);
        return TCBS_ROBOT_COMM_OK;
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_state", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_get_ref_sequence(TcbsRobotController_C* handle, int32_t which,
                                                 double* out, int32_t max_len) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (which != 0 && which != 1) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const RobotController::State st = handle->rc->getState();
        return copySeqTo(st.mpc.ref_azimuth_seq[which], out, max_len);
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_ref_sequence", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_get_pred_sequence(TcbsRobotController_C* handle, int32_t which,
                                                  double* out, int32_t max_len) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (which != 0 && which != 1) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const RobotController::State st = handle->rc->getState();
        return copySeqTo(st.mpc.pred_azimuth_seq[which], out, max_len);
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_pred_sequence", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

extern "C" int tcbs_robot_controller_get_pred_joint_sequence(TcbsRobotController_C* handle, int32_t which,
                                                        double* out, int32_t max_len) {
    if (handle == nullptr) return TCBS_ROBOT_COMM_ERR_NULL_HANDLE;
    if (which != 0 && which != 1) return TCBS_ROBOT_COMM_ERR_INVALID_ARG;
    try {
        std::lock_guard<std::mutex> lock(handle->mtx);
        if (!handle->rc) return TCBS_ROBOT_COMM_ERR_RUNTIME;
        const RobotController::State st = handle->rc->getState();
        return copySeqTo(st.mpc.pred_joint_seq[which], out, max_len);
    } catch (const std::exception& e) {
        reportException("tcbs_robot_controller_get_pred_joint_sequence", e);
        return TCBS_ROBOT_COMM_ERR_RUNTIME;
    } catch (...) {
        return TCBS_ROBOT_COMM_ERR_UNKNOWN;
    }
}

} // namespace tcbs
