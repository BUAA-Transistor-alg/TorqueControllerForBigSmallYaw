// ============================================================================
// StrictPose.h — 「严格反解数据包」: 以 IMU 为准确值，反解底盘姿态，
//                并把**反解用到的全部数据**打包出去，供外部独立复算整车姿态。
//
// 设计目标（用户要求）:
//   1) **IMU 数据是准确值**（唯一绝对姿态来源），其余姿态全部由它反解；
//   2) **反解底盘**（chassis）姿态；
//   3) 包里带齐"反解时用到的每一项数据"（IMU 欧拉角、各关节角、安装矩阵参数、构型开关），
//      使得外部**只依赖这一个包**就能重构整车姿态；
//   4) 忽略浮点误差时，用本包重构出的 **IMU 所在位置**的姿态必须**严格等于** IMU 实际数据
//      —— 即 `strictPoseReconstructImu()` 的结果与 `R_world_imu` 逐元素相等（测试里按 1e-15 校核）。
//
// ── 坐标链（A 系 = 大 yaw 转子系, C 系 = 底盘系, H 系 = 头部, IMU = IMU 自身坐标系）──
//   构型 0（`IMU_ON_BIG_YAW`，默认: IMU 固定在大 yaw 转子上）:
//       R_world_imu = R_world_chassis · Rz(θ_b) · R_A_IMU
//       ⇒ 反解:  R_world_chassis = R_world_imu · R_A_IMUᵀ · Rz(θ_b)ᵀ
//   构型 1（`IMU_ON_HEAD`: IMU 在头上, pitch 之后）:
//       R_world_imu = R_world_chassis · Rz(θ_b) · Rz(θ_s) · Rx(θ_p) · R_H_IMU
//       ⇒ 反解:  R_world_chassis = R_world_imu · R_H_IMUᵀ · Rx(θ_p)ᵀ · Rz(θ_s)ᵀ · Rz(θ_b)ᵀ
//   其中 R_A_IMU / R_H_IMU 由 Config 的安装欧拉角（ZXY）给出。
//
// ── 精度说明（重要）──
//   `R_world_imu` 来自 IMU，实时且（在 ON_BIG_YAW 下）无链路延迟；
//   而反解底盘还要用到 **θ_b（大 yaw 关节角）**，它来自"延迟 + 被保持"的编码器链路
//   （已做 IMU 速率一阶外推补偿，见 `YawStateEstimator`）。因此:
//     · **重构恒等式**（用包内 θ_b 复算 IMU 姿态）永远严格成立，与 θ_b 是否准确无关；
//     · **反解出的底盘姿态**的误差 = θ_b 的误差（外加安装矩阵标定误差）。
//   包里同时给出 `big_joint_angle` 与其年龄 `big_joint_angle_age`，便于外部判断可信度。
//
// 说明: 本结构只包含**姿态（旋转）**。整车平动/位置没有绝对观测量（无 GNSS/UWB 等），
//       因此"位姿"中的平移部分不在本包内；几何量（d、c、bore 等）见 `ModelParams` /
//       `YawStateEstimator::Config::bore`。
// ============================================================================
#ifndef TCBS_DUAL_YAW_STRICT_POSE_H
#define TCBS_DUAL_YAW_STRICT_POSE_H

#include "tcbs/common/RotationUtils.h"

namespace tcbs {

namespace dual_yaw {

// IMU 安装位置（与 YawStateEstimator::Config::ImuLocation 取值一致）
enum class StrictPoseImuLocation : int {
    ON_BIG_YAW = 0,   // IMU 固定在大 yaw 转子 A 上（默认）
    ON_HEAD    = 1,   // IMU 装在头上（pitch 之后）
};

struct StrictPose {
    // ── ① 反解输入快照（全部 wrap 到 (−π, π]）──
    double imu_euler_yaw = 0.0;      // IMU 原始欧拉角（世界←IMU, ZXY）
    double imu_euler_pitch = 0.0;
    double imu_euler_roll = 0.0;
    int    imu_location = 0;         // StrictPoseImuLocation
    double big_joint_angle = 0.0;    // θ_b（反解所用；已做延迟补偿，见文件头"精度说明"）
    double small_joint_angle = 0.0;  // θ_s
    double pitch_joint_angle = 0.0;  // θ_p
    // ── ② 安装旋转参数（反解用到的一切标定量的**快照**）──
    double mount_yaw = 0.0, mount_pitch = 0.0, mount_roll = 0.0;            // R_A_IMU
    double head_mount_yaw = 0.0, head_mount_pitch = 0.0, head_mount_roll = 0.0;  // R_H_IMU
    // ── ③ 反解结果 ──
    double R_world_imu[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};      // 由 ① 的欧拉角重构（与 IMU 数据一致）
    double chassis_euler_yaw = 0.0, chassis_euler_pitch = 0.0, chassis_euler_roll = 0.0;
    double R_world_chassis[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // ★ 反解出来的底盘姿态
    // ── ④ 顺带给出的整车其余环节（同一链条，供外部一次拿全）──
    double R_world_platform[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // 大 yaw 转子 A 系
    double R_world_head[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};     // 头 H 系
    // ── ④' 各环节 x 轴的**世界方位角**（wrap 到 (−π, π]）──
    //   与 `YawStateEstimator::Estimate::platform_azimuth / chassis_azimuth /
    //     small_output_azimuth` **同一定义**（估计器里那些量是多圈解卷绕值，这里按圈 wrap）。
    //   注意: 当该环节相对水平面有俯仰/横滚倾斜时，"x 轴方位角" 与 ZXY 欧拉 yaw **不相等**
    //   （差约 pitch·roll），这是两种不同约定；两者本包都给出。
    double platform_azimuth = 0.0;
    double chassis_azimuth = 0.0;
    double head_azimuth = 0.0;
    // ── ⑤ 自洽性（外部可据此判断"包是否完整/一致"）──
    // 纯自洽性自检: ‖reconstruct(R_world_imu) − R_world_imu‖_F。由构造保证 ≈0（实测 ~1e-16），
    // 外部可用它校验"包是否被正确搬运/序列化"，不是可用性标志。
    double recon_err_rot = 0.0;
    double big_joint_angle_age = -1.0;  // θ_b 的实测年龄（s；-1 = 未知/从未更新）
};

// 说明（与原仓库 `YawChassisFusion::StrictPose` 一致的设计约定）:
//   · **没有 valid 标志，始终解算** —— 本包任何时刻都可读、都有意义；
//   · 所需数据缺失时用**历史值或 0** 参与计算（IMU 未到达 ⇒ 欧拉角 0 ⇒ 姿态为单位阵；
//     关节角未到达 ⇒ 0；安装参数来自配置，恒有值），因此**重构关系恒成立**；
//   · 所有角度 wrap 到 (−π, π]。

inline void matToArray(const rot::Mat3& R, double out[9]) {
    out[0] = R.m[0][0]; out[1] = R.m[0][1]; out[2] = R.m[0][2];
    out[3] = R.m[1][0]; out[4] = R.m[1][1]; out[5] = R.m[1][2];
    out[6] = R.m[2][0]; out[7] = R.m[2][1]; out[8] = R.m[2][2];
}

inline rot::Mat3 arrayToMat(const double in[9]) {
    rot::Mat3 R;
    R.m[0][0] = in[0]; R.m[0][1] = in[1]; R.m[0][2] = in[2];
    R.m[1][0] = in[3]; R.m[1][1] = in[4]; R.m[1][2] = in[5];
    R.m[2][0] = in[6]; R.m[2][1] = in[7]; R.m[2][2] = in[8];
    return R;
}

// ── 用包内数据重构 IMU 所在位置的姿态（反解的逆运算）──
//   构型 0: R = R_chassis · Rz(θ_b) · R_A_IMU
//   构型 1: R = R_chassis · Rz(θ_b) · Rz(θ_s) · Rx(θ_p) · R_H_IMU
//   忽略浮点误差时结果应严格等于 `sp.R_world_imu`。
inline rot::Mat3 strictPoseReconstructImu(const StrictPose& sp) {
    const rot::Mat3 R_chassis = arrayToMat(sp.R_world_chassis);
    rot::Mat3 R = rot::mul(R_chassis, rot::rotZ(sp.big_joint_angle));
    if (sp.imu_location == static_cast<int>(StrictPoseImuLocation::ON_HEAD)) {
        R = rot::mul(R, rot::rotZ(sp.small_joint_angle));
        R = rot::mul(R, rot::rotX(sp.pitch_joint_angle));
        R = rot::mul(R, rot::eulerZXY(sp.head_mount_yaw, sp.head_mount_pitch,
                                      sp.head_mount_roll));
    } else {
        R = rot::mul(R, rot::eulerZXY(sp.mount_yaw, sp.mount_pitch, sp.mount_roll));
    }
    return R;
}

// ── 由 IMU 姿态 + 各关节角 + 安装参数构造整包（机器人侧调用）──
inline StrictPose makeStrictPose(double imu_yaw, double imu_pitch, double imu_roll,
                                 int imu_location,
                                 double theta_big, double theta_small, double theta_pitch,
                                 double mount_yaw, double mount_pitch, double mount_roll,
                                 double head_mount_yaw, double head_mount_pitch,
                                 double head_mount_roll,
                                 double big_joint_age = -1.0) {
    StrictPose sp;
    sp.imu_euler_yaw = std::remainder(imu_yaw, 2.0 * M_PI);
    sp.imu_euler_pitch = std::remainder(imu_pitch, 2.0 * M_PI);
    sp.imu_euler_roll = std::remainder(imu_roll, 2.0 * M_PI);
    sp.imu_location = imu_location;
    sp.big_joint_angle = std::remainder(theta_big, 2.0 * M_PI);
    sp.small_joint_angle = std::remainder(theta_small, 2.0 * M_PI);
    sp.pitch_joint_angle = std::remainder(theta_pitch, 2.0 * M_PI);
    sp.mount_yaw = mount_yaw; sp.mount_pitch = mount_pitch; sp.mount_roll = mount_roll;
    sp.head_mount_yaw = head_mount_yaw;
    sp.head_mount_pitch = head_mount_pitch;
    sp.head_mount_roll = head_mount_roll;
    sp.big_joint_angle_age = big_joint_age;

    const rot::Mat3 R_imu = rot::eulerZXY(sp.imu_euler_yaw, sp.imu_euler_pitch,
                                         sp.imu_euler_roll);
    matToArray(R_imu, sp.R_world_imu);

    // ── 反解: 由 IMU 往底盘走（运动学链的逆）──
    //   chassis → Rz(θ_b) → [Rz(θ_s) → Rx(θ_p)] → 安装矩阵 → IMU
    rot::Mat3 R_chassis, R_platform, R_head;
    if (sp.imu_location == static_cast<int>(StrictPoseImuLocation::ON_HEAD)) {
        const rot::Mat3 R_H_IMU = rot::eulerZXY(head_mount_yaw, head_mount_pitch,
                                                head_mount_roll);
        R_head = rot::mul(R_imu, rot::transpose(R_H_IMU));                 // 头（世界）
        const rot::Mat3 R_B = rot::mul(R_head, rot::transpose(rot::rotX(sp.pitch_joint_angle)));
        R_platform = rot::mul(R_B, rot::transpose(rot::rotZ(sp.small_joint_angle)));  // A 系
        R_chassis = rot::mul(R_platform, rot::transpose(rot::rotZ(sp.big_joint_angle)));
    } else {
        const rot::Mat3 R_A_IMU = rot::eulerZXY(mount_yaw, mount_pitch, mount_roll);
        R_platform = rot::mul(R_imu, rot::transpose(R_A_IMU));              // A 系
        R_head = rot::mul(R_platform, rot::rotZ(sp.small_joint_angle));
        R_head = rot::mul(R_head, rot::rotX(sp.pitch_joint_angle));
        R_chassis = rot::mul(R_platform, rot::transpose(rot::rotZ(sp.big_joint_angle)));
    }
    matToArray(R_chassis, sp.R_world_chassis);
    matToArray(R_platform, sp.R_world_platform);
    matToArray(R_head, sp.R_world_head);
    // x 轴世界方位角（与估计器 Estimate 同定义）
    sp.platform_azimuth = std::atan2(R_platform.m[1][0], R_platform.m[0][0]);
    sp.chassis_azimuth = std::atan2(R_chassis.m[1][0], R_chassis.m[0][0]);
    sp.head_azimuth = std::atan2(R_head.m[1][0], R_head.m[0][0]);
    rot::matToEulerZXY(R_chassis, sp.chassis_euler_yaw, sp.chassis_euler_pitch,
                       sp.chassis_euler_roll);
    sp.chassis_euler_yaw = std::remainder(sp.chassis_euler_yaw, 2.0 * M_PI);
    sp.chassis_euler_pitch = std::remainder(sp.chassis_euler_pitch, 2.0 * M_PI);
    sp.chassis_euler_roll = std::remainder(sp.chassis_euler_roll, 2.0 * M_PI);

    // ── 自洽性: 用同一包数据重构 IMU 姿态，应与 R_world_imu 严格一致 ──
    const rot::Mat3 R_rec = strictPoseReconstructImu(sp);
    double sse = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double d = R_rec.m[i][j] - R_imu.m[i][j];
            sse += d * d;
        }
    sp.recon_err_rot = std::sqrt(sse);
    return sp;
}

} // namespace dual_yaw

} // namespace tcbs

#endif // TCBS_DUAL_YAW_STRICT_POSE_H