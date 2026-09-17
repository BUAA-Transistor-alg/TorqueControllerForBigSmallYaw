#pragma once

#include "tcbs/communication/Protocol.hpp"
#include <cmath>

namespace tcbs {

// ============================================================================
// McuDataPreprocessor — MCU 通信数据预处理（所有"编码器/指令 → 角度"映射集中于此）
//
// 本结构是**全部可修改的映射参数**的唯一入口（协议里 yaw 仍由电控换算成弧度，
// pitch 由上位机映射；每种映射都保留 scale/offset，默认恒等或沿用旧标定值）。
//
// 映射定义（线性）:
//   接收: angle = recv_*_scale * raw + recv_*_offset        （弧度）
//          omega = recv_*_omega_scale * raw_omega           （rad/s）
//   发送: value = send_*_scale * value + send_*_offset
//          torque = send_*_torque_scale * torque            （N·m → 电控单位）
//
// 关于 yaw: 协议规定电控侧完成"编码器计数 → 弧度、多圈累计"，因此默认 scale=1/offset=0；
// 若某天改为电控发原始计数，只需把 scale 设为 计数→弧度的系数即可，无需改协议。
// ============================================================================
class McuDataPreprocessor {
public:
    struct LinearParams {
        // ── pitch（占位恒等值；**必须按 docs/calibration.md §3.3 重新标定**）──
        // 注意: 旧版这四个数对应旧构型的 "imu_euler_pitch ↔ mcu_pitch_angle" 语义
        // （IMU 装在云台终端），二者互不为逆（复合比例 ≈23）。本构型 IMU 移到大 yaw 上，
        // pitch 只表示"关节角 ↔ 电控原始值"的映射，因此收敛为恒等占位；
        // 若照抄旧值会导致 pitch 目标角被放大 20 倍以上（危险）。
        double send_pitch_scale  =   1.0;        // 关节角 → 电控 pitch 目标值
        double send_pitch_offset =   0.0;
        double recv_pitch_scale  =   1.0;        // 电控原始 pitch 值 → 关节角
        double recv_pitch_offset =   0.0;

        // ── 大 yaw（★ 电控侧该轴的正方向与本工程约定**相反** ⇒ 位置/速度/力矩
        //    收发两个方向都取负号；**温度、模式位**不参与映射，不受影响）──
        //   本工程约定: yaw 绕 +z、从上方看逆时针为正（x→y，见 docs/model.md §2.1）。
        //   `mapped = scale·raw + offset`、`raw_cmd = scale·θ + offset`，
        //   这里 scale = −1、offset = 0 ⇒ 收发互为逆映射，等价于"整体镜像"。
        //   平衡校验（与用户给的判据一致: 下发值 == 编码器回读值 ⇒ 不动）:
        //     发 raw_cmd = −θ，回读 raw_fb = −θ ⇒ 二者相等 ⇒ 不动 ✔
        double recv_big_yaw_scale      = -1.0;
        double recv_big_yaw_offset     =  0.0;
        double recv_big_omega_scale    = -1.0;
        double send_big_yaw_scale      = -1.0;
        double send_big_yaw_offset     =  0.0;
        double send_big_velocity_scale = -1.0;
        double send_big_torque_scale   = -1.0;

        // ── 小 yaw ──
        // ★ 小 yaw 零位（已标定）: 物理零点处电控上报 **+1.025466 rad**
        //   ⇒ 按 `mapped = raw·scale + offset` 反推: offset = −1.025466
        //   （与 calibrate_small_zero.py 的约定一致: Δoffset = −mapped_zero，见该脚本 §"零位建议"）。
        //   校验: raw = +1.025466 ⇒ mapped = 1.0·1.025466 − 1.025466 = 0 ✔
        //   ⇒ **全系统（MPC 限位 ±30°、回中中心 0、电控夹取）都以这个零点解释角度**。
        double recv_small_yaw_scale      = 1.0;
        double recv_small_yaw_offset     = -1.025466;   // = −(零点处的电控原始读数)
        double recv_small_omega_scale    = 1.0;
        // ★ 下发侧必须**反向**补同一个偏移: 电控内环的判据是"**下发值 == 编码器值 ⇒ 不动**"，
        //   即下发与上报共用同一个原始坐标系 ⇒ 要停在关节角 θ，必须发 `raw = θ + 1.025466`。
        //   平衡校验: 发 raw_cmd = θ+1.025466，回读 raw_fb = θ+1.025466 ⇒ 二者相等 ⇒ 不动 ✔
        //   （若只改 recv 不改 send，则发 0 会被电控理解成 −58.75° 的位置，往错误方向跑满行程。）
        double send_small_yaw_scale      = 1.0;
        double send_small_yaw_offset     = +1.025466;   // = +(零点处的电控原始读数)
        double send_small_velocity_scale = 1.0;
        double send_small_torque_scale   = 1.0;
    };

    static LinearParams defaultParams() { return LinearParams{}; }

    // 注意: 默认实参必须调用静态函数（函数体属"完整类上下文"），
    // 直接写 LinearParams{} 会因默认成员初始化器尚未就绪而编译失败
    explicit McuDataPreprocessor(const LinearParams& params = defaultParams())
        : params_(params) {}

    const LinearParams& params() const { return params_; }
    void setParams(const LinearParams& p) { params_ = p; }

    // ── 发送包预处理 ──
    mcu::SendPacket processSend(const mcu::SendPacket& packet) const;

    // ── 接收包预处理（yaw 映射为线性；pitch 沿用旧标定）──
    mcu::ReceivePacket processReceive(const mcu::ReceivePacket& packet) const;

private:
    LinearParams params_;
};

} // namespace tcbs
