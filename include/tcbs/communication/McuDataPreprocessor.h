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
        // ── pitch（★ **已标定**；由 `./build/tcbs_pitch_calibration` 两段拟合得到）──
        // 标定前提: IMU 必须在头上（构型 ON_HEAD，**本工程默认**），此时 `imu.euler_pitch`
        // 就是 pitch 关节角；若切成 ON_BIG_YAW 则真值变成合成倾角、这四个数无效。
        // ★ **两段互不为逆**（这是正常的，别"顺手改成互逆"）: 电控的目标角通道与反馈通道
        // 用的是**不同的单位/零点** ——
        //   recv: 关节角 = 0.006060·raw_fb − 198.645875   （raw_fb 是计数：关节角 0 ⇒ raw ≈ 32780 ≈ 2^15）
        //   send: raw_cmd = 21.337421·关节角 − 6.708668    （关节角 0 ⇒ raw ≈ 0）
        // 两个通道各自线性，但斜率与零点都不同 ⇒ 必须分别标定、分别使用。
        // 改机械/换电控/动过 IMU 安装后都要重标。
        double send_pitch_scale  =  21.337421;   // 关节角 → 电控 pitch 目标值
        double send_pitch_offset =  -6.708668;
        double recv_pitch_scale  =   0.006060;   // 电控原始 pitch 值 → 关节角
        double recv_pitch_offset = -198.645875;

        // ── 大 yaw（★ 电控侧该轴的正方向与本工程约定**相反** ⇒ 位置/速度/力矩
        //    收发两个方向都取负号；**温度、模式位**不参与映射，不受影响）──
        //   本工程约定: yaw 绕 +z、从上方看逆时针为正（x→y）。
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
        //   —— 读法: 跑 `./build/tcbs_test_serial`，人工把 **小 yaw 摆到机械零点**，
        //   读它打印的 `yaw_small_angle`（电控原始弧度），**取负**就是本 offset。
        //   不需要单独的标定程序（用户确认: 串口测试里直接读即可）。
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
