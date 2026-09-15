#include "tcbs/communication/McuDataPreprocessor.h"

namespace tcbs {

mcu::SendPacket McuDataPreprocessor::processSend(const mcu::SendPacket& packet) const {
    mcu::SendPacket r = packet;
    // pitch: 沿用旧标定（pitch 的编码器/执行器映射由上位机负责）
    r.pitch_target_angle = static_cast<float>(
        params_.send_pitch_scale * packet.pitch_target_angle + params_.send_pitch_offset);
    // 大 yaw（默认恒等）
    r.yaw_big_target_angle =
        params_.send_big_yaw_scale * packet.yaw_big_target_angle + params_.send_big_yaw_offset;
    r.yaw_big_target_velocity =
        static_cast<float>(params_.send_big_velocity_scale * packet.yaw_big_target_velocity);
    r.yaw_big_torque =
        static_cast<float>(params_.send_big_torque_scale * packet.yaw_big_torque);
    // 小 yaw（默认恒等）
    r.yaw_small_target_angle = static_cast<float>(
        params_.send_small_yaw_scale * packet.yaw_small_target_angle + params_.send_small_yaw_offset);
    r.yaw_small_target_velocity =
        static_cast<float>(params_.send_small_velocity_scale * packet.yaw_small_target_velocity);
    r.yaw_small_torque =
        static_cast<float>(params_.send_small_torque_scale * packet.yaw_small_torque);
    return r;
}

mcu::ReceivePacket McuDataPreprocessor::processReceive(const mcu::ReceivePacket& packet) const {
    mcu::ReceivePacket r = packet;
    // pitch: 沿用旧标定（mcu_pitch_angle → 实际关节角）
    r.pitch_angle = static_cast<float>(
        params_.recv_pitch_scale * packet.pitch_angle + params_.recv_pitch_offset);
    // 大 yaw（默认恒等）
    r.yaw_big_angle   = params_.recv_big_yaw_scale * packet.yaw_big_angle
                      + params_.recv_big_yaw_offset;
    r.yaw_big_omega   = static_cast<float>(params_.recv_big_omega_scale * packet.yaw_big_omega);
    // 小 yaw（默认恒等）
    r.yaw_small_angle = static_cast<float>(
        params_.recv_small_yaw_scale * packet.yaw_small_angle + params_.recv_small_yaw_offset);
    r.yaw_small_omega = static_cast<float>(params_.recv_small_omega_scale * packet.yaw_small_omega);
    return r;
}

} // namespace tcbs
