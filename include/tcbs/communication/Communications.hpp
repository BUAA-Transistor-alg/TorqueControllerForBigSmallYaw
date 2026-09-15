// Communications.hpp — 基于 SerialProtocol 模板的具体通信类型定义（双级 yaw 版本）
//
// McuCommunication : 与电控（MCU）通信，CRC8，前导 0x42 0x52 0x03
// ImuCommunication : 与大 yaw 上的 IMU 通信，CRC32，前导 0xA7 0xB6 0xC5
//
// RobotCommunication 组合: 两个串口 + McuDataPreprocessor（映射）+ YawStateEstimator
//   - MCU 回调里做映射（processReceive）并喂状态估计器
//   - IMU 回调里喂状态估计器（IMU 固定在大 yaw 转子，可信、高频）
#ifndef TCBS_COMMUNICATIONS_HPP
#define TCBS_COMMUNICATIONS_HPP

#include "tcbs/communication/SerialProtocol.hpp"
#include "tcbs/communication/Protocol.hpp"
#include "tcbs/communication/CRC.h"
#include "tcbs/communication/McuDataPreprocessor.h"
#include "tcbs/communication/YawStateEstimator.h"
#include <string>
#include <mutex>

namespace tcbs {

// ── 端口筛选函数 ──
inline bool mcuPortSelector(const std::string& product_info) {
    return product_info != "AutoAim_IMU_Com";
}
inline bool imuPortSelector(const std::string& product_info) {
    return product_info == "AutoAim_IMU_Com";
}

// ── 具体通信类型别名 ──
using McuCommunication = SerialProtocol<
    mcu::SendPacket,
    mcu::ReceivePacket,
    CRC8_Check_Sum,
    mcuPortSelector,
    mcu::PREAMBLE_SIZE
>;

using ImuCommunication = SerialProtocol<
    imu::SendPacket,
    imu::ReceivePacket,
    CRC32_Calculate,
    imuPortSelector,
    imu::PREAMBLE_SIZE
>;

// ============================================================================
// RobotCommunication — 组合 MCU 与 IMU 通信、数据预处理与状态估计
// ============================================================================
class RobotCommunication {
public:
    struct LatestData {
        bool               imu_valid = false;
        imu::ReceivePacket imu_packet{};
        bool               mcu_valid = false;
        mcu::ReceivePacket mcu_packet{};   // 已映射
        uint8_t            mcu2_seq = 0;               // MCU2 新样本序号（值保持时不变）
    };

    explicit RobotCommunication(
        const McuDataPreprocessor::LinearParams& mcu_linear_params = McuDataPreprocessor::LinearParams{},
        const YawStateEstimator::Config& estimator_cfg = YawStateEstimator::Config{})
        : preprocessor_(mcu_linear_params)
        , estimator_(estimator_cfg)
        , mcu_serial_([this](const mcu::ReceivePacket& pkt) { onMcuReceive(pkt); }, false)
        , imu_serial_([this](const imu::ReceivePacket& pkt) { onImuReceive(pkt); }, false)
    {
        mcu_serial_.startWorker();
        imu_serial_.startWorker();
    }

    ~RobotCommunication() {
        mcu_serial_.stopWorker();
        imu_serial_.stopWorker();
    }

    // 获取最新原始数据（MCU 数据已按映射参数预处理）
    LatestData getLatestData() {
        LatestData data;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            if (has_imu_data_) {
                data.imu_packet = latest_imu_packet_;
                data.imu_valid = true;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            if (has_mcu_data_) {
                data.mcu_packet = latest_mcu_packet_;
                data.mcu2_seq = latest_mcu2_seq_;
                data.mcu_valid = true;
            }
        }
        return data;
    }

    // 发送 MCU 数据（发送前按映射参数预处理）
    bool sendToMcu(mcu::SendPacket packet) {
        mcu::SendPacket processed = preprocessor_.processSend(packet);
        return mcu_serial_.sendData(processed);
    }

    // 发送 IMU 数据（心跳等，无预处理）
    bool sendToImu(imu::SendPacket packet) {
        return imu_serial_.sendData(packet);
    }

    void stop() {
        mcu_serial_.stopWorker();
        imu_serial_.stopWorker();
    }

    // 状态估计输出（线程安全）
    YawStateEstimator::Estimate getEstimate() const { return estimator_.estimate(); }

    YawStateEstimator& estimator() { return estimator_; }

    const McuDataPreprocessor& preprocessor() const { return preprocessor_; }
    void setLinearParams(const McuDataPreprocessor::LinearParams& p) { preprocessor_.setParams(p); }

private:
    void onImuReceive(const imu::ReceivePacket& packet) {
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            latest_imu_packet_ = packet;
            has_imu_data_ = true;
        }
        estimator_.onImu(packet.euler_yaw, packet.euler_pitch, packet.euler_roll,
                         packet.gx, packet.gy, packet.gz);
    }

    void onMcuReceive(const mcu::ReceivePacket& packet) {
        const mcu::ReceivePacket processed = preprocessor_.processReceive(packet);
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            latest_mcu_packet_ = processed;
            latest_mcu2_seq_ = packet.mcu2_seq;
            has_mcu_data_ = true;
        }
        estimator_.onMcu(processed.yaw_big_angle, processed.yaw_big_omega,
                         processed.yaw_small_angle, processed.yaw_small_omega,
                         processed.pitch_angle,
                         processed.chassis_imu_yaw, processed.chassis_imu_omega,
                         packet.mcu2_seq);
    }

    McuDataPreprocessor  preprocessor_;
    YawStateEstimator    estimator_;
    McuCommunication     mcu_serial_;
    ImuCommunication     imu_serial_;

    std::mutex         imu_mutex_;
    imu::ReceivePacket latest_imu_packet_{};
    bool               has_imu_data_ = false;

    std::mutex         mcu_mutex_;
    mcu::ReceivePacket latest_mcu_packet_{};
    uint8_t            latest_mcu2_seq_ = 0;
    bool               has_mcu_data_ = false;
};

} // namespace tcbs

#endif // TCBS_COMMUNICATIONS_HPP