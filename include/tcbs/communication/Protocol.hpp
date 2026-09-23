#pragma once
#include <cstdint>
#include <cstddef>

namespace tcbs {

// ============================================================================
// MCU（电控）通信协议 v0x03 —— 双级 yaw（大yaw + 小yaw）版本
//
// 帧格式（与 SerialProtocol 一致）: [前导 3B][data_size 1B][payload data_size B][CRC8 1B]
// 前导: 0x42 0x52 0x03
//
// 设计要点:
//   - 两个 yaw 关节各有独立通道: 模式位 + 目标角/角速度 + 力矩
//   - 大 yaw 目标角为 double（多圈连续语义，不限位）
//   - 小 yaw 目标角为 float（关节系相对角，机械行程 −25° ~ +20°（非对称），由上位机按该行程限位，
//     电控侧再做目标夹取/硬限位保护）
//   - pitch 仍为单通道；其"编码器/指令 → 角度"线性映射由上位机
//     McuDataPreprocessor 完成（电控侧不做该映射）
//   - yaw 轴的角度换算（编码器计数 → 弧度、多圈累计）仍由电控完成
//   - **反馈通道的更新率并不一致**（真实系统: 上位机直连 MCU1 与大 yaw 上的 IMU，
//     MCU2 经 MCU1↔MCU2 链路，该链路约 10Hz 量级且间隔不规则）:
//       * 小 yaw / pitch 编码器（MCU1 直控）: 每帧都是新值（实时可信）
//       * 大 yaw 反馈、底盘 IMU（MCU2）: **更新率低且间隔不规则**；两次更新之间
//         MCU1 会**重复发送上一次的值**（值被保持）。因此上位机不能"每收到一帧就
//         当作新样本"。
//   - **不使用任何电控侧时钟**（MCU 端计时不可用，只有上位机能计时）。改用
//     **单个"MCU2 新样本序号"** 解决"新样本 vs 被保持的旧值"的歧义:
//       mcu2_seq : MCU1 每从 MCU2 取到一次新数据就 +1（大 yaw 与底盘 IMU 同源、
//                  同一次取数里一起刷新，故共用一个序号）
//     判定规则（上位机）: **序号变化** 或 **值变化** ⇒ 视为新样本（即使值恰好相同，
//     序号变化也能识别）；序号与值都不变 ⇒ 值被保持。
//     值的年龄由上位机自己的时钟测量（从"首次看到该序号"起算）+ 链路传输时延。
//     不做"可用性"检测: 上电后第一帧即视为一次更新，无需 0 哨兵约定。
// ============================================================================
namespace mcu {

// 帧同步前导字节数（frame_header1 + frame_header2 + protocol_version）
constexpr size_t PREAMBLE_SIZE = 3;

// 协议版本
constexpr uint8_t PROTOCOL_VERSION = 0x03;

// ── yaw 关节控制模式 ──
// ★ 2026-09-23（用户要求）: **1 = 仅力矩**、**2 = 力矩 + 内环**；
//   其余任何取值（含 0）一律视为**非法模式** ⇒ 力矩按 0 处理（保守，见电控端实现）。
//   与旧版单轴示例（`yaw_torque_only_mode`，1 = 仅力矩）以及采集脚本一致。
enum YawMode : uint8_t {
    YAW_MODE_TORQUE_ONLY     = 1, // 仅力矩: 电控直接施加 yaw_torque
    YAW_MODE_TORQUE_PLUS_PID = 2, // 力矩 + 位置/速度内环: τ = kp(θ*−θ) + kd(ω*−ω) + yaw_torque
};

#pragma pack(push, 1)
struct SendPacket
{
    uint8_t frame_header1 = 0x42;
    uint8_t frame_header2 = 0x52;
    uint8_t protocol_version = PROTOCOL_VERSION;
    uint8_t data_size = 36;             // 1+1+4 +1+8+4+4 +1+4+4+4
    // ── 通用 ──
    uint8_t auto_aim_enable;            // 自瞄总开关（与电控手动开关相与）
    uint8_t fire;                       // 火控
    float   pitch_target_angle;         // pitch 目标角（原始语义，由预处理器线性映射后发出）
    // ── 大 yaw ──
    uint8_t yaw_big_mode;               // YawMode
    double  yaw_big_target_angle;       // rad，关节系，多圈连续
    float   yaw_big_target_velocity;    // rad/s
    float   yaw_big_torque;             // N·m（前馈/纯力矩）
    // ── 小 yaw ──
    uint8_t yaw_small_mode;             // YawMode
    float   yaw_small_target_angle;     // rad，关节系（相对大yaw），行程 −25° ~ +20°（非对称）
    float   yaw_small_target_velocity;  // rad/s
    float   yaw_small_torque;           // N·m
    uint8_t crc8;
};

struct ReceivePacket
{
    uint8_t frame_header1 = 0x42;
    uint8_t frame_header2 = 0x52;
    uint8_t protocol_version = PROTOCOL_VERSION;
    uint8_t data_size;                  // 应为 42
    // ── 弹速/火控相关信息 ──
    float   bullet_velocity;            // m/s
    float   pitch_angle;                // pitch 关节原始角（由预处理器映射为实际角度）
    // ── 大 yaw（本通道存在链路延迟与误差，另有 IMU 提供其高频可信分量）──
    double  yaw_big_angle;              // rad，多圈连续（电控按编码器累计）
    float   yaw_big_omega;              // rad/s
    // ── 小 yaw（可信、实时）──
    float   yaw_small_angle;            // rad，相对大yaw的关节角
    float   yaw_small_omega;            // rad/s
    // ── 底盘 IMU（底盘自身姿态，用于底盘旋转补偿）──
    float   chassis_imu_yaw;            // 0 ~ 2π
    float   chassis_imu_omega;          // rad/s（底盘 yaw 角速度）
    // ── 状态 ──
    uint8_t mark;                       // 递增标志位
    uint8_t color;                      // 颜色
    uint8_t auto_aim_switch;            // 电控自瞄开关
    uint8_t yaw_big_temperature;        // 大 yaw 电机温度
    uint8_t yaw_small_temperature;      // 小 yaw 电机温度
    uint8_t mcu2_seq;                   // MCU2 数据新样本序号（每次取到新数据 +1；值保持时不变）
    uint8_t crc8;
};
#pragma pack(pop)

// 编译期一致性检查（payload 长度必须与 data_size 一致）
static_assert(sizeof(SendPacket) == 3 + 1 + 36 + 1,
              "SendPacket 布局与 data_size 不一致");
static_assert(sizeof(ReceivePacket) == 3 + 1 + 42 + 1,
              "ReceivePacket 布局与 data_size 不一致");
// 关键字段偏移（电控侧应逐字段核对，避免静默错位）
static_assert(offsetof(ReceivePacket, yaw_big_angle) == 12, "偏移变化");
static_assert(offsetof(ReceivePacket, yaw_small_angle) == 24, "偏移变化");
static_assert(offsetof(ReceivePacket, chassis_imu_yaw) == 32, "偏移变化");
static_assert(offsetof(ReceivePacket, yaw_big_temperature) == 43, "偏移变化");
static_assert(offsetof(ReceivePacket, mcu2_seq) == 45, "偏移变化");
static_assert(offsetof(ReceivePacket, crc8) == 46, "偏移变化");

} // namespace mcu


// ============================================================================
// IMU 通信协议（与大yaw 转子固连的 IMU）
// 语义变化（相对旧版本）: IMU 不再位于云台终端（pitch 之后），
// 而是固定在大 yaw 转子上，因此 R_imu = R_C · Rz(θ_big) · R_mount，
// 其欧拉角与陀螺可实时给出"大 yaw 平台的世界姿态与角速度"。
// ============================================================================
namespace imu {

constexpr size_t PREAMBLE_SIZE = 3;

#pragma pack(push, 1)
struct SendPacket
{
    uint8_t frame_header1 = 0xA7;
    uint8_t frame_header2 = 0xB6;
    uint8_t frame_header3 = 0xC5;
    uint8_t data_size = 0;              // 无数据载荷，仅心跳
    uint32_t crc32;
};

struct ReceivePacket
{
    uint8_t frame_header1 = 0xA7;
    uint8_t frame_header2 = 0xB6;
    uint8_t frame_header3 = 0xC5;
    uint8_t data_size;
    float gx;                           // IMU 本体系角速度 (rad/s)
    float gy;
    float gz;
    float ax;                           // 加速度 (m/s²)
    float ay;
    float az;
    double euler_yaw;                   // 世界系欧拉角 (rad)，ZXY 约定
    double euler_pitch;
    double euler_roll;
    uint32_t dt_one_tenth_ms;           // 本帧间隔（0.1ms 单位）
    uint32_t crc32;
};
#pragma pack(pop)

} // namespace imu

} // namespace tcbs
