/* ============================================================================
 * dual_yaw_control.c —— RoboMaster 双级 yaw 云台 电控(MCU)侧示例实现
 *
 * 对应上位机侧权威定义: include/tcbs/communication/Protocol.hpp (协议 v0x03)
 *                        include/tcbs/communication/CRC.h  / src/communication/CRC.cpp
 * 旧版单轴写法参考:      mcu_code_demo/yaw_control_single_reference.c
 *
 * 本文件是**纯 C11 单文件示例**（本文件 = MCU1 的角色），不依赖任何上位机(C++)代码、
 * 不使用动态内存。需要用户填写的硬件相关桩函数集中在第 5 节，均以 "// TODO: 用户填写" 标注。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 零、真实的系统结构（决定了本文件里"新样本序号"这套机制的由来）
 *
 *     上位机(PC) ⟷ 串口 ⟷ MCU1 ⟷ 内部链路(不稳定) ⟷ MCU2
 *                          │                            │
 *                          ├─ pitch   (每帧新值)         ├─ 大 yaw 编码器/电机
 *                          ├─ 小 yaw  (每帧新值)         └─ 底盘 IMU
 *                          └─ 大 yaw 电机指令转发
 *
 *   - 上位机 ↔ MCU1 可视为实时；**MCU1 ↔ MCU2 链路约 10Hz 量级**（间隔不规则，
 *     基本不低于 3Hz）。
 *   - MCU1 在没收到 MCU2 新数据时会**继续沿用旧值**（值被保持）。
 *   - 因此上位机看到的每一帧里: pitch / 小 yaw 一定是新值; 大 yaw / 底盘 IMU 可能
 *     是几十到几百毫秒前的旧值。上位机必须能区分"新样本"与"被保持的旧值"。
 *   - **电控侧不提供任何时间信息**（MCU 端没有可靠毫秒计数，只有上位机能计时），
 *     所以用**一个"MCU2 新样本序号"**来表达"这是不是一个新样本":
 *         mcu2_seq : MCU1 每从 MCU2 取到一次新数据就 +1（值保持的帧里原样重复）
 *     大 yaw 与底盘 IMU **同源**: MCU2 一次把两路一起送来, 同一次取数里两路一起刷新,
 *     因此只需要一个序号（任一路报告"有新数据"就 +1, 同一次取数只 +1 一次）。
 *     序号变化 = 新样本（**即使值恰好与上一次相同**）; 序号不变 = 值被保持。
 *     上位机判定规则: **首帧到达 或 序号变化 或 值变化** 三者取或 ⇒ 视为新样本。
 *     （不做"可用性"检测: 不需要"0 = 从未收到"的哨兵约定, 上电初值 0 即可。）
 *     值的年龄由上位机用自己的时钟测量（从"首次看到该序号"起算）+ 链路传输时延，
 *     据此对大 yaw / 底盘姿态角做**角速度一阶外推**，对底盘角速度做**零阶保持**。
 *
 *   ⚠ 本文件的 `YAW_BIG_SOURCE_FROM_MCU2`（默认 1）就是描述这个结构:
 *     大 yaw 的**反馈值来自 MCU2**（带序号、可能被保持），大 yaw 的**控制指令
 *     转发给 MCU2**（MCU2 用自己实时的大 yaw 编码器跑内环）。
 *     若你的机器是"单 MCU 直接驱动大 yaw"（上一版示例的做法），把它设为 0 即可，
 *     此时大 yaw 每个反馈周期都是新样本，序号随之 +1。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 一、帧格式（与上位机 SerialProtocol 一致）
 *
 *   [前导 3B][data_size 1B][payload data_size B][CRC8 1B]
 *   前导 = 0x42 0x52 (版本 0x03) ; CRC8 = 查表法, 初值 0xFF,
 *   覆盖范围 = 整帧除最后一个 CRC 字节（即 sizeof(wire) - 1 字节）。
 *   注意: 上位机对"发送给电控的帧"和"电控发回的帧"使用**相同前导**，
 *         两者靠 data_size 区分（36 vs 42），本文件在接收状态机里同时校验两者。
 *   假设: 小端字节序（Cortex-M）、IEEE-754 float/double、double 按 8 字节裸拷贝。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 二、PC → MCU 字节布局  (上位机 mcu::SendPacket, 总长 41B, data_size = 36)
 *
 *   偏移  长度 类型  字段                        语义
 *   ----  ---- ----  --------------------------  --------------------------------
 *    0     1   u8    frame_header1               固定 0x42
 *    1     1   u8    frame_header2               固定 0x52
 *    2     1   u8    protocol_version            固定 0x03
 *    3     1   u8    data_size                   固定 36
 *    4     1   u8    auto_aim_enable             自瞄总开关(与电控手动开关相与)
 *    5     1   u8    fire                        火控
 *    6     4   f32   pitch_target_angle          pitch 目标角(原始语义, 上位机已映射)
 *   10     1   u8    yaw_big_mode                0=仅力矩 1=力矩+位置/速度内环
 *   11     8   f64   yaw_big_target_angle        rad, 关节系, **多圈连续**
 *   19     4   f32   yaw_big_target_velocity     rad/s
 *   23     4   f32   yaw_big_torque              N·m(前馈 / 纯力矩)
 *   27     1   u8    yaw_small_mode              0=仅力矩 1=力矩+位置/速度内环
 *   28     4   f32   yaw_small_target_angle      rad, 关节系(相对大yaw), 行程 −25° ~ +20°
 *   32     4   f32   yaw_small_target_velocity   rad/s
 *   36     4   f32   yaw_small_torque            N·m
 *   40     1   u8    crc8                        CRC8(byte 0..39), 初值 0xFF
 *   ---- 41 字节 (payload = 偏移 4..39 = 36 字节) ----
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 三、MCU → PC 字节布局  (上位机 mcu::ReceivePacket, 总长 47B, data_size = 42)
 *
 *   偏移  长度 类型  字段                        语义
 *   ----  ---- ----  --------------------------  --------------------------------
 *    0     1   u8    frame_header1               固定 0x42
 *    1     1   u8    frame_header2               固定 0x52
 *    2     1   u8    protocol_version            固定 0x03
 *    3     1   u8    data_size                   固定 42
 *    4     4   f32   bullet_velocity             m/s
 *    8     4   f32   pitch_angle                 pitch 关节原始角(上位机做映射)
 *   12     8   f64   yaw_big_angle               rad,多圈连续; 来自 MCU2, **值可能被保持**
 *   20     4   f32   yaw_big_omega               rad/s(同上, 与 yaw_big_angle 同批)
 *   24     4   f32   yaw_small_angle             rad, 相对大yaw的关节角(每帧新值)
 *   28     4   f32   yaw_small_omega             rad/s(每帧新值)
 *   32     4   f32   chassis_imu_yaw             0 ~ 2π; 来自 MCU2, **值可能被保持**
 *   36     4   f32   chassis_imu_omega           rad/s(底盘 yaw 角速度, 同上同批)
 *   40     1   u8    mark                        递增标志位
 *   41     1   u8    color                       颜色
 *   42     1   u8    auto_aim_switch             电控自瞄开关
 *   43     1   u8    yaw_big_temperature         大 yaw 电机温度
 *   44     1   u8    yaw_small_temperature       小 yaw 电机温度
 *   45     1   u8    mcu2_seq                    MCU2 新样本序号; **值保持时不变**
 *   46     1   u8    crc8                        CRC8(byte 0..45), 初值 0xFF
 *   ---- 47 字节 (payload = 偏移 4..45 = 42 字节) ----
 *
 *   协议核对结论（已用工具链实测）:
 *     sizeof(mcu::SendPacket)    = 41 = 3+1+36+1  ✔（未变）
 *     sizeof(mcu::ReceivePacket) = 47 = 3+1+42+1  ✔（本次由 48 改为 47）
 *     （注意: 42 是 payload 长度; 整帧 = 3 前导 + 1 data_size + 42 payload + 1 CRC = 47。
 *       只发 46 字节会漏掉最后的 CRC，上位机将永远解析不出帧。）
 *   本文件用同样的 #pragma pack(1) 复刻这两个布局, 并用 _Static_assert 逐字段核对
 *   偏移/大小 —— 只要 Protocol.hpp 被改动, 这里会编译期报错而不是静默错位。
 *   帧内**不再有任何时间字段**（MCU 端计时不可用）。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 三之二、单个"MCU2 新样本序号"的语义（**协议修订的核心, 实现时必须严格遵守**）
 *
 *   1) `mcu2_seq` 是**新样本计数器**（不是时间戳）:
 *      MCU1 每**真正从 MCU2 取到一次新数据**就 +1
 *      （即 mcu2_get_yaw_big_sample() 或 mcu2_get_chassis_imu_sample() 返回"有新数据"）。
 *      **这是上位机区分"新样本"与"被保持的陈旧值"的依据之一。**
 *   2) 大 yaw 与底盘 IMU **同源且同批**: MCU2 一次把两路一起送来, 一次取数里两路一起刷新,
 *      所以共用一个序号 —— 同一次取数里**只 +1 一次**（不要两路各加一次）。
 *   3) 值被保持的那些帧里，**序号与值都原样重复**，绝不递增。
 *      （上位机判定: 首帧到达 或 序号变化 或 值变化 ⇒ 新样本;
 *        即使新样本的值与上一帧恰好相同, 序号也会变。）
 *   4) 序号用 uint8，自然回绕即可（上位机只判断"是否与上次不同"）。
 *   5) **不需要"0 = 从未收到"的哨兵约定**: 不做可用性检测, 上电初值 0 即可,
 *      上位机把收到的第一帧就当一次更新（三者取或的规则里"首帧到达"已覆盖）。
 *   6) 看门狗超时、未使能等路径**只冻结力矩输出**, 反馈与序号**照常更新**
 *      （看门狗管"不输出力矩", 不管"不读反馈"; 详见 dual_yaw_control_step() 步骤 ③）。
 *      好处: 恢复控制时锚点是新鲜的; 且链路若真断了序号自然不递增、年龄照样增长。
 *   7) 小 yaw / pitch 每帧都是新样本，**不带**序号（照旧直接刷新）。
 *
 *   实现见 `yaw_feedback_update()`: 取完两路后, 只要有任意一路报告"有新数据"就 `序号++` 一次;
 *   两路都没新数据时**值与序号都原样保持**。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 四、可调宏一览（全部集中在下面第 1 节, 调参只改这一处）
 *
 *   [协议常量]   YAW_PREAMBLE_1/2, YAW_PROTOCOL_VERSION,
 *                PC_TO_MCU_PAYLOAD_SIZE=36, PC_TO_MCU_FRAME_SIZE=41,
 *                MCU_TO_PC_PAYLOAD_SIZE=42, MCU_TO_PC_FRAME_SIZE=47,
 *                YAW_CRC8_INIT_VALUE=0xFF
 *   [系统结构]   YAW_BIG_SOURCE_FROM_MCU2 = 1
 *                （1 = 大 yaw 反馈来自 MCU2、指令转发 MCU2，本文件的默认；
 *                  0 = 单 MCU 直接读大 yaw 编码器 + 本地 CAN 下发）
 *   [内环增益]   YAW_BIG_KP, YAW_BIG_KD, YAW_SMALL_KP, YAW_SMALL_KD
 *                （大 yaw 由 MCU2 驱动时, 其 KP/KD 与力矩换算在 MCU2 的工程里）
 *   [力矩换算]   YAW_BIG_KT_NM_PER_A / YAW_BIG_GEAR_RATIO / YAW_BIG_CURRENT_FS_A
 *                → YAW_BIG_TORQUE_TO_CMD_SCALE
 *                YAW_SMALL_* 同上（两电机力矩常数/减速比不同, 必须分别标定）
 *                YAW_TORQUE_CMD_FULL_SCALE = 16384（最终限幅）
 *   [编码器换算] YAW_BIG_ENCODER_COUNTS_PER_TURN=8192, YAW_BIG_RPM_TO_RAD_S
 *                YAW_SMALL_RAD_PER_COUNT, YAW_SMALL_ENCODER_ZERO_COUNTS,
 *                YAW_SMALL_RPM_TO_RAD_S, YAW_SMALL_ANGLE_WRAP_TO_PM_PI
 *   [小 yaw 限位] YAW_SMALL_MIN_RAD=−25°, YAW_SMALL_MAX_RAD=+20°（**非对称**硬限位）,
 *                YAW_SMALL_SOFT_MARGIN_RAD=2°
 *                → 目标角夹取 [YAW_SMALL_TARGET_MIN_RAD, YAW_SMALL_TARGET_MAX_RAD] = [−23°, +18°],
 *                YAW_SMALL_DECEL_ZONE_LEN_RAD=10°（距任一侧限位 10° 开始降速 ⇒ [−15°, +10°]）,
 *                YAW_SMALL_RATE_MAX_RAD_S, YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S,
 *                YAW_SMALL_HARD_CENTER_TORQUE_NM (默认 0 = 硬限位处零力矩)
 *   [看门狗]     YAW_WATCHDOG_TIMEOUT_MS = 50
 *   [其它]       YAW_FEEDBACK_TX_PERIOD_MS, YAW_AUTO_AIM_GATING_ENABLE,
 *                YAW_RX_BUFFER_SIZE
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 五、调用方式（在 MCU 主循环 / 中断中的接线）
 *
 *   1) 上电:  dual_yaw_control_init();
 *   2) 串口每收到 1 字节:  dual_yaw_rx_byte(b);      // 可放在 USART 中断里
 *      （或整块: dual_yaw_rx_bytes(buf, len);）
 *   3) MCU1↔MCU2 内部链路收到数据时: 由你的链路驱动缓存/置标志即可,
 *      本文件在 1kHz 任务里通过 mcu2_get_yaw_big_sample() /
 *      mcu2_get_chassis_imu_sample() 询问"有没有新样本"（见第 5.6 节桩函数）。
 *   4) 定时器 1kHz 周期任务: dual_yaw_control_step(now_ms);  // now_ms = HAL_GetTick()
 *
 *   dual_yaw_control_step() 内部完成: 消费最新合法帧 → 刷新 MCU2 数据(值 + 新样本序号,
 *   在故障判断之前) → 看门狗判定（只冻力矩）→ 两关节控制（含小 yaw 安全层）→ 下发
 *   → 按周期回传反馈帧（47B, 含 mcu2_seq）。
 * ==========================================================================*/

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * 1. 可调宏（集中配置区）
 * ==========================================================================*/

/* ---- 1.1 协议常量（必须与 Protocol.hpp 完全一致, 不要随手机改） ---- */
#define YAW_PREAMBLE_1              0x42u
#define YAW_PREAMBLE_2              0x52u
#define YAW_PROTOCOL_VERSION        0x03u

#define PC_TO_MCU_PREAMBLE_SIZE     3u      /* 前导字节数 */
#define PC_TO_MCU_PAYLOAD_SIZE      36u     /* 上位机 SendPacket.data_size */
#define PC_TO_MCU_FRAME_SIZE        41u     /* 3 + 1 + 36 + 1 */

#define MCU_TO_PC_PREAMBLE_SIZE     3u
#define MCU_TO_PC_PAYLOAD_SIZE      42u     /* 本机 ReceivePacket.data_size（v0x03 修订: 43 → 42） */
#define MCU_TO_PC_FRAME_SIZE        47u     /* 3 + 1 + 42 + 1（v0x03 修订: 48 → 47） */

#define YAW_CRC8_INIT_VALUE         0xFFu   /* 与 CRC8_Check_Sum() 的初值一致 */

/* MCU2 新样本序号: 初值 0, 每**真正取到一次** MCU2 新数据就 +1（uint8 自然回绕）。
 * 不需要"0 = 从未收到"的哨兵约定（上位机按"首帧到达 或 序号变化 或 值变化"判定）。
 * 帧里没有任何时间字段。 */
#define YAW_MCU2_SEQ_INIT           0u

/* ---- 系统结构: 大 yaw 的反馈与控制走哪条路 ----
 *   1 = 真实系统结构: MCU1 只负责 pitch + 小 yaw; 大 yaw 编码器/电机与底盘 IMU 在 MCU2 上。
 *       大 yaw 的**反馈值**经内部链路从 MCU2 取（约 10Hz、值会被保持, 因此带**新样本序号**）,
 *       大 yaw 的**控制指令**原样转发给 MCU2（由 MCU2 用自己实时的编码器跑内环）。
 *   0 = 单 MCU 方案（上一版示例的做法）: 本机直接读大 yaw 编码器、本地做多圈累计与内环、
 *       本地 CAN 下发; 此时大 yaw 每个反馈周期都是新样本, 序号随之 +1。
 * 可用 -DYAW_BIG_SOURCE_FROM_MCU2=0 从编译命令覆盖，便于两种接法共存。 */
#ifndef YAW_BIG_SOURCE_FROM_MCU2
#define YAW_BIG_SOURCE_FROM_MCU2    1
#endif

/* yaw 控制模式位（与 mcu::YawMode 一致）
 * 注意: 与旧版单轴示例的 yaw_torque_only_mode 语义相反 —— 旧版 1 = 仅力矩,
 *       新版协议规定 0 = 仅力矩, 1 = 力矩 + 位置/速度内环。 */
#define YAW_MODE_TORQUE_ONLY        0u
#define YAW_MODE_TORQUE_PLUS_PID    1u

/* ---- 1.2 内环增益（每关节独立, 需调参） ---- */
#define YAW_BIG_KP                  0.10f   /* N·m/rad   */
#define YAW_BIG_KD                  0.10f   /* N·m·s/rad */
#define YAW_SMALL_KP                0.05f   /* N·m/rad   */
#define YAW_SMALL_KD                0.02f   /* N·m·s/rad */

/* ---- 1.3 力矩 → 电流/力矩指令 换算系数（每关节独立） ----
 * 协议里的 yaw_*_torque 单位是 N·m（关节侧），电控需要换算成电机电流/力矩指令:
 *     τ_fs[N·m] = Kt[N·m/A] · G(减速比) · I_fs[A]
 *     scale     = 1 / τ_fs      [1/(N·m)]
 * 两电机力矩常数/减速比不同, 必须分别标定; 若电调直接收 N·m 归一化指令,
 * 则把 Kt/G/I_fs 设成让 scale = 1/τ_fs 即可。 */
#define YAW_BIG_KT_NM_PER_A        0.32f    /* 大 yaw 电机力矩常数 */
#define YAW_BIG_GEAR_RATIO         1.0f     /* 大 yaw 减速比(电机→关节) */
#define YAW_BIG_CURRENT_FS_A       12.0f    /* 大 yaw 满量程电流 */
#define YAW_BIG_TORQUE_TO_CMD_SCALE \
    (1.0f / (YAW_BIG_KT_NM_PER_A * YAW_BIG_GEAR_RATIO * YAW_BIG_CURRENT_FS_A))

#define YAW_SMALL_KT_NM_PER_A      0.10f    /* 小 yaw 电机力矩常数 */
#define YAW_SMALL_GEAR_RATIO       1.0f     /* 小 yaw 减速比(电机→关节) */
#define YAW_SMALL_CURRENT_FS_A     5.0f     /* 小 yaw 满量程电流 */
#define YAW_SMALL_TORQUE_TO_CMD_SCALE \
    (1.0f / (YAW_SMALL_KT_NM_PER_A * YAW_SMALL_GEAR_RATIO * YAW_SMALL_CURRENT_FS_A))

#define YAW_TORQUE_CMD_NORM_LIMIT   1.0f    /* 归一化限幅 ±1.0（沿用旧示例风格） */
#define YAW_TORQUE_CMD_FULL_SCALE   16384.0f/* 最终指令限幅 ±16384 */

/* ---- 1.4 编码器 / 速度换算 ---- */
#define YAW_PI                      3.14159265358979323846
#define YAW_TWO_PI                  (2.0 * YAW_PI)
#define YAW_DEG2RAD                 (YAW_PI / 180.0)
#define YAW_RPM2RADS                (YAW_PI / 30.0)

/* 大 yaw: 单圈 0~8191 计数, 多圈由软件累计（见 yaw_big_update_multiturn） */
#define YAW_BIG_ENCODER_COUNTS_PER_TURN 8192.0f
/* 电机速度报文 rpm → 关节角速度 rad/s（含减速比） */
#define YAW_BIG_RPM_TO_RAD_S        ((float)(YAW_RPM2RADS / YAW_BIG_GEAR_RATIO))

/* 小 yaw: 编码器给出该关节的绝对/相对计数
 *   counts → 关节角(rad) = (counts - ZERO) * RAD_PER_COUNT
 * 若为单圈 8192 计数 + 减速比 G: RAD_PER_COUNT = 2π/(8192·G) */
#define YAW_SMALL_ENCODER_ZERO_COUNTS   0
#define YAW_SMALL_RAD_PER_COUNT \
    ((float)(YAW_TWO_PI / ((double)YAW_BIG_ENCODER_COUNTS_PER_TURN * (double)YAW_SMALL_GEAR_RATIO)))
/* 0 = 计数直读（绝对/相对计数语义, 默认）; 1 = 单圈编码器绕到 ±π */
#define YAW_SMALL_ANGLE_WRAP_TO_PM_PI   0
/* 小 yaw 电机速度报文 rpm → 关节角速度 rad/s（含减速比） */
#define YAW_SMALL_RPM_TO_RAD_S      ((float)(YAW_RPM2RADS / YAW_SMALL_GEAR_RATIO))

/* ---- 1.5 小 yaw 安全限位（**非对称行程 −25° ~ +20°**；最后一道安全防线, 宁可误触发不可漏触发）
 *
 * 机械行程**两侧不对称**: 负侧只能到 −25°, 正侧只能到 +20°（0 不是行程中心, 中心是 −2.5°）。
 * 因此这里**两个方向各用独立的宏**, 严禁写成 `±LIMIT` / `fabsf(angle) > LIMIT` 这种
 * 对称写法 —— 那会让负侧按正侧的余量算, 提前 5° 就限速/夹目标角。
 *
 *   硬限位   : [YAW_SMALL_MIN_RAD, YAW_SMALL_MAX_RAD] = [−25°, +20°]
 *   目标夹取 : [YAW_SMALL_TARGET_MIN_RAD, YAW_SMALL_TARGET_MAX_RAD] = [−23°, +18°]（各留 2°）
 *   减速区   : 距**任一侧**硬限位 10° 开始降速 ⇒ [−15°, +10°] 之外开始降速
 */
#define YAW_SMALL_MIN_RAD               ((float)(-25.0 * YAW_DEG2RAD)) /* 机械硬限位（负侧）−25° */
#define YAW_SMALL_MAX_RAD               ((float)( 20.0 * YAW_DEG2RAD)) /* 机械硬限位（正侧）+20° */
#define YAW_SMALL_SOFT_MARGIN_RAD       ((float)(2.0 * YAW_DEG2RAD))   /* 目标角夹取余量 2°（两侧各自留） */
#define YAW_SMALL_TARGET_MIN_RAD \
    (YAW_SMALL_MIN_RAD + YAW_SMALL_SOFT_MARGIN_RAD)                    /* = −23° 目标角下限 */
#define YAW_SMALL_TARGET_MAX_RAD \
    (YAW_SMALL_MAX_RAD - YAW_SMALL_SOFT_MARGIN_RAD)                    /* = +18° 目标角上限 */
/* 减速区长度（距硬限位的距离）: 10° ⇒ 负侧从 −15° 起、正侧从 +10° 起开始降速 */
#define YAW_SMALL_DECEL_ZONE_LEN_RAD    ((float)(10.0 * YAW_DEG2RAD))
#define YAW_SMALL_DECEL_START_MIN_RAD \
    (YAW_SMALL_MIN_RAD + YAW_SMALL_DECEL_ZONE_LEN_RAD)                 /* = −15° */
#define YAW_SMALL_DECEL_START_MAX_RAD \
    (YAW_SMALL_MAX_RAD - YAW_SMALL_DECEL_ZONE_LEN_RAD)                 /* = +10° */
#define YAW_SMALL_RATE_MAX_RAD_S        6.0f    /* 远离限位时允许的 |ω| */
#define YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S 0.2f /* 贴到硬限位时允许的 |ω| */
/* 超过硬限位时施加的"回中"力矩幅值: 0 = 直接零力矩（默认, 最保守）。
 * 若设为 >0, 则正侧越限施 −该幅值、负侧越限施 +该幅值（把关节拉回行程内）。 */
#define YAW_SMALL_HARD_CENTER_TORQUE_NM 0.0f

/* ---- 1.6 看门狗 ----
 * 超过该时间未收到**合法**(前导+长度+CRC 全通过)的上位机帧, 两关节力矩清零。
 * 这是必需的: 上位机(MPC/自瞄)死机、串口掉线或线缆松脱时, 若继续执行最后一帧
 * 的力矩指令, 云台会持续朝一个方向加速(典型的"满舵打转"), 极易损坏机械与
 * 伤及人员; 因此宁可失去控制权也绝不允许保持上一帧力矩。 */
#define YAW_WATCHDOG_TIMEOUT_MS     50u

/* ---- 1.7 其它 ---- */
#define YAW_FEEDBACK_TX_PERIOD_MS   5u      /* 反馈帧发送周期（200Hz） */
#define YAW_RX_BUFFER_SIZE          64u     /* 单帧 41B, 留余量 */
#define YAW_AUTO_AIM_GATING_ENABLE  1u      /* 1 = 自瞄总开关/电控开关未开时力矩清零 */

/* ---- 1.8 接收结果码（调试用, 可挂到示波器/日志） ---- */
#define YAW_PARSE_OK                0u
#define YAW_PARSE_ERR_LENGTH        1u      /* 帧长不足 */
#define YAW_PARSE_ERR_PREAMBLE      2u      /* 前导错 */
#define YAW_PARSE_ERR_DATA_SIZE     3u      /* data_size != 36 */
#define YAW_PARSE_ERR_CRC           4u      /* CRC8 错 */
#define YAW_PARSE_ERR_NOT_FINITE    5u      /* 出现 NaN/Inf, 拒绝该帧 */

/* ============================================================================
 * 2. 线格式（wire）结构体 + 编译期布局核对
 *    只用它们来取 offsetof / sizeof, 不做成员直接访问 —— 避免在某些内核/编译
 *    配置下对 packed 成员产生非对齐访问。所有读写都走 memcpy(见第 4 节)。
 * ==========================================================================*/
#pragma pack(push, 1)

/* 与 mcu::SendPacket 逐字段对应（PC → MCU） */
typedef struct {
    uint8_t frame_header1;
    uint8_t frame_header2;
    uint8_t protocol_version;
    uint8_t data_size;
    uint8_t auto_aim_enable;
    uint8_t fire;
    float   pitch_target_angle;
    uint8_t yaw_big_mode;
    double  yaw_big_target_angle;
    float   yaw_big_target_velocity;
    float   yaw_big_torque;
    uint8_t yaw_small_mode;
    float   yaw_small_target_angle;
    float   yaw_small_target_velocity;
    float   yaw_small_torque;
    uint8_t crc8;
} pc_to_mcu_wire_t;

/* 与 mcu::ReceivePacket 逐字段对应（MCU → PC） */
typedef struct {
    uint8_t frame_header1;
    uint8_t frame_header2;
    uint8_t protocol_version;
    uint8_t data_size;
    float   bullet_velocity;
    float   pitch_angle;
    double  yaw_big_angle;
    float   yaw_big_omega;
    float   yaw_small_angle;
    float   yaw_small_omega;
    float   chassis_imu_yaw;
    float   chassis_imu_omega;
    uint8_t mark;
    uint8_t color;
    uint8_t auto_aim_switch;
    uint8_t yaw_big_temperature;
    uint8_t yaw_small_temperature;
    uint8_t mcu2_seq;                   /* 45: MCU2 新样本序号(值保持时不变) */
    uint8_t crc8;
} mcu_to_pc_wire_t;

#pragma pack(pop)

/* ── 整帧大小核对（对应 Protocol.hpp 里的 static_assert） ── */
_Static_assert(sizeof(pc_to_mcu_wire_t) == PC_TO_MCU_FRAME_SIZE,
               "PC->MCU 帧长必须为 41 (= 3+1+36+1), 与 mcu::SendPacket 不一致");
_Static_assert(sizeof(mcu_to_pc_wire_t) == MCU_TO_PC_FRAME_SIZE,
               "MCU->PC 帧长必须为 47 (= 3+1+42+1), 与 mcu::ReceivePacket 不一致");
/* 帧内不含任何时间字段: payload 里最后一个多字节字段是 chassis_imu_omega(偏移 36) */
_Static_assert(offsetof(mcu_to_pc_wire_t, mcu2_seq) == 45, "mcu2_seq 偏移必须为 45");
_Static_assert(offsetof(mcu_to_pc_wire_t, crc8) == 46, "MCU->PC crc8 偏移必须为 46");

/* ── 关键字段偏移核对（数值来自 Protocol.hpp 实测布局） ── */
_Static_assert(offsetof(pc_to_mcu_wire_t, data_size) == 3, "data_size 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, auto_aim_enable) == 4, "auto_aim_enable 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, fire) == 5, "fire 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, pitch_target_angle) == 6, "pitch_target_angle 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_big_mode) == 10, "yaw_big_mode 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_big_target_angle) == 11, "yaw_big_target_angle 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_big_target_velocity) == 19, "yaw_big_target_velocity 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_big_torque) == 23, "yaw_big_torque 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_small_mode) == 27, "yaw_small_mode 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_small_target_angle) == 28, "yaw_small_target_angle 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_small_target_velocity) == 32, "yaw_small_target_velocity 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, yaw_small_torque) == 36, "yaw_small_torque 偏移错");
_Static_assert(offsetof(pc_to_mcu_wire_t, crc8) == 40, "PC->MCU crc8 偏移必须为 40");

_Static_assert(offsetof(mcu_to_pc_wire_t, data_size) == 3, "data_size 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, bullet_velocity) == 4, "bullet_velocity 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, pitch_angle) == 8, "pitch_angle 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, yaw_big_angle) == 12, "yaw_big_angle 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, yaw_big_omega) == 20, "yaw_big_omega 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, yaw_small_angle) == 24, "yaw_small_angle 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, yaw_small_omega) == 28, "yaw_small_omega 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, chassis_imu_yaw) == 32, "chassis_imu_yaw 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, chassis_imu_omega) == 36, "chassis_imu_omega 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, mark) == 40, "mark 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, color) == 41, "color 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, auto_aim_switch) == 42, "auto_aim_switch 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, yaw_big_temperature) == 43, "yaw_big_temperature 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, yaw_small_temperature) == 44, "yaw_small_temperature 偏移错");
_Static_assert(offsetof(mcu_to_pc_wire_t, crc8) == (MCU_TO_PC_FRAME_SIZE - 1u),
               "crc8 必须正好是整帧的最后一个字节");

/* ============================================================================
 * 3. CRC8（与 src/communication/CRC.cpp 的 CRC8_Check_Sum 逐字节一致）
 *    算法: 查表法, 初值 0xFF, 无输出异或, 每次 index = crc ^ byte
 * ==========================================================================*/
static const uint8_t YAW_CRC8_TAB[256] = {
    0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
    0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
    0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
    0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
    0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
    0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
    0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
    0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
    0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
    0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
    0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
    0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
    0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
    0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
    0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
    0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35
};

/* CRC8: data 为整帧起始, len = 帧长 - 1（不含 CRC 字节本身） */
static uint8_t crc8_calc(const uint8_t *data, size_t len)
{
    uint8_t crc = (uint8_t)YAW_CRC8_INIT_VALUE;
    size_t i;
    if (data == NULL) {
        return (uint8_t)YAW_CRC8_INIT_VALUE;
    }
    for (i = 0; i < len; ++i) {
        crc = YAW_CRC8_TAB[(uint8_t)(crc ^ data[i])];
    }
    return crc;
}

/* ============================================================================
 * 4. 解码 / 编码辅助（小端 + memcpy, 规避 packed 结构体的非对齐访问）
 * ==========================================================================*/
static float get_f32_le(const uint8_t *p)
{
    float v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static double get_f64_le(const uint8_t *p)
{
    double v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static void put_f32_le(uint8_t *p, float v)
{
    memcpy(p, &v, sizeof(v));
}

static void put_f64_le(uint8_t *p, double v)
{
    memcpy(p, &v, sizeof(v));
}

/* 解包后的上位机指令（PC → MCU） */
typedef struct {
    uint8_t auto_aim_enable;            /* 自瞄总开关 */
    uint8_t fire;                       /* 火控（本示例只解析, 发射逻辑由火控模块实现） */
    float   pitch_target_angle;         /* 原始语义, 电控不做映射 */
    uint8_t yaw_big_mode;               /* 0=仅力矩 1=力矩+PID */
    double  yaw_big_target_angle;       /* rad, 多圈连续 */
    float   yaw_big_target_velocity;    /* rad/s */
    float   yaw_big_torque;             /* N·m */
    uint8_t yaw_small_mode;             /* 0=仅力矩 1=力矩+PID */
    float   yaw_small_target_angle;     /* rad, 相对角 */
    float   yaw_small_target_velocity;  /* rad/s */
    float   yaw_small_torque;           /* N·m */
} yaw_cmd_t;

/* 待组包的反馈（MCU → PC） */
typedef struct {
    float    bullet_velocity;
    float    pitch_angle;
    double   yaw_big_angle;             /* rad, 多圈连续 */
    float    yaw_big_omega;             /* rad/s */
    float    yaw_small_angle;           /* rad */
    float    yaw_small_omega;           /* rad/s */
    float    chassis_imu_yaw;
    float    chassis_imu_omega;
    uint8_t  mark;
    uint8_t  color;
    uint8_t  auto_aim_switch;
    uint8_t  yaw_big_temperature;
    uint8_t  yaw_small_temperature;
    uint8_t  mcu2_seq;                  /* MCU2 新样本序号（值保持时不变） */
} yaw_feedback_t;

/* ---------------------------------------------------------------------------
 * parse_packet(): 解析上位机发来的整帧（PC → MCU）
 *   frame : 指向帧首（偏移 0 = 0x42）
 *   len   : 至少 PC_TO_MCU_FRAME_SIZE(41)
 *   out   : 解析结果
 * 返回 YAW_PARSE_OK 或 YAW_PARSE_ERR_*
 * 校验: 帧长 → 前导 0x42 0x52 0x03 → data_size == 36 → CRC8 → 数值有限性
 * ------------------------------------------------------------------------- */
uint8_t parse_packet(const uint8_t *frame, uint32_t len, yaw_cmd_t *out)
{
    uint8_t crc_rx;
    uint8_t crc_calc;

    if ((frame == NULL) || (out == NULL) || (len < PC_TO_MCU_FRAME_SIZE)) {
        return YAW_PARSE_ERR_LENGTH;
    }
    /* ① 前导（3 字节）+ 版本 */
    if ((frame[0] != (uint8_t)YAW_PREAMBLE_1) ||
        (frame[1] != (uint8_t)YAW_PREAMBLE_2) ||
        (frame[2] != (uint8_t)YAW_PROTOCOL_VERSION)) {
        return YAW_PARSE_ERR_PREAMBLE;
    }
    /* ② 长度字段必须为 36；同时可用来拒绝本机回传帧(data_size=53)的串口回环 */
    if (frame[offsetof(pc_to_mcu_wire_t, data_size)] != (uint8_t)PC_TO_MCU_PAYLOAD_SIZE) {
        return YAW_PARSE_ERR_DATA_SIZE;
    }
    /* ③ CRC8: 覆盖 byte 0 .. 39（frame_size - 1） */
    crc_rx   = frame[offsetof(pc_to_mcu_wire_t, crc8)];
    crc_calc = crc8_calc(frame, (size_t)(PC_TO_MCU_FRAME_SIZE - 1u));
    if (crc_rx != crc_calc) {
        return YAW_PARSE_ERR_CRC;
    }

    /* ④ 逐字段解析（偏移取自 packed 布局, double 用 memcpy 拷贝） */
    out->auto_aim_enable          = frame[offsetof(pc_to_mcu_wire_t, auto_aim_enable)];
    out->fire                     = frame[offsetof(pc_to_mcu_wire_t, fire)];
    out->pitch_target_angle       = get_f32_le(frame + offsetof(pc_to_mcu_wire_t, pitch_target_angle));
    out->yaw_big_mode             = frame[offsetof(pc_to_mcu_wire_t, yaw_big_mode)];
    out->yaw_big_target_angle     = get_f64_le(frame + offsetof(pc_to_mcu_wire_t, yaw_big_target_angle));
    out->yaw_big_target_velocity  = get_f32_le(frame + offsetof(pc_to_mcu_wire_t, yaw_big_target_velocity));
    out->yaw_big_torque           = get_f32_le(frame + offsetof(pc_to_mcu_wire_t, yaw_big_torque));
    out->yaw_small_mode           = frame[offsetof(pc_to_mcu_wire_t, yaw_small_mode)];
    out->yaw_small_target_angle   = get_f32_le(frame + offsetof(pc_to_mcu_wire_t, yaw_small_target_angle));
    out->yaw_small_target_velocity= get_f32_le(frame + offsetof(pc_to_mcu_wire_t, yaw_small_target_velocity));
    out->yaw_small_torque         = get_f32_le(frame + offsetof(pc_to_mcu_wire_t, yaw_small_torque));

    /* ⑤ 数值健全性: 带 NaN/Inf 的帧一律拒绝（否则会算出 NaN 力矩直接下发） */
    if (!isfinite((double)out->pitch_target_angle) ||
        !isfinite(out->yaw_big_target_angle) ||
        !isfinite((double)out->yaw_big_target_velocity) ||
        !isfinite((double)out->yaw_big_torque) ||
        !isfinite((double)out->yaw_small_target_angle) ||
        !isfinite((double)out->yaw_small_target_velocity) ||
        !isfinite((double)out->yaw_small_torque)) {
        return YAW_PARSE_ERR_NOT_FINITE;
    }

    /* ⑥ 模式位只认 0/1, 其它值按"仅力矩"处理（保守） */
    if (out->yaw_big_mode != (uint8_t)YAW_MODE_TORQUE_PLUS_PID) {
        out->yaw_big_mode = (uint8_t)YAW_MODE_TORQUE_ONLY;
    }
    if (out->yaw_small_mode != (uint8_t)YAW_MODE_TORQUE_PLUS_PID) {
        out->yaw_small_mode = (uint8_t)YAW_MODE_TORQUE_ONLY;
    }
    return YAW_PARSE_OK;
}

/* ---------------------------------------------------------------------------
 * build_packet(): 组包回传（MCU → PC）
 *   fb      : 反馈内容
 *   out     : 输出缓冲, 至少 MCU_TO_PC_FRAME_SIZE(47) 字节
 *   out_cap : 缓冲容量
 * 返回写入的字节数（正常 47）, 容量不足返回 0
 * ------------------------------------------------------------------------- */
uint32_t build_packet(const yaw_feedback_t *fb, uint8_t *out, uint32_t out_cap)
{
    if ((fb == NULL) || (out == NULL) || (out_cap < MCU_TO_PC_FRAME_SIZE)) {
        return 0u;
    }
    memset(out, 0, (size_t)MCU_TO_PC_FRAME_SIZE);

    /* 前导 + 版本 + data_size */
    out[offsetof(mcu_to_pc_wire_t, frame_header1)]    = (uint8_t)YAW_PREAMBLE_1;
    out[offsetof(mcu_to_pc_wire_t, frame_header2)]    = (uint8_t)YAW_PREAMBLE_2;
    out[offsetof(mcu_to_pc_wire_t, protocol_version)] = (uint8_t)YAW_PROTOCOL_VERSION;
    out[offsetof(mcu_to_pc_wire_t, data_size)]        = (uint8_t)MCU_TO_PC_PAYLOAD_SIZE; /* 42 */

    /* 载荷（偏移严格按 packed 布局） */
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, bullet_velocity),     fb->bullet_velocity);
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, pitch_angle),         fb->pitch_angle);
    put_f64_le(out + offsetof(mcu_to_pc_wire_t, yaw_big_angle),       fb->yaw_big_angle);   /* double */
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, yaw_big_omega),       fb->yaw_big_omega);
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, yaw_small_angle),     fb->yaw_small_angle);
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, yaw_small_omega),     fb->yaw_small_omega);
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, chassis_imu_yaw),     fb->chassis_imu_yaw);
    put_f32_le(out + offsetof(mcu_to_pc_wire_t, chassis_imu_omega),   fb->chassis_imu_omega);
    out[offsetof(mcu_to_pc_wire_t, mark)]                 = fb->mark;
    out[offsetof(mcu_to_pc_wire_t, color)]                = fb->color;
    out[offsetof(mcu_to_pc_wire_t, auto_aim_switch)]      = fb->auto_aim_switch;
    out[offsetof(mcu_to_pc_wire_t, yaw_big_temperature)]  = fb->yaw_big_temperature;
    out[offsetof(mcu_to_pc_wire_t, yaw_small_temperature)]= fb->yaw_small_temperature;
    /* MCU2 新样本序号: 由 yaw_feedback_update() 维护, 值被保持时**原样透传**（不递增） */
    out[offsetof(mcu_to_pc_wire_t, mcu2_seq)]             = fb->mcu2_seq;

    /* CRC8: 覆盖 byte 0 .. 45（整帧去掉最后的 CRC 字节） */
    out[offsetof(mcu_to_pc_wire_t, crc8)] =
        crc8_calc(out, (size_t)(MCU_TO_PC_FRAME_SIZE - 1u));

    return (uint32_t)MCU_TO_PC_FRAME_SIZE;
}

/* ============================================================================
 * 5. 用户桩函数（硬件相关, 全部需要按自己的工程填写）
 * ==========================================================================*/

/* ---- 5.1 编码器 / 速度反馈 ----
 * 注意: 在真实结构里(YAW_BIG_SOURCE_FROM_MCU2 = 1)大 yaw 的编码器挂在 MCU2 上,
 *       下面两个大 yaw 桩函数只在"单 MCU 方案"(= 0)下才会被调用。 */
uint16_t get_encoder_big_raw(void);             /* TODO: 用户填写 —— 大 yaw 单圈原始计数 0~8191 */
int16_t  get_encoder_big_velocity_rpm(void);    /* TODO: 用户填写 —— 大 yaw 速度报文 rpm */
int32_t  get_encoder_small_counts(void);        /* TODO: 用户填写 —— 小 yaw 该关节绝对/相对计数 */
int16_t  get_encoder_small_velocity_rpm(void);  /* TODO: 用户填写 —— 小 yaw 速度报文 rpm */

/* ---- 5.2 力矩下发（CAN） ----
 * 同上: can_send_torque_big() 只在"单 MCU 方案"下被调用;
 *       真实结构下大 yaw 力矩由 MCU2 下发（见 5.6 的转发桩函数）。 */
void can_send_torque_big(int16_t cmd);          /* TODO: 用户填写 —— 大 yaw ±16384 力矩/电流指令 */
void can_send_torque_small(int16_t cmd);        /* TODO: 用户填写 —— 小 yaw ±16384 力矩/电流指令 */

/* ---- 5.3 串口发送 ---- */
void uart_send_bytes(const uint8_t *data, uint32_t len); /* TODO: 用户填写 —— 回传反馈帧 */

/* ---- 5.4 反馈所需其它量 ---- */
float   get_pitch_angle_raw(void);              /* TODO: 用户填写 —— pitch 关节原始角(电控不做映射) */
float   get_bullet_velocity(void);              /* TODO: 用户填写 —— 弹速 m/s */
float   get_chassis_imu_yaw(void);              /* TODO: 用户填写 —— 底盘 IMU yaw, 0~2π */
float   get_chassis_imu_omega(void);            /* TODO: 用户填写 —— 底盘 yaw 角速度 rad/s */
uint8_t get_mark_color(void);                   /* TODO: 用户填写 —— 颜色 */
uint8_t get_auto_aim_switch(void);              /* TODO: 用户填写 —— 电控自瞄拨杆/开关 */
uint8_t get_motor_temperature_big(void);        /* TODO: 用户填写 —— 大 yaw 电机温度 */
uint8_t get_motor_temperature_small(void);      /* TODO: 用户填写 —— 小 yaw 电机温度 */

/* ---- 5.5 临界区（接收中断与主循环交换命令缓冲时使用） ---- */
void yaw_enter_critical(void);                  /* TODO: 用户填写 —— 关中断/加锁 */
void yaw_exit_critical(void);                   /* TODO: 用户填写 —— 开中断/解锁 */

/* ---- 5.6 MCU1 ↔ MCU2 内部链路（**协议修订的关键接口**） ----
 * 真实结构下(默认 YAW_BIG_SOURCE_FROM_MCU2 = 1):
 *   大 yaw 反馈、底盘 IMU 都由 MCU2 采集, 经内部链路送到 MCU1（约 10Hz、间隔不规则）。
 *   MCU1 在没收到新数据时会一直沿用旧值 —— 所以这两个"取新样本"函数必须**如实**反映
 *   "这次调用有没有拿到新数据":
 *     返回 1 = 本次拿到了 MCU2 的新样本（值有效, 已写入 *出参）
 *              → 调用方刷新值并把对应的**新样本序号 +1**
 *     返回 0 = 没有新数据（**输出参数不要改**; 调用方会继续保持旧值与旧序号）
 *
 *   典型实现: 链路驱动的接收回调里把最新值写进缓冲并置 new_flag / 递增"已收帧数",
 *             本函数检查该标志（取走并清标志）或比较计数, 从而区分"新值"与"同一旧值重复收到"。
 *   ⚠ 不要在这里"每次都返回 1": 那会让序号每帧都 +1, 上位机会把被保持的旧值
 *     当成每帧都刷新的新样本（延迟补偿/外推彻底失效）。
 *   ⚠ 链路是"同一帧被重复读到"也不能算新样本（只有真的收到新数据才算）。
 */
uint8_t mcu2_get_yaw_big_sample(double *angle_rad, float *omega_rad_s);
                                                /* TODO: 用户填写 —— 大 yaw 是否有新样本(含多圈连续角, double) */
uint8_t mcu2_get_chassis_imu_sample(float *yaw_rad, float *omega_rad_s);
                                                /* TODO: 用户填写 —— 底盘 IMU 是否有新样本 */

/* 把大 yaw 控制指令原样转发给 MCU2（由其用自己实时的大 yaw 编码器跑内环并下发 CAN）。
 * 传 0 力矩时即"大 yaw 力矩清零"（看门狗 / 未使能 / 上电初始化都会用到）。
 * 若你的内部链路直接透传上位机 41B 帧中的大 yaw 段, 也可以在这里打包转发。 */
void mcu2_send_yaw_big_command(uint8_t mode, double target_angle,
                               float target_velocity, float torque_nm);
                                                /* TODO: 用户填写 —— 大 yaw 指令转发(帧率跟随上位机即可) */

/* ============================================================================
 * 6. 状态量
 * ==========================================================================*/

/* ---- 6.1 大 yaw 多圈累计（沿用旧版单轴示例的做法） ----
 * 圈内角度用 double 而非 float 参与运算: 协议字段 yaw_big_angle 是 double,
 * 若中途落到 float, 6 rad 量级的角度会有 ~5e-7 rad 的舍入抖动（虽远小于
 * 2π/8192 的编码器量化步长, 但会让上位机看到虚假的高频抖动）。 */
static int32_t g_big_turns = 0;                      /* 累计圈数 */
static double  g_big_last_angle_in_one_turn = 0.0;   /* 上一周期的圈内角度 */
static uint8_t g_big_turns_inited = 0u;              /* 首次运行标志 */

/* ---- 6.2 接收（中断侧写 staging, 主循环侧取走） ---- */
static uint8_t  g_rx_buf[YAW_RX_BUFFER_SIZE];
static uint32_t g_rx_len = 0u;
static yaw_cmd_t g_rx_staging;                      /* 中断侧写入 */
static volatile uint8_t g_rx_staging_ready = 0u;    /* 1 = 有新的合法帧待取 */
static yaw_cmd_t g_cmd;                             /* 主循环侧使用的最新指令 */
static volatile uint8_t g_rx_err_last = YAW_PARSE_OK; /* 最近一次解析错误码(调试) */

/* ---- 6.3 运行状态 ---- */
static uint32_t g_last_valid_rx_ms = 0u;            /* 最近一次收到合法帧的时刻 */
static uint8_t  g_ever_valid_rx = 0u;               /* 是否收到过合法帧 */
static uint8_t  g_watchdog_fault = 1u;              /* 上电默认处于看门狗故障态（力矩为 0） */
static uint8_t  g_mark = 0u;                        /* 回传帧递增标志位 */
static uint32_t g_last_tx_ms = 0u;                  /* 上次回传时刻 */

/* ---- 6.4 "MCU2 数据"的反馈保持值 + **单个新样本序号** ----
 * 大 yaw 与底盘 IMU 都由 MCU2 采集, **同一次取数里两路一起刷新**（链路约 10Hz、间隔不规则）;
 * 没收到新数据时**值与序号一起保持不动**（序号不动才是上位机判断"这是被保持的旧值"的依据）。
 * `g_mcu2_seq` 只在真的取到 MCU2 新数据时 +1（同一次取数只加一次）;
 * 上电初值 0（无"0 = 从未收到"约定, 上位机首帧到达即视为一次更新）。
 * 帧里**没有任何时间字段**: 值的年龄由上位机用自己的时钟测量。 */
static double  g_yaw_big_fb_angle = 0.0;            /* 大 yaw 多圈连续角(来自 MCU2, 可能已变旧) */
static float   g_yaw_big_fb_omega = 0.0f;           /* 大 yaw 角速度(与下面底盘 IMU 同批) */
static float   g_chassis_fb_yaw   = 0.0f;           /* 底盘 IMU yaw(来自 MCU2, 可能已变旧) */
static float   g_chassis_fb_omega = 0.0f;           /* 底盘 yaw 角速度(与上面大 yaw 同批) */
static uint8_t g_mcu2_seq         = YAW_MCU2_SEQ_INIT;  /* MCU2 新样本序号(值保持时不变) */

/* ============================================================================
 * 7. 控制
 * ==========================================================================*/

/* 力矩[N·m] → 电机指令（±1.0 归一化 + ±16384 最终限幅, 沿用旧示例风格） */
static int16_t yaw_torque_to_cmd(float torque_nm, float torque_to_cmd_scale)
{
    float   norm;
    float   raw;
    int32_t cmd;

    if (!isfinite(torque_nm)) {
        return 0;                                   /* NaN/Inf 一律按零力矩处理 */
    }
    norm = torque_nm * torque_to_cmd_scale;
    if (norm > YAW_TORQUE_CMD_NORM_LIMIT) {         /* 第一次限幅: ±1.0 */
        norm = YAW_TORQUE_CMD_NORM_LIMIT;
    } else if (norm < -YAW_TORQUE_CMD_NORM_LIMIT) {
        norm = -YAW_TORQUE_CMD_NORM_LIMIT;
    }

    raw = YAW_TORQUE_CMD_FULL_SCALE * norm;
    cmd = (int32_t)raw;
    if (cmd > (int32_t)YAW_TORQUE_CMD_FULL_SCALE) {         /* 第二次限幅: ±16384 */
        cmd = (int32_t)YAW_TORQUE_CMD_FULL_SCALE;
    } else if (cmd < -(int32_t)YAW_TORQUE_CMD_FULL_SCALE) {
        cmd = -(int32_t)YAW_TORQUE_CMD_FULL_SCALE;
    }
    return (int16_t)cmd;
}

#if YAW_BIG_SOURCE_FROM_MCU2 == 0
/* ===========================================================================
 * 【仅单 MCU 方案】大 yaw 编码器在本机时才会用到的代码
 *   真实结构（默认）下大 yaw 编码器挂在 MCU2 上, 下面三个函数应当放在 **MCU2 的
 *   工程**里（MCU1 不需要, 也不该用经不稳定链路传来的旧角度做本地闭环）。
 *   这里保留作为参考实现: 若 MCU1 确实直连大 yaw 电机, 把
 *   YAW_BIG_SOURCE_FROM_MCU2 设为 0 即可启用。
 * =========================================================================== */

/* ---------------------------------------------------------------------------
 * 大 yaw: 编码器单圈计数 → 多圈连续角（圈数累计, 同旧版单轴示例）
 * ------------------------------------------------------------------------- */
static void yaw_big_update_multiturn(void)
{
    /* 圈内角度 0 ~ 2π */
    double angle_in_one_turn =
        (double)get_encoder_big_raw() * (YAW_TWO_PI / (double)YAW_BIG_ENCODER_COUNTS_PER_TURN);

    if (g_big_turns_inited) {
        /* 过零判据: 与"上一角度/上一角度±2π"三者中最近的那个"是跨越了圈边界" */
        double distance_to_next_turn = fabs(g_big_last_angle_in_one_turn - YAW_TWO_PI - angle_in_one_turn);
        double distance_to_this_turn = fabs(g_big_last_angle_in_one_turn - angle_in_one_turn);
        double distance_to_last_turn = fabs(g_big_last_angle_in_one_turn + YAW_TWO_PI - angle_in_one_turn);

        if ((distance_to_next_turn < distance_to_this_turn) &&
            (distance_to_next_turn < distance_to_last_turn)) {
            g_big_turns += 1;                       /* 正向跨圈 8191 → 0 */
        } else if ((distance_to_last_turn < distance_to_next_turn) &&
                   (distance_to_last_turn < distance_to_this_turn)) {
            g_big_turns -= 1;                       /* 反向跨圈 0 → 8191 */
        }
    } else {
        g_big_turns_inited = 1u;                    /* 首次上电: 以当前位置为零点 */
    }
    g_big_last_angle_in_one_turn = angle_in_one_turn;
}

/* 大 yaw 多圈连续角（rad） */
static double yaw_big_get_angle(void)
{
    double angle_in_one_turn =
        (double)get_encoder_big_raw() * (YAW_TWO_PI / (double)YAW_BIG_ENCODER_COUNTS_PER_TURN);
    return angle_in_one_turn + ((double)g_big_turns * YAW_TWO_PI);
}

/* 大 yaw 角速度（rad/s, 由速度报文 rpm 换算） */
static float yaw_big_get_omega(void)
{
    return (float)get_encoder_big_velocity_rpm() * YAW_BIG_RPM_TO_RAD_S;
}
#endif /* YAW_BIG_SOURCE_FROM_MCU2 == 0 */

/* ---------------------------------------------------------------------------
 * yaw_feedback_update(): 维护 MCU2 数据（大 yaw + 底盘 IMU）的保持值与新样本序号
 *
 * 语义（**本函数是协议修订的核心**, 上位机区分"新样本 vs 陈旧值"全靠它）:
 *   - 大 yaw 与底盘 IMU **同源同批**: MCU2 一次把两路一起送来, 所以共用一个序号;
 *     只要**任意一路**报告"有新数据", 取完两路后 `g_mcu2_seq++` **一次**
 *     （同一次取数里两路都刷新 → 只 +1 一次, 不要两路各加一次）;
 *   - 两路都没新数据时: **值与序号都原样保持**（上位机看到序号不变 → 知道这是被保持的
 *     旧值, 于是对姿态角只按已知角速度做一阶外推、对底盘角速度做零阶保持）;
 *   - 上电初值 0, **没有"0 = 从未收到"的哨兵约定**（上位机判定规则是
 *     "首帧到达 或 序号变化 或 值变化", 不做可用性检测）;
 *   - 序号用 uint8, 自然回绕（255 → 0）即可: 上位机只比较"是否与上次不同"。
 *
 * ⚠ 序号**只在本函数里递增**, 别在别处（发送每帧等路径）加 `seq++` ——
 *   那会让上位机把"陈旧的被保持值"误判成新样本。
 * ⚠ 本函数在 dual_yaw_control_step() 里位于**看门狗/使能判断之前**, 也就是说
 *   **故障/未使能期间照样跟踪 MCU2 并照常递增序号**, 但那些路径下**力矩输出**仍然被冻结
 *   （见 step 里的 yaw_big_zero_torque() / can_send_torque_small(0)）:
 *     · "不要输出力矩"与"是否读取反馈"是两件事, 回传帧在故障期间本来就照常发送;
 *     · 若连反馈也冻结, 恢复控制的那一刻只能拿一个陈旧值当锚点（大 yaw 若在故障期间
 *       被外力转动过, 上位机的修正还要被 big_enc_max_jump 限幅, 需 2~3 个新样本
 *       ≈0.3s@10Hz 才收敛）; 保持跟踪则恢复瞬间锚点就是新鲜的, 无恢复瞬态;
 *     · 告警能力不丢: 故障期间若 MCU2 也断了, 序号同样不再递增, 上位机看到的"年龄"
 *       照样持续增长。
 * ------------------------------------------------------------------------- */
static void yaw_feedback_update(void)
{
#if YAW_BIG_SOURCE_FROM_MCU2
    /* ── 真实结构: 值来自 MCU2 内部链路（同一次取数里两路一起刷新, 共用一个序号） ── */
    uint8_t got_new = 0u;

    {
        double big_angle = 0.0;                     /* 注意 double: 协议里 yaw_big_angle 是 double */
        float  big_omega = 0.0f;
        if (mcu2_get_yaw_big_sample(&big_angle, &big_omega)) {  /* TODO: 用户填写 */
            g_yaw_big_fb_angle = big_angle;
            g_yaw_big_fb_omega = big_omega;
            got_new = 1u;
        }
        /* else: 没有新样本 → 值与序号都不动（值被保持, 序号重复发送） */
    }
    {
        float yaw   = 0.0f;
        float omega = 0.0f;
        if (mcu2_get_chassis_imu_sample(&yaw, &omega)) {        /* TODO: 用户填写 */
            g_chassis_fb_yaw   = yaw;
            g_chassis_fb_omega = omega;
            got_new = 1u;
        }
        /* else: 保持旧值 */
    }

    if (got_new) {
        g_mcu2_seq++;       /* ← 唯一递增点: 本次取数拿到了 MCU2 新数据（两路只加 1 次） */
    }
#else
    /* ── 单 MCU 方案: 每周期都能从本地编码器/IMU 拿到新样本 → 序号每周期 +1（一次） ── */
    g_yaw_big_fb_angle = yaw_big_get_angle();       /* double, 多圈连续 */
    g_yaw_big_fb_omega = yaw_big_get_omega();
    g_chassis_fb_yaw   = get_chassis_imu_yaw();     /* TODO: 用户填写 */
    g_chassis_fb_omega = get_chassis_imu_omega();   /* TODO: 用户填写 */
    g_mcu2_seq++;                                   /* 本周期确实拿到了新数据（只加一次） */
#endif
}

/* 大 yaw 力矩"归零"出口: 按当前结构走本地 CAN 或转发给 MCU2。
 * 上电初始化、看门狗超时、未使能时都调用它 —— 大 yaw 的力矩清零绝不能漏。 */
static void yaw_big_zero_torque(void)
{
#if YAW_BIG_SOURCE_FROM_MCU2
    /* 转发一条零力矩指令给 MCU2:
     * mode = 仅力矩, 目标角 = 0, 目标角速度 = 0, τ = 0 N·m —— 即"大 yaw 不施力" */
    mcu2_send_yaw_big_command((uint8_t)YAW_MODE_TORQUE_ONLY, 0.0, 0.0f, 0.0f);
#else
    can_send_torque_big(0);                         /* TODO: 用户填写 can_send_torque_big() */
#endif
}

/* ---------------------------------------------------------------------------
 * yaw_big_control(): 大 yaw 控制
 *   mode 0: τ = yaw_big_torque                       （仅力矩, 直接施加）
 *   mode 1: τ = kp·(θ*−θ) + kd·(ω*−ω) + yaw_big_torque（力矩 + 位置/速度内环）
 * 大 yaw 可多圈自由旋转, 目标角为多圈连续角, 不做角度限位。
 *
 * 两种结构（YAW_BIG_SOURCE_FROM_MCU2）:
 *   1 = 真实结构: 本函数只把 (mode, θ*, ω*, τ_ff) **原样转发给 MCU2**, 由 MCU2 用它
 *       自己实时的大 yaw 编码器跑同样的内环并下发 CAN —— 这样闭环不会用到经由
 *       不稳定链路传来的旧角度; 本机不产生 CAN 指令, 返回 0。
 *   0 = 单 MCU: 本机读编码器 → 内环 → 力矩换算 → CAN 下发, 返回下发的指令值。
 * 无论哪种, 调用方（dual_yaw_control_step）都不要重复下发。
 * ------------------------------------------------------------------------- */
int16_t yaw_big_control(const yaw_cmd_t *cmd)
{
#if YAW_BIG_SOURCE_FROM_MCU2
    /* 转发给 MCU2。力矩换算系数/大 yaw 的 kp,kd 都属于 MCU2 的工程。 */
    mcu2_send_yaw_big_command(cmd->yaw_big_mode, cmd->yaw_big_target_angle,
                              cmd->yaw_big_target_velocity, cmd->yaw_big_torque);
    return 0;                                       /* 本机没有直接的 CAN 指令 */
#else
    double big_angle = yaw_big_get_angle();         /* rad, 多圈连续 */
    float  big_omega = yaw_big_get_omega();         /* rad/s */
    float  torque;
    int16_t cmd_out;

    if (cmd->yaw_big_mode == (uint8_t)YAW_MODE_TORQUE_PLUS_PID) {
        float kp = YAW_BIG_KP;
        float kd = YAW_BIG_KD;
        torque = kp * (float)(cmd->yaw_big_target_angle - big_angle)
               + kd * (cmd->yaw_big_target_velocity - big_omega)
               + cmd->yaw_big_torque;
    } else {
        torque = cmd->yaw_big_torque;               /* 仅力矩模式 */
    }

    cmd_out = yaw_torque_to_cmd(torque, YAW_BIG_TORQUE_TO_CMD_SCALE);
    can_send_torque_big(cmd_out);                   /* TODO: 用户填写 can_send_torque_big() */
    return cmd_out;
#endif
}

/* ---------------------------------------------------------------------------
 * 小 yaw: 该关节的绝对/相对计数 → 关节角（rad）
 * ------------------------------------------------------------------------- */
static float yaw_small_read_angle(void)
{
    int32_t counts = get_encoder_small_counts();
    float   angle  = (float)(counts - (int32_t)YAW_SMALL_ENCODER_ZERO_COUNTS) * YAW_SMALL_RAD_PER_COUNT;

#if YAW_SMALL_ANGLE_WRAP_TO_PM_PI
    /* 单圈编码器: 绕到 (−π, π]。注意这会掩盖"编码器整圈跳变", 仅在确认
     * 编码器为单圈且关节绝不可能转满一圈时启用。 */
    while (angle > (float)YAW_PI)  { angle -= (float)YAW_TWO_PI; }
    while (angle < -(float)YAW_PI) { angle += (float)YAW_TWO_PI; }
#endif
    return angle;
}

/* 小 yaw 角速度（rad/s, 由速度报文 rpm 换算） */
static float yaw_small_read_omega(void)
{
    return (float)get_encoder_small_velocity_rpm() * YAW_SMALL_RPM_TO_RAD_S;
}

/* 安全层输出。
 * ⚠ 行程**非对称**（−25° ~ +20°）, 所以"越限"必须分成两侧各自判断:
 *   硬限位: hard_min = θ ≤ YAW_SMALL_MIN_RAD（撞负侧）; hard_max = θ ≥ YAW_SMALL_MAX_RAD（撞正侧）
 *   软限位: soft_min = θ ≤ YAW_SMALL_TARGET_MIN_RAD; soft_max = θ ≥ YAW_SMALL_TARGET_MAX_RAD
 *   hard_limit / soft_limit 是两者的"或", 只作汇总位（日志/示波器）用, 逻辑判断请用分侧位。 */
typedef struct {
    float   target_angle;     /* 已限位目标角 rad */
    float   target_velocity;  /* 已限速目标角速度 rad/s */
    float   torque;           /* 已约束的力矩 N·m（含前馈） */
    float   allowed_rate;     /* 当前允许的 |ω| rad/s */
    uint8_t hard_min;         /* 1 = 已越过负侧硬限位（θ ≤ −25°） */
    uint8_t hard_max;         /* 1 = 已越过正侧硬限位（θ ≥ +20°） */
    uint8_t soft_min;         /* 1 = 已越过负侧软限位（θ ≤ −23°） */
    uint8_t soft_max;         /* 1 = 已越过正侧软限位（θ ≥ +18°） */
    uint8_t hard_limit;       /* 1 = hard_min || hard_max（汇总） */
    uint8_t soft_limit;       /* 1 = soft_min || soft_max（汇总） */
} small_yaw_guard_t;

/* ---------------------------------------------------------------------------
 * small_yaw_guard(): 小 yaw 安全层（**最后一道安全防线**）
 *
 * 机械行程**非对称 −25° ~ +20°**（中心 −2.5°）, 一旦撞死会打坏电机/线束, 因此这里做四件事:
 *   ① 目标角限位: θ* 夹到 [−23°, +18°]（两侧各留 2° 余量）。
 *      这一步天然使"越界后位置误差指向内侧", 即内环只会朝回中方向施力。
 *   ② 速度限速: 距**任一侧**硬限位 10° 以内（即 θ < −15° 或 θ > +10°）后, 允许 |ω*|
 *      随剩余角度线性下降, 贴到限位时只剩 0.2 rad/s; 越过软限位后禁止"向外"的目标速度。
 *   ③ 前馈约束: 越过软限位后, 朝外方向的前馈力矩清零 ——
 *      "超过软限位时禁止继续向外施力, 只允许回中方向力矩"（正侧朝外 = 正力矩, 负侧朝外 = 负力矩）。
 *   ④ 硬限位: θ ≤ −25° 或 θ ≥ +20° 时整帧力矩作废, 只允许零力矩/回中力矩
 *      （YAW_SMALL_HARD_CENTER_TORQUE_NM, 默认 0 = 零力矩）。
 *
 * 注意: 安全层优先于一切控制模式 —— 无论 mode 0/1、无论上位机下发什么,
 *       输出到 CAN 的力矩都必须再经过 small_yaw_apply_limit_torque()。
 * ------------------------------------------------------------------------- */
void small_yaw_guard(float angle, float target_angle, float target_velocity,
                     float torque_in, small_yaw_guard_t *out)
{
    float remaining;
    float dist_to_max;
    float dist_to_min;
    float allowed;

    if (out == NULL) {
        return;
    }
    out->hard_min = 0u;
    out->hard_max = 0u;
    out->soft_min = 0u;
    out->soft_max = 0u;
    out->hard_limit = 0u;
    out->soft_limit = 0u;
    out->torque     = torque_in;

    if (!isfinite(angle)) {
        /* 角度异常: 视为最危险情况, 零力矩（两侧都算越限, 便于上层监控发现） */
        out->hard_min = 1u;
        out->hard_max = 1u;
        out->hard_limit = 1u;
        out->target_angle = 0.0f;
        out->target_velocity = 0.0f;
        out->torque = 0.0f;
        out->allowed_rate = 0.0f;
        return;
    }

    /* ① 目标角限位到 [-23°, +18°]（两侧各自夹, 非对称） */
    if (target_angle > YAW_SMALL_TARGET_MAX_RAD) {
        target_angle = YAW_SMALL_TARGET_MAX_RAD;
    } else if (target_angle < YAW_SMALL_TARGET_MIN_RAD) {
        target_angle = YAW_SMALL_TARGET_MIN_RAD;
    }

    /* ② 允许速度: 按**到最近一侧硬限位**的剩余角度线性插值（两侧独立, 不用 fabsf 对称写法） */
    dist_to_max = YAW_SMALL_MAX_RAD - angle;      /* 到 +20° 侧还有多少 */
    dist_to_min = angle - YAW_SMALL_MIN_RAD;      /* 到 −25° 侧还有多少 */
    remaining = (dist_to_max < dist_to_min) ? dist_to_max : dist_to_min;
    if (remaining <= 0.0f) {
        allowed = YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S;                 /* 已贴死限位 */
    } else if (remaining >= YAW_SMALL_DECEL_ZONE_LEN_RAD) {
        allowed = YAW_SMALL_RATE_MAX_RAD_S;                           /* 远离限位 */
    } else {
        float ratio = remaining / YAW_SMALL_DECEL_ZONE_LEN_RAD;       /* 0 ~ 1 */
        allowed = YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S
                + (YAW_SMALL_RATE_MAX_RAD_S - YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S) * ratio;
    }
    if (allowed < 0.0f) {
        allowed = 0.0f;
    }
    out->allowed_rate = allowed;

    /* 目标速度限幅（限速） */
    if (target_velocity > allowed) {
        target_velocity = allowed;
    } else if (target_velocity < -allowed) {
        target_velocity = -allowed;
    }

    /* 越过软限位: 两侧**各自**判断, 且只禁止"继续向外"的那个方向 */
    if (angle >= YAW_SMALL_TARGET_MAX_RAD) {
        out->soft_max = 1u;
        if (target_velocity > 0.0f) {
            target_velocity = 0.0f;                                   /* 正侧: 禁止继续往外(正) */
        }
    }
    if (angle <= YAW_SMALL_TARGET_MIN_RAD) {
        out->soft_min = 1u;
        if (target_velocity < 0.0f) {
            target_velocity = 0.0f;                                   /* 负侧: 禁止继续往外(负) */
        }
    }
    /* 越过硬限位: 同样两侧独立 */
    if (angle >= YAW_SMALL_MAX_RAD) {
        out->hard_max = 1u;
    }
    if (angle <= YAW_SMALL_MIN_RAD) {
        out->hard_min = 1u;
    }
    out->hard_limit = (uint8_t)((out->hard_min || out->hard_max) ? 1u : 0u);
    out->soft_limit = (uint8_t)((out->soft_min || out->soft_max) ? 1u : 0u);

    /* ③④ 力矩约束 */
    if (out->hard_max) {
        /* 正向越限: 只允许负向（回中）力矩 */
        out->torque = -YAW_SMALL_HARD_CENTER_TORQUE_NM;
    } else if (out->hard_min) {
        /* 负向越限: 只允许正向（回中）力矩 */
        out->torque = YAW_SMALL_HARD_CENTER_TORQUE_NM;
    } else if (out->soft_max && (torque_in > 0.0f)) {
        out->torque = 0.0f;                                           /* 正侧软限位外禁止向外施力 */
    } else if (out->soft_min && (torque_in < 0.0f)) {
        out->torque = 0.0f;                                           /* 负侧软限位外禁止向外施力 */
    }

    out->target_angle    = target_angle;
    out->target_velocity = target_velocity;
}

/* 末级力矩约束: 对**内环算完之后**的总力矩再执行一次限位规则（与安全层构成双保险）。
 * 两侧判断全部来自 g 的分侧标志, 不再需要传入 θ 去"按符号推方向"。 */
static float small_yaw_apply_limit_torque(float torque, const small_yaw_guard_t *g)
{
    if (!isfinite(torque)) {
        return 0.0f;
    }
    if (g->hard_max) {
        /* 正向越限 → 丢弃内环输出, 只留零力矩/负向回中力矩 */
        return -YAW_SMALL_HARD_CENTER_TORQUE_NM;
    }
    if (g->hard_min) {
        /* 负向越限 → 只留零力矩/正向回中力矩 */
        return YAW_SMALL_HARD_CENTER_TORQUE_NM;
    }
    if (g->soft_max && (torque > 0.0f)) {
        return 0.0f;    /* 正侧软限位外禁止继续向外（正）施力 */
    }
    if (g->soft_min && (torque < 0.0f)) {
        return 0.0f;    /* 负侧软限位外禁止继续向外（负）施力 */
    }
    return torque;
}

/* ---------------------------------------------------------------------------
 * yaw_small_control(): 小 yaw 控制
 *   mode 0: τ = yaw_small_torque
 *   mode 1: τ = kp·(θ*−θ) + kd·(ω*−ω) + yaw_small_torque
 * 之后必须再过一遍限位规则（与安全层构成"双保险"）。
 * 本函数负责 CAN 下发, 调用方不要再重复下发。
 * 返回下发的指令值。
 * ------------------------------------------------------------------------- */
int16_t yaw_small_control(const yaw_cmd_t *cmd)
{
    float angle = yaw_small_read_angle();
    float omega = yaw_small_read_omega();
    small_yaw_guard_t g;
    float torque;
    int16_t cmd_out;

    /* ① 安全层: 限位/限速, 并约束前馈力矩 */
    small_yaw_guard(angle, cmd->yaw_small_target_angle, cmd->yaw_small_target_velocity,
                    cmd->yaw_small_torque, &g);

    /* ② 力矩合成（用上面安全层返回的、已限位的目标角与已限速的目标角速度） */
    if (cmd->yaw_small_mode == (uint8_t)YAW_MODE_TORQUE_PLUS_PID) {
        torque = YAW_SMALL_KP * (g.target_angle - angle)
               + YAW_SMALL_KD * (g.target_velocity - omega)
               + g.torque;
    } else {
        torque = g.torque;
    }

    /* ③ 末级限位约束: 内环输出同样不允许把关节继续往限位外推 */
    torque = small_yaw_apply_limit_torque(torque, &g);

    cmd_out = yaw_torque_to_cmd(torque, YAW_SMALL_TORQUE_TO_CMD_SCALE);
    can_send_torque_small(cmd_out);                 /* TODO: 用户填写 can_send_torque_small() */
    return cmd_out;
}

/* ============================================================================
 * 8. 接收状态机（字节流 → 合法帧）
 * ==========================================================================*/

static void yaw_rx_drop(uint32_t n)
{
    if (n >= g_rx_len) {
        g_rx_len = 0u;
        return;
    }
    memmove(g_rx_buf, g_rx_buf + n, (size_t)(g_rx_len - n));
    g_rx_len -= n;
}

/* 尝试从缓冲里切出一个完整合法帧; 不完整就等下一个字节 */
void yaw_rx_try_parse(void)
{
    yaw_cmd_t parsed;
    uint8_t   res;

    /* ① 找前导 0x42 0x52 0x03（逐字节滑动, 只丢 1 字节以免漏掉真正的帧头） */
    for (;;) {
        if (g_rx_len < PC_TO_MCU_PREAMBLE_SIZE) {
            return;                                     /* 字节不够, 等 */
        }
        if ((g_rx_buf[0] == (uint8_t)YAW_PREAMBLE_1) &&
            (g_rx_buf[1] == (uint8_t)YAW_PREAMBLE_2) &&
            (g_rx_buf[2] == (uint8_t)YAW_PROTOCOL_VERSION)) {
            break;                                      /* 前导对齐 */
        }
        yaw_rx_drop(1u);
    }

    /* ② 长度字段必须为 36; 若为 53 说明这是串口回环的本机回传帧, 直接丢弃 */
    if (g_rx_len < (PC_TO_MCU_PREAMBLE_SIZE + 1u)) {
        return;
    }
    if (g_rx_buf[PC_TO_MCU_PREAMBLE_SIZE] != (uint8_t)PC_TO_MCU_PAYLOAD_SIZE) {
        yaw_rx_drop(1u);                                /* 假头, 重新同步 */
        return;
    }

    /* ③ 帧是否收齐 */
    if (g_rx_len < PC_TO_MCU_FRAME_SIZE) {
        return;
    }

    /* ④ 整帧校验 + 解析 */
    res = parse_packet(g_rx_buf, PC_TO_MCU_FRAME_SIZE, &parsed);
    if (res == YAW_PARSE_OK) {
        yaw_enter_critical();                           /* TODO: 用户填写 —— 关中断 */
        memcpy(&g_rx_staging, &parsed, sizeof(g_rx_staging));
        g_rx_staging_ready = 1u;
        yaw_exit_critical();                            /* TODO: 用户填写 —— 开中断 */
    } else {
        g_rx_err_last = res;                            /* 出错计数/调试用 */
    }
    yaw_rx_drop(PC_TO_MCU_FRAME_SIZE);                  /* 无论成败都消费掉这一帧 */
}

/* 串口每收到 1 字节调用一次（可放在 USART RX 中断里） */
void dual_yaw_rx_byte(uint8_t b)
{
    if (g_rx_len < YAW_RX_BUFFER_SIZE) {
        g_rx_buf[g_rx_len] = b;
        g_rx_len++;
    } else {
        /* 缓冲满: 丢最老的一字节, 保证不越界（正常不会发生, 41B/帧） */
        memmove(g_rx_buf, g_rx_buf + 1u, (size_t)(YAW_RX_BUFFER_SIZE - 1u));
        g_rx_buf[YAW_RX_BUFFER_SIZE - 1u] = b;
    }
    yaw_rx_try_parse();
}

/* 一次收到一块数据时使用 */
void dual_yaw_rx_bytes(const uint8_t *data, uint32_t len)
{
    uint32_t i;
    if (data == NULL) {
        return;
    }
    for (i = 0u; i < len; ++i) {
        dual_yaw_rx_byte(data[i]);
    }
}

/* ============================================================================
 * 9. 看门狗
 * ==========================================================================*/

/* 返回 1 = 看门狗超时（必须清零力矩） */
static uint8_t yaw_watchdog_expired(uint32_t now_ms)
{
    if (!g_ever_valid_rx) {
        return 1u;      /* 上电后从未收到合法帧 → 一切力矩保持为 0 */
    }
    /* 无符号相减, 天然处理 32 位 ms 计数回绕 */
    return (uint32_t)(now_ms - g_last_valid_rx_ms) > (uint32_t)YAW_WATCHDOG_TIMEOUT_MS;
}

/* ============================================================================
 * 10. 初始化与主循环入口
 * ==========================================================================*/

void dual_yaw_control_init(void)
{
    memset(&g_cmd, 0, sizeof(g_cmd));
    memset(&g_rx_staging, 0, sizeof(g_rx_staging));
    g_rx_len = 0u;
    g_rx_staging_ready = 0u;

    g_big_turns = 0;
    g_big_last_angle_in_one_turn = 0.0;
    g_big_turns_inited = 0u;

    g_last_valid_rx_ms = 0u;
    g_ever_valid_rx = 0u;
    g_watchdog_fault = 1u;      /* 收到第一帧合法指令前, 力矩恒为 0 */
    g_mark = 0u;
    g_last_tx_ms = 0u;

    /* MCU2 数据: 值清 0, 新样本序号初值 0（无"0 = 从未收到"约定, 上位机首帧即视为一次更新） */
    g_yaw_big_fb_angle = 0.0;
    g_yaw_big_fb_omega = 0.0f;
    g_chassis_fb_yaw   = 0.0f;
    g_chassis_fb_omega = 0.0f;
    g_mcu2_seq         = YAW_MCU2_SEQ_INIT;

    /* 上电先显式下发零力矩, 防止电调保持上电前的值 */
    yaw_big_zero_torque();      /* 本地 CAN 或转发 MCU2, 由 YAW_BIG_SOURCE_FROM_MCU2 决定 */
    can_send_torque_small(0);   /* TODO: 用户填写 can_send_torque_small() */
}

/* ---------------------------------------------------------------------------
 * dual_yaw_control_step(): 周期任务（建议 1kHz）
 *   ① 取走最新合法帧, 刷新看门狗
 *   ② 大 yaw 多圈累计（仅单 MCU 方案）
 *   ③ **刷新 MCU2 数据**(大 yaw + 底盘 IMU 同源同批)的值与新样本序号 `mcu2_seq`
 *      —— 与自瞄/看门狗**无关**: 看门狗管的是"不要输出力矩", 不是"不要读反馈"。
 *         故障期间继续跟踪的好处: 恢复控制的瞬间锚点就是新鲜的, 不会拿陈旧值重锚;
 *         同时若链路真的断了, 序号同样不再递增、上位机看到的"年龄"照样增长。
 *   ④ 看门狗超时 → 两关节力矩清零（绝不保持上一帧力矩）
 *   ⑤ 正常受控时: 两关节控制（小 yaw 内含安全层; 大 yaw 本地闭环或转发 MCU2）
 *   ⑥ 按周期回传反馈帧（47B, 含 mcu2_seq）
 * ------------------------------------------------------------------------- */
void dual_yaw_control_step(uint32_t now_ms)
{
    yaw_feedback_t fb;
    uint8_t        tx_buf[MCU_TO_PC_FRAME_SIZE];
    uint8_t        any_valid_cmd = 0u;

    /* ① 取走中断侧解析好的最新指令 */
    if (g_rx_staging_ready) {
        yaw_enter_critical();                           /* TODO: 用户填写 —— 关中断 */
        memcpy(&g_cmd, &g_rx_staging, sizeof(g_cmd));
        g_rx_staging_ready = 0u;
        yaw_exit_critical();                            /* TODO: 用户填写 —— 开中断 */
        g_last_valid_rx_ms = now_ms;                    /* 看门狗喂狗 */
        g_ever_valid_rx = 1u;
        any_valid_cmd = 1u;
    }

#if YAW_BIG_SOURCE_FROM_MCU2 == 0
    /* ② 单 MCU 方案: 大 yaw 多圈累计必须每周期都做（与自瞄/看门狗无关） */
    yaw_big_update_multiturn();
#endif

    /* ③ 反馈采集: **在看门狗/使能判断之前**取 MCU2 的新样本, 有新样本就刷新值并让序号 +1。
     *    理由: 看门狗/未使能的语义是"不要输出力矩", 与"是否读取反馈"无关 ——
     *      回传帧在故障期间本来就照常发送, 若连反馈一起冻结, 恢复控制的那一刻只能拿
     *      一个陈旧值当锚点(大 yaw 若在故障期间被外力转动过, 上位机的修正还要被
     *      big_enc_max_jump 限幅, 需 2~3 个新样本 ≈0.3s@10Hz 才收敛);
     *      保持跟踪则恢复瞬间锚点就是新鲜的, 控制恢复无瞬态。
     *    同时"年龄"仍真实反映链路状况: 故障期间若 MCU2 也断了, 序号同样不再递增。 */
    yaw_feedback_update();

    /* ④ 看门狗: 超时 → 两关节力矩清零
     *    这是必需的: 上位机死机/掉线时若继续执行最后一帧力矩, 云台会持续
     *    朝一个方向加速, 属于必须避免的失控工况。
     *    注意: 这里冻的是**力矩输出**, 反馈与序号已经在上面照常更新。 */
    g_watchdog_fault = yaw_watchdog_expired(now_ms);
    if (g_watchdog_fault) {
        yaw_big_zero_torque();                          /* 本地 CAN 或转发 MCU2 */
        can_send_torque_small(0);                       /* TODO: 用户填写 can_send_torque_small() */
    } else {
        /* ⑤ 自瞄门控: 上位机自瞄开关 与 电控自瞄开关 相与（沿用旧示例语义） */
        uint8_t gating_ok = 1u;
#if YAW_AUTO_AIM_GATING_ENABLE
        gating_ok = (uint8_t)((g_cmd.auto_aim_enable && get_auto_aim_switch()) ? 1u : 0u);
#endif

        if (gating_ok) {
            /* 两关节控制（小 yaw 内含安全层; 大 yaw 按结构本地闭环或转发 MCU2） */
            (void)yaw_big_control(&g_cmd);
            (void)yaw_small_control(&g_cmd);
        } else {
            /* 非自瞄/未使能: 同样只冻力矩（手动操控逻辑由电控自己实现, 本示例不涉及） */
            yaw_big_zero_torque();                      /* 本地 CAN 或转发 MCU2 */
            can_send_torque_small(0);                   /* TODO: 用户填写 can_send_torque_small() */
        }
    }
    (void)any_valid_cmd;    /* 需要"新帧到达"事件时可用它触发一次动作 */

    /* ⑥ 回传反馈帧（MCU → PC, data_size = 42, 整帧 47B） */
    if ((uint32_t)(now_ms - g_last_tx_ms) >= (uint32_t)YAW_FEEDBACK_TX_PERIOD_MS) {
        g_last_tx_ms = now_ms;
        g_mark++;                                       /* 递增标志位 */

        fb.bullet_velocity      = get_bullet_velocity();        /* TODO: 用户填写 */
        fb.pitch_angle          = get_pitch_angle_raw();        /* TODO: 用户填写 */
        /* ── MCU2 数据(大 yaw + 底盘 IMU): 直接透传"保持值"(可能是几十~几百 ms 前的旧值) ── */
        fb.yaw_big_angle        = g_yaw_big_fb_angle;           /* double, 多圈连续 */
        fb.yaw_big_omega        = g_yaw_big_fb_omega;
        fb.chassis_imu_yaw      = g_chassis_fb_yaw;
        fb.chassis_imu_omega    = g_chassis_fb_omega;
        /* ── 小 yaw / pitch: 每帧都是新样本, 不需要序号 ── */
        fb.yaw_small_angle      = yaw_small_read_angle();
        fb.yaw_small_omega      = yaw_small_read_omega();
        fb.mark                 = g_mark;
        fb.color                = get_mark_color();             /* TODO: 用户填写 */
        fb.auto_aim_switch      = get_auto_aim_switch();        /* TODO: 用户填写 */
        fb.yaw_big_temperature  = get_motor_temperature_big();  /* TODO: 用户填写 */
        fb.yaw_small_temperature= get_motor_temperature_small();/* TODO: 用户填写 */
        /* MCU2 新样本序号: 值保持期间原样重复（不递增）;
         * 上位机判定: 首帧到达 或 序号变化 或 值变化 ⇒ 新样本 */
        fb.mcu2_seq             = g_mcu2_seq;

        if (build_packet(&fb, tx_buf, (uint32_t)sizeof(tx_buf)) == MCU_TO_PC_FRAME_SIZE) {
            uart_send_bytes(tx_buf, (uint32_t)MCU_TO_PC_FRAME_SIZE); /* TODO: 用户填写 */
        }
    }
}

/* ============================================================================
 * 11. 安全层自检（可选编译, 默认关闭 —— 定义 YAW_SMALL_LIMIT_SELFTEST=1 打开）
 *
 * 目的: 行程改成**非对称 −25° ~ +20°** 之后, 最容易犯的错误就是"某一处还在用对称写法"
 *       （`±LIMIT` / `fabsf(angle) > LIMIT`）。本节把安全层的**两侧**逐条钉死:
 *         · 宏关系（目标夹取落在硬限位内、减速区起点 = 距限位 10°、两侧独立）
 *         · 目标角夹取两侧各自的数值（−23° / +18°）
 *         · 减速区两侧各自的距离→允许速度映射（到限位距离相同 ⇒ 允许速度相同）
 *         · 越软限位只禁止"向外"方向（两侧独立）、越硬限位只允许回中方向
 *         · 末级力矩约束（small_yaw_apply_limit_torque）与安全层一致
 *
 * 纯函数 + 无硬件依赖 ⇒ 可直接在 PC 上跑:
 *     gcc -std=c11 -Wall -Wextra -pedantic mcu_code_demo/dual_yaw_control_selftest.c -lm
 *     （runner 里提供桩函数与 main, 见该文件）
 * MCU 上想跑就把它编进来、传一个打印回调即可（回调可为 NULL, 只统计失败数）。
 * ==========================================================================*/
#ifndef YAW_SMALL_LIMIT_SELFTEST
#define YAW_SMALL_LIMIT_SELFTEST 0    /* 0 = 不编入（不占 MCU flash）; 1 = 编入 */
#endif

#if YAW_SMALL_LIMIT_SELFTEST

/* 自检结果回调: name = 检查项, ok = 1 通过 / 0 失败（可为 NULL = 只统计不打印） */
typedef void (*yaw_selftest_report_fn)(const char *name, int ok);

/* 浮点比较容差（自检里所有比较都用它, 避免 == 比较浮点） */
#define YAW_SELFTEST_EPS   1e-6f

static int yaw_selftest_near(float a, float b)
{
    float d = a - b;
    if (d < 0.0f) { d = -d; }
    return (d <= YAW_SELFTEST_EPS) ? 1 : 0;
}
static int yaw_selftest_deg(float rad, float deg)
{
    return yaw_selftest_near(rad, (float)(deg * YAW_DEG2RAD));
}

/* 返回失败条数（0 = 全部通过） */
int yaw_small_limit_selftest(yaw_selftest_report_fn report)
{
    int fail = 0;
    small_yaw_guard_t g;
    small_yaw_guard_t g2;

#define YAW_SELFTEST_CHK(cond, label)                       \
    do {                                                    \
        const int yaw_st_ok = (cond) ? 1 : 0;                \
        if (!yaw_st_ok) { fail++; }                          \
        if (report != NULL) { report((label), yaw_st_ok); }  \
    } while (0)

    /* ── 1) 宏关系（运行期核对: _Static_assert 不允许浮点常量表达式） ── */
    YAW_SELFTEST_CHK(yaw_selftest_deg(YAW_SMALL_MIN_RAD, -25.0f),
                     "[宏] 硬限位负侧 = −25°");
    YAW_SELFTEST_CHK(yaw_selftest_deg(YAW_SMALL_MAX_RAD, 20.0f),
                     "[宏] 硬限位正侧 = +20°");
    YAW_SELFTEST_CHK((YAW_SMALL_MIN_RAD < 0.0f) && (YAW_SMALL_MAX_RAD > 0.0f),
                     "[宏] 行程跨过 0（−25° < 0 < +20°）");
    YAW_SELFTEST_CHK(yaw_selftest_deg(YAW_SMALL_TARGET_MIN_RAD, -23.0f),
                     "[宏] 目标角下限 = −23°（硬限位 + 2° 余量）");
    YAW_SELFTEST_CHK(yaw_selftest_deg(YAW_SMALL_TARGET_MAX_RAD, 18.0f),
                     "[宏] 目标角上限 = +18°（硬限位 − 2° 余量）");
    YAW_SELFTEST_CHK((YAW_SMALL_TARGET_MIN_RAD > YAW_SMALL_MIN_RAD) &&
                     (YAW_SMALL_TARGET_MAX_RAD < YAW_SMALL_MAX_RAD),
                     "[宏] 目标夹取区间严格落在硬限位内");
    YAW_SELFTEST_CHK(yaw_selftest_deg(YAW_SMALL_DECEL_START_MIN_RAD, -15.0f),
                     "[宏] 负侧减速区起点 = −15°（距 −25° 正好 10°）");
    YAW_SELFTEST_CHK(yaw_selftest_deg(YAW_SMALL_DECEL_START_MAX_RAD, 10.0f),
                     "[宏] 正侧减速区起点 = +10°（距 +20° 正好 10°）");
    YAW_SELFTEST_CHK(yaw_selftest_near(YAW_SMALL_DECEL_ZONE_LEN_RAD,
                                       (float)(10.0 * YAW_DEG2RAD)),
                     "[宏] 减速区长度 = 10°");

    /* ── 2) 目标角夹取: 两侧各自夹到 [−23°, +18°] ── */
    small_yaw_guard(0.0f, (float)(30.0 * YAW_DEG2RAD), 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_deg(g.target_angle, 18.0f), "[夹取] θ*=+30° → +18°");
    small_yaw_guard(0.0f, (float)(-30.0 * YAW_DEG2RAD), 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_deg(g.target_angle, -23.0f), "[夹取] θ*=−30° → −23°");
    small_yaw_guard(0.0f, (float)(5.0 * YAW_DEG2RAD), 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_deg(g.target_angle, 5.0f), "[夹取] 行程内目标角不被改");

    /* ── 3) 减速区: 两侧按**各自**到硬限位的距离线性降速 ── */
    small_yaw_guard(0.0f, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, YAW_SMALL_RATE_MAX_RAD_S),
                     "[限速] θ=0° → 允许 |ω|=6.0");
    small_yaw_guard(YAW_SMALL_DECEL_START_MAX_RAD, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, YAW_SMALL_RATE_MAX_RAD_S),
                     "[限速] θ=+10°（正侧减速区起点）→ 6.0");
    small_yaw_guard(YAW_SMALL_DECEL_START_MIN_RAD, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, YAW_SMALL_RATE_MAX_RAD_S),
                     "[限速] θ=−15°（负侧减速区起点）→ 6.0");
    /* +15° 距 +20° 还有 5°（半程） ⇒ 0.2 + (6.0−0.2)*0.5 = 3.1 */
    small_yaw_guard((float)(15.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, 3.1f), "[限速] θ=+15° → 3.1（半程）");
    /* −20° 距 −25° 还有 5°（半程） ⇒ 同样是 3.1 —— **按各自到硬限位的距离算, 不看 |θ|** */
    small_yaw_guard((float)(-20.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g2);
    YAW_SELFTEST_CHK(yaw_selftest_near(g2.allowed_rate, 3.1f), "[限速] θ=−20° → 3.1（半程）");
    YAW_SELFTEST_CHK(yaw_selftest_near(g2.allowed_rate, g.allowed_rate),
                     "[限速] 距各自硬限位同为 5° ⇒ 两侧允许速度相同（对称写法在此会不等）");
    small_yaw_guard(YAW_SMALL_MAX_RAD, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S),
                     "[限速] θ=+20°（贴正侧硬限位）→ 0.2");
    small_yaw_guard(YAW_SMALL_MIN_RAD, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S),
                     "[限速] θ=−25°（贴负侧硬限位）→ 0.2");
    small_yaw_guard((float)(-26.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.allowed_rate, YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S),
                     "[限速] θ=−26°（越限）→ 仍只允许 0.2");

    /* ── 4) 软限位标志: 两侧独立 ── */
    small_yaw_guard((float)(19.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK((g.soft_max == 1u) && (g.soft_min == 0u) && (g.hard_max == 0u),
                     "[软限位] θ=+19° → 只有 soft_max, 未越硬限位");
    small_yaw_guard((float)(-24.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK((g.soft_min == 1u) && (g.soft_max == 0u) && (g.hard_min == 0u),
                     "[软限位] θ=−24° → 只有 soft_min, 未越硬限位");
    small_yaw_guard((float)(5.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK((g.soft_min == 0u) && (g.soft_max == 0u) && (g.hard_limit == 0u) &&
                     (g.soft_limit == 0u),
                     "[软限位] θ=+5°（行程中部）→ 四个标志全 0");
    /* 负侧 12° 处（−12°）在负侧软限位（−23°）内 ⇒ 不应报警; 而正侧 +12° 已越软限位 +18°? 否 */
    small_yaw_guard((float)(-12.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(g.soft_limit == 0u, "[软限位] θ=−12° → 未越软限位");
    small_yaw_guard((float)(12.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(g.soft_limit == 0u, "[软限位] θ=+12° → 未越软限位");

    /* ── 5) 越软限位: 只禁止"向外"的前馈力矩, 反向（回中）保留 ── */
    small_yaw_guard((float)(19.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.5f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.torque, 0.0f),
                     "[力矩] θ=+19° 且 τ=+0.5（向外）→ 清零");
    small_yaw_guard((float)(19.0 * YAW_DEG2RAD), 0.0f, 0.0f, -0.5f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.torque, -0.5f),
                     "[力矩] θ=+19° 且 τ=−0.5（回中）→ 保留");
    small_yaw_guard((float)(-24.0 * YAW_DEG2RAD), 0.0f, 0.0f, -0.5f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.torque, 0.0f),
                     "[力矩] θ=−24° 且 τ=−0.5（向外）→ 清零");
    small_yaw_guard((float)(-24.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.5f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.torque, 0.5f),
                     "[力矩] θ=−24° 且 τ=+0.5（回中）→ 保留");
    small_yaw_guard((float)(-12.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.5f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.torque, 0.5f),
                     "[力矩] 未越软限位时前馈力矩原样通过");

    /* ── 6) 越硬限位: 只允许零力矩/回中力矩（CENTER 默认 0） ── */
    small_yaw_guard((float)(20.5 * YAW_DEG2RAD), 0.0f, 0.0f, -0.5f, &g);
    YAW_SELFTEST_CHK((g.hard_max == 1u) && (g.hard_limit == 1u) &&
                     yaw_selftest_near(g.torque, -YAW_SMALL_HARD_CENTER_TORQUE_NM),
                     "[硬限位] θ=+20.5° → hard_max, 只留回中力矩");
    small_yaw_guard((float)(-25.5 * YAW_DEG2RAD), 0.0f, 0.0f, 0.5f, &g);
    YAW_SELFTEST_CHK((g.hard_min == 1u) && (g.hard_limit == 1u) &&
                     yaw_selftest_near(g.torque, YAW_SMALL_HARD_CENTER_TORQUE_NM),
                     "[硬限位] θ=−25.5° → hard_min, 只留回中力矩");
    /* 边界: 恰好等于硬限位也算越限（宁可误触发不可漏触发） */
    small_yaw_guard(YAW_SMALL_MAX_RAD, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(g.hard_max == 1u, "[硬限位] θ 恰为 +20° → 记为越限（保守）");
    small_yaw_guard(YAW_SMALL_MIN_RAD, 0.0f, 0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(g.hard_min == 1u, "[硬限位] θ 恰为 −25° → 记为越限（保守）");

    /* ── 7) 越软限位禁止"向外"的目标速度（两侧独立） ── */
    small_yaw_guard((float)(19.0 * YAW_DEG2RAD), 0.0f, 5.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.target_velocity, 0.0f),
                     "[限速] θ=+19° 且 ω*=+5（向外）→ 目标速度归零");
    small_yaw_guard((float)(19.0 * YAW_DEG2RAD), 0.0f, -5.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(g.target_velocity < 0.0f,
                     "[限速] θ=+19° 且 ω*=−5（回中）→ 保留（仅被限速夹到 allowed）");
    small_yaw_guard((float)(-24.0 * YAW_DEG2RAD), 0.0f, -5.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_near(g.target_velocity, 0.0f),
                     "[限速] θ=−24° 且 ω*=−5（向外）→ 目标速度归零");
    small_yaw_guard((float)(-24.0 * YAW_DEG2RAD), 0.0f, 5.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(g.target_velocity > 0.0f,
                     "[限速] θ=−24° 且 ω*=+5（回中）→ 保留（仅被限速夹到 allowed）");

    /* ── 8) 末级力矩约束（内环算完之后再过一遍）与安全层一致 ── */
    small_yaw_guard((float)(19.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g2);
    YAW_SELFTEST_CHK(yaw_selftest_near(small_yaw_apply_limit_torque(1.0f, &g2), 0.0f),
                     "[末级] θ=+19°: 内环输出 +1.0（向外）→ 清零");
    YAW_SELFTEST_CHK(yaw_selftest_near(small_yaw_apply_limit_torque(-1.0f, &g2), -1.0f),
                     "[末级] θ=+19°: 内环输出 −1.0（回中）→ 保留");
    small_yaw_guard((float)(-24.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g2);
    YAW_SELFTEST_CHK(yaw_selftest_near(small_yaw_apply_limit_torque(-1.0f, &g2), 0.0f),
                     "[末级] θ=−24°: 内环输出 −1.0（向外）→ 清零");
    small_yaw_guard((float)(20.5 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g2);
    YAW_SELFTEST_CHK(yaw_selftest_near(small_yaw_apply_limit_torque(-1.0f, &g2),
                                       -YAW_SMALL_HARD_CENTER_TORQUE_NM),
                     "[末级] θ=+20.5°（越硬限位）→ 只留回中力矩");
    small_yaw_guard((float)(5.0 * YAW_DEG2RAD), 0.0f, 0.0f, 0.0f, &g2);
    YAW_SELFTEST_CHK(yaw_selftest_near(small_yaw_apply_limit_torque(0.7f, &g2), 0.7f),
                     "[末级] 行程中部: 内环输出原样通过");

    /* ── 9) 异常输入 ── */
    small_yaw_guard((float)NAN, 0.0f, 0.0f, 0.5f, &g);
    YAW_SELFTEST_CHK((g.hard_limit == 1u) && yaw_selftest_near(g.torque, 0.0f) &&
                     yaw_selftest_near(g.allowed_rate, 0.0f),
                     "[异常] θ=NaN → 两侧都算越限 + 零力矩");
    small_yaw_guard((float)(-26.0 * YAW_DEG2RAD), (float)(-30.0 * YAW_DEG2RAD),
                    0.0f, 0.0f, &g);
    YAW_SELFTEST_CHK(yaw_selftest_deg(g.target_angle, -23.0f),
                     "[异常] θ=−26°（越限）时目标角仍被夹到 −23°（误差天然指向内侧）");
    small_yaw_guard(0.0f, 0.0f, 0.0f, 0.0f, NULL);   /* 必须不崩 */
    YAW_SELFTEST_CHK(1, "[异常] out = NULL 不崩溃");

#undef YAW_SELFTEST_CHK
    return fail;
}

#endif /* YAW_SMALL_LIMIT_SELFTEST */
