// ============================================================================
// test_serial.cpp — 串口链路自检（MCU / IMU 收发包 + 协议字段解码）
//
// 移植自原仓库 `TorqueController/src/test_serial.cpp`（target `tcs_test_serial`）：
// 「打印收到的每一帧协议字段 + 按固定节拍发一包无害的帧」这个核心用途**完全保留**，
// 本仓库按双级 yaw 协议与安全约定做了适配（差异见文末列表）。
//
// 定位: **上实车第一个跑的程序**。它不碰任何控制逻辑、不建 RobotController、
//   不跑状态估计/MPC —— 只回答四个问题:
//     ① 串口在不在、选中的是哪个口、是不是 MCU / IMU 该有的那个口；
//     ② 我们的帧能不能被电控接受（CRC8 对不对、data_size 对不对）；
//     ③ 电控 / IMU 的帧能不能被我们解出来（前导、CRC、字段偏移是否与协议一致）；
//     ④ 链路质量: 收发频率、丢帧、MCU2 那条低速链路多久刷新一次（mcu2_seq 递增率）。
//
// ★ 安全（本仓库约定，与 tools/pitch_calibration.cpp / python/scripts/collect_sysid.py 一致）:
//   · 两个 yaw 关节一律 **YAW_MODE_TORQUE_ONLY + 力矩 0**（绝不发位置/速度目标）；
//   · `auto_aim_enable` **默认 0**（原仓库写 1）——本工具不发任何运动指令，
//     写 0 保证即使电控收到帧也不会因为本工具而动作；只有在要验证"电控确实接受
//     我们的力矩通道"时才加 `--auto-aim`；
//   · `fire` 恒为 0，**不提供**任何开关（避免误触发射）；
//   · `pitch_target_angle` 默认 0。原仓库硬编码 `10.0f`，其语义是**旧构型的电控原始单位**；
//     本构型 pitch 映射是恒等占位（raw 即弧度）⇒ 照抄会被当成 10 rad ≈ 573° 下发，
//     可能顶到机械限位。要动 pitch 必须显式 `--pitch=<rad>`，且 |pitch| > 0.5 rad 直接拒绝；
//   · 任何退出路径（正常结束 / 报错 / Ctrl+C / Ctrl+C 再来一次）都先连发若干帧**零力矩**；
//   · 另提供 `--no-send` 纯监听模式（一个字节都不发），用于排查"到底是谁在动"。
//
// 用法:
//   ./build/tcbs_test_serial --list                     # 只列串口 + 选中结果（无硬件也能跑）
//   ./build/tcbs_test_serial --selftest                 # 纯软件自检（协议布局/CRC/安全不变量）
//   ./build/tcbs_test_serial --help
//   ./build/tcbs_test_serial                            # 实车: 100 Hz 发零力矩帧 + 打印收到的每帧
//   ./build/tcbs_test_serial --no-send                  # 实车: 只监听不发（最安全）
//   ./build/tcbs_test_serial --dur=10 --imu             # 跑 10 s，连 IMU 帧一起打印
//   ./build/tcbs_test_serial --summary-only             # 不逐帧打印，只每秒一行统计
//
// 与原仓库 `tcs_test_serial` 的差异（只列适配点，用途不变）:
//   1) `namespace tcs` → `tcbs`、`#include "tcbs/..."`；
//   2) 发送包按本仓库协议 v0x03 的**双关节**字段填写（原为单 yaw 字段），语义等价:
//      两关节均"仅力矩 + 0 N·m"；
//   3) `auto_aim_enable` 默认 0（原 1）、pitch 目标默认 0（原硬编码 10.0f）——安全理由见上；
//   4) Ctrl+C 不再直接 `exit(0)`，改为置标志 → 主流程先发零力矩再关句柄 → 退出码 130；
//      再按一次 Ctrl+C 立即退出（急停用）；
//   5) 预检首个数据帧（`--wait`，默认 3 s），拿不到就**明确报错 + 诊断 + 退出码 1**，
//      不静默空转（原仓库无硬件时会一直打印 "Sending packets..."）；
//   6) 新增 --list / --selftest / --no-send / --dur / --count / --hz / --every /
//      --raw / --imu / --imu-heartbeat / --summary-only / --auto-aim / --pitch，
//      以及 1 Hz 统计行（收发频率、mcu2_seq 递增率、链路静默告警）；
//   7) 打印实际选中的串口名与是否打开成功（原仓库只由 SerialProtocol 内部 printf）。
// ============================================================================
#include "tcbs/communication/Communications.hpp"
#include "tcbs/communication/Protocol.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tcbs {

namespace {

// ============================================================================
// 常量 / 选项
// ============================================================================
constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;

constexpr int    kZeroTorqueFrames = 20;      // 退出前连发的零力矩帧数（与其它工具一致）
constexpr double kFramePeriodS     = 0.01;    // 100 Hz 发送节拍（原仓库 10 ms/帧，一致）
constexpr double kDefaultWaitS     = 3.0;     // 预检首个数据帧的最长等待
constexpr double kStaleWarnS       = 1.0;     // 超过这么久没有新帧 ⇒ 告警
constexpr double kMaxPitchAbsRad   = 0.50;    // --pitch 的硬上限（≈28.6°），见文件头安全说明

struct Options {
    bool   help = false;
    bool   list = false;
    bool   selftest = false;
    bool   no_send = false;                 // 纯监听
    bool   print_imu = false;               // 逐帧打印 IMU
    bool   raw = false;                     // 附加十六进制原始帧
    bool   summary_only = false;            // 只打 1 Hz 统计
    bool   auto_aim = false;                // auto_aim_enable: 默认 0（安全）
    bool   imu_heartbeat = false;           // 是否给 IMU 发心跳（默认不发）
    double pitch = 0.0;                     // pitch 目标（rad），默认 0
    bool   pitch_given = false;
    double hz = 1.0 / kFramePeriodS;        // 发送频率
    double dur = 0.0;                       // >0: 跑这么多秒后正常退出
    long   count = 0;                       // >0: 收到这么多 MCU 帧后正常退出
    long   every = 1;                       // 每 N 帧打印一次
    double wait = kDefaultWaitS;
};

void printUsage(const char* prog) {
    std::cout <<
        "用法: " << prog << " [选项]\n"
        "\n"
        "模式（互斥，按此优先级）:\n"
        "  --list                只枚举 /dev/ttyACM* 与各自 iProduct，并打印 MCU / IMU 选择器\n"
        "                        会选中哪个口（**不需要硬件在线也能跑**，现场排查第一步）\n"
        "  --selftest            纯软件自检: 包布局/字段偏移 vs 协议常量、CRC8 往返、\n"
        "                        零力矩安全不变量、--pitch 守卫\n"
        "  (默认)                实车链路自检: 按 --hz 发无害帧 + 打印收到的每一帧\n"
        "\n"
        "选项:\n"
        "  --no-send             一个字节都不发（纯监听）\n"
        "  --auto-aim            发送帧里 auto_aim_enable=1（**默认 0**，见文件头安全说明）\n"
        "  --pitch=<rad>         pitch 目标角（默认 0，|值| > 0.5 rad 直接拒绝）\n"
        "  --hz=<v>              发送频率（默认 100）\n"
        "  --dur=<s>             跑 s 秒后正常退出（默认 0 = 一直跑，Ctrl+C 结束）\n"
        "  --count=<n>           收到 n 帧 MCU 数据后正常退出\n"
        "  --every=<n>           每 n 帧打印一次（默认 1；高频链路可调大）\n"
        "  --raw                 额外打印每帧的十六进制字节（按本机结构体布局 = 线格式）\n"
        "  --imu                 连 IMU 帧一起逐帧打印（默认只在统计行里显示频率）\n"
        "  --imu-heartbeat       给 IMU 发心跳帧（默认不发；实测收帧不依赖它）\n"
        "  --summary-only        不逐帧打印，只每秒一行统计\n"
        "  --wait=<s>            预检等待首个数据帧（默认 3；**=0 则跳过预检**直接进主循环，\n"
        "                        用于电控比上位机晚上电的场合）\n"
        "  --help                显示本帮助\n"
        "\n"
        "退出码: 0 正常 / 1 预检失败或无硬件 / 2 参数错误 / 130 Ctrl+C\n"
        "\n"
        "安全: 两个 yaw 关节恒为「仅力矩模式 + 0 N·m」，fire 恒 0；任何退出路径先发零力矩。\n";
}

bool needValue(const std::string& a, const char* key, std::string& out) {
    const size_t n = std::strlen(key);
    if (a.compare(0, n, key) != 0) return false;
    out = a.substr(n);
    return true;
}

bool parseArgs(int argc, char** argv, Options& o, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        std::string v;
        if (a == "--help" || a == "-h") { o.help = true; }
        else if (a == "--list") { o.list = true; }
        else if (a == "--selftest") { o.selftest = true; }
        else if (a == "--no-send") { o.no_send = true; }
        else if (a == "--raw") { o.raw = true; }
        else if (a == "--imu") { o.print_imu = true; }
        else if (a == "--imu-heartbeat") { o.imu_heartbeat = true; }
        else if (a == "--summary-only") { o.summary_only = true; }
        else if (a == "--auto-aim") { o.auto_aim = true; }
        else if (needValue(a, "--pitch=", v)) { o.pitch = std::atof(v.c_str()); o.pitch_given = true; }
        else if (needValue(a, "--hz=", v)) { o.hz = std::atof(v.c_str()); }
        else if (needValue(a, "--dur=", v)) { o.dur = std::atof(v.c_str()); }
        else if (needValue(a, "--count=", v)) { o.count = std::atol(v.c_str()); }
        else if (needValue(a, "--every=", v)) { o.every = std::atol(v.c_str()); }
        else if (needValue(a, "--wait=", v)) { o.wait = std::atof(v.c_str()); }
        else { err = "未知参数: " + a; return false; }
    }
    if (o.hz <= 0.0 || o.hz > 1000.0) { err = "--hz 必须在 (0, 1000] 内"; return false; }
    if (o.every < 1) { err = "--every 必须 ≥ 1"; return false; }
    if (o.dur < 0.0) { err = "--dur 不能为负"; return false; }
    if (o.count < 0) { err = "--count 不能为负"; return false; }
    if (o.wait < 0.0) { err = "--wait 不能为负"; return false; }
    // ★ pitch 守卫（本构型 pitch 目标 raw 即弧度，给大了可能顶到机械限位）
    if (o.pitch_given && std::fabs(o.pitch) > kMaxPitchAbsRad) {
        err = "--pitch=" + std::to_string(o.pitch) + " 超过硬上限 "
              + std::to_string(kMaxPitchAbsRad) + " rad（≈28.6°）: 本工具只做链路自检，"
              "不应大幅驱动 pitch。确认要动请改用 tcbs_pitch_calibration";
        return false;
    }
    return true;
}

// ============================================================================
// 信号 / 安全退出
// ============================================================================
std::atomic<bool> g_stop{false};
std::atomic<int>  g_sigcount{0};

void onSignal(int) {
    g_stop = true;
    g_sigcount.fetch_add(1);
}

// 构造"无害帧": 两关节仅力矩 + 0 N·m（本文件头声明的安全不变量）
mcu::SendPacket makeSafePacket(const Options& o) {
    mcu::SendPacket p;
    p.auto_aim_enable = o.auto_aim ? 1 : 0;
    p.fire = 0;                                  // 恒 0
    p.pitch_target_angle = static_cast<float>(o.pitch);
    p.yaw_big_mode = mcu::YAW_MODE_TORQUE_ONLY;
    p.yaw_big_target_angle = 0.0;
    p.yaw_big_target_velocity = 0.0f;
    p.yaw_big_torque = 0.0f;
    p.yaw_small_mode = mcu::YAW_MODE_TORQUE_ONLY;
    p.yaw_small_target_angle = 0.0f;
    p.yaw_small_target_velocity = 0.0f;
    p.yaw_small_torque = 0.0f;
    return p;
}

// 编译期波特率 → 数值（SERIAL_BAUD_RATE 是 termios 的 B* 常量，其**数值不是波特率**，
//   例如 B115200 == 4098；直接 printf 会打印 4098，误导现场排查）。
int serialBaudRate() {
#if   SERIAL_BAUD_RATE == B9600
    return 9600;
#elif SERIAL_BAUD_RATE == B19200
    return 19200;
#elif SERIAL_BAUD_RATE == B38400
    return 38400;
#elif SERIAL_BAUD_RATE == B57600
    return 57600;
#elif SERIAL_BAUD_RATE == B115200
    return 115200;
#elif SERIAL_BAUD_RATE == B230400
    return 230400;
#elif SERIAL_BAUD_RATE == B460800
    return 460800;
#elif SERIAL_BAUD_RATE == B500000
    return 500000;
#elif SERIAL_BAUD_RATE == B921600
    return 921600;
#elif SERIAL_BAUD_RATE == B1000000
    return 1000000;
#elif SERIAL_BAUD_RATE == B1500000
    return 1500000;
#elif SERIAL_BAUD_RATE == B2000000
    return 2000000;
#elif SERIAL_BAUD_RATE == B3000000
    return 3000000;
#else
    return -1;   // 非标准/未知: 调用方打印原始宏值并提示"数值不是波特率"
#endif
}

void hexDump(const char* tag, const void* data, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(data);
    std::printf("  %s (%zu B):\n", tag, n);
    for (size_t i = 0; i < n; i += 16) {
        std::printf("    %04zx  ", i);
        for (size_t j = 0; j < 16; ++j) {
            if (i + j < n) std::printf("%02X ", b[i + j]); else std::printf("   ");
            if (j == 7) std::printf(" ");
        }
        std::printf(" |");
        for (size_t j = 0; j < 16 && i + j < n; ++j) {
            const uint8_t c = b[i + j];
            std::printf("%c", (c >= 32 && c < 127) ? static_cast<char>(c) : '.');
        }
        std::printf("|\n");
    }
}

// ============================================================================
// --list: 枚举串口 + 选择结果（无硬件也能跑）
// ============================================================================
int runList() {
    std::printf("==============================================================\n");
    std::printf(" 串口枚举（/dev/ttyACM*）\n");
    std::printf("==============================================================\n");
    // 直接调用 SerialProtocol 的 static 工具（与 initializeSerial 内部同一实现）
    const std::vector<std::string> ports = McuCommunication::findAvailableSerialPorts();
    if (ports.empty()) {
        std::printf("  ✗ 没有可打开的 /dev/ttyACM*\n");
        std::printf("    常见原因: ① 未插 USB；② 无权限（需在 dialout 组: groups | grep dialout）；\n");
        std::printf("              ③ 端口被其它进程占用（该口会 open 失败而被跳过）。\n");
    } else {
        std::printf("  找到 %zu 个可打开的口:\n\n", ports.size());
        std::printf("    %-16s %-26s %-6s %-6s %s\n",
                    "port", "iProduct", "MCU?", "IMU?", "说明");
        std::printf("    ---------------- -------------------------- ------ ------ ----\n");
        for (const std::string& p : ports) {
            std::string info = McuCommunication::getSerialProductInfo(p.substr(5));
            // 端口选择器的入参就是 iProduct；这里如实复现选择逻辑
            bool is_mcu = mcuPortSelector(info);
            bool is_imu = imuPortSelector(info);
            const char* note = "";
            if (!is_mcu && !is_imu)  note = "★ 两个选择器都不认（iProduct 不匹配）";
            if (is_imu)              note = "IMU 口（iProduct == \"AutoAim_IMU_Com\"）";
            std::printf("    %-16s %-26s %-6s %-6s %s\n",
                        p.c_str(), info.c_str(),
                        is_mcu ? "YES" : "-", is_imu ? "YES" : "-", note);
        }
        std::printf("\n  选择规则（include/tcbs/communication/Communications.hpp）:\n");
        std::printf("    mcuPortSelector: iProduct != \"AutoAim_IMU_Com\"  ⇒ 取**第一个**匹配的口\n");
        std::printf("    imuPortSelector: iProduct == \"AutoAim_IMU_Com\"  ⇒ 取**第一个**匹配的口\n");
        std::printf("  ⇒ 若两个 USB 设备的 iProduct 相同（比如都是默认值），MCU/IMU 会被分到**错的**口。\n");
    }
    const int baud = serialBaudRate();
    if (baud > 0) {
        std::printf("\n  编译期常量: 波特率 %d、MCU 前导 0x42 0x52 0x%02X、IMU 前导 0xA7 0xB6 0xC5\n",
                    baud, mcu::PROTOCOL_VERSION);
    } else {
        std::printf("\n  编译期常量: 波特率 SERIAL_BAUD_RATE=%d（非标准值，注意这是 termios 常量而非波特率）、"
                    "MCU 前导 0x42 0x52 0x%02X、IMU 前导 0xA7 0xB6 0xC5\n",
                    static_cast<int>(SERIAL_BAUD_RATE), mcu::PROTOCOL_VERSION);
    }
    std::printf("  （波特率可在 CMake 编译期用 -DSERIAL_BAUD_RATE=B921600 覆盖）\n");
    std::printf("==============================================================\n");
    return 0;
}

// ============================================================================
// --selftest: 纯软件自检（无硬件）
// ============================================================================
struct FieldExpect { const char* name; size_t offset; };

int runSelftest() {
    int fail = 0, total = 0;
    auto check = [&](bool ok, const char* name, const std::string& detail) {
        std::printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
        ++total;
        if (!ok) ++fail;
    };

    std::printf("==============================================================\n");
    std::printf(" 串口链路工具自检（纯软件，不需要硬件）\n");
    std::printf("==============================================================\n");

    // ── 1) 包布局与协议常量 ──
    std::printf("\n[1] 包布局 vs 协议常量\n");
    std::printf("    sizeof(mcu::SendPacket)    = %zu (期望 3+1+%u+1 = %u)\n",
                sizeof(mcu::SendPacket), 36u, 3u + 1u + 36u + 1u);
    std::printf("    sizeof(mcu::ReceivePacket) = %zu (期望 3+1+%u+1 = %u)\n",
                sizeof(mcu::ReceivePacket), 42u, 3u + 1u + 42u + 1u);
    check(sizeof(mcu::SendPacket) == 41, "SendPacket == 41 B",
          std::to_string(sizeof(mcu::SendPacket)) + " B");
    check(sizeof(mcu::ReceivePacket) == 47, "ReceivePacket == 47 B",
          std::to_string(sizeof(mcu::ReceivePacket)) + " B");
    check(mcu::SendPacket{}.data_size == 36, "SendPacket.data_size == 36",
          std::to_string(static_cast<int>(mcu::SendPacket{}.data_size)));
    check(mcu::PROTOCOL_VERSION == 0x03, "PROTOCOL_VERSION == 0x03",
          std::to_string(static_cast<int>(mcu::PROTOCOL_VERSION)));

    // ── 2) 关键字段偏移（与 Protocol.hpp 的 static_assert 同一张表；这里打印实际值供电控核对）──
    std::printf("\n[2] ReceivePacket 关键字段偏移（电控侧应逐字段核对，避免静默错位）\n");
    const FieldExpect expected[] = {
        {"yaw_big_angle",        offsetof(mcu::ReceivePacket, yaw_big_angle)},
        {"yaw_small_angle",      offsetof(mcu::ReceivePacket, yaw_small_angle)},
        {"chassis_imu_yaw",      offsetof(mcu::ReceivePacket, chassis_imu_yaw)},
        {"yaw_big_temperature",  offsetof(mcu::ReceivePacket, yaw_big_temperature)},
        {"mcu2_seq",             offsetof(mcu::ReceivePacket, mcu2_seq)},
        {"crc8",                 offsetof(mcu::ReceivePacket, crc8)},
    };
    const size_t want[] = {12, 24, 32, 43, 45, 46};
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        std::printf("    %-22s offset = %2zu (期望 %2zu)\n",
                    expected[i].name, expected[i].offset, want[i]);
        check(expected[i].offset == want[i],
              (std::string("offsetof(") + expected[i].name + ")").c_str(),
              std::to_string(expected[i].offset) + " vs " + std::to_string(want[i]));
    }

    // ── 3) CRC8 往返 + 单字节敏感性（SendPacket 的 CRC 只覆盖前 sizeof-1 字节）──
    std::printf("\n[3] CRC8 往返与敏感性\n");
    {
        mcu::SendPacket p{};
        const size_t n = sizeof(mcu::SendPacket) - 1;   // CRC 字段本身不参与
        const uint8_t crc_a = CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(&p), n);
        const uint8_t crc_b = CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(&p), n);
        check(crc_a == crc_b, "CRC8 对同一缓冲两次结果一致", "crc=0x" + std::to_string(crc_a));
        // 翻转一个参与运算的字节 ⇒ CRC 必须变（否则 CRC 形同虚设）
        uint8_t buf[64];
        std::memcpy(buf, &p, n);
        buf[n / 2] ^= 0x01;
        const uint8_t crc_c = CRC8_Check_Sum(buf, n);
        check(crc_c != crc_a, "翻转 1 bit 后 CRC8 改变（CRC 有效）",
              "0x" + std::to_string(crc_a) + " → 0x" + std::to_string(crc_c));
        // 长度敏感性
        check(CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(&p), n - 1) != crc_a,
              "长度变化后 CRC8 改变", "-");
    }

    // ── 4) 安全不变量: makeSafePacket 必须是"零力矩 + 仅力矩模式 + fire=0" ──
    std::printf("\n[4] 安全不变量（工具的核心承诺）\n");
    {
        Options o;
        o.auto_aim = false;
        const mcu::SendPacket p = makeSafePacket(o);
        check(p.yaw_big_mode == mcu::YAW_MODE_TORQUE_ONLY, "大 yaw 模式 = 仅力矩", "-");
        check(p.yaw_small_mode == mcu::YAW_MODE_TORQUE_ONLY, "小 yaw 模式 = 仅力矩", "-");
        check(p.yaw_big_torque == 0.0f && p.yaw_small_torque == 0.0f, "两关节力矩 = 0 N·m", "-");
        check(p.yaw_big_target_angle == 0.0 && p.yaw_big_target_velocity == 0.0f &&
              p.yaw_small_target_angle == 0.0f && p.yaw_small_target_velocity == 0.0f,
              "两关节目标角/角速度 = 0（虽然仅力矩模式下电控不理会）", "-");
        check(p.fire == 0, "fire = 0", "-");
        check(p.auto_aim_enable == 0, "默认 auto_aim_enable = 0（安全）", "-");
        check(p.pitch_target_angle == 0.0f, "默认 pitch 目标 = 0（不是原仓库的 10.0f）", "-");
        Options o2; o2.auto_aim = true;
        check(makeSafePacket(o2).auto_aim_enable == 1, "--auto-aim 时才置 1", "-");
    }

    // ── 5) --pitch 守卫 ──
    std::printf("\n[5] --pitch 守卫（本构型 raw 即弧度，给大了危险）\n");
    {
        auto try_pitch = [](const char* s) {
            Options o; std::string err;
            std::vector<std::string> argv_s = {"prog", std::string("--pitch=") + s};
            std::vector<char*> argv;
            for (auto& x : argv_s) argv.push_back(const_cast<char*>(x.c_str()));
            const bool ok = parseArgs(static_cast<int>(argv.size()), argv.data(), o, err);
            return ok;
        };
        check(try_pitch("0") == true,   "--pitch=0 接受", "-");
        check(try_pitch("0.3") == true, "--pitch=0.3 接受（≈17°）", "-");
        check(try_pitch("10") == false, "--pitch=10 拒绝（原仓库那个值，本构型=573°）", "-");
        check(try_pitch("-10") == false, "--pitch=-10 拒绝", "-");
        // 参数错误路径
        {
            Options o; std::string err;
            std::vector<std::string> argv_s = {"prog", "--hz=0"};
            std::vector<char*> argv;
            for (auto& x : argv_s) argv.push_back(const_cast<char*>(x.c_str()));
            check(!parseArgs(static_cast<int>(argv.size()), argv.data(), o, err), "--hz=0 拒绝", err);
        }
        {
            Options o; std::string err;
            std::vector<std::string> argv_s = {"prog", "--every=0"};
            std::vector<char*> argv;
            for (auto& x : argv_s) argv.push_back(const_cast<char*>(x.c_str()));
            check(!parseArgs(static_cast<int>(argv.size()), argv.data(), o, err), "--every=0 拒绝", err);
        }
    }

    std::printf("\n==============================================================\n");
    if (fail == 0) std::printf("✓ --selftest 全部通过（%d 项）\n", total);
    else           std::printf("✗ --selftest 有 %d 项失败\n", fail);
    std::printf("==============================================================\n");
    return fail == 0 ? 0 : 1;
}

// ============================================================================
// 收帧统计与打印
// ============================================================================
struct Counters {
    std::atomic<long> mcu_frames{0};
    std::atomic<long> imu_frames{0};
    std::atomic<long> mcu2_new{0};         // mcu2_seq 变化次数 = MCU2 那条低速链路的刷新次数
    std::atomic<long> printed{0};
    std::atomic<bool> mcu_seen{false};
    std::atomic<bool> imu_seen{false};
    std::atomic<int>  last_seq{-1};
    // 最后一次收到数据的时间（用 steady_clock 的计数表示，避免在回调里取锁）
    std::atomic<long long> last_mcu_ns{0};
    std::atomic<long long> last_imu_ns{0};
};

void printMcuPacket(const mcu::ReceivePacket& p, long idx, bool raw) {
    std::printf("\n[MCU #%ld] header=0x%02X 0x%02X ver=0x%02X data_size=%u\n",
                idx, p.frame_header1, p.frame_header2, p.protocol_version, p.data_size);
    std::printf("  bullet_velocity    : %.3f m/s\n", p.bullet_velocity);
    std::printf("  pitch_angle(raw)   : %+.5f rad (%+.3f°)  ★ 需经 McuDataPreprocessor 映射才是关节角\n",
                p.pitch_angle, p.pitch_angle * kRadToDeg);
    std::printf("  yaw_big_angle      : %+.6f rad (%+.3f°)   ← 电控侧多圈累计\n",
                p.yaw_big_angle, p.yaw_big_angle * kRadToDeg);
    std::printf("  yaw_big_omega      : %+.5f rad/s\n", p.yaw_big_omega);
    std::printf("  yaw_small_angle    : %+.6f rad (%+.3f°)   ← 相对大 yaw 的关节角（可信、实时）\n",
                p.yaw_small_angle, p.yaw_small_angle * kRadToDeg);
    std::printf("  yaw_small_omega    : %+.5f rad/s\n", p.yaw_small_omega);
    std::printf("  chassis_imu_yaw    : %+.5f rad (%+.3f°)   ← 底盘自身 yaw\n",
                p.chassis_imu_yaw, p.chassis_imu_yaw * kRadToDeg);
    std::printf("  chassis_imu_omega  : %+.5f rad/s\n", p.chassis_imu_omega);
    std::printf("  mark=%u color=%u auto_aim_switch=%u\n",
                p.mark, p.color, p.auto_aim_switch);
    std::printf("  temp: big=%u C small=%u C\n",
                p.yaw_big_temperature, p.yaw_small_temperature);
    std::printf("  mcu2_seq           : %u\n", p.mcu2_seq);
    std::printf("  crc8               : 0x%02X（已由 SerialProtocol 校验通过）\n", p.crc8);
    if (raw) hexDump("raw(ReceivePacket 布局 = 线格式)", &p, sizeof(p));
}

void printImuPacket(const imu::ReceivePacket& p, long idx, bool raw) {
    std::printf("\n[IMU #%ld] header=0x%02X 0x%02X 0x%02X data_size=%u\n",
                idx, p.frame_header1, p.frame_header2, p.frame_header3, p.data_size);
    std::printf("  gyro (本体系)      : gx=%+.5f gy=%+.5f gz=%+.5f rad/s\n", p.gx, p.gy, p.gz);
    std::printf("  accel (本体系)     : ax=%+.4f ay=%+.4f az=%+.4f m/s²  (|a|=%.4f)\n",
                p.ax, p.ay, p.az,
                std::sqrt(p.ax * p.ax + p.ay * p.ay + p.az * p.az));
    std::printf("  euler (ZXY, 世界系): yaw=%+.5f pitch=%+.5f roll=%+.5f rad\n",
                p.euler_yaw, p.euler_pitch, p.euler_roll);
    std::printf("    ⇒ yaw=%+.3f° pitch=%+.3f° roll=%+.3f°\n",
                p.euler_yaw * kRadToDeg, p.euler_pitch * kRadToDeg, p.euler_roll * kRadToDeg);
    std::printf("  dt_one_tenth_ms    : %u (%.2f ms)\n",
                p.dt_one_tenth_ms, p.dt_one_tenth_ms * 0.1);
    std::printf("  crc32              : 0x%08X（已校验通过）\n", p.crc32);
    if (raw) hexDump("raw(ReceivePacket 布局 = 线格式)", &p, sizeof(p));
}

}  // namespace

}  // namespace tcbs

// ============================================================================
// 入口
// ============================================================================
using namespace tcbs;

int main(int argc, char** argv) {
    Options opt;
    {
        std::string err;
        if (!parseArgs(argc, argv, opt, err)) {
            std::cerr << "参数错误: " << err << "\n\n";
            printUsage(argv[0]);
            return 2;
        }
    }
    if (opt.help) { printUsage(argv[0]); return 0; }
    if (opt.list) { return runList(); }
    if (opt.selftest) { return runSelftest(); }

    // ── 实车模式 ──
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    std::printf("==============================================================\n");
    std::printf(" tcbs_test_serial — 串口链路自检\n");
    std::printf("==============================================================\n");
    std::printf(" 包大小: SendPacket=%zu B, ReceivePacket=%zu B（协议 v0x%02X）\n",
                sizeof(mcu::SendPacket), sizeof(mcu::ReceivePacket),
                static_cast<int>(mcu::PROTOCOL_VERSION));
    std::printf(" 发送: %s", opt.no_send ? "【关闭】纯监听模式\n" : "");
    if (!opt.no_send) {
        std::printf("%.1f Hz", opt.hz);
        std::printf(" | auto_aim_enable=%d | fire=0", opt.auto_aim ? 1 : 0);
        std::printf(" | yaw 两关节: 仅力矩 + 0 N·m");
        std::printf(" | pitch=%.4f rad", opt.pitch);
        std::printf("\n");
    }
    std::printf(" 打印: every=%ld 帧%s%s%s\n", opt.every,
                opt.print_imu ? " + IMU" : "", opt.raw ? " + raw hex" : "",
                opt.summary_only ? " (summary-only)" : "");
    if (opt.dur > 0.0)   std::printf(" 时长: %.1f s\n", opt.dur);
    if (opt.count > 0)   std::printf(" 收到 %ld 帧 MCU 后退出\n", opt.count);
    std::printf(" Ctrl+C 结束（再按一次立即退出）。\n");
    std::printf("--------------------------------------------------------------\n");

    // 先列一下串口情况（含选择结果），现场第一眼就能看出"口对不对"
    (void)runList();
    std::printf("\n正在打开串口（MCU + IMU，自动重连由 SerialProtocol 负责）…\n");

    Counters cnt;
    std::mutex print_mtx;

    auto now_ns = []() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    // ★ 构造时 auto_start=false，成员就绪后再 startWorker（避免回调在读成员前触发）
    McuCommunication mcu_serial([&](const mcu::ReceivePacket& pkt) {
        const long n = cnt.mcu_frames.fetch_add(1) + 1;
        cnt.mcu_seen = true;
        cnt.last_mcu_ns = now_ns();
        const int seq = static_cast<int>(pkt.mcu2_seq);
        const int prev = cnt.last_seq.exchange(seq);
        if (prev != seq) cnt.mcu2_new.fetch_add(1);
        if (!opt.summary_only && (n % opt.every == 0)) {
            std::lock_guard<std::mutex> lk(print_mtx);
            printMcuPacket(pkt, n, opt.raw);
        }
    }, false);

    ImuCommunication imu_serial([&](const imu::ReceivePacket& pkt) {
        const long n = cnt.imu_frames.fetch_add(1) + 1;
        cnt.imu_seen = true;
        cnt.last_imu_ns = now_ns();
        if (opt.print_imu && !opt.summary_only && (n % opt.every == 0)) {
            std::lock_guard<std::mutex> lk(print_mtx);
            printImuPacket(pkt, n, opt.raw);
        }
    }, false);

    mcu_serial.startWorker();
    imu_serial.startWorker();

    std::printf(" MCU 串口: %s (open=%s)\n",
                mcu_serial.portName().empty() ? "<未选中>" : mcu_serial.portName().c_str(),
                mcu_serial.isOpen() ? "yes" : "no");
    std::printf(" IMU 串口: %s (open=%s)\n",
                imu_serial.portName().empty() ? "<未选中>" : imu_serial.portName().c_str(),
                imu_serial.isOpen() ? "yes" : "no");

    // ── 预检: 等待首个数据帧（--wait=0 ⇒ **跳过预检**，直接进主循环）──
    //   跳过预检的用途: 电控比上位机晚上电（上位机先跑、等电控上来）、
    //   或只想先看我们自己发帧是否正常。此时主循环会用统计行里的 ⚠ 静默告警提示收不到数据。
    bool precheck_failed = false;
    if (opt.wait > 0.0) {
        std::printf("\n---- 预检: 等待 MCU / IMU 首帧（最多 %.1f s）----\n", opt.wait);
        {
            const auto t0 = std::chrono::steady_clock::now();
            while (!g_stop) {
                if (cnt.mcu_seen.load() || cnt.imu_seen.load()) break;
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > opt.wait) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        const bool mcu_ok = cnt.mcu_seen.load();
        const bool imu_ok = cnt.imu_seen.load();
        std::printf("  MCU: %s   IMU: %s\n", mcu_ok ? "有数据 ✓" : "无数据 ✗",
                    imu_ok ? "有数据 ✓" : "无数据 ✗");
        precheck_failed = (!mcu_ok && !imu_ok && !g_stop);
    } else {
        std::printf("\n---- 跳过预检（--wait=0）: 直接进入主循环 ----\n");
    }

    if (precheck_failed) {
        std::printf("\n✗ 预检失败（%.1f s 内无任何可用数据）\n", opt.wait);
        std::printf("  诊断（按顺序查）:\n");
        std::printf("    ① 端口: 上面 --list 的表里有没有 YES？没有 ⇒ iProduct 不匹配或没插好\n");
        std::printf("    ② 权限: 需在 dialout 组（groups | grep dialout; 改完要重新登录）\n");
        std::printf("    ③ 占用: 别的进程（自瞄程序 / 上一轮工具）占着口 ⇒ open 失败会被跳过\n");
        std::printf("    ④ 供电 / 波特率: 电控是否上电；波特率是否与电控一致（编译期 SERIAL_BAUD_RATE）\n");
        std::printf("    ⑤ 协议: 电控发的是不是 v0x%02X（前导 0x42 0x52 0x%02X）？前导不对会被丢\n",
                    static_cast<int>(mcu::PROTOCOL_VERSION), static_cast<int>(mcu::PROTOCOL_VERSION));
        std::printf("  退出前先发零力矩（无论 --no-send，除纯监听模式外）…\n");
        if (!opt.no_send) {
            mcu::SendPacket safe = makeSafePacket(opt);
            int sent = 0;
            for (int i = 0; i < kZeroTorqueFrames; ++i) { if (mcu_serial.sendData(safe)) ++sent; }
            std::printf("    已尝试发送 %d/%d 帧零力矩（成功 %d 帧）\n", kZeroTorqueFrames, kZeroTorqueFrames, sent);
        }
        mcu_serial.stopWorker();
        imu_serial.stopWorker();
        std::printf(" 已关闭串口句柄，退出（退出码 1）。\n");
        return 1;
    }

    std::printf("\n---- 开始链路自检（Ctrl+C 结束）----\n");

    // ── 主循环: 发送 + 1 Hz 统计 ──
    const auto period = std::chrono::duration<double>(1.0 / opt.hz);
    auto next_send = std::chrono::steady_clock::now();
    auto next_stat = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    const auto t_start = std::chrono::steady_clock::now();
    long tx_ok = 0, tx_fail = 0;
    long last_mcu = 0, last_imu = 0, last_mcu2 = 0;
    int  stat_sec = 0;

    while (!g_stop) {
        const auto now = std::chrono::steady_clock::now();

        // 发送（原仓库语义: 固定节拍发同一包）
        if (!opt.no_send && now >= next_send) {
            mcu::SendPacket safe = makeSafePacket(opt);
            if (mcu_serial.sendData(safe)) ++tx_ok; else ++tx_fail;
            if (opt.imu_heartbeat) { imu::SendPacket hb; imu_serial.sendData(hb); }
            // ★ 不用 now += period: 落后时直接对齐到"下一个未来时刻"，避免追赶式连发刷屏
            next_send += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            if (next_send < now) next_send = now;
        }

        // 退出条件
        if (opt.count > 0 && cnt.mcu_frames.load() >= opt.count) break;
        if (opt.dur > 0.0 &&
            std::chrono::duration<double>(now - t_start).count() >= opt.dur) break;

        // 1 Hz 统计
        if (now >= next_stat) {
            const long m = cnt.mcu_frames.load(), i = cnt.imu_frames.load();
            const long q = cnt.mcu2_new.load();
            ++stat_sec;
            const double age_m = cnt.last_mcu_ns.load() ? (now_ns() - cnt.last_mcu_ns.load()) * 1e-9 : -1.0;
            const double age_i = cnt.last_imu_ns.load() ? (now_ns() - cnt.last_imu_ns.load()) * 1e-9 : -1.0;
            std::printf("[t=%3ds] MCU %4ld 帧/s (Σ%ld)  IMU %4ld 帧/s (Σ%ld)  "
                        "MCU2 新样本 %3ld/s  TX ok=%ld fail=%ld",
                        stat_sec, m - last_mcu, m, i - last_imu, i, q - last_mcu2, tx_ok, tx_fail);
            if (!opt.no_send) std::printf("  (%.1f Hz)", opt.hz);
            std::printf("\n");
            if (m == 0) std::printf("         ⚠ 还没收到任何 MCU 帧\n");
            else if (age_m > kStaleWarnS) std::printf("         ⚠ MCU 已静默 %.2f s（链路断了？看门狗会把目标清零）\n", age_m);
            if (i == 0) std::printf("         ⚠ 还没收到任何 IMU 帧\n");
            else if (age_i > kStaleWarnS) std::printf("         ⚠ IMU 已静默 %.2f s\n", age_i);
            if (q == 0 && m > 0)
                std::printf("         ⚠ mcu2_seq 一次都没变 ⇒ MCU1↔MCU2 链路没通（大 yaw 与底盘 IMU 是保持值）\n");
            last_mcu = m; last_imu = i; last_mcu2 = q;
            next_stat += std::chrono::seconds(1);
            if (next_stat < now) next_stat = now + std::chrono::seconds(1);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const bool interrupted = (g_sigcount.load() > 0);
    std::printf("\n---- 收尾 ----\n");
    std::printf(" 统计: MCU %ld 帧, IMU %ld 帧, MCU2 新样本 %ld, TX 成功 %ld / 失败 %ld\n",
                cnt.mcu_frames.load(), cnt.imu_frames.load(), cnt.mcu2_new.load(), tx_ok, tx_fail);

    if (opt.no_send) {
        std::printf(" 纯监听模式（--no-send）⇒ 未发送任何帧，直接关句柄。\n");
    } else {
        std::printf(" 退出前先发零力矩（yaw 两关节: 仅力矩 + 0 N·m）…\n");
        mcu::SendPacket safe = makeSafePacket(opt);
        int sent = 0;
        for (int i = 0; i < kZeroTorqueFrames; ++i) {
            if (mcu_serial.sendData(safe)) ++sent;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::printf("  已发 %d/%d 帧零力矩\n", sent, kZeroTorqueFrames);
        if (sent < kZeroTorqueFrames)
            std::printf("  ⚠ 零力矩帧**未全部发出**（串口不可用 / 已断开）⇒ 电控若仍在自瞄状态，"
                        "请人工确认/断电，或重新上电后再操作。\n");
    }

    mcu_serial.stopWorker();
    imu_serial.stopWorker();
    std::printf(" 已关闭串口句柄，退出（退出码 %d）。\n", interrupted ? 130 : 0);
    return interrupted ? 130 : 0;
}
