// ============================================================================
// pitch_calibration.cpp — pitch 轴映射标定（两段线性拟合）
//
// ★★★ 构型前提（用之前必须先读这一条）★★★
//   本工具假定 **IMU 临时装在头上**（IMU 与 head 固连、位于 pitch 之后，
//   即 YawStateEstimator::Config::ImuLocation::ON_HEAD）—— 与原仓库
//   `TorqueController/src/pitch_calibration.cpp` 的构型完全一致。
//   只有在这种构型下，`imu.euler_pitch` **就是 pitch 关节角**（物理角真值）。
//   若 IMU 仍固定在大 yaw 转子上（本工程默认构型 ON_BIG_YAW），
//   `imu.euler_pitch` 是"大 yaw 平台 + 小 yaw + pitch"的**合成倾角**，
//   **不是** pitch 关节角 ⇒ 本工具标出来的 4 个数完全无效，
//   照抄进 McuDataPreprocessor 会把 pitch 目标角映射到错误值（可能顶到机械限位）。
//   程序启动时也会把这条警告再打一遍（大字/多行）。
//
// 算法（与原仓库**逐段一致**）: 采样 target_angle(pitch_target_angle)、
//   imu_euler_pitch、mcu_pitch_angle 三者的关系，做两级线性拟合
//     Fit1: mcu_pitch_angle(电控原始值) → imu_euler_pitch(物理关节角)
//           ⇒ recv_pitch_scale/offset （物理角 = scale·raw + offset）
//     Fit2: imu_euler_pitch(物理关节角) → pitch_target_angle(下发的原始值)
//           ⇒ send_pitch_scale/offset （raw_cmd = scale·物理角 + offset）
//
// 流程（= 原仓库流程）:
//   Step 1 在扫描范围两端各测一次，得到 y_left / y_right 与相关性符号；
//   Step 2 二分查找 y 中心值（y_mid）对应的 target_angle ⇒ target_center（8 次迭代）；
//   Step 3 取 y 范围 0.1 / 0.9 分位对应的 target_angle（各 8 次二分迭代）；
//   Step 4 在 [target_0.1, target_0.9] 内**从两端向中间交替**采样（默认 20 点）；
//   Step 5 两级线性拟合 + 组合公式 + R² / 残差 RMS / 样本统计；
//   Step 6 打印可直接替换 LinearParams 的四行。
//
// 使用 0.1 和 0.9 分位值而非直接使用 min/max 的原因（原仓库注释，照抄）:
//   物理系统在行程端点附近通常存在非线性（如机械限位、电机力矩饱和、
//   传感器边缘效应等），端点处的测量值噪声也更大。取 0.1~0.9 分位范围
//   可以剔除两端各 10% 的不可靠数据，聚焦在线性度最好的中间区域进行
//   拟合，得到的斜率和截距更能代表系统的真实线性特性，避免端点异常值
//   拉偏回归结果。
//
// 安全（本仓库约定，见 tools/identify_params.cpp / python/scripts/collect_sysid.py）:
//   · yaw **两个关节一律"仅力矩模式 + 零力矩"**（YAW_MODE_TORQUE_ONLY，τ_big = τ_small = 0），
//     本工具绝不驱动 yaw；
//   · pitch 只有目标角通道（协议里没有独立力矩通道），因此只发 pitch_target_angle；
//   · 任何退出路径（正常 / 报错 / Ctrl+C）都先连发若干帧"零力矩"再关句柄；
//   · 稳定等待期间以 100 Hz 重发同一目标（喂电控看门狗 + 目标零阶保持）。
//
// 用法:
//   ./build/tcbs_pitch_calibration                      # 实车（需硬件 + IMU 在头上）
//   ./build/tcbs_pitch_calibration --sim                # 无硬件自检（虚拟 pitch 台架）
//   ./build/tcbs_pitch_calibration --selftest           # 纯数学自检（拟合核心）
//   ./build/tcbs_pitch_calibration --help
//   ./build/tcbs_pitch_calibration --points=20 --min=-0.30 --max=0.30 --dwell=1.0 --out=x.txt
//
// 扫描范围（★ 与原仓库默认不同）:
//   默认 --min=-0.30 / --max=+0.30，单位**弧度**；跨度 > 1.20 rad（≈69°）且未加
//   --force-range 时**拒绝开跑**（参数错误，退出码 1，不碰串口）。原仓库默认 -10/30 是
//   旧电控原始单位下的经验值；本构型 pitch 映射为恒等占位（raw 即弧度），照抄会去要
//   -573°/+1719°，可能顶到机械限位。--sim / --selftest 不受此限。
//
// 与原仓库的差异（只列适配点，算法/流程不变）:
//   1) `namespace tcbs` / `#include "tcbs/..."`；串口用本仓库的 tcbs::McuCommunication、
//      tcbs::ImuCommunication（构造时 auto_start=false，成员就绪后再 startWorker，
//      避免回调在读成员前触发）；
//   2) 发送包按本仓库协议 v0x03 的**双关节**字段（yaw_big_*/yaw_small_* 各自的
//      模式位 + 目标角/角速度 + 力矩）填写，语义与原仓库的单 yaw 字段一致（仅力矩 + 0）；
//   3) Ctrl+C 不再 exit(0)，改为置中断标志 → 主流程收尾（零力矩 + 关句柄）→ 退出码 130；
//   4) 拟合内部用 double（原用 float），协议字段该是 float 的就是 float；仅数值精度提升；
//   5) 新增 --sim（虚拟台架自检）/ --selftest（纯数学自检）/ --points / --min / --max /
//      --dwell / --wait / --out / --park / --seed，以及"无数据/链路过期"的显式报错；
//   6) 新增打印 R² 之外的残差 RMS、参与/未参与拟合的样本数（原仓库只给 R²）；
//   7) 稳定等待期间以 100 Hz 重发同一目标（原仓库"发一帧 → sleep 1000 ms"，
//      在带看门狗的电控上会中途清零目标）—— 只改发送节拍，不改采样流程与判据；
//   8) 实车默认扫描范围改为弧度（-0.30/+0.30）并新增跨度保护（> 1.20 rad 需 --force-range）。
// ============================================================================
#include "tcbs/communication/Communications.hpp"
#include "tcbs/communication/Protocol.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace tcbs {

namespace {

// ============================================================================
// 基础常量
// ============================================================================
constexpr double kPi = 3.14159265358979323846;

// 与原仓库一致的默认参数（硬编码改为 CLI，默认值不变）
// ★ 实车默认扫描范围（**本构型的单位是弧度**）:
//   原仓库那两个数（−10 / 30）是**旧仓库原始单位**下的经验值；本仓库协议里 pitch 目标默认
//   恒等映射 ⇒ raw 即弧度，沿用 −10/30 会去要 −573°/+1719°，属于危险操作。
//   这里给一个保守的弧度默认（−0.30 ~ +0.30 rad ≈ ∓17°）；上实车前请按本车 pitch 实际行程
//   与电控单位显式给 --min/--max。跨度明显离谱时工具会**拒绝开跑**（见 kMaxSpanRad）。
constexpr double kDefaultTargetMin = -0.30;
constexpr double kDefaultTargetMax =  0.30;
// 扫描跨度上限（rad）: 超过它且未显式 --force-range ⇒ 拒绝开跑（防单位搞错 / 防超程）
constexpr double kMaxSpanRad = 1.20;   // ≈ 69°
constexpr int    kDefaultFitPoints = 20;    // 拟合采样点数（Step 4）
constexpr double kDefaultDwellS    = 1.0;   // 每个目标点的稳定等待（原仓库 1000 ms）
constexpr double kGapS             = 0.5;   // 采样后间隔（原仓库 500 ms）

// 二分查找迭代次数（原仓库 Step 2 / Step 3 均为 8）
constexpr int kBinaryIter = 8;

// 安全：退出前连发多少帧"零力矩"（与 python/scripts/collect_sysid.py 的 20 帧一致）
constexpr int    kZeroTorqueFrames = 20;
constexpr double kFramePeriodS     = 0.01;   // 100 Hz（保持目标 / 零力矩的发送节拍）

// 链路新鲜度阈值：超过这个时间没有新帧就认为链路断了（显式报错，不静默用旧值）
constexpr double kLinkStaleS = 1.0;

// --sim 虚拟台架的默认原始扫描范围（**仅在未显式给 --min/--max 时使用**）:
//   台架真值 recv_pitch_scale = 1.0 ⇒ 原始单位 ≈ 弧度，取 −0.35 ~ +0.55 rad
//   （≈ −20° ~ +31.5°，与真实 pitch 行程同量级），避免用默认 −10/30 跑出台架的荒谬量程。
constexpr double kSimRawMin = -0.35;
constexpr double kSimRawMax =  0.55;

// 台架真值（--sim 用；"例如 recv: 1.0/−0.02、send: 1.03/0.05"）
constexpr double kSimRecvScale  = 1.0;
constexpr double kSimRecvOffset = -0.02;
constexpr double kSimSendScale  = 1.03;
constexpr double kSimSendOffset = 0.05;

// 台架端点非线性: 两端各 10% 行程内加二次项，端点幅值 = kEndQuad·(0.1·span)²
constexpr double kSimEndWindow = 0.10;
constexpr double kSimEndQuad   = 5.0;

// 台架测量噪声（每点独立高斯）: 电控原始值 / IMU pitch
constexpr double kSimNoiseRaw = 0.002;
constexpr double kSimNoiseImu = 0.002;

// --sim 断言容差: 随**本次配置**自适应（推导与实测见 runSim 打印）
//   tol = max(kSimTolFloor, kSimTolPerSigma·SE + kSimTolSystematic)
//   · SE = σ_res/(σ_x·√n): 由台架噪声与本次拟合跨度/点数算出 —— 配置越差（点更少、
//     行程更窄）容差越松，这本身就是"该配置精度不够"的量化说明，而不是把判据写死；
//   · 端点残余系统项: 两端对称的二次失真把实测端点值抬高，使 0.9 分位边界**内移**进
//     非线性区，少量非线性样本进入拟合 ⇒ send_pitch_scale 系统性偏低；默认台架实测
//     均值 ≈ −3.9e-3（−0.38%），取 5e-3 覆盖。这是该裁剪法的固有残余（不改算法）。
constexpr double kSimTolPerSigma   = 4.0;
constexpr double kSimTolSystematic = 5e-3;
constexpr double kSimTolFloor      = 0.010;
constexpr double kSimMinR2         = 0.999;
// --selftest 用的固定容差（= 默认台架配置下的容差 4σ+系统项 ≈ 0.015）
constexpr double kSimTolParam = 0.015;

// ============================================================================
// 中断处理: 信号处理函数只置标志，收尾一律回到主流程（保证"先零力矩再关句柄"）
// ============================================================================
std::atomic<bool> g_interrupted{false};
void onSignal(int) { g_interrupted.store(true); }

// ============================================================================
// 数据结构与拟合（与原仓库同构，只把 float 提到 double）
// ============================================================================
struct DataPoint {
    double target_angle = 0.0;   // mcu::SendPacket.pitch_target_angle（下发的原始值）
    double imu_pitch    = 0.0;   // imu::ReceivePacket.euler_pitch（IMU 在头上时 = 物理关节角）
    double mcu_pitch    = 0.0;   // mcu::ReceivePacket.pitch_angle（电控上报的原始值）
    bool   valid        = false;
};

struct LinearFit {
    double slope     = 0.0;
    double intercept = 0.0;
    double r_squared = 0.0;
    double rms       = 0.0;      // 残差 RMS（新增: 原仓库只给 R²）
    double max_abs_res = 0.0;    // 最大绝对残差（新增）
    size_t n         = 0;        // 参与拟合的样本数
    bool   ok        = false;
};

// 一元线性最小二乘（与原仓库 fitLinear 逐步一致: 正规方程 + R²），额外返回残差统计
LinearFit fitLinear(const std::vector<double>& xs, const std::vector<double>& ys) {
    LinearFit result;
    const size_t n = xs.size();
    result.n = n;
    if (n < 2 || ys.size() != n) return result;
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = xs[i], y = ys[i];
        sx  += x;  sy  += y;
        sxy += x * y;
        sx2 += x * x;
        sy2 += y * y;
    }
    const double denom = static_cast<double>(n) * sx2 - sx * sx;
    if (std::fabs(denom) < 1e-9) return result;          // 退化（x 全同）: 不产生 NaN
    result.slope     = (static_cast<double>(n) * sxy - sx * sy) / denom;
    result.intercept = (sy - result.slope * sx) / static_cast<double>(n);

    const double my = sy / static_cast<double>(n);
    double ssr = 0, sst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double pred = result.slope * xs[i] + result.intercept;
        ssr += (ys[i] - pred) * (ys[i] - pred);
        sst += (ys[i] - my)   * (ys[i] - my);
    }
    result.r_squared = (sst > 1e-9) ? 1.0 - ssr / sst : 1.0;
    result.rms       = std::sqrt(ssr / static_cast<double>(n));
    result.max_abs_res = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double pred = result.slope * xs[i] + result.intercept;
        result.max_abs_res = std::max(result.max_abs_res, std::fabs(ys[i] - pred));
    }
    result.ok = true;
    return result;
}

inline double lerp(double a, double b, double t) { return a + t * (b - a); }

// ============================================================================
// 台架抽象: 硬件（真实串口）与虚拟（--sim）走同一套标定流程
// ============================================================================
class PitchRig {
public:
    virtual ~PitchRig() = default;

    // 发送 pitch **原始**目标值（yaw 恒为"仅力矩模式 + 零力矩"）
    virtual bool sendPitchTarget(double raw_target, std::string& err) = 0;
    // 保持当前目标并等待 seconds（硬件: 100 Hz 重发同一帧；仿真: 立即返回）
    virtual bool hold(double seconds, std::string& err) = 0;
    // 读取最新反馈: 电控原始 pitch 与 IMU pitch
    virtual bool readFeedback(double& mcu_pitch, double& imu_pitch, std::string& err) = 0;
    // 链路自检（未收到任何帧 / 数据过期 / 串口未打开）
    virtual bool checkLink(std::string& err) = 0;
    // 当前（最后一个）下发的 pitch 原始目标 —— 退出时用它做默认停位，避免突跳
    virtual double currentTarget() const = 0;
    // 退出: 先连发零力矩（yaw 两关节），再关句柄。返回是否真的发出了至少一帧
    // （串口不可用/已断开时返回 false —— 调用方必须把这件事明确告诉操作者）
    virtual bool emergencyZeroTorque(double hold_pitch_raw) = 0;
    virtual void close() = 0;
    virtual const char* name() const = 0;
};

// ============================================================================
// HardwareRig — 真实串口（tcbs::McuCommunication / tcbs::ImuCommunication）
// ============================================================================
class HardwareRig final : public PitchRig {
public:
    HardwareRig()
        // auto_start = false: 成员（互斥量等）构造完成后再 startWorker，
        // 否则串口线程可能在互斥量构造前回调（与 RobotCommunication 的做法一致）
        : mcu_serial_([this](const mcu::ReceivePacket& p) { onMcu(p); }, false)
        , imu_serial_([this](const imu::ReceivePacket& p) { onImu(p); }, false)
    {
        mcu_serial_.startWorker();
        imu_serial_.startWorker();
    }

    ~HardwareRig() override { close(); }

    const char* name() const override { return "硬件（真实串口）"; }
    double currentTarget() const override { return current_target_; }

    bool sendPitchTarget(double raw_target, std::string& err) override {
        current_target_ = raw_target;
        return rawSend(raw_target, err);
    }

    bool hold(double seconds, std::string& err) override {
        // 原仓库是"发一帧 → sleep"；这里在等待期间按 100 Hz 重发同一目标：
        //   · 电控看门狗（~50 ms 无帧清零力矩）不会误触发；
        //   · 目标零阶保持，测量的是"稳定后"的稳态值（与原意一致）。
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(seconds));
        while (std::chrono::steady_clock::now() < deadline) {
            if (g_interrupted.load()) { err = "用户中断（Ctrl+C）"; return false; }
            if (!rawSend(current_target_, err)) return false;
            std::this_thread::sleep_for(
                std::chrono::duration<double>(kFramePeriodS));
        }
        return true;
    }

    bool readFeedback(double& mcu_pitch, double& imu_pitch, std::string& err) override {
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            if (!has_mcu_) { err = "尚未收到任何 MCU（电控）帧"; return false; }
            mcu_pitch = static_cast<double>(latest_mcu_.pitch_angle);
        }
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            if (!has_imu_) { err = "尚未收到任何 IMU 帧"; return false; }
            imu_pitch = latest_imu_.euler_pitch;
        }
        return true;
    }

    bool checkLink(std::string& err) override {
        const auto now = std::chrono::steady_clock::now();
        bool has_mcu = false, has_imu = false;
        double age_mcu = 0.0, age_imu = 0.0;
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            has_mcu = has_mcu_;
            age_mcu = std::chrono::duration<double>(now - last_mcu_rx_).count();
        }
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            has_imu = has_imu_;
            age_imu = std::chrono::duration<double>(now - last_imu_rx_).count();
        }
        if (!has_mcu && !has_imu) {
            err = "未收到 MCU 与 IMU 的任何数据（找不到串口 / 未上电 / 端口被占用）";
            return false;
        }
        if (!has_mcu) { err = "未收到 MCU（电控）数据"; return false; }
        if (!has_imu) { err = "未收到 IMU 数据"; return false; }
        if (age_mcu > kLinkStaleS) {
            err = "MCU 数据已过期 " + std::to_string(age_mcu) + " s（超过 " +
                  std::to_string(kLinkStaleS) + " s）";
            return false;
        }
        if (age_imu > kLinkStaleS) {
            err = "IMU 数据已过期 " + std::to_string(age_imu) + " s（超过 " +
                  std::to_string(kLinkStaleS) + " s）";
            return false;
        }
        return true;
    }

    bool emergencyZeroTorque(double hold_pitch_raw) override {
        // pitch 在协议里**没有**独立力矩通道（只有目标角），所以"零力矩"作用于两个 yaw 关节；
        // pitch 保持给定原始目标（默认 = 最后一个目标）——比强行回到 0 安全（不产生突跳）。
        std::string err;
        bool sent_any = false;
        for (int i = 0; i < kZeroTorqueFrames; ++i) {
            if (!rawSend(hold_pitch_raw, err)) break;   // 串口已不可用: 不再重试
            sent_any = true;
            std::this_thread::sleep_for(std::chrono::duration<double>(kFramePeriodS));
        }
        if (!sent_any) {
            std::cout << "⚠ 零力矩帧**未能发出**（串口不可用）: " << err
                      << "\n  ⇒ 电控若仍在自瞄状态，请人工确认/断电，或重新上电后再操作。\n";
        }
        return sent_any;
    }

    void close() override {
        mcu_serial_.stopWorker();
        imu_serial_.stopWorker();
        // fd 由 SerialProtocol 析构时关闭（此时回调线程已 join）
    }

private:
    bool rawSend(double pitch_raw, std::string& err) {
        mcu::SendPacket pkt;
        pkt.auto_aim_enable    = 1;          // 与旧脚本一致（与电控手动开关相与）
        pkt.fire               = 0;
        pkt.pitch_target_angle = static_cast<float>(pitch_raw);   // ★ 原始值，不经预处理器
        // ── 两个 yaw 关节: 仅力矩模式 + 零力矩（本工具绝不驱动 yaw）──
        pkt.yaw_big_mode            = mcu::YAW_MODE_TORQUE_ONLY;
        pkt.yaw_big_target_angle    = 0.0;
        pkt.yaw_big_target_velocity = 0.0f;
        pkt.yaw_big_torque          = 0.0f;
        pkt.yaw_small_mode            = mcu::YAW_MODE_TORQUE_ONLY;
        pkt.yaw_small_target_angle    = 0.0f;
        pkt.yaw_small_target_velocity = 0.0f;
        pkt.yaw_small_torque          = 0.0f;
        if (!mcu_serial_.sendData(pkt)) {
            err = "串口写入失败（电控串口未打开 / 已断开）";
            return false;
        }
        return true;
    }

    void onMcu(const mcu::ReceivePacket& pkt) {
        std::lock_guard<std::mutex> lock(mcu_mutex_);
        latest_mcu_  = pkt;
        has_mcu_     = true;
        last_mcu_rx_ = std::chrono::steady_clock::now();
    }

    void onImu(const imu::ReceivePacket& pkt) {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        latest_imu_  = pkt;
        has_imu_     = true;
        last_imu_rx_ = std::chrono::steady_clock::now();
    }

    McuCommunication mcu_serial_;
    ImuCommunication imu_serial_;

    std::mutex mcu_mutex_;
    mcu::ReceivePacket latest_mcu_{};
    bool   has_mcu_ = false;
    std::chrono::steady_clock::time_point last_mcu_rx_{};

    std::mutex imu_mutex_;
    imu::ReceivePacket latest_imu_{};
    bool   has_imu_ = false;
    std::chrono::steady_clock::time_point last_imu_rx_{};

    double current_target_ = 0.0;
};

// ============================================================================
// SimRig — 虚拟 pitch 台架（--sim，无硬件、不依赖串口）
//
// 台架模型（"物理"含义写清楚，便于判断断言是否合理）:
//   1) 电控内环: 收到原始目标 r 后，实际到达的**编码器原始值**
//        e = k_track·r + c_track + nl(r) + noise_raw
//      其中 k_track / c_track 由台架真值反推:
//        要求 r = send_s·θ + send_o 且 θ = recv_s·e + recv_o
//        ⇒ k_track = 1/(recv_s·send_s),  c_track = (−send_o/send_s − recv_o)/recv_s
//      （物理含义: 电控内环本身有增益/偏置误差 ⇒ "该下发多少" 与 "编码器读多少"
//        不是简单互逆 ⇒ **两段拟合必须分别做**，这正是本工具存在的意义。）
//   2) nl(r): 两端各 10% 行程内的二次非线性（模拟机械限位/力矩饱和/边缘效应）
//   3) IMU: θ = recv_s·e_ideal + recv_o + noise_imu（IMU 在头上 = 直接测关节角）
//   4) e_ideal 不含测量噪声（真实位置）⇒ 编码器读出值 = e_ideal + noise_raw
// ============================================================================
class SimRig final : public PitchRig {
public:
    SimRig(double raw_min, double raw_max, unsigned seed)
        : raw_min_(raw_min), raw_max_(raw_max), rng_(seed)
    {
        recv_scale_  = kSimRecvScale;   recv_offset_ = kSimRecvOffset;
        send_scale_  = kSimSendScale;   send_offset_ = kSimSendOffset;
        // 由真值反推"电控内环"的跟踪增益/偏置
        k_track_ = 1.0 / (recv_scale_ * send_scale_);
        c_track_ = (-send_offset_ / send_scale_ - recv_offset_) / recv_scale_;
    }

    const char* name() const override { return "虚拟 pitch 台架（--sim）"; }
    double currentTarget() const override { return cmd_; }

    // 台架真值（供 --sim 断言用）
    double truthRecvScale()  const { return recv_scale_; }
    double truthRecvOffset() const { return recv_offset_; }
    double truthSendScale()  const { return send_scale_; }
    double truthSendOffset() const { return send_offset_; }
    double kTrack() const { return k_track_; }
    double cTrack() const { return c_track_; }

    // 端点非线性幅值（原始单位）: 两端各 kSimEndWindow 行程处为 0，端点为 kSimEndQuad·w²
    double endpointNonlinearityAmplitude() const {
        const double span = raw_max_ - raw_min_;
        const double w = kSimEndWindow * span;
        return kSimEndQuad * w * w;
    }

    bool sendPitchTarget(double raw_target, std::string& err) override {
        (void)err;
        cmd_ = raw_target;
        ++sent_frames_;
        return true;
    }

    bool hold(double, std::string&) override {
        if (g_interrupted.load()) return false;   // 仿真下不等（几秒内跑完）
        return true;
    }

    bool readFeedback(double& mcu_pitch, double& imu_pitch, std::string& err) override {
        (void)err;
        // ① 位置 → 编码器原始值（含端点非线性 + 编码器噪声）
        const double e_ideal = k_track_ * cmd_ + c_track_ + endpointNonlinearity(cmd_);
        // ② 编码器原始值 → 物理关节角（IMU 在头上的 euler_pitch）
        const double theta   = recv_scale_ * e_ideal + recv_offset_;
        mcu_pitch = e_ideal + raw_noise_(rng_);
        imu_pitch = theta   + imu_noise_(rng_);
        return true;
    }

    bool checkLink(std::string&) override { return true; }

    bool emergencyZeroTorque(double) override { return true; }   // 仿真: 无需零力矩
    void close() override {}

    size_t sentFrames() const { return sent_frames_; }

private:
    double endpointNonlinearity(double r) const {
        const double span = raw_max_ - raw_min_;
        if (!(span > 0.0)) return 0.0;
        const double rel = (r - raw_min_) / span;
        double d = 0.0;
        if (rel < kSimEndWindow) {
            const double x = kSimEndWindow - rel;
            d += kSimEndQuad * x * x;
        }
        if (rel > 1.0 - kSimEndWindow) {
            const double x = rel - (1.0 - kSimEndWindow);
            d += kSimEndQuad * x * x;
        }
        return d;
    }

    double raw_min_ = 0.0, raw_max_ = 0.0;
    double cmd_ = 0.0;
    double k_track_ = 1.0, c_track_ = 0.0;
    double recv_scale_ = 1.0, recv_offset_ = 0.0;
    double send_scale_ = 1.0, send_offset_ = 0.0;
    std::mt19937 rng_;
    std::normal_distribution<double> raw_noise_{0.0, kSimNoiseRaw};
    std::normal_distribution<double> imu_noise_{0.0, kSimNoiseImu};
    size_t sent_frames_ = 0;
};

// ============================================================================
// 标定结果
// ============================================================================
struct CalibrationResult {
    LinearFit fit1, fit2;
    double target_min = 0.0, target_max = 0.0;
    double target_center = 0.0, target_01 = 0.0, target_09 = 0.0;
    double y_left = 0.0, y_right = 0.0, y_min = 0.0, y_max = 0.0;
    double y_01 = 0.0, y_09 = 0.0;
    double target_lo = 0.0, target_hi = 0.0;
    int    fit_points = 0;
    size_t scanned_samples = 0;    // 全流程采过的样本数（含端点/二分）
    std::vector<DataPoint> fit_samples;
    std::vector<DataPoint> all_samples;
};

// ============================================================================
// PitchCalibrator — 与原仓库同名同流程
// ============================================================================
class PitchCalibrator {
public:
    PitchCalibrator(PitchRig& rig, double target_min, double target_max,
                    int fit_points, double dwell_s)
        : rig_(rig)
        , target_min_(target_min), target_max_(target_max)
        , fit_points_(fit_points), dwell_(dwell_s) {}

    bool run(CalibrationResult& out, std::string& err, bool& interrupted);

private:
    bool sample(double target_angle, DataPoint& out, std::string& err, bool& interrupted);
    bool binarySearchTarget(double target_y, double lo, double hi, double sign,
                            int max_iter, double& out, std::string& err, bool& interrupted);

    PitchRig& rig_;
    double target_min_ = 0.0, target_max_ = 0.0;
    int    fit_points_ = 20;
    double dwell_ = 1.0;
    std::vector<DataPoint> all_samples_;
};

// ── 单点采样（= 原仓库 sample()）: 发目标 → 稳定等待 → 读 MCU + IMU ──
bool PitchCalibrator::sample(double target_angle, DataPoint& out, std::string& err,
                             bool& interrupted) {
    interrupted = false;
    if (!rig_.checkLink(err)) return false;
    if (!rig_.sendPitchTarget(target_angle, err)) return false;
    if (!rig_.hold(dwell_, err)) { interrupted = g_interrupted.load(); return false; }

    out = DataPoint{};
    out.target_angle = target_angle;
    if (!rig_.readFeedback(out.mcu_pitch, out.imu_pitch, err)) return false;
    out.valid = true;
    all_samples_.push_back(out);

    if (!rig_.hold(kGapS, err)) { interrupted = g_interrupted.load(); return false; }
    return true;
}

// ── 二分查找（= 原仓库 binarySearchTarget()）: 找 target 使 mcu_pitch ≈ target_y ──
bool PitchCalibrator::binarySearchTarget(double target_y, double lo, double hi, double sign,
                                         int max_iter, double& out, std::string& err,
                                         bool& interrupted) {
    for (int i = 0; i < max_iter; ++i) {
        const double mid = (lo + hi) * 0.5;
        DataPoint dp;
        if (!sample(mid, dp, err, interrupted)) return false;
        const double y = dp.mcu_pitch;
        std::cout << "    [" << i + 1 << "/" << max_iter << "] target="
                  << std::fixed << std::setprecision(4) << mid
                  << " -> mcu=" << y << " (target_y=" << target_y << ")\n";
        if (sign * (y - target_y) < 0) lo = mid;
        else                           hi = mid;
    }
    out = (lo + hi) * 0.5;
    return true;
}

// ── 主流程（Step 1 ~ Step 6，与原仓库一致 + 新增统计）──
bool PitchCalibrator::run(CalibrationResult& out, std::string& err, bool& interrupted) {
    interrupted = false;
    out = CalibrationResult{};
    out.target_min = target_min_;
    out.target_max = target_max_;
    out.fit_points = fit_points_;
    all_samples_.clear();

    // ── Step 1: 测量两端点 ──
    std::cout << "\n========== Step 1: 测量两端点 ==========\n";
    std::cout << "target_min = " << target_min_ << ", target_max = " << target_max_ << "\n";

    DataPoint p_left{}, p_right{};
    if (!sample(target_min_, p_left, err, interrupted)) return false;
    if (!sample(target_max_, p_right, err, interrupted)) return false;

    const double y_left  = p_left.mcu_pitch;
    const double y_right = p_right.mcu_pitch;
    out.y_left = y_left;
    out.y_right = y_right;

    std::cout << "y_left  = " << y_left  << "  (at target=" << target_min_ << ")\n";
    std::cout << "y_right = " << y_right << "  (at target=" << target_max_ << ")\n";

    const double sign = (y_right > y_left) ? 1.0 : -1.0;
    std::cout << "相关性: " << (sign > 0 ? "正相关" : "负相关") << "\n";

    // ── Step 2: 二分查找 y 中心值对应的 target_angle ──
    const double y_mid = (y_left + y_right) * 0.5;
    std::cout << "\n========== Step 2: 二分查找 y 中心 ==========\n";
    std::cout << "y_mid = " << y_mid << "\n";

    double target_center = 0.0;
    if (!binarySearchTarget(y_mid, target_min_, target_max_, sign, kBinaryIter,
                            target_center, err, interrupted)) return false;
    out.target_center = target_center;
    std::cout << "target_center = " << target_center << "\n";

    // ── Step 3: 二分查找 0.1 / 0.9 分位值对应的 target_angle ──
    const double y_min = std::min(y_left, y_right);
    const double y_max = std::max(y_left, y_right);
    const double y_01 = lerp(y_min, y_max, 0.1);
    const double y_09 = lerp(y_min, y_max, 0.9);
    out.y_min = y_min; out.y_max = y_max; out.y_01 = y_01; out.y_09 = y_09;

    std::cout << "\n========== Step 3: 查找 0.1 / 0.9 分位 ==========\n";
    std::cout << "y 范围: [" << y_min << ", " << y_max << "]\n";
    std::cout << "y_0.1 = " << y_01 << ", y_0.9 = " << y_09 << "\n";

    double target_01 = 0.0, target_09 = 0.0;
    if (y_left < y_right) {
        std::cout << "\n--- 向 target_min 方向搜索 target_0.1 ---\n";
        if (!binarySearchTarget(y_01, target_min_, target_center, sign, kBinaryIter,
                                target_01, err, interrupted)) return false;
        std::cout << "--- 向 target_max 方向搜索 target_0.9 ---\n";
        if (!binarySearchTarget(y_09, target_center, target_max_, sign, kBinaryIter,
                                target_09, err, interrupted)) return false;
    } else {
        std::cout << "\n--- 向 target_max 方向搜索 target_0.1 ---\n";
        if (!binarySearchTarget(y_01, target_center, target_max_, sign, kBinaryIter,
                                target_01, err, interrupted)) return false;
        std::cout << "--- 向 target_min 方向搜索 target_0.9 ---\n";
        if (!binarySearchTarget(y_09, target_min_, target_center, sign, kBinaryIter,
                                target_09, err, interrupted)) return false;
    }
    out.target_01 = target_01; out.target_09 = target_09;
    std::cout << "target_0.1 = " << target_01 << ", target_0.9 = " << target_09 << "\n";

    // ── Step 4: 在 [target_01, target_09] 范围内交替采样 ──
    const double target_lo = std::min(target_01, target_09);
    const double target_hi = std::max(target_01, target_09);
    out.target_lo = target_lo; out.target_hi = target_hi;

    std::cout << "\n========== Step 4: 交替采样拟合数据 ==========\n";
    std::cout << "采样范围: [" << target_lo << ", " << target_hi << "]\n";

    // 构建交替测量顺序：从两端向中间交替取值，避免单向漂移引入系统误差（原仓库同）
    std::vector<double> target_order;
    target_order.reserve(static_cast<size_t>(fit_points_));
    {
        int lo = 0, hi = fit_points_ - 1;
        while (lo <= hi) {
            const double t_lo = static_cast<double>(lo) / std::max(1, fit_points_ - 1);
            target_order.push_back(lerp(target_lo, target_hi, t_lo));
            lo++;
            if (lo > hi) break;
            const double t_hi = static_cast<double>(hi) / std::max(1, fit_points_ - 1);
            target_order.push_back(lerp(target_lo, target_hi, t_hi));
            hi--;
        }
    }

    std::vector<DataPoint> samples;
    samples.reserve(static_cast<size_t>(fit_points_));
    for (int i = 0; i < fit_points_; ++i) {
        DataPoint dp;
        if (!sample(target_order[static_cast<size_t>(i)], dp, err, interrupted)) return false;
        samples.push_back(dp);
        std::cout << "  [" << std::setw(2) << i + 1 << "/" << fit_points_ << "]"
                  << " target=" << std::fixed << std::setprecision(4) << dp.target_angle
                  << " -> imu=" << dp.imu_pitch << " mcu=" << dp.mcu_pitch << "\n";
    }
    out.fit_samples = samples;
    out.all_samples = all_samples_;
    out.scanned_samples = all_samples_.size();

    // ── Step 5: 两级线性拟合 ──
    std::cout << "\n========== Step 5: 线性拟合结果 ==========\n";

    std::vector<double> target_vec, imu_vec, mcu_vec;
    target_vec.reserve(samples.size());
    imu_vec.reserve(samples.size());
    mcu_vec.reserve(samples.size());
    for (const auto& dp : samples) {
        target_vec.push_back(dp.target_angle);
        imu_vec.push_back(dp.imu_pitch);
        mcu_vec.push_back(dp.mcu_pitch);
    }

    // Fit1: mcu_pitch_angle → imu_euler_pitch
    const LinearFit fit1 = fitLinear(mcu_vec, imu_vec);
    out.fit1 = fit1;

    std::cout << "\n--- Fit1: mcu_pitch_angle → imu_euler_pitch ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "斜率 (slope)      : " << fit1.slope << "\n";
    std::cout << "截距 (intercept)  : " << fit1.intercept << "\n";
    std::cout << "R²                : " << fit1.r_squared << "\n";
    std::cout << "拟合公式: imu_euler_pitch = " << fit1.slope
              << " * mcu_pitch_angle + " << fit1.intercept << "\n";

    // Fit2: imu_euler_pitch → pitch_target_angle
    const LinearFit fit2 = fitLinear(imu_vec, target_vec);
    out.fit2 = fit2;

    std::cout << "\n--- Fit2: imu_euler_pitch → pitch_target_angle ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "斜率 (slope)      : " << fit2.slope << "\n";
    std::cout << "截距 (intercept)  : " << fit2.intercept << "\n";
    std::cout << "R²                : " << fit2.r_squared << "\n";
    std::cout << "拟合公式: pitch_target_angle = " << fit2.slope
              << " * imu_euler_pitch + " << fit2.intercept << "\n";

    // 组合公式: 由 mcu_pitch_angle 直接推算 pitch_target_angle
    std::cout << "\n--- 组合公式: mcu_pitch_angle → pitch_target_angle ---\n";
    const double comb_slope     = fit2.slope * fit1.slope;
    const double comb_intercept = fit2.slope * fit1.intercept + fit2.intercept;
    std::cout << "pitch_target_angle = " << comb_slope << " * mcu_pitch_angle + "
              << comb_intercept << "\n";

    // y 极值（MCU pitch_angle）取自 Step 3 已计算的 y_min / y_max
    const double target_at_y_min = (std::fabs(comb_slope) > 1e-9)
                                       ? comb_slope * y_min + comb_intercept : target_min_;
    const double target_at_y_max = (std::fabs(comb_slope) > 1e-9)
                                       ? comb_slope * y_max + comb_intercept : target_max_;

    std::cout << std::setprecision(4);
    std::cout << "\nMCU pitch_angle 最小值 : " << y_min
              << "  (pitch_target_angle=" << target_at_y_min << ")\n";
    std::cout << "MCU pitch_angle 最大值 : " << y_max
              << "  (pitch_target_angle=" << target_at_y_max << ")\n";

    // ── Step 5b: 拟合质量与样本统计（新增: 原仓库只给 R²）──
    const size_t excluded = out.scanned_samples - samples.size();
    std::cout << "\n--- 拟合质量与样本统计 ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Fit1  残差 RMS    : " << fit1.rms << " (rad) = "
              << fit1.rms * 180.0 / kPi << " (°)   最大 |残差| = " << fit1.max_abs_res << "\n";
    std::cout << "Fit2  残差 RMS    : " << fit2.rms << " (raw) = "
              << fit2.rms * 180.0 / kPi << " (°)   最大 |残差| = " << fit2.max_abs_res << "\n";
    std::cout << "参与拟合的样本数  : " << samples.size()
              << "（Fit1 / Fit2 使用同一批样本）\n";
    std::cout << "未参与拟合的样本数: " << excluded
              << "（Step 1 两端端点 2 + Step 2 中心二分 " << kBinaryIter
              << " + Step 3 分位二分 " << 2 * kBinaryIter << "）\n";
    std::cout << "★ 两端端点未参与拟合: 拟合区间 = [target_0.1, target_0.9] = ["
              << std::setprecision(4) << target_lo << ", " << target_hi << "]"
              << "，占原始扫描范围 [" << target_min_ << ", " << target_max_ << "] 的 "
              << std::setprecision(1)
              << 100.0 * std::fabs(target_hi - target_lo) /
                     std::max(1e-12, std::fabs(target_max_ - target_min_))
              << "%；两端各 10% 行程（含两个极值端点本身）被 0.1/0.9 分位裁剪剔除，\n"
              << "  以避开端点非线性/力矩饱和/边缘噪声。\n";
    if (!fit1.ok || !fit2.ok) {
        err = "拟合失败（样本退化: x 全部相同或样本数不足）";
        return false;
    }

    // ── Step 6: 输出可直接替换 LinearParams 的四行（不自动写回代码）──
    std::cout << "\n========== Step 6: LinearParams 默认值替换片段 ==========\n";
    std::cout << "以下 4 行可整体替换 include/tcbs/communication/McuDataPreprocessor.h 中\n";
    std::cout << "struct LinearParams 的 pitch 部分（缩进与注释与当前代码一致）：\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "        double send_pitch_scale  = " << std::setw(10) << fit2.slope
              << ";   // 关节角 → 电控 pitch 目标值\n";
    std::cout << "        double send_pitch_offset = " << std::setw(10) << fit2.intercept << ";\n";
    std::cout << "        double recv_pitch_scale  = " << std::setw(10) << fit1.slope
              << ";   // 电控原始 pitch 值 → 关节角\n";
    std::cout << "        double recv_pitch_offset = " << std::setw(10) << fit1.intercept << ";\n";

    std::cout << "\n========================================\n";
    std::cout << "标定完成。\n";
    return true;
}

// ============================================================================
// 命令行
// ============================================================================
struct Options {
    bool   help = false;
    bool   sim = false;
    bool   selftest = false;
    int    points = kDefaultFitPoints;
    double min_raw = kDefaultTargetMin;
    double max_raw = kDefaultTargetMax;
    bool   range_given = false;     // 是否显式给了 --min/--max（--sim 用它决定台架范围）
    bool   force_range = false;     // --force-range: 跳过跨度安全上限检查
    double dwell = kDefaultDwellS;
    double wait_s = 3.0;            // 实车预检: 等待首帧的时间
    bool   park_given = false;
    double park = 0.0;              // 退出时把 pitch 停到该原始值（默认保持最后目标）
    std::string out;                // 非空则把结果写成文本
    unsigned seed = 12345;          // --sim 随机种子
};

void usage(const char* prog) {
    std::cout <<
        "用法: " << prog << " [选项]\n"
        "\n"
        "pitch 轴映射标定（两段线性拟合，与原仓库 TorqueController 的 pitch_calibration 同法）:\n"
        "  Fit1: mcu.pitch_angle(电控原始值)  → imu.euler_pitch(物理关节角)  ⇒ recv_pitch_scale/offset\n"
        "  Fit2: imu.euler_pitch(物理关节角)  → pitch_target_angle(下发原始值) ⇒ send_pitch_scale/offset\n"
        "  裁剪: 先二分找出 MCU 值 0.1/0.9 分位对应的目标角，只在该区间内采样拟合\n"
        "        （两端各 10% 行程的端点样本不参与拟合，避开端点非线性/饱和/噪声）\n"
        "  安全: yaw 两个关节一律「仅力矩模式 + 零力矩」，本工具不驱动 yaw；\n"
        "        任何退出路径都先连发零力矩帧再关句柄。\n"
        "\n"
        "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n"
        "★ 前提: IMU 必须**临时装在头上**（与 head 固连，pitch 之后 = YawStateEstimator::\n"
        "★       Config::ImuLocation::ON_HEAD）—— 与原仓库构型一致。\n"
        "★ 若 IMU 仍在大 yaw 转子上（本工程默认构型 ON_BIG_YAW），imu.euler_pitch 是\n"
        "★ 「大 yaw + 小 yaw + pitch」的合成倾角，**不是** pitch 关节角 ⇒ 标定结果无效，\n"
        "★ 照抄进 McuDataPreprocessor 会把 pitch 目标角映射到错误值（可能顶到机械限位）。\n"
        "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n"
        "\n"
        "选项:\n"
        "  --sim                 无硬件自检: 用内置虚拟 pitch 台架（已知真值 + 端点非线性 + 噪声）\n"
        "                        跑完整流程，并断言恢复出的 4 个参数落在容差内\n"
        "  --selftest            纯数学自检: 只验拟合核心（精确直线/噪声/退化输入），不扫描\n"
        "  --points=<n>          拟合采样点数（默认 " << kDefaultFitPoints << "，与原仓库一致）\n"
        "  --min=<raw> --max=<raw>  pitch 原始目标的扫描范围（默认 " << kDefaultTargetMin
        << " / " << kDefaultTargetMax << "，单位=**弧度**）\n"
        "                            注: 原仓库的 -10/30 是**旧单位**经验值；本构型默认恒等映射，\n"
        "                            请按本车实际单位与行程显式给出\n"
        "  --force-range            跳过「跨度 > 1.2 rad」的安全检查（确认单位/行程无误后使用）\n"
        "                        ★ 单位是**电控原始单位**（本构型默认映射为恒等 ⇒ raw 即弧度）；\n"
        "                          请先确认单位与 pitch 机械行程余量再上实车\n"
        "  --dwell=<s>           每个目标点的稳定等待（默认 " << kDefaultDwellS
        << " s，与原仓库 1000 ms 一致；采样后另有 " << kGapS << " s 间隔；仿真下为 0）\n"
        "  --wait=<s>            实车预检等待首帧的时间（默认 " << 3.0 << " s，超时即报错退出）\n"
        "  --park=<raw>          退出时把 pitch 目标停到该原始值（默认: 保持最后一个目标）\n"
        "  --out=<path>          把结果（四行 + 统计）写成文本（默认只打印）\n"
        "  --seed=<n>            --sim 的随机种子（默认 12345）\n"
        "  --help                显示本帮助\n"
        "\n"
        "示例:\n"
        "  " << prog << " --sim                       # 无硬件自检（几秒）\n"
        "  " << prog << " --selftest                  # 拟合核心自检\n"
        "  " << prog << " --points=20 --min=-0.35 --max=0.55 --dwell=1.0 --out=pitch_calib.txt\n";
}

bool parseArgs(int argc, char** argv, Options& o, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* key, std::string& v) -> bool {
            const size_t n = std::strlen(key);
            if (a.size() > n && a.compare(0, n, key) == 0) { v = a.substr(n); return true; }
            return false;
        };
        std::string v;
        if (a == "--help" || a == "-h") {
            o.help = true;
        } else if (a == "--sim") {
            o.sim = true;
        } else if (a == "--selftest") {
            o.selftest = true;
        } else if (value("--points=", v)) {
            o.points = std::atoi(v.c_str());
            if (o.points < 4 || o.points > 200) { err = "--points 需在 4..200"; return false; }
        } else if (value("--min=", v)) {
            o.min_raw = std::atof(v.c_str()); o.range_given = true;
        } else if (value("--max=", v)) {
            o.max_raw = std::atof(v.c_str()); o.range_given = true;
        } else if (value("--dwell=", v)) {
            o.dwell = std::atof(v.c_str());
            if (!(o.dwell >= 0.0)) { err = "--dwell 必须 ≥ 0"; return false; }
        } else if (value("--wait=", v)) {
            o.wait_s = std::atof(v.c_str());
            if (!(o.wait_s >= 0.0)) { err = "--wait 必须 ≥ 0"; return false; }
        } else if (value("--park=", v)) {
            o.park = std::atof(v.c_str()); o.park_given = true;
        } else if (a == "--force-range") {
            o.force_range = true;
        } else if (value("--out=", v)) {
            o.out = v;
        } else if (value("--seed=", v)) {
            o.seed = static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 10));
        } else {
            err = "未知选项 " + a;
            return false;
        }
    }
    if (o.max_raw <= o.min_raw) { err = "--max 必须大于 --min"; return false; }
    // ★ 扫描跨度安全检查（实车）: 超过上限且未显式 --force-range ⇒ 拒绝开跑。
    //   动机: 原仓库默认 -10/30 是**旧单位**经验值；本构型 pitch 目标默认恒等映射（raw 即弧度），
    //   沿用会把目标角放大成 −573°/+1719°，可能把 pitch 顶到机械限位。--sim 不受此限。
    if (!o.sim) {
        const double span = o.max_raw - o.min_raw;
        if (span > kMaxSpanRad && !o.force_range) {
            err = "pitch 扫描跨度 " + std::to_string(span) + " 超过安全上限 "
                  + std::to_string(kMaxSpanRad) + "（≈69°）: 请确认本车 pitch 的单位（本构型恒等映射 ⇒ 弧度）"
                  "与机械行程，确认后加 --force-range；或先小范围试探 --min=-0.1 --max=0.1 --points=5";
            return false;
        }
    }
    return true;
}

// ============================================================================
// 启动横幅 + 构型警告（醒目）
// ============================================================================
void printBanner(const Options& o, const char* mode) {
    std::cout <<
        "Pitch 轴标定程序（tcbs_pitch_calibration）\n"
        "运行模式: " << mode << "\n"
        "target_angle 范围: [" << o.min_raw << ", " << o.max_raw << "]\n"
        "拟合采样点数: " << o.points << "\n"
        "按 Ctrl+C 可随时中断（会先发零力矩再退出）\n";

    std::cout <<
        "\n"
        "╔════════════════════════════════════════════════════════════════════════════╗\n"
        "║  ★★ 构型警告: 本工具假定 IMU **临时装在头上**（ON_HEAD，与 head 固连）★★  ║\n"
        "╚════════════════════════════════════════════════════════════════════════════╝\n"
        "  · 只有 IMU 在头上（pitch 之后）时，imu.euler_pitch 才**就是 pitch 关节角**，\n"
        "    Fit1 / Fit2 才有物理意义 —— 这与原仓库 TorqueController 的构型一致。\n"
        "  · 若 IMU 仍固定在大 yaw 转子上（本工程默认构型 ON_BIG_YAW），\n"
        "    imu.euler_pitch = 「大 yaw 平台 + 小 yaw + pitch」的合成倾角，**不是** pitch 关节角：\n"
        "    标定出的 recv_pitch_*/send_pitch_* **完全无效**；照抄进 McuDataPreprocessor 会把\n"
        "    pitch 目标角放大/缩小到错误值，可能把 pitch 顶到机械限位（危险）。\n"
        "  · 标定前请确认: ① IMU 已临时装到头上并拧紧；② 大 yaw / 小 yaw 静止；\n"
        "    ③ pitch 行程两端有余量（本工具不做限位，超程由电控硬限位保护）。\n";
    std::cout <<
        "  · 安全: yaw 两个关节 = 仅力矩模式 + 零力矩（本工具不驱动 yaw）；pitch 只发目标角。\n";

    const bool hardware = (std::strcmp(mode, "实车（真实串口）") == 0);
    if (hardware || o.range_given) {
        std::cout <<
            "\n注意（扫描范围）: 默认 --min=" << kDefaultTargetMin << " / --max="
                  << kDefaultTargetMax
                  << " 沿用的是**旧仓库原始单位**下的经验值。\n"
            "  本工具按「电控原始单位」发送 pitch 目标（本构型默认映射为恒等 ⇒ raw 即弧度），请确认:\n"
            "    · 电控上报/接收的 pitch 原始值单位与量程（弧度 / 度 / 计数 …）；\n"
            "    · pitch 机械行程两端余量（本工具不做限位，超程由电控硬限位保护）；\n"
            "    · 单位不确定时，先用小范围试探: --min=-0.1 --max=0.1 --points=5，看反馈是否合理。\n";
    }
    std::cout <<
        "\n安全: yaw 两个关节 = 仅力矩模式 + 零力矩（本工具不驱动 yaw）；pitch 只发目标角；\n"
        "      任何退出路径（正常 / 报错 / Ctrl+C）都先连发零力矩帧再关句柄。\n";
    if (!hardware) {
        std::cout << "[" << mode << "] 不打开串口、不依赖硬件、无副作用，秒级跑完。\n";
    }
}

// ============================================================================
// --out 文本
// ============================================================================
bool writeOutFile(const std::string& path, const Options& o, const CalibrationResult& res,
                  std::string& err) {
    std::ofstream f(path);
    if (!f) { err = "无法写入 " + path; return false; }
    f << "# ============================================================================\n"
         "# tcbs_pitch_calibration 结果 —— pitch 轴映射（两段线性拟合）\n"
         "# ★ 前提: IMU 临时装在头上（YawStateEstimator::Config::ImuLocation::ON_HEAD）\n"
         "#   若 IMU 在大 yaw 转子上，下面的 4 个数无效。\n";
    if (o.sim) {
        f << "# ⚠ 本文件由 --sim 虚拟台架生成（自检用，**不是**实车标定值）。\n";
    }
    f << "# 扫描: target ∈ [" << res.target_min << ", " << res.target_max
      << "], 拟合点数 = " << o.points << "\n"
         "# 拟合区间（0.1~0.9 分位裁剪后的实际采样范围）= ["
      << res.target_lo << ", " << res.target_hi << "]\n";
    f << std::fixed << std::setprecision(6)
      << "# Fit1 (mcu_pitch_angle → imu_euler_pitch): R²=" << res.fit1.r_squared
      << ", 残差 RMS=" << res.fit1.rms << ", n=" << res.fit1.n << "\n"
      << "# Fit2 (imu_euler_pitch → pitch_target_angle): R²=" << res.fit2.r_squared
      << ", 残差 RMS=" << res.fit2.rms << ", n=" << res.fit2.n << "\n"
      << "# 参与拟合样本 = " << res.fit_samples.size()
      << "，未参与拟合样本 = " << (res.scanned_samples - res.fit_samples.size())
      << "（含两端端点，端点非线性/饱和/噪声样本被剔除）\n"
         "# ----------------------------------------------------------------------------\n"
         "# 以下 4 行可直接替换 include/tcbs/communication/McuDataPreprocessor.h\n"
         "# 中 struct LinearParams 的 pitch 部分:\n\n"
      << "        double send_pitch_scale  = " << std::setw(10) << res.fit2.slope
      << ";   // 关节角 → 电控 pitch 目标值\n"
      << "        double send_pitch_offset = " << std::setw(10) << res.fit2.intercept << ";\n"
      << "        double recv_pitch_scale  = " << std::setw(10) << res.fit1.slope
      << ";   // 电控原始 pitch 值 → 关节角\n"
      << "        double recv_pitch_offset = " << std::setw(10) << res.fit1.intercept << ";\n";
    return true;
}

// ============================================================================
// --sim: 虚拟台架自检
// ============================================================================
struct SimAssertion {
    std::string name;
    double truth = 0.0, got = 0.0, tol = 0.0;
    bool   pass = false;
};

int runSim(const Options& opt) {
    const double raw_min = opt.range_given ? opt.min_raw : kSimRawMin;
    const double raw_max = opt.range_given ? opt.max_raw : kSimRawMax;
    const double dwell   = 0.0;   // 仿真: 不等

    SimRig rig(raw_min, raw_max, opt.seed);
    std::cout << std::fixed;
    std::cout << "\n========== --sim 虚拟 pitch 台架自检（无硬件，不依赖串口）==========\n";
    std::cout << std::setprecision(6);
    std::cout << "台架真值（写入台架、待恢复）:\n"
              << "  recv_pitch_scale  = " << rig.truthRecvScale()
              << "    recv_pitch_offset = " << rig.truthRecvOffset()
              << "    （电控原始值 → 关节角）\n"
              << "  send_pitch_scale  = " << rig.truthSendScale()
              << "    send_pitch_offset = " << rig.truthSendOffset()
              << "    （关节角 → 下发原始值）\n";
    std::cout << "台架内部（由上面 4 个数反推，等效「电控内环」的增益/偏置误差，物理含义: 编码器值与\n"
                 "  下发值不是简单互逆 ⇒ 两段拟合必须分别做）:\n"
              << "  e = k_track·r + c_track + nl(r)，k_track = " << rig.kTrack()
              << "（=1/(recv_s·send_s)）, c_track = " << rig.cTrack() << "\n"
              << "  θ = recv_s·e + recv_o（拟合区间内严格线性）\n";
    std::cout << std::setprecision(6)
              << "端点非线性 nl(r): 两端各 " << kSimEndWindow * 100.0
              << "% 行程内加二次项，端点幅值 = " << rig.endpointNonlinearityAmplitude()
              << " 原始单位（≈" << rig.endpointNonlinearityAmplitude() * 180.0 / kPi << "°）\n"
              << "测量噪声: σ(电控原始值) = " << kSimNoiseRaw
              << "，σ(IMU pitch) = " << kSimNoiseImu << " rad（每点独立）\n";
    std::cout << std::setprecision(4)
              << "台架扫描范围: [" << raw_min << ", " << raw_max << "]（原始单位"
              << (opt.range_given ? "，来自 --min/--max" : "，--min/--max 未显式给出 ⇒ 用台架可行范围")
              << "）\n"
              << "扫描点数: " << opt.points << "，稳定等待: " << dwell << " s（仿真下不等）\n"
              << "随机种子: " << opt.seed << "\n";

    // ── 跑完整流程（Step 1 ~ Step 6 与实车完全相同）──
    PitchCalibrator calib(rig, raw_min, raw_max, opt.points, dwell);
    CalibrationResult res;
    std::string err;
    bool interrupted = false;
    if (!calib.run(res, err, interrupted)) {
        std::cout << "\n✗ --sim 流程失败: " << err << "\n";
        return 1;
    }

    // ── 断言: 恢复出的 4 个参数 vs 台架真值 ──
    // 容差: 由本次配置（拟合跨度、点数）与台架噪声算出，见上方常量注释
    const double span_fit  = std::fabs(res.target_hi - res.target_lo);
    const double sigma_res = std::sqrt(kSimNoiseRaw * kSimNoiseRaw + kSimNoiseImu * kSimNoiseImu);
    const double sigma_x   = span_fit / std::sqrt(12.0);
    const double n_fit     = static_cast<double>(res.fit_samples.size());
    const double se_slope  = (sigma_x > 1e-12 && n_fit > 0.0)
                                 ? sigma_res / (sigma_x * std::sqrt(n_fit)) : 1.0;
    const double tol = std::max(kSimTolFloor, kSimTolPerSigma * se_slope + kSimTolSystematic);

    std::vector<SimAssertion> items = {
        {"recv_pitch_scale",  rig.truthRecvScale(),  res.fit1.slope,     tol, false},
        {"recv_pitch_offset", rig.truthRecvOffset(), res.fit1.intercept, tol, false},
        {"send_pitch_scale",  rig.truthSendScale(),  res.fit2.slope,     tol, false},
        {"send_pitch_offset", rig.truthSendOffset(), res.fit2.intercept, tol, false},
    };

    std::cout << "\n========== --sim 自检结论 ==========\n";
    std::cout << std::left << std::setw(20) << "参数" << std::right
              << std::setw(14) << "台架真值" << std::setw(14) << "恢复值"
              << std::setw(14) << "误差" << std::setw(12) << "容差" << "   判定\n";
    bool all_pass = true;
    for (auto& it : items) {
        it.pass = std::fabs(it.got - it.truth) <= it.tol;
        all_pass = all_pass && it.pass;
        std::cout << std::left << std::setw(20) << it.name << std::right << std::fixed
                  << std::setprecision(6)
                  << std::setw(14) << it.truth << std::setw(14) << it.got
                  << std::setw(14) << (it.got - it.truth) << std::setw(12) << it.tol
                  << "   " << (it.pass ? "✓" : "✗") << "\n";
    }
    const bool r2_pass = (res.fit1.r_squared >= kSimMinR2) && (res.fit2.r_squared >= kSimMinR2);
    all_pass = all_pass && r2_pass;
    std::cout << std::setprecision(6)
              << "\n拟合质量: Fit1 R² = " << res.fit1.r_squared
              << "，残差 RMS = " << res.fit1.rms
              << "；Fit2 R² = " << res.fit2.r_squared
              << "，残差 RMS = " << res.fit2.rms << "\n"
              << "R² 判据: 均须 ≥ " << kSimMinR2 << " ⇒ " << (r2_pass ? "✓" : "✗") << "\n";

    std::cout << std::setprecision(6)
              << "\n容差依据 tol = max(" << kSimTolFloor << ", "
              << kSimTolPerSigma << "·SE + " << kSimTolSystematic << ") = "
              << std::setprecision(6) << tol << "（两项均为可推导量，不是拍脑袋）:\n"
                 "  ① 噪声项: 每点叠加 σ_raw = "
              << kSimNoiseRaw << "（原始单位）与 σ_imu = " << kSimNoiseImu
              << " rad 的独立测量噪声；\n"
                 "     本次拟合区间跨度 " << std::setprecision(3) << span_fit
              << " 原始单位、n = " << res.fit_samples.size()
              << " ⇒ σ_res = √(σ_raw²+σ_imu²) = " << std::setprecision(4) << sigma_res
              << "、σ_x = span/√12 = " << sigma_x
              << "\n     ⇒ SE(slope) = σ_res/(σ_x·√n) = " << se_slope
              << " ⇒ " << std::setprecision(3) << kSimTolPerSigma << "σ = "
              << kSimTolPerSigma * se_slope << "。\n"
                 "  ② 端点残余系统项: 两端对称的二次失真会把**实测**端点值抬高，使 0.9 分位\n"
                 "     对应的目标角反而**内移**进非线性区（0.1 分位那一侧则外移），于是拟合区间里\n"
                 "     残留少量非线性样本 ⇒ send_pitch_scale 系统性偏低（本台架实测均值 ≈ −3.9e-3，\n"
                 "     即 −0.38%，取 " << kSimTolSystematic << " 覆盖）。这是「按实测端点定 0.1/0.9 边界」\n"
                 "     这一做法的固有残余，不改算法就只能由容差覆盖。\n"
                 "  ③ 默认配置（20 点、默认跨度）实测: 40 个随机种子的最大误差 7.4e-3 < tol；\n"
                 "     若把 --points 调小或把 --min/--max 收窄，tol 会随 SE 自动放宽（越差越松）。\n"
                 "  R² 理论期望 ≈ 1 − (σ_res/σ_y)² ≈ 0.99987（σ_y ≈ span/√12）⇒ 判据取 "
              << kSimMinR2 << "（实测 0.99977~0.99989）。\n";

    // ── 诊断对照: 不裁剪（含两端端点）会怎样 ──
    {
        std::vector<double> t_all, i_all, m_all;
        for (const auto& dp : res.all_samples) {
            t_all.push_back(dp.target_angle);
            i_all.push_back(dp.imu_pitch);
            m_all.push_back(dp.mcu_pitch);
        }
        const LinearFit f1u = fitLinear(m_all, i_all);
        const LinearFit f2u = fitLinear(i_all, t_all);
        const double e1 = f1u.slope - rig.truthRecvScale();
        const double e2 = f2u.slope - rig.truthSendScale();
        std::cout << std::setprecision(6)
                  << "\n诊断对照（不参与标定输出，只说明 0.1/0.9 裁剪的价值）:\n"
                  << "  若用全部 " << res.all_samples.size()
                  << " 个样本（含两端端点与二分点）做同样的两段拟合:\n"
                  << "    Fit1' slope = " << f1u.slope << "（真值 " << rig.truthRecvScale()
                  << "，误差 " << std::setprecision(2) << 100.0 * e1 / rig.truthRecvScale()
                  << "%），offset = " << std::setprecision(6) << f1u.intercept << "\n"
                  << "    Fit2' slope = " << f2u.slope << "（真值 " << rig.truthSendScale()
                  << "，误差 " << std::setprecision(2) << 100.0 * e2 / rig.truthSendScale()
                  << "%），offset = " << std::setprecision(6) << f2u.intercept << "\n"
                  << "  ⇒ 端点二次非线性会把斜率拉偏上述百分比；裁剪后（Step 4 采样区间）的\n"
                     "    恢复误差已回到容差以内。\n";
    }

    std::cout << "\n" << (all_pass ? "✓ --sim 自检通过" : "✗ --sim 自检失败") << "：";
    if (all_pass) {
        std::cout << "4 个参数全部落在容差 ±" << std::setprecision(4) << tol
                  << " 内，且 R² ≥ " << kSimMinR2 << "。\n";
    } else {
        std::cout << "有参数超出容差或 R² 不足（见上表）。\n";
    }
    std::cout << "台架共发送 " << rig.sentFrames() << " 帧（yaw 恒零力矩，仿真无副作用）。\n";

    // --sim 也支持 --out（便于无硬件时先验证"打印 + 落盘"整条链路）
    if (!opt.out.empty()) {
        std::string werr;
        if (writeOutFile(opt.out, opt, res, werr)) {
            std::cout << "（--sim 结果已按 --out 写入: " << opt.out
                      << "；注意这是台架数据，不是实车标定值）\n";
        } else {
            std::cout << "⚠ 写文件失败: " << werr << "\n";
            all_pass = false;
        }
    }
    return all_pass ? 0 : 1;
}

// ============================================================================
// --selftest: 纯数学自检（不扫描、不碰台架、不碰串口）
// ============================================================================
int runSelfTest() {
    std::cout << "\n========== --selftest 纯数学自检（拟合核心）==========\n";
    bool all = true;

    // 1) 精确直线: y = 2.5x − 0.75
    {
        std::vector<double> x, y;
        for (int i = 0; i < 20; ++i) {
            const double xv = -1.0 + 0.1 * i;
            x.push_back(xv);
            y.push_back(2.5 * xv - 0.75);
        }
        const LinearFit f = fitLinear(x, y);
        const bool ok = f.ok && std::fabs(f.slope - 2.5) < 1e-9 &&
                        std::fabs(f.intercept + 0.75) < 1e-9 &&
                        std::fabs(f.r_squared - 1.0) < 1e-9 && f.rms < 1e-12;
        all = all && ok;
        std::cout << std::fixed << std::setprecision(12)
                  << "1) 精确直线 y = 2.5x − 0.75 ⇒ slope=" << f.slope
                  << " intercept=" << f.intercept << " R²=" << f.r_squared
                  << " RMS=" << f.rms << "   " << (ok ? "✓" : "✗") << "\n";
    }

    // 2) 带噪直线: y = 1.03x + 0.05，σ = 2e-3（与 --sim 同量级）
    {
        std::mt19937 rng(2026);
        std::normal_distribution<double> nz(0.0, 2e-3);
        std::vector<double> x, y;
        for (int i = 0; i < 20; ++i) {
            const double xv = -0.43 + 0.045 * i;
            x.push_back(xv);
            y.push_back(1.03 * xv + 0.05 + nz(rng));
        }
        const LinearFit f = fitLinear(x, y);
        const bool ok = f.ok && std::fabs(f.slope - 1.03) <= kSimTolParam &&
                        std::fabs(f.intercept - 0.05) <= kSimTolParam && f.r_squared >= kSimMinR2;
        all = all && ok;
        std::cout << std::setprecision(6)
                  << "2) 带噪直线 y = 1.03x + 0.05（σ=2e-3, n=20）⇒ slope=" << f.slope
                  << "（误差 " << std::setprecision(2) << 100.0 * (f.slope - 1.03) / 1.03
                  << "%） intercept=" << std::setprecision(6) << f.intercept
                  << " R²=" << f.r_squared << "   " << (ok ? "✓" : "✗")
                  << "（容差 ±" << kSimTolParam
                  << "，= --sim 默认配置（20 点 / 默认跨度）下的容差）\n";
    }

    // 3) 端点非线性: 真值 y = 2.5x − 0.75，两端各 10% 行程内**横坐标被二次项抬高**
    //    （对应"端点处编码器/IMU 读数偏离线性"）⇒ 丢掉两端各 10% 后应精确恢复，
    //    含端点一起拟合则被拉偏（这正是工具做 0.1/0.9 分位裁剪的原因）
    {
        std::vector<double> xa, ya, xc, yc;
        const int N = 41;
        const double w = 0.10;                                        // 两端各 10% 行程
        for (int i = 0; i < N; ++i) {
            const double rel = static_cast<double>(i) / (N - 1);      // 0..1
            const double x_true = -1.0 + 2.0 * rel;
            const double y = 2.5 * x_true - 0.75;                     // 真值直线
            double x_meas = x_true;
            if (rel < w)       { const double d = w - rel;         x_meas += 5.0 * d * d; }
            if (rel > 1.0 - w) { const double d = rel - (1.0 - w); x_meas += 5.0 * d * d; }
            xa.push_back(x_meas); ya.push_back(y);
            if (rel >= w && rel <= 1.0 - w) { xc.push_back(x_meas); yc.push_back(y); }
        }
        const LinearFit f_all  = fitLinear(xa, ya);
        const LinearFit f_crop = fitLinear(xc, yc);
        const bool ok = f_crop.ok && std::fabs(f_crop.slope - 2.5) < 1e-9 &&
                        std::fabs(f_crop.intercept + 0.75) < 1e-9 &&
                        std::fabs(f_all.slope - 2.5) > 1e-6;          // 不裁剪确实被拉偏
        all = all && ok;
        std::cout << std::setprecision(6)
                  << "3) 两端各 10% 行程加二次端点非线性 ⇒ 裁剪后 slope=" << f_crop.slope
                  << " intercept=" << f_crop.intercept << "（精确恢复真值）; 不裁剪 slope="
                  << f_all.slope << "（偏差 " << std::setprecision(2)
                  << 100.0 * (f_all.slope - 2.5) / 2.5 << "%）   " << (ok ? "✓" : "✗")
                  << "\n   （说明: 工具里的裁剪由「0.1/0.9 分位二分选目标角」实现，见 Step 3/4）\n";
    }

    // 4) 退化输入: 空 / 单点 / x 全同 ⇒ 不崩、不产生 NaN、ok=false
    {
        const LinearFit f_empty = fitLinear({}, {});
        std::vector<double> x1{1.0}, y1{2.0};
        const LinearFit f_one = fitLinear(x1, y1);
        std::vector<double> xd(5, 3.0), yd{1, 2, 3, 4, 5};
        const LinearFit f_deg = fitLinear(xd, yd);
        const bool ok = !f_empty.ok && !f_one.ok && !f_deg.ok &&
                        std::isfinite(f_empty.slope) && std::isfinite(f_one.slope) &&
                        std::isfinite(f_deg.slope) && std::isfinite(f_deg.r_squared);
        all = all && ok;
        std::cout << "4) 退化输入（空 / 单点 / x 全同）⇒ 均安全返回 ok=false 且无 NaN/Inf   "
                  << (ok ? "✓" : "✗") << "\n";
    }

    std::cout << "\n" << (all ? "✓ --selftest 全部通过" : "✗ --selftest 有失败项") << "\n";
    return all ? 0 : 1;
}

// ============================================================================
// 实车: 预检（等首帧）
// ============================================================================
bool waitForLink(PitchRig& rig, double timeout_s, std::string& err) {
    std::cout << "\n---- 预检: 等待 MCU / IMU 首帧（最多 " << timeout_s << " s）----\n";
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(timeout_s));
    std::string last;
    while (std::chrono::steady_clock::now() < deadline) {
        if (g_interrupted.load()) { err = "用户中断（Ctrl+C）"; return false; }
        if (rig.checkLink(last)) {
            std::cout << "✓ 链路正常: 已收到 MCU 与 IMU 数据。\n";
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    err = "预检失败（" + std::to_string(timeout_s) + " s 内无可用数据）: " + last;
    return false;
}

}  // namespace

}  // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；
// 下面把 namespace tcbs 内的名字引入作用域（与原仓库一致）。
using namespace tcbs;

int main(int argc, char** argv) {
    Options opt;
    std::string err;
    if (!parseArgs(argc, argv, opt, err)) {
        std::cout << "参数错误: " << err << "\n\n";
        usage(argv[0]);
        return 1;
    }
    if (opt.help) { usage(argv[0]); return 0; }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    printBanner(opt, opt.selftest ? "--selftest（纯数学自检）"
                                  : (opt.sim ? "--sim（虚拟台架）" : "实车（真实串口）"));

    // ── 纯数学自检 ──
    if (opt.selftest) return runSelfTest();
    // ── 虚拟台架自检（无硬件）──
    if (opt.sim) return runSim(opt);

    // ── 实车 ──
    std::cout << "\n串口筛选: MCU 端口 = USB 产品串 ≠ \"AutoAim_IMU_Com\" 的 /dev/ttyACM*；"
                 "IMU 端口 = 产品串 == \"AutoAim_IMU_Com\"\n"
              << "发送内容: pitch_target_angle = 目标原始值（不经预处理器）；"
                 "yaw 仅力矩模式 + 零力矩（τ_big = τ_small = 0）\n";

    std::unique_ptr<PitchRig> rig;
    try {
        rig = std::make_unique<HardwareRig>();
    } catch (const std::exception& e) {
        std::cout << "✗ 串口初始化异常: " << e.what() << "\n";
        return 1;
    }

    // 预检: 无硬件时在这里给出明确中文报错（先发零力矩 → 关句柄 → 非 0 退出）
    if (!waitForLink(*rig, opt.wait_s, err)) {
        if (g_interrupted.load()) {   // 预检期间被 Ctrl+C
            std::cout << "\n用户中断（Ctrl+C）。\n";
            std::cout << "退出前先发零力矩（yaw 两关节: 仅力矩 + 0 N·m）…\n";
            rig->emergencyZeroTorque(rig->currentTarget());
            rig->close();
            std::cout << "已关闭串口句柄，退出（退出码 130）。\n";
            return 130;
        }
        std::cout << "\n✗ " << err << "\n"
                  << "  常见原因: ① 串口不存在/无权限（ls -l /dev/ttyACM*，需要 dialout 组）；\n"
                  << "            ② 电控或 IMU 未上电 / USB 未插好；\n"
                  << "            ③ 端口被其它程序占用（control_demo / collect_sysid.py / python 绑定）。\n";
        std::cout << "退出前先发零力矩（yaw 两关节: 仅力矩 + 0 N·m）…\n";
        rig->emergencyZeroTorque(rig->currentTarget());
        rig->close();
        std::cout << "已关闭串口句柄，退出（退出码 1）。\n";
        return 1;
    }

    const int total_samples = 2 + 3 * kBinaryIter + opt.points;
    std::cout << "预计耗时 ≈ " << (total_samples * (opt.dwell + kGapS))
              << " s（" << total_samples << " 个采样点 × (" << opt.dwell << " + " << kGapS << ") s）\n";

    PitchCalibrator calib(*rig, opt.min_raw, opt.max_raw, opt.points, opt.dwell);
    CalibrationResult res;
    bool interrupted = false;
    const bool ok = calib.run(res, err, interrupted);

    // ── 任何退出路径: 先发零力矩，再关句柄（pitch 默认停在"最后一个下发目标"，不突跳）──
    const double park_raw = opt.park_given ? opt.park : rig->currentTarget();
    if (!ok || interrupted) {
        std::cout << "\n" << (interrupted ? "用户中断。" : "✗ 标定失败: " + err) << "\n";
        std::cout << "退出前先发零力矩（yaw 两关节: 仅力矩 + 0 N·m，共 "
                  << kZeroTorqueFrames << " 帧）…\n";
        rig->emergencyZeroTorque(park_raw);
        rig->close();
        std::cout << "已关闭串口句柄，退出（退出码 " << (interrupted ? 130 : 1) << "）。\n";
        return interrupted ? 130 : 1;
    }

    if (!opt.out.empty()) {
        std::string werr;
        if (writeOutFile(opt.out, opt, res, werr)) {
            std::cout << "\n结果已写入: " << opt.out << "\n";
        } else {
            std::cout << "\n⚠ 写文件失败: " << werr
                      << "（标定结果仍有效，已在上方打印）\n";
        }
    }

    std::cout << "\n退出: 先发零力矩（yaw 两关节: 仅力矩 + 0 N·m，共 " << kZeroTorqueFrames
              << " 帧），pitch 目标停在原始值 " << park_raw
              << (opt.park_given ? "（--park）" : "（= 最后一次下发的目标）") << "…\n";
    rig->emergencyZeroTorque(park_raw);
    rig->close();
    std::cout << "已关闭串口句柄，退出（退出码 0）。\n";
    return 0;
}
