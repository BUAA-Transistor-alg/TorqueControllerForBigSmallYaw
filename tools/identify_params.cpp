// ============================================================================
// identify_params.cpp — 双级 yaw **平面 8 参模型**参数辨识（加权最小二乘 + SVD 截断）
//
// 模型（推导见 include/tcbs/mpc/planar_yaw_model.h 文件头）:
//     τ = M(q) q̈ + h(q, q̇, g_A, ω_c, α_c)
//   8 个待辨识参数（顺序 = paramsToVector / regressor 的列序）:
//     0 Jbig_eff  1 Js  2 Px  3 Py  4 fc_big  5 fv_big  6 fc_small  7 fv_small
//   方程对 φ **线性**且 Y = ∂τ/∂φ 与 φ 无关（regressor() 是解析式，已数值校验）
//   ⇒ 一次线性 LS 即可；不可辨识方向（YᵀWY 的零空间）保持先验（默认参数）不变。
//
// 处理链:
//   1) 读任意多个 CSV（按**列名**取值，多余列忽略；必需列缺失时给出清晰错误）
//      必需: t, theta_big, theta_small, dtheta_big, dtheta_small, tau_big, tau_small
//      可选: axis(0=大 yaw 被激励, 1=小 yaw 被激励), held_target, mcu2_seq,
//            chassis_*, big_enc_age（默认忽略）, gravity_ax/gravity_ay（有则逐样本用）
//   2) q̈ = dtheta_* 中心差分 + 3 点平滑（--smooth 可多遍），dt 由 t 列逐段计算
//   3) held 轴处理: --held=ideal（held 轴 q̇/q̈ 置 0）/ measured（实测，默认）/ drop（丢该轴方程）
//   4) 残差 r = τ_meas − inverseDynamics(q, q̇, q̈, φ)；两轴各按 τ 标准差归一化加权
//   5) 列级 SNR 加权（默认开，--no-snr-weight 关闭）: 逐列估「信号 σ / 噪声 σ」→ Wiener 权重，
//      并把 SNR 与权重打印出来（哪些列没资格进回归一目了然）
//   6) --align-shift=auto（默认）: 在 −1/0/+1 帧内按**拟合后残差**搜 τ↔状态平移；
//      改善 < --align-margin(1%) 时保持自然对齐 shift=0（避免追噪声）。
//      另有 --align-shift=fine（0.125 帧步长、τ 线性插值）与
//      **对齐不确定度带**（--align-band）: 打印"残差在最优 +2% 内的 shift 区间"上各参数的散布，
//      这是 LS 的 σ 之外的系统不确定度（摩擦参数对它尤其敏感，见 tools/README_identify.md §5.4）
//   7) 归一化法方程 + SVD 截断（--trunc，相对最大奇异值）+ 可选 Tikhonov（--tikhonov）
//   8) 列共线性诊断（|corr| ≥ 0.90 的列对）
//   9) 物理约束 Jbig_eff>0, Js>0, fc≥0, fv≥0（越界钳制 + 警告）
//  10) 写 data/sysid/identified_params.txt: 可直接替换 planar_yaw_params.h 的 8 行赋值
//  11) --validate=<csv>: 在验证集上算力矩预测 RMS 与开环前向仿真的角度误差
//  12) 指令力矩峰值 ≥ --sat（默认 0.98）时警告可能饱和（饱和样本会污染辨识）
//
// 用法（详见 tools/README_identify.md）:
//   ./build/identify_params data/sysid/sysid_*.csv [options]
//   ./build/identify_params train/*.csv --held=drop --align-shift=auto --trunc=1e-7
//        --validate=val.csv [--tikhonov=1e-3] [--truth=Js=0.017,Px=0.012,...]
// ============================================================================
#include "tcbs/mpc/planar_yaw_params.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace tcbs {

using namespace dual_yaw;

namespace {

// ============================================================================
// 基础工具
// ============================================================================
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false;
    for (char ch : line) {
        if (ch == '"') { in_quote = !in_quote; continue; }
        if (ch == '\r') continue;
        if (ch == ',' && !in_quote) { out.push_back(trim(cur)); cur.clear(); continue; }
        cur.push_back(ch);
    }
    out.push_back(trim(cur));
    return out;
}

bool toNum(const std::string& s, double& out) {
    const std::string t = trim(s);
    if (t.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(t.c_str(), &end);
    if (end == t.c_str()) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0') return false;
    if (!std::isfinite(v)) return false;   // nan/inf 视为无效
    out = v;
    return true;
}

// 按别名顺序找列（第一个命中的别名优先），找不到返回 -1
int findCol(const std::vector<std::string>& hdr,
            const std::vector<std::string>& aliases) {
    for (const auto& a : aliases) {
        if (a.empty()) continue;
        for (size_t i = 0; i < hdr.size(); ++i) {
            if (hdr[i] == a) return static_cast<int>(i);
        }
    }
    return -1;
}

// 无 NaN 的极值/分位数辅助
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = p * static_cast<double>(v.size() - 1);
    const size_t i = static_cast<size_t>(std::floor(x));
    const size_t j = std::min(i + 1, v.size() - 1);
    const double f = x - static_cast<double>(i);
    return v[i] * (1.0 - f) + v[j] * f;
}

std::string numSigned(double v, int prec = 6) {
    if (!std::isfinite(v)) return std::string("inf");
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%+.*g", prec, v);
    return std::string(buf);
}

std::string num(double v, int prec = 6) {
    if (!std::isfinite(v)) return std::string("inf");
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    return std::string(buf);
}

// ============================================================================
// 命令行
// ============================================================================
enum class HeldMode { Ideal, Measured, Drop };

const char* heldName(HeldMode m) {
    switch (m) {
        case HeldMode::Ideal:    return "ideal";
        case HeldMode::Measured: return "measured";
        default:                 return "drop";
    }
}

struct Options {
    std::vector<std::string> files;
    std::vector<std::string> validate;

    HeldMode held = HeldMode::Measured;
    std::string align = "auto";      // "auto" | "fine" | 帧数
    double align_band = 0.02;        // 对齐不确定度带（拟合后残差在最优 +2% 内）
    double align_margin = 0.01;      // 接受非零 shift 所需的最小残差改善（默认 1%）
    double trunc = 1e-7;             // SVD 相对截断阈值
    double tikhonov = 0.0;           // >0 时启用 Tikhonov（归一化坐标下的 λ）
    bool   snr_weight = true;
    double snr_min = 0.05;           // SNR 低于此值的列直接丢（权重 0）
    double snr_floor = 0.0;          // 权重下限
    double qdd_max[2] = {500.0, 2000.0};   // 角加速度差分尖峰阈值 (rad/s²)
    int    smooth = 1;               // 3 点平滑遍数
    bool   joint_weight = true;      // 两轴按力矩 σ 归一化加权
    double gravity_a[2] = {0.0, 0.0};   // 大 yaw 转子系里的重力平面分量
    double tilt_deg = 0.0;           // --tilt-deg=X ⇒ gravity_a = (g·sinX, 0)
    bool   tilt_given = false;
    double dx = -1.0, dy = -1.0;     // <0 ⇒ 用 defaultModelParams()
    double lambda = -1.0;            // <0 ⇒ 用 defaultModelParams().frictionLambda（默认 100）
    double max_big_age = -1.0;       // >0 且存在 big_enc_age/big_age 列时筛样本
    double sat = 0.98;               // 力矩饱和告警阈值
    bool   fwd_sim = true;           // 验证集里做开环前向仿真
    std::string out = "data/sysid/identified_params.txt";
    std::string truth;               // --truth=name=value,...（合成数据自检用）
};

void usage(const char* prog) {
    std::printf(
        "用法: %s <sysid.csv> [more.csv ...] [选项]\n"
        "\n"
        "数据: CSV 列 t,theta_big,theta_small,dtheta_big,dtheta_small,tau_big,tau_small[,axis,...]\n"
        "      (axis: 0=大 yaw 被激励, 1=小 yaw 被激励; 其余列一律忽略)\n"
        "\n"
        "选项:\n"
        "  --held=ideal|measured|drop   held 轴处理（默认 measured）\n"
        "  --align-shift=auto|fine|<x>  τ↔状态时间对齐（auto=搜 −1/0/+1；fine=0.125 帧步长；\n"
        "                               也可直接给帧数，允许小数）\n"
        "  --align-band=<f>             对齐不确定度带阈值（默认 0.02 = 残差在最优 +2%% 内；0=关闭）\n"
        "  --align-margin=<f>           接受非零 shift 所需的最小残差改善（默认 0.01=1%%）\n"
        "  --trunc=<r>                  SVD 相对截断阈值（默认 1e-7）\n"
        "  --tikhonov=<lam>             Tikhonov 正则（归一化坐标；默认 0=关）\n"
        "  --no-snr-weight              关闭列级 SNR 加权\n"
        "  --snr-min=<s>                列 SNR 低于此值则丢弃该列（默认 0.05）\n"
        "  --snr-floor=<f>              列权重下限（默认 0）\n"
        "  --qdd-max=<v>                角加速度尖峰阈值（两轴同值；默认 500/2000）\n"
        "  --qdd-max-big=<v> --qdd-max-small=<v>\n"
        "  --smooth=<n>                 角加速度 3 点平滑遍数（默认 1）\n"
        "  --no-joint-weight            不做两轴力矩 σ 归一化加权\n"
        "  --gravity-a=<gx,gy>          重力在大 yaw 转子系的平面分量 (m/s²)\n"
        "  --tilt-deg=<deg>             底盘绕 y 轴倾斜角 ⇒ gravity_a=(g·sin,0)\n"
        "  --dx=<m> --dy=<m>            覆盖几何偏置（默认取 defaultModelParams()）\n"
        "  --lambda=<v>                 摩擦软符号陡度（默认取 defaultModelParams().frictionLambda）\n"
        "  --max-big-age=<s>            仅当 CSV 有 big_enc_age/big_age 列时筛样本\n"
        "  --sat=<v>                    力矩饱和告警阈值（默认 0.98）\n"
        "  --no-fwd-sim                 验证集里跳过开环前向仿真\n"
        "  --validate=<csv>             验证用 CSV（可重复）\n"
        "  --truth=<name=value,...>     打印真值对比（合成数据自检用）\n"
        "  --out=<path>                 输出文件（默认 data/sysid/identified_params.txt）\n",
        prog);
}

bool needValue(const std::string& a, const char* key, std::string& out) {
    const size_t n = std::strlen(key);
    if (a.size() > n && a.compare(0, n, key) == 0) { out = a.substr(n); return true; }
    return false;
}

bool parseArgs(int argc, char** argv, Options& o, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        std::string v;
        if (a.rfind("--", 0) != 0) { o.files.push_back(a); continue; }

        if (needValue(a, "--held=", v)) {
            if (v == "ideal") o.held = HeldMode::Ideal;
            else if (v == "measured") o.held = HeldMode::Measured;
            else if (v == "drop") o.held = HeldMode::Drop;
            else { err = "--held 只支持 ideal|measured|drop"; return false; }
        } else if (needValue(a, "--align-shift=", v)) {
            if (v != "auto" && v != "fine") {
                char* e = nullptr;
                const double sv = std::strtod(v.c_str(), &e);
                if (e == v.c_str() || *e != '\0' || !std::isfinite(sv) || std::fabs(sv) > 50.0) {
                    err = "--align-shift 只支持 auto | fine | 帧数(−50..50, 允许小数)";
                    return false;
                }
            }
            o.align = v;
        } else if (needValue(a, "--align-band=", v)) {
            o.align_band = std::atof(v.c_str());
        } else if (needValue(a, "--align-margin=", v)) {
            o.align_margin = std::atof(v.c_str());
        } else if (needValue(a, "--trunc=", v)) {
            o.trunc = std::atof(v.c_str());
        } else if (needValue(a, "--tikhonov=", v)) {
            o.tikhonov = std::atof(v.c_str());
        } else if (a == "--no-snr-weight") {
            o.snr_weight = false;
        } else if (needValue(a, "--snr-min=", v)) {
            o.snr_min = std::atof(v.c_str());
        } else if (needValue(a, "--snr-floor=", v)) {
            o.snr_floor = std::atof(v.c_str());
        } else if (needValue(a, "--qdd-max=", v)) {
            o.qdd_max[0] = o.qdd_max[1] = std::atof(v.c_str());
        } else if (needValue(a, "--qdd-max-big=", v)) {
            o.qdd_max[0] = std::atof(v.c_str());
        } else if (needValue(a, "--qdd-max-small=", v)) {
            o.qdd_max[1] = std::atof(v.c_str());
        } else if (needValue(a, "--smooth=", v)) {
            o.smooth = std::max(0, std::atoi(v.c_str()));
        } else if (a == "--no-joint-weight") {
            o.joint_weight = false;
        } else if (needValue(a, "--gravity-a=", v)) {
            const size_t c = v.find(',');
            if (c == std::string::npos) { err = "--gravity-a 需要 gx,gy"; return false; }
            o.gravity_a[0] = std::atof(v.substr(0, c).c_str());
            o.gravity_a[1] = std::atof(v.substr(c + 1).c_str());
        } else if (needValue(a, "--tilt-deg=", v)) {
            o.tilt_deg = std::atof(v.c_str());
            o.tilt_given = true;
        } else if (needValue(a, "--dx=", v)) {
            o.dx = std::atof(v.c_str());
        } else if (needValue(a, "--dy=", v)) {
            o.dy = std::atof(v.c_str());
        } else if (needValue(a, "--lambda=", v)) {
            o.lambda = std::atof(v.c_str());
        } else if (needValue(a, "--max-big-age=", v)) {
            o.max_big_age = std::atof(v.c_str());
        } else if (needValue(a, "--sat=", v)) {
            o.sat = std::atof(v.c_str());
        } else if (a == "--no-fwd-sim") {
            o.fwd_sim = false;
        } else if (needValue(a, "--validate=", v)) {
            o.validate.push_back(v);
        } else if (needValue(a, "--truth=", v)) {
            o.truth = v;
        } else if (needValue(a, "--out=", v)) {
            o.out = v;
        } else {
            err = "未知选项 " + a;
            return false;
        }
    }
    return true;
}

// ============================================================================
// 数据
// ============================================================================
struct Sample {
    double t = 0.0;
    double q[2] = {0.0, 0.0};      // theta_big, theta_small
    double qd[2] = {0.0, 0.0};     // dtheta_*
    double qdd[2] = {0.0, 0.0};    // 由 qd 差分得到
    double tau[2] = {0.0, 0.0};    // tau_big, tau_small（指令力矩）
    double gA[2] = {0.0, 0.0};     // gravity_ax/ay（可选列）
    bool   has_gA = false;
    double big_age = -1.0;         // big_enc_age / big_age（可选列）
    int    axis = -1;              // 0: 大 yaw 被激励, 1: 小 yaw 被激励, -1: 未知
    int    seg = 0;                // 连续段编号（文件边界/时间跳变处断开）
    bool   tainted = false;        // 差分边界/尖峰/无效 ⇒ 该样本的状态不可用于拟合
};

struct Cols {
    int t = -1, tb = -1, ts = -1, vb = -1, vs = -1, taub = -1, taus = -1;
    int axis = -1, gax = -1, gay = -1, age = -1;
};

struct LoadReport {
    size_t rows = 0, bad_rows = 0, dropped_age = 0;
    size_t axis_big = 0, axis_small = 0, axis_unknown = 0;
    size_t segs = 0;
    double dt_med = 0.0, dt_min = 0.0, dt_max = 0.0;
    double t_span = 0.0;
    bool has_axis_col = false, has_gravity_cols = false;
};

bool loadCsv(const std::string& path, const Options& opt, std::vector<Sample>& out,
             int& seg_counter, LoadReport& rep, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "无法打开 " + path; return false; }
    std::string line;
    if (!std::getline(f, line)) { err = "空文件 " + path; return false; }
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF) {
        line.erase(0, 3);   // UTF-8 BOM
    }
    std::vector<std::string> hdr = splitCsv(line);
    for (auto& h : hdr) h = trim(h);

    Cols c;
    c.t    = findCol(hdr, {"t", "time", "timestamp"});
    c.tb   = findCol(hdr, {"theta_big", "big_yaw", "theta_big_rad"});
    c.ts   = findCol(hdr, {"theta_small", "small_yaw", "theta_small_rad"});
    c.vb   = findCol(hdr, {"dtheta_big", "omega_big", "theta_big_dot"});
    c.vs   = findCol(hdr, {"dtheta_small", "omega_small", "theta_small_dot"});
    c.taub = findCol(hdr, {"tau_big", "tau_big_cmd", "tau_big_meas"});
    c.taus = findCol(hdr, {"tau_small", "tau_small_cmd", "tau_small_meas"});
    c.axis = findCol(hdr, {"axis", "excite_axis", "excited_axis", "excite"});
    c.gax  = findCol(hdr, {"gravity_ax", "g_a_x", "gravity_a_x", "gravity_a0"});
    c.gay  = findCol(hdr, {"gravity_ay", "g_a_y", "gravity_a_y", "gravity_a1"});
    c.age  = findCol(hdr, {"big_enc_age", "big_age"});

    {
        std::string miss;
        auto need = [&](int idx, const char* name) {
            if (idx < 0) { if (!miss.empty()) miss += ", "; miss += name; }
        };
        need(c.t, "t"); need(c.tb, "theta_big"); need(c.ts, "theta_small");
        need(c.vb, "dtheta_big"); need(c.vs, "dtheta_small");
        need(c.taub, "tau_big"); need(c.taus, "tau_small");
        if (!miss.empty()) {
            std::string have;
            for (size_t i = 0; i < hdr.size(); ++i) {
                if (i) have += ",";
                have += hdr[i];
            }
            err = "缺少必需列 [" + miss + "]；文件实际列: " + have;
            return false;
        }
    }
    rep.has_axis_col = (c.axis >= 0);
    rep.has_gravity_cols = (c.gax >= 0 && c.gay >= 0);

    std::vector<Sample> rows;
    size_t lineno = 1;
    while (std::getline(f, line)) {
        ++lineno;
        if (trim(line).empty()) continue;
        const std::vector<std::string> v = splitCsv(line);
        auto get = [&](int idx, double def, bool& ok) {
            if (idx < 0 || idx >= static_cast<int>(v.size())) { ok = false; return def; }
            double x = def;
            ok = toNum(v[static_cast<size_t>(idx)], x);
            return x;
        };
        Sample s;
        bool ok = true, o1 = true;
        s.t = get(c.t, 0.0, o1); ok = ok && o1;
        s.q[0] = get(c.tb, 0.0, o1); ok = ok && o1;
        s.q[1] = get(c.ts, 0.0, o1); ok = ok && o1;
        s.qd[0] = get(c.vb, 0.0, o1); ok = ok && o1;
        s.qd[1] = get(c.vs, 0.0, o1); ok = ok && o1;
        s.tau[0] = get(c.taub, 0.0, o1); ok = ok && o1;
        s.tau[1] = get(c.taus, 0.0, o1); ok = ok && o1;
        if (!ok) { ++rep.bad_rows; continue; }

        if (c.axis >= 0) {
            double a = -1.0; bool oa = true;
            a = get(c.axis, -1.0, oa);
            if (oa) s.axis = static_cast<int>(std::lround(a));   // 其它值 ⇒ 未知（两轴都激励）
        }
        if (c.gax >= 0 && c.gay >= 0) {
            bool og1 = true, og2 = true;
            const double gx = get(c.gax, 0.0, og1);
            const double gy = get(c.gay, 0.0, og2);
            if (og1 && og2) { s.gA[0] = gx; s.gA[1] = gy; s.has_gA = true; }
        }
        if (c.age >= 0) { bool oa = true; s.big_age = get(c.age, -1.0, oa); }
        rows.push_back(s);
        (void)lineno;
    }
    if (rows.empty()) { err = "无有效数据行: " + path; return false; }

    // ── dt 中位数 + 分段（文件边界必断；时间跳变 > 50% 也断）──
    std::vector<double> dts;
    dts.reserve(rows.size());
    for (size_t i = 1; i < rows.size(); ++i) {
        const double dt = rows[i].t - rows[i - 1].t;
        if (dt > 0.0) dts.push_back(dt);
    }
    const double dt_med = dts.empty() ? 0.0 : percentile(dts, 0.5);
    if (!(dt_med > 0.0)) { err = "t 列非严格递增（无法计算 dt）: " + path; return false; }
    rep.dt_med = dt_med;
    rep.dt_min = *std::min_element(dts.begin(), dts.end());
    rep.dt_max = *std::max_element(dts.begin(), dts.end());

    int seg = seg_counter++;
    ++rep.segs;
    rows[0].seg = seg;
    for (size_t i = 1; i < rows.size(); ++i) {
        const double dt = rows[i].t - rows[i - 1].t;
        if (!(dt > 0.0) || dt > 1.5 * dt_med) { seg = seg_counter++; ++rep.segs; }
        rows[i].seg = seg;
    }

    // ── 可选: 按大 yaw 编码器年龄筛样本（列不存在时完全不筛）──
    if (opt.max_big_age > 0.0 && c.age >= 0) {
        std::vector<Sample> keep;
        keep.reserve(rows.size());
        for (const auto& s : rows) {
            if (s.big_age >= 0.0 && s.big_age > opt.max_big_age) { ++rep.dropped_age; continue; }
            keep.push_back(s);
        }
        rows.swap(keep);
    }

    rep.rows += rows.size();
    rep.t_span += (rows.back().t - rows.front().t);
    for (const auto& s : rows) {
        if (s.axis == 0) ++rep.axis_big;
        else if (s.axis == 1) ++rep.axis_small;
        else ++rep.axis_unknown;
    }
    out.insert(out.end(), rows.begin(), rows.end());
    return true;
}

// ── 角加速度: 段内中心差分 + 3 点平滑 ──
struct AccelReport {
    size_t edge = 0, spike[2] = {0, 0};
    double p999[2] = {0, 0}, absmax[2] = {0, 0};
};

void computeAccel(std::vector<Sample>& d, const Options& opt, AccelReport& rep) {
    size_t i = 0;
    while (i < d.size()) {
        size_t j = i;
        while (j + 1 < d.size() && d[j + 1].seg == d[i].seg) ++j;
        const size_t n = j - i + 1;

        for (size_t k = i; k <= j; ++k) {
            if (n < 2) { d[k].qdd[0] = d[k].qdd[1] = 0.0; continue; }
            const size_t a = (k == i) ? k : k - 1;
            const size_t b = (k == j) ? k : k + 1;
            double dt = d[b].t - d[a].t;
            if (!(dt > 1e-6)) dt = 1e-3;
            for (int ax = 0; ax < 2; ++ax) d[k].qdd[ax] = (d[b].qd[ax] - d[a].qd[ax]) / dt;
        }
        // 3 点平滑 (0.25, 0.5, 0.25)，可多遍；段边界点不更新
        std::vector<std::array<double, 2>> buf(n);
        for (int pass = 0; pass < opt.smooth && n >= 3; ++pass) {
            for (size_t k = 0; k < n; ++k)
                for (int ax = 0; ax < 2; ++ax) buf[k][ax] = d[i + k].qdd[ax];
            for (size_t k = 1; k + 1 < n; ++k) {
                for (int ax = 0; ax < 2; ++ax) {
                    d[i + k].qdd[ax] = 0.25 * buf[k - 1][ax] + 0.5 * buf[k][ax] +
                                       0.25 * buf[k + 1][ax];
                }
            }
        }
        if (n >= 3) {                 // 差分边界不可靠 ⇒ 不参与拟合
            d[i].tainted = true;
            d[j].tainted = true;
            rep.edge += 2;
        }
        i = j + 1;
    }

    // 差分尖峰剔除（阈值可参数化）
    std::vector<double> mag[2];
    for (const auto& s : d)
        for (int ax = 0; ax < 2; ++ax) mag[ax].push_back(std::fabs(s.qdd[ax]));
    for (int ax = 0; ax < 2; ++ax) {
        rep.p999[ax] = percentile(mag[ax], 0.999);
        rep.absmax[ax] = mag[ax].empty() ? 0.0 : *std::max_element(mag[ax].begin(), mag[ax].end());
    }
    for (auto& s : d) {
        for (int ax = 0; ax < 2; ++ax) {
            if (std::fabs(s.qdd[ax]) > opt.qdd_max[ax]) {
                s.tainted = true;
                ++rep.spike[ax];
            }
        }
    }
}

// ============================================================================
// 行（方程）组装
//
// 对齐约定 shift: τ 的第 (j − shift) 号样本与状态第 j 号样本配对（shift 允许小数，
// 此时 τ 在线性插值后使用）。**为什么需要小数**: q̈ 用中心差分得到（覆盖 ±1 帧，
// 重心在 j），而 τ 是零阶保持（τ[i] 作用在 [i, i+1]，重心在 i+0.5）；再叠加 3 点
// 平滑后，整条链的等效配对准心约在 j − 0.375 帧。若强制用整数帧，摩擦参数
// （fc/fv）会被系统性偏移几十 %（见 tools/README_identify.md 的自检数据）。
// ============================================================================
struct Row {
    int j;         // 状态样本下标
    int eq;        // 方程/关节下标 0=大 yaw, 1=小 yaw
    double tau;    // 该方程的实测（指令）力矩，已按 shift 插值
};

void buildRows(const std::vector<Sample>& d, double shift, HeldMode mode,
               std::vector<Row>& rows) {
    rows.clear();
    const int n = static_cast<int>(d.size());
    for (int j = 0; j < n; ++j) {
        if (d[static_cast<size_t>(j)].tainted) continue;
        const double fi = static_cast<double>(j) - shift;
        if (fi < 0.0 || fi > static_cast<double>(n - 1)) continue;
        int i0 = static_cast<int>(std::floor(fi));
        double a = fi - static_cast<double>(i0);
        int i1 = i0 + 1;
        if (i1 > n - 1) { i1 = i0; a = 0.0; }
        if (d[static_cast<size_t>(i0)].seg != d[static_cast<size_t>(j)].seg) continue;
        if (d[static_cast<size_t>(i1)].seg != d[static_cast<size_t>(j)].seg) { i1 = i0; a = 0.0; }
        const int inear = std::min(std::max(static_cast<int>(std::lround(fi)), 0), n - 1);
        const int axis = d[static_cast<size_t>(inear)].axis;
        const double t0 = d[static_cast<size_t>(i0)].tau[0] * (1.0 - a) +
                          d[static_cast<size_t>(i1)].tau[0] * a;
        const double t1 = d[static_cast<size_t>(i0)].tau[1] * (1.0 - a) +
                          d[static_cast<size_t>(i1)].tau[1] * a;
        if (mode == HeldMode::Drop && (axis == 0 || axis == 1)) {
            rows.push_back({j, axis, axis == 0 ? t0 : t1});
        } else {
            rows.push_back({j, 0, t0});
            rows.push_back({j, 1, t1});
        }
    }
}

inline int heldAxisOf(int axis) {
    if (axis == 0) return 1;
    if (axis == 1) return 0;
    return -1;
}

inline void effState(const Sample& s, HeldMode mode, double qd[2], double qdd[2]) {
    for (int a = 0; a < 2; ++a) { qd[a] = s.qd[a]; qdd[a] = s.qdd[a]; }
    if (mode == HeldMode::Ideal) {
        const int h = heldAxisOf(s.axis);
        if (h >= 0) { qd[h] = 0.0; qdd[h] = 0.0; }   // held 轴视作静止（位置仍用实测）
    }
}

inline ModelExo makeExo(const Sample& s, const Options& opt) {
    ModelExo e;
    if (s.has_gA) { e.gravity_a[0] = s.gA[0]; e.gravity_a[1] = s.gA[1]; }
    else          { e.gravity_a[0] = opt.gravity_a[0]; e.gravity_a[1] = opt.gravity_a[1]; }
    e.base_omega = 0.0;   // 本工具不辨识底盘耦合（需要时在采集侧写入并按 0 处理）
    e.base_alpha = 0.0;
    return e;
}

// 加权残差平方和（先验参数下的残差；仅作参考，对齐选择用 probeFitRms 的拟合后残差）
double sseWeighted(const std::vector<Sample>& d, const std::vector<Row>& rows,
                   const ModelParams& p, const Options& opt, const double w[2],
                   HeldMode mode) {
    double sse = 0.0;
    for (const auto& r : rows) {
        const Sample& s = d[static_cast<size_t>(r.j)];
        double qd[2], qdd[2];
        effState(s, mode, qd, qdd);
        const double q[2] = {s.q[0], s.q[1]};
        double tau[2];
        inverseDynamics(q, qd, qdd, p, makeExo(s, opt), tau);
        const double e = r.tau - tau[r.eq];
        sse += w[r.eq] * w[r.eq] * e * e;
    }
    return sse;
}

// ============================================================================
// 列级 SNR
// ============================================================================
struct SnrInfo {
    double sig[kNumParams] = {0};
    double noise[kNumParams] = {0};
    double snr[kNumParams] = {0};
    double wgt[kNumParams] = {1, 1, 1, 1, 1, 1, 1, 1};
    bool   dropped[kNumParams] = {false};
};

// 每列: σ_signal = 段内合并信号标准差；σ_noise = 高通残差 x_i−½(x_{i−1}+x_{i+1})
// 的估计噪声（白噪声假设下 ×1/√1.5）。结构上为 0 的流（如摩擦的交叉列）跳过。
SnrInfo estimateSnr(const std::vector<Sample>& d, const std::vector<Row>& rows,
                    const Eigen::MatrixXd& Yw, const Options& opt) {
    SnrInfo info;
    // 按 (seg, eq) 分组，保持时间顺序
    std::vector<int> keys;
    std::vector<std::vector<int>> groups;
    for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
        const int key = d[static_cast<size_t>(rows[static_cast<size_t>(r)].j)].seg * 2 +
                        rows[static_cast<size_t>(r)].eq;
        size_t g = 0;
        for (; g < keys.size(); ++g) if (keys[g] == key) break;
        if (g == keys.size()) { keys.push_back(key); groups.emplace_back(); }
        groups[g].push_back(r);
    }

    for (int c = 0; c < kNumParams; ++c) {
        double ss_sig = 0.0, ss_noi = 0.0;
        double cnt_sig = 0.0, cnt_noi = 0.0;
        for (const auto& g : groups) {
            if (g.size() < 5) continue;
            double mean = 0.0;
            for (int r : g) mean += Yw(r, c);
            mean /= static_cast<double>(g.size());
            double ss = 0.0;
            for (int r : g) { const double x = Yw(r, c) - mean; ss += x * x; }
            const double var = ss / static_cast<double>(g.size());
            if (var < 1e-20) continue;              // 结构上为 0 的流（如摩擦交叉列）
            ss_sig += ss;
            cnt_sig += static_cast<double>(g.size());
            double sh = 0.0;
            for (size_t k = 1; k + 1 < g.size(); ++k) {
                const double x = Yw(g[k], c) - 0.5 * (Yw(g[k - 1], c) + Yw(g[k + 1], c));
                sh += x * x;
            }
            const double cnt_h = static_cast<double>(g.size() - 2);
            if (cnt_h > 0.0) { ss_noi += sh / 1.5; cnt_noi += cnt_h; }
        }
        if (cnt_sig < 1.0) {                        // 该列在所有流里都是 0
            info.sig[c] = 0.0; info.noise[c] = 0.0; info.snr[c] = 0.0;
            info.wgt[c] = 0.0; info.dropped[c] = true;
            continue;
        }
        const double var_sig = ss_sig / cnt_sig;
        const double var_noi = (cnt_noi > 0.0) ? (ss_noi / cnt_noi) : 0.0;
        info.sig[c] = std::sqrt(var_sig);
        info.noise[c] = std::sqrt(var_noi);
        if (var_noi < 1e-18) info.snr[c] = 1e9;
        else info.snr[c] = std::sqrt(var_sig / var_noi);
        double w = info.snr[c] * info.snr[c] / (1.0 + info.snr[c] * info.snr[c]);
        if (info.snr[c] < opt.snr_min) { w = 0.0; info.dropped[c] = true; }
        w = std::max(w, std::min(1.0, opt.snr_floor));
        info.wgt[c] = w;
    }
    return info;
}

// ── 轻量 LS 探针: 用给定行集做一次线性 LS，返回**拟合后**的加权残差 RMS ──
// 只做列归一化 + SVD 截断，用于 --align-shift 的候选比较（该准则与先验参数的
// 误差无关 —— 用先验残差挑对齐会被"先验不准"误导成错误的平移）。
// phi_out != nullptr 时同时给出该 shift 下的参数估计（已做物理钳制）。
double probeFitRms(const std::vector<Sample>& d, const std::vector<Row>& rows,
                   const ModelParams& p0, const double phi0[kNumParams],
                   const Options& opt, const double w[2],
                   double phi_out[kNumParams] = nullptr) {
    const int nr = static_cast<int>(rows.size());
    if (nr < 20) return kInf;
    Eigen::MatrixXd Y(nr, kNumParams);
    Eigen::VectorXd rhs(nr);
    for (int r = 0; r < nr; ++r) {
        const Row& row = rows[static_cast<size_t>(r)];
        const Sample& s = d[static_cast<size_t>(row.j)];
        double qd[2], qdd[2];
        effState(s, opt.held, qd, qdd);
        const double q[2] = {s.q[0], s.q[1]};
        double Yr[2][kNumParams];
        regressor(q, qd, qdd, p0, makeExo(s, opt), Yr);
        double base = 0.0;
        for (int c = 0; c < kNumParams; ++c) {
            Y(r, c) = Yr[row.eq][c] * w[row.eq];
            base += Y(r, c) * phi0[c];
        }
        rhs(r) = row.tau * w[row.eq] - base;
    }
    double nrmv[kNumParams];
    for (int c = 0; c < kNumParams; ++c) {
        nrmv[c] = Y.col(c).norm();
        if (nrmv[c] > 1e-12) Y.col(c) /= nrmv[c];
        else nrmv[c] = 1.0;
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd sv = svd.singularValues();
    const double tol = std::max(1e-14, opt.trunc * (sv.size() ? sv(0) : 0.0));
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(kNumParams);
    for (int i = 0; i < sv.size(); ++i) {
        if (sv(i) <= tol) continue;
        delta += (svd.matrixU().col(i).dot(rhs) / sv(i)) * svd.matrixV().col(i);
    }
    if (phi_out) {
        for (int c = 0; c < kNumParams; ++c) {
            double v = phi0[c] + delta(c) / nrmv[c];
            if (c == 0 || c == 1) v = std::max(1e-5, v);   // Jbig_eff, Js > 0
            if (c >= 4) v = std::max(0.0, v);              // fc/fv ≥ 0
            phi_out[c] = v;
        }
    }
    const Eigen::VectorXd res = rhs - Y * delta;
    return std::sqrt(res.squaredNorm() / static_cast<double>(nr));
}

// ============================================================================
// 报告
// ============================================================================
void printHeader() {
    std::printf(
        "==============================================================\n"
        " identify_params — 平面 8 参模型辨识 (加权 LS + SVD 截断)\n"
        " 参数: Jbig_eff Js Px Py fc_big fv_big fc_small fv_small\n"
        "==============================================================\n");
}

}  // namespace

// ============================================================================
} // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；下面把 tcbs 内的类型与测试辅助函数引入作用域
using namespace tcbs;

int main(int argc, char** argv) {
    Options opt;
    std::string err;
    if (!parseArgs(argc, argv, opt, err)) {
        std::printf("参数错误: %s\n\n", err.c_str());
        usage(argv[0]);
        return 1;
    }
    if (opt.files.empty()) { usage(argv[0]); return 1; }
    if (opt.tilt_given) {
        const double g = 9.81;
        opt.gravity_a[0] = g * std::sin(opt.tilt_deg * kPi / 180.0);
        opt.gravity_a[1] = 0.0;
    }

    printHeader();
    std::printf("几何: dx=%s dy=%s (defaultModelParams%s)\n",
                num(opt.dx >= 0 ? opt.dx : defaultModelParams().dx, 6).c_str(),
                num(opt.dy >= 0 ? opt.dy : defaultModelParams().dy, 6).c_str(),
                (opt.dx >= 0 || opt.dy >= 0) ? " 被 --dx/--dy 覆盖" : "");
    std::printf("held=%s  align-shift=%s  trunc=%g  tikhonov=%g  SNR加权=%s  smooth=%d\n",
                heldName(opt.held), opt.align.c_str(), opt.trunc, opt.tikhonov,
                opt.snr_weight ? "on" : "off", opt.smooth);
    std::printf("gravity_a(默认, m/s²) = (%s, %s)%s\n",
                num(opt.gravity_a[0], 6).c_str(), num(opt.gravity_a[1], 6).c_str(),
                (opt.tilt_given ? "  [--tilt-deg]" : ""));

    // ── 1) 读数据 ──────────────────────────────────────────────────────────
    std::vector<Sample> data;
    int seg_counter = 0;
    bool any_axis_col = true, any_gravity = false;
    std::printf("\n---- 数据 ----\n");
    for (const auto& f : opt.files) {
        const size_t before = data.size();
        LoadReport r;
        int sc = seg_counter;
        if (!loadCsv(f, opt, data, sc, r, err)) {
            std::printf("读取失败: %s\n", err.c_str());
            return 1;
        }
        seg_counter = sc;
        std::printf("  %s: %zu 样本, dt中位=%s s, 段数=%zu, axis(big/small/未知)=%zu/%zu/%zu%s\n",
                    f.c_str(), r.rows, num(r.dt_med, 4).c_str(), r.segs,
                    r.axis_big, r.axis_small, r.axis_unknown,
                    r.bad_rows ? (" , 丢弃无效行 " + std::to_string(r.bad_rows)).c_str() : "");
        if (r.dropped_age)
            std::printf("    (按 --max-big-age=%g 剔除 %zu 样本)\n", opt.max_big_age, r.dropped_age);
        if (data.size() == before) { std::printf("  该文件无有效样本\n"); }
        any_axis_col = any_axis_col && r.has_axis_col;
        any_gravity = any_gravity || r.has_gravity_cols;
    }
    if (data.empty()) { std::printf("无有效数据\n"); return 1; }
    std::printf("合计 %zu 样本（%d 段）\n", data.size(), seg_counter);
    if (!any_axis_col) {
        std::printf("⚠ 数据里没有 axis 列 ⇒ 无法判断哪一轴被\"保持\"：\n"
                    "  --held=ideal/measured/drop 将退化为\"两轴都按实测\"处理（等价 measured 的两轴方程）\n");
    }
    std::printf("gravity_a: %s\n", any_gravity ? "使用 CSV 的 gravity_ax/ay 列"
                                               : "使用命令行常数（CSV 无该列）");
    if (opt.held == HeldMode::Drop) {
        size_t nb = 0, ns = 0;
        for (const auto& s : data) { if (s.axis == 0) ++nb; else if (s.axis == 1) ++ns; }
        if (nb == 0 || ns == 0) {
            std::printf("\n✗ --held=drop 只保留**被激励轴**的方程：本数据里\n"
                        "  大yaw 被激励 %zu 个样本 / 小yaw 被激励 %zu 个样本 ⇒ 其中一轴的方程会被全部丢弃，\n"
                        "  该轴的参数（Js/Px/Py/fc/fv 中的一部分）不可辨识。\n"
                        "  请改用 --held=measured（默认）；若两轴都有激励数据请把它们一起传入。\n", nb, ns);
            return 1;
        }
    }

    // ── 2) 角加速度 ────────────────────────────────────────────────────────
    AccelReport ar;
    computeAccel(data, opt, ar);
    std::printf("\n---- 角加速度（中心差分 + %d 遍 3 点平滑）----\n", opt.smooth);
    for (int ax = 0; ax < 2; ++ax) {
        std::printf("  %s: |qdd| p99.9=%s, max=%s rad/s²；尖峰阈值 %g ⇒ 剔除 %zu\n",
                    ax == 0 ? "大yaw" : "小yaw", num(ar.p999[ax], 4).c_str(),
                    num(ar.absmax[ax], 4).c_str(), opt.qdd_max[ax], ar.spike[ax]);
    }
    std::printf("  差分边界剔除 %zu 个样本状态（每段首尾）\n", ar.edge);

    // ── 3) 力矩饱和检查 ────────────────────────────────────────────────────
    {
        double mx[2] = {0, 0};
        size_t over[2] = {0, 0};
        for (const auto& s : data)
            for (int j = 0; j < 2; ++j) {
                mx[j] = std::max(mx[j], std::fabs(s.tau[j]));
                if (std::fabs(s.tau[j]) >= opt.sat) ++over[j];
            }
        std::printf("\n---- 力矩检查 ----\n");
        std::printf("  指令力矩峰值: 大yaw %s N·m, 小yaw %s N·m（饱和阈值/sample数 %g / %zu,%zu）\n",
                    num(mx[0], 5).c_str(), num(mx[1], 5).c_str(), opt.sat, over[0], over[1]);
        if (mx[0] >= opt.sat || mx[1] >= opt.sat) {
            std::printf("  ⚠ 警告: 指令力矩已达/接近饱和阈值 %g —— 电控可能已限幅截断，\n"
                        "    饱和样本会让辨识结果产生系统偏差（残差被\"削平\"）；\n"
                        "    建议降低激励幅值重采，或确认采集侧的峰值上限设置。\n", opt.sat);
        }
    }

    // ── 4) 关节权重 + 时间对齐 ─────────────────────────────────────────────
    ModelParams p0 = defaultModelParams();
    if (opt.lambda > 0.0) p0.frictionLambda = opt.lambda;
    if (opt.dx >= 0) p0.dx = opt.dx;
    if (opt.dy >= 0) p0.dy = opt.dy;
    const double phi0[kNumParams] = {p0.Jbig_eff, p0.Js, p0.Px, p0.Py,
                                     p0.fcBig, p0.fvBig, p0.fcSmall, p0.fvSmall};

    double w[2] = {1.0, 1.0};
    std::printf("\n---- 加权 ----\n");
    for (int eq = 0; eq < 2; ++eq) {
        double s1 = 0.0, s2 = 0.0;
        size_t n = 0;
        for (const auto& s : data) {
            const double tv = s.tau[eq];
            if (std::fabs(tv) < 1e-12) continue;
            s1 += tv; s2 += tv * tv; ++n;
        }
        if (n < 10) { std::printf("关节 %d 的力矩样本过少\n", eq); return 1; }
        const double mean = s1 / static_cast<double>(n);
        const double var = std::max(0.0, s2 / static_cast<double>(n) - mean * mean);
        const double sd = std::sqrt(var);
        if (!opt.joint_weight) { w[eq] = 1.0; }
        else if (sd < 1e-6) {
            w[eq] = 1.0;
            std::printf("  ⚠ 关节 %d 的力矩几乎无变化（σ=%s）⇒ 权重取 1（该轴大概率不可辨识）\n",
                        eq, num(sd, 3).c_str());
        } else {
            w[eq] = 1.0 / sd;
        }
    }
    std::printf("  关节力矩标准差: 大yaw %s, 小yaw %s N·m ⇒ 权重 (%s, %s)%s\n",
                num(1.0 / w[0], 5).c_str(), num(1.0 / w[1], 5).c_str(),
                num(w[0], 4).c_str(), num(w[1], 4).c_str(),
                opt.joint_weight ? "" : "  [--no-joint-weight]");

    std::printf("\n---- 时间对齐（τ 与状态平移，允许小数帧）----\n");
    const char* const* nm = paramNames();
    double best_shift = 0.0;
    {
        std::vector<Row> rows;
        std::vector<double> cand;
        if (opt.align == "fine") {
            for (double s = -1.0; s <= 1.0001; s += 0.125) cand.push_back(s);
            std::printf("  --align-shift=fine: 在 [−1,+1] 上以 0.125 帧步长搜索（τ 线性插值）\n");
        } else {
            cand = {-1.0, 0.0, 1.0};   // 规范要求的整数帧搜索
        }
        std::vector<double> post_of(cand.size(), kInf);
        double best = kInf;
        for (size_t ci = 0; ci < cand.size(); ++ci) {
            buildRows(data, cand[ci], opt.held, rows);
            if (rows.size() < 20) {
                std::printf("  shift %6s: 有效方程 %zu（过少，跳过）\n",
                            numSigned(cand[ci], 4).c_str(), rows.size());
                continue;
            }
            // 先验残差（仅供参考）+ **拟合后**残差（用于选择，与先验参数误差无关）
            const double rms_pre = std::sqrt(sseWeighted(data, rows, p0, opt, w, opt.held) /
                                             static_cast<double>(rows.size()));
            post_of[ci] = probeFitRms(data, rows, p0, phi0, opt, w);
            std::printf("  shift %6s: 方程 %5zu, 先验残差 %s → 拟合后残差 %s N·m%s\n",
                        numSigned(cand[ci], 4).c_str(), rows.size(), num(rms_pre, 5).c_str(),
                        num(post_of[ci], 5).c_str(), (post_of[ci] < best ? "   ← 目前最优" : ""));
            if (post_of[ci] < best) { best = post_of[ci]; best_shift = cand[ci]; }
        }
        // ── 对齐不确定度带: 残差对亚帧错位不敏感，但 fc/fv 很敏感 ⇒ 给出参数散布 ──
        if (opt.align_band > 0.0) {
            const int NG = 17;
            std::vector<double> rmsg(NG, kInf);
            std::vector<std::array<double, kNumParams>> phig(NG);
            double rbest = kInf;
            for (int g = 0; g < NG; ++g) {
                const double sg = -1.0 + 0.125 * g;
                buildRows(data, sg, opt.held, rows);
                if (rows.size() < 20) continue;
                rmsg[static_cast<size_t>(g)] = probeFitRms(data, rows, p0, phi0, opt, w,
                                                           phig[static_cast<size_t>(g)].data());
                rbest = std::min(rbest, rmsg[static_cast<size_t>(g)]);
            }
            if (std::isfinite(rbest)) {
                const double lim = rbest * (1.0 + opt.align_band);
                double smin = kInf, smax = -kInf;
                double pmax[kNumParams] = {0}, pmin[kNumParams] = {0};
                for (int c = 0; c < kNumParams; ++c) { pmax[c] = -kInf; pmin[c] = kInf; }
                int used = 0;
                for (int g = 0; g < NG; ++g) {
                    if (!(rmsg[static_cast<size_t>(g)] <= lim)) continue;
                    const double sg = -1.0 + 0.125 * g;
                    smin = std::min(smin, sg); smax = std::max(smax, sg);
                    for (int c = 0; c < kNumParams; ++c) {
                        const double v = phig[static_cast<size_t>(g)][static_cast<size_t>(c)];
                        pmax[c] = std::max(pmax[c], v); pmin[c] = std::min(pmin[c], v);
                    }
                    ++used;
                }
                std::printf("  对齐不确定度带（拟合后残差 ≤ 最优+%.1f%%）: shift %s … %s（%d 个候选）\n",
                            100.0 * opt.align_band, numSigned(smin, 4).c_str(),
                            numSigned(smax, 4).c_str(), used);
                std::printf("    该带内各参数的散布（**LS 的 σ 不含这一项**，是系统不确定度）:\n");
                for (int c = 0; c < kNumParams; ++c) {
                    const double mid = 0.5 * (pmax[c] + pmin[c]);
                    const double rel = (std::fabs(mid) > 1e-12) ? 100.0 * (pmax[c] - pmin[c]) / std::fabs(mid) : 0.0;
                    std::printf("      %-10s %12s … %-12s (±%.1f%%)\n", nm[c], num(pmin[c], 5).c_str(),
                                num(pmax[c], 5).c_str(), 0.5 * rel);
                }
            }
        }
        if (opt.align != "auto" && opt.align != "fine") {
            best_shift = std::atof(opt.align.c_str());
            std::printf("  --align-shift=%s ⇒ 强制使用 shift %s\n", opt.align.c_str(),
                        numSigned(best_shift, 4).c_str());
        } else if (std::isfinite(best)) {
            // 是否接受非零 shift: 需要相对 shift 0 有足够改善（否则是噪声/半帧效应在作祟）
            const double rms_zero = post_of[1];   // cand[1] == 0
            const double gain = (std::isfinite(rms_zero) && rms_zero > 0.0)
                                    ? (rms_zero - best) / rms_zero : 0.0;
            if (opt.align_margin > 0.0 && best_shift != 0.0 && gain < opt.align_margin) {
                std::printf("⇒ 保持 shift = +0：最优 shift %s 相对 shift 0 仅改善 %.2f%% < 阈值 %.2f%%\n"
                            "  ⇒ 认为对齐无法由残差判定（残差对亚帧错位不敏感），按自然对齐（τ[i]↔state[i]）处理；\n"
                            "    要强制使用最优 shift 请显式 --align-shift=%s，或调小 --align-margin。\n",
                            numSigned(best_shift, 4).c_str(), 100.0 * gain, 100.0 * opt.align_margin,
                            numSigned(best_shift, 4).c_str());
                best_shift = 0.0;
            } else {
                double second = kInf;
                for (size_t ci = 0; ci < cand.size(); ++ci)
                    if (std::fabs(cand[ci] - best_shift) > 1e-9 && post_of[ci] < second) second = post_of[ci];
                std::printf("⇒ 选择 shift = %s（拟合后残差最小，相对 shift 0 改善 %.2f%%）\n",
                            numSigned(best_shift, 4).c_str(), 100.0 * gain);
                const double margin = std::isfinite(second) ? 100.0 * (second - best) / best : 100.0;
                std::printf("  相对次优 shift 领先 %.2f%%\n", margin);
            }
            if (gain < 2.0 * opt.align_margin) {
                std::printf("  ⚠ 对齐差异不显著（<2%%）：残差本身对半帧错位不敏感，但摩擦参数\n"
                            "    （fc/fv）对它很敏感 —— 建议同时跑 --align-shift=fine 与\n"
                            "    --align-shift=0/-1 比较参数稳定性。\n");
            }
        }
    }

    // ── 5) 组装方程 ────────────────────────────────────────────────────────
    std::vector<Row> rows;
    buildRows(data, best_shift, opt.held, rows);
    if (rows.size() < 50) {
        std::printf("有效方程过少（%zu），无法辨识\n", rows.size());
        return 1;
    }
    for (int eq = 0; eq < 2; ++eq) {
        size_t n = 0;
        for (const auto& r : rows) if (r.eq == eq) ++n;
        if (n < 10) { std::printf("关节 %d 的方程过少（%zu）\n", eq, n); return 1; }
    }
    std::printf("  方程数 %zu（held=%s, shift=%s）\n", rows.size(), heldName(opt.held),
                numSigned(best_shift, 4).c_str());

    const size_t NR = rows.size();
    Eigen::MatrixXd Yw = Eigen::MatrixXd::Zero(static_cast<int>(NR), kNumParams);
    Eigen::VectorXd tau_w = Eigen::VectorXd::Zero(static_cast<int>(NR));
    for (size_t r = 0; r < NR; ++r) {
        const Sample& s = data[static_cast<size_t>(rows[r].j)];
        double qd[2], qdd[2];
        effState(s, opt.held, qd, qdd);
        const double q[2] = {s.q[0], s.q[1]};
        double Y[2][kNumParams];
        regressor(q, qd, qdd, p0, makeExo(s, opt), Y);
        for (int c = 0; c < kNumParams; ++c)
            Yw(static_cast<int>(r), c) = Y[rows[r].eq][c] * w[rows[r].eq];
        tau_w(static_cast<int>(r)) = rows[r].tau * w[rows[r].eq];
    }

    // ── 6) 列级 SNR 加权 ───────────────────────────────────────────────────
    SnrInfo snr;
    if (opt.snr_weight) {
        snr = estimateSnr(data, rows, Yw, opt);
        std::printf("\n  列级 SNR（信号 σ / 噪声 σ，噪声由段内高通残差估计）:\n");
        std::printf("    %-10s %12s %12s %10s %8s   %s\n",
                    "列", "σ_signal", "σ_noise", "SNR", "w", "说明");
        for (int c = 0; c < kNumParams; ++c) {
            const char* note = "";
            if (snr.dropped[c]) note = "列被丢弃（SNR 过低 / 恒零）⇒ 该参数保持先验";
            else if (snr.snr[c] < 1.0) note = "信号弱于噪声，权重已压低";
            std::printf("    %-10s %12s %12s %10s %8.4f   %s\n", nm[c],
                        num(snr.sig[c], 4).c_str(), num(snr.noise[c], 4).c_str(),
                        num(snr.snr[c], 4).c_str(), snr.wgt[c], note);
        }
    } else {
        std::printf("\n  --no-snr-weight: 列权重全部取 1\n");
        for (int c = 0; c < kNumParams; ++c) { snr.wgt[c] = 1.0; }
    }

    // ── 7) 归一化 + SVD 截断求解 ───────────────────────────────────────────
    std::vector<double> col_norm(kNumParams, 1.0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<int>(NR), kNumParams);
    for (int c = 0; c < kNumParams; ++c) {
        double s2 = 0.0;
        for (size_t r = 0; r < NR; ++r) {
            const double x = Yw(static_cast<int>(r), c) * snr.wgt[c];
            s2 += x * x;
        }
        const double nrm = std::sqrt(s2);
        col_norm[c] = (nrm > 1e-12) ? nrm : 1.0;
        for (size_t r = 0; r < NR; ++r)
            A(static_cast<int>(r), c) = Yw(static_cast<int>(r), c) * snr.wgt[c] / col_norm[c];
    }
    Eigen::VectorXd rhs = tau_w;
    for (size_t r = 0; r < NR; ++r) {
        double pred = 0.0;
        for (int c = 0; c < kNumParams; ++c)
            pred += Yw(static_cast<int>(r), c) * phi0[c];
        rhs(static_cast<int>(r)) -= pred;
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd sv = svd.singularValues();
    const double sv_max = sv.size() > 0 ? sv(0) : 0.0;
    const double tol = std::max(1e-14, opt.trunc * sv_max);
    int rank = 0;
    for (int i = 0; i < sv.size(); ++i) if (sv(i) > tol) ++rank;

    std::printf("\n---- 求解 ----\n");
    // 列共线性诊断（|corr| 大的列对 ⇒ 参数互相抵消，SNR 高也可能不可辨识）
    {
        double mean[kNumParams] = {0}, sd[kNumParams] = {0};
        for (int c = 0; c < kNumParams; ++c) {
            double m = 0.0;
            for (size_t r = 0; r < NR; ++r) m += Yw(static_cast<int>(r), c);
            mean[c] = m / static_cast<double>(NR);
            double v = 0.0;
            for (size_t r = 0; r < NR; ++r) {
                const double x = Yw(static_cast<int>(r), c) - mean[c];
                v += x * x;
            }
            sd[c] = std::sqrt(v / static_cast<double>(NR));
        }
        struct Pair { double corr; int a, b; };
        std::vector<Pair> pairs;
        for (int a = 0; a < kNumParams; ++a) {
            for (int b = a + 1; b < kNumParams; ++b) {
                if (sd[a] < 1e-12 || sd[b] < 1e-12) continue;
                double cov = 0.0;
                for (size_t r = 0; r < NR; ++r)
                    cov += (Yw(static_cast<int>(r), a) - mean[a]) *
                           (Yw(static_cast<int>(r), b) - mean[b]);
                cov /= static_cast<double>(NR);
                pairs.push_back({cov / (sd[a] * sd[b]), a, b});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& x, const Pair& y) { return std::fabs(x.corr) > std::fabs(y.corr); });
        std::printf("  列共线性诊断（关节加权后）: ");
        if (pairs.empty() || std::fabs(pairs[0].corr) < 0.90) {
            std::printf("无 |corr| ≥ 0.90 的列对\n");
        } else {
            std::printf("\n");
            for (size_t i = 0; i < pairs.size() && i < 6; ++i) {
                if (std::fabs(pairs[i].corr) < 0.90) break;
                std::printf("    %-10s ↔ %-10s corr = %+6.3f  ⚠ 高度共线 ⇒ 二者只能定出组合量\n",
                            nm[pairs[i].a], nm[pairs[i].b], pairs[i].corr);
            }
        }
    }
    std::printf("  方程 %zu, 参数 %d, 数值秩 %d（不可辨识方向 %d 个保持先验）\n",
                NR, kNumParams, rank, kNumParams - rank);
    std::printf("  奇异值(前 8): ");
    for (int i = 0; i < std::min<int>(kNumParams, sv.size()); ++i)
        std::printf("%s%s", num(sv(i), 4).c_str(), (i + 1 == std::min<int>(kNumParams, sv.size())) ? "" : " ");
    std::printf("\n  列归一化因子 ||col||: ");
    for (int c = 0; c < kNumParams; ++c)
        std::printf("%s%s", num(col_norm[c], 4).c_str(), (c + 1 == kNumParams) ? "" : " ");
    std::printf("\n  Tikhonov λ = %g（归一化坐标）, 截断阈值 = %s\n", opt.tikhonov, num(tol, 3).c_str());

    const double lam2 = opt.tikhonov * opt.tikhonov;
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(kNumParams);
    std::vector<double> gain(std::min<int>(kNumParams, sv.size()), 0.0);
    for (int i = 0; i < sv.size(); ++i) {
        if (sv(i) <= tol) continue;
        gain[static_cast<size_t>(i)] = sv(i) / (sv(i) * sv(i) + lam2);
        const double coef = gain[static_cast<size_t>(i)] * svd.matrixU().col(i).dot(rhs);
        delta += coef * svd.matrixV().col(i);
    }

    // φ = φ0 + diag(w_snr)·diag(1/||col||)·δ
    std::vector<double> phi(kNumParams, 0.0), dphi(kNumParams, 0.0);
    for (int c = 0; c < kNumParams; ++c) {
        dphi[c] = snr.wgt[c] * delta(c) / col_norm[c];
        phi[c] = phi0[c] + dphi[c];
    }

    // 残差与 σ²
    double sse_ls = 0.0;
    for (size_t r = 0; r < NR; ++r) {
        double pred = 0.0;
        for (int c = 0; c < kNumParams; ++c) pred += A(static_cast<int>(r), c) * delta(c);
        const double e = rhs(static_cast<int>(r)) - pred;
        sse_ls += e * e;
    }
    const double dof = std::max(1.0, static_cast<double>(NR) - static_cast<double>(rank));
    const double sigma2 = sse_ls / dof;

    // 标准差: Cov(δ) = σ²·Σ gain_j² v_j v_jᵀ；截断方向记 ∞
    std::vector<double> sd(kNumParams, 0.0);
    std::vector<bool> sd_inf(kNumParams, false);
    for (int c = 0; c < kNumParams; ++c) {
        double var = 0.0;
        bool inf = false;
        for (int j = 0; j < sv.size(); ++j) {
            const double v = svd.matrixV()(c, j);
            if (sv(j) > tol) {
                const double g = gain[static_cast<size_t>(j)];
                var += sigma2 * v * v * g * g;
            } else if (std::fabs(v) > 1e-3) {
                inf = true;   // 该参数在不可辨识方向上还有分量 ⇒ 标准差 ∞
            }
        }
        const double scale = snr.wgt[c] / col_norm[c];
        sd[c] = std::sqrt(std::max(0.0, var)) * scale;
        sd_inf[c] = inf;
    }

    // ── 8) 物理约束 ────────────────────────────────────────────────────────
    std::vector<double> phi_raw = phi;
    bool clamped[kNumParams] = {false, false, false, false, false, false, false, false};
    if (phi[0] < 1e-5) { phi[0] = 1e-5; clamped[0] = true; }
    if (phi[1] < 1e-5) { phi[1] = 1e-5; clamped[1] = true; }
    for (int c = 4; c < kNumParams; ++c) {
        if (phi[c] < 0.0) { phi[c] = 0.0; clamped[c] = true; }
    }
    ModelParams p1 = p0;
    {
        double v[kNumParams];
        for (int c = 0; c < kNumParams; ++c) v[c] = phi[c];
        vectorToParams(v, p1);
    }
    p1.frictionLambda = p0.frictionLambda;   // λ 固定 10，不辨识
    p1.tau_offset_big = p0.tau_offset_big;
    p1.tau_offset_small = p0.tau_offset_small;

    // ── 9) 报告 ────────────────────────────────────────────────────────────
    std::printf("\n---- 参数 ----\n");
    std::printf("  %-10s %13s %13s %13s %13s\n", "name", "先验(默认)", "估计值", "变化量", "标准差 σ");
    for (int c = 0; c < kNumParams; ++c) {
        std::string sd_s = sd_inf[c] ? "inf(不可辨识方向)" : num(sd[c], 4);
        std::printf("  %-10s %13s %13s %13s %13s%s\n", nm[c], num(phi0[c], 6).c_str(),
                    num(phi[c], 6).c_str(), numSigned(phi[c] - phi0[c], 6).c_str(), sd_s.c_str(),
                    clamped[c] ? "   ← 已钳制" : "");
        if (clamped[c])
            std::printf("      ⚠ 物理约束: %s 越界（原始估计 %s）⇒ 钳到下限 %s，该方向不再是最优拟合\n",
                        nm[c], num(phi_raw[c], 6).c_str(), num(phi[c], 6).c_str());
    }
    if (rank < kNumParams)
        std::printf("  注: 数值秩 %d < 8 ⇒ 不可辨识方向上的估计值 = 先验值；请检查激励是否覆盖该方向。\n", rank);

    // 加权 / 未加权残差
    double sse_b = 0.0, sse_a = 0.0, ss_tau = 0.0;
    double ss_eq_b[2] = {0, 0}, ss_eq_a[2] = {0, 0}, ss_tau_eq[2] = {0, 0};
    size_t n_eq[2] = {0, 0};
    for (size_t r = 0; r < NR; ++r) {
        const Sample& s = data[static_cast<size_t>(rows[r].j)];
        double qd[2], qdd[2];
        effState(s, opt.held, qd, qdd);
        const double q[2] = {s.q[0], s.q[1]};
        double pred0[2], pred1[2];
        const ModelExo e = makeExo(s, opt);
        inverseDynamics(q, qd, qdd, p0, e, pred0);
        inverseDynamics(q, qd, qdd, p1, e, pred1);
        const double tm = rows[r].tau;
        const double e0 = tm - pred0[rows[r].eq], e1 = tm - pred1[rows[r].eq];
        sse_b += w[rows[r].eq] * w[rows[r].eq] * e0 * e0;
        sse_a += w[rows[r].eq] * w[rows[r].eq] * e1 * e1;
        ss_tau += w[rows[r].eq] * w[rows[r].eq] * tm * tm;
        ss_eq_b[rows[r].eq] += e0 * e0;
        ss_eq_a[rows[r].eq] += e1 * e1;
        ss_tau_eq[rows[r].eq] += tm * tm;
        ++n_eq[rows[r].eq];
    }
    const double rms_b = std::sqrt(sse_b / static_cast<double>(NR));
    const double rms_a = std::sqrt(sse_a / static_cast<double>(NR));
    const double rms_ls = std::sqrt(sse_ls / static_cast<double>(NR));
    std::printf("\n---- 残差（τ_meas − inverseDynamics）----\n");
    std::printf("  加权 RMS: 先验 %s → 辨识后 %s（最小二乘解 %s），实测 τ 加权 RMS %s\n",
                num(rms_b, 5).c_str(), num(rms_a, 5).c_str(), num(rms_ls, 5).c_str(),
                num(std::sqrt(ss_tau / static_cast<double>(NR)), 5).c_str());
    std::printf("  相对降低 %.1f%%\n", 100.0 * (1.0 - rms_a / std::max(1e-12, rms_b)));
    for (int eq = 0; eq < 2; ++eq) {
        if (n_eq[eq] == 0) continue;
        std::printf("  %s（未加权, N·m）: 先验 %s → 辨识后 %s（τ RMS %s）\n",
                    eq == 0 ? "大yaw " : "小yaw ", 
                    num(std::sqrt(ss_eq_b[eq] / static_cast<double>(n_eq[eq])), 5).c_str(),
                    num(std::sqrt(ss_eq_a[eq] / static_cast<double>(n_eq[eq])), 5).c_str(),
                    num(std::sqrt(ss_tau_eq[eq] / static_cast<double>(n_eq[eq])), 5).c_str());
    }

    // 用辨识出的参数复核对齐选择
    if (opt.align == "auto" || opt.align == "fine") {
        std::vector<Row> rr;
        double best = kInf, bs = best_shift;
        const double step = (opt.align == "fine") ? 0.125 : 1.0;
        for (double s = -1.0; s <= 1.0001; s += step) {
            buildRows(data, s, opt.held, rr);
            if (rr.size() < 20) continue;
            const double rms = std::sqrt(sseWeighted(data, rr, p1, opt, w, opt.held) /
                                         static_cast<double>(rr.size()));
            if (rms < best) { best = rms; bs = s; }
        }
        if (std::fabs(bs - best_shift) > 1e-9)
            std::printf("  ⚠ 复核: 用辨识参数重新搜索得到 shift=%s（当前 %s）；"
                        "如需可显式指定 --align-shift=%s 重跑\n", numSigned(bs, 4).c_str(),
                        numSigned(best_shift, 4).c_str(), numSigned(bs, 4).c_str());
        else
            std::printf("  对齐复核: 用辨识参数重搜仍是 shift=%s ✓\n", numSigned(best_shift, 4).c_str());
    }

    // 数值可积性（λ 固定 10 的检查）
    const double lam_max = recommendedFrictionLambda(p1, 0.01, true);
    std::printf("  数值可积性: recommendedFrictionLambda(dt=0.01) = %s（λ 固定 %s）%s\n",
                num(lam_max, 4).c_str(), num(p1.frictionLambda, 4).c_str(),
                (lam_max + 1e-9 < p1.frictionLambda) ? "  ⚠ λ 偏大，RK4 在 ω≈0 附近可能失稳" : " ✓");

    // 真值对比（合成数据自检）
    if (!opt.truth.empty()) {
        std::printf("\n---- 真值对比（--truth）----\n");
        std::printf("  %-10s %13s %13s %12s\n", "name", "真值", "估计值", "相对误差");
        std::string s = opt.truth;
        size_t pos = 0;
        while (pos <= s.size()) {
            const size_t c = s.find(',', pos);
            const std::string item = s.substr(pos, (c == std::string::npos ? s.size() : c) - pos);
            pos = (c == std::string::npos) ? s.size() + 1 : c + 1;
            if (item.empty()) continue;
            const size_t eq = item.find('=');
            if (eq == std::string::npos) { std::printf("  (无法解析 %s)\n", item.c_str()); continue; }
            const std::string key = trim(item.substr(0, eq));
            const double tv = std::atof(item.substr(eq + 1).c_str());
            int idx = -1;
            for (int k = 0; k < kNumParams; ++k) if (key == nm[k]) { idx = k; break; }
            if (idx < 0) { std::printf("  (未知参数名 %s)\n", key.c_str()); continue; }
            const double rel = (std::fabs(tv) > 1e-12) ? (phi[idx] - tv) / tv : (phi[idx] - tv);
            std::printf("  %-10s %13s %13s %11.3f%%\n", key.c_str(), num(tv, 6).c_str(),
                        num(phi[idx], 6).c_str(), 100.0 * rel);
        }
    }

    // ── 10) 写文件 ─────────────────────────────────────────────────────────
    {
        std::ofstream of(opt.out);
        if (of) {
            of.precision(9);
            of << "// ============================================================================\n"
               << "// identified_params.txt — 由 tools/identify_params 生成\n"
               << "// 可直接替换 include/tcbs/mpc/planar_yaw_params.h 中 defaultModelParams() 的以下 8 行\n"
               << "//\n"
               << "// 数据: ";
            for (size_t i = 0; i < opt.files.size(); ++i)
                of << (i ? ", " : "") << opt.files[i];
            of << "\n// 样本 " << data.size() << ", 方程 " << NR << ", held=" << heldName(opt.held)
               << ", align-shift=" << numSigned(best_shift, 4) << ", 数值秩 " << rank << "/" << kNumParams << "\n"
               << "// 加权残差 RMS: 先验 " << num(rms_b, 5) << " → 辨识后 " << num(rms_a, 5)
               << " N·m; 未加权 大yaw " << num(std::sqrt(ss_eq_a[0] / std::max<size_t>(1, n_eq[0])), 5)
               << " / 小yaw " << num(std::sqrt(ss_eq_a[1] / std::max<size_t>(1, n_eq[1])), 5) << " N·m\n"
               << "// 几何(未辨识): dx=" << p1.dx << ", dy=" << p1.dy
               << "; gravity_a=(" << num(opt.gravity_a[0], 5) << ", " << num(opt.gravity_a[1], 5)
               << "); frictionLambda=" << p1.frictionLambda << " (固定, 不辨识)\n"
               << "// 估计标准差 σ: ";
            for (int c = 0; c < kNumParams; ++c)
                of << nm[c] << "=" << (sd_inf[c] ? std::string("inf") : num(sd[c], 3))
                   << (c + 1 == kNumParams ? "\n" : ", ");
            of << "// ============================================================================\n";
            of << std::fixed;
            of << "    p.Jbig_eff = " << p1.Jbig_eff << ";  // 大 yaw 侧惯量（含 m_u|d|²）\n";
            of << "    p.Js       = " << p1.Js << ";  // 上装绕小 yaw 轴总惯量\n";
            of << "    p.Px       = " << p1.Px << ";  // 上装一阶矩 m_u·ρ_x (kg·m)\n";
            of << "    p.Py       = " << p1.Py << ";\n";
            of << "    p.fcBig    = " << p1.fcBig << ";  p.fvBig   = " << p1.fvBig << ";\n";
            of << "    p.fcSmall  = " << p1.fcSmall << ";  p.fvSmall = " << p1.fvSmall << ";\n";
            of << "// 注意: 以上为辨识值，使用前请确认残差与数值秩（见上），并跑 test_planar_yaw_model 回归。\n";
            std::printf("\n已写出: %s\n", opt.out.c_str());
        } else {
            std::printf("\n⚠ 无法写出 %s（目录不存在或无权限）\n", opt.out.c_str());
        }
    }

    // ── 11) 验证集 ─────────────────────────────────────────────────────────
    if (!opt.validate.empty()) {
        std::printf("\n================ 验证集 ================\n");
        for (const auto& vf : opt.validate) {
            std::vector<Sample> vd;
            LoadReport vr;
            int sc = 0;
            std::string verr;
            if (!loadCsv(vf, opt, vd, sc, vr, verr)) {
                std::printf("  %s: 读取失败: %s\n", vf.c_str(), verr.c_str());
                continue;
            }
            AccelReport var;
            computeAccel(vd, opt, var);   // 复用同一套差分/平滑参数
            std::printf("  %s: %zu 样本, %zu 段, dt中位 %s s\n", vf.c_str(), vd.size(), vr.segs,
                        num(vr.dt_med, 4).c_str());

            // 验证集上的对齐搜索（用辨识参数）
            double vshift = best_shift;
            {
                std::vector<Row> rr;
                double best = kInf;
                const double step = (opt.align == "fine") ? 0.125 : 1.0;
                for (double s = -1.0; s <= 1.0001; s += step) {
                    buildRows(vd, s, opt.held, rr);
                    if (rr.size() < 20) continue;
                    const double rms = std::sqrt(sseWeighted(vd, rr, p1, opt, w, opt.held) /
                                                 static_cast<double>(rr.size()));
                    if (rms < best) { best = rms; vshift = s; }
                }
            }
            std::vector<Row> rr;
            buildRows(vd, vshift, opt.held, rr);
            if (rr.empty()) { std::printf("    无有效方程\n"); continue; }

            double ss_f[2] = {0, 0}, ss_p[2] = {0, 0};
            size_t nq[2] = {0, 0};
            for (const auto& r : rr) {
                const Sample& s = vd[static_cast<size_t>(r.j)];
                double qd[2], qdd[2];
                effState(s, opt.held, qd, qdd);
                const double q[2] = {s.q[0], s.q[1]};
                double tf[2], tp[2];
                const ModelExo e = makeExo(s, opt);
                inverseDynamics(q, qd, qdd, p1, e, tf);
                inverseDynamics(q, qd, qdd, p0, e, tp);
                const double tm = r.tau;
                ss_f[r.eq] += (tm - tf[r.eq]) * (tm - tf[r.eq]);
                ss_p[r.eq] += (tm - tp[r.eq]) * (tm - tp[r.eq]);
                ++nq[r.eq];
            }
            std::printf("    力矩预测残差 RMS (未加权, N·m), align-shift=%s:\n", numSigned(vshift, 4).c_str());
            for (int eq = 0; eq < 2; ++eq) {
                if (!nq[eq]) continue;
                const double rf = std::sqrt(ss_f[eq] / static_cast<double>(nq[eq]));
                const double rp = std::sqrt(ss_p[eq] / static_cast<double>(nq[eq]));
                std::printf("      %s: 先验 %s → 辨识后 %s%s\n", eq == 0 ? "大yaw " : "小yaw ",
                            num(rp, 5).c_str(), num(rf, 5).c_str(),
                            (rf < rp ? "  ✓改善" : "  ⚠变差"));
            }

            // 开环前向仿真（用实测 τ 驱动；held 轴闭环保稳时误差会体现控制作用）
            if (opt.fwd_sim) {
                const int n = static_cast<int>(vd.size());
                size_t i0 = 0;
                while (i0 < vd.size()) {
                    size_t i1 = i0;
                    while (i1 + 1 < vd.size() && vd[i1 + 1].seg == vd[i0].seg) ++i1;
                    if (i1 > i0) {
                        for (int which = 0; which < 2; ++which) {
                            const ModelParams& pp = (which == 0) ? p0 : p1;
                            double q[2] = {vd[i0].q[0], vd[i0].q[1]};
                            double qd[2] = {vd[i0].qd[0], vd[i0].qd[1]};
                            double ss[2] = {0, 0}, mx[2] = {0, 0};
                            size_t cnt = 0;
                            for (size_t k = i0; k < i1; ++k) {
                                const double fi = static_cast<double>(k) - vshift;
                                const int it0 = std::min(std::max(static_cast<int>(std::floor(fi)), 0), n - 1);
                                const int it1 = std::min(it0 + 1, n - 1);
                                const double fa = std::min(std::max(fi - std::floor(fi), 0.0), 1.0);
                                const double tau_in[2] = {
                                    vd[static_cast<size_t>(it0)].tau[0] * (1.0 - fa) + vd[static_cast<size_t>(it1)].tau[0] * fa,
                                    vd[static_cast<size_t>(it0)].tau[1] * (1.0 - fa) + vd[static_cast<size_t>(it1)].tau[1] * fa};
                                const double dt = vd[k + 1].t - vd[k].t;
                                if (!(dt > 1e-6)) continue;
                                const int sub = std::max(1, static_cast<int>(std::lround(dt / 0.002)));
                                double qn[2], qdn[2];
                                integrateStep(q, qd, tau_in, pp, makeExo(vd[k], opt), dt, sub, qn, qdn);
                                for (int a = 0; a < 2; ++a) {
                                    q[a] = qn[a]; qd[a] = qdn[a];
                                    const double e = q[a] - vd[k + 1].q[a];
                                    ss[a] += e * e;
                                    mx[a] = std::max(mx[a], std::fabs(e));
                                }
                                ++cnt;
                            }
                            if (which == 0)
                                std::printf("    前向仿真（开环、用实测 τ 驱动）角度误差 RMS / max [rad]，顺序 大yaw / 小yaw:\n");
                            if (!cnt) continue;
                            std::printf("      用%s参数:   RMS %s / %s   max %s / %s\n",
                                        which == 0 ? "先验  " : "辨识后",
                                        num(std::sqrt(ss[0] / static_cast<double>(cnt)), 4).c_str(),
                                        num(std::sqrt(ss[1] / static_cast<double>(cnt)), 4).c_str(),
                                        num(mx[0], 4).c_str(), num(mx[1], 4).c_str());
                        }
                    }
                    i0 = i1 + 1;
                }
            }
        }
    }

    std::printf("\n完成。\n");
    return 0;
}
