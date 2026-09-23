# TorqueControllerForBigSmallYaw — 双级 yaw 云台力矩 MPC 控制

两级平行 yaw 的云台控制系统（**大 yaw 可多圈自由转、响应慢；小 yaw 行程 ±35°（软限位 ±30°）、响应快；
`IMU 固定在头上`（构型 `ON_HEAD`，默认；`ON_BIG_YAW` 为备选，运行时切换）**）。
上位机通过串口与电控（MCU）和 IMU 通信，
完成状态估计与**耦合非线性 MPC** 求解，输出两关节力矩（可选叠加电控内环）。

> 本工程是 `TorqueController`（单 yaw 版本）的改版：模型从"单自由度 + J/τ_c/b"升级为
> **平面 2 自由度的严格刚体动力学**（8 个可辨识参数：4 个摩擦 + 2 个惯量 + 2 个一阶矩，
> 两 yaw 的非共轴偏置带来的交叉惯量与离心项、底盘转动耦合、重力项全部显式建模；
> 推导与化简前提见 `docs/model.md`），接口从"一个 yaw 目标"升级为
> **大/小 yaw 两个世界方位角目标**，状态从"融合 yaw 位置/速度"升级为
> **可信量 + 大 yaw 延迟补偿估计 + 反解真实位姿 + 数据来源**。
> 详细差异见 §10。

---

## 0. 每台新车必须标定的参数（总清单）

> **换一台车，下面 A/B/C/D 四类全部要重做**（E 类一般沿用）。详细步骤与判据见
> `docs/calibration.md`（§2 参数总表、§3 运动学、§4 参数辨识、§5 力矩常数、§6 顺序）；
> 本节是"一张纸的清单"，含**必须自己手工测量**的量。

### 0. 先决条件: 串口链路自检（不是标定，但不过这一步后面全是白做）

```bash
./build/tcbs_test_serial --list      # 两个口选对了吗？（iProduct 是否互不相同）
./build/tcbs_test_serial --selftest  # 包布局/字段偏移/CRC8/安全不变量（纯软件）
./build/tcbs_test_serial             # 实车: 100 Hz 发零力矩帧 + 逐帧打印协议字段
```

`--list` 里 MCU / IMU 两列都必须有 `YES`；逐帧打印必须出现 `[MCU #n]`（说明 CRC8 与
字段偏移都对得上）；1 Hz 统计行里 **`MCU2 新样本 n/s`** 就是 §3.1 要标的
`transport_delay_s` 所依赖的那条低速链路的真实刷新率。安全约定与常见坑见
`docs/calibration.md` §3.0（含 `--wait=0` 跳过预检、`--no-send` 纯监听）。

### A. 手工测量（不进辨识，但缺一不可）

| 量 | 怎么测 | 精度 | 填到哪 |
|---|---|---|---|
| **两 yaw 轴平面偏置 `d = (dx, dy)`** | 卡尺/三坐标量两轴中心距与方向（A 系 x-y 平面内）。**本构型已实测 = `(0, 0.07)` m**（横向无偏置、小 yaw 轴在大 yaw 轴**前方** 0.07 m） | ±0.5 mm | `ModelParams::dx/dy`（`planar_yaw_params.h`，**已按实测填好**） |
| **小 yaw 实际行程两端角度** | 手动（力矩 0）转到两侧机械限位，读编码器；确认 `±35°`（软限位 ±30°；也可在 `./build/tcbs_test_serial` 里读） | ±0.2° | `defaultMpcConfig().small.min_angle/max_angle` + 电控宏 `YAW_SMALL_MIN/MAX_RAD` |
| **两关节力矩能力**（峰值力矩 × 减速比 × 效率） | 电机手册 + **实测堵转/斜坡**（不要只信手册） | — | `big/small.max_torque`、`max_torque_rate`（**直接决定控制权限与安全**） |
| （可选）上装质量 `m_u` | 电子秤 | ±10 g | `ModelParams::m_u_known`（不称重填 0，代价见 §8-8） |
| （可选）上装质心偏置 `ρ` | 吊线/称重法 | ±2 mm | 仅用于核对辨识出的 `P`（`P = m_u·ρ`），**不填进模型** |

### B. 编码器/指令映射（`McuDataPreprocessor::LinearParams`）

| 项 | 标定方法 | 目标精度 |
|---|---|---|
| `recv_pitch_*` / `send_pitch_*` | ★ `./build/tcbs_pitch_calibration --points=20 --min=<原始单位下限> --max=<原始单位上限>`（两段线性拟合：`recv_*`: 电控原始值→关节角，`send_*`: 关节角→下发值；无硬件先跑 `--sim`/`--selftest`）。**前提: IMU 临时装到头上**（`ImuLocation::ON_HEAD`）—— 否则 `imu.euler_pitch` 不是 pitch 关节角、结果无效（§3.3） | 0.2° |
| `recv_small_yaw_scale` | **临时 head IMU 法**（`docs/calibration.md` §3.3）——大 yaw 静止、小 yaw 慢速三角波 | 0.05° |
| `recv_small_yaw_offset`（**零位**） | ★ **在串口测试里直接读**：跑 `./build/tcbs_test_serial`（力矩恒 0），人工把小 yaw 摆到**机械零点**，读它打印的 `yaw_small_angle`（电控原始弧度），**取负**写进 `recv_small_yaw_offset`；`send_small_yaw_offset` 取**相反符号**（下发与上报同一原始坐标系）。**不需要单独的标定程序**。重复性 = 人工摆放的可重复性 | 0.2~1°（取决于人工重复性） |
| `recv_big_yaw_scale/offset` | **IMU 法**（§3.1）：底盘静止时 `Δ(IMU 方位角) == Δ(关节角)` | 0.05° |
| `send_*_torque_scale` | 力矩常数独立校核（§5：已知惯量体或吊质量块测稳态力矩） | 1% |

### C. 状态估计器（`YawStateEstimator::Config`）

| 项 | 标定方法 | 备注 |
|---|---|---|
| `imu_location` | 装配决定（**默认 `ON_HEAD`**：IMU 在头上；备选 `ON_BIG_YAW`） | 运行时可切换，一份二进制支持两种构型 |
| `mount_yaw/pitch/roll`（或 `head_mount_*`） | **静止时**用 IMU 加速度计把安装倾斜标到 0.1°；yaw 部分按约定（"机械零位处 x 轴指向世界 +x ⇒ 方位角 0"） | 重力方向直接乘这个矩阵 ⇒ 直接影响 `P` 的辨识 |
| `transport_delay_s` | §3.1：用 `big_enc_innovation` 与 `big_enc_age` 在线校核（默认 15 ms） | MCU 无时钟，只标传输时延 |
| `bore[3]` | §3.4：激光/照准器或相机像素反解视轴方向 | 默认 `(0,1,0)`（本工程 x=右/y=前/z=上，pitch 绕 x ⇒ 光轴在 y-z 平面） |
| `chassis_imu_timeout_s`、`small_rate_lpf_alpha`/`big_rate_lpf_alpha`、`stale_age_s`… | 保持默认，按实测噪声/带宽微调 | 不影响正确性，只影响平滑度 |

### D. 动力学参数（16 个，必须**辨识**，不要手填）

| 参数 | 含义 |
|---|---|
| `Jbig_eff`, `Js` | 大 yaw 侧惯量（含 `m_u\|d\|²`）、上装绕小 yaw 轴惯量（含 `m_u\|ρ\|²`） |
| `Px`, `Py` | 上装一阶矩 `m_u·ρ` |
| `fc_big/fv_big/fc_small/fv_small` | 两轴库仑/粘滞摩擦 |
| ★ `backlash_delta` | 大 yaw **背隙宽度**（唯一可离线标定的背隙量；每台车都要重标） |
| ★ `backlash_k`, `backlash_c` | 背隙接触刚度/阻尼（**当前是占位值**，等 3-DOF 拟合给实测值） |
| ★ `Jmotor`, `fc_motor`, `fv_motor` | 大 yaw 电机侧等效惯量/库仑/粘滞摩擦（同为占位） |
| ★ `backlash_through` γ | 死区直通线性项（给死区内提供梯度）—— **默认冻结在 0.002、不参与拟合**（`--no-freeze-backlash-through` 可放开） |
| ★ `backlash_beta` β | 死区**中心**偏置 —— **运行期由估计器在线给**（`Estimate::backlash_center`），不要用离线常数 |

> 全套 16 参的含义、δ 与 β 为什么一个能离线一个不能、底盘 IMU 的延迟补偿、MPC 接线、
> 以及"哪些是实测值、哪些还是占位"的可信度表，见 **`docs/backlash_model.md`**；
> 该文还给了**仿真环境 2**（`--sim-rigid`: 接触完全刚性 + 死区完全自由 + β 随机/漂移）与
> 100 段 × 1000 epoch 的完整验证结果（收敛曲线 / 模型 vs 环境 / 控制 vs 目标三条曲线）。

采集与拟合：

```bash
# ① 采集（分轴：一轴激励、另一轴 PID 保持；pitch≡0；100 Hz；建议全程固定同一个 10° 静态倾角）
python3 python/scripts/collect_sysid.py --tag=big   --segments=6 --tilted --held-big-stratified
python3 python/scripts/collect_sysid.py --tag=small --segments=6 --tilted
# ↑ 默认还会把每个"静止保持段"也落盘（文件名后缀 _hold，首段除外）并参与辨识 ——
#   它补的是采样轨迹里稀缺的**大角度阶跃**激励；不要就用 --no-record-hold
# ★ 默认**只写 npz**（列是 csv 的超集、体积约为一半；辨识只读 npz，同名成对时也只读 npz）；
#   确实要留一份 csv 给人看/给老脚本用，加 --save-csv
# ①' 只采**测试数据**（无硬件）: 用"刚性接触 + 死区完全自由 + β 每条数据随机/漂移"的仿真环境
#    python3 python/scripts/collect_sysid.py --dry-run --sim-rigid --segments=100 \
#            --duration-sec=3 --no-record-hold --out=/tmp/rigid
# ② 拟合（torch 输出误差法；λ 固定 100，不改。拟合 15 参：平面 8 + 背隙/电机侧 8 里去掉了
#   **默认冻结**的直通项 γ；保持段默认只取前 3 s，见 --hold-max-sec）
#   需要数据里同时有 theta_big_motor（电机侧）与 theta_big_platform（云台侧）两列
#   δ 的独立校验/初值: python3 python/scripts/calibrate_backlash.py --data='data/sysid/*.npz'
# ★ 几何已按实测填好（(dx, dy) = (0, 0.07)），实机数据**不需要**再给 --dx/--dy；
#   只有换机械或跑旧归档数据（用 (0.10, 0) 生成的那批）时才显式覆盖
python3 python/scripts/identify_params_torch.py --data='data/sysid/*.npz' \
        --truth-params=<若有真值> --epochs=1000
# ③ 结果填进 include/tcbs/mpc/planar_yaw_params.h 的 defaultModelParams()（或运行时 setModelParams）
```

要点：**两轴力矩都必须记录**（被保持轴的力矩是 `P` 的观测量）；**小 yaw 要尽量用满行程**；
**水平数据下 `Px/Py` 不可辨识** ⇒ 必须加**静态倾斜段**（固定一个倾角贯穿全程即可，
方向多样性由"各段 held 大 yaw 方位角不同 + 大 yaw 自身被激励"提供）；
判据看 `|Px|/σ ≥ 3`，以及**留出段**的前向仿真 RMSE（不是拟合残差）。

### 结果放哪里（沿用原仓库约定）

标定结果**一车一目录**：`data/cars/<车名>/`（`LinearParams.txt` + `params/Identified_parameters.txt`
+ `params/Figure_1.png` + `sysid_samples/*.npz`，约定见 `data/cars/README.md`）。
`data/sysid/` 是**工作区**（也在版本管理内，但定型后请归到对应车目录）；本轮仿真验证数据归档在 `data/archive/20260915_sim_sysid/`。

### E. MPC / 控制调参（不是标定，但每台车要过一遍）

`N`、`substeps`（λ=100 时默认 4）、`max_iter`、各权重 `w_*/r_*/rd_*`、软限位比例、
回中中心（默认 = 行程中心）、积分增益。**力矩上限与变化率必须来自 A 的实测值**。
自检：`cd build && ctest --output-on-failure`（模型/估计/闭环）+ `./tcbs_control_demo --dur=10`。

---

## 1. 快速开始

```bash
# 依赖: Linux, libudev, Eigen3, Ceres Solver, g++ (C++17), python3（绑定/脚本只用标准库）
bash build.sh                 # = cmake + make -j$(nproc)

# 测试（含模型验证 / 状态估计 / 闭环仿真，无需硬件）
cd build && ctest --output-on-failure
./tcbs_test_planar_yaw_model        # 平面 8 参模型验证（独立实现比对/回归矩阵/能量）
./tcbs_test_yaw_state_estimator     # 用真实时钟跑约 4s
./tcbs_test_dual_yaw_mpc            # 闭环仿真（λ=100 被控对象 vs λ=10 MPC 模型）

# 实车 —— ★ 第一步永远是串口链路自检（移植自原仓库 test_serial）
./tcbs_test_serial --list                    # 枚举串口 + 打印 MCU/IMU 会选中哪个口（无需硬件）
./tcbs_test_serial --selftest                # 纯软件自检（包布局/字段偏移/CRC8/安全不变量）
./tcbs_test_serial --no-send                 # 实车最安全: 只监听不发，逐帧打印收到的数据
./tcbs_test_serial --dur=10 --imu --raw      # 发零力矩帧 + 打印 MCU/IMU 每帧 + 十六进制原文

./tcbs_control_demo --dur=10                 # 正弦方位角跟踪

# 参数辨识: 采集（Python，无硬件可 --dry-run）+ 拟合（C++ 线性最小二乘 / torch 可导仿真）
python3 python/scripts/collect_sysid.py --tag=big --segments=6      # 大 yaw 被激励
python3 python/scripts/collect_sysid.py --tag=small --segments=6    # 小 yaw 被激励
python3 python/scripts/identify_params_torch.py --data='data/sysid/sysid_*.npz'   # 唯一辨识路径

# pitch 映射标定（两段线性拟合）: ★ 需把 IMU 临时装到头上（ON_HEAD）
./tcbs_pitch_calibration --sim                # 无硬件自检（虚拟台架 + 断言，秒级）
./tcbs_pitch_calibration --selftest           # 纯数学自检（拟合核心）
./tcbs_pitch_calibration --help               # 选项与构型警告
./tcbs_pitch_calibration --points=20 --min=<原始单位下限> --max=<原始单位上限>   # 实车
# ↑ 默认范围 -10/+30，单位 = **电控 pitch 原始值**（与原仓库相同；跨度 > 100 会拒绝开跑）
```

产物（`build/`）: `libtcbs_robot_comm_c.so`（C++/C/Python 动态库）、`libtcbs_communication.a`（静态库）、
`tcbs_control_demo`、`tcbs_mpc_param_eval`、`tcbs_pitch_calibration`、`tcbs_test_serial`（实车链路自检，**不注册 ctest**）、四个 ctest 测试程序。

> **作为子模组嵌入父工程**（模块标识 `tcbs`）: 本仓库的全部对外名字都带 `tcbs_` 前缀，
> C++ 代码整体在 `namespace tcbs` 内，头文件一律走 `#include "tcbs/..."`，
> 因此可与同源的 `TorqueController`（单 yaw 版）等子模组**共存于同一个 CMake 工程、同一个进程**。
> 约定与父工程侧改法见 [`docs/embedding.md`](docs/embedding.md)；父工程用
> `add_subdirectory(<本目录> <binary_dir> EXCLUDE_FROM_ALL)` 引入后，
> 链接 `tcbs::robot_comm_c`（或 `tcbs::communication`）即可。

---

## 2. 系统拓扑与反馈通道特性（决定了整套估计/滤波设计）

```
                    ┌───────────── 直连（可视为实时） ─────────────┐
   上位机 ──────────┤ MCU1（控制 pitch + 小 yaw）                  │
        │           └──────────────┬───────────────────────────────┘
        │                          │  MCU1 ↔ MCU2 内部链路：**不稳定**
        │                          │  · 更新率远低于帧率、间隔不规则
        │                          │  · 未收到新数据时 MCU1 一直沿用旧值
        │           ┌──────────────┴───────────────┐
        └───────────┤ MCU2（控制 大 yaw + 底盘 IMU） │
   （IMU 直连）      └──────────────────────────────┘
```

由此，**上位机收到的同一帧里，各通道的"新鲜程度"完全不同**:

| 通道 | 更新特性 | 上位机的处理 |
|---|---|---|
| 小 yaw 角/角速度、pitch 角 | 每帧新值（实时可信，MCU1 直控） | 直接当实时量；并用其角速度外推消除帧周期内的采样滞后 |
| **大 yaw 角/角速度** | 经 MCU2: **约 10Hz、间隔不规则、值被保持** | 靠 `mcu2_seq`（MCU2 每送来一次新数据 +1）识别新样本 → 用 IMU 角速度做**一阶（速度）外推**把该值推到当前时刻；保持期间用 IMU 速率继续积分 |
| **底盘 IMU 角/角速度** | 同上（值被保持） | **零阶保持**（底盘角速度变化缓慢，刻意不做加速度级外推、避免发散）；另上报年龄供上层判断 |

> **不使用任何电控侧时钟**（MCU 端计时不可用，只有上位机能计时）。协议因此带一个
> **MCU2 新样本序号 `mcu2_seq`**（MCU1 每从 MCU2 取到一次新数据就 +1；大 yaw 与底盘 IMU
> 同源、在同一次取数里一起刷新，故共用一个序号；值被保持时该序号不变）。
> 判定规则: **首帧到达 或 序号变化 或 值变化** ⇒ 新样本（不做可用性检测、无 0 哨兵）。
> 值的**年龄由上位机自己的时钟测量**（从首次看到该序号起算 + 传输时延），并通过
> `st.est.big_enc_age`、`st.est.big_sample_interval`、`prov.*.stale` 上报。
> **外推阶数最高到速度（一阶）**：位置用角速度外推、角速度只做零阶保持。

---

### 2.1 架构与数据流（一张图看全链路）

```
                  ┌──────────────────────────── 串口 1: MCU (CRC8, 协议 v0x03) ─────┐
                  │  发: 两关节 模式位 + θ* + ω* + τ, pitch 目标, 自瞄/火控        │
                  │  收: 大yaw角(多圈,有延迟/误差) 小yaw角(可信) pitch(可信) 底盘IMU │
                  ▼                                                                 │
  ┌──────────────────────────────┐                                                  │
  │ McuDataPreprocessor          │  ← 所有"编码器/指令 → 角度"映射参数都在这里       │
  └──────────────┬───────────────┘                                                  │
                 ▼                                                                   │
  ┌──────────────────────────────┐        ┌────────────────────────────┐             │
  │ YawStateEstimator            │◀───────│ 串口 2: IMU (CRC32)        │             │
  │  · IMU 在大 yaw 上 → 平台世界方位角/角速度（实时可信）              │             │
  │  · 小 yaw / pitch 编码器 → 实时可信量（按角速度外推补采样滞后）      │             │
  │  · 大 yaw 编码器（延迟+误差）→ 用 IMU 速率前推做延迟补偿            │             │
  │  · 可信量 + 标定参数 → 反解 head 真实位姿 / LOS 方位俯仰            │             │
  │  · 模型外生量: 底盘 ω_c、重力在关节系的分量 g_C、pitch 角/角速度/角加速度 │        │
  │  · Provenance: 每个源的 valid/age/计数/是否被使用（"所用数据"）      │             │
  └──────────────┬───────────────┘                                                  │
                 ▼                                                                   │
  ┌──────────────────────────────┐   后台线程 (loop_period, 默认 100Hz)              │
  │ McuMpcController             │                                                   │
  │  · 组装 MPC 输入（含 pitch 外生量、底盘 ω、重力）                                 │
  │  · tcbs::DualYawMpc::solve()  → 两关节力矩 + 第一步预测(θ*, ω*)                         │
  │  · 可选逐关节积分补偿                                                             │
  │  · 组包发送 ──────────────────────────────────────────────────────────────────────┘
  └──────────────────────────────┘
                 ▲
  tcbs::RobotController::set(ψ_big*, ψ_small*, pitch*, fire*)   ← 外部只给两个世界方位角
```

`RobotController` 把上述全部封装成一个类，外部只需 `set()` / `getState()`。

---

## 3. 对外 C++ 接口（`include/tcbs/RobotController.h`）

> 所有对外类型都在 `namespace tcbs` 内: `tcbs::RobotController`、`tcbs::RobotCommunication`、
> `tcbs::YawStateEstimator`、`tcbs::McuDataPreprocessor`、`tcbs::McuMpcController`、
> `tcbs::FrameRateCounter`、`tcbs::dual_yaw::ModelParams`、`tcbs::rot::Mat3`、
> `tcbs::mcu::SendPacket` / `tcbs::imu::ReceivePacket`。
> C ABI 头 `include/tcbs/c_api/RobotCommunicationC.h` 是**例外**（保持合法 C11 + `extern "C"`，
> 不进命名空间），其函数/类型/宏分别带 `tcbs_` / `Tcbs` / `TCBS_` 前缀。

### 3.1 构造

```cpp
#include "tcbs/RobotController.h"

using namespace tcbs;   // 也可逐处写 tcbs:: 限定名（下面按清晰起见混用）

tcbs::RobotController::Config cfg;
cfg.model       = tcbs::dual_yaw::defaultModelParams();   // 几何/惯量/摩擦（标定后替换）
cfg.mpc         = tcbs::dual_yaw::defaultMpcConfig();     // N/权重/限位/积分器
cfg.estimator   = tcbs::YawStateEstimator::Config{};      // IMU 安装旋转、大yaw延迟、视轴
cfg.mcu_linear  = tcbs::McuDataPreprocessor::LinearParams{};  // 编码器/执行器映射
cfg.controller  = tcbs::McuMpcController::Config{};       // loop 周期、模式位、积分补偿
cfg.sequence_mode = false;                          // 序列模式需在构造时选定
RobotController rc(cfg);                            // 内部自动启动后台线程
```

### 3.2 设置目标（**世界方位角**语义，分别设置两个轴）

- `big_yaw_azimuth`   —— 大 yaw 平台 x 轴的世界方位角 `ψ_big`（IMU 直测，多圈连续）
- `small_yaw_azimuth` —— 小 yaw 输出 x 轴的世界方位角 `ψ_small = ψ_big + θ_small`

两者之差就是小 yaw 的关节角指令。**快慢分配由调用方决定**:

| 调用方意图 | 设 `ψ_big*` | 设 `ψ_small*` |
|---|---|---|
| 小 yaw 承担全部快速运动、大 yaw 不动 | = 当前 `ψ_big` | = 目标方位角 |
| 大 yaw 承担全部（小 yaw 保持 0°） | = 目标方位角 | = 目标方位角 |
| 快速段小 yaw 追、随后大 yaw 展开回中 | 目标方位角的慢速滤波 | 目标方位角 |

```cpp
// 单目标
rc.set(/*auto_aim_enable=*/true,
       /*big_torque_only=*/false,      // 模式位: false = 力矩 + 电控位置/速度内环
       /*small_torque_only=*/false,
       psi_big_target, psi_small_target,
       /*pitch_target_angle=*/0.1, /*fire=*/false,
       /*integral_enable=*/true);

// 序列（sequence_mode=true 时可用）: 各通道独立序列，不截断；空序列=该通道保持当前值
rc.set(true, false, false, big_seq, small_seq, pitch_seq, fire_seq, true);

// 便捷: 以**关节系**角度设置（内部按当前底盘方位角估计换算成世界方位角）
rc.setJointAngles(true, false, false, big_joint_angle, small_joint_angle, 0.1, false, true);
```

> 推荐给**未来 N 步**的参考序列（`mpc.dt_control × N` 时长），MPC 才能提前规划减速与展开；
> 若只给当拍常数参考，等价于每拍给一个阶跃。`tcbs::McuMpcController::Config::ref_delay_steps`
> 保留旧版"目标延迟前瞻"语义（默认 0 = 不延迟）。

### 3.3 读取状态（`getState()`，按来源分组）

```cpp
auto st = rc.getState();

st.mcu     // MCU 原始反馈（已映射）: 大/小 yaw 角与角速度、pitch、底盘 IMU、温度、tick
st.imu     // 大 yaw 上 IMU 的原始数据: 欧拉角、陀螺、加速度、dt
st.strict_pose  // ★ 严格反解数据包（与 mcu/imu/est/mpc 并列；见 §3.3.1）

st.est     // ★ 状态估计（本次改版重点）
  .valid, .imu_yaw/pitch/roll
  .platform_azimuth        // ψ_big（IMU 直测，实时无延迟）
  .platform_rate
  .small_joint_angle/rate  // θ_small（可信）
  .pitch_joint_angle/rate  // θ_pitch（可信）
  .big_joint_angle_meas    // 大 yaw 原始测量（可能被保持，偏旧）
  .big_joint_angle/rate    // ★ 延迟补偿后的关节角估计（控制用）
  .big_enc_age             // ★ 该值的**年龄**（秒，上位机计时）: 保持期间持续增大
  .big_sample_interval     // 最近两次新样本间隔（s）→ 反映 MCU1↔MCU2 链路状况
  .chassis_imu_age         // 底盘 IMU 值的实测年龄
  .big_enc_innovation, .big_has_encoder
  .head_world_yaw/pitch/roll   // ★ 由可信量严格反解的真实 head 姿态
  .small_output_azimuth        // ψ_small（反解）
  .los_azimuth, .los_elevation // 视轴方向（bore 参数决定语义）
  .chassis_azimuth, .chassis_yaw_rate
  .base_omega[3]           // 底盘角速度（关节参考系）
  .gravity_a[3]            // ★ 重力矢量在 **A 系(大 yaw 转子系)** 的投影 —— 底盘倾斜显式进入模型
  .pitch_acc
  .prov                    // ★ "所用数据": 每个源 valid/age/new_samples/count/rejected/stale、
                           //    used_mask、年龄与残差、反解是否全部由可信量完成

st.mpc     // 控制输出/参考/预测/性能
  .torque[2], .torque_mpc[2], .integral[2]
  .target_joint[2], .target_joint_rate[2]   // 发给电控的 θ*/ω*
  .big_torque_only, .small_torque_only
  .ref_azimuth[2], .delayed_ref_azimuth[2]
  .ref_azimuth_seq[2], .pred_azimuth_seq[2], .pred_joint_seq[2]   // 长度 N
  .small_ref_over_limit
  .solve_ms, .loop_fps, .ticks_since_set, .solve_count, .solve_fail_count
  .estimator_valid, .sent_ok
```

#### 3.3.1 `st.strict_pose` —— 严格反解数据包（`include/tcbs/common/StrictPose.h`）

把"**IMU 是准确值**"这件事做成一个**自包含的数据包**（与 `mcu/imu/est/mpc` **并列**的顶层包）：
用 IMU 姿态反解**底盘**姿态，并把反解用到的**每一项数据**一起打包，
使得外部**只凭这一包**就能重构整车姿态：

```cpp
auto sp = rc.getState().strict_pose;
sp.imu_euler_yaw/pitch/roll      // ① IMU 原始姿态（世界←IMU, ZXY）——唯一绝对基准
sp.imu_location;                 //   构型: 0 = IMU 在大 yaw 转子上, 1 = 在头上（默认）
sp.big_joint_angle;              //   θ_b（延迟补偿估计值）＋ sp.big_joint_angle_age
sp.small_joint_angle;            //   θ_s
sp.pitch_joint_angle;            //   θ_p
sp.mount_*; sp.head_mount_*;     // ② 反解用到的安装矩阵参数（快照，便于外部复算）
sp.R_world_imu[9];               // ③ IMU 姿态矩阵（由 ① 重构，与 IMU 数据一致）
sp.R_world_chassis[9];           // ★ 反解出来的底盘姿态矩阵
sp.chassis_euler_yaw/pitch/roll; //   同上的 ZXY 欧拉角（wrap 到 (−π,π]）
sp.platform_azimuth / chassis_azimuth / head_azimuth;  // x 轴世界方位角（与 Estimate 同定义）
sp.R_world_platform[9]; sp.R_world_head[9];            // 大 yaw 转子 A 系 / 头 H 系
sp.recon_err_rot;                // ④ 自洽性自检: 用包内数据重构 IMU 姿态的残差（实测 ~1e-16）
```

> **与原仓库同款设计约定**: **没有 `valid` 标志、始终解算** —— 任何时刻读它都有意义；
> 所需数据缺失时用**历史值或 0** 参与（IMU 未到达 ⇒ 欧拉角 0 ⇒ 单位阵；关节角未到达 ⇒ 0；
> 安装参数来自配置恒有值），因此**重构关系恒成立**（连默认构造的空包都成立，测试有断言）。
> 所有角度 wrap 到 (−π, π]。

重构公式（`tcbs::dual_yaw::strictPoseReconstructImu(sp)` 就是它）：

```
构型 0（默认）: R_world_imu = R_world_chassis · Rz(θ_b) · R_A_IMU
构型 1（头上）: R_world_imu = R_world_chassis · Rz(θ_b) · Rz(θ_s) · Rx(θ_p) · R_H_IMU
```

**保证**：忽略浮点误差，用本包重构出的 **IMU 所在位置**的姿态**严格等于** IMU 实际数据
（`tcbs_test_yaw_state_estimator` 两种构型都断言 `< 1e-12`，实测 6e-16 / 8e-16）。
**注意**：反解出的**底盘**姿态还要用到 θ_b（来自"延迟 + 被保持"的大 yaw 编码器链路，
已做 IMU 速率一阶外推），因此它的误差就等于 θ_b 的估计误差（测试场景里 1.2e-2 rad），
包里同时给出 `big_joint_angle_age` 供外部判断。
顺带一个改进：`chassis_azimuth` 现在由运动学链矩阵精确算出（而不是 `ψ_platform − θ_b`
的标量近似），底盘有俯仰/横滚倾斜时后者会有约 `pitch·roll` 的误差（8°/5° 下 ~0.01 rad）。
本包只含**姿态**；整车平动无绝对观测量，不在包内。

### 3.4 在线改参数（标定/调参，不必重编译）

```cpp
rc.setLinearParams(linear);            // 编码器/执行器映射
rc.setEstimatorConfig(est_cfg);        // IMU 安装旋转、大 yaw 延迟、视轴
rc.setModelParams(model);              // 几何/惯量/摩擦（会复位 MPC 热启动）
rc.setMpcConfig(mpc_cfg);              // 权重/限位/N/积分器
rc.communication(); rc.mcuMpc(); rc.estimator();   // 子系统直访问
```

---

## 4. 耦合非线性 MPC（`DualYawMpc`）

优化问题: 决策变量 = 两关节力矩增量序列（`2N` 个），状态 = 两个 yaw 关节角/角速度，
pitch 与底盘量作为外生量按步刷新。

```
min Σ_k  w_b·|ψ_big(k)   − ψ_big*(k)|_smooth      ← 大 yaw 世界方位角跟踪
       + w_s·|ψ_small(k) − ψ_small*(k)|_smooth    ← 小 yaw 世界方位角跟踪（瞄准）
       + w_c·(θ_small(k) − small_center_angle)²   ← 冗余自由度回中（打破多解；中心 = 行程中心
                                                     0°）
       + w_lim·ρ(θ_small(k))                      ← 小 yaw 软限位（**双侧** 4 次幂障碍）
       + r_b·u_b² + r_s·u_s²                      ← 力矩惩罚
       + rd_b·Δu_b² + rd_s·Δu_s²                  ← 力矩变化率惩罚
s.t.   |u| ≤ max_torque（内部 clamp，硬限位）
       |Δu| ≤ max_torque_rate·dt（硬约束，代价函数内 clamp）
       ψ_big(k)   = ψ_chassis + ω_chassis·t_k + θ_big(k)     ← 底盘转动在窗内线性外推
       ψ_small(k) = ψ_big(k) + θ_small(k)
       动力学: 严格模型（见 docs/model.md），RK4 / 半隐式欧拉离散
```

要点:
- **两关节的相互影响被显式建模**（非共轴偏置 `d` 与上装一阶矩 `P` 引起的交叉惯量
  `M12(θ_s)=Js+d·R(θ_s)P`、离心/科氏项、底盘转动耦合、重力项），因此"大 yaw 快速展开时
  小 yaw 被带偏"这类现象由 MPC 直接补偿，而不是靠事后调参掩盖；
- **小 yaw 行程由 `small.min_angle/max_angle` 给出（当前对称 ±35°，软限位 ±30° ⇒ `small_limit_soft_ratio ≈ 0.9286`）**，软限位**两侧各自**从硬限位向内推导：
  `inset = (1−ratio)·(max−min)` ⇒ `soft_min = min + inset`、`soft_max = max − inset`
  （默认 `ratio=0.75`、`inset=11.25°` ⇒ 软限位区 `[−13.75°, +8.75°]`）。
  旧的 `soft = ratio·max_angle` 隐含"行程对称 ±max_angle"：行程一旦非对称，负侧会被算成
  `−0.75·|max| = −15°`、障碍宽度算成 5°（真实的施工余量是 11.25°），负侧行程根本用不满；
- **回中目标角 `small_center_angle` 是显式配置量**（`defaultMpcConfig()` 取行程中心
  `0.5·(min_angle+max_angle)`，当前对称行程下 = 0°），不是内部隐式平均 —— 换非对称行程会自动跟着变；
- **小 yaw 限位是硬性安全约束**: 代价里的双侧障碍项 + 电控侧硬限位（§8）双重保障；
- **求解器**: Ceres + 动态自动微分（模型模板化，梯度精确）。
  注意: **不要用 Ceres 的参数箱式边界**——边界会强制切换到线搜索，单次求解的函数求值次数
  上升约 6 倍；本实现把变化率约束放进代价函数内部 clamp，从而可用 LM 信任域。

### 4.1 实测性能（本机, 单核）

平面 8 参模型的动力学生成成本远低于原 40 参版本，单次求解进入亚毫秒级（`tcbs_test_dual_yaw_mpc` §9/§10）:

| 配置 | 平均求解 | 最坏 | 等效频率 | 闭环 RMS 误差 |
|---|---|---|---|---|
| **RK4 N=12 max_iter=8（默认）** | **0.58–0.67 ms** | 1.0–2.9 ms\* | ~1500 Hz | 0.010 rad |
| RK4 N=12 max_iter=15 | 1.5 ms | 1.7 ms | ~660 Hz | 0.010 rad |
| 半隐式欧拉 N=12 max_iter=8 | 0.50 ms | 0.59 ms | ~2000 Hz | 0.007 rad |
| RK4 N=20 max_iter=8 | 2.5 ms | 3.3–8.8 ms\* | ~400 Hz | 0.015 rad |

\* 最坏值受本机负载影响较大（同一配置多次运行在 1.0~8.8 ms 间波动），
选型时应以**平均**值留 3~5 倍裕量，并在目标机上看 `getState().mpc.solve_ms`。

调优旋钮（按需权衡实时性与预测精度）: `mpc.N`、`mpc.use_rk4`（false 用半隐式欧拉，快约 3 倍）、
`mpc.max_iter`、`mpc.substeps`。**上线后请先看 `getState().mpc.loop_fps` 与 `solve_ms`**，
确保最坏求解耗时 < `loop_period`。车载计算机更慢时，优先降 `N` 或改欧拉。

---

## 5. 通信协议 v0x03（`include/tcbs/communication/Protocol.hpp`）

| 方向 | 帧 | 字段 |
|---|---|---|
| PC→MCU (41B，含 CRC) | `0x42 0x52 0x03` + size(36) + payload + CRC8 | `auto_aim_enable, fire, pitch_target_angle`、**大 yaw**: `mode, θ*, ω*, τ`、**小 yaw**: `mode, θ*, ω*, τ` |
| MCU→PC (**47B**，含 CRC) | `0x42 0x52 0x03` + size(42) + payload + CRC8 | `bullet_velocity, pitch_angle`、**大 yaw**: `angle(double,多圈), omega`、**小 yaw**: `angle(相对), omega`、`chassis_imu_yaw/omega`、`mark, color, auto_aim_switch,` 两个温度、**`mcu2_seq`（MCU2 新样本序号）** |

- 两个 yaw 关节各有**模式位**: **`1 = 仅力矩`**、**`2 = 力矩 + 电控位置/速度内环`**
  （`τ = kp(θ*−θ) + kd(ω*−ω) + τ_ff`）；**其余取值（含 `0`）= 非法模式 ⇒ 电控按 0 力矩处理**；
- `pitch_angle` 为原始语义，由上位机 `McuDataPreprocessor` 做线性映射（电控侧不做）；
- yaw 轴的"计数→弧度、多圈累计"仍由电控完成；映射参数保留在上位机（`recv_big_yaw_scale/offset` 等）；
- **`mcu2_seq` 是"值被保持"通道的关键**: 值保持期间必须重复发送同一个序号，
  **只有 MCU1 真正从 MCU2 取到新数据时才 +1**；上位机据此区分新样本与旧值（见 §2）。
  仍需注意: 上位机判定新样本时**序号变化与值变化二者取或**，因此即使序号不递增，
  值变了也会更新（反之亦然）。
- 波特率: 双向合计约 88B/帧；100 Hz 时 ≈8.8 kB/s，115200(≈11.5 kB/s) 可用但余量不大 →
  若同时要跑高帧率建议改到 460800（上位机侧只需定义 `SERIAL_BAUD_RATE=B460800`，
  见 `include/tcbs/communication/SerialProtocol.hpp` 顶部注释）。

---

## 6. 电控侧要求

见 `mcu_code_demo/dual_yaw_control.c`（协议解析、组帧、双电机控制、小 yaw 安全层、看门狗）
与 `mcu_code_demo/README.md`（接入说明、桩函数表、易踩坑）。

**两个 MCU 的分工（与真实结构一致，示例代码默认按此实现）**:
- **MCU1**（与上位机直连）: 解析/组帧、控制 **pitch + 小 yaw**（本地闭环、每帧新值）、
  并把 **大 yaw 的控制指令原样转发给 MCU2**、把 **MCU2 的大 yaw 与底盘 IMU 反馈**连同
  **新样本序号 `mcu2_seq`** 一起回传上位机（**只在真正取到新数据时 +1，值保持时序号不变**；
  上位机不用任何 MCU 端时钟，年龄全部由上位机自己测量）。
  - 示例里用宏 `YAW_BIG_SOURCE_FROM_MCU2`（默认 1）切换；置 0 可回到"单 MCU 本地控制大 yaw"
    的旧结构（两条路径都在 `#if` 内，均编译干净）。
  - 需填写的转发桩: `mcu2_send_yaw_big_command(mode, θ*, ω*, τ)`、
    `mcu2_get_yaw_big_sample(double* angle, float* omega)`、
    `mcu2_get_chassis_imu_sample(float* yaw, float* omega)`。
    约定: 桩返回 0 时**不要修改出参**（表示"本次没有新数据，继续用旧值"），
    且**不要每帧都返回 1**（否则 `mcu2_seq` 失去意义，上位机会误判"每帧都有新样本"）。
- **MCU2**: 控制 **大 yaw + 采集底盘 IMU**，按自己的节奏回传给 MCU1。

协议侧的硬性要求:

1. **每关节力矩限幅与最终电流限幅**（力矩换算常数每关节独立）；
2. **小 yaw 的硬限位保护（行程 ±30°）**: 目标角先夹到 `[−28°, +28°]`（各留 2° 余量）、
   越限只允许回中方向力矩、接近限位（距任一侧 10° 起）按剩余角度限制速度
   （这是最后一道安全防线）。
   ⚠ **这套限位是以"小 yaw 编码器零点"为基准的，而这个零点必须先由 §8 步骤 3（手动零点捕获）标定**
   （`recv_small_yaw_offset`）。**零点没标定之前，电控与 MPC 的限位判断都是错的** ——
   此时既不能用绝对角度指挥小 yaw，也应把电控侧的限位保护临时放宽（只保留低力矩限幅 + 机械硬限位）；
3. **看门狗**: 超过 ~50 ms 未收到合法帧 → 两关节力矩清零；
4. 大 yaw 的多圈累计与打包。

---

## 7. 目录结构

```
include/tcbs/                  # ★ 所有头文件都在 include/tcbs/ 下（仓库内外一律 #include "tcbs/..."）
  RobotController.h            # 对外主接口（tcbs::RobotController）
  communication/
    Protocol.hpp               # 协议 v0x03（权威定义）
    Communications.hpp         # 串口 + 映射 + 估计器组合
    McuDataPreprocessor.h      # 全部可配置的编码器/指令映射
    YawStateEstimator.h        # 状态估计（可信量/延迟补偿/反解/来源）
    SerialProtocol.hpp CRC.h   # 通用串口协议框架
  mpc/
    planar_yaw_model.h         # ★ 平面 2 自由度 8 参模型（含推导注释/回归矩阵/摩擦/积分器）
    planar_yaw_params.h        # 默认参数与默认 MPC 配置（占位值，标定后替换）
    dual_yaw_mpc.h             # 耦合非线性 MPC（Ceres + 动态自动微分）
    mcu_mpc_controller.h       # 后台发送线程 + 组包 + 积分补偿
  common/RotationUtils.h       # 旋转矩阵/欧拉角/解卷绕
  common/StrictPose.h          # ★ 严格反解数据包（IMU 为准确值 + 反解底盘 + 可重构性）
  c_api/RobotCommunicationC.h  # C ABI（供 python/外部程序）
src/                           # 对应实现
tools/
  test_serial.cpp              # ★ 串口链路自检（原仓库 test_serial 的移植；--list/--selftest/--no-send）
  pitch_calibration.cpp        # ★ pitch 映射标定（两段线性拟合；--sim/--selftest；★需 IMU 在头上）
  mpc_param_eval.cpp           # MPC 参数评估（用辨识参数跑闭环）
tests/
  test_planar_yaw_model.cpp    # 模型验证（独立实现比对/解析特例/回归矩阵/能量一致性）
  test_yaw_state_estimator.cpp # 状态估计验证（延迟补偿/反解/来源）
  test_dual_yaw_mpc.cpp        # 闭环仿真（跟踪/限位/抗扰/失配/性能）
mcu_code_demo/                 # 电控侧示例 C 代码
python/
  scripts/collect_sysid.py     # ★ 辨识数据采集（录制目标序列+增强+PID，分轴，100Hz）
  scripts/identify_params_torch.py    # ★ 唯一的参数辨识路径（torch 可导仿真输出误差法）
  scripts/mpc_demo.py          # 控制台示例（小 yaw 正弦跟踪；可切 IMU 构型/临时改 8 参）
  scripts/c_api_selftest.py    # C API / 绑定自检（无硬件可跑）
  torque_controller/           # ctypes 绑定（对应 C API v4: 平面 8 参模型）
docs/
  model.md                     # ★ 平面 8 参模型: 化简前提/推导/可辨识性/验证/IMU 构型开关
  calibration.md               # ★ 本构型下的完整标定方法（含辨识方法与采集规格）
  sysid_data.md                # 辨识数据格式与采集协议（CSV/NPZ 列头、增强规则、安全策略）
  sysid_torch.md               # ★ torch 辨识结果 + MPC 闭环验证（λ、N、ON_HEAD、倾斜数据）
```

---

## 8. 标定（详见 `docs/calibration.md`）

一句话流程: **先运动学后动力学**。
1. 用大 yaw 上的 IMU 标定大 yaw 编码器的比例/零位**与链路延迟**（底盘静止时
   Δ(IMU 方位角) 必须等于 Δ(关节角)）；
2. 用**临时装回云台终端的 IMU** 标定小 yaw/pitch 编码器映射与 IMU 安装旋转，
   并校核反解精度（< 0.5°）；其中 **pitch 映射用本仓库工具**
   `./build/tcbs_pitch_calibration --points=20 --min=<原始单位下限> --max=<原始单位上限>`
   （两段线性拟合，直接打印可粘贴的 `recv_pitch_*`/`send_pitch_*` 四行；
   **前提是 IMU 临时装在头上**，见 `docs/calibration.md` §3.3）；
   无硬件时先跑 `./build/tcbs_pitch_calibration --sim`（虚拟台架 + 断言）与 `--selftest`；
3. **小 yaw 零位（必须先做，后面所有小 yaw 角度语义都依赖它）**:
   跑 `./build/tcbs_test_serial`（两关节恒「仅力矩 + 0 N·m」，工具不驱动任何关节）——
   **人工把小 yaw 摆到"准确的零点"位置并扶稳，读它打印的 `yaw_small_angle`**（电控原始弧度，
   多摆几次取平均；**重复性就是零点精度上限**），然后
   `recv_small_yaw_offset = −(零点读数)`、`send_small_yaw_offset = +(零点读数)`
   （下发与上报共用同一原始坐标系 ⇒ 互为逆映射；电控判据是"下发值 == 编码器回读值 ⇒ 不动"）。
   **不需要单独的标定程序**（用户确认: 串口测试里直接读即可）。
   注意: 零点只给出"哪个读数对应 0" ⇒ **`P` 的方向仍然未知，不能假设 `Py=0`**。
4. 用 `python/scripts/collect_sysid.py` 采集（**录制目标序列 + 增强 + 上位机 PID**、
   **分轴激励**、另一轴 PID 保持在固定/随机位置、pitch≡0、100 Hz）→
   用 `python/scripts/identify_params_torch.py`（**唯一辨识路径**: torch 可导仿真输出误差法）
   辨识 8 个参数；
4. 关键: **两轴力矩都必须记录**（被保持轴的力矩就是耦合项 `P` 的传感器，见
   `docs/calibration.md` §4.2）；小 yaw 应**尽量用满行程**（±30°，
   两侧各留 8° 余量 ⇒ 约 44° 摆幅），摆幅越小 `Px/Py` 与惯量越共线；
   **强烈建议加静态倾斜段**（底盘静止但静置成 ±10° 左右，`--tilted`），
   否则水平数据下 `Px/Py` 几乎不可辨识（详见 `docs/sysid_data.md` §6.4）；
5. 辨识完做三项检查: 参数物理合理（`J>0`、`fc/fv ≥ 0`）、`|Px|/σ ≥ 3`、
   **未参与拟合的留出段**做开环前向仿真的 RMSE；再上实车跑 `tcbs_control_demo` 低幅验证。

---

## 9. 安全与实时性注意事项

1. **摩擦软符号系数 λ 与积分子步必须配套**：模型 λ=100（`planar_yaw_params.h`）、
   `mpc.substeps=4`（有效 2.5 ms）⇒ 显式 RK4 稳定。λ 越大越接近真库仑（|ω|≳1°/s 饱和），
   但稳定上限 `dt < 2.78/(fc·λ·M⁻¹)` 会变小；λ=1000 需要 `substeps≈32`，单次求解
   3~6 ms（最坏 12~20 ms）**超出 10 ms 控制周期**。`DualYawMpc` 构造时会按
   `recommendedFrictionLambda()`（已按子步折算）告警；
2. **小 yaw 限位三层保护**: MPC 软限位 → 电控目标限位 → 电控硬限位/机械限位；
3. **求解耗时**必须 < `loop_period`；`getState().mpc.solve_fail_count` 应保持 0
   （求解失败时本实现退化为零力矩，属安全但会失去控制）；
4. **估计未就绪时**（`st.est.valid == false`，即从未收到大 yaw 绝对基准）不发控制力矩；
5. 串口中断/重连由 `SerialProtocol` 处理；电控侧看门狗负责断流保护。

---

## 10. 与旧版 `TorqueController` 的差异（改版清单）

| 方面 | 旧版 | 本版 |
|---|---|---|
| 被控对象 | 单 yaw，模型 `Jω̇ = τ − τ_c·sgn − bω` | **平面 2 自由度（8 参）**，两 yaw 耦合 + 底盘转动 + 重力 |
| IMU 位置 | 云台终端（pitch 之后） | **大 yaw 转子上**（决定了整套可观测性设计） |
| 状态 | 融合 yaw 位置/速度 + 底盘姿态 | 可信量 / 大 yaw 延迟补偿估计 / **反解真实位姿** / **数据来源** 四组 |
| 大 yaw 编码器 | 唯一绝对基准（延迟即误差） | 低频基准，高频由 IMU 速率前推补齐（延迟补偿，实测改善 ~5×） |
| 对外目标 | 单个 yaw 角度（自动卷绕到 imu 同圈） | **大/小 yaw 两个世界方位角**（或关节角便捷接口） |
| 约束 | 力矩 + 力矩变化率 | 同上 + **小 yaw 关节限位**（软限位代价项） |
| 求解器 | Ceres + 参数箱式边界（线搜索） | Ceres + LM 信任域（约束内移，求值次数降 ~6×） |
| 参数 | J/τ_c/b 四个标量 | **8 个可辨识参数**（4 摩擦 + `Jbig_eff` + `Js` + `P` 两分量）+ 线性最小二乘/torch 辨识工具链 |
| 协议 | 单 yaw 通道 | 双 yaw 通道 + 模式位 + 时戳 |
| 链路工具 | `tcs_test_serial`（打印收到的每帧） | `tcbs_test_serial`（同用途，另加 `--list` 端口/选择器诊断、`--selftest` 包布局与安全不变量自检、`--no-send` 纯监听、1 Hz 链路统计；安全上 `auto_aim_enable` 默认 0、pitch 目标默认 0 而不是硬编码 `10.0f`） |

---

## 11. 已知局限 / 待确认事项

1. **两轴平行是硬前提**。若实测夹角 > 0.2°，姿态链不能合并、反解与"方位角之和"语义都不成立，
   需要改成完整 3 关节 IK（改动量较大）。**请先测量这个夹角**。
2. 底盘**角加速度**默认置 0（作为慢变扰动）；若需要更激进的前馈，可由 `ω_c` 微分估计填入。
3. 底盘平动与其旋转轴偏置未建模。
4. `planar_yaw_params.h` 里所有数值都是**占位值**，必须按 `docs/calibration.md` 标定后替换；
   尤其 `small.max_torque`（小 yaw 力矩能力）与 `max_torque_rate` 直接影响控制权限。
   另注意两个**不可分辨**的量（见 `docs/model.md`）: `m_u` 与质心偏置 `ρ` 只能得到乘积
   `P=m_u·ρ`；大 yaw 自身惯量与 `m_u|d|²` 只能得到和 `Jbig_eff`。
5. 结构柔度/回差未建模；若非共轴偏置较大且上装较重，注意低频谐振。
6. 电控协议需要电控侧同步改到 v0x03（示例代码已给）。
