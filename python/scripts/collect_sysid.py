#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_sysid.py — 双级 yaw 云台的**系统辨识数据采集**（分轴激励版）

用法::

    # 真实硬件（先在 build/ 里构建 libtcbs_robot_comm_c.so）
    python3 python/scripts/collect_sysid.py --segments=10
    python3 python/scripts/collect_sysid.py --segments=4 --tag=exp1 --max-temp=50
    # 无硬件自检（内置仿真代替串口；用来验证脚本逻辑与数据格式）
    python3 python/scripts/collect_sysid.py --dry-run --segments=1

数据格式见 ``docs/sysid_data.md``。

================================================================================
一、为什么这样采集（设计理由，先读这一段）
================================================================================

1) **不用固定激励（正弦/傅里叶/多正弦）**，而是"**提前录制的目标序列 + 增强 + 上位机 PID**"。
   原因: 待辨识的是 ``include/tcbs/mpc/planar_yaw_model.h`` 的 8 参模型，其中摩擦是
   ``fc·tanh(λθ̇) + fv·θ̇``、耦合是 ``M11/M12/μ(θ_s)``、重力/一阶矩项与 θ_s 强相关。
   这些项在真实工况下的**力矩-速度工作点在整个范围内连续变化**（含换向、低速滞留、
   变速段）；单一正弦只在一条窄带（固定频率/幅值）里激励，摩擦的库仑段与换向段
   几乎没有数据，参数会病态。录制的目标序列来自真实跟踪工况 ⇒ **工况相符**
   （``data/targets/*.npz`` 就是实机录的参考角序列），再叠加"随机截取 + 连续化 +
   随机等比缩放 + 随机偏置"做数据增强，得到多样化的激励形态。

2) **大小 yaw 分开采集（每段只激励一个关节，另一个由 PID 保持在固定位置）**。
   原因: 两轴通过 ``M12 = J_s + d·Q(θ_s)`` 强耦合。若两轴同时被驱动，回归矩阵的两列
   高度相关（分不清谁动了谁），8 参最小二乘会病态。分轴采集让"被激励轴的
   τ↔θ↔θ̇"这条主链干净可辨。
   同时 **held 轴不是"关掉不管"**: 它由同一套 PID 闭环守在固定角度上，其**下发力矩
   恰好是耦合项的观测量** —— 保持轴要顶住被激励轴带来的惯性/离心/科氏扰动力矩才能
   不动，因此 ``tau_small``(held) 里含有 ``M12·θ̈_big + μ 项`` 的信息，是辨识耦合与
   惯量参数的关键通道（这正是"两轴力矩都要记录"的原因）。

3) **held 轴的目标位置随机化**（大 yaw: 现有角度附近 ±π 内随机；小 yaw: ±40° 内随机）。
   原因: ``M12``、``μ = ∂M11/∂θ_s``、重力项 ``G_s = Q_x g_y − Q_y g_x`` 都是 **θ_s 的
   函数**。只有让 θ_s 在多个不同值上取数据，才能把它们辨识出来（只在一点标定只能定
   一个切片，Px/Py 与 μ 完全不可辨）。

4) **两个关节在采样前都要像 driven 轴一样先 PID 到位**（固定保持 --settle-sec，不判稳定判据）。
   原因: 采样段必须是"纯激励响应"。若保持轴在采样开始时还在大幅移动，它的加减速会给
   被激励轴注入额外的、未记录的扰动力矩，把数据污染成"激励 + 未知阶跃"；同理被激励轴
   也要从静止、无积分饱和的状态起步，这样数据段起点的状态是已知的（θ̇≈0）。

5) **小 yaw 超范围序列整体等比缩放到 ±40° 以内（不截断）**。
   原因: 小 yaw 机械/电控硬限位是 ±45°。截断会破坏轨迹形状（人为引入尖角与额外高频），
   而**等比缩放保形状**，只把幅值映射到安全区；形状（换向次数、速度分布、停留段）
   才是激励多样性的来源。

6) **pitch 固定 0**。原因: ``planar_yaw_model.h`` 的化简前提③明确"忽略 pitch 转动对
   上装质心的影响 ⇒ pitch 不进动力学"。本项目只用两个 yaw 的数据标定，动 pitch 只会
   白白引入未建模扰动。

7) **100 Hz 固定、用 ``time.perf_counter_ns()`` 忙等到绝对时间点**。
   原因: 上位机 PID 的 dt 必须真是 0.01 s（积分/微分项直接乘除 dt）；``time.sleep()``
   有毫秒级抖动 + 调度延迟，会让 dt 不准、力矩抖动，直接污染辨识。忙等绝对时间点可把
   抖动压到微秒级。**不允许更高**频率: 电控的力矩接收周期、状态估计的更新率
   （大 yaw/底盘 IMU 只有 ~10 Hz）都跟不上，高于 100 Hz 只是重复下发同一份估计值，
   反而破坏"力矩-状态"的时间对应关系。

8) **仅力矩模式下发（``yaw_*_mode = 0``）+ 上位机 PID**。原因: 要辨识的是**力矩 → 运动**
   的动力学，力矩通道必须由上位机独占；若开电控内环（mode 1），电控的位置环会吃掉上位机
   力矩的因果性，辨识出来的就不是机械参数而是"电控环 + 机械"的混合模型。
   反馈: 大 yaw 用 IMU 直测的 ``platform_azimuth``（实时无延迟），小 yaw 用编码器
   ``small_joint_angle``（实时可信）—— 两者都是估计器里的**可信实时量**。

9) **力矩变化率限幅 0.1 N·m/步**。原因: 保护减速器（力矩阶跃会激发齿隙冲击），并让激励
   在高频段的能量更接近真实工况；**记录的是限幅后的最终下发值**，否则辨识用的力矩与实际
   作用力矩不一致。

10) ``theta_big`` 记录的是**延迟补偿估计** ``big_joint_angle``（而不是原始编码器值）:
    大 yaw 编码器经 MCU1↔MCU2 链路（~10 Hz、间隔不规则、值被保持），原始值带不确定延迟；
    估计器用 IMU 角速度做一阶外推补偿到当前时刻。同时记录 ``big_enc_age``，供拟合时判断
    哪些时刻的补偿可信（用户要求"可以记录但拟合不用"）。

================================================================================
二、每段数据流（一个 segment）
================================================================================

    读当前状态 → 规划本段:
        · 选轴: 偶数段驱动大 yaw(axis=0)，奇数段驱动小 yaw(axis=1)
        · [可选 --tilted] 打印显著提示: 请把底盘以固定倾角静置（见 §五）
        · driven 参考 = 录制序列 → 连续化(np.unwrap) → 去直流/归一化 → 随机幅值
                        → TrajectoryPlanner+StepRefinementWrapper 平滑
                        → **校验平滑后确实有激励幅值**(大 yaw ≥14° / 小 yaw ≥7°)，
                          不够则放大输入重试、再换窗口
                        → 叠加随机中心
                          （大 yaw: 现有方位角附近 ±30°；小 yaw: 落在参考包络
                            [−22°, +22°] 内，中心在可行中心区间内随机，见 §七）
        · held 目标 = 大 yaw: 现有方位角 ±π 随机；小 yaw: 行程中心 ± 0.7×包络半宽
          （当前行程对称 ±30° ⇒ 中心 0°、范围 ±15.4°；式子按区间运算，非对称行程也对）
    到位: 两个 PID 把 driven 轴拉到 ref[0]、held 轴拉到 held_target，固定保持 --settle-sec（默认 5 s）
          （移动参考同样由轨迹规划器整形 —— 直接给阶跃会饱和过冲，把小 yaw 顶到硬限位；
            未收敛最多再等 2 轮；到位后 PID 状态**不清零**，见"与旧脚本差异"）
    采样: 300 点 @100 Hz（3 s），每点:
          忙等到绝对时间点 → 读 est/mcu → 安全判定（小 yaw 行程界限、温度）
          → 两轴各跑 PID(误差 e = wrap(目标 − 反馈)) → 各轴力矩变化率限幅
          → 仅力矩模式下发 → 记录（含重力 A 系平面分量 gravity_ax/ay）
    收尾: 主动回到**行程中心**保持（当前行程 ±30° ⇒ 中心 0°；大 yaw 保持当前平台方位角）
    保存: ``data/sysid/sysid_<tag>_<时间戳>_<序号>.npz``（★ **默认只写 npz**，
          ``--save-csv`` 才再加一份同名 ``.csv``；npz 是 csv 的超集，辨识只读 npz）
          **零力矩只在程序退出时发**（见 safe_shutdown）

================================================================================
三、安全策略
================================================================================

* **静止保持段也落盘**（`--record-hold`，默认开；文件名后缀 `_hold`）: 到位等待期间
  两轴都在走大角度阶跃，这段数据同样逐样本记录、同样参与辨识（首段除外）。
* 小 yaw θ 超出**硬限位 [−30°, +30°]** → 立即中止本段（**不保存**被污染的数据）→ PID 回
  **行程中心**（当前 0°）→ 零力矩；距界限 < 3° 时只打印告警（见 §七）;
* 电机温度 ≥ ``--max-temp`` → 中止本段 → 零力矩 100 Hz 保温等待降温后重试（超时退出）；
* ``--tilt-rolling`` 段间改倾角时，两轴**保持闭环守位**（倾斜后重力会在小 yaw 上产生力矩，
  撒手会让它自己滑到限位），不撒手、也不做补偿；
* Ctrl+C → 立刻按限幅斜坡把力矩压到 0 并**连发若干帧零力矩**再关闭句柄；
* 任何退出路径（正常/异常/中断）都经过同一个 ``safe_shutdown()``。

================================================================================
四、dry-run（无硬件）
================================================================================

``--dry-run`` 用**内置仿真**代替串口: 被控对象用 ``include/tcbs/mpc/planar_yaw_model.h``
的同一组方程（M(θ_s)、μ、科氏/离心、重力、``fc·tanh(λθ̇)+fv·θ̇``）在 Python 里积分，
λ = 100（模拟真实库仑摩擦的陡峭软符号），积分步长 0.05 ms（RK4，200 子步/控制周期）。
仿真对象对外暴露与 ``TcbsRobotCommunication`` 相同的字段语义
（``platform_azimuth`` / ``small_joint_angle`` / ``big_joint_angle``(延迟补偿) /
``mcu2_seq`` / ``big_enc_age`` …），因此**同一套采集代码**在仿真与实机上走同一条路径
—— dry-run 能真正验证脚本逻辑与输出格式。

================================================================================
五、静态倾斜段（可选: --tilted / --tilt-rolling，默认关闭）
================================================================================

**为什么需要**: `P = m_u·ρ`（上装一阶矩）是本项目的重点（小 yaw 载荷质心不在小 yaw 转轴上）。
但底盘**水平**时 `gravity_a` 的平面分量为 0 ⇒ 重力项 `G_s = Qx·gy − Qy·gx` 恒为 0，
`Px/Py` 只能靠 `M11/M12/μ` 里的 `d·Q = d·R(θ_s)·P` 间接观测，而 `d ≈ 0.03 m` 很小 ——
此前的 LS 自检显示此时 `P` 与惯量参数**共线（corr ≈ −0.93）**，
虽然仍能估到 ~10% 以内，但一旦有摩擦形状失配/柔度等未建模误差，`P` 的偏差会被放大。

**倾斜为什么有效**: 底盘以固定倾角静置后，`gravity_a` 的平面分量 ≈ `g·sin(tilt)`（10° ⇒ 1.7 m/s²），
而 `∂G_s/∂P` 的灵敏度正是这个 g 量级，比水平时仅 `d ≈ 0.03 m` 的惯性耦合项强两个数量级
⇒ `Px/Py` 的回归条件数改善约两个数量级。

**倾斜 ≠ 底盘运动**（关键）: 采集期间底盘仍然**静止**，只是**静置姿态**不同（如垫起一侧车轮）；
模型里的 `base_omega` / `base_alpha` **依旧取 0**。倾斜只改变重力在 A 系的投影。

**脚本只做两件事**（都不影响默认行为）:
  1) 逐样本记录 `gravity_ax` / `gravity_ay`（CSV 末两列 + npz 数组），元数据记 `tilted=1`；
  2) 每段前打印显著提示，要求把底盘以固定倾角静置。
**不做**: 任何倾角补偿、任何激励方式改动、任何"倾角是否足够"的自动判断。
`--tilt-rolling` 额外在段间提示轮换倾角（+10° / 0° / −10° 槽位循环），并留 10 s 让操作者调整，
期间两轴**保持闭环守位**（倾斜后重力会在小 yaw 上产生力矩，撒手会滑到限位）。

不传 `--tilted` 时，脚本行为与之前**逐字相同**，只是 CSV/npz 多了两列恒为 0 的重力列
（全 0 ⇒ 下游等价于原来的"水平假设"）。

================================================================================
七、小 yaw 行程三档（**对称 ±30°**）
================================================================================

实测机械行程是 **min = −30°、max = +30°**。三档含义:

| 档 | 区间 | 用途 |
|---|---|---|
| ① 硬限位（机械行程） | **[−35°, +35°]** | 电控侧也按它限位；`SMALL_TRAVEL_MIN/MAX`（2026-09-21 由 ±30° 放宽） |
| ② 中止阈值（上位机） | 同上（触及即中止本段、数据不保存、PID 回中心） | 对应旧版的 ±45° 中止 |
| ③ **软限位** | **[−30°, +30°]** | **±30° 之外才开始软限位**：越界只**告警**（每秒一句，报出离硬限位还剩多少），不中止 |
| ④ 参考包络 | **[−22°, +22°]**（软限位两侧各留 8° 跟踪超调余量） | 激励参考、到位目标、held 保持目标都用它 |

**中心 = 0°**（`SMALL_CENTER_RAD = (min+max)/2`，当前行程对称）。回中心/段尾保持/初始条件都用它；
式子不假设对称 ⇒ 以后改成非对称行程（例如 [−20°,+25°] ⇒ +2.5°）会自动跟着走。

**所有取值都写成区间运算（不假设对称，对称/非对称行程都能用）**:
  · 参考中心: 从**可行中心区间** `[env_min + 半幅, env_max − 半幅]` 里随机取
    （`random_center_for()`；而不是 `±(band − 半幅)`）；
  · 越界 guard: **整体等比缩放 + 平移到包络内**（`fit_into_interval()`；而不是绕 0 缩放）；
  · held（小 yaw 作为保持轴）目标: 行程中心 ± `0.7 × 包络半宽`（保留 0.7 安全余量，
    因为大 yaw 摆动会通过 M12 把小 yaw 推偏十几度）；
  · 中止判定: `θ > max` **或** `θ < min`（两个阈值不再同号对称）。

================================================================================
八、小 yaw 零点（**直接在串口测试里读，不需要单独程序**）
================================================================================

做法: 跑 `./build/tcbs_test_serial`（两个 yaw 关节恒为「仅力矩 + 0 N·m」，不会动），
人工把小 yaw 摆到**机械零点**，读它打印的 `yaw_small_angle`（电控原始弧度）：

* `recv_small_yaw_offset = −(零点处的 yaw_small_angle)`
* `send_small_yaw_offset = +(零点处的 yaw_small_angle)`（下发与上报同一原始坐标系 ⇒ 互为逆映射）

写回 `McuDataPreprocessor::LinearParams` 后，**全系统的角度语义都切到这个新零点**
（MPC 的小 yaw 限位 ±30°、回中中心 0、电控夹取都用新零点）⇒ 写回后必须复核限位。

================================================================================
九、与旧采集脚本（TorqueController/python/scripts/collect_sysid_data.py）的差异
================================================================================

1. **分轴采集**: 旧脚本只采一个 yaw 轴。新脚本每段只激励**一个**关节（driven），
   另一个（held）由同一套 PID 守在随机固定位置，**两轴力矩都记录** —— held 轴的力矩
   是耦合项（M12·θ̈ 等）的观测量，是辨识耦合/惯量的关键通道。
2. **过热保护换了判据**: 旧脚本靠"两次 PID 移动的角度差 < 20°"猜过热/卡死；新脚本直接
   用协议里的电机温度 ``yaw_big_temperature`` / ``yaw_small_temperature``（阈值
   ``--max-temp``），过温就零力矩保温等待降温后重采该段。
3. **到位移动也整形**: 旧脚本直接把阶跃目标丢给 PID；新脚本把"当前 → 目标"交给同一个
   轨迹规划器（``homing_sequence``），避免 PID 饱和过冲把小 yaw 顶到 45° 保护。
4. **PID 状态跨相位连续**: 旧脚本在采样开始时 ``reset()`` PID —— 等于把顶着重力/摩擦的
   积分力矩瞬间清零，那是一个**未记录的阶跃扰动**；新脚本"到位 → 采样"之间不清零。
5. **采样后不撒手**: 旧脚本每段结束发零力矩（被激励轴还有残余角速度时，会通过耦合把
   另一轴推着走）；新脚本每段结束**主动回中保持**，只在程序退出时才连发零力矩。
6. **参考增强更严格**: 平滑**之后**校验激励幅值（≥14°），不够就放大输入重试、再换窗口
   —— 旧脚本没有这层校验，可能采到"平滑完几乎不动"的静止段。
7. **数据格式**: **默认只落 npz**（``--save-csv`` 才另写一份 csv），含 ``axis`` /
   ``held_target`` / ``mcu2_seq`` / 底盘数据 / ``big_enc_age`` 以及全部标量元数据
   （dt、PID 增益、参考来源与幅值）。
8. **两个关节都要到位**: 旧脚本只有"把单轴 PID 到某角度"；新脚本 driven 与 held **两轴
   都先 PID 到位并保持 `--settle-sec`**（固定时长、不判稳定判据），否则段内会混入未记录的扰动力矩。
9. **dry-run 内置仿真**: 旧脚本必须有硬件才能跑；新脚本用 ``planar_yaw_model.h`` 的同一
   组方程在 Python 里积分被控对象（λ=100、0.05 ms 步长），无硬件即可验证全流程与格式。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ── 让脚本能直接 `python3 python/scripts/collect_sysid.py` 跑（不依赖 PYTHONPATH）──
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (_REPO, os.path.join(_REPO, "python"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torque_controller import TcbsRobotCommunication  # noqa: E402  (低层 ctypes API)
from trajectory_planner import StepRefinementWrapper, TrajectoryPlanner  # noqa: E402

# ============================================================================
# 常量（采样/控制规格）
# ============================================================================
RATE = 100.0                       # 采样率固定 100 Hz（规格: 不允许更高）
DT = 1.0 / RATE                    # 0.01 s（写进 npz 的 dt）
DT_NS = int(round(DT * 1e9))       # 10_000_000 ns
SAMPLE_LEN = 300                   # 标准段长 = 3 s @100 Hz
TWO_PI = 2.0 * math.pi

# ── 协议模式位（include/tcbs/communication/Protocol.hpp）──
YAW_MODE_TORQUE_ONLY = 1           # ★ 1 = 仅力矩（2026-09-23 与 mcu::YawMode 对齐；上位机独占力矩通道）
AUTO_AIM_ENABLE = 1                # 自瞄总开关（与电控手动开关相与）；与旧脚本一致
PITCH_TARGET_ANGLE = 0.0           # pitch 固定 0（不进动力学）

# ── 上位机 PID（与旧采集脚本一致，两批数据可直接对比/合并）──
# ★ PID 增益（可用 --kp/--ki/--kd 覆盖）。
#   原仓库是 2.0 / 0.1 / 0.2。实测反馈: **P 偏小（到位慢/有静差）+ 有震荡** ——
#   震荡的来源基本是 kd 那项: 它用的是**未滤波的中心差分** ė，100 Hz 下角度量化
#   (2π/8192 ≈ 7.7e-4 rad) 就会让 ė 抖 ±0.08 rad/s，×kd 直接变成力矩抖动。
#   因此: kp 2.0→4.0（P 加大, 减少静差/加快到位）、kd 0.2→0.05（削弱差分噪声放大）、
#   ki 保持 0.1（有抗饱和，不动）。这两项要按实车手感再调时用 CLI。
PID_KP, PID_KI, PID_KD = 1.0, 0.1, 0.1
PID_OUT_MIN, PID_OUT_MAX = -1.0, 1.0
MAX_TORQUE_DELTA = 0.5             # 相邻两步力矩变化限幅 N·m（保护减速器）
MAX_TX_FAIL = 50                   # 连续 50 帧发不出去 ⇒ 判定链路断开，报错退出（0.5 s）

PID_DEADBAND_DEG = 1.0            # [CLI] --pid-deadband-deg（见 PidController 的死区说明）

# ============================================================================
# ★★ 命令行默认值（**统一在这里管理**）
#    · `build_arg_parser()` 只引用这些名字 ⇒ 改默认值只改这一处，不在 add_argument 里写裸字面量;
#    · 命名约定: 只有 CLI 用的加 `[CLI]` 注释；别的代码也引用的沿用原名（RATE / DEFAULT_OUT_DIR /
#      PID_KP… / SETTLE_SEC / SIM_* 等），它们同样是"默认值的唯一来源"。
# ============================================================================
SEGMENTS_DEFAULT = 1              # [CLI] --segments（偶数段激励大 yaw，奇数段激励小 yaw）
DURATION_SEC = 3.0                # [CLI] --duration-sec：每段采样时长 s（3.0 s = 300 点 @100 Hz）
SEED_DEFAULT = 1048596                 # [CLI] --seed（★ 确定性: 换 seed 才是另一批激励）
MAX_TEMP_C = 55.0                 # [CLI] --max-temp：电机过温阈值 ℃
TAG_DEFAULT = None                # [CLI] --tag（None = 按 axis 自动取 big/small）
RECORD_HOLD_DEFAULT = True        # [CLI] --record-hold / --no-record-hold
SAVE_CSV_DEFAULT = False          # [CLI] --save-csv / --no-save-csv
SIM_TILT_DEG = 0.0                # [CLI] --sim-tilt-deg（dry-run 底盘静态倾角；0 = 水平）
SIM_COM_SEED = None               # [CLI] --sim-com-seed（None = 用本次 --seed）

# ── 小 yaw 行程限位（**对称: 软限位 ±30° / 硬限位 ±35°**）──
#   四档含义（详见文件头 "七、小 yaw 行程三档" 与 docs/sysid_data.md §5）:
#     ① 硬限位（机械行程, 电控侧也按它限位）: [SMALL_TRAVEL_MIN, SMALL_TRAVEL_MAX] = [−35°, +35°]
#        ★ 用户要求（2026-09-21）: 限幅从 ±30° **放宽到 ±35°**，给超调留 5° 机械余量。
#     ② 软限位（正常允许范围）: [SMALL_SOFT_MIN, SMALL_SOFT_MAX] = [−30°, +30°]
#        ★ 用户要求: **±30° 之外才开始软限位** —— 采集时越过软限位只**告警**（每秒一句，
#          并报出离硬限位还剩多少），**不中止**；只有碰到硬限位（±35°）才中止本段。
#     ③ 中止阈值（上位机）: 与硬限位同值 —— 一旦触碰立刻中止本段、数据不保存、PID 回中心
#     ④ 参考包络（激励/到位/保持都用它）: [SMALL_ENV_MIN, SMALL_ENV_MAX] = [−30°, +30°]
#        ★ 用户要求（2026-09-21）: **包络 = 软限位**（SMALL_TRACK_MARGIN 改成 0°，
#          不再留跟踪余量）⇒ 参考可以一直摆到 ±30°，超出部分（到 ±35° 硬限位）就是
#          留给 PID 超调的余量。想再收回来就把 SMALL_TRACK_MARGIN 调大。
#   ★ 改行程要**三处一起改**: 这里、C++ 的 defaultMpcConfig().small.min/max_angle、
#     电控侧 mcu_code_demo 的 YAW_SMALL_MIN_RAD/MAX_RAD。
#     C++ 侧（defaultMpcConfig）已同步: 硬限位 ±35°、small_limit_soft_ratio = 0.9286
#       ⇒ 软限位 = max − (1−ratio)·(max−min) = 35° − 0.0714·70° = 30° ✓
#   ★ 下面的取值全部写成基于 [min, max] 的**区间运算**（不假设对称），所以对称/非对称行程
#     都能直接用；行程若是非对称（例如 [−20°,+25°]，中心 +2.5°），中心/包络会自动跟着走。
SMALL_TRAVEL_MIN = math.radians(-35.0)   # 硬限位下界（机械行程；中止阈值同值）
SMALL_TRAVEL_MAX = math.radians(35.0)    # 硬限位上界
SMALL_SOFT_MIN = math.radians(-30.0)     # 软限位下界（越界只告警，不中止）
SMALL_SOFT_MAX = math.radians(30.0)      # 软限位上界
SMALL_CENTER_RAD = 0.5 * (SMALL_TRAVEL_MIN + SMALL_TRAVEL_MAX)   # 0°（当前行程对称）
#   ↑ 回中/段尾保持/初始条件都用**行程中心**而不是硬编码 0: 当前行程对称 ⇒ 中心就是 0，
#     但一旦行程改成非对称（例如 [−20°,+25°] 的中心是 +2.5°），这个式子会自动跟着变，
#     保证到两端的余量相等。
SMALL_ABORT_MIN = SMALL_TRAVEL_MIN       # 中止阈值（规格: 触及硬界限即中止本段）
SMALL_ABORT_MAX = SMALL_TRAVEL_MAX
SMALL_WARN_MARGIN = math.radians(3.0)    # 距**软限位** < 3° 只**告警**（不中止）
SMALL_TRACK_MARGIN = math.radians(0.0)   # 参考包络相对**软限位**留的跟踪超调余量（★ 0° = 包络与软限位同值）
SMALL_ENV_MIN = SMALL_SOFT_MIN + SMALL_TRACK_MARGIN     # −30°（= 软限位，无额外收窄）
SMALL_ENV_MAX = SMALL_SOFT_MAX - SMALL_TRACK_MARGIN     # +30°
SMALL_ENV_HALF = 0.5 * (SMALL_ENV_MAX - SMALL_ENV_MIN)  # 包络半宽 = 30°
SMALL_REF_AMP = SMALL_ENV_HALF           # driven 参考的半幅上限（= 包络半宽 = 30°）
#   ↑ PID 跟带尖角（换向）的参考时实际角度会超出参考峰值（仿真实测 2°~14°）:
#     现在余量由**软限位(30°) → 硬限位(35°)** 那 5° 提供（参考到 30°、超调落在 30~35°）。
HELD_SMALL_MAX = 0.7 * SMALL_ENV_HALF    # held 小 yaw 随机目标半宽（保留 0.7 安全余量）
#   ↑ driven 轴是**大 yaw** 时，held 小 yaw 的目标在包络内随机抽，但收进 0.7 倍:
#     大 yaw 摆动会通过 M12 给小 yaw 注入扰动力矩（M12·θ̈_big 可达 ~0.3 N·m），
#     PID 顶回来需要几度到十几度的瞬时偏差，收窄一点才有余量不触碰中止阈值。
#     抽取范围 = SMALL_CENTER_RAD ± HELD_SMALL_MAX（⊂ 参考包络），见 docs/sysid_data.md §6.2。
BIG_REF_AMP = math.radians(120.0)     # driven 大 yaw 参考**半幅**上限（★ 2026-09-21: 60°→120°
                                      #   ⇒ 参考峰峰可达 240°；大 yaw 多圈自由，仅防大摆）
BIG_CENTER_JITTER = math.radians(30.0)  # driven 大 yaw 参考中心相对当前方位角的随机抖动
HELD_BIG_OFFSET = math.pi             # held 大 yaw 目标: 现有角度 ±π 内随机（多圈连续）
# （到位判据已删: 与原仓库一样"PID 跑固定 2 s 就算到位"，不做收敛判定/多轮重试 ——
#   判据不满足时的处理反而更麻烦，且原仓库就是这么做的，两批数据口径一致。）

# ── 参考幅值: 既要"每段都有有效激励"，又不能超过该轴的安全半幅 ──
#   下限**按轴给**: 小 yaw 行程只有 45° 宽（包络半宽才 14.5°），下限不能沿用大 yaw 的 14°。
MIN_EXCITE_AMP_BIG = math.radians(14.0)     # 大 yaw 参考半幅下限（≈14°）
MIN_EXCITE_AMP_SMALL = math.radians(7.0)    # 小 yaw 参考半幅下限（≈7°: 峰峰 14°，
#   仍远高于编码器噪声/到位判据 0.02 rad ≈ 1.1°，且能产生可观的力矩变化）
MIN_SHAPE_SPAN = 0.20                 # 挑选录制窗口的峰峰值下限 rad（≈11°）: 丢掉"平段"
WINDOW_TRIES = 20                     # 挑窗口最多重试次数（录制序列里有大量静止段）
AMP_JITTER = (0.6, 1.0)               # 半幅在上限的 60%~100% 间随机（幅值多样性）
REF_TRIES = 6                         # 参考幅值不达标时换窗口重试次数
AMP_BOOST_TRIES = 4                   # 平滑抹平了激励时，放大输入重试次数
AMP_BOOST_FACTOR = 4.0                # 每次放大倍数

# ── 参考轨迹规划器（按轴给参数；理由: 参考必须是执行器跟得动的）──
#   · 大 yaw: 惯量大、力矩上限 ±1 N·m，允许较快的摆动（主要激励惯量/耦合）
#   · 小 yaw: 必须一直守在 ±45° 内，参考越"温柔"，PID 跟踪误差越小、越安全
REFINE_N = 1000                       # StepRefinementWrapper 细化系数（与旧脚本一致）
BIG_PLANNER = dict(max_velocity=8.0, max_acceleration=30.0, max_jerk=800.0)
SMALL_PLANNER = dict(max_velocity=3.0, max_acceleration=15.0, max_jerk=400.0)

# ── 时序 ──
HOLD_SUFFIX = "_hold"                 # 静止保持段落盘文件名的后缀（见 --record-hold）
# 采样前的**到位等待**（固定时长，**不判稳定性** —— 稳定判据已按用户要求删除，2026-09-20）。
# 两次采样的间隔 = SETTLE_SEC（原来还要再等一段"连续稳定 STABLE_SEC"）。
SETTLE_SEC = 5.0                      # 采样前 PID 到位并保持这么久（固定时长，不判据）
ZERO_FRAMES_AT_EXIT = 20              # 退出前必发的零力矩帧数（规格: 连发几帧）
MAX_COOL_WAIT_S = 600.0               # 过热等待上限（超过则退出）
COOL_HYSTERESIS_C = 5.0               # 降温到 max_temp − 5 ℃ 才恢复
RECENTER_SEC = 1.5                    # 小 yaw 越限后的回中时间

# ── 静态倾斜段（--tilted / --tilt-rolling）──
#   倾斜只为让重力在 A 系有平面分量（gravity_a[0..1] ≠ 0），从而把 P = m_u·ρ 的回归
#   条件数改善约两个数量级，见文件头 "五、静态倾斜段" 与 docs/sysid_data.md §6.4。
#   **倾斜 ≠ 底盘运动**: 采集期间底盘仍是静止的（只是静置姿态不同），
#   模型外生量 base_omega / base_alpha 依旧取 0。
TILT_SLOTS = (
    "+10° 倾角（例如垫起一侧车轮）",
    "0° 水平（恢复水平静置）",
    "−10° 倾角（例如垫起另一侧车轮）",
)
TILT_CHANGE_SEC = 10.0                # --tilt-rolling: 段间留给操作者改倾角的时间（闭环守位）

# ── 路径 ──
TARGET_DIR = os.path.join(_REPO, "data", "targets")
DEFAULT_OUT_DIR = os.path.join(_REPO, "data", "sysid")

# ── 轴编号（写进数据的约定）──
AXIS_BIG = 0        # 0 = 大 yaw 被激励
AXIS_SMALL = 1      # 1 = 小 yaw 被激励
AXIS_NAME = {AXIS_BIG: "big", AXIS_SMALL: "small"}

# ── CSV 列头（前 10 列与 docs/sysid_data.md §2 逐字一致；末尾两列是可选的重力列）──
#   gravity_ax / gravity_ay: 重力在 **A 系（大 yaw 转子系）** 的平面分量 (m/s²)，
#   水平静置时 ≈ 0。**追加在最后**是为了让按列名取列的读取器
#   (findCol) 继续工作；只有这两列"有非零值"时，下游才会启用重力项。
# ════════════════════════════════════════════════════════════════════════════
# 落盘列（★ 全量: 收到的、下发的、估计出来的全部保存）
#
# 前置 12 列沿用旧名，保证老数据与老辨识器继续可用。**注意两个历史列的含义**:
#   `theta_big`  = 大 yaw **电机侧**角度（MCU 编码器 + 延时补偿）—— 旧语义，未变
#   `dtheta_big` = 大 yaw **云台侧**角速度（IMU 陀螺投影）—— 旧语义，**与 theta_big 不同源**
#   （这正是背隙问题的根源；新代码请用下面显式的 `*_motor` / `*_platform` 四列）
# ════════════════════════════════════════════════════════════════════════════
CSV_HEADER = [
    # ── 前置列（旧名; 前 10 列与 docs/sysid_data.md §2 逐字一致）──
    "t", "theta_big", "theta_small", "dtheta_big", "dtheta_small",
    "tau_big", "tau_small", "axis", "held_target", "mcu2_seq",
    "gravity_ax", "gravity_ay",
    # ── ★ 大 yaw 电机侧 / 云台侧 显式分离（背隙标定的核心列）──
    "theta_big_motor", "dtheta_big_motor",
    "theta_big_platform", "dtheta_big_platform",
    "theta_big_motor_meas",
    # ── 估计器其余输出与诊断（只记录，拟合不用）──
    "platform_azimuth", "platform_rate",
    "small_joint_angle_est", "small_joint_rate_est",
    "pitch_joint_angle", "pitch_joint_rate", "pitch_acc",
    "chassis_azimuth", "chassis_yaw_rate",
    "big_enc_age", "big_sample_interval", "big_enc_innovation", "chassis_imu_age",
    "gravity_a_x", "gravity_a_y", "gravity_a_z",
    "base_omega_x", "base_omega_y", "base_omega_z",
    "los_azimuth", "los_elevation",
    # ── MCU 反馈（已按 LinearParams 映射；原始值 = 反解映射常量）──
    "mcu_bullet_velocity", "mcu_pitch_angle",
    "mcu_yaw_big_angle", "mcu_yaw_big_omega",
    "mcu_yaw_small_angle", "mcu_yaw_small_omega",
    "mcu_chassis_imu_yaw", "mcu_chassis_imu_omega",
    "mcu_mark", "mcu_color", "mcu_auto_aim_switch",
    "mcu_temp_big", "mcu_temp_small",
    # ── IMU 反馈（原始值，未滤波）──
    "imu_gx", "imu_gy", "imu_gz", "imu_ax", "imu_ay", "imu_az",
    "imu_euler_yaw", "imu_euler_pitch", "imu_euler_roll", "imu_dt_one_tenth_ms",
    # ── 下发（本拍实际发出的值; 力矩 N·m / 关节角 rad）──
    "tx_auto_aim_enable", "tx_fire", "tx_pitch_target_angle",
    "tx_yaw_big_mode", "tx_yaw_big_target_angle", "tx_yaw_big_target_velocity",
    "tx_yaw_big_torque",
    "tx_yaw_small_mode", "tx_yaw_small_target_angle", "tx_yaw_small_target_velocity",
    "tx_yaw_small_torque",
    # ── 参考（本拍目标的参考序列值）与有效位 ──
    "target_big", "target_small",
    "est_valid", "mcu_valid", "imu_valid",
    # ── ★ 背隙中心 β（死区中心）──
    #   `backlash_center`    = **运行期**估计器的在线值（滑动 min/max；实机 = est.backlash_center，
    #                          dry-run 里用同一套规则在 SimRobotLink 内复算）
    #                          ⇒ 3-DOF 辨识用它当**逐样本外生量**（与 MPC 运行期一致）
    #   `backlash_beta_true` = 仿真环境的**真值 β**（仅 dry-run 非 0；实机恒 0 = 未知）
    "backlash_center", "backlash_beta_true",
    # ── ★ 仿真**真值状态**（仅 dry-run 非 0；实机恒 0 = 未知）──
    #   用途: ① 量化"电机侧延时补偿估计"的误差（直接对比 theta_big_motor）；
    #         ② 辨识的**上限对照**（`--state-mode=true`: 用真值当初值与拟合目标，
    #            把"模型误差"和"状态估计误差"分开）。
    "theta_true_motor", "theta_true_platform", "theta_true_small",
    "dtheta_true_motor", "dtheta_true_platform", "dtheta_true_small"]

# 由链路 ``read()`` 提供的列（其余列在 make_row 里按控制步填）
_SAMPLE_FIELDS = (
    "platform_azimuth", "platform_rate",
    "big_joint_angle", "big_joint_rate", "big_joint_angle_meas",
    "theta_big_motor", "dtheta_big_motor", "theta_big_platform", "dtheta_big_platform",
    "big_enc_age", "big_sample_interval", "big_enc_innovation",
    "small_joint_angle", "small_joint_rate",
    "pitch_joint_angle", "pitch_joint_rate", "pitch_acc",
    "chassis_yaw", "chassis_omega", "chassis_imu_age",
    "gravity_ax", "gravity_ay", "gravity_az",
    "base_omega_x", "base_omega_y", "base_omega_z",
    "los_azimuth", "los_elevation",
    "mcu_bullet_velocity", "mcu_pitch_angle",
    "mcu_yaw_big_angle", "mcu_yaw_big_omega",
    "mcu_yaw_small_angle", "mcu_yaw_small_omega",
    "mcu_chassis_imu_yaw", "mcu_chassis_imu_omega",
    "mcu_mark", "mcu_color", "mcu_auto_aim_switch",
    "mcu_temp_big", "mcu_temp_small", "mcu2_seq",
    "backlash_center", "backlash_beta_true",
    "theta_true_motor", "theta_true_platform", "theta_true_small",
    "dtheta_true_motor", "dtheta_true_platform", "dtheta_true_small",
    "imu_gx", "imu_gy", "imu_gz", "imu_ax", "imu_ay", "imu_az",
    "imu_euler_yaw", "imu_euler_pitch", "imu_euler_roll", "imu_dt_one_tenth_ms",
    "est_valid", "mcu_valid", "imu_valid")

# ── dry-run 仿真参数（与 mpc/planar_yaw_model.h 的 ModelParams 默认值一致）──
SIM_INT_STEP = 5e-5          # 0.05 ms 积分步长（RK4 稳定: 摩擦模态时间常数 ~2 ms ≫ 0.05 ms）
SIM_FRICTION_LAMBDA = 100.0  # ★ 大 λ: 模拟真实库仑摩擦（tanh 软符号很陡）
SIM_CHASSIS_AZIMUTH = 0.0    # 仿真里底盘的初始方位角（底盘数据只记录、不参与拟合）
# ★ 底盘 IMU 也在这条链路上（与大 yaw 编码器同一包、同一序号、同样延迟 + 值保持），
#   所以它的方位角/角速度也从"被保持的样本"里取（见 SimRobotLink）；采集规范要求底盘静止
#   （倾斜 ≠ 底盘运动），因此这里没有底盘运动学，只有链路语义。
SIM_TRANSPORT_DELAY = 0.015  # 链路传输时延（s），与估计器默认 transport_delay_s 一致
# ── ★ 仿真环境 2（--sim-rigid）: "接触完全刚性 + 死区完全自由 + β 随机/漂移" ──
SIM_BETA_RANDOM_FRAC = 0.30   # 每条数据的 β0 在 ±0.30·δ 内随机
SIM_BETA_DRIFT_FRAC = 0.05    # 漂移幅值 = 0.05·δ
SIM_BETA_DRIFT_PERIOD = 30.0  # 漂移周期 (s)
SIM_BETA_TAU_S = 3.0          # β 在线估计的遗忘时间常数（= 估计器默认 backlash_center_tau_s）
SIM_ENC_NOISE = 2e-5         # 编码器噪声标准差（rad），仅让 PID 微分项有真实感
# ── ★ 仿真环境 3（--sim-no-backlash）: "没有背隙"（诊断/可辨识性实验用）──
#   δ = 0 ⇒ 电机与云台之间没有空行程，只剩一条刚度为 SIM_NO_BACKLASH_K 的"同步带"弹簧
#   （τ_t = k·Δ + c·Δ̇）。它是"真实系统其实没有背隙"时当前 16 参模型的识别极限实验。
SIM_NO_BACKLASH_K = 200.0     # 无背隙时那条"同步带"的刚度（N·m/rad）
# ── ★ 随机重心偏置（--sim-com-random）: 每次运行抽一次（不是每段抽！）──
#   为什么不是每段抽: P/Pb 是**全局**参数，逐段变化会让"拟合一个常数"这件事本身无解
#   （真机重心也不会每段变）。所以每次采集运行抽一组，写进 npz 供诊断。
SIM_COM_P_RANGE = (0.005, 0.020)      # |P| = |小 yaw 上装一阶矩| (kg·m)
SIM_COM_PB_RANGE = (0.003, 0.015)     # |Pb| = |大 yaw 侧一阶矩|  (kg·m)


def log(msg: str = "") -> None:
    """带 flush 的打印（串口/仿真循环里需要立即看到进度）。"""
    print(msg, flush=True)


# ============================================================================
# 小工具
# ============================================================================
def busy_wait_until(target_ns: int) -> None:
    """忙等到绝对时间点（微秒级抖动；sleep 的毫秒抖动会污染辨识）。"""
    while time.perf_counter_ns() < target_ns:
        pass


def wrap_pi(x: float) -> float:
    """把角度折叠到 [−π, π]（大 yaw 多圈方位角/小 yaw 相对角都安全）。"""
    return math.remainder(x, TWO_PI)


def _deg(rad: float) -> float:
    return math.degrees(rad)


def _gravity_a_plane(est):
    """从估计结果里取重力在 **A 系（大 yaw 转子系）** 的平面分量 ``(gx, gy)``。

    字段名: C API v4 起是 ``gravity_a``（旧版本叫 ``gravity_c``，语义不同），因此这里
    两种都试着取；字段不存在时返回 ``(0, 0)``（= 与旧数据等价的"水平假设"），
    保证脚本在任何库版本下都不会因为缺字段而崩掉。
    """
    for name in ("gravity_a", "gravity_c"):
        g = getattr(est, name, None)
        if g is None:
            continue
        try:
            return float(g[0]), float(g[1])
        except (TypeError, IndexError, ValueError):
            continue
    return 0.0, 0.0


# ============================================================================
# PID + 力矩变化率限幅
# ============================================================================
class PidController:
    """位置式 PID: out = kp·e + ki·∫e + kd·ė，输出限幅 ±1.0 N·m，条件积分抗饱和。

    ★ **误差死区**（`deadband` = d > 0 时启用，用户要求）—— 用的是**「到死区边界的距离」**，
      不是「到目标点的距离」::

          e = 目标 − 实际
          |e| ≤ d  ⇒  e_eff = 0          （死区内: 只保留积分项 ⇒ 不追噪声、不在背隙里蹭）
          e >  +d  ⇒  e_eff = e − d      （到**正侧死区边界**的距离）
          e <  −d  ⇒  e_eff = e + d      （到**负侧死区边界**的距离）

      好处: `e_eff` 在 `|e| = d` 处**连续**（从 0 开始长起来）⇒ 输出不会像"死区内直接置 0"
      那样在边界上跳变 `kp·d`（kp=5、d=3° 时那是 0.26 N·m 的阶跃，会激起背隙撞击/极限环）。
      等价说法: 死区把**等效目标**从"目标点"变成了"离实际值最近的那个死区边界"。
      · P / D / 积分**都用 e_eff**；死区内 `e_eff=0` ⇒ 积分自然冻结（不清零，否则会撒手）；
      · 抗饱和判据用 `e_eff` 的符号（死区内为 0 ⇒ 不积分）。

    与旧采集脚本**逐行同构**（`deadband=0` 时逐字同构），便于两批数据合并。
    """

    def __init__(self, kp, ki, kd, out_min, out_max, name="pid", deadband: float = 0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.name = name
        self.deadband = max(0.0, float(deadband))
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        # ★ 死区: |e| ≤ d ⇒ e_eff = 0；|e| > d ⇒ e_eff = e ∓ d（**到死区边界**的距离）
        #   ⇒ e_eff 在边界处连续，输出无跳变（见类 docstring）。
        d = self.deadband
        if d <= 0.0:
            e_eff = error
        elif error > d:
            e_eff = error - d
        elif error < -d:
            e_eff = error + d
        else:
            e_eff = 0.0
        deriv = (e_eff - self.prev_error) / dt if dt > 1e-6 else 0.0
        self.prev_error = e_eff
        out = self.kp * e_eff + self.ki * self.integral + self.kd * deriv
        sat_hi = out > self.out_max
        sat_lo = out < self.out_min
        if sat_hi:
            out = self.out_max
        if sat_lo:
            out = self.out_min
        do_int = True
        if sat_hi and e_eff > 0:
            do_int = False
        if sat_lo and e_eff < 0:
            do_int = False
        if do_int:
            self.integral += e_eff * dt        # 死区内 e_eff=0 ⇒ 积分自然冻结
        return out

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class TorqueRateLimiter:
    """相邻两步力矩变化 ≤ MAX_TORQUE_DELTA（每轴一个实例，跨相位共享状态）。

    记录进数据的必须是**限幅后真正下发**的值，否则辨识用的力矩 ≠ 实际作用力矩。
    """

    def __init__(self, max_delta=MAX_TORQUE_DELTA):
        self.max_delta = max_delta
        self.last = 0.0

    def limit(self, torque: float) -> float:
        delta = torque - self.last
        if delta > self.max_delta:
            torque = self.last + self.max_delta
        elif delta < -self.max_delta:
            torque = self.last - self.max_delta
        self.last = torque
        return torque

    def clear(self):
        self.last = 0.0


# ============================================================================
# 录制目标序列: 加载 / 增强 / 平滑
# ============================================================================
def load_all_targets() -> list:
    """加载 ``data/targets/*.npz`` 的 ``target`` 列（100 Hz 录制的参考角序列, rad）。"""
    if not os.path.isdir(TARGET_DIR):
        raise SystemExit(f"[ERROR] 找不到目标序列目录: {TARGET_DIR}")
    files = sorted(f for f in os.listdir(TARGET_DIR) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"[ERROR] {TARGET_DIR} 下没有 *.npz 目标序列")
    targets = []
    for name in files:
        with np.load(os.path.join(TARGET_DIR, name)) as data:
            targets.append((name, np.asarray(data["target"], dtype=np.float64)))
    total = sum(len(t) for _n, t in targets)
    log(f"加载目标序列: {len(files)} 个文件, 共 {total} 点 @100Hz")
    return targets


def _load_unit_shape(rng, targets, n: int):
    """随机挑一个"确实有运动"的录制窗口，返回**半幅归一化为 1 的零均值形状**。

    返回 ``(文件名, 起点, 单位形状, 原始半幅)``。

    · **连续化**: 录制的是绝对角度，可能带 ±2π 卷绕 → ``np.unwrap`` 成连续曲线；
    · **去直流**: 只保留"形状"（激励形态 + 换向次数），绝对位置交给随机中心
      （给平滑后的序列整体加常量不影响规划器行为: ``step()`` 只看 ``target − p``）；
    · **挑窗口重试**: 录制序列里存在大量静止段（整段一字不动），随机截取可能截到平段 ——
      那样的段没有任何激励信息。因此重试直到峰峰值 ≥ ``MIN_SHAPE_SPAN``。
    """
    best = None
    for _ in range(WINDOW_TRIES):
        idx = int(rng.integers(0, len(targets)))
        name, arr = targets[idx]
        if len(arr) <= n:
            start, seq = 0, arr.copy()
        else:
            start = int(rng.integers(0, len(arr) - n + 1))
            seq = arr[start:start + n].copy()
        seq = np.unwrap(seq)
        span = float(seq.max() - seq.min()) if len(seq) else 0.0
        if best is None or span > best[0]:
            best = (span, name, start, seq)
        if span >= MIN_SHAPE_SPAN:
            break
    _span, name, start, seq = best
    seq = seq - float(seq.mean())
    amp0 = float(np.max(np.abs(seq))) if len(seq) else 0.0
    if amp0 <= 1e-9:                       # 理论上到不了（上面已挑过有运动的窗口）
        amp0 = 1.0
    return name, start, seq / amp0, amp0


def build_driven_reference(rng, targets, n: int, cap: float, min_amp: float,
                           refined: StepRefinementWrapper):
    """构造 driven 轴的参考: 单位形状 → 随机幅值 → 平滑 → **校验平滑后确实激励起来了**。

    这条流水线必须校验**平滑之后**的幅值，而不是缩放之前的幅值: 录制片段里有不少
    "尖刺型"窗口（相邻点跳变 1~2 rad），轨迹规划器会把它们抹平成一条几乎不动的直线 ——
    只看输入幅值会以为"这段有激励"，实际采到的是一段静止数据（实测出现过 3° 的段）。
    因此: 平滑后幅值 < ``min_amp`` 就**放大输入重试**，仍不行就**换窗口**。

    ``cap`` / ``min_amp`` **按轴给**: 小 yaw 行程只有 45° 宽，包络半宽才 14.5°，
    所以它的下限（7°）比大 yaw（14°）小得多。

    返回 ``(文件名, 起点, 平滑参考形状, 实际缩放系数)``；参考半幅不超过 ``cap``，
    且 ≥ ``min_amp``（除非所有候选窗口都是尖刺型，此时取最好的一个并告警）。
    """
    best = None
    for _ in range(REF_TRIES):
        name, start, unit, amp0 = _load_unit_shape(rng, targets, n)
        # 目标半幅: 该轴安全上限的 60%~100%，再乘一次随机缩放 ⇒ 幅值有多样性
        amp_goal = max(min_amp,
                       cap * float(rng.uniform(*AMP_JITTER)) * float(rng.uniform(0.5, 1.0)))
        amp_try = amp_goal
        ref = unit * amp_try
        amp_now = float(np.max(np.abs(ref))) if len(ref) else 0.0
        for _boost in range(AMP_BOOST_TRIES):
            ref = smooth_sequence(unit * amp_try, refined)
            amp_now = float(np.max(np.abs(ref))) if len(ref) else 0.0
            if amp_now >= min_amp or amp_now <= 1e-9:
                break
            amp_try = min(amp_try * AMP_BOOST_FACTOR, amp_goal * AMP_BOOST_FACTOR ** 2)
        if best is None or amp_now > best[0]:
            best = (amp_now, name, start, ref, amp_try, amp0)
        if amp_now >= min_amp:
            break
    amp_now, name, start, ref, amp_try, amp0 = best
    if amp_now < min_amp:
        log(f"  [WARN] 参考平滑后幅值仅 {_deg(amp_now):.1f}°（< {_deg(min_amp):.0f}°）: "
            f"候选窗口都是尖刺型，本段激励偏弱")
    ref = fit_into_band(ref, cap)          # 统一等比缩放，绝不截断
    return name, start, ref, (amp_try / amp0)


def smooth_sequence(shape: np.ndarray, refined: StepRefinementWrapper) -> np.ndarray:
    """用 TrajectoryPlanner + StepRefinementWrapper 把形状序列变成**可跟踪的平滑参考**。

    录制序列含真实工况的换向与跳变；直接当参考会让 PID 长期饱和打滑（力矩恒为 ±1，
    回归矩阵几乎常数，信息量极低）。规划器把参考限制在速度/加速度/加加速度上限内，
    得到"跟得住"的参考 ⇒ 力矩随工况连续变化，辨识才有信息。
    """
    n = len(shape)
    out = np.zeros(n, dtype=np.float64)
    pos = float(shape[0])
    vel = 0.0
    acc = 0.0
    for i in range(n):
        pos, vel, acc, _ = refined.step(float(shape[i]), pos, vel, acc, DT)
        out[i] = pos
    return out


def fit_into_band(seq: np.ndarray, limit: float) -> np.ndarray:
    """超范围时**整体等比缩放**到 ±limit 以内（保留轨迹形状，绝不截断）。"""
    peak = float(np.max(np.abs(seq))) if len(seq) else 0.0
    if peak > limit and peak > 0.0:
        seq = seq * (limit / peak)
    return seq


def fit_into_interval(seq: np.ndarray, lo: float, hi: float):
    """**整体等比缩放 + 平移**，把 ``seq`` 放进非对称区间 ``[lo, hi]``（保形状、绝不截断）。

    为什么不是 ``fit_into_band`` 那种"绕 0 缩放": 小 yaw 行程可能是**非对称**区间（例如 −20° … +25°），
    绕 0 缩放会把轨迹推向一侧、白吃余量。这里:
      1) 先按区间**宽度**统一等比缩放（形状不变）: ``k = min(1, (hi−lo)/峰峰值)``；
      2) 再把缩放后序列的**中点平移到区间中点**（此时必定落在区间内，因为峰峰值 ≤ 宽度）；
      3) 最后做一次数值兜底平移（浮点误差/极端形状时也不会越界）。
    返回 ``(新序列, 缩放系数 k, 平移量 shift)``。
    """
    seq = np.asarray(seq, dtype=np.float64)
    if len(seq) == 0:
        return seq, 1.0, 0.0
    width = float(hi) - float(lo)
    span = float(seq.max() - seq.min())
    k = 1.0 if span <= 1e-12 else min(1.0, width / span)
    out = seq * k
    mid = 0.5 * (float(hi) + float(lo))
    shift = mid - 0.5 * float(out.max() + out.min())      # 中点对齐 ⇒ 整体落在区间内
    out = out + shift
    if float(out.max()) > hi:                             # 数值兜底（不该发生）
        out = out - (float(out.max()) - float(hi))
    if float(out.min()) < lo:
        out = out + (float(lo) - float(out.min()))
    return out, k, shift


def random_center_for(rng, amp: float, lo: float, hi: float,
                      jitter: float | None = None) -> float:
    """在"让整条 ±amp 的轨迹落在 [lo, hi] 内"的**可行中心区间**里随机取一个中心。

    非对称行程下**不能**再用 ``±(band − amp)`` 那种对称写法:
    可行中心区间 = ``[lo + amp, hi − amp]``（lo=硬限位下界+余量, hi=硬限位上界−余量）。
    可行区间为空（幅值比区间还宽）时退回区间中点。
    """
    c_lo, c_hi = float(lo) + amp, float(hi) - amp
    if c_lo > c_hi:
        return 0.5 * (float(lo) + float(hi))
    if jitter is not None:                                # 额外围绕区间中点收窄（可选）
        mid = 0.5 * (float(lo) + float(hi))
        c_lo, c_hi = max(c_lo, mid - jitter), min(c_hi, mid + jitter)
        if c_lo > c_hi:
            c_lo = c_hi = mid
    return float(rng.uniform(c_lo, c_hi))


def homing_sequence(current: float, target: float, n: int,
                    refined: StepRefinementWrapper) -> np.ndarray:
    """到位阶段的参考: 把"当前角 → 目标角"的**阶跃**交给同一个轨迹规划器整形。

    为什么不能直接给阶跃: PID 面对阶跃会一直饱和到限幅，靠 kd 与限幅刹车，
    **必然过冲**（仿真实测: 小 yaw 从 +15° 走到 −31° 时中途冲到 47°，直接触发
    45° 中止保护，白白浪费一段）。规划器给的参考有速度/加速度/加加速度上限，
    并且会"提前刹车"，移动平滑且不过冲。
    """
    raw = np.empty(n, dtype=np.float64)
    raw[0] = float(current)
    raw[1:] = float(target)
    return smooth_sequence(raw, refined)


def make_planners() -> dict:
    """两条轴的参考规划器（细化 1000 子步/控制周期，与旧脚本一致）。"""
    return {
        "big": StepRefinementWrapper(
            TrajectoryPlanner(**BIG_PLANNER).step, REFINE_N),
        "small": StepRefinementWrapper(
            TrajectoryPlanner(**SMALL_PLANNER).step, REFINE_N),
    }


# ============================================================================
# 数据容器
# ============================================================================
class RobotSample:
    """一次读数。两种链路（真实/仿真）的 ``read()`` 返回同一种对象，
    采集逻辑因此与硬件完全解耦 —— dry-run 与实机走同一条代码路径。"""

    __slots__ = _SAMPLE_FIELDS

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, 0.0))


@dataclass
class SegmentPlan:
    """一段采集的完整计划。"""

    axis: int
    held_target: float        # held 轴目标角（大 yaw: 平台方位角；小 yaw: 关节角）
    ref_big: np.ndarray       # 每步大 yaw 目标（平台方位角语义；held 时为常量）
    ref_small: np.ndarray     # 每步小 yaw 目标（关节角语义；held 时为常量）
    ref_center: float = 0.0   # driven 参考的随机中心（大 yaw: 平台方位角; 小 yaw: 关节角）
    ref_amp: float = 0.0      # driven 参考的半幅（相对 ref_center）
    tilted: int = 0           # 1 = 本段是在 --tilted（静态倾斜静置）下采集的
    tilt_slot: int = -1       # --tilt-rolling 时的倾角槽位下标（-1 = 未轮换）
    held_strat_index: int = -1   # --held-big-stratified: 本段 held 方位角的下标
    held_strat_count: int = 0    # --held-big-stratified: 分层总数（0 = 未启用）
    held_strat_offset: float = 0.0   # 相对分层基准的偏移 rad
    no_backlash: int = 0      # 1 = dry-run 用的被控对象**没有背隙**（δ=0；诊断/可辨识性实验）
    # 本段实际使用的 PID 配置（★ 记**真正用的参数**，而不是模块默认常量）
    kp: float = PID_KP
    ki: float = PID_KI
    kd: float = PID_KD
    pid_deadband: float = 0.0    # PID 误差死区（rad；0 = 关闭）
    # dry-run 用的被控对象**重心真值**（kg·m；实机恒 0 = 未知，只用于诊断/出报告）
    px_true: float = 0.0
    py_true: float = 0.0
    pbx_true: float = 0.0
    pby_true: float = 0.0
    src_file: str = ""        # 参考来源（录制文件名）
    src_start: int = 0        # 参考在录制文件中的起点下标
    src_scale: float = 0.0    # 随机缩放系数


class SegmentRecord:
    """一段采集的落盘数据（各列等长，长度 = 段点数）。"""

    def __init__(self):
        # 按列名存列表（列集 = CSV_HEADER）⇒ 新增记录列不用改这个类
        self.cols: dict = {name: [] for name in CSV_HEADER}

    def append(self, row: dict) -> None:
        for key in row:
            if key not in self.cols:
                raise KeyError(f"未知记录列 {key!r}（请加进 collect_sysid.CSV_HEADER）")
        miss = [k for k in self.cols if k not in row]
        if miss:
            raise KeyError(f"本拍缺少这些记录列: {miss}（make_row 里没填）")
        for key, value in row.items():
            self.cols[key].append(value)

    def col(self, name: str) -> list:
        return self.cols[name]

    def __len__(self) -> int:
        return len(self.cols["t"])


# ============================================================================
# 采集计划: driven 参考 + held 目标
# ============================================================================
def build_segment_plan(rng, targets, axis: int, st, planners: dict, n: int,
                       held_big_override: float | None = None) -> SegmentPlan:
    """构造一段的参考: driven 轴 = 增强后的录制序列；held 轴 = 常量目标。

    · driven = 大 yaw: 参考中心取**现有平台方位角附近**（±BIG_CENTER_JITTER），
      半幅 ≤ BIG_REF_AMP(120°)；held = 小 yaw 目标在**参考包络内**随机
      （收进 0.7 倍 ⇒ SMALL_CENTER_RAD ± 10.15°，给大 yaw 摆动经 M12 传来的耦合偏移留余量）。
    · driven = 小 yaw: 参考落在**参考包络 [−22°, +22°]** 内（硬限位 ±30°
      两侧各留 8° 跟踪超调余量），中心在可行中心区间 [env_min+amp, env_max−amp] 内随机；
      超出包络时**整体等比缩放 + 平移到包络内**（保形状、不截断）；
      held = 大 yaw 目标取现有方位角 ±π 内随机（多圈连续，无需限幅）。
    """
    if axis == AXIS_BIG:
        # ── 大 yaw 被激励; 小 yaw 由 PID 保持在包络内的随机固定角 ──
        held_target = float(SMALL_CENTER_RAD) + float(
            rng.uniform(-HELD_SMALL_MAX, HELD_SMALL_MAX))
        name, start, ref_shape, scale = build_driven_reference(
            rng, targets, n, BIG_REF_AMP, MIN_EXCITE_AMP_BIG, planners["big"])
        # 随机中心: 现在角度附近 ±BIG_CENTER_JITTER（不跳到大角度，避免采样前的大行程）
        center = float(st.platform_azimuth) + float(
            rng.uniform(-BIG_CENTER_JITTER, BIG_CENTER_JITTER))
        ref_big = ref_shape + center
        ref_small = np.full(n, held_target, dtype=np.float64)
        ref_center, ref_amp = center, float(np.max(np.abs(ref_big - center)))
    else:
        # ── 小 yaw 被激励; 大 yaw 由 PID 保持在随机方位角 ──
        held_target = (float(held_big_override) if held_big_override is not None
                       else float(st.platform_azimuth) + float(
                           rng.uniform(-HELD_BIG_OFFSET, HELD_BIG_OFFSET)))
        name, start, ref_shape, scale = build_driven_reference(
            rng, targets, n, SMALL_REF_AMP, MIN_EXCITE_AMP_SMALL, planners["small"])
        amp = float(np.max(np.abs(ref_shape)))
        # 随机中心: 在"让整条轨迹落在参考包络内"的**可行中心区间**里随机取
        # （行程可能非对称 ⇒ 不能用 ±(band−amp) 的对称写法）
        center = random_center_for(rng, amp, SMALL_ENV_MIN, SMALL_ENV_MAX)
        ref_small = ref_shape + center
        # 规格: 若仍超出包络（规划器过冲/浮点）→ **整体等比缩放 + 平移到包络内**（不截断）
        ref_small, _k, _shift = fit_into_interval(ref_small, SMALL_ENV_MIN, SMALL_ENV_MAX)
        ref_big = np.full(n, held_target, dtype=np.float64)
        ref_center, ref_amp = center, float(np.max(np.abs(ref_small - center)))

    return SegmentPlan(
        axis=axis, held_target=held_target,
        ref_big=np.asarray(ref_big, dtype=np.float64),
        ref_small=np.asarray(ref_small, dtype=np.float64),
        ref_center=ref_center, ref_amp=ref_amp,
        src_file=name, src_start=start, src_scale=scale)


# ============================================================================
# 链路 1: 真实硬件（低层 TcbsRobotCommunication，直接下发力矩）
# ============================================================================
class HwRobotLink:
    """串口链路。**只用低层 API** ``get_latest_data`` / ``get_estimate`` /
    ``send_to_mcu`` —— 采集必须自己掌握力矩通道，不能用高层 MPC 控制器。"""

    def __init__(self):
        self.comm = TcbsRobotCommunication()
        self.tx_fail = 0

    def begin_segment(self, index: int = 0) -> None:
        """实机没有"每段随机 β"这件事（β 由估计器在线给）⇒ 空操作。"""
        pass

    def wait_ready(self, timeout_s: float = 10.0) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            data = self.comm.get_latest_data()
            if data.mcu.valid and data.imu.valid:
                return True
            time.sleep(0.01)
        return False

    def read(self) -> RobotSample:
        """读一帧，**把收到的所有量都填进 RobotSample**（由 make_row 全部落盘）。

        大 yaw 的三个角度在这里被显式分开:
          * ``theta_big_motor``    = 电机侧（MCU 编码器 + 延时补偿）—— 控制/辨识的"电机角"
          * ``theta_big_motor_meas`` = 同上但**未做延时补偿**（原始滞后测量）
          * ``theta_big_platform`` = 云台侧（IMU 反解）: θ_p = platform_azimuth − chassis_azimuth
        两者之差就是传动形变 Δ（背隙建模要用的量）。
        """
        data = self.comm.get_latest_data()
        est = self.comm.get_estimate()
        imu = data.imu
        mcu = data.mcu
        gx, gy = _gravity_a_plane(est)
        # 云台侧关节角 = 平台世界方位角 − 底盘方位角（两者都是解卷绕后的连续量）
        big_platform = float(est.platform_azimuth) - float(est.chassis_azimuth)
        big_motor = float(est.big_joint_angle)
        # 云台侧角速度: 估计器的 platform_rate 就是"相对底盘的关节角速度"（IMU 陀螺投影）
        big_platform_rate = float(est.platform_rate)
        bw = est.base_omega
        ga = est.gravity_a
        return RobotSample(
            # ── 大 yaw: 电机侧 / 云台侧 显式分离 ──
            big_joint_angle=big_motor,                        # 兼容旧语义
            big_joint_angle_meas=float(est.big_joint_angle_meas),
            big_joint_rate=big_platform_rate,                 # 兼容旧语义（= 云台侧）
            theta_big_motor=big_motor,
            dtheta_big_motor=float(mcu.yaw_big_omega),        # 电机侧角速度来自 MCU 编码器
            theta_big_platform=big_platform,
            dtheta_big_platform=big_platform_rate,
            # ── 平台/关节角度与角速度（估计器输出）──
            platform_azimuth=float(est.platform_azimuth),
            platform_rate=float(est.platform_rate),
            small_joint_angle=float(est.small_joint_angle),
            small_joint_rate=float(est.small_joint_rate),
            small_joint_angle_est=float(est.small_joint_angle),
            small_joint_rate_est=float(est.small_joint_rate),
            pitch_joint_angle=float(est.pitch_joint_angle),
            pitch_joint_rate=float(est.pitch_joint_rate),
            pitch_acc=float(est.pitch_acc),
            chassis_yaw=float(est.chassis_azimuth),
            chassis_omega=float(est.chassis_yaw_rate),
            chassis_azimuth=float(est.chassis_azimuth),
            chassis_yaw_rate=float(est.chassis_yaw_rate),
            # ── 链路诊断（值年龄/间隔/新样本统计）──
            big_enc_age=float(est.big_enc_age),
            big_sample_interval=float(est.big_sample_interval),
            big_enc_innovation=float(est.big_enc_innovation),
            chassis_imu_age=float(est.chassis_imu_age),
            # ── 模型外生量 ──
            gravity_ax=gx, gravity_ay=gy, gravity_az=float(ga[2]),
            base_omega_x=float(bw[0]), base_omega_y=float(bw[1]), base_omega_z=float(bw[2]),
            # ★ 背隙中心: 实机直接用估计器的**在线**值（真值未知 ⇒ 恒 0）
            backlash_center=float(est.backlash_center), backlash_beta_true=0.0,
            los_azimuth=float(est.los_azimuth), los_elevation=float(est.los_elevation),
            # ── MCU 反馈（全部字段；已按 LinearParams 映射）──
            mcu_bullet_velocity=float(mcu.bullet_velocity),
            mcu_pitch_angle=float(mcu.pitch_angle),
            mcu_yaw_big_angle=float(mcu.yaw_big_angle),
            mcu_yaw_big_omega=float(mcu.yaw_big_omega),
            mcu_yaw_small_angle=float(mcu.yaw_small_angle),
            mcu_yaw_small_omega=float(mcu.yaw_small_omega),
            mcu_chassis_imu_yaw=float(mcu.chassis_imu_yaw),
            mcu_chassis_imu_omega=float(mcu.chassis_imu_omega),
            mcu_mark=int(mcu.mark), mcu_color=int(mcu.color),
            mcu_auto_aim_switch=int(mcu.auto_aim_switch),
            mcu_temp_big=int(mcu.yaw_big_temperature),
            mcu_temp_small=int(mcu.yaw_small_temperature),
            mcu2_seq=int(mcu.mcu2_seq),
            # ── IMU 反馈（原始 6 轴 + 欧拉 + 帧间隔）──
            imu_gx=float(imu.gx), imu_gy=float(imu.gy), imu_gz=float(imu.gz),
            imu_ax=float(imu.ax), imu_ay=float(imu.ay), imu_az=float(imu.az),
            imu_euler_yaw=float(imu.euler_yaw),
            imu_euler_pitch=float(imu.euler_pitch),
            imu_euler_roll=float(imu.euler_roll),
            imu_dt_one_tenth_ms=int(imu.dt_one_tenth_ms),
            # ── 有效位 ──
            est_valid=int(est.valid), mcu_valid=int(mcu.valid), imu_valid=int(imu.valid))

    def send(self, tau_big, tau_small, big_joint_target, small_joint_target) -> bool:
        # 仅力矩模式（yaw_*_mode = 0）: 电控直接施加 yaw_*_torque。
        # 目标角/角速度字段在 mode 0 下电控不使用，但仍按"关节角语义"填上当前意图:
        #   大 yaw = 关节角（多圈连续）; 小 yaw = 相对角（±45° 内）。
        ok = self.comm.send_to_mcu(
            auto_aim_enable=AUTO_AIM_ENABLE, fire=0,
            pitch_target_angle=PITCH_TARGET_ANGLE,
            yaw_big_mode=YAW_MODE_TORQUE_ONLY,
            yaw_big_target_angle=float(big_joint_target),
            yaw_big_target_velocity=0.0,
            yaw_big_torque=float(tau_big),
            yaw_small_mode=YAW_MODE_TORQUE_ONLY,
            yaw_small_target_angle=float(small_joint_target),
            yaw_small_target_velocity=0.0,
            yaw_small_torque=float(tau_small))
        if not ok:
            self.tx_fail += 1
            # 连续发不出去 = 串口断了。此时继续"采样"只是在录一份没有力矩的垃圾数据，
            # 而且电控侧收不到指令会进保护 → 直接报错退出（finally 里仍会尝试发零力矩）。
            if self.tx_fail >= MAX_TX_FAIL:
                raise RuntimeError(
                    f"连续 {self.tx_fail} 帧未写入串口（send_to_mcu 返回 0）—— 链路断开，停止采集")
        else:
            self.tx_fail = 0
        return bool(ok)

    def close(self) -> None:
        try:
            self.comm.stop()
        finally:
            self.comm.close()


# ============================================================================
# 链路 2: dry-run 内置仿真（planar_yaw_model.h 同方程）
# ============================================================================
class PlanarYawPlant:
    """dry-run 的被控对象: 与 ``include/tcbs/mpc/planar_yaw_model.h`` **同一组方程**。

        Q(θ_s) = R(θ_s)·P,  dQ = d·Q,  μ = 2(d_y Q_x − d_x Q_y)
        M11 = Jbig_eff + J_s + 2dQ,  M12 = J_s + dQ,  M22 = J_s
        G_s = Q_x g_y − Q_y g_x,  G_b = m_u_known(d_x g_y − d_y g_x) + G_s
        h_b = μ θ̇_b θ̇_s + ½μ θ̇_s² − G_b + μ θ̇_s ω_c + M11 α_c + fric_b + τ_off_b
        h_s = −½μ θ̇_b² − G_s − μ θ̇_b ω_c − ½μ ω_c² + M12 α_c + fric_s + τ_off_s
        fric = fc·tanh(λ θ̇) + fv θ̇
        q̈ = M⁻¹(τ − h)          （RK4，力矩零阶保持）
    """

    def __init__(self, int_step: float = SIM_INT_STEP, tilt_deg: float = 0.0,
                 gravity_a=None, **overrides):
        # ★ 真实量级参数（用户 + 原仓库 data/cars/*/params/Identified_parameters.txt:
        #   J=0.0165/0.0250, tau_c=0.0973/0.1225, b=0.0321/0.0256; 大小 yaw 同量级、
        #   大 yaw 因承载小 yaw ≈翻倍; 轴间距 ≈0.1 m; 小 yaw 上装 m_u≈0.1 kg、ρ≈0.1 m
        #   ⇒ |P|≈0.01 kg·m, 方向与 d 偏 30°）
        p = dict(
            dx=0.0, dy=0.07, gravity=9.81, m_u_known=0.0,
            Jbig_eff=0.050, Js=0.020, Px=0.00866, Py=0.005,
            # ★ 大 yaw 侧一阶矩（kg·m）: "只随大 yaw 转、不随小 yaw 转"的质量偏心。
            #   默认 0；`--sim-com-random` 会给 P 与 Pb 都抽一个随机偏置（见 main）。
            Pbx=0.0, Pby=0.0,
            fcBig=0.22, fvBig=0.055, fcSmall=0.0973, fvSmall=0.028,
            frictionLambda=SIM_FRICTION_LAMBDA,     # ★ 大 λ = 更接近真实库仑摩擦
            tau_offset_big=0.0, tau_offset_small=0.0)
        p.update(overrides)
        self.p = p
        # ★ 背隙参数（默认 = C++ defaultModelParams() 同一组）:
        #   τ_t = k·[dz(Δ) + γ·Δ] + c·Δ̇,  Δ = θ_motor − θ_platform（仿真里 β=0）
        p.setdefault("backlash_delta", 0.0873)
        p.setdefault("backlash_k", 200.0)
        p.setdefault("backlash_c", 2.0)
        p.setdefault("backlash_through", 0.002)
        p.setdefault("backlash_smooth_eps", 1.0e-4)
        p.setdefault("Jmotor", 0.006)
        p.setdefault("fcMotor", 0.030)
        p.setdefault("fvMotor", 0.010)
        p.setdefault("tau_offset_motor", 0.0)
        # 刚性环境的"每条数据随机 β"幅度（占 δ 的比例）；平滑环境不用
        p.setdefault("beta_random_frac", 0.0)
        self.int_step = float(int_step)
        # ★ 3-DOF: [θ_motor, θ_platform, θ_small]
        self.q = [0.0, 0.0, 0.0]
        self.qd = [0.0, 0.0, 0.0]
        # 外生量 (g_x, g_y, ω_c, α_c)。默认底盘**水平静止** ⇒ 重力平面分量为 0、
        # 底盘角速度/角加速度为 0。两种给重力的方式（这**不是**底盘运动，只是静置姿态不同，
        # base_omega/base_alpha 仍为 0）:
        #   · tilt_deg=X  : 绕 y 轴倾斜 X 度 ⇒ 底盘系平面分量 g_C = (g·sinX, 0)
        #   · gravity_a=(gx, gy): 直接给 A 系平面分量（标定脚本的"倾斜消融"需要 g_y ≠ 0）
        # ★★ `tilt_deg ≠ 0` 时 A 系重力**随大 yaw 平台角旋转**（物理正确）:
        #     g_A = Rz(−θ_p)·g_C ⇒ g_Ax = g_Cx·cosθ_p + g_Cy·sinθ_p, g_Ay = −g_Cx·sinθ_p + g_Cy·cosθ_p
        #   实机由估计器/固件给出 A 系 gravity_a（本来就跟着转）；之前 dry-run 把它**冻结**在
        #   `__init__` 的值上 ⇒ 倾斜 + 大 yaw 转动时动力学是错的（水平时 g_C=0，看不出问题）。
        #   显式给 `gravity_a=` 的老用法（标定消融）仍保持"恒定 A 系分量"语义。
        self._rot_grav = (gravity_a is None) and (abs(float(tilt_deg)) > 1e-12)
        if gravity_a is None:
            gx = p["gravity"] * math.sin(math.radians(float(tilt_deg)))
            gy = 0.0
        else:
            gx, gy = float(gravity_a[0]), float(gravity_a[1])
        self._g_c = (gx, gy)                     # 底盘系（C）平面分量
        self.exo = (gx, gy, 0.0, 0.0)

    def sync_exo(self) -> None:
        """把 A 系重力按当前平台角刷新（`g_A = Rz(−θ_p)·g_C`）。

        每个控制周期开头调用一次（与实机"每拍刷新一次 exo"的语义一致）。水平静置或
        显式给了 `gravity_a=` 时不做任何事。
        """
        if not getattr(self, "_rot_grav", False):
            return
        cp, sp = math.cos(self.q[1]), math.sin(self.q[1])
        gx, gy = self._g_c
        self.exo = (gx * cp + gy * sp, -gx * sp + gy * cp, 0.0, 0.0)

    # ── 派生量 ──
    def _derived(self, qs):
        p = self.p
        cs, sn = math.cos(qs), math.sin(qs)
        Qx = p["Px"] * cs - p["Py"] * sn
        Qy = p["Px"] * sn + p["Py"] * cs
        dQ = p["dx"] * Qx + p["dy"] * Qy
        mu = 2.0 * (p["dy"] * Qx - p["dx"] * Qy)
        M11 = p["Jbig_eff"] + p["Js"] + 2.0 * dQ
        M12 = p["Js"] + dQ
        return Qx, Qy, M11, M12, mu

    def _fric(self, w, fc, fv):
        return fc * math.tanh(self.p["frictionLambda"] * w) + fv * w

    def _backlash_torque(self, D, Dd):
        """τ_t = k·[dz(Δ) + γ·Δ] + c·Δ̇（dz 与 C++ 的平滑死区同式）"""
        p = self.p
        h = 0.5 * p["backlash_delta"]
        eps = p["backlash_smooth_eps"]

        def relu(x):
            return 0.5 * (x + math.sqrt(x * x + eps * eps))

        return (p["backlash_k"] * (relu(D - h) - relu(-D - h) + p["backlash_through"] * D)
                + p["backlash_c"] * Dd)

    def _h(self, q, qd):
        """云台+小 yaw 子块（与 C++ `eomBacklash` 内 `eom(qb, qdb, ...)` 同式）。

        ★ 3-DOF 下 q/qd 是 (电机, 云台, 小 yaw) ⇒ 子块的自变量是
        ``θ_small = q[2]``、``qd_b = (qd[1], qd[2])``（云台/小 yaw 角速度）——
        因为 2-DOF 子块里的 ``q[1]`` 指的就是**小 yaw 关节角**（耦合项 M11/M12/μ 与
        重力项 Gs 都按 R(θ_small)·P 算）。
        ⚠ 两处历史 bug（都必须用"被控对象 vs 辨识模型/C++ 逐点比对"才发现）:
          · 早期把 ``qd[0], qd[1]`` 当云台/小 yaw 角速度（用了**电机**角速度）；
          · 早期把 ``q[1]``（云台角，大 yaw 多圈）当小 yaw 角 ⇒ 耦合项与 Gs 的
            **θ 依赖整个错了**（Q 会随大 yaw 转好几圈），2026-09-20 修正为 ``q[2]``。
          受影响的只有 **dry-run 仿真数据**里与 P 相关的结论（δ/k/c/β 那套不受影响：
          τ_t 只依赖 Δ）；实机数据与 C++/辨识模型一直是对的。
        """
        p = self.p
        Qx, Qy, M11, M12, mu = self._derived(q[2])
        gx, gy, wc, ac = self.exo
        Gs = Qx * gy - Qy * gx
        # ★ 大 yaw 侧: 已知上装质量那份 + 偏心 Pb（与 C++ eom/辨识模型逐字同式）
        Gb = ((p["Pbx"] + p["m_u_known"] * p["dx"]) * gy
              - (p["Pby"] + p["m_u_known"] * p["dy"]) * gx) + Gs
        tb, ts = qd[1], qd[2]          # ★ 云台 / 小 yaw 角速度（不是电机/云台）
        h0 = (mu * tb * ts + 0.5 * mu * ts * ts - Gb + mu * ts * wc + M11 * ac
              + self._fric(tb, p["fcBig"], p["fvBig"]) + p["tau_offset_big"])
        h1 = (-0.5 * mu * tb * tb - Gs - mu * tb * wc - 0.5 * mu * wc * wc + M12 * ac
              + self._fric(ts, p["fcSmall"], p["fvSmall"]) + p["tau_offset_small"])
        return M11, M12, h0, h1

    def _accel(self, q, qd, u):
        """3-DOF: u = (τ_cmd_motor, 0, τ_small)。电机行与云台行只通过 τ_t 耦合。"""
        p = self.p
        M11, M12, h0, h1 = self._h(q, qd)
        tt = self._backlash_torque(q[0] - q[1], qd[0] - qd[1])
        hM = tt + self._fric(qd[0], p["fcMotor"], p["fvMotor"]) + p["tau_offset_motor"]
        hp = h0 - tt
        M22 = p["Js"]
        det = M11 * M22 - M12 * M12
        if abs(det) <= 1e-12 or abs(p["Jmotor"]) <= 1e-12:
            return 0.0, 0.0, 0.0
        inv = 1.0 / det
        r1 = u[1] - hp
        r2 = u[2] - h1
        return ((u[0] - hM) / p["Jmotor"],
                (M22 * r1 - M12 * r2) * inv,
                (-M12 * r1 + M11 * r2) * inv)

    def _rk4(self, hh, u):
        q, qd = self.q, self.qd
        k1 = self._accel(q, qd, u)
        q2 = [q[i] + 0.5 * hh * qd[i] for i in range(3)]
        qd2 = [qd[i] + 0.5 * hh * k1[i] for i in range(3)]
        k2 = self._accel(q2, qd2, u)
        q3 = [q[i] + 0.5 * hh * qd2[i] for i in range(3)]
        qd3 = [qd[i] + 0.5 * hh * k2[i] for i in range(3)]
        k3 = self._accel(q3, qd3, u)
        q4 = [q[i] + hh * qd3[i] for i in range(3)]
        qd4 = [qd[i] + hh * k3[i] for i in range(3)]
        k4 = self._accel(q4, qd4, u)
        h6 = hh / 6.0
        self.q = [q[i] + h6 * (qd[i] + 2.0 * qd2[i] + 2.0 * qd3[i] + qd4[i]) for i in range(3)]
        self.qd = [qd[i] + h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(3)]

    def step(self, tau, dt: float) -> None:
        """积分一个控制周期（力矩零阶保持）；``tau`` = (τ_big, τ_small)。

        3-DOF: τ_big 作用在**电机**（q[0]），经背隙传到云台（q[1]）；小 yaw 直接驱动。
        """
        u = (float(tau[0]), 0.0, float(tau[1]))
        n = max(1, int(round(dt / self.int_step)))
        hh = dt / float(n)
        self.sync_exo()                          # ★ 倾斜时 A 系重力随平台角旋转
        for _ in range(n):
            self._rk4(hh, u)
        self.sync_exo()                          # 让"被读出的 exo"与步末状态同刻（记录一致）


# ============================================================================
# ★ 仿真环境 2: 背隙"完全刚性 + 中间完全自由"（**仅用于采集测试数据**）
# ============================================================================
class RigidBacklashPlant(PlanarYawPlant):
    """背隙**接触面完全刚性**、死区内**完全自由**、且 β 随机 + 微弱漂移的仿真环境。

    与 ``PlanarYawPlant``（平滑死区 + 弹簧接触，`τ_t = k[dz(Δ)+γΔ] + cΔ̇`）的区别 —— 这里
    **没有 k/c/γ**，背隙是纯几何间隙 + 单向刚性约束:

      · **死区内完全自由**（|Δ_raw − β| < δ/2，Δ_raw = θ_motor − θ_platform）: ``τ_t ≡ 0``，
        电机与云台互不传力（连阻尼都没有）；
      · **接触后完全刚性**（Δ_raw − β = ±δ/2）: 电机与云台被**刚性锁定**（相对角、相对角速度
        都恒为常数/0），动力学退化成"电机+云台合并惯量"的 2-DOF 系统
        （``M11 ← M11 + Jmotor``、``h_b ← h_b + 电机摩擦``），接触力矩由电机行反解:
        ``τ_t = τ_cmd − h_motor − Jmotor·θ̈``；
      · **约束单向**: 若反解出的 ``τ_t`` 与接触侧符号相反（说明需要"拉"而不是"推"）⇒ 约束释放，
        回到自由段；
      · **撞击 = 完全非弹性冲击**: 自由段撞到边界时，用广义动量守恒 + 冲击后相对速度 = 0
        求解冲击（等价于沿约束方向施加脉冲），然后转入刚性锁定；
      · ★ **β 每条数据随机、并随时间微弱漂移**: ``β(t) = β0 + A·sin(2πt/T)``，
        ``β0 ~ U(−f_rand·δ, +f_rand·δ)``（每条数据重新抽），``A = f_drift·δ``。
        β 在**每个控制周期内视为常数**（漂移率 ≲1e-3 rad/s，10 ms 内的变化 <1e-5 rad，
        远小于 δ），这样边界不随时间瞬变、事件检测简单且准确。

    为什么要有这个环境（用户要求）: 它把"背隙建模"逼到**最不利**的情形 ——
      · 死区内零刚度 ⇒ 损失对 δ/k/c 的梯度在死区内**几乎为零**（这正是模型里 γ 的用武之地）；
      · 接触完全刚性 ⇒ 真实 k = ∞，而模型只能用有限 k 去近似（`k·δ/2 ≫ τ_max` 时才"看起来刚性"）；
      · β 每条数据随机 + 漂移 ⇒ **单个全局 β 不可能对**，必须靠估计器的在线值（`backlash_center`），
        离线拟合只能拟合 δ/k/c/γ/电机侧那部分。

    接口与 ``PlanarYawPlant`` 完全一致（``q``/``qd``/``exo``/``step``/``new_segment``）。
    """

    def __init__(self, int_step: float = SIM_INT_STEP, tilt_deg: float = 0.0,
                 gravity_a=None, beta0: float = 0.0, beta_drift: float = 0.0,
                 beta_period: float = 30.0, **overrides):
        super().__init__(int_step=int_step, tilt_deg=tilt_deg, gravity_a=gravity_a,
                         **overrides)
        # 这个环境里 k/c/γ 无效（保留字段只为与 PlanarYawPlant 同接口/元数据）
        self.rigid = True
        self.t = 0.0                      # 仿真时间（β 漂移用）
        self.beta0 = float(beta0)
        self.beta_drift = float(beta_drift)      # 漂移幅值 A
        self.beta_period = max(1e-3, float(beta_period))
        self.beta = self._beta(0.0)              # 当前控制周期内保持常数
        self.contact = 0                          # 0 = 自由(死区内), +1/-1 = 贴在一侧
        self.n_impact = 0
        self.n_release = 0

    # ── β(t) 与接触半宽 ──
    def _beta(self, t: float) -> float:
        if self.beta_drift <= 0.0:
            return self.beta0
        return self.beta0 + self.beta_drift * math.sin(TWO_PI * t / self.beta_period)

    def _half(self) -> float:
        return 0.5 * self.p["backlash_delta"]

    def new_segment(self, rng, index: int = 0) -> None:
        """每条数据开始时重新抽 β0（用户要求: 背隙中心位置**每条数据随机**）。"""
        frac = float(self.p.get("beta_random_frac", 0.0))
        self.beta0 = (float(rng.uniform(-frac, frac)) * self.p["backlash_delta"]
                      if frac > 0.0 else self.beta0)
        self.t = 0.0
        self.beta = self._beta(0.0)
        # 抽完 β0 后把状态放到死区内（否则可能一上来就"穿模"）
        d = self.q[0] - self.q[1] - self.beta
        h = self._half()
        if abs(d) > h:
            self.q[1] = self.q[0] - self.beta
            self.contact = 0

    # ── 两种模式的加速度 ──
    def _open_accel(self, q, qd, u):
        """自由段: τ_t ≡ 0（死区内完全自由，连阻尼都没有）。"""
        p = self.p
        M11, M12, h0, h1 = self._h(q, qd)
        hM = self._fric(qd[0], p["fcMotor"], p["fvMotor"]) + p["tau_offset_motor"]
        det = M11 * p["Js"] - M12 * M12
        if abs(det) <= 1e-12 or abs(p["Jmotor"]) <= 1e-12:
            return 0.0, 0.0, 0.0
        r1 = u[1] - h0
        r2 = u[2] - h1
        return ((u[0] - hM) / p["Jmotor"],
                (p["Js"] * r1 - M12 * r2) / det,
                (-M12 * r1 + M11 * r2) / det)

    def _closed_accel(self, q, qd, u):
        """刚性锁定段: Δ_raw ≡ β + s·h（⇒ θ̈_motor = θ̈_platform），电机与云台合并。

        返回 ``(θ̈_common, θ̈_small, τ_t)``；``τ_t`` 是维持该约束所需的接触力矩（电机行反解）。
        """
        p = self.p
        M11, M12, h0, h1 = self._h(q, qd)
        hM = self._fric(qd[0], p["fcMotor"], p["fvMotor"]) + p["tau_offset_motor"]
        M11c = M11 + p["Jmotor"]
        h0c = h0 + hM
        det = M11c * p["Js"] - M12 * M12
        if abs(det) <= 1e-12:
            return 0.0, 0.0, 0.0
        r1 = u[0] - h0c
        r2 = u[2] - h1
        qdd = (p["Js"] * r1 - M12 * r2) / det
        qdds = (-M12 * r1 + M11c * r2) / det
        tt = u[0] - hM - p["Jmotor"] * qdd     # 电机行反解 ⇒ 约束力
        return qdd, qdds, tt

    def _rk4_open(self, hh, u, q0, qd0):
        k1 = self._open_accel(q0, qd0, u)
        q2 = [q0[i] + 0.5 * hh * qd0[i] for i in range(3)]
        qd2 = [qd0[i] + 0.5 * hh * k1[i] for i in range(3)]
        k2 = self._open_accel(q2, qd2, u)
        q3 = [q0[i] + 0.5 * hh * qd2[i] for i in range(3)]
        qd3 = [qd0[i] + 0.5 * hh * k2[i] for i in range(3)]
        k3 = self._open_accel(q3, qd3, u)
        q4 = [q0[i] + hh * qd3[i] for i in range(3)]
        qd4 = [qd0[i] + hh * k3[i] for i in range(3)]
        k4 = self._open_accel(q4, qd4, u)
        h6 = hh / 6.0
        qn = [q0[i] + h6 * (qd0[i] + 2.0 * qd2[i] + 2.0 * qd3[i] + qd4[i]) for i in range(3)]
        qdn = [qd0[i] + h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(3)]
        return qn, qdn

    def _impact(self, s: int) -> None:
        """完全非弹性冲击: 沿约束方向 (1,−1,0) 施加脉冲，使冲击后 Δ̇ = 0（广义动量守恒）。"""
        p = self.p
        q, qd = self.q, self.qd
        M11, M12, _h0, _h1 = self._h(q, qd)
        Js = p["Js"]
        det = M11 * Js - M12 * M12
        inv_bb = Js / det
        inv_sb = -M12 / det
        jmj = 1.0 / p["Jmotor"] + inv_bb
        lam = -(qd[0] - qd[1]) / jmj
        qd[0] += lam / p["Jmotor"]
        qd[1] += -inv_bb * lam
        qd[2] += -inv_sb * lam
        qd[1] = qd[0]                       # 数值上强制相对速度 = 0
        # 位置夹到接触面（消除插值残差）
        q[1] = q[0] - self.beta - s * self._half()
        self.contact = s
        self.n_impact += 1

    def _free_advance(self, hh, u) -> float:
        """自由段推进 ``hh``；若中途撞到边界则只走到边界 + 冲击 + 转闭锁。

        返回**尚未使用**的时间（用于让调用方把剩下的一小段按新接触状态走完）。
        """
        q, qd = self.q, self.qd
        h = self._half()
        qn, qdn = self._rk4_open(hh, u, q, qd)
        d_new = qn[0] - qn[1] - self.beta
        if abs(d_new) <= h:
            self.q, self.qd = qn, qdn
            return 0.0
        # 二分定位穿越时刻（始终从本步起点积分 ⇒ 状态可复现）
        lo, hi = 0.0, hh
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            qm, qdm = self._rk4_open(mid, u, q, qd)
            if abs(qm[0] - qm[1] - self.beta) <= h:
                lo = mid
            else:
                hi = mid
        qs, qds = self._rk4_open(lo, u, q, qd)
        self.q, self.qd = qs, qds
        s = 1 if (self.q[0] - self.q[1] - self.beta) > 0.0 else -1
        self._impact(s)
        return hh - lo

    def _closed_advance(self, hh, u) -> bool:
        """刚性锁定推进 ``hh``。返回 False = 约束释放（本步转为自由段，调用方重跑）。"""
        p = self.p
        q, qd = self.q, self.qd
        s = self.contact
        # 先判约束是否还需要（贴 +侧需要把电机往回推 ⇒ τ_t > 0）
        _a, _b, tt = self._closed_accel(q, qd, u)
        if s * tt <= 0.0:
            self.contact = 0
            self.n_release += 1
            return False

        # 2-DOF 刚性锁定积分: 状态 (θ_平台, θ_小yaw)，θ_电机 = θ_平台（+常数间隙）
        #   质量阵 [[Jmotor + M11, M12], [M12, Js]]，h_b ← h_b + 电机摩擦（τ_t 变成内力）
        def acc2(xb, xs, vb, vs):
            qq = [xb, xb, xs]              # θ_motor = θ_platform（锁定）
            qdd_ = [vb, vb, vs]
            Ma, Mb, ha, hb = self._h(qq, qdd_)
            hm = self._fric(vb, p["fcMotor"], p["fvMotor"]) + p["tau_offset_motor"]
            A, B, C = Ma + p["Jmotor"], Mb, p["Js"]
            d = A * C - B * B
            r1 = u[0] - (ha + hm)
            r2 = u[2] - hb
            return (C * r1 - B * r2) / d, (-B * r1 + A * r2) / d

        xb, xs, vb, vs = q[1], q[2], qd[1], qd[2]
        k1b, k1s = acc2(xb, xs, vb, vs)
        k2b, k2s = acc2(xb + 0.5 * hh * vb, xs + 0.5 * hh * vs,
                        vb + 0.5 * hh * k1b, vs + 0.5 * hh * k1s)
        k3b, k3s = acc2(xb + 0.5 * hh * (vb + 0.5 * hh * k1b),
                        xs + 0.5 * hh * (vs + 0.5 * hh * k1s),
                        vb + 0.5 * hh * k2b, vs + 0.5 * hh * k2s)
        k4b, k4s = acc2(xb + hh * (vb + 0.5 * hh * k2b), xs + hh * (vs + 0.5 * hh * k2s),
                        vb + hh * k3b, vs + hh * k3s)
        h6 = hh / 6.0
        xb_n = xb + h6 * (vb + 2.0 * (vb + 0.5 * hh * k1b) + 2.0 * (vb + 0.5 * hh * k2b)
                          + (vb + hh * k3b))
        vb_n = vb + h6 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
        xs_n = xs + h6 * (vs + 2.0 * (vs + 0.5 * hh * k1s) + 2.0 * (vs + 0.5 * hh * k2s)
                          + (vs + hh * k3s))
        vs_n = vs + h6 * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
        self.q = [xb_n + self.beta + s * self._half(), xb_n, xs_n]
        self.qd = [vb_n, vb_n, vs_n]
        return True

    def step(self, tau, dt: float) -> None:
        u = (float(tau[0]), 0.0, float(tau[1]))
        n = max(1, int(round(dt / self.int_step)))
        hh = dt / float(n)
        self.beta = self._beta(self.t)          # β 在一个控制周期内视为常数（漂移很慢）
        self.sync_exo()                         # ★ 倾斜时 A 系重力随平台角旋转
        for _ in range(n):
            remaining = hh
            guard = 0
            while remaining > 1e-12 and guard < 64:
                guard += 1
                if self.contact != 0:
                    if self._closed_advance(remaining, u):
                        remaining = 0.0
                    # 释放 ⇒ 不消耗时间，下一轮走自由段
                else:
                    remaining = self._free_advance(remaining, u)
            self.t += hh
        self.sync_exo()                          # 同上: 步末再同步一次


class SimRobotLink:
    """无硬件时替代 ``TcbsRobotCommunication``: 内置仿真 + 模拟 MCU2 链路语义。

    仿真对外暴露的字段与实机一致:
      · ``platform_azimuth`` = 真值平台世界方位角（IMU 直测、实时）
      · ``small_joint_angle`` = 真值小 yaw 关节角（编码器，加微小噪声）
      · ``big_joint_angle``   = **延迟补偿**估计 = 最近一次链路新样本值 + 平台角速度×年龄
      · ``chassis_azimuth``   = **延迟补偿**后的底盘方位角（与大 yaw 编码器同一包、同一序号）
      · ``mcu2_seq``/``big_enc_age``/``chassis_imu_age`` = 模拟 ~10 Hz、间隔不规则、
        值被保持的 MCU2 链路（年龄从"上位机首次看到该新样本"起算，与估计器语义一致）
    ★ 底盘 IMU **与大 yaw 编码器在同一条 MCU2 链路上**（同一序号、同一刷新时刻、同样被保持）
      ⇒ 这里也用同一套"延迟 + 值保持"取样，而不是直接给真值。采集规范要求底盘静止，
      所以链路语义在数值上表现为常数；它对 θ_p = ψ_platform − ψ_chassis 的影响
      （以及估计器的一阶延时补偿）由 tests/test_yaw_state_estimator.cpp 的 [1]/[3] 场景验证。
    """

    def __init__(self, rng, int_step: float = SIM_INT_STEP, tilt_deg: float = 0.0,
                 rigid: bool = False, beta_random_frac: float = SIM_BETA_RANDOM_FRAC,
                 beta_drift_frac: float = SIM_BETA_DRIFT_FRAC,
                 beta_drift_period: float = SIM_BETA_DRIFT_PERIOD,
                 no_backlash: bool = False,
                 no_backlash_k: float = SIM_NO_BACKLASH_K):
        """``rigid=True`` ⇒ 用 **RigidBacklashPlant**（接触完全刚性 + 死区完全自由 +
        每条数据随机 β + 微弱漂移）替代平滑背隙被控对象，仅用于采集测试数据。

        ``no_backlash=True`` ⇒ 被控对象**没有背隙**（``backlash_delta = 0``，电机与云台之间
        只剩一条刚度为 ``no_backlash_k`` 的"同步带"弹簧）—— 用于**可辨识性诊断**：
        当前 16 参模型（δ 走 log 参数化、死区近乎不可微）在这种数据上还能不能识别出
        "其实没有背隙"。与 ``rigid`` 互斥（刚性约束 + 零宽死区是退化情形）。
        """
        self.rng = rng
        self.rigid = bool(rigid)
        self.no_backlash = bool(no_backlash)
        if self.rigid and self.no_backlash:
            raise ValueError("--sim-rigid 与 --sim-no-backlash 互斥"
                             "（零宽死区 + 刚性约束是退化情形，无法积分）")
        if self.rigid:
            self.plant = RigidBacklashPlant(
                int_step=int_step, tilt_deg=tilt_deg,
                beta_drift=float(beta_drift_frac) * 0.0873,
                beta_period=float(beta_drift_period),
                beta_random_frac=float(beta_random_frac))
        elif self.no_backlash:
            self.plant = PlanarYawPlant(int_step=int_step, tilt_deg=tilt_deg,
                                        backlash_delta=0.0,
                                        backlash_k=float(no_backlash_k))
        else:
            self.plant = PlanarYawPlant(int_step=int_step, tilt_deg=tilt_deg)
        self.t = 0.0                 # 仿真时钟（每个控制周期 +DT）
        self.frames = 0              # 已下发的帧数
        self._hist = deque(maxlen=128)
        self._meas = {"q0": 0.0, "t": 0.0, "seq": 0, "psi_c": SIM_CHASSIS_AZIMUTH, "w_c": 0.0}
        self._since = 1
        self._next = 1               # 首帧即视为一次新样本
        # ★ 背隙中心 β 的**在线**估计（与 C++ YawStateEstimator 同一套带遗忘滑动 min/max）
        self._bl_min = 0.0
        self._bl_max = 0.0
        self._bl_seen = False
        self._bl_t = -1.0

    def begin_segment(self, index: int = 0) -> None:
        """每段开始时调用: 刚性环境重新抽 β0（"每条数据的背隙中心随机"）。

        ★ β 的**在线估计器状态不重置**（只重抽 β0）—— 运行期的估计器是连续跑的，
        不会每段清零；靠它自己的遗忘（τ=3 s）+ 段前这几秒的到位过程把新的两侧极值
        采到，采样开始时已经收敛。第一段会有一次预热，与实机一致。
        """
        if self.rigid:
            self.plant.new_segment(self.rng, index)

    def wait_ready(self, timeout_s: float = 10.0) -> bool:
        return True

    def _beta_online(self, draw: float, fresh: bool = True) -> float:
        """β 的在线估计（= C++ `YawStateEstimator::estimate()` 里那段滑动 min/max）。

        ★ 只在**刚刷新的样本**上更新极值（``fresh``）: `theta_big_motor` 在两次样本之间是
        按角速度外推的，年龄越大外推误差越大（½α·age²），拿它撑极值会把 Δ 的极差撑开、
        把 β 中心带偏。C++ 侧同一个门限 (`kBacklashFreshS = 20 ms`)。
        """
        if not fresh:
            return 0.5 * (self._bl_max + self._bl_min)
        tau = SIM_BETA_TAU_S
        if not self._bl_seen:
            self._bl_min = self._bl_max = draw
            self._bl_seen = True
            self._bl_t = self.t
            return 0.5 * (self._bl_max + self._bl_min)
        dtb = self.t - self._bl_t
        if dtb > 1e-6:
            a = 1.0 - math.exp(-dtb / tau)
            self._bl_min = draw if draw < self._bl_min else self._bl_min + a * (draw - self._bl_min)
            self._bl_max = draw if draw > self._bl_max else self._bl_max + a * (draw - self._bl_max)
            self._bl_t = self.t
        return 0.5 * (self._bl_max + self._bl_min)

    def _lookup(self, t_target: float) -> float:
        """取 (t_target − 传输时延) 时刻的真值（模拟链路里的采样时刻）。"""
        best = self._hist[0][1] if self._hist else 0.0
        for ts, q0, _qd0 in self._hist:
            if ts <= t_target:
                best = q0
            else:
                break
        return float(best)

    def _lookup_rate(self, t_target: float) -> float:
        """同 ``_lookup``，但取电机侧**角速度**真值（`_hist` 的第三项）。"""
        best = self._hist[0][2] if self._hist else 0.0
        for ts, _q0, qd0 in self._hist:
            if ts <= t_target:
                best = qd0
            else:
                break
        return float(best)

    def read(self) -> RobotSample:
        q, qd = self.plant.q, self.plant.qd
        age = max(0.0, self.t - self._meas["t"])
        psi_c_true, w_c_true = SIM_CHASSIS_AZIMUTH, 0.0     # 底盘静止（采集规范）
        # ★ 电机侧（q[0]）走 MCU2 链路: 延迟 + 值保持 + 传输时延
        #   云台侧（q[1]）由"头上 IMU 反解"给出 ⇒ **无延迟、实时**
        motor_rate = float(self._meas.get("qd0", 0.0))
        big_est = self._meas["q0"] + motor_rate * age      # 估计器的延时补偿（一阶）
        # ★ 底盘侧同样走 MCU2 链路（与大 yaw 同一包、同一序号）:
        #   原始被保持值记进 mcu_chassis_imu_*；估计器输出 = 一阶延时补偿后的底盘方位角
        #   （补偿用被保持的底盘角速度，与估计器实现一致 ⇒ 真值是一阶准确解）
        psi_c_held = float(self._meas.get("psi_c", SIM_CHASSIS_AZIMUTH))
        w_c_held = float(self._meas.get("w_c", 0.0))
        # ★ 背隙中心 β: 用**在线估计器**（与运行期同一套规则），输入 = Δ_raw = θ_motor − θ_platform
        # "新鲜样本" = 首次看到该新样本后的 20 ms 内（与 C++ kBacklashFreshS 一致）
        beta_est = self._beta_online(big_est - q[1], fresh=(age <= 0.02))
        beta_true = float(self.plant.beta) if self.rigid else 0.0
        return RobotSample(
            big_joint_angle=big_est, big_joint_angle_meas=float(self._meas["q0"]),
            big_joint_rate=qd[1],                          # 云台侧（旧语义）
            theta_big_motor=big_est,
            dtheta_big_motor=motor_rate * float(self.rng.normal(1.0, 0.01)),
            theta_big_platform=q[1],                       # ★ 云台侧真值
            dtheta_big_platform=qd[1],
            platform_azimuth=q[1] + psi_c_true,            # 平台世界方位角 = 关节角 + 底盘方位角
            platform_rate=qd[1],
            small_joint_angle=q[2] + float(self.rng.normal(0.0, SIM_ENC_NOISE)),
            small_joint_rate=qd[2],
            small_joint_angle_est=q[2], small_joint_rate_est=qd[2],
            chassis_yaw=psi_c_true, chassis_omega=w_c_held,   # 估计器输出（方位角已补偿）
            chassis_azimuth=psi_c_true, chassis_yaw_rate=w_c_held,
            big_enc_age=age, big_sample_interval=float(self._next) * DT,
            big_enc_innovation=0.0, chassis_imu_age=age,
            mcu2_seq=int(self._meas["seq"]),
            mcu_temp_big=30, mcu_temp_small=30,
            # 仿真里把估计器输出直接当"收到的 MCU 量"填（用于验证记录列非空）；
            # 底盘那一对是**原始被保持值**（未补偿），供下游复核链路语义
            mcu_bullet_velocity=0.0, mcu_pitch_angle=0.0,
            mcu_yaw_big_angle=float(self._meas["q0"]), mcu_yaw_big_omega=motor_rate,
            mcu_yaw_small_angle=q[2], mcu_yaw_small_omega=qd[2],
            mcu_chassis_imu_yaw=psi_c_held, mcu_chassis_imu_omega=w_c_held,
            mcu_mark=0, mcu_color=0, mcu_auto_aim_switch=1,
            imu_gx=0.0, imu_gy=0.0, imu_gz=qd[1] + w_c_true,
            imu_ax=0.0, imu_ay=0.0, imu_az=9.81,
            imu_euler_yaw=q[1] + psi_c_true, imu_euler_pitch=0.0, imu_euler_roll=0.0,
            imu_dt_one_tenth_ms=100,
            est_valid=1, mcu_valid=1, imu_valid=1,
            gravity_ax=float(self.plant.exo[0]), gravity_ay=float(self.plant.exo[1]),
            gravity_az=-9.81,
            # 估计器的 base_omega_z = 被保持的底盘角速度（角速度不做延时候补偿）
            base_omega_x=0.0, base_omega_y=0.0, base_omega_z=w_c_held,
            los_azimuth=q[1] + psi_c_true, los_elevation=0.0,
            pitch_joint_angle=0.0, pitch_joint_rate=0.0, pitch_acc=0.0,
            backlash_center=beta_est, backlash_beta_true=beta_true,
            # ★ 仿真真值状态（仅用于诊断/上限对照；实机这 6 列恒 0）
            theta_true_motor=q[0], theta_true_platform=q[1], theta_true_small=q[2],
            dtheta_true_motor=qd[0], dtheta_true_platform=qd[1], dtheta_true_small=qd[2])

    def send(self, tau_big, tau_small, big_joint_target, small_joint_target) -> bool:
        self.frames += 1
        self._hist.append((self.t, self.plant.q[0], self.plant.qd[0]))
        t_new = self.t + DT
        # ── 模拟 MCU2 链路: ~10 Hz、间隔不规则（80~120 ms）、两次之间值被保持 ──
        self._since += 1
        if self._since >= self._next:
            self._since = 0
            self._next = int(self.rng.integers(8, 13))
            t_smp = t_new - SIM_TRANSPORT_DELAY
            self._meas = {"q0": self._lookup(t_smp),
                          "qd0": self._lookup_rate(t_smp),
                          # ★ 底盘 IMU 与编码器**同一包** ⇒ 同一采样时刻、同一序号、同样被保持
                          "psi_c": SIM_CHASSIS_AZIMUTH, "w_c": 0.0,
                          "t": t_new,
                          "seq": (self._meas["seq"] + 1) % 256}
        self.t = t_new
        # 被控对象推进一个控制周期（力矩零阶保持，内部 0.05 ms 细分）
        self.plant.step([float(tau_big), float(tau_small)], DT)
        return True

    def close(self) -> None:
        pass


# ============================================================================
# 控制相位驱动（到位 / 采样 / 回中 共用同一段代码）
# ============================================================================
def make_row(st: "RobotSample", t: float, tau_big: float, tau_small: float, axis: int,
             held_target: float, tgt_big: float, tgt_small: float) -> dict:
    """构造一条记录（覆盖全部列）。

    * 链路侧（收到的 MCU/IMU、估计器输出）直接取自 ``st``；
    * 下发侧按**本拍实际发出**的内容填（仅力矩模式、目标速度 0）；
    * 前置历史列 ``theta_big`` = **电机侧**角度、``dtheta_big`` = **云台侧**角速度
      （旧语义未变）；新列 ``*_motor`` / ``*_platform`` 是显式分离后的值。
    """
    row = {"t": t, "tau_big": tau_big, "tau_small": tau_small,
           "axis": axis, "held_target": held_target,
           "target_big": tgt_big, "target_small": tgt_small}
    colset = set(CSV_HEADER)
    for name in _SAMPLE_FIELDS:
        if name in colset:                 # 同名列直接搬
            row[name] = getattr(st, name)
    # 需要改名的两处: gravity_ax/ay/az → gravity_a_x/y/z
    row["theta_big_motor_meas"] = st.big_joint_angle_meas
    row["small_joint_angle_est"] = st.small_joint_angle
    row["small_joint_rate_est"] = st.small_joint_rate
    row["chassis_azimuth"] = st.chassis_yaw
    row["chassis_yaw_rate"] = st.chassis_omega
    row["gravity_a_x"] = st.gravity_ax
    row["gravity_a_y"] = st.gravity_ay
    row["gravity_a_z"] = st.gravity_az
    row["theta_big"] = st.big_joint_angle          # 电机侧（旧语义）
    row["dtheta_big"] = st.dtheta_big_platform     # 云台侧（旧语义）
    row["theta_small"] = st.small_joint_angle
    row["dtheta_small"] = st.small_joint_rate
    row["mcu2_seq"] = st.mcu2_seq
    row["gravity_ax"] = st.gravity_ax
    row["gravity_ay"] = st.gravity_ay
    row["backlash_center"] = st.backlash_center
    row["backlash_beta_true"] = st.backlash_beta_true
    row["theta_true_motor"] = st.theta_true_motor
    row["theta_true_platform"] = st.theta_true_platform
    row["theta_true_small"] = st.theta_true_small
    row["dtheta_true_motor"] = st.dtheta_true_motor
    row["dtheta_true_platform"] = st.dtheta_true_platform
    row["dtheta_true_small"] = st.dtheta_true_small
    row["tx_auto_aim_enable"] = AUTO_AIM_ENABLE
    row["tx_fire"] = 0
    row["tx_pitch_target_angle"] = PITCH_TARGET_ANGLE
    row["tx_yaw_big_mode"] = YAW_MODE_TORQUE_ONLY
    row["tx_yaw_big_target_angle"] = tgt_big
    row["tx_yaw_big_target_velocity"] = 0.0
    row["tx_yaw_big_torque"] = tau_big
    row["tx_yaw_small_mode"] = YAW_MODE_TORQUE_ONLY
    row["tx_yaw_small_target_angle"] = tgt_small
    row["tx_yaw_small_target_velocity"] = 0.0
    row["tx_yaw_small_torque"] = tau_small
    return row


def drive_steps(link, ref_big: np.ndarray, ref_small: np.ndarray, pids, limiters,
                max_temp: float, small_guard: bool = True,
                record: SegmentRecord | None = None, axis: int = AXIS_BIG,
                held_target: float = 0.0):
    """按 100 Hz 跑完 ``len(ref_big)`` 个控制周期（绝对时间点忙等）。

    返回 ``(abort_reason, steps_done)``；``abort_reason`` 为 None 表示正常结束。
    每个周期: 读状态 → 安全检查 → 两轴 PID → 各轴力矩限幅 → 仅力矩下发 →（可选）记录。
    """
    n = len(ref_big)
    t0_ns = time.perf_counter_ns()
    for k in range(n):
        busy_wait_until(t0_ns + k * DT_NS)
        st = link.read()

        # ── 安全 1: 小 yaw 硬限位（当前行程对称 ±30°，但判据按 [min,max] 写，非对称也对）──
        if small_guard:
            th_s = float(st.small_joint_angle)
            # ★ 三档（2026-09-21 放宽）: 软限位 ±30°（越界**只告警**）；硬限位 ±35°（触碰中止）
            if th_s > SMALL_ABORT_MAX or th_s < SMALL_ABORT_MIN:
                log(f"  [SAFETY] 小 yaw θ={_deg(th_s):+.1f}° 触及**硬限位** "
                    f"[{_deg(SMALL_ABORT_MIN):+.0f}°, {_deg(SMALL_ABORT_MAX):+.0f}°] "
                    f"→ 中止本段并回中心")
                return "small_limit", k
            if (th_s > SMALL_SOFT_MAX or th_s < SMALL_SOFT_MIN):
                if k % 50 == 0:
                    _m = (SMALL_ABORT_MAX - th_s if th_s > 0 else th_s - SMALL_ABORT_MIN)
                    log(f"  [WARN] 小 yaw θ={_deg(th_s):+.1f}° 已越**软限位** "
                        f"[{_deg(SMALL_SOFT_MIN):+.0f}°, {_deg(SMALL_SOFT_MAX):+.0f}°]，"
                        f"距硬限位还有 {_deg(_m):.1f}°（**不中止**，只是超调余量在变小）")
            elif (th_s > SMALL_SOFT_MAX - SMALL_WARN_MARGIN
                    or th_s < SMALL_SOFT_MIN + SMALL_WARN_MARGIN) and k % 100 == 0:
                log(f"  [WARN] 小 yaw θ={_deg(th_s):+.1f}° 接近软限位"
                    f"（距 {_deg(SMALL_SOFT_MAX if th_s > 0 else SMALL_SOFT_MIN):+.0f}° "
                    f"余量 < {_deg(SMALL_WARN_MARGIN):.0f}°）")
        # ── 安全 2: 电机温度 ──
        if max(st.mcu_temp_big, st.mcu_temp_small) >= max_temp:
            return "overheat", k

        tgt_big = float(ref_big[k])
        tgt_small = float(ref_small[k])
        # 误差: 大 yaw 用平台方位角（多圈，wrap 到 ±π）；小 yaw 相对角误差本身 ≪π，wrap 无副作用
        e_big = wrap_pi(tgt_big - st.platform_azimuth)
        e_small = wrap_pi(tgt_small - st.small_joint_angle)
        tau_big = limiters[0].limit(pids[0].update(e_big, DT))
        tau_small = limiters[1].limit(pids[1].update(e_small, DT))
        # 下发给电控的"关节角目标"（mode=0 时电控不使用，仅供电控限位/日志参考）:
        #   大 yaw = 估计关节角 + 平台误差; 小 yaw = 关节相对角目标
        big_joint_target = st.big_joint_angle + e_big
        link.send(tau_big, tau_small, big_joint_target, tgt_small)

        if record is not None:
            record.append(make_row(st, (time.perf_counter_ns() - t0_ns) * 1e-9,
                                   tau_big, tau_small, axis, held_target,
                                   tgt_big, tgt_small))
    return None, n


def run_zero_torque(link, limiters, seconds: float, stop_temp: float | None = None):
    """零力矩保温（过热等待用）: 100 Hz 发零力矩，必要时监测温度。

    返回 ``(ok, cooled)``；``stop_temp`` 给定时，温度降到该值以下提前返回 True。
    """
    n = max(1, int(round(seconds * RATE)))
    t0_ns = time.perf_counter_ns()
    cooled = False
    for k in range(n):
        busy_wait_until(t0_ns + k * DT_NS)
        tb = limiters[0].limit(0.0)
        ts = limiters[1].limit(0.0)
        link.send(tb, ts, 0.0, 0.0)
        if stop_temp is not None and k % 100 == 99:
            st = link.read()
            if k % 1000 == 999:
                log(f"    降温中… temp=({st.mcu_temp_big},{st.mcu_temp_small})℃")
            if max(st.mcu_temp_big, st.mcu_temp_small) < stop_temp:
                cooled = True
                break
    return True, cooled


def cooldown(link, limiters, max_temp: float, reason: str) -> bool:
    """过热保护: 零力矩 + 100 Hz 保温等待降温（规格: 等待或退出）。"""
    target = max_temp - COOL_HYSTERESIS_C
    log(f"  [SAFETY] {reason}: 零力矩降温等待（目标 < {target:.0f}℃, 上限 "
        f"{MAX_COOL_WAIT_S:.0f}s）")
    ok, cooled = run_zero_torque(link, limiters, MAX_COOL_WAIT_S, stop_temp=target)
    if cooled:
        log("    温度已回落，继续采集")
        return True
    log(f"    [ERROR] {MAX_COOL_WAIT_S:.0f}s 内未降到 {target:.0f}℃ 以下")
    return False


def recenter(link, pids, limiters, max_temp: float, seconds: float = RECENTER_SEC) -> None:
    """回中心/守位: 小 yaw 回到**行程中心**（当前行程对称 ⇒ 0°），大 yaw 保持当前平台方位角。

    为什么写"行程中心"而不是硬编码 0: 行程由 `SMALL_TRAVEL_MIN/MAX` 决定，式子按
    `(min+max)/2` 算 ⇒ 以后改成非对称行程（例如 [−20°, +25°] ⇒ +2.5°）会自动跟着走；
    停在中心时到两端的余量相等，这是"段间静置/初始条件"最安全的位置。

    用在小 yaw 触碰行程界限之后、每段结束、以及 `--tilt-rolling` 段间改倾角的等待
    （倾斜后重力会在小 yaw 上产生力矩，"撒手"会让它自己滑到限位，所以这里保持闭环）。
    过程中关闭小 yaw 限位判定（否则刚越限时会被立刻再次中止），力矩仍受限幅保护。
    """
    st = link.read()
    n = max(1, int(round(seconds * RATE)))
    ref_big = np.full(n, float(st.platform_azimuth), dtype=np.float64)
    ref_small = np.full(n, float(SMALL_CENTER_RAD), dtype=np.float64)
    pids[0].reset()
    pids[1].reset()
    drive_steps(link, ref_big, ref_small, pids, limiters, max_temp, small_guard=False)


# ============================================================================
# 保存（★ 默认只写 npz；--save-csv 才同时写 csv）
# ============================================================================
def _unique_paths(out_dir: str, tag: str, segment_index: int, suffix: str = "",
                  save_csv: bool = False):
    base = f"sysid_{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{segment_index:02d}{suffix}"
    npz_path = os.path.join(out_dir, base + ".npz")
    csv_path = os.path.join(out_dir, base + ".csv") if save_csv else None
    k = 1
    while os.path.exists(npz_path) or (csv_path is not None and os.path.exists(csv_path)):
        npz_path = os.path.join(out_dir, f"{base}_{k}.npz")
        csv_path = os.path.join(out_dir, f"{base}_{k}.csv") if save_csv else None
        k += 1
    return npz_path, csv_path


def save_segment(rec: SegmentRecord, plan: SegmentPlan, out_dir: str,
                 tag_override: str | None, segment_index: int, suffix: str = "",
                 save_csv: bool = False):
    """写 npz（**默认只写这一份**；``save_csv=True`` 时再写一份同名 csv）。

    npz 里的 ``axis``/``held_target`` 是**标量**（段内恒定），CSV 里它们是每行一列（同值）；
    其余列一一对应。详见 docs/sysid_data.md。

    ★ 为什么默认不写 CSV: npz 的列是 CSV 的**超集**（多了打包的 ``theta_true``/``dtheta_true``、
    仿真真值 β 列，且 csv 只保留了 6 位小数），辨识脚本本来就只按列名从 npz 读；
    CSV 的体积却与 npz 同量级 ⇒ 实测一个 360 段的数据目录，去掉 csv 后体积减半。

    CSV（``--save-csv``）= 10 个固定列 + 末尾两列 ``gravity_ax,gravity_ay``（重力 A 系平面分量，
    水平静置时全 0）——追加在最后，保证按列名取列的读取器不失效。
    """
    tag = tag_override or AXIS_NAME[plan.axis]
    npz_path, csv_path = _unique_paths(out_dir, tag, segment_index, suffix, save_csv)
    axis = int(plan.axis)

    def arr(name):
        return np.asarray(rec.col(name), dtype=np.float64)

    np.savez(
        npz_path,
        # ── 与 CSV 逐列对应的时段数组 ──
        t=arr("t"), theta_big=arr("theta_big"), theta_small=arr("theta_small"),
        dtheta_big=arr("dtheta_big"), dtheta_small=arr("dtheta_small"),
        tau_big=arr("tau_big"), tau_small=arr("tau_small"),
        mcu2_seq=np.asarray(rec.col("mcu2_seq"), dtype=np.float64),
        # ── 底盘/诊断（用户: 可以记录但拟合不用）──
        chassis_yaw=arr("chassis_azimuth"), chassis_yaw_rate=arr("chassis_yaw_rate"),
        big_enc_age=arr("big_enc_age"),
        chassis_imu_age=arr("chassis_imu_age"),
        base_omega_x=arr("base_omega_x"), base_omega_y=arr("base_omega_y"),
        base_omega_z=arr("base_omega_z"),
        # ── ★ 大 yaw 电机侧 / 云台侧 显式分离（3-DOF 背隙辨识的核心列）──
        #   旧列 `theta_big` = 电机侧（延时补偿后）；这里再给**原始滞后**值与云台侧值
        theta_big_motor=arr("theta_big_motor"),
        theta_big_motor_meas=arr("theta_big_motor_meas"),
        theta_big_platform=arr("theta_big_platform"),
        dtheta_big_motor=arr("dtheta_big_motor"),
        dtheta_big_platform=arr("dtheta_big_platform"),
        small_joint_angle_est=arr("small_joint_angle_est"),
        small_joint_rate_est=arr("small_joint_rate_est"),
        # ── ★ 背隙中心 β（在线值 / 仿真真值）──
        backlash_center=arr("backlash_center"),
        backlash_beta_true=arr("backlash_beta_true"),
        # ── ★ 仿真真值状态（[T,3] 打包；实机全 0）──
        theta_true=np.stack([arr("theta_true_motor"), arr("theta_true_platform"),
                             arr("theta_true_small")], axis=-1),
        dtheta_true=np.stack([arr("dtheta_true_motor"), arr("dtheta_true_platform"),
                              arr("dtheta_true_small")], axis=-1),
        # ── 重力 A 系平面分量（m/s²）: 水平静置全 0；倾斜静置非 0 ⇒ 下游启用重力项 ──
        gravity_ax=arr("gravity_ax"), gravity_ay=arr("gravity_ay"),
        # ── 下发的参考（便于复核/画图）──
        target_big=arr("target_big"), target_small=arr("target_small"),
        # ── 标量元数据（规格要求）──
        axis=np.int32(axis),
        dt=np.float64(DT),
        held_target=np.float64(plan.held_target),
        kp=np.float64(plan.kp), ki=np.float64(plan.ki), kd=np.float64(plan.kd),
        pid_deadband=np.float64(plan.pid_deadband),
        # ── 附加元数据（便于溯源；不影响拟合）──
        n_points=np.int32(len(rec)),
        rate=np.float64(RATE),
        pid_out_limit=np.float64(PID_OUT_MAX),
        max_torque_delta=np.float64(MAX_TORQUE_DELTA),
        tag=np.str_(tag),
        source_file=np.str_(plan.src_file),
        source_start=np.int32(plan.src_start),
        source_scale=np.float64(plan.src_scale),
        ref_center=np.float64(plan.ref_center),
        ref_amp=np.float64(plan.ref_amp),
        # ── 静态倾斜段标记（--tilted / --tilt-rolling）──
        tilted=np.int32(plan.tilted),
        tilt_slot=np.int32(plan.tilt_slot),
        # ── ★ 无背隙仿真标记（--sim-no-backlash；诊断用，实机恒 0）──
        no_backlash=np.int32(plan.no_backlash),
        # ── ★ 被控对象重心真值（kg·m；实机恒 0 = 未知，只给诊断/报告用）──
        px_true=np.float64(plan.px_true), py_true=np.float64(plan.py_true),
        pbx_true=np.float64(plan.pbx_true), pby_true=np.float64(plan.pby_true),
        # ── held 大 yaw 方位角分层（--held-big-stratified）──
        held_big_stratified=np.int32(1 if plan.held_strat_count else 0),
        held_strat_index=np.int32(plan.held_strat_index),
        held_strat_count=np.int32(plan.held_strat_count),
        held_strat_offset=np.float64(plan.held_strat_offset),
        # ── 小 yaw 行程（**非对称**）: 三档数值都写进去，便于下游核对待遇 ──
        small_travel_min=np.float64(SMALL_TRAVEL_MIN),
        small_travel_max=np.float64(SMALL_TRAVEL_MAX),
        small_soft_min=np.float64(SMALL_SOFT_MIN),
        small_soft_max=np.float64(SMALL_SOFT_MAX),
        small_env_min=np.float64(SMALL_ENV_MIN),
        small_env_max=np.float64(SMALL_ENV_MAX),
        small_center=np.float64(SMALL_CENTER_RAD))

    _INT_COLS = {"axis", "mcu_mark", "mcu_color", "mcu_auto_aim_switch",
                 "mcu_temp_big", "mcu_temp_small", "tx_auto_aim_enable", "tx_fire",
                 "tx_yaw_big_mode", "tx_yaw_small_mode",
                 "est_valid", "mcu_valid", "imu_valid", "mcu2_seq",
                 "mcu_dt_one_tenth_ms", "imu_dt_one_tenth_ms"}
    if csv_path is None:
        return npz_path, None          # ★ 默认路径: 只落 npz，不写 csv
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for i in range(len(rec)):
            writer.writerow([
                (f"{float(rec.col(n)[i]):.4f}" if n == "t"
                 else (str(int(rec.col(n)[i])) if n in _INT_COLS
                       else f"{float(rec.col(n)[i]):.6f}"))
                for n in CSV_HEADER])
    return npz_path, csv_path


# ============================================================================
# 单段采集
# ============================================================================
def tilt_banner(segment_index: int, slot: int, rolling: bool) -> None:
    """静态倾斜段的显著提示（每段一次）。

    **只提示，不做任何补偿**: 脚本不会去估计/抵消倾角，也不改变激励方式 ——
    模型的重力项 `G_s = Qx·gy − Qy·gx` 只在数据里"本来就非零"时才有信息量，
    由下游辨识工具决定是否启用（见 docs/sysid_data.md §6.4）。
    """
    name = TILT_SLOTS[slot % len(TILT_SLOTS)]
    log("  " + "!" * 74)
    log(f"  !! 静态倾斜段（--tilted）  第 {segment_index + 1} 段")
    if rolling:
        log(f"  !! 本段要求的静置姿态: 【{name}】")
    else:
        log(f"  !! 请把底盘以固定倾角静置（例如垫起一侧车轮，约 10°）: 建议【{name}】")
    log("  !! 采集期间底盘**保持静止不动**（倾斜 ≠ 底盘运动: 模型里 base_omega/")
    log("  !! base_alpha 仍取 0）; 脚本不做事后倾角补偿，重力分量原样记录进数据。")
    log("  " + "!" * 74)


def collect_segment(link, rng, targets, planners, pids, limiters, args,
                    segment_index: int, samples: int):
    """采集一段。返回 ``"saved"`` / ``"retry"``（过热，未保存）/ ``"abort"``（中止，未保存）。"""
    axis = AXIS_BIG if segment_index % 2 == 0 else AXIS_SMALL
    drive_axis = AXIS_NAME[axis]
    hold_axis = AXIS_NAME[AXIS_SMALL if axis == AXIS_BIG else AXIS_BIG]

    link.begin_segment(segment_index)     # ★ 刚性环境: 每段重新抽 β0
    st = link.read()
    log(f"\n=== 段 {segment_index + 1} === driven={drive_axis} 轴 / held={hold_axis} 轴"
        f"  温度=({st.mcu_temp_big},{st.mcu_temp_small})℃")
    if max(st.mcu_temp_big, st.mcu_temp_small) >= args.max_temp:
        if not cooldown(link, limiters, args.max_temp, "段前温度过高"):
            return "abort"
        st = link.read()

    # ── 静态倾斜段: 每段前显著提示（默认关闭，开启时元数据记 tilted=1）──
    tilt_slot = -1
    if args.tilted:
        tilt_slot = segment_index % len(TILT_SLOTS)
        tilt_banner(segment_index, tilt_slot, bool(args.tilt_rolling))
        if args.tilt_rolling and segment_index > 0:
            # 段间给操作者留出改倾角的时间。期间**保持闭环守位**（不是撒手零力矩）:
            # 倾斜后重力会在小 yaw 上产生力矩，撒手会让它自己滑到限位。
            log(f"  请在 {TILT_CHANGE_SEC:.0f}s 内把底盘调到该倾角（两轴保持闭环守位）…")
            recenter(link, pids, limiters, args.max_temp, seconds=TILT_CHANGE_SEC)
            st = link.read()

    # ── 分层 held 大 yaw 方位角（可选）: 均匀铺满 ±π, 让固定倾角下的 g_A 方向覆盖更均匀 ──
    held_override = None
    strat_k = -1
    if getattr(args, "held_big_stratified", False) and axis == AXIS_SMALL:
        n_small = max(1, args.segments // 2)
        strat_k = segment_index // 2
        if getattr(args, "_held_base", None) is None:
            args._held_base = float(st.platform_azimuth)
        held_override = args._held_base + (-math.pi + (strat_k + 0.5)
                                           * 2.0 * math.pi / n_small)
        log(f"  held 大 yaw 方位角（分层 {strat_k + 1}/{n_small}）= "
            f"{held_override:+.3f} rad ({_deg(held_override):+.1f}°)"
            f"（基准 {args._held_base:+.3f} rad, 偏移 "
            f"{_deg(held_override - args._held_base):+.1f}°）")

    plan = build_segment_plan(rng, targets, axis, st, planners, samples,
                              held_big_override=held_override)
    plan.tilted = 1 if (args.tilted or abs(getattr(args, "sim_tilt_deg", 0.0)) > 1e-12) else 0
    plan.no_backlash = 1 if getattr(args, "sim_no_backlash", False) else 0
    plan.kp, plan.ki, plan.kd = float(args.kp), float(args.ki), float(args.kd)
    plan.pid_deadband = math.radians(max(0.0, float(args.pid_deadband_deg)))
    # 被控对象的重心真值（实机未知 ⇒ 0；dry-run 从 plant 读）
    _pl = getattr(link, "plant", None)
    if _pl is not None:
        plan.px_true = float(_pl.p["Px"]); plan.py_true = float(_pl.p["Py"])
        plan.pbx_true = float(_pl.p["Pbx"]); plan.pby_true = float(_pl.p["Pby"])
    plan.tilt_slot = tilt_slot
    plan.held_strat_index = strat_k
    plan.held_strat_count = (max(1, args.segments // 2)
                             if getattr(args, "held_big_stratified", False) else 0)
    plan.held_strat_offset = (held_override - args._held_base) if held_override is not None else 0.0
    log(f"  参考来源: {plan.src_file}[{plan.src_start}:{plan.src_start + samples}] "
        f"随机缩放={plan.src_scale:.2f}")
    if axis == AXIS_BIG:
        log(f"  driven 大 yaw: 中心={plan.ref_center:+.3f} rad "
            f"半幅=±{plan.ref_amp:.3f} rad(±{_deg(plan.ref_amp):.1f}°)")
        log(f"  held   小 yaw: 目标={plan.held_target:+.3f} rad "
            f"({_deg(plan.held_target):+.1f}°)  [包络 "
            f"{_deg(SMALL_ENV_MIN):+.0f}°…{_deg(SMALL_ENV_MAX):+.0f}°]")
    else:
        log(f"  driven 小 yaw: 中心={plan.ref_center:+.3f} rad"
            f"({_deg(plan.ref_center):+.1f}°) 半幅=±{plan.ref_amp:.3f} rad"
            f"(±{_deg(plan.ref_amp):.1f}°)")
        log(f"    整条位于参考包络 [{_deg(SMALL_ENV_MIN):+.0f}°, {_deg(SMALL_ENV_MAX):+.0f}°] 内"
            f"（硬限位 [{_deg(SMALL_TRAVEL_MIN):+.0f}°, {_deg(SMALL_TRAVEL_MAX):+.0f}°]，"
            f"两侧各留 {_deg(SMALL_TRACK_MARGIN):.0f}° 跟踪余量; 中心 {_deg(SMALL_CENTER_RAD):+.1f}°）")
        log(f"  held   大 yaw: 目标={plan.held_target:+.3f} rad（现有方位角 ±π 内随机）")

    # ── 到位等待（★ 单段；稳定判据已删除）──
    #   PID 把两轴带到目标并保持 `--settle-sec`（默认 5.0 s），**固定时长、不判据**，
    #   走完就直接采样（原来后面还有「第二段: 连续 --stable-sec 达标」，已按用户要求删除）。
    #   参考仍由轨迹规划器整形（不是阶跃），否则 PID 会饱和过冲把小 yaw 顶到限位。
    settle_n = max(1, int(round(args.settle_sec * RATE)))
    pids[0].reset()
    pids[1].reset()
    # ── ★ 静止保持段也记录（可选, 默认开）──
    #   这一段是"从当前位姿 → 本段目标"的**大角度阶跃**（由轨迹规划器整形），
    #   刚好补上采样轨迹里稀缺的"大幅阶跃"激励 ⇒ 一并落盘、一并参与辨识。
    #   跳过**最开始的第一次**（segment_index == 0）: 那一拍的起始位姿是任意的
    #   （可能是人工摆放/上电瞬态），不是一个有意义的受控阶跃。
    rec_hold = (SegmentRecord()
                if (args.record_hold and segment_index > 0) else None)
    st = link.read()
    ref_big_home = homing_sequence(st.platform_azimuth, float(plan.ref_big[0]),
                                   settle_n, planners["big"])
    ref_small_home = homing_sequence(st.small_joint_angle, float(plan.ref_small[0]),
                                     settle_n, planners["small"])
    reason, _ = drive_steps(link, ref_big_home, ref_small_home, pids, limiters,
                            args.max_temp, record=rec_hold, axis=axis,
                            held_target=plan.held_target)
    if reason == "small_limit":
        recenter(link, pids, limiters, args.max_temp)
        return "abort"
    if reason == "overheat":
        if not cooldown(link, limiters, args.max_temp, "到位阶段温度过高"):
            return "abort"
        return "retry"
    st = link.read()
    log(f"  到位(固定 {args.settle_sec:g}s): err_big={_deg(wrap_pi(plan.ref_big[0] - st.platform_azimuth)):+.2f}°"
        f" err_small={_deg(wrap_pi(plan.ref_small[0] - st.small_joint_angle)):+.2f}°")

    # ── ★ 稳定性判据已删除（用户要求，2026-09-20）──
    #   原来这里是"第二段: 误差/速度连续 `--stable-sec` 全部达标才开采，不满足就一直等"。
    #   现在: 到位段（固定 `--settle-sec`，默认 5 s）走完就**直接采样**，不再判稳定。
    #   · 两次采样的间隔只由 `--settle-sec` 决定（原来是 settle + stable）；
    #   · PID 状态仍然跨相位连续（不 reset），与之前一致；
    #   · 到位段的实际余差仍打印出来供人工看（不再当门限）。
    st = link.read()
    log(f"  ✓ 到位等待结束（固定 {args.settle_sec:g}s，**不判稳定性**）→ 直接开始采样"
        f"（当前 err_big={_deg(wrap_pi(plan.ref_big[0] - st.platform_azimuth)):+.2f}° "
        f"err_small={_deg(wrap_pi(plan.ref_small[0] - st.small_joint_angle)):+.2f}°）")
    if rec_hold is not None and len(rec_hold) > 0:
        h_npz, h_csv = save_segment(rec_hold, plan, args.out, args.tag, segment_index,
                                    suffix=HOLD_SUFFIX, save_csv=args.save_csv)
        log(f"  静止保持段已记录: {h_npz}  ({len(rec_hold)} 行, "
            f"t=0~{rec_hold.col('t')[-1]:.2f}s, 含大角度阶跃)")
        if h_csv:
            log(f"        {h_csv}")
        if args.dry_run:
            pass

    # ── 采样: 300 点 @100 Hz ──
    log(f"  采样 {samples} 点 ({samples * DT:.2f} s @100Hz)…")
    rec = SegmentRecord()
    reason, steps_done = drive_steps(link, plan.ref_big, plan.ref_small, pids, limiters,
                                     args.max_temp, record=rec, axis=axis,
                                     held_target=plan.held_target)

    if reason == "small_limit":
        recenter(link, pids, limiters, args.max_temp)
        log("  [SKIP] 本段因小 yaw 越限中止，数据不保存")
        return "abort"
    if reason == "overheat":
        if not cooldown(link, limiters, args.max_temp, "采样阶段温度过高"):
            return "abort"
        log("  [SKIP] 本段因过热中止，数据不保存（降温后重采）")
        return "retry"
    if steps_done != samples:
        log(f"  [SKIP] 只采到 {steps_done}/{samples} 点，丢弃")
        return "abort"

    # ── 段尾: 主动回中保持（小 yaw → 0，大 yaw 保持当前方位角）──
    #    比"直接零力矩撒手"更安全: 段末被激励轴仍有残余角速度，它通过耦合会把
    #    已撒手的另一轴推着走（仿真实测可漂 20°+），主动闭环可以把它按回去。
    #    * 真正的"零力矩"只在程序退出时发（规格要求），见 safe_shutdown()。
    recenter(link, pids, limiters, args.max_temp)

    npz_path, csv_path = save_segment(rec, plan, args.out, args.tag, segment_index,
                                      save_csv=args.save_csv)
    log(f"  保存: {npz_path}  ({len(rec)} 行)")
    if csv_path:
        log(f"        {csv_path}")
    return "saved"


# ============================================================================
# 退出: 任何路径都走这里（力矩斜坡到零 + 连发零力矩）
# ============================================================================
def safe_shutdown(link, limiters) -> None:
    """退出前: 按 0.1 N·m/步的斜坡把力矩压到 0，再连发若干帧"纯零力矩"。

    斜坡而不是直接置零: 力矩阶跃会激发齿隙冲击（保护减速器）；斜坡总共 ≤ 0.4 s。
    """
    limiters[0].last = float(limiters[0].last)
    limiters[1].last = float(limiters[1].last)
    t0_ns = time.perf_counter_ns()
    k = 0
    ok = True
    try:
        while k < 40 and (abs(limiters[0].last) > 1e-12 or abs(limiters[1].last) > 1e-12):
            busy_wait_until(t0_ns + k * DT_NS)
            k += 1
            tb = limiters[0].limit(0.0)
            ts = limiters[1].limit(0.0)
            link.send(tb, ts, 0.0, 0.0)
        limiters[0].clear()
        limiters[1].clear()
        for i in range(ZERO_FRAMES_AT_EXIT):
            busy_wait_until(t0_ns + (k + i) * DT_NS)
            link.send(0.0, 0.0, 0.0, 0.0)
    except KeyboardInterrupt:
        # Ctrl+C 之后仍然尽力把零力矩发出去
        try:
            for _ in range(ZERO_FRAMES_AT_EXIT):
                link.send(0.0, 0.0, 0.0, 0.0)
        except Exception:
            ok = False
    except Exception as exc:  # pragma: no cover - 串口异常
        ok = False
        log(f"  [WARN] 退出时发零力矩失败: {exc}")
    log(f"  已发送 {ZERO_FRAMES_AT_EXIT} 帧零力矩并停止" if ok
        else "  [WARN] 零力矩帧未能全部发出（链路已断）")


# ============================================================================
# 命令行
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="双级 yaw 系统辨识数据采集（分轴激励 + 录制序列增强 + 上位机 PID）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--segments", type=int, default=SEGMENTS_DEFAULT, help="采集段数（偶数段驱动大 yaw，奇数段驱动小 yaw）")
    p.add_argument("--duration-sec", type=float, default=DURATION_SEC,
                   help="每段秒数（标准 3 s = 300 点 @100 Hz）")
    p.add_argument("--rate", type=float, default=RATE,
                   help="采样率 Hz（**固定 100**；传更高会被拒绝并回到 100）")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="保存目录")
    p.add_argument("--tag", default=TAG_DEFAULT, help="文件名 tag（默认按 axis 自动取 big/small）")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                   help="随机数种子（增强/激励序列/每段随机 β0 **都**由它决定 ⇒ 采集是"
                        "**确定性**的）。★ 采留出/测试集时必须换一个 seed，否则会得到与"
                        "训练集**逐位相同**的数据（辨识脚本会做指纹比对并告警）")
    p.add_argument("--max-temp", type=float, default=MAX_TEMP_C, help="电机过温阈值 ℃")
    p.add_argument("--kp", type=float, default=PID_KP, help=f"PID 比例增益（默认 {PID_KP}）")
    p.add_argument("--ki", type=float, default=PID_KI, help=f"PID 积分增益（默认 {PID_KI}）")
    p.add_argument("--kd", type=float, default=PID_KD,
                   help=f"PID 微分增益（默认 {PID_KD}；注意用的是未滤波差分, 给大会抖）")
    p.add_argument("--pid-deadband-deg", type=float, default=PID_DEADBAND_DEG,
                   help="★ **PID 误差死区**（度，默认 0 = 关闭）: |e| ≤ 死区 ⇒ e_eff=0"
                        "（只留积分项，被保持轴不再追编码器噪声/在背隙里蹭）；"
                        "**|e| > 死区 ⇒ 误差按「到死区边界的距离」e∓d 算**（不是到目标点的距离）"
                        "⇒ 输出在边界处连续、没有 kp·d 的力矩跳变。"
                        "建议取 0.3~0.5°（越大保持段静态偏差越大，但力矩越干净）；"
                        "记进 npz 的 `pid_deadband`。")
    p.add_argument("--record-hold", action=argparse.BooleanOptionalAction, default=RECORD_HOLD_DEFAULT,
                   help="把每次的静止保持段（到位等待, 含大角度阶跃）也落盘并参与辨识"
                        "（文件名后缀 %s；**最开始的第一次不记**）。默认开；--no-record-hold 关闭"
                        % HOLD_SUFFIX)
    p.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=SAVE_CSV_DEFAULT,
                   help="★ **默认不写 CSV**: 只落 npz（列更全、体积约小 2 倍、辨识直接可读）。"
                        "给 --save-csv 才同时写同名 .csv（**只为人眼看/给老脚本用**；"
                        "注意目录里 csv 与 npz 成对存在时，辨识脚本只读 npz 那一份）")
    p.add_argument("--settle-sec", type=float, default=SETTLE_SEC,
                   help=f"采样前的到位等待时长 s（默认 {SETTLE_SEC:g}）。"
                        "★ 现在**只有这一段**: 走完就直接采样，不再判稳定性")
    p.add_argument("--tilted", action="store_true",
                   help="静态倾斜段: 每段前提示把底盘以固定倾角静置（**不做任何倾角补偿、"
                        "不改变激励方式**），并把 gravity_ax/ay 记进数据、元数据记 tilted=1")
    p.add_argument("--tilt-rolling", action="store_true",
                   help="静态倾斜段 + 段间提示操作者轮换倾角（隐含 --tilted）；"
                        "段与段之间留出改倾角的时间，采集期间底盘仍然不动")
    p.add_argument("--held-big-stratified", action="store_true",
                   help="小 yaw 被激励的段里, held 大 yaw 方位角**按下标均匀铺满 ±π**"
                        "（默认关: 纯随机 ±π）。固定一个底盘倾角时, 这样能让 A 系里的 "
                        "g_A 方向覆盖更均匀 ⇒ P 的条件数更好")
    p.add_argument("--dry-run", action="store_true",
                   help="无硬件自检: 用内置仿真（planar_yaw_model.h 同方程）代替串口")
    sg = p.add_argument_group(
        "★ dry-run 仿真环境 2/3",
        "环境 2（--sim-rigid）: 背隙「接触完全刚性 + 死区完全自由 + β 随机/漂移」，"
        "仅用于采集**测试数据**。与默认的平滑背隙被控对象（τ_t = k[dz(Δ)+γΔ]+cΔ̇）"
        "不同: 死区内 τ_t≡0（连阻尼都没有）、接触后电机/云台**刚性锁定**（k = ∞）、"
        "撞击为完全非弹性冲击；且**每条数据**的背隙中心 β0 重新随机抽样并随时间微弱漂移。"
        "目的是把背隙建模逼到最不利情形: 死区内零刚度 ⇒ δ/k/c 几乎无梯度；k=∞ 只能用有限 k 近似；"
        "β 每条数据都不同 ⇒ 单个全局 β 不可能对，必须用估计器的在线值（数据里的 backlash_center 列）。\n"
        "环境 3（--sim-no-backlash）: **被控对象根本没有背隙**（δ=0），"
        "用于可辨识性诊断 —— 看当前 18 参模型会不会在无背隙数据上\"认\"出一个假的死区。")
    sg.add_argument("--sim-rigid", action="store_true",
                    help="dry-run 用**刚性接触**环境（隐含: 记录 backlash_center / "
                         "backlash_beta_true 两列；实机也会记，实机真值恒 0）")
    sg.add_argument("--sim-no-backlash", action="store_true",
                    help="★ dry-run 用**没有背隙**的被控对象（δ = 0 ⇒ 电机与云台之间没有空行程，"
                         "只剩一条刚度为 --sim-no-backlash-k 的同步带弹簧）。用途: 可辨识性诊断 —— "
                         "喂给当前 18 参（含 δ/k/c/γ/Pb）模型，看它能不能识出'其实没有背隙'。"
                         "与 --sim-rigid 互斥")
    sg.add_argument("--sim-no-backlash-k", type=float, default=SIM_NO_BACKLASH_K,
                    help=f"无背隙环境里那条同步带的刚度（N·m/rad，默认 {SIM_NO_BACKLASH_K:g}）；"
                         "给大（如 1e4）就等价于'既无空行程、又近似刚性'")
    sg.add_argument("--sim-com-random", action="store_true",
                    help="★ dry-run 给**小 yaw 上装 (P) 与大 yaw 转子 (Pb) 各抽一个随机重心偏置**"
                         "（每次运行抽一次，由 --seed 决定；方向均匀、幅值 "
                         f"|P|~U{SIM_COM_P_RANGE} / |Pb|~U{SIM_COM_PB_RANGE} kg·m），"
                         "免得辨识只在某一组恰好方便的偏置上验证；真值写进 npz（*_true 标量）",
                    )
    sg.add_argument("--sim-com-seed", type=int, default=SIM_COM_SEED,
                    help="★ 随机重心偏置用**独立**的种子（默认 None = 用本次运行的 --seed）。"
                         "train/val、水平/倾斜 几次运行要**共享同一组重心真值**时，"
                         "都传同一个 --sim-com-seed（否则各自抽各自的，留出集与训练集物理对象不同）")
    sg.add_argument("--sim-tilt-deg", type=float, default=SIM_TILT_DEG,
                    help="★ dry-run 的**底盘静态倾角**（度，绕底盘 y 轴，默认 0 = 水平）。"
                         "非 0 时被控对象的 A 系重力**随大 yaw 平台角旋转**"
                         "（g_A = Rz(−θ_p)·g_C，与实机估计器给 gravity_a 的口径一致），"
                         "并把 gravity_ax/ay 记进数据、元数据 tilted=1 ⇒ 用于验证'倾斜下 P 可辨识'。"
                         "注意: 实机采倾斜数据用 --tilted（只提示静置姿态），两者互不影响")
    sg.add_argument("--sim-beta-random-frac", type=float, default=SIM_BETA_RANDOM_FRAC,
                    help="每条数据的 β0 随机幅度（占 δ 的比例，默认 0.30）")
    sg.add_argument("--sim-beta-drift-frac", type=float, default=SIM_BETA_DRIFT_FRAC,
                    help="β 漂移幅值（占 δ 的比例，默认 0.05）")
    sg.add_argument("--sim-beta-drift-period", type=float, default=SIM_BETA_DRIFT_PERIOD,
                    help="β 漂移周期 (s)，默认 30")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    # ── 采样率: 固定 100 Hz，不允许更高 ──
    if abs(args.rate - RATE) > 1e-9:
        log(f"[WARN] --rate={args.rate:g} 不被支持: 采集固定 {RATE:g} Hz"
            f"（更高频率只会重复下发同一份估计值，破坏力矩-状态的时间对应关系）→ 使用 {RATE:g} Hz")
        args.rate = RATE
    samples = int(round(args.duration_sec * RATE))
    if samples != SAMPLE_LEN:
        log(f"[WARN] --duration-sec={args.duration_sec:g} → {samples} 点（标准段为 "
            f"{SAMPLE_LEN} 点 = 3 s @100Hz）")
    if samples <= 0:
        raise SystemExit("[ERROR] --duration-sec 必须 > 0")

    # ── 静态倾斜段: --tilt-rolling 隐含 --tilted（只提示 + 记录，不改激励、不做补偿）──
    if args.tilt_rolling and not args.tilted:
        log("[INFO] --tilt-rolling 隐含 --tilted（段间会提示轮换倾角）")
        args.tilted = True

    os.makedirs(args.out, exist_ok=True)
    targets = load_all_targets()
    planners = make_planners()
    rng = np.random.default_rng(args.seed)
    _db = math.radians(max(0.0, float(args.pid_deadband_deg)))
    pids = [PidController(args.kp, args.ki, args.kd, PID_OUT_MIN, PID_OUT_MAX, "big", _db),
            PidController(args.kp, args.ki, args.kd, PID_OUT_MIN, PID_OUT_MAX, "small", _db)]
    limiters = [TorqueRateLimiter(), TorqueRateLimiter()]

    log("=" * 78)
    log("双级 yaw 系统辨识数据采集 — 分轴激励（录制序列 + 增强 + 上位机 PID）")
    log(f"  段数={args.segments}  每段={samples} 点 ({samples * DT:.2f}s @{RATE:g}Hz)  "
        f"到位={args.settle_sec:g}s  种子={args.seed}")
    log(f"  PID: kp={args.kp} ki={args.ki} kd={args.kd} 输出限幅 ±{PID_OUT_MAX:g} N·m  "
        f"力矩变化限幅 {MAX_TORQUE_DELTA:g} N·m/步")
    log(f"  静止保持段记录={'开（后缀 ' + HOLD_SUFFIX + '）' if args.record_hold else '关'}"
        f"（首段除外）")
    log(f"  仅力矩模式(mode=0)  pitch=0  小 yaw 行程="
        f"[{_deg(SMALL_TRAVEL_MIN):+.0f}°, {_deg(SMALL_TRAVEL_MAX):+.0f}°]"
        f"（软限位 [{_deg(SMALL_SOFT_MIN):+.0f}°, {_deg(SMALL_SOFT_MAX):+.0f}°]，越界只告警）"
        f"  参考包络=[{_deg(SMALL_ENV_MIN):+.0f}°, {_deg(SMALL_ENV_MAX):+.0f}°]"
        f"  中止阈值=同硬限位  中心={_deg(SMALL_CENTER_RAD):+.1f}°  过温={args.max_temp:g}℃")
    if getattr(args, "held_big_stratified", False):
        n_small = max(1, args.segments // 2)
        offs = [(-180.0 + (k + 0.5) * 360.0 / n_small) for k in range(n_small)]
        log(f"  ★ held 大 yaw 方位角分层: 开（{n_small} 个小 yaw 段）; 相对首个"
            f"小 yaw 段的平台方位角的偏移 = [{', '.join(f'{o:+.1f}°' for o in offs)}]"
            f" ⇒ 固定一个倾角时 A 系里的 g_A 方向覆盖 ±π")
    if args.tilted:
        log(f"  静态倾斜段: 开（tilted=1{'; 段间轮换倾角' if args.tilt_rolling else ''}）"
            f" —— 只提示静置姿态 + 记录 gravity_ax/ay，不做补偿、不改激励")
    if args.dry_run and abs(getattr(args, "sim_tilt_deg", 0.0)) > 1e-12:
        log(f"  [DRY-RUN] ★ 底盘静态倾角 = {args.sim_tilt_deg:+.2f}°（绕 y）"
            f" ⇒ A 系重力随大 yaw 平台角旋转，|g_A| = "
            f"{9.81 * abs(math.sin(math.radians(args.sim_tilt_deg))):.2f} m/s²，记 tilted=1")
    if _db > 0.0:
        log(f"  ★ PID 误差死区: ±{args.pid_deadband_deg:g}°（{_db:.5f} rad）"
            f" —— 死区内 P/D 按 0 算、积分冻结")
        log("    （稳定判据已删除 ⇒ 死区多大都不会「卡在门限外」，它只决定保持段的静态偏差）")
    log(f"  保存目录: {args.out}")
    log(f"  保存格式: {'npz + csv' if args.save_csv else 'npz（默认，不写 csv）'}"
        + ("  ← 也记静止保持段" if args.record_hold else "  ← 不记静止保持段"))
    if args.dry_run:
        log("  [DRY-RUN] 无硬件: 用内置仿真代替串口"
            f"（planar_yaw_model.h 同方程, λ={SIM_FRICTION_LAMBDA:g}, "
            f"积分步长 {SIM_INT_STEP * 1e3:.3f} ms）")
        if args.sim_rigid:
            log(f"  [DRY-RUN] ★ 仿真环境 2: 接触**完全刚性** + 死区**完全自由**"
                f"（τ_t≡0 in |Δ−β|<δ/2），撞击=完全非弹性冲击; "
                f"β0 每条数据随机 ±{args.sim_beta_random_frac:.2f}·δ, "
                f"漂移 ±{args.sim_beta_drift_frac:.2f}·δ / {args.sim_beta_drift_period:g}s")
            log(f"  [DRY-RUN] 注意: 这个环境里 k/c/γ **无效**；"
                f"记录列 backlash_center(在线估计) / backlash_beta_true(真值)")
        if getattr(args, "sim_no_backlash", False):
            log(f"  [DRY-RUN] ★ 仿真环境 3: **没有背隙**（δ=0）—— 电机与云台之间没有空行程，"
                f"只剩 τ_t = k·Δ + c·Δ̇（k={args.sim_no_backlash_k:g} N·m/rad, "
                f"c=defaultModelParams 的 {2.0:g}）")
            log(f"  [DRY-RUN] 用途: 可辨识性诊断 —— 喂给当前 16 参（含 δ）模型，"
                f"看它能不能识出'其实没有背隙'；npz 里记 no_backlash=1")
    log("=" * 78)

    args._held_base = None          # --held-big-stratified 的基准平台方位角（首个小 yaw 段时确定）
    link = (SimRobotLink(rng, rigid=args.sim_rigid,
                         tilt_deg=getattr(args, "sim_tilt_deg", 0.0),
                         beta_random_frac=args.sim_beta_random_frac,
                         beta_drift_frac=args.sim_beta_drift_frac,
                         beta_drift_period=args.sim_beta_drift_period,
                         no_backlash=getattr(args, "sim_no_backlash", False),
                         no_backlash_k=getattr(args, "sim_no_backlash_k",
                                               SIM_NO_BACKLASH_K))
            if args.dry_run else HwRobotLink())
    # ── ★ 随机重心偏置（--sim-com-random）: **每次运行抽一次**（由 --seed 决定）──
    #   小 yaw 上装 P 与大 yaw 转子 Pb 各抽一个（方向均匀、幅值在 SIM_COM_*_RANGE）——
    #   免得"辨识成功"只发生在某一组恰好方便的偏置上。真值由 plan.*_true 落进 npz。
    if args.dry_run and getattr(args, "sim_com_random", False):
        _com_seed = getattr(args, "sim_com_seed", None)
        com_rng = np.random.default_rng(_com_seed) if _com_seed is not None else rng
        if _com_seed is not None:
            log(f"  [DRY-RUN] 随机重心偏置用独立种子 --sim-com-seed={_com_seed}"
                f"（train/val、水平/倾斜 共享同一组真值）")
        for key, mag_range, tag in (("P", SIM_COM_P_RANGE, "P  (小 yaw 上装一阶矩)"),
                                    ("Pb", SIM_COM_PB_RANGE, "Pb (大 yaw 转子一阶矩)")):
            mag = float(com_rng.uniform(*mag_range))
            ang = float(com_rng.uniform(0.0, TWO_PI))
            link.plant.p[key + "x"] = mag * math.cos(ang)
            link.plant.p[key + "y"] = mag * math.sin(ang)
            log(f"  [DRY-RUN] ★ 随机重心偏置 {tag}: |{key}| = {mag:.5f} kg·m, "
                f"方向 {math.degrees(ang):+7.1f}° ⇒ ({key}x, {key}y) = "
                f"({link.plant.p[key + 'x']:+.5f}, {link.plant.p[key + 'y']:+.5f})")

    saved, attempts = 0, 0
    exit_code = 0
    interrupted = False
    try:
        if not link.wait_ready(15.0):
            log("[ERROR] 15s 内没有收到有效的 MCU/IMU 数据（串口未连接？）")
            return 2
        log("数据链路就绪")

        idx = 0                                   # 逻辑段号（决定 driven 轴与文件名序号）
        while idx < args.segments:
            attempts += 1
            if attempts > args.segments * 4:
                log("[ERROR] 连续中止/过热次数过多，退出")
                exit_code = 1
                break
            result = collect_segment(link, rng, targets, planners, pids, limiters,
                                     args, idx, samples)
            if result == "saved":
                saved += 1
                idx += 1
            elif result == "retry":
                continue                          # 过热: 重采同一段号
            else:
                log("  [SKIP] 本段中止（小 yaw 越限或链路异常），跳过该段号")
                idx += 1
    except KeyboardInterrupt:
        interrupted = True
        log("\n[Ctrl+C] 立即停止激励并回零…")
    finally:
        try:
            safe_shutdown(link, limiters)
        finally:
            try:
                link.close()
            except Exception:
                pass

    log("-" * 78)
    log(f"完成: 保存 {saved}/{args.segments} 段 → {args.out}")
    if isinstance(link, SimRobotLink):
        log(f"  [DRY-RUN] 仿真下发 {link.frames} 帧, 结束状态 "
            f"θ_big={link.plant.q[0]:+.4f} θ_small={link.plant.q[1]:+.4f} rad")
    if interrupted:
        return 130
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
