#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_small_zero.py — 小 yaw 编码器零位标定（离心平衡点法 · **相对偏置版**）

用法::

    python3 python/scripts/calibrate_small_zero.py --omega=40 --repeats=5 \
            --ack-mcu-limits-bypassed                 # 真机（先读完三条前置条件!）
    python3 python/scripts/calibrate_small_zero.py --report-only   # 只测平衡点, 不写 offset
    python3 python/scripts/calibrate_small_zero.py --probe-travel  # 标定后再量机械行程自检
    python3 python/scripts/calibrate_small_zero.py --sim           # 虚拟台架 + 消融（无硬件）

================================================================================
★ 三条前置条件（**必须**先读; 违反任何一条都会把数据做成垃圾）
================================================================================

**(1) 标定前，"绝对角度"信息量为零 —— 本脚本只用相对读数。**
    电控送来的小 yaw 编码器 **0 点对应哪个物理位置是未知的**；而 `[−30°, +30°]` 这套
    行程限位是**以标定后的 0 点（= 离心平衡点）为基准**定义的。所以在标定完成之前:
      · 不知道"行程中心 −2.5°"在哪;
      · 不知道"现在离限位还有多少度";
      · 因此**任何"绝对到位"的做法都是错的**。
    本脚本一律把位置指令写成 **相对当前读数的偏置**: ``target = mapped_now + δ``，
    δ 只有几度（``--bias-deg-lo/hi``，默认 2°~5°）。**第一次测量甚至完全不预置**:
    直接从当前位置松手（τ=0）→ 大 yaw 加速到 Ω → 稳定 → 等小 yaw 停住 → 读 θ̂₀。

**(2) "两侧逼近"必须用"旋转稳定之后的相对偏置"实现，不能靠绝对位置。**
    Ω 稳定之后，才给一个相对偏置把起点放到平衡点估计的两侧（逐次交替）:
    ``target = mapped_now ± δ``，用 PID 走完这段**相对**位移（低力矩 + 失速检测）。
    之后松手（τ=0），等它停在平衡点（±粘滞带）上，读该次结果。

**(3) 电控侧的小 yaw 限位/软限位保护用的是同一个（错误的）零点，标定前它会误判。**
    标定期间必须**临时放宽或旁路电控的小 yaw 限位保护**（保留脚本的低力矩限幅 +
    机械硬限位作为最后一道防线），否则电控会在"它以为越界"时把小 yaw 力矩清零 ——
    脚本会检测"**下发了力矩但角度不跟随**"（失速）并明确提示，而不是默默采一堆垃圾。
    标定出 `Δoffset` 并写入之后，**再恢复** `[−28°, +28°]` 那套目标夹取。
    真机路径必须显式确认: ``--ack-mcu-limits-bypassed``（否则脚本拒绝开跑）。

因此安全策略也**全部换成相对量**（绝对守卫在标定前没有意义）:

    · **相对起始读数位移预算** |mapped − mapped0| ≤ --rel-budget-deg（默认 12°）
    · **相对路径预算**（累计移动量）≤ --rel-path-budget-deg（默认 150°）
    · **失速检测**: 施加了力矩但 |Δθ| 在 --stall-sec 内不跟随 ⇒ 判"撞到机械限位/卡住"
      ⇒ **立即停止并反向退回**（撞限位时不许继续加力）
    · **低力矩限幅** --bias-torque（默认 0.15 N·m; 只用于相对偏置与探针）
    · **|θ̇_small| 过大立即中止**（--rate-abort，默认 8 rad/s）
    · **加速段冲量约束**（见 §四）: 自由松手 + 快斜坡会把小 yaw 甩飞（实测 −1085°），
      因此默认用**慢斜坡**（--accel-limit 默认 2.0 rad/s²，Ω=40 时 20 s），
      或者用 --spinup-hold hold 在加速段做**相对保持**（可快 4 倍）

================================================================================
一、物理原理
================================================================================

平面模型（``include/tcbs/mpc/planar_yaw_model.h``）里小 yaw 的力矩方程::

    τ_s = M12·θ̈_b + M22·θ̈_s − ½·μ(θ_s)·θ̇_b² − G_s
          + ω_c(½μ θ̇_s − μ θ̇_b) − ½μ ω_c² + M12·α_c + fric_s + τ_drag,s

    M11 = Jbig_eff + J_s + 2·d·Q,  M12 = J_s + d·Q,  M22 = J_s
    Q(θ_s) = R(θ_s)·P,  μ(θ_s) = ∂M11/∂θ_s = 2(dy·Qx − dx·Qy)
    P = m_u·ρ（上装一阶矩）,  d = (dx,dy) 两轴平面偏置

在三个前提下 ①**大 yaw 恒定 Ω**（θ̇_b = Ω、θ̈_b = 0）②**底盘绕关节轴静止**（ω_c = α_c = 0）
③**底盘水平**（gravity_a 平面分量 = 0 ⇒ G_s = 0），令**小 yaw 力矩为 0 且静止**，
**忽略阻力**时方程只剩 ``0 = −½·μ(θ_s)·Ω²`` ⇒:

    ★ μ(θ_s*) = 0  ⇔  Q(θ_s*) = R(θ_s*)·P ∥ d  ⇔  θ* = angle(d) − angle(P)

含义: **离心力把上装质心甩到"大 yaw 轴 ↔ 小 yaw 轴"连线的径向外侧**（ρ 与 d 同向）。
等效刚度 ``k = Ω²·|d|·|P|``。该位置**只由机械几何决定、与编码器零位无关、可重复** ⇒
它就是"天然物理参考"，测得它对应的编码器读数 `mapped_eq` 就得零位偏移。

================================================================================
二、★ 两类系统误差、两套抵消手段（**两个都必须做**）
================================================================================

| 误差源 | 对 Ω 的奇偶性 | 抵消手段 |
|---|---|---|
| **静摩擦/粘滞带**（从哪一侧逼近 ⇒ 停在 θ* ± fc/k） | **偶函数**（Ω → −Ω 不变） | **两侧交替逼近后平均** |
| **空气阻力等与 Ω 成正比的耗散** ``τ_drag,s = −c(|Q|²θ̇_s + Ω(d·Q + |Q|²))`` | **奇函数**（含线性于 Ω 的项） | **±Ω 双向平均** |
| **底盘倾斜**（重力项 G_s） | **偶函数** | ❌ 消不掉 ⇒ 只能**把底盘放平** |

* ``单侧偏差 = fc/k = fc_small/(Ω²|d||P|)``（= 粘滞带半宽）: 左侧逼近停在 θ* − fc/k，
  右侧停在 θ* + fc/k，两侧平均才得 θ*。**阻力是"平衡点整体平移"**（与逼近侧无关），
  所以**两侧交替消不掉它**；但它恰好是 **Ω 的奇次项**，**±Ω 平均正好抵消**。
* 理想（无未建模 Ω 奇次项）模型里 ±Ω 完全等价；**一旦存在 Ω 奇次项（阻力/风阻/线缆拖曳），
  ±Ω 就是必需的**。
* 另外两个"没有信息量"的坑:
  · **起点落在粘滞带内**（距 θ* < fc/k）⇒ 小 yaw 根本不动 ⇒ 该次测量等于"起点本身"。
    脚本记录 `moved_deg`/`stuck` 并在卡住过半时告警（应加大 Ω 或加大 δ）。
  · **加速段冲量** ``∫M12·θ̈_b dt = M12·Ω``（与斜坡快慢无关）: 自由松手时会被甩出
    ``Δθ̇_s ≈ (M12/M22)Ω``。**摩擦能吸收的量是 fc·t_ramp** ⇒ 必须
    ``t_ramp ≥ M12·Ω/fc_small``，否则小 yaw 直接飞出去（本仓库参数实测:
    accel=8 rad/s² 时位移 **−1085°**、|θ̇|max = 10.9 rad/s; accel=2 时只有 −6.1°、0.04 rad/s）。

================================================================================
三、每次测量的流程（相对偏置）
================================================================================

    第 1 次（θ̂ 未知）: 不预置 —— 直接松手（τ=0）→ 大 yaw 加速到 Ω → 等 Ω 稳定
                        → 等小 yaw 停住（|θ̇|<tol 持续 settle_sec）→ 采样窗口平均 ⇒ θ̂₀
    之后每次          : 松手 → 大 yaw 加速到 Ω → 等 Ω 稳定
                        → ★**此刻**施加相对偏置 target = mapped_now ± δ（交替; 低力矩+失速检测）
                        → 松手 → 等收敛 → 采样窗口平均 ⇒ 本次停位
    收尾              : 相对保持（把当前读数当目标）+ 减速到 0

    每方向重复 --repeats 次; ±Ω 两个方向都做; 最后**两侧合并 + 双向合并**。

================================================================================
四、时间与安全参数（为什么这么定）
================================================================================

* ``--accel-limit``（默认 2.0 rad/s²）: 由冲量约束反解 —— ``t_ramp = Ω/accel ≥ M12·Ω/fc``
  ⇒ 与 Ω 无关地要求 ``accel ≤ fc_small/M12``。本仓库参数（M12 ≈ 0.0178、fc = 0.052）给出
  **accel ≤ 2.9 rad/s²**，故默认 2.0。给出 `--fc-small/--p-moment/--d-offset` 时脚本会算出
  "吸收比"并在 < 1 时**告警**。``--spinup-hold hold`` 允许加速段做相对保持（可把 accel 提到 8）。
* ``--bias-torque``（默认 0.15 N·m）: 只用于相对偏置/探针，必须**大于 fc_small**
  （否则连静摩擦都推不动），但远小于额定的 1 N·m。
* ``--stall-sec``（0.8 s）/``--stall-deg``（0.15°）: "加了力却不跟随" ⇒ 撞限位/卡住。

================================================================================
五、输出
================================================================================

* 终端: 每次原始值（起点读数、偏置 δ 与**实际达成**、松手角、停位、位移、是否卡住/失速）、
  **左侧均值 / 右侧均值 / 两侧合并均值**、**单侧偏差**（(右−左)/2）、**双向合并均值**、
  ±Ω 差、实测极差、理论 fc/k、卡住次数、相对预算已用量;
* ``--out``（默认 ``data/sysid/small_zero_calib.json`` + 同名 ``.csv``）;
* **零位建议**: ``mapped = raw·scale + offset`` ⇒ ``Δoffset = −mapped_eq``
  （`--report-only` 不给建议）;
* **★ 副产品**: 标定后模型里 **P = |P|·d̂**（dy = 0 ⇒ **Py = 0**）—— P 的**方向**被物理标定
  确定、只剩模长待辨识（缓解"水平数据下 Px/Py 与惯量共线"）; 前提: d 方向可信、
  平衡点可达、底盘水平;
* ``--probe-travel``: 标定后用低力矩 + 失速检测找两侧机械停止位 ⇒ ① 行程宽度 W
  （应 ≈60° ⇒ **scale 校核**）② 平衡点到两侧距离（应 ≈ +30° / ≈ −30°）③ 自洽性结论。

================================================================================
六、--sim 虚拟台架
================================================================================

* **1 自由度小 yaw 被控对象**（大 yaw 视为理想速度源 θ̇_b = Ω、θ̈_b = Ω̇）::

      J_s·θ̈_s = τ_s + F_drive(θ_s, θ̇_s) − fric(θ̇_s)
      F_drive  = ½μ(θ_s)Ω² + G_s − M12·Ω̇ − c·(|Q|²θ̇_s + Ω(d·Q + |Q|²))   ← 阻力项
  · **真·库仑摩擦 + stick 判定**: ``|θ̇|<ε 且 |F|≤fc ⇒ θ̇=0、θ̈=0（原地不动）``
  · **λ=100 tanh** 版本保留作对照（静止摩擦=0 ⇒ 必然收敛到 μ=0，**过于乐观**）
  · ``--sim-drag=<c>`` 打开阻力项（默认 0.5, 同时跑 c=0 组）
* **仿真也遵守"相对偏置"**: 起点一律由"当前读数 + δ"给出，**绝不用真值零点设起点**，
  真值只用于最后算误差。
* 消融: ①单侧单次 ②单侧多次 ③两侧交替多次 ④±Ω 对照 ⑤不同 Ω ⑥倾斜 ⑦stick/tanh
  ⑧撞限位/失速 ⑨ fc/k ∝ 1/Ω² ⑩ **阻力组（Ω 奇次项）** ⑪ 甩飞验证。

结果表写进 ``docs/small_zero_calib.md``（``docs/sysid_data.md`` 里有链接）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

# ── 让脚本能直接 `python3 python/scripts/calibrate_small_zero.py` 跑 ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (_REPO, os.path.join(_REPO, "python"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torque_controller import TcbsRobotCommunication  # noqa: E402

# 与采集脚本共用同一份行程常量（硬限位/包络/中心是"标定后"的语义）
from collect_sysid import (  # noqa: E402
    DT,
    RATE,
    SMALL_CENTER_RAD,
    SMALL_TRAVEL_MAX,
    SMALL_TRAVEL_MIN,
    PidController,
    TorqueRateLimiter,
    log,
    wrap_pi,
)

# ============================================================================
# 常量
# ============================================================================
OMEGA_MAX = math.pi
#   ↑ ★ **硬上限: 2 s 一圈** ⇒ Ω ≤ π ≈ 3.1416 rad/s（用户明确要求; 超过则钳到 π 并告警）
DEFAULT_OMEGA = OMEGA_MAX       # 默认就取上限（离心法越慢越糟, 所以取允许的最快值）
MAX_ACCEPTABLE_DEG = 5.0       # 单侧偏差的可接受上限: 超过就判"该方法不可行"
MANUAL_TOL_DEG = 0.2           # ★ manual 模式: 窗口内极差 ≤ 该值才接受（手扶稳的判据）
MANUAL_TRIES = 5               # manual: 每次重复最多重摆几次
GRAVITY_G = 9.81
DEFAULT_TILT_DEG = 90.0        # 重力法主倾角（把小 yaw 轴放到接近水平 ⇒ 刚度最大）
DEFAULT_TILT_LIST = "90,60"    # 多倾角平均（粘滞带 ∝ 1/sinφ 不同 ⇒ 平均掉粘滞带偏差）
DEFAULT_REPEATS = 5             # 每个方向的重复次数
ACCEL_LIMIT = 2.0              # 加速段斜坡上限 rad/s²（由冲量约束反解, 见 §四）
BIAS_TORQUE = 0.25             # 相对偏置/探针力矩上限 N·m
#   ↑ 稳态 Ω 下把 yaw 推离平衡点需要克服 **离心等效弹簧 + 静摩擦**: τ ≥ fc + k·δ
#     （k = Ω²|d||P|）。所以"接近 fc 的 0.05 N·m"在稳态下**推不动**（本仓库参数:
#     fc=0.052、Ω=40 时 k·δ(δ=5°)=0.06 ⇒ 至少 0.11 N·m）。默认取 0.25 N·m
#     （≈5×fc，仍远小于额定的 1 N·m），并在启动时按已知参数打印**建议值**。
BIAS_SEC = 3.0                 # 相对偏置最长用时
BIAS_TOL = math.radians(0.5)   # 偏置到位容差
BIAS_KP = 4.0
#   ↑ 偏置 P 增益 N·m/rad: 必须在目标误差(~5°)处就能**饱和**到 --bias-torque，
#     否则 P 输出 ≈ k·δ 刚好等于弹簧力、差一个 fc 就卡住（实测会把每次测量都判成失速）。
BIAS_LO = math.radians(2.0)
BIAS_HI = math.radians(5.0)
REL_BUDGET_DEG = 12.0          # 相对起始读数位移预算
REL_PATH_BUDGET_DEG = 400.0    # 累计路径预算
#   ↑ 这是"异常振荡/反复来回"的探测器（行程风险由**位移预算**管），不是行程守卫:
#     一次正常标定（10 次测量 × 每次偏置移动 + 收敛摆动）累计路径约 100~300°。
STALL_SEC = 0.8                # 失速判定时间
STALL_DEG = 0.15               # 失速判定位移阈值
STALL_HARD_DEG = 0.05          # 失速且位移 < 该值 ⇒ 判"完全推不动"（疑似撞机械限位）
RATE_ABORT = 8.0
RATE_TOL = 0.15
OMEGA_TOL = 0.5
SETTLE_SEC = 2.0
SETTLE_TIMEOUT = 20.0
WINDOW_SEC = 1.0
PROBE_TORQUE = 0.12
PROBE_MAX_DEG = 30.0
PROBE_SEC = 15.0
PROBE_SCALE_TOL_DEG = 3.0
BIG_KP, BIG_KI = 0.25, 0.15
BIG_TAU_LIMIT = 1.0
SMALL_TAU_LIMIT = 1.0
PID_KP, PID_KI, PID_KD = 2.0, 0.1, 0.2
TORQUE_DELTA = 0.1
GRAVITY_TOL = 0.5
ENC_NOISE = 2e-5
STICK_EPS = 1e-3
SIM_INT_STEP = 2e-4

DEFAULT_OUT = "data/sysid/small_zero_calib.json"
DEFAULT_OUT_SIM = "data/sysid/small_zero_calib_sim.json"

CSV_HEADER = ["index", "direction", "approach", "omega_nominal", "omega_mean",
              "start_read", "bias_cmd", "bias_achieved", "bias_ok", "release_read",
              "theta_mean", "moved_rad", "theta_std", "theta_min", "theta_max",
              "rate_residual", "settle_time_s", "temp_big", "temp_small",
              "gravity_norm", "stuck", "stalled", "ok", "abort_reason"]

APPROACH_NAME = {"L": "左侧(低读数侧)逼近", "R": "右侧(高读数侧)逼近", "C": "盲松手(首测)"}
EXIT_NEED_ACK = 3


def deg(rad: float) -> float:
    return math.degrees(rad)


# ============================================================================
# 解析工具
# ============================================================================
def theta_star(dx, dy, px, py) -> float:
    """μ(θ*)=0 的稳定解: θ* = angle(d) − angle(P)。"""
    return math.atan2(dy * px - dx * py, dy * py + dx * px)


def mu_of(theta, dx, dy, px, py) -> float:
    c, s = math.cos(theta), math.sin(theta)
    qx, qy = px * c - py * s, px * s + py * c
    return 2.0 * (dy * qx - dx * qy)


def stiffness(omega, dx, dy, px, py) -> float:
    return omega * omega * math.hypot(dx, dy) * math.hypot(px, py)


def side_bias_deg(omega, fc_small, dx, dy, px, py) -> float:
    k = stiffness(omega, dx, dy, px, py)
    if k <= 0 or fc_small is None:
        return float("nan")
    return deg(float(fc_small) / k)


def required_omega(fc_small, dx, dy, px, py, target_deg) -> float:
    kk = math.hypot(dx, dy) * math.hypot(px, py)
    if kk <= 0 or fc_small is None:
        return float("nan")
    return math.sqrt(float(fc_small) / (kk * math.radians(target_deg)))


def whip_absorb_ratio(accel, omega, fc_small, m12) -> float:
    """加速段冲量吸收比 = fc·t_ramp / (M12·Ω)（<1 ⇒ 摩擦吸收不了 ⇒ 会被甩飞）。"""
    if not (accel > 0 and omega > 0 and m12 and fc_small):
        return float("nan")
    return (float(fc_small) * (omega / accel)) / (float(m12) * omega)


def grav_stiffness(p_moment, tilt_deg) -> float:
    """重力法等效刚度 k_grav = |P|·g·sinφ（N·m/rad）。"""
    if not p_moment or tilt_deg is None:
        return float("nan")
    return abs(float(p_moment)) * GRAVITY_G * math.sin(math.radians(tilt_deg))


def side_bias_grav_deg(fc_small, p_moment, tilt_deg) -> float:
    k = grav_stiffness(p_moment, tilt_deg)
    if not (k > 0) or fc_small is None:
        return float("nan")
    return deg(float(fc_small) / k)


def required_p_moment(fc_small, tilt_deg, omega, target_deg) -> float:
    """要把单侧偏差压到 target_deg 需要多大的 |P|（重力法）/ 以及需要多大 |d||P|（离心法）。"""
    if fc_small is None:
        return float("nan")
    if omega:
        return float(fc_small) / (math.radians(target_deg) * omega * omega * 1.0)   # 还要除 |d|
    k = float(fc_small) / math.radians(target_deg)
    return k / (GRAVITY_G * math.sin(math.radians(tilt_deg)))


def choose_method(args, geom) -> dict:
    """按"能不能推动/精度够不够"选方法（--method=auto 时的判定），并给出理由。

    · 离心法单侧偏差 = fc/(Ω²|d||P|)，Ω 被硬上限钳在 π;
    · 重力法单侧偏差 = fc/(|P|·g·sinφ)。
    判据: 单侧偏差 ≤ --max-acceptable-deg 才算可行; 否则**明确报告不可行**。
    """
    fc = geom.get("fc_small")
    dd, pp = geom.get("d_offset"), geom.get("p_moment")
    out = dict(fc=fc, d=dd, p=pp, omega=args.omega_used,
               tilt_list=list(args.tilt_list), reasons=[])
    out["side_cent"] = (side_bias_deg(args.omega_used, fc, dd or 0.0, 0.0, pp or 0.0, 0.0)
                        if (fc and dd and pp) else float("nan"))
    out["feasible_cent"] = bool(out["side_cent"] == out["side_cent"]
                                and out["side_cent"] <= args.max_acceptable_deg)
    grav = {}
    for phi in args.tilt_list:
        grav[phi] = side_bias_grav_deg(fc, pp, phi)
    out["side_grav"] = grav
    vals = [v for v in grav.values() if v == v]
    out["side_grav_best"] = min(vals) if vals else float("nan")
    out["feasible_grav"] = bool(vals and out["side_grav_best"] <= args.max_acceptable_deg)
    out["k_ratio_90"] = (grav_stiffness(pp, 90.0) / stiffness(args.omega_used, dd or 0.0, 0.0,
                                                              pp or 0.0, 0.0)
                         if (pp and dd and args.omega_used) else float("nan"))
    if args.method != "auto":
        out["method"] = args.method
        out["reasons"].append("用户显式指定 --method=" + args.method)
        return out
    out["method"] = "gravity" if out["feasible_grav"] else (
        "centrifugal" if out["feasible_cent"] else "gravity")
    out["reasons"].append("auto: 优先重力法" if out["feasible_grav"] else
                          ("auto: 退回离心法" if out["feasible_cent"] else
                           "auto: **两种方法在 Ω≤π 下都不可行**（仍按重力法测量以便取证）"))
    return out


def drag_shift_deg(omega, c, truth) -> float:
    """阻力引起的平衡点偏移（"Ω 奇次项"的理论值）。"""
    if not c or not omega:
        return 0.0
    dx, dy, px, py = truth["dx"], truth["dy"], truth["Px"], truth["Py"]

    def f(th):
        cth, sth = math.cos(th), math.sin(th)
        qx, qy = px * cth - py * sth, px * sth + py * cth
        mu = 2.0 * (dy * qx - dx * qy)
        return 0.5 * mu * omega * omega - c * omega * (dx * qx + dy * qy + qx * qx + qy * qy)

    lo, hi = math.radians(-60.0), math.radians(60.0)
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return deg(0.5 * (lo + hi) - theta_star(dx, dy, px, py))


# ============================================================================
# 虚拟台架真值
# ============================================================================
# ── ★ 真实量级参数（用户 + 原仓库 TorqueController/data/cars/*/params/
#    Identified_parameters.txt: J=0.016541/0.024986, tau_c=0.097297/0.122514,
#    b=0.0321/0.025648；大小 yaw 同量级、大 yaw 因承载小 yaw ≈翻倍；
#    轴间距 ≈0.1 m；小 yaw 上装 m_u≈0.1 kg、质心位移 ρ≈0.1 m ⇒ |P|≈0.01 kg·m）──
# ★ 实测几何 (dx, dy) = (0, 0.07): 两轴横向无偏置、小 yaw 轴在大 yaw 轴前方 0.07 m
#   ⇒ d 沿 +y（90°）; P 取 30° ⇒ 两者夹角 60° = θ*（非 0，避免退化；见 theta_star）。
REAL = dict(dx=0.0, dy=0.07, p_moment=0.01, p_angle_deg=30.0,
            Js=0.020, Jbig_eff=0.050, fcSmall=0.0973, fvSmall=0.028,
            fcBig=0.22, fvBig=0.055, frictionLambda=100.0,
            zero_offset=0.05, gravity=9.81)


def sim_truth_from_args(args) -> dict:
    """★ 让 --d-offset/--p-moment/--fc-small 等 CLI 参数**真正驱动仿真 plant**。

    默认值就是上面的真实量级参数（用户给的处方）。
    """
    # --d-offset 是**标量轴间距**（|d|）；本构型 d 纯在 y 方向 ⇒ 放到 dy，dx 保持 0
    d = args.d_offset if args.d_offset else REAL["dy"]
    pm = args.p_moment if args.p_moment else REAL["p_moment"]
    ang = math.radians(args.p_angle_deg)
    fc = args.fc_small if args.fc_small else REAL["fcSmall"]
    return dict(dx=REAL["dx"], dy=float(d), Px=float(pm) * math.cos(ang), Py=float(pm) * math.sin(ang),
                Js=float(args.js), Jbig_eff=float(args.jbig), fcSmall=float(fc),
                fvSmall=float(args.fv_small), fcBig=float(args.fc_big),
                fvBig=float(args.fv_big), frictionLambda=100.0,
                zero_offset=float(args.zero_offset_deg) * math.pi / 180.0,
                gravity=GRAVITY_G)



#   （旧的占位几何预设已删除: 现在一律由 CLI/真实参数驱动, 见 sim_truth_from_args）


# ============================================================================
# 1 自由度小 yaw 被控对象
# ============================================================================
class SmallAxisPlant:
    """大 yaw 速度由速度环控制 ⇒ 理想速度源（θ̇_b = Ω、θ̈_b = Ω̇）。

        J_s·θ̈_s = τ_s + F_drive − fric(θ̇_s)
        F_drive  = ½μ(θ_s)Ω² + G_s − M12·Ω̇ − c·(|Q|²θ̇_s + Ω(d·Q + |Q|²))
    """

    def __init__(self, truth: dict, gravity_a=(0.0, 0.0), friction="stick", drag=0.0,
                 int_step=SIM_INT_STEP):
        self.dx, self.dy = truth["dx"], truth["dy"]
        self.px, self.py = truth["Px"], truth["Py"]
        self.Js = truth["Js"]
        self.fc, self.fv = truth["fcSmall"], truth["fvSmall"]
        self.lam = truth["frictionLambda"]
        self.gx, self.gy = float(gravity_a[0]), float(gravity_a[1])
        self.drag = float(drag)
        self.friction = friction
        self.int_step = float(int_step)
        self.q = 0.0
        self.qd = 0.0
        self.omega = 0.0
        self.omega_dot = 0.0

    def derived(self, qs):
        c, s = math.cos(qs), math.sin(qs)
        qx, qy = self.px * c - self.py * s, self.px * s + self.py * c
        return qx, qy, self.dx * qx + self.dy * qy

    def drive_torque(self, qs, qd=0.0):
        qx, qy, dq = self.derived(qs)
        mu = 2.0 * (self.dy * qx - self.dx * qy)
        gs = qx * self.gy - qy * self.gx
        f = 0.5 * mu * self.omega * self.omega + gs - (self.Js + dq) * self.omega_dot
        if self.drag:
            q2 = qx * qx + qy * qy
            f -= self.drag * (q2 * qd + self.omega * (dq + q2))
        return f

    def friction_torque(self, qd, f_drive):
        if self.friction == "stick":
            if abs(qd) < STICK_EPS:
                if abs(f_drive) <= self.fc:
                    return f_drive
                return self.fc * (1.0 if f_drive > 0 else -1.0)
            return self.fc * (1.0 if qd > 0 else -1.0) + self.fv * qd
        return self.fc * math.tanh(self.lam * qd) + self.fv * qd

    def accel(self, qs, qd, tau_s):
        f = self.drive_torque(qs, qd)
        return (tau_s + f - self.friction_torque(qd, f)) / self.Js

    def is_stuck(self, qs, qd, tau_s):
        if self.friction != "stick" or abs(qd) >= STICK_EPS:
            return False
        return abs(tau_s + self.drive_torque(qs, 0.0)) <= self.fc

    def _rk4(self, hh, tau_s):
        q, qd = self.q, self.qd
        if self.is_stuck(q, qd, tau_s):
            self.qd = 0.0                     # 粘滞: θ̇ 必须清零（否则会以残余速度蠕行）
            return
        k1 = self.accel(q, qd, tau_s)
        q2, v2 = q + 0.5 * hh * qd, qd + 0.5 * hh * k1
        k2 = self.accel(q2, v2, tau_s)
        q3, v3 = q + 0.5 * hh * v2, qd + 0.5 * hh * k2
        k3 = self.accel(q3, v3, tau_s)
        q4, v4 = q + hh * v3, qd + hh * k3
        k4 = self.accel(q4, v4, tau_s)
        self.q = q + hh / 6.0 * (qd + 2.0 * v2 + 2.0 * v3 + v4)
        self.qd = qd + hh / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if abs(self.qd) < 1e-9:
            self.qd = 0.0

    def step(self, tau_s, dt):
        n = max(1, int(round(dt / self.int_step)))
        hh = dt / n
        for _ in range(n):
            self._rk4(hh, tau_s)


class Sample:
    __slots__ = ("small_angle", "small_rate", "platform_rate", "gravity_ax",
                 "gravity_ay", "temp_big", "temp_small", "est_valid", "mcu_valid")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, 0.0))

    @property
    def gravity_norm(self):
        return math.hypot(float(self.gravity_ax), float(self.gravity_ay))


class SimRig:
    """虚拟台架。仿真**也遵守"相对偏置"约束**: 控制侧只看到 mapped 读数与 δ。"""

    def __init__(self, truth, gravity_a=(0.0, 0.0), friction="stick", drag=0.0, seed=0,
                 int_step=SIM_INT_STEP, travel=None, accel_limit=ACCEL_LIMIT):
        self.truth = truth
        self.plant = SmallAxisPlant(truth, gravity_a=gravity_a, friction=friction,
                                    drag=drag, int_step=int_step)
        self.zero_offset = float(truth["zero_offset"])
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.frames = 0
        self.travel = travel
        self.limit_hits = 0
        self._omega_ref = 0.0
        self.accel_limit = float(accel_limit)   # 速度源的变化率上限（对应真实速度环）

    def wait_ready(self, timeout_s=15.0):
        return True

    def close(self):
        pass

    def now(self):
        return self.t

    def tick(self, t_phase0, k):
        pass

    def read(self) -> Sample:
        return Sample(small_angle=self.plant.q + self.zero_offset
                      + float(self.rng.normal(0.0, ENC_NOISE)),
                      small_rate=self.plant.qd,
                      platform_rate=self.plant.omega if self.frames else 0.0,
                      gravity_ax=self.plant.gx, gravity_ay=self.plant.gy,
                      temp_big=30, temp_small=30, est_valid=1, mcu_valid=1)

    def send(self, tau_big, tau_small, omega_ref, small_target):
        p = self.plant
        # 速度环不能瞬间改变转速 ⇒ 按 accel_limit 限幅地逼近参考
        step = self.accel_limit * DT
        prev = p.omega
        new = prev + max(-step, min(step, float(omega_ref) - prev))
        p.omega_dot = (new - prev) / DT if self.frames else 0.0
        p.omega = new
        self._omega_ref = float(omega_ref)
        p.step(float(tau_small), DT)
        self.t += DT
        self.frames += 1
        if self.travel is not None:
            lo, hi = self.travel
            if p.q < lo or p.q > hi:
                p.q = min(max(p.q, lo), hi)
                p.qd = 0.0
                self.limit_hits += 1
        return True


class HwRig:
    """真机: 低层 TcbsRobotCommunication，自己独占力矩通道（小 yaw mode=0）。"""

    def __init__(self, mode="host"):
        self.comm = TcbsRobotCommunication()
        self.mode = mode
        self.tx_fail = 0
        self._big_target_angle = 0.0

    def wait_ready(self, timeout_s=15.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            data = self.comm.get_latest_data()
            if data.mcu.valid and data.imu.valid:
                return True
            time.sleep(0.01)
        return False

    def close(self):
        try:
            self.comm.stop()
        finally:
            self.comm.close()

    def now(self):
        return time.perf_counter()

    def tick(self, t_phase0, k):
        target_ns = int(t_phase0 * 1e9) + k * int(round(DT * 1e9))
        while time.perf_counter_ns() < target_ns:
            pass

    def read(self) -> Sample:
        data = self.comm.get_latest_data()
        est = self.comm.get_estimate()
        mcu = data.mcu
        gx = gy = 0.0
        for name in ("gravity_a", "gravity_c"):
            g = getattr(est, name, None)
            if g is not None:
                try:
                    gx, gy = float(g[0]), float(g[1])
                    break
                except (TypeError, IndexError, ValueError):
                    continue
        self._big_target_angle = float(est.big_joint_angle)
        return Sample(small_angle=float(est.small_joint_angle),
                      small_rate=float(est.small_joint_rate),
                      platform_rate=float(est.platform_rate),
                      gravity_ax=gx, gravity_ay=gy,
                      temp_big=int(mcu.yaw_big_temperature),
                      temp_small=int(mcu.yaw_small_temperature),
                      est_valid=int(est.valid), mcu_valid=int(mcu.valid))

    def send(self, tau_big, tau_small, omega_ref, small_target):
        if self.mode == "mcu":
            big_mode, big_vel, big_tau = 1, float(omega_ref), float(tau_big)
        else:
            big_mode, big_vel, big_tau = 0, 0.0, float(tau_big)
        ok = self.comm.send_to_mcu(
            auto_aim_enable=1, fire=0, pitch_target_angle=0.0,
            yaw_big_mode=big_mode, yaw_big_target_angle=float(self._big_target_angle),
            yaw_big_target_velocity=big_vel, yaw_big_torque=big_tau,
            yaw_small_mode=0, yaw_small_target_angle=float(small_target),
            yaw_small_target_velocity=0.0, yaw_small_torque=float(tau_small))
        if not ok:
            self.tx_fail += 1
            if self.tx_fail >= 50:
                raise RuntimeError("连续 50 帧未写入串口 —— 链路断开，停止标定")
        else:
            self.tx_fail = 0
        return bool(ok)


# ============================================================================
# 控制与安全
# ============================================================================
class VelocityPi:
    def __init__(self, kp, ki, out_min, out_max):
        self.kp, self.ki = kp, ki
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0

    def update(self, error, dt):
        out = self.kp * error + self.ki * self.integral
        hi, lo = out > self.out_max, out < self.out_min
        if hi:
            out = self.out_max
        if lo:
            out = self.out_min
        if not ((hi and error > 0) or (lo and error < 0)):
            self.integral += error * dt
        return out

    def reset(self):
        self.integral = 0.0


class Controller:
    def __init__(self, args):
        self.args = args
        self.big_pi = VelocityPi(args.big_kp, args.big_ki, -BIG_TAU_LIMIT, BIG_TAU_LIMIT)
        self.big_lim = TorqueRateLimiter(args.torque_delta)
        self.small_pid = PidController(args.bias_kp, 0.0, 0.0, -BIAS_TORQUE, BIAS_TORQUE,
                                       "small")
        self.small_lim = TorqueRateLimiter(args.torque_delta)
        self.omega_ref = 0.0
        self.small_tau_cap = SMALL_TAU_LIMIT

    def reset(self):
        self.big_pi.reset()
        self.big_lim.clear()
        self.small_pid.reset()
        self.small_lim.clear()
        self.omega_ref = 0.0

    def torques(self, st, omega_target, small_target):
        step = self.args.accel_limit * DT
        self.omega_ref += max(-step, min(step, omega_target - self.omega_ref))
        if self.args.mode == "mcu":
            tau_big = 0.0
        else:
            tau_big = self.big_lim.limit(
                self.big_pi.update(self.omega_ref - st.platform_rate, DT))
        if small_target is None:
            tau_small = self.small_lim.limit(0.0)
        else:
            cap = self.small_tau_cap
            e = wrap_pi(small_target - st.small_angle)
            self.small_pid.integral += e * DT
            out = self.small_pid.kp * e + self.small_pid.ki * self.small_pid.integral
            tau_small = self.small_lim.limit(max(-cap, min(cap, out)))
        return tau_big, tau_small, self.omega_ref


class Safety:
    """标定前的**相对**安全（绝对守卫在标定前无意义）。"""

    def __init__(self, mapped0, args):
        self.args = args
        self.mapped0 = float(mapped0)
        self.last = float(mapped0)
        self.path = 0.0
        self.peak = 0.0

    def note(self, mapped):
        self.path += abs(mapped - self.last)
        self.last = float(mapped)
        self.peak = max(self.peak, abs(mapped - self.mapped0))

    def check(self, st):
        self.note(st.small_angle)
        if abs(st.small_rate) > self.args.rate_abort:
            return f"rate(θ̇={st.small_rate:+.2f} rad/s > {self.args.rate_abort:g})"
        if self.peak > math.radians(self.args.rel_budget_deg):
            return (f"rel_budget(相对起始读数 {deg(self.peak):.1f}° > "
                    f"{self.args.rel_budget_deg:g}°)")
        if self.path > math.radians(self.args.rel_path_budget_deg):
            return (f"rel_path_budget(累计 {deg(self.path):.0f}° > "
                    f"{self.args.rel_path_budget_deg:g}°)")
        return None

    def report(self):
        return (f"相对预算用量: 位移 {deg(self.peak):.2f}°/{self.args.rel_budget_deg:g}°, "
                f"累计 {deg(self.path):.1f}°/{self.args.rel_path_budget_deg:g}°")


def one_step(rig, ctl, args, omega_target, small_target, t_phase0, k, safety=None):
    rig.tick(t_phase0, k)
    st = rig.read()
    if safety is not None:
        reason = safety.check(st)
        if reason:
            return f"safety:{reason}", st
    if max(st.temp_big, st.temp_small) >= args.max_temp:
        return "overheat", st
    tau_big, tau_small, w_ref = ctl.torques(st, omega_target, small_target)
    rig.send(tau_big, tau_small, w_ref, 0.0 if small_target is None else small_target)
    return "ok", st


def run_phase(rig, ctl, args, seconds, omega_target, small_target, safety=None,
              collect=False):
    n = max(1, int(round(seconds * RATE)))
    t0 = rig.now()
    rows = []
    st = None
    for k in range(n):
        status, st = one_step(rig, ctl, args, omega_target, small_target, t0, k, safety)
        if status != "ok":
            return dict(status=status, st=st, rows=rows)
        if collect:
            rows.append((rig.now(), st.small_angle, st.small_rate, st.platform_rate,
                         st.temp_big, st.temp_small, st.gravity_norm))
    return dict(status="ok", st=st, rows=rows)


def bias_move(rig, ctl, args, omega, delta, safety):
    """Ω 稳定后，给小 yaw 一个**相对当前读数**的偏置 δ（低力矩 + 失速检测）。"""
    st = rig.read()
    mapped_now = float(st.small_angle)
    target = mapped_now + delta
    ctl.small_tau_cap = args.bias_torque
    ctl.small_pid.reset()
    t_phase = rig.now()
    n = max(1, int(round(args.bias_sec * RATE)))
    stall_t = 0.0
    stall_ref = mapped_now
    out = dict(ok=False, reason="timeout", achieved=0.0, stalled=False, hard_stop=False)
    for k in range(n):
        status, st = one_step(rig, ctl, args, omega, target, t_phase, k, safety)
        achieved = float(st.small_angle) - mapped_now
        if status != "ok":
            out.update(reason=status, achieved=achieved)
            return out
        err = wrap_pi(target - st.small_angle)
        if abs(err) < args.bias_tol:
            out.update(ok=True, reason="", achieved=achieved)
            return out
        if abs(float(st.small_angle) - stall_ref) < math.radians(args.stall_deg):
            stall_t += DT
        else:
            stall_t = 0.0
            stall_ref = float(st.small_angle)
        if stall_t >= args.stall_sec:
            # 两种失速要分开:
            #  · achieved ≈ 0 ⇒ 完全推不动 ⇒ 疑似**机械限位**（或电控把小 yaw 力矩清零）
            #  · achieved > 0 但到不了目标 ⇒ 只是**弹簧+摩擦的力平衡点**（正常）
            hard = abs(achieved) < math.radians(args.stall_hard_deg)
            out.update(reason="stall", achieved=achieved, stalled=True, hard_stop=hard)
            return out
    out.update(reason="timeout", achieved=float(st.small_angle) - mapped_now)
    return out


def stall_backoff(rig, ctl, args, omega, back, safety):
    """撞限位/失速后的**反向退回**（小力矩、短距离）。"""
    ctl.small_tau_cap = args.bias_torque
    ctl.small_pid.reset()
    st = rig.read()
    run_phase(rig, ctl, args, 1.0, omega, float(st.small_angle) - back, safety)
    return float(rig.read().small_angle)


# ============================================================================
# 一次测量（相对偏置）
# ============================================================================
@dataclass
class RunResult:
    index: int
    direction: int
    approach: str = "C"
    omega_nominal: float = 0.0
    omega_mean: float = 0.0
    start_read: float = 0.0
    bias_cmd: float = 0.0
    bias_achieved: float = 0.0
    bias_ok: bool = True
    release_read: float = 0.0
    theta_mean: float = 0.0
    moved_rad: float = 0.0
    theta_std: float = 0.0
    theta_min: float = 0.0
    theta_max: float = 0.0
    rate_residual: float = 0.0
    settle_time: float = 0.0
    temp_big: float = 0.0
    temp_small: float = 0.0
    gravity_norm: float = 0.0
    stuck: bool = False
    stalled: bool = False
    ok: bool = False
    abort_reason: str = ""


def _settle_and_sample(rig, ctl, args, res, safety, omega, spinning=True, verbose=True):
    """等收敛 → 采样窗口平均 → 相对保持 + 减速（离心/重力两种方法共用）。"""
    t_start = rig.now()
    hold = 0.0
    converged = False
    while rig.now() - t_start < args.settle_timeout:
        r = run_phase(rig, ctl, args, 0.2, omega, None, safety)
        if r["status"] != "ok":
            return bail_from(res, r, "收敛阶段", safety)
        st = r["st"]
        ok_rate = abs(st.small_rate) < args.rate_tol
        ok_omega = (abs(st.platform_rate - omega) < args.omega_tol) if spinning else True
        if ok_rate and ok_omega:
            hold += 0.2
            if hold >= args.settle_sec:
                converged = True
                break
        else:
            hold = 0.0
    res.settle_time = rig.now() - t_start
    if not converged:
        res.abort_reason = f"收敛超时(>{args.settle_timeout:.0f}s)"
        return res
    r = run_phase(rig, ctl, args, args.window_sec, omega, None, safety, collect=True)
    if r["status"] != "ok":
        return bail_from(res, r, "采样窗口", safety)
    th = [row[1] for row in r["rows"]]
    res.theta_mean = float(statistics.fmean(th))
    res.theta_std = float(statistics.pstdev(th)) if len(th) > 1 else 0.0
    res.theta_min, res.theta_max = float(min(th)), float(max(th))
    res.rate_residual = float(statistics.fmean(abs(row[2]) for row in r["rows"]))
    res.omega_mean = float(statistics.fmean(row[3] for row in r["rows"]))
    res.temp_big = r["rows"][-1][4]
    res.temp_small = r["rows"][-1][5]
    res.gravity_norm = r["rows"][-1][6]
    res.moved_rad = abs(res.theta_mean - res.release_read)
    res.stuck = res.moved_rad < math.radians(0.2)
    res.ok = True
    grab = float(r["rows"][-1][1])
    ctl.small_tau_cap = SMALL_TAU_LIMIT
    run_phase(rig, ctl, args, 0.3, omega, grab, safety)
    if spinning:
        run_phase(rig, ctl, args, abs(omega) / args.accel_limit + 0.3, 0.0, grab, safety)
    if verbose:
        log(f"    [{res.direction:+d}{'Ω' if spinning else '重力'} #{res.index}] "
            f"{APPROACH_NAME[res.approach]} 起点={deg(res.start_read):+7.3f}° "
            f"δ={deg(res.bias_cmd):+5.2f}°(达成{deg(res.bias_achieved):+5.2f}°) "
            f"松手={deg(res.release_read):+7.3f}° → 停位={deg(res.theta_mean):+7.3f}°"
            f"(移动{deg(res.moved_rad):5.3f}°{', 卡住' if res.stuck else ''}) "
            f"σ={deg(res.theta_std):.3f}° |g_A|={res.gravity_norm:.3f} "
            f"收敛={res.settle_time:.1f}s")
    return res


def bail_from(res, r, stage, safety):
    st = r.get("st")
    res.abort_reason = f"{stage}:{r['status']}（{safety.report()}）"
    if st is not None:
        res.temp_big, res.temp_small = st.temp_big, st.temp_small
        res.gravity_norm = st.gravity_norm
        res.theta_mean = float(st.small_angle)
    return res


def measure_once(rig, ctl, args, omega, index, direction, rep, theta_hat, rng, safety,
                 verbose=True) -> RunResult:
    """一次测量。``theta_hat is None`` ⇒ 首测（盲松手，不做任何预置）。"""
    res = RunResult(index=index, direction=direction, omega_nominal=omega)
    first = theta_hat is None
    res.start_read = float(rig.read().small_angle)
    ramp_sec = abs(omega) / args.accel_limit + 0.3

    def bail(r, stage):
        st = r.get("st")
        res.abort_reason = f"{stage}:{r['status']}（{safety.report()}）"
        if st is not None:
            res.temp_big, res.temp_small = st.temp_big, st.temp_small
            res.gravity_norm = st.gravity_norm
            res.theta_mean = float(st.small_angle)
        return res

    ctl.reset()
    spinning = (args.method_used == "centrifugal")
    # ── 0) 重力法: 大 yaw **不动**（Ω=0）, 只靠静态倾角的 g_A 把载荷"挂"到最低点 ──
    if not spinning:
        ctl.reset()
        r = run_phase(rig, ctl, args, 0.5, 0.0, None, safety)
        if r["status"] != "ok":
            return bail(r, "稳定阶段")
        # 记录倾角（由 IMU 实测, 不是假定值）
        res.gravity_norm = r["st"].gravity_norm
        st_hold = r["st"]
        # 偏置（非首测）: Ω=0 下没有离心弹簧, 只需克服重力刚度 k_grav + fc
        if not first:
            delta = float(rng.uniform(args.bias_lo, args.bias_hi))
            side = 1.0 if (args.one_side or rep % 2 == 0) else -1.0
            res.bias_cmd = delta * side
            res.approach = "R" if side > 0 else "L"
            mv = bias_move(rig, ctl, args, 0.0, res.bias_cmd, safety)
            res.bias_achieved = mv["achieved"]
            res.bias_ok = bool(mv["ok"] or abs(mv["achieved"]) > 0.3 * abs(res.bias_cmd))
            if mv["stalled"] and mv["hard_stop"]:
                res.stalled = True
                stall_backoff(rig, ctl, args, 0.0, abs(res.bias_cmd), safety)
                res.abort_reason = "bias_stall(完全推不动: 疑似撞限位)"
                return res
        # 松手 → 等收敛 → 采样（与离心法共用后半段）
        r = run_phase(rig, ctl, args, 0.3, 0.0, None, safety)
        if r["status"] != "ok":
            return bail(r, "松手阶段")
        res.release_read = float(r["st"].small_angle)
        return _settle_and_sample(rig, ctl, args, res, safety, omega=0.0, spinning=False,
                                  verbose=verbose)
    # ── 1) 松手（首测直接松手; 之后也是先松手, 偏置在 Ω 稳定后才加）──
    if args.spinup_hold == "hold":
        ctl.small_tau_cap = SMALL_TAU_LIMIT          # 相对保持（目标=当前读数, 非绝对）
        r = run_phase(rig, ctl, args, ramp_sec, omega, res.start_read, safety)
    else:
        r = run_phase(rig, ctl, args, ramp_sec, omega, None, safety)
    if r["status"] != "ok":
        return bail(r, "加速阶段")
    # ── 2) 等 Ω 稳定（保持模式下继续相对保持）──
    t0 = rig.now()
    omega_ok = False
    while rig.now() - t0 < 5.0:
        tgt = None
        if args.spinup_hold == "hold":
            tgt = float(rig.read().small_angle)
        r = run_phase(rig, ctl, args, 0.1, omega, tgt, safety)
        if r["status"] != "ok":
            return bail(r, "恒定Ω等待")
        if abs(r["st"].platform_rate - omega) < args.omega_tol:
            omega_ok = True
            break
    if not omega_ok:
        res.abort_reason = "未达到恒定 Ω"
        return res
    # ── 3) ★ Ω 稳定之后才施加相对偏置（首测跳过）──
    if not first:
        delta = float(rng.uniform(args.bias_lo, args.bias_hi))
        side = 1.0 if (args.one_side or rep % 2 == 0) else -1.0
        res.bias_cmd = delta * side
        res.approach = "R" if side > 0 else "L"
        mv = bias_move(rig, ctl, args, omega, res.bias_cmd, safety)
        res.bias_achieved = mv["achieved"]
        res.bias_ok = bool(mv["ok"] or abs(mv["achieved"]) > 0.3 * abs(res.bias_cmd))
        if mv["stalled"] and mv["hard_stop"]:
            res.stalled = True
            log(f"    [{direction:+d}Ω #{index}] [STALL] 加了力矩但**完全没动** ⇒ 疑似撞机械"
                f"限位，或**电控把小 yaw 力矩清零**（请确认已旁路电控小 yaw 限位保护）"
                f" → 反向退回")
            stall_backoff(rig, ctl, args, omega, abs(res.bias_cmd), safety)
            res.abort_reason = "bias_stall(完全推不动: 疑似撞限位/电控保护清零力矩)"
            return res
        if mv["stalled"]:
            res.stalled = True
            log(f"    [{direction:+d}Ω #{index}] 偏置未到目标（达成 "
                f"{deg(mv['achieved']):+.2f}°/目标{deg(res.bias_cmd):+.2f}°）: 稳态 Ω 下"
                f"的**弹簧+摩擦力平衡点**（正常现象）⇒ 就从这里松手测")
        elif mv["reason"] not in ("", "ok"):
            res.abort_reason = f"bias_move:{mv['reason']}"
            return res
    # ── 4) 松手 ──
    r = run_phase(rig, ctl, args, 0.3, omega, None, safety)
    if r["status"] != "ok":
        return bail(r, "松手阶段")
    res.release_read = float(r["st"].small_angle)
    return _settle_and_sample(rig, ctl, args, res, safety, omega=omega, spinning=True,
                              verbose=verbose)


# ============================================================================
# 一轮标定
# ============================================================================
def run_calibration(rig, omega, repeats, args, seed, verbose=True, directions=(+1, -1),
                    mapped0=None, tilt_list=None) -> dict:
    """跑一轮标定。

    · 离心法: ``directions`` = ±Ω（Ω 符号不改变平衡点; ±平均消 Ω 奇次项如阻力）;
    · 重力法: 大 yaw 不动（Ω=0），``tilt_list`` = 多个**倾角大小**（平衡点与 |g_A| 无关,
      所以不同倾角测的是**同一个物理点**, 而它们的粘滞带 fc/(|P|g sinφ) 不同 ⇒
      多倾角平均正好消掉粘滞带偏差, 起到与 ±Ω 完全对应的作用，而且没有空气阻力）。
    """
    ctl = Controller(args)
    rng = np.random.default_rng(seed)
    if mapped0 is None:
        mapped0 = float(rig.read().small_angle)
    safety = Safety(mapped0, args)
    grav = (args.method_used == "gravity")
    passes = list(tilt_list) if (grav and tilt_list) else [0.0]
    log(f"\n=== 标定({args.method_used}): "
        + (f"倾角 {passes}°（大 yaw 不动 Ω=0）" if grav
           else f"Ω=±{omega:g} rad/s（上限 π）")
        + f", 每轮 {repeats} 次, 偏置={'单侧' if args.one_side else '两侧交替'} ===")
    log(f"  起始读数 mapped0 = {deg(mapped0):+.3f}°（**绝对含义未知**, 仅作相对基准）")
    log(f"  相对预算: 位移 {args.rel_budget_deg:g}°, 累计 {args.rel_path_budget_deg:g}°")
    runs: list[RunResult] = []
    theta_hat = None
    idx = 0
    for tilt in passes:
        if grav:
            set_tilt(rig, tilt)          # 仿真: 直接设定; 真机: 提示操作者(见 set_tilt)
        label = (f"倾角 φ={tilt:g}°" if grav else
                 f"方向 {'+Ω' if tilt > 0 else '−Ω'} (Ω={tilt * omega if tilt else omega:+.2f} rad/s)")
        log(f"  {label}:")
        for rep in range(repeats):
            om = 0.0 if grav else tilt * omega
            res = measure_once(rig, ctl, args, om, idx, 1 if tilt >= 0 else -1, rep,
                              theta_hat, rng, safety, verbose=verbose)
            if not res.ok:
                log(f"    [#{idx}] 失败: {res.abort_reason}")
            runs.append(res)
            idx += 1
            if res.ok:
                theta_hat = float(statistics.fmean([r.theta_mean for r in runs if r.ok]))
    ok = [r for r in runs if r.ok]
    left = [r.theta_mean for r in ok if r.approach == "L"]
    right = [r.theta_mean for r in ok if r.approach == "R"]
    blind = [r.theta_mean for r in ok if r.approach == "C"]

    def stat(xs):
        if not xs:
            return dict(n=0, mean=float("nan"), std=float("nan"), min=float("nan"),
                        max=float("nan"))
        return dict(n=len(xs), mean=float(statistics.fmean(xs)),
                    std=float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
                    min=float(min(xs)), max=float(max(xs)))

    s_l, s_r, s_b = stat(left), stat(right), stat(blind)
    pos = stat([r.theta_mean for r in ok if r.direction > 0])
    neg = stat([r.theta_mean for r in ok if r.direction < 0])
    all_vals = [r.theta_mean for r in ok]
    total_mean = float(statistics.fmean(all_vals)) if all_vals else float("nan")
    total_std = float(statistics.pstdev(all_vals)) if len(all_vals) > 1 else 0.0
    # ── 重力法: 换算到"水平旋转平衡点"（与离心法同一个物理点）──
    level_equiv = float("nan")
    tilt_shift = float("nan")
    if grav:
        gd = rig_gravity_direction(rig)
        tilt_shift = wrap_pi(angle_of(gd) - angle_of_d(args))
        if ok:
            eq = [wrap_pi(r.theta_mean - tilt_shift) for r in ok]
            level_equiv = float(statistics.fmean(eq))
    return dict(omega=omega, repeats=repeats, runs=runs, ok=ok, left=s_l, right=s_r,
                blind=s_b, pos=pos, neg=neg, total_mean=total_mean, total_std=total_std,
                level_equiv=level_equiv, tilt_shift=tilt_shift, method=args.method_used,
                single_side=(0.5 * (s_r["mean"] - s_l["mean"]) if (s_l["n"] and s_r["n"])
                             else float("nan")),
                omega_gap=(abs(pos["mean"] - neg["mean"]) if (pos["n"] and neg["n"])
                           else float("nan")),
                band_emp=(max(all_vals) - min(all_vals)) if all_vals else float("nan"),
                n_stuck=sum(1 for r in ok if r.stuck),
                n_bias_bad=sum(1 for r in ok if not r.bias_ok),
                mapped0=mapped0, safety=safety)


def angle_of(v) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def rig_gravity_direction(rig) -> tuple:
    st = rig.read()
    return (float(st.gravity_ax), float(st.gravity_ay))


def angle_of_d(args) -> float:
    """d 的方向（实测几何; 取 +x, 可用 --d-angle-deg 覆盖）。"""
    return math.radians(getattr(args, "d_angle_deg", 0.0))


def set_tilt(rig, tilt_deg) -> None:
    """设/提示倾角。仿真: 直接改 plant 的 g_A（保持方向 = d 方向, 只改大小以免平衡点跑出行程）;
    真机: 打印提示要求操作者把底盘摆到该倾角（脚本从 IMU 读实测值）。"""
    if isinstance(rig, SimRig):
        g = GRAVITY_G * math.sin(math.radians(tilt_deg))
        psi = math.radians(getattr(rig, "g_psi_deg", 20.0))
        rig.plant.gx, rig.plant.gy = g * math.cos(psi), g * math.sin(psi)
        # g_A 方向取 20°（故意不沿 d）⇒ 平衡点与"水平旋转平衡点"不同 ⇒ 真正走一遍换算
    else:
        log(f"    [操作] 请把底盘静态倾斜到 φ≈{tilt_deg:g}°（把小 yaw 轴放倒接近水平）;"
            f" 脚本会从 IMU 读实测 |g_A| 核对")


def to_deg(stat_d):
    out = dict(stat_d)
    for k in ("mean", "std", "min", "max"):
        v = out.get(k)
        if isinstance(v, float) and not math.isnan(v):
            out[k + "_deg"] = deg(v)
    return out


# ============================================================================
# 标定后自检: 探针找两侧机械停止位
# ============================================================================
def place_sim(rig, args, target_mapped, rng):
    """--sim manual: 用"人手"把 plant 摆到目标位置（含摆放散布 + 扶稳抖动）。"""
    if isinstance(rig, SimRig):
        err = float(rng.normal(0.0, math.radians(args.sim_placement_sd_deg)))
        rig.plant.q = (target_mapped + err) - rig.zero_offset
        rig.plant.qd = 0.0
        rig.manual_tremor = math.radians(args.sim_tremor_deg)


def run_manual(rig, args, seed) -> dict:
    """★ manual 模式: 人工把小 yaw 摆到"准确的零点"位置, 读编码器即可。

    · 小 yaw **力矩恒 0**（松手） —— 工具不做任何自动运动、不加任何标定力;
    · 大 yaw 保持在当前位置（`--manual-hold-big hold` 低增益速度环守住, 或 free 完全松手）;
    · 采样 `est.small_joint_angle` 一个窗口（--window-sec）; 窗口内**极差 ≤ --manual-tol-deg**
      才接受, 否则提示"手还在动/没扶稳, 重来"; 角速度超阈值也会提示;
    · 重复 --repeats 次, 每次请操作者**重新摆放** ⇒ 报告均值/标准差/极差
      = **人工摆放的可重复性**（这个数直接决定零点精度上限）。
    """
    rng = np.random.default_rng(seed)
    ctl = Controller(args)
    ctl.small_tau_cap = 0.0                     # ★ 小 yaw 力矩恒 0
    st0 = rig.read()
    mapped0 = float(st0.small_angle)
    big_target = float(st0.platform_azimuth) if hasattr(st0, "platform_azimuth") else 0.0
    log("=" * 100)
    log("小 yaw 零点捕获（**manual 模式: 人工摆位 + 读编码器**）")
    log(f"  小 yaw: **力矩恒 0（松手）**; 大 yaw: {'低增益守住当前位置' if args.manual_hold_big == 'hold' else '完全松手（力矩 0）'}")
    log(f"  采样窗口 {args.window_sec:g}s（{int(round(args.window_sec * RATE))} 点 @100Hz）, "
        f"稳定性判据: 窗口内极差 ≤ {args.manual_tol_deg:g}°")
    log(f"  重复 {args.repeats} 次, **每次都请重新摆放**（先推离再摆回）")
    log(f"  起始读数（绝对含义未知, 仅作相对基准）: {deg(mapped0):+.3f}°")
    log(f"  ★ 工具不做任何自动运动; 唯一保护是机械限位与人手; |θ̇|>{args.rate_abort:g} rad/s 会提示'被外力带动'")
    if args.note:
        log(f"  备注（工装/基准）: {args.note}")
    log("=" * 100)

    rows = []
    for rep in range(args.repeats):
        if isinstance(rig, SimRig):
            place_sim(rig, args, float(args._manual_true_mapped), rng)
        else:
            log(f"  [操作 {rep + 1}/{args.repeats}] 请把小 yaw 摆到**零点位置**并扶稳…")
        accepted = False
        for attempt in range(args.manual_tries):
            st = rig.read()
            log(f"    当前读数 {deg(st.small_angle):+8.3f}°（相对起点 "
                f"{deg(st.small_angle - mapped0):+7.3f}°）采样中…")
            t_phase = rig.now()
            vals, rates = [], []
            n = max(1, int(round(args.window_sec * RATE)))
            aborted = False
            for k in range(n):
                status, st_k = one_step(rig, ctl, args, 0.0 if args.manual_hold_big == "hold"
                                        else None, None, t_phase, k, None)
                if isinstance(rig, SimRig) and getattr(rig, "manual_tremor", 0.0):
                    rig.plant.q += float(rng.normal(0.0, rig.manual_tremor / 3.0))
                vals.append(float(st_k.small_angle))
                rates.append(abs(float(st_k.small_rate)))
                if abs(float(st_k.small_rate)) > args.rate_abort:
                    log(f"    [WARN] |θ̇|={abs(float(st_k.small_rate)):.2f} rad/s > "
                        f"{args.rate_abort:g} ⇒ **被外力带动**, 本次作废重摆")
                    aborted = True
                    break
            if aborted:
                continue
            mean = float(statistics.fmean(vals))
            p2p = (max(vals) - min(vals))
            rate_max = max(rates)
            ok = p2p <= math.radians(args.manual_tol_deg) and rate_max <= args.rate_tol
            log(f"    窗口: 均值={deg(mean):+8.3f}° σ={deg(statistics.pstdev(vals)):.4f}° "
                f"极差={deg(p2p):.4f}° |θ̇|max={rate_max:.3f} rad/s ⇒ "
                f"{'**接受**' if ok else '**不接受**'}")
            if not ok:
                log(f"    [重来] {'手还在动/没扶稳' if p2p > math.radians(args.manual_tol_deg) else '角速度超阈值'}"
                    f"（极差需 ≤ {args.manual_tol_deg:g}°）—— 请扶稳后重摆")
                if isinstance(rig, SimRig):
                    place_sim(rig, args, float(args._manual_true_mapped), rng)
                continue
            rows.append(dict(index=rep, ok=True, mapped_mean=mean,
                             mapped_std=float(statistics.pstdev(vals)), mapped_p2p=p2p,
                             rate_max=rate_max, temp_big=st_k.temp_big,
                             temp_small=st_k.temp_small, gravity_norm=st_k.gravity_norm,
                             note=args.note))
            accepted = True
            break
        if not accepted:
            rows.append(dict(index=rep, ok=False, mapped_mean=float("nan"), mapped_std=0.0,
                             mapped_p2p=float("nan"), rate_max=float("nan"), temp_big=0.0,
                             temp_small=0.0, gravity_norm=0.0, note="未扶稳, 放弃"))
            log(f"    [#{rep}] 多次重摆仍未稳定 ⇒ 本次放弃")
    okr = [r for r in rows if r["ok"]]
    vals = [r["mapped_mean"] for r in okr]
    mapped_zero = float(statistics.fmean(vals)) if vals else float("nan")
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    rng_deg = (max(vals) - min(vals)) if vals else float("nan")
    delta = -mapped_zero
    log("\n" + "-" * 84)
    log(f"manual 零点捕获结果: 接受 {len(okr)}/{args.repeats} 次")
    log(f"  ★ mapped_zero（窗口均值）= {deg(mapped_zero):+.4f}° = {mapped_zero:+.6f} rad")
    log(f"  ★ 人工摆放可重复性: σ = {deg(std):.4f}°  极差 = {deg(rng_deg):.4f}°"
        f"  （**这个数直接决定零点精度上限**）")
    log(f"  ★ Δoffset = −mapped_zero = {delta:+.6f} rad = {deg(delta):+.4f}°")
    for r in rows:
        log(f"    第 {r['index'] + 1} 次: {'接受' if r['ok'] else '放弃'} "
            f"mapped={deg(r['mapped_mean']):+8.4f}° 极差={deg(r['mapped_p2p']):.4f}° "
            f"|θ̇|max={r['rate_max']:.3f} 温度=({r['temp_big']:.0f},{r['temp_small']:.0f})℃")
    gr = next((r["gravity_norm"] for r in okr), 0.0)
    log(f"  底盘 |gravity_a| = {gr:.4f} m/s²（manual 模式对倾角不敏感: 只要摆位与标定工况一致）")
    offset_old = args.offset_old
    if offset_old is None:
        try:
            from torque_controller import default_linear_params
            offset_old = float(default_linear_params().recv_small_yaw_offset)
        except Exception:
            offset_old = 0.0
            log("  [WARN] 读不到 default_linear_params ⇒ offset_old 记 0（可用 --offset-old）")
    if not args.report_only:
        log("  编码器零位建议（mapped = raw*scale + offset）:")
        log(f"    offset_old = {offset_old:+.6f} rad ⇒ offset_new = "
            f"{offset_new_val(offset_old, delta):+.6f} rad（Δoffset={delta:+.6f} rad）")
        log("    ★ 写回之后**全系统就在这个新零点坐标系里做绝对控制**: MPC 的小 yaw 限位 "
            "[−30°,+30°]、回中中心 0°、电控 [−28°,+28°] 夹取、所有目标/参考都用新零点。")
        log("    ★ **Δoffset 写错 = 限位对应的物理位置整体平移**（写大 3° ⇒ 真实可达行程变成"
            "一边 28°、另一边 17°）⇒ 写回后必须复核限位！")
        log("    ★ 标定脚本自身**不做绝对定位**（写 offset 之前坐标系无意义）。")
    else:
        log("  [--report-only] 只测量、不写 offset 建议")
    log("  ★ 重要: **manual 零点与 P 的方向无关** —— 所以这里**不给出** P 方向结论。")
    log("     若想知道 P 的方向，需要另做**离心/重力实验**测平衡点读数 θ*_meas，此时")
    log("     P = |P|·R(−θ*_meas)·d̂; **手动零点下这个角度未知 ⇒ 辨识时不要假设 Py = 0**。")
    log("-" * 84)

    payload = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                   mode_desc="manual（人工摆位捕获零点）", method="manual",
                   report_only=bool(args.report_only), repeats=args.repeats,
                   window_sec=args.window_sec, manual_tol_deg=args.manual_tol_deg,
                   manual_tries=args.manual_tries,
                   manual_hold_big=args.manual_hold_big, note=args.note,
                   mapped0_deg=deg(mapped0), runs=rows,
                   summary=dict(n_ok=len(okr), mapped_zero_rad=mapped_zero,
                                mapped_zero_deg=deg(mapped_zero),
                                repeatability_std_deg=deg(std),
                                repeatability_range_deg=deg(rng_deg)),
                   level_check=dict(gravity_norm=gr, tol=GRAVITY_TOL,
                                    ok=bool(gr < GRAVITY_TOL)),
                   P_direction_note=("manual 零点与 P 方向无关; 要得到 P 方向须另做离心/重力"
                                     "实验测 θ*_meas, 再 P=|P|·R(−θ*_meas)·d̂; "
                                     "手动零点下角度未知 ⇒ 辨识时不要假设 Py=0"),
                   zero_offset=(None if args.report_only else dict(
                       offset_old_rad=offset_old, mapped_zero_rad=mapped_zero,
                       delta_offset_rad=delta, delta_offset_deg=deg(delta),
                       offset_new_rad=offset_new_val(offset_old, delta))))
    if args._manual_true_mapped is not None:
        payload["sim_truth"] = dict(
            true_mapped_zero_deg=deg(args._manual_true_mapped),
            est_mapped_zero_deg=deg(mapped_zero),
            capture_error_deg=deg(mapped_zero - args._manual_true_mapped),
            injected_placement_sd_deg=args.sim_placement_sd_deg,
            injected_tremor_p2p_deg=args.sim_tremor_deg)
        log(f"  [仿真真值] 真实 mapped 零点 = {deg(args._manual_true_mapped):+.4f}° ⇒ "
            f"捕获误差 = {deg(mapped_zero - args._manual_true_mapped):+.4f}°; "
            f"注入的摆放散布 = {args.sim_placement_sd_deg:g}°（σ）")
    out_json = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    csv_path = os.path.splitext(out_json)[0] + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["index", "ok", "mapped_mean", "mapped_std", "mapped_p2p", "rate_max",
                    "temp_big", "temp_small", "gravity_norm", "note"])
        for r in rows:
            w.writerow([int(r["index"]), int(bool(r["ok"])), f"{r['mapped_mean']:.6f}",
                        f"{r['mapped_std']:.6f}", f"{r['mapped_p2p']:.6f}",
                        f"{r['rate_max']:.4f}" if r["rate_max"] == r["rate_max"] else "",
                        f"{r['temp_big']:.0f}", f"{r['temp_small']:.0f}",
                        f"{r['gravity_norm']:.4f}", r["note"]])
    log(f"  写出: {out_json}")
    log(f"        {csv_path}")
    return payload


def probe_travel(rig, ctl, args, mapped_eq, scale, safety) -> dict:
    """低力矩 + 失速检测找两侧机械停止位（编码器读数），做行程/scale 自洽检查。

    ★ 探针**不受标定阶段的相对预算约束**（那个预算是"标定前不知道绝对位置"时的保护）:
      标定完成后我们已经知道平衡点在哪，探针用的是它自己的两道保护 ——
      ``--probe-max-deg``（单侧最大行程）+ 失速检测（撞上就停）。
    """
    log("\n=== 标定后自检（--probe-travel）: 低力矩 + 失速检测找机械停止位 ===")
    st0 = rig.read()
    if st0.gravity_norm > GRAVITY_TOL:
        log(f"  [WARN] 当前底盘不水平（|g_A|={st0.gravity_norm:.3f} m/s²）: 重力会把小 yaw"
            f" 拉向重力平衡点, 探针会把'推不动'误判成机械停止位 ⇒ **请先把底盘放平**"
            f"（探针必须在 |g_A|≈0 且大 yaw 静止时做）")
    saved = (args.rel_budget_deg, args.rel_path_budget_deg)
    args.rel_budget_deg = args.probe_max_deg + 5.0
    args.rel_path_budget_deg = 4.0 * args.probe_max_deg + 50.0
    p_safety = Safety(mapped_eq, args)
    ctl.reset()
    ctl.small_tau_cap = SMALL_TAU_LIMIT                    # 停车阶段先守住小 yaw
    st = rig.read()
    spin_down = min(60.0, abs(st.platform_rate) / args.accel_limit + 1.0)
    run_phase(rig, ctl, args, spin_down, 0.0, float(st.small_angle), p_safety)
    ctl.small_tau_cap = args.probe_torque
    run_phase(rig, ctl, args, 2.0, 0.0, mapped_eq, p_safety)      # 再回到平衡点附近
    stops, stalled_map = {}, {}
    for sign, name in ((+1.0, "+"), (-1.0, "-")):
        target = mapped_eq + sign * math.radians(args.probe_max_deg)
        ctl.small_pid.reset()
        stalled = False
        stall_t = 0.0
        ref = float(rig.read().small_angle)
        t_phase = rig.now()
        n = max(1, int(round(args.probe_sec * RATE)))
        for k in range(n):
            status, st = one_step(rig, ctl, args, 0.0, target, t_phase, k, p_safety)
            if status != "ok":
                log(f"  探针{name}侧中止: {status}")
                break
            if abs(float(st.small_angle) - mapped_eq) > math.radians(args.probe_max_deg):
                break                                   # 到探针行程上限
            if abs(float(st.small_angle) - ref) < math.radians(args.stall_deg):
                stall_t += DT
            else:
                stall_t, ref = 0.0, float(st.small_angle)
            if stall_t >= args.stall_sec:               # ★ 加了力却不动 ⇒ 机械停止位
                stalled = True
                break
        stops[name], stalled_map[name] = float(rig.read().small_angle), stalled
        log(f"  探针{name}侧: 停止位读数 = {deg(stops[name]):+8.3f}°"
            f"{'（失速 ⇒ 判定为机械停止位）' if stalled else '（未失速: 到探针上限）'}")
        stall_backoff(rig, ctl, args, 0.0, math.radians(1.0), p_safety)
    w_deg = deg(abs(stops["+"] - stops["-"]) * scale)
    d_plus = deg((stops["+"] - mapped_eq) * scale)
    d_minus = deg((mapped_eq - stops["-"]) * scale)
    span = deg(SMALL_TRAVEL_MAX - SMALL_TRAVEL_MIN)
    implied = (span / w_deg * scale) if w_deg else float("nan")
    ok_scale = abs(w_deg - span) <= args.probe_scale_tol_deg
    ok_dist = (abs(d_plus - deg(SMALL_TRAVEL_MAX)) <= args.probe_scale_tol_deg
               and abs(d_minus + deg(SMALL_TRAVEL_MIN)) <= args.probe_scale_tol_deg)
    log(f"  行程宽度 W = {w_deg:.2f}°（应 ≈{span:.1f}° ⇒ scale 校核"
        f"{'通过' if ok_scale else '**不通过**'}）")
    log(f"  平衡点到两侧: +{d_plus:.2f}°（应 ≈{deg(SMALL_TRAVEL_MAX):.1f}°）, "
        f"−{d_minus:.2f}°（应 ≈{-deg(SMALL_TRAVEL_MIN):.1f}°） ⇒ "
        f"{'自洽' if ok_dist else '**不自洽**'}")
    if not ok_scale and w_deg:
        log(f"  ⚠ 建议 scale 修正: {scale:g} → {implied:.6g}（= 旧 scale × {span:.1f}/W）")
    if not (ok_scale and ok_dist):
        log("  ⚠ 结论: 零点或行程假设可能有问题 —— 核对 P 方向/装配, 或改用外部角度基准")
    args.rel_budget_deg, args.rel_path_budget_deg = saved
    return dict(stops=stops, stalled=stalled_map, width_deg=w_deg, d_plus_deg=d_plus,
                d_minus_deg=d_minus, implied_scale=implied, ok_scale=bool(ok_scale),
                ok_dist=bool(ok_dist))


# ============================================================================
# 报告与落盘
# ============================================================================
def report_and_save(calib, args, mode_desc, out_json, truth=None, extra=None,
                    probe=None) -> dict:
    s_l, s_r, s_b = calib["left"], calib["right"], calib["blind"]
    total_mean = calib["total_mean"]
    log("\n" + "-" * 84)
    log(f"标定结果（{mode_desc}）")
    if s_b["n"]:
        log(f"  首次盲松手: n={s_b['n']} 读数均值={deg(s_b['mean']):+.3f}°")
    if s_l["n"]:
        log(f"  ★ {APPROACH_NAME['L']}: n={s_l['n']} 均值={deg(s_l['mean']):+.3f}° "
            f"σ={deg(s_l['std']):.3f}°")
    if s_r["n"]:
        log(f"  ★ {APPROACH_NAME['R']}: n={s_r['n']} 均值={deg(s_r['mean']):+.3f}° "
            f"σ={deg(s_r['std']):.3f}°")
    log(f"  ★ 两侧合并(mapped_eq) = {deg(total_mean):+.4f}° = {total_mean:+.6f} rad")
    if calib["pos"]["n"] and calib["neg"]["n"]:
        log(f"  ★ 仅+Ω = {deg(calib['pos']['mean']):+.4f}°  仅−Ω = "
            f"{deg(calib['neg']['mean']):+.4f}°  **±Ω合并** = {deg(total_mean):+.4f}°")
    log(f"  ★ 单侧偏差 (右−左)/2 = {deg(calib['single_side']):+.4f}°"
        f"（理论 fc/(Ω²|d||P|); Ω 偶函数 ⇒ 靠两侧交替消）")
    log(f"  ★ ±Ω 差 = {deg(calib['omega_gap']):+.4f}°（Ω 奇次项——如空气阻力——体现在这里;"
        f" ±Ω 平均消掉它）")
    log(f"     两侧合并σ = {deg(calib['total_std']):.4f}°  实测极差 = "
        f"{deg(calib['band_emp']):.4f}°")
    n_ok = len(calib["ok"])
    log(f"  被推动的测量: {n_ok - calib['n_stuck']}/{n_ok}（卡住 {calib['n_stuck']}）; "
        f"偏置未达成: {calib['n_bias_bad']}")
    if calib["n_stuck"] > max(1, n_ok // 2):
        log("    [WARN] 多数测量没推动小 yaw ⇒ 估计值≈偏置起点平均, **信息量低** —— "
            "提高 Ω（单侧偏差 ∝ 1/Ω²）或加大 --bias-deg")
    log(f"  {calib['safety'].report()}")
    gr = next((r.gravity_norm for r in calib["ok"]), 0.0)
    log(f"  ★ **卡住比例 = {calib['n_stuck']}/{n_ok}"
        f"（{100.0 * calib['n_stuck'] / max(1, n_ok):.0f}%）**"
        f" —— 卡住=没被推动 ⇒ 读数就是起点, **误差小是假象、信息量为零**")
    method = calib.get("method", args.method_used)
    lev = calib.get("level_equiv", float("nan"))
    theta_meas = total_mean
    if method == "gravity":
        log(f"  ★ 倾角换算: 本工况平衡点比'水平旋转平衡点'偏了 "
            f"{deg(calib.get('tilt_shift', 0.0)):+.2f}°（= angle(g_A) − angle(d)，已知量）")
        log(f"  ★ θ*_meas(本工况读数) = {deg(theta_meas):+.4f}°;  "
            f"**水平等效 θ*_level = {deg(lev):+.4f}°**（与离心法同一物理点）")
    else:
        log(f"  ★ θ*_meas(读数) = {deg(theta_meas):+.4f}°（离心法本来就是水平旋转平衡点）")
        lev = theta_meas
    log("  ★ 消误差三件套的贡献:")
    log(f"     ① 两侧交替逼近: 单侧偏差 (右−左)/2 = {deg(calib['single_side']):+.4f}°"
        f"（≥ 理论上界 fc/k 的一部分）⇒ 消掉**粘滞带/逼近侧**这一类（Ω 偶函数）")
    if method == "gravity":
        log(f"     ② 多倾角平均 {args.tilt_list}: 各倾角的粘滞带 fc/(|P|g·sinφ) 不同 "
            f"⇒ 平均掉粘滞带偏差; **重力法没有空气阻力项**（Ω=0 ⇒ 无 Ω 奇次项）")
    else:
        log(f"     ② ±Ω 双向: ±Ω 差 = {deg(calib['omega_gap']):+.4f}°"
            f"（未建模 Ω 奇次项——如空气阻力——体现在这里）⇒ 消掉**阻力**这一类")
    log(f"     ③ 水平/倾角校核: |g_A|={gr:.4f} m/s²（{'水平' if gr < GRAVITY_TOL else '倾斜'}）"
        f" —— 重力偏移是 Ω 偶函数、**任何平均都消不掉**, 只能靠放平")
    if truth is not None:
        th_true = theta_star(truth["dx"], truth["dy"], truth["Px"], truth["Py"]) \
            + truth["zero_offset"]
        msg = f"  [仿真真值] 平衡点读数真值 = {deg(th_true):+.3f}°"
        if calib["pos"]["n"] and calib["neg"]["n"]:
            msg += (f" ⇒ ★双向合并误差 = {deg(total_mean - th_true):+.3f}°"
                    f"（仅+Ω {deg(calib['pos']['mean'] - th_true):+.3f}°, "
                    f"仅−Ω {deg(calib['neg']['mean'] - th_true):+.3f}°）")
        log(msg)
    gr = next((r.gravity_norm for r in calib["ok"]), 0.0)
    log(f"  底盘水平检查: |gravity_a| = {gr:.4f} m/s² (阈值 {GRAVITY_TOL:g})")

    offset_old, scale = args.offset_old, args.scale
    if offset_old is None and not args.report_only:
        try:
            from torque_controller import default_linear_params
            lp = default_linear_params()
            offset_old = float(lp.recv_small_yaw_offset)
            scale = float(lp.recv_small_yaw_scale)
        except Exception:
            log("  [WARN] 读不到 default_linear_params ⇒ offset_old 记 0（可用 --offset-old）")
            offset_old = 0.0
    if scale is None:
        scale = 1.0
    delta = -total_mean
    # ── 零点建议 / 手动零点 ──
    theta_meas_new = total_mean                      # 本工况读数
    theta_ref = (lev if (args.zero_target == "level" and lev == lev) else theta_meas_new)
    delta_suggest = -theta_ref
    if args.zero_deg is not None:
        log(f"  ★ 手动零点: --zero-deg={args.zero_deg:+g}° ⇒ **用这个值**, 不用标定建议值")
        log(f"    （= 用户直接声明把零点挪 {args.zero_deg:+g}°; 标定只提供 θ*_meas 作为参考）")
    if args.report_only:
        log("  [--report-only] 只测量、不写 offset 建议（先确认平衡点可达）")
        log(f"    θ*_meas(本工况读数) = {deg(theta_meas_new):+.4f}°"
            + (f"   θ*_level(水平等效) = {deg(lev):+.4f}°" if lev == lev and method == "gravity"
               else ""))
        theta_after = theta_meas_new
    else:
        delta = math.radians(args.zero_deg) if args.zero_deg is not None else delta_suggest
        theta_after = wrap_pi(theta_meas_new + delta)   # 新坐标系里平衡点的读数
        log("  编码器零位建议（mapped = raw*scale + offset）:")
        if args.zero_deg is None:
            log(f"    ★ Δoffset(建议) = {delta:+.6f} rad = {deg(delta):+.4f}°"
                f"（= −θ*_{args.zero_target}）")
        else:
            log(f"    ★ Δoffset(手动) = {delta:+.6f} rad = {deg(delta):+.4f}°"
                f"（标定建议值本来会是 {delta_suggest:+.6f} rad = {deg(delta_suggest):+.4f}°）")
        log(f"    offset_old = {offset_old:+.6f} rad ⇒ offset_new = "
            f"{offset_new_val(offset_old, delta):+.6f} rad")
        log("    ★ 写回之后**全系统就在这个新零点坐标系里做绝对控制**: MPC 的小 yaw 限位 "
            "[−30°,+30°]、回中中心 0°、电控 [−28°,+28°] 夹取、所有目标/参考都用新零点。")
        log("    ★ **Δoffset 写错 = 限位对应的物理位置整体平移**（写大 3° ⇒ 真实可达行程变成"
            "一边 28°、另一边 17°）⇒ 写回后必须复核限位！")
        log("    ★ 标定脚本自身**不做绝对定位**（写 offset 之前坐标系无意义），只用相对偏置。")
    log(f"  ★ **implied_P_direction_offset_deg = {deg(theta_after):+.4f}°**"
        f"（平衡点在新坐标系里的读数）")
    log(f"     ⇒ 辨识时的 P 方向约束: **P = |P|·R(−{deg(theta_after):+.4f}°)·d̂**"
        f"（{('就是 P ∥ d, dy=0 ⇒ Py=0' if abs(deg(theta_after)) < 1e-6 else '**不能**再假设 Py=0')}）")
    if probe:
        log(f"  行程自检: W={probe['width_deg']:.2f}°（scale 校核"
            f"{'通过' if probe['ok_scale'] else '不通过'}）, 到两侧 "
            f"+{probe['d_plus_deg']:.2f}°/−{probe['d_minus_deg']:.2f}°"
            f"（{'自洽' if probe['ok_dist'] else '不自洽'}）")
    log("-" * 84)

    payload = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), mode_desc=mode_desc,
        report_only=bool(args.report_only), omega_nominal=calib["omega"],
        repeats=calib["repeats"],
        approach=("one-side" if args.one_side else "both-sides-alternating"),
        spinup=args.spinup_hold, accel_limit=args.accel_limit,
        safety=dict(rel_budget_deg=args.rel_budget_deg,
                    rel_path_budget_deg=args.rel_path_budget_deg,
                    peak_deg=deg(calib["safety"].peak), path_deg=deg(calib["safety"].path),
                    stall_sec=args.stall_sec, stall_deg=args.stall_deg,
                    bias_torque=args.bias_torque, rate_abort=args.rate_abort),
        mapped0_deg=deg(calib["mapped0"]),
        runs=[{**asdict(r), "theta_mean_deg": deg(r.theta_mean),
               "start_read_deg": deg(r.start_read), "release_read_deg": deg(r.release_read),
               "bias_cmd_deg": deg(r.bias_cmd), "bias_achieved_deg": deg(r.bias_achieved),
               "moved_deg": deg(r.moved_rad)} for r in calib["runs"]],
        summary=dict(left=to_deg(s_l), right=to_deg(s_r), blind=to_deg(s_b),
                     pos=to_deg(calib["pos"]), neg=to_deg(calib["neg"]),
                     total_mean_rad=total_mean, total_mean_deg=deg(total_mean),
                     total_std_deg=deg(calib["total_std"]),
                     single_side_bias_deg=deg(calib["single_side"]),
                     single_side_bias_theory_deg=side_bias_deg(
                         calib["omega"], (truth or {}).get("fcSmall"),
                         (truth or {}).get("dx", 0.0), (truth or {}).get("dy", 0.0),
                         (truth or {}).get("Px", 0.0), (truth or {}).get("Py", 0.0)),
                     omega_gap_deg=deg(calib["omega_gap"]),
                     band_emp_deg=deg(calib["band_emp"]), n_ok=n_ok,
                     n_stuck=calib["n_stuck"], n_bias_bad=calib["n_bias_bad"]),
        level_check=dict(gravity_norm=gr, tol=GRAVITY_TOL, ok=bool(gr < GRAVITY_TOL)),
        probe=probe,
        method=calib.get("method", args.method_used),
        feasibility=args._feasibility,
        theta_star_meas_rad=total_mean, theta_star_meas_deg=deg(total_mean),
        theta_star_level_equiv_deg=(deg(calib["level_equiv"])
                                    if calib.get("level_equiv", float("nan")) ==
                                    calib.get("level_equiv", float("nan")) else None),
        tilt_shift_deg=(deg(calib.get("tilt_shift", 0.0)) if method == "gravity" else 0.0),
        implied_P_direction_offset_deg=deg(theta_after),
        P_direction_constraint=f"P = |P| * R(-{deg(theta_after):.6f} deg) * d_hat",
        zero_manual_deg=args.zero_deg,
        stuck_ratio=(calib["n_stuck"] / n_ok if n_ok else None),
        zero_offset=(None if args.report_only else dict(
            scale=scale, offset_old_rad=offset_old, mapped_eq_rad=total_mean,
            delta_offset_rad=delta, delta_offset_deg=deg(delta),
            offset_new_rad=offset_new_val(offset_old, delta),
            manual=(args.zero_deg is not None))),
    )
    if extra:
        payload.update(extra)
    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    csv_path = os.path.splitext(out_json)[0] + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_HEADER)
        for r in calib["runs"]:
            w.writerow([int(r.index), int(r.direction), r.approach,
                        f"{r.omega_nominal:.3f}", f"{r.omega_mean:.4f}",
                        f"{r.start_read:.6f}", f"{r.bias_cmd:.6f}",
                        f"{r.bias_achieved:.6f}", int(bool(r.bias_ok)),
                        f"{r.release_read:.6f}", f"{r.theta_mean:.6f}",
                        f"{r.moved_rad:.6f}", f"{r.theta_std:.6f}", f"{r.theta_min:.6f}",
                        f"{r.theta_max:.6f}", f"{r.rate_residual:.4f}",
                        f"{r.settle_time:.3f}", f"{r.temp_big:.0f}", f"{r.temp_small:.0f}",
                        f"{r.gravity_norm:.4f}", int(bool(r.stuck)), int(bool(r.stalled)),
                        int(bool(r.ok)), r.abort_reason])
    log(f"  写出: {out_json}")
    log(f"        {csv_path}")
    return payload


def offset_new_val(offset_old, delta):
    return (offset_old or 0.0) + delta


# ============================================================================
# --sim 消融
# ============================================================================
def _row(name, truth, args, omega, repeats, directions, one_side=False, gravity_a=(0, 0),
         friction="stick", drag=0.0, bias_deg=None, spinup=None, guard_travel=None,
         rel_budget=None, accel=None):
    saved = (args.one_side, args._sim_gravity, args.bias_lo, args.bias_hi,
             args.spinup_hold, args.rel_budget_deg, args.accel_limit)
    args.one_side = one_side
    args._sim_gravity = gravity_a
    if bias_deg:
        args.bias_lo, args.bias_hi = math.radians(bias_deg[0]), math.radians(bias_deg[1])
    if spinup:
        args.spinup_hold = spinup
    if rel_budget:
        args.rel_budget_deg = rel_budget
    if accel:
        args.accel_limit = accel
    rig = SimRig(truth, gravity_a=gravity_a, friction=friction, drag=drag, seed=args.seed,
                 travel=guard_travel, accel_limit=accel or args.accel_limit)
    calib = run_calibration(rig, omega, repeats, args, seed=args.seed,
                            verbose=args.verbose, directions=directions)
    th_true = theta_star(truth["dx"], truth["dy"], truth["Px"], truth["Py"]) \
        + truth["zero_offset"]
    ok = calib["ok"]
    row = dict(case=name, plant=friction, drag=drag, omega=omega, n_dirs=len(directions),
               repeats=repeats, n_ok=len(ok), n_try=len(calib["runs"]),
               n_stuck=calib["n_stuck"],
               pos_deg=(deg(calib["pos"]["mean"]) if calib["pos"]["n"] else None),
               neg_deg=(deg(calib["neg"]["mean"]) if calib["neg"]["n"] else None),
               left_deg=(deg(calib["left"]["mean"]) if calib["left"]["n"] else None),
               right_deg=(deg(calib["right"]["mean"]) if calib["right"]["n"] else None),
               est_deg=(deg(calib["total_mean"]) if ok else None),
               err_deg=(deg(calib["total_mean"] - th_true) if ok else None),
               single_side_deg=(deg(calib["single_side"]) if ok else None),
               single_side_theory_deg=side_bias_deg(omega, truth["fcSmall"], truth["dx"],
                                                    truth["dy"], truth["Px"], truth["Py"]),
               omega_gap_deg=(deg(calib["omega_gap"]) if ok else None),
               std_deg=(deg(calib["total_std"]) if ok else None),
               peak_deg=deg(calib["safety"].peak), theta_true_deg=deg(th_true),
               limit_hits=rig.limit_hits,
               abort_note=next((r.abort_reason for r in calib["runs"] if not r.ok), ""))
    (args.one_side, args._sim_gravity, args.bias_lo, args.bias_hi, args.spinup_hold,
     args.rel_budget_deg, args.accel_limit) = saved
    return row


def omega_sign_drag_check(truth, omega, c, theta0_deg, gravity_a=(0.0, 0.0),
                          seconds=25.0) -> tuple:
    """确定性验证"阻力是 Ω 的奇次项": 同一初始状态、同一 |Ω|、仅符号不同。

    无阻力 ⇒ 停位完全相同（Ω 偶函数）; 有阻力 ⇒ 两个方向停在**相反方向平移**的平衡点上,
    **但它们的平均正好回到无阻力平衡点** ⇒ 这就是"±Ω 双向平均"能消掉阻力的原因。
    """
    out = []
    for sign in (+1.0, -1.0):
        plant = SmallAxisPlant(truth, gravity_a=gravity_a, friction="stick", drag=c)
        plant.q = math.radians(theta0_deg)
        plant.qd = 0.0
        plant.omega = sign * omega
        plant.omega_dot = 0.0
        for _ in range(int(round(seconds / DT))):
            plant.step(0.0, DT)
        out.append(deg(plant.q))
    return out[0], out[1]


def _grav_row(name, truth, args, tilts, repeats, manual_deg=None,
              rel_budget=None) -> dict:
    """重力法一行: 大 yaw 不动、多倾角平均、并给出"水平等效"与 P 方向。"""
    saved = (args.method_used, args.tilt_list, args.zero_deg, args.one_side,
             args.rel_budget_deg)
    args.method_used = "gravity"
    args.tilt_list = list(tilts)
    args.one_side = False
    if rel_budget:
        args.rel_budget_deg = rel_budget
    rig = SimRig(truth, friction="stick", seed=args.seed, accel_limit=args.accel_limit)
    rig.g_psi_deg = 20.0
    calib = run_calibration(rig, 0.0, repeats, args, seed=args.seed,
                            verbose=args.verbose, tilt_list=list(tilts))
    th_true = theta_star(truth["dx"], truth["dy"], truth["Px"], truth["Py"]) \
        + truth["zero_offset"]                       # 水平旋转平衡点的读数真值
    ok = calib["ok"]
    lev = calib["level_equiv"]
    row = dict(case=name, plant="stick", drag=0.0, omega=0.0, n_dirs=len(tilts),
               repeats=repeats, n_ok=len(ok), n_try=len(calib["runs"]),
               n_stuck=calib["n_stuck"],
               pos_deg=None, neg_deg=None,
               left_deg=(deg(calib["left"]["mean"]) if calib["left"]["n"] else None),
               right_deg=(deg(calib["right"]["mean"]) if calib["right"]["n"] else None),
               est_deg=(deg(calib["total_mean"]) if ok else None),
               err_deg=(deg(lev - th_true) if (ok and lev == lev) else None),
               single_side_deg=(deg(calib["single_side"]) if ok else None),
               single_side_theory_deg=side_bias_grav_deg(truth["fcSmall"],
                                                         math.hypot(truth["Px"], truth["Py"]),
                                                         max(tilts)),
               omega_gap_deg=None, std_deg=(deg(calib["total_std"]) if ok else None),
               peak_deg=deg(calib["safety"].peak), theta_true_deg=deg(th_true),
               limit_hits=rig.limit_hits, manual_deg=manual_deg,
               tilt_shift_deg=(deg(calib["tilt_shift"]) if calib["tilt_shift"] == calib["tilt_shift"]
                               else None),
               implied_P_deg=(deg(wrap_pi(lev + math.radians(manual_deg)))
                              if (ok and lev == lev and manual_deg is not None) else
                              (deg(lev) if (ok and lev == lev) else None)),
               abort_note=next((r.abort_reason for r in calib["runs"] if not r.ok), ""))
    (args.method_used, args.tilt_list, args.zero_deg, args.one_side,
     args.rel_budget_deg) = saved
    return row


def sim_ablation(args) -> None:
    """消融（**真实量级参数**，由 CLI 驱动 plant）:

    A 离心法 @Ω≤π（不可行判定 + 卡住比例 + 建议 Ω）
    B 两侧交替 vs 单侧
    D 重力法 φ=30/60/90（多倾角平均）
    E 手动零点 δ 的对照
    F 临时配重（|P|×K）—— 现实出路
    """
    reps = args.sim_repeats
    real = sim_truth_from_args(args)
    fc = real["fcSmall"]
    dd = real["dx"]
    pp = math.hypot(real["Px"], real["Py"])
    k_c = stiffness(OMEGA_MAX, real["dx"], real["dy"], real["Px"], real["Py"])
    th = theta_star(real["dx"], real["dy"], real["Px"], real["Py"])
    log("=" * 104)
    log("虚拟台架自检（--sim）: 真·库仑摩擦 + stick 判定；**真实量级参数**（CLI 驱动）")
    log(f"  真值: d=({real['dx']},{real['dy']}) |d|={dd:.4g} m; "
        f"P=({real['Px']:.5f},{real['Py']:.5f}) |P|={pp:.4g} kg·m (与 d 偏 {args.p_angle_deg:g}°)")
    log(f"        Js={real['Js']} Jbig_eff={real['Jbig_eff']} fc_small={fc} fv_small={real['fvSmall']}"
        f" fc_big={real['fcBig']} fv_big={real['fvBig']}  零位偏移={args.zero_offset_deg:g}°")
    log(f"  ★ |d||P| = {dd * pp:.4g} kg·m² ; Ω=π ⇒ k_cent = {k_c:.5f} N·m/rad ; "
        f"±5° 内离心驱动力 = {k_c * math.radians(5):.6f} N·m vs fc={fc} "
        f"⇒ {'能推动' if k_c * math.radians(5) > fc else '**完全推不动**'}")
    log(f"  ★ θ* = {deg(th):+.3f}°（行程 [{deg(SMALL_TRAVEL_MIN):+.0f}°,{deg(SMALL_TRAVEL_MAX):+.0f}°] "
        f"⇒ {'在行程内' if SMALL_TRAVEL_MIN < th < SMALL_TRAVEL_MAX else '**在行程外（不可达）**'}）")
    log("  ★ fc_small 敏感性（Ω≤π 下的离心法 vs 重力法 φ=90°）:")
    log(f"    {'fc_small':>9}{'fc/k_cent(°)':>14}{'要≤5°需Ω':>10}{'要≤2°需Ω':>10}"
        f"{'每圈(s) 5°/2°':>16}{'fc/k_grav(°)':>13}{'配重 K(5°/2°)':>15}")
    for f_ in sorted({0.03, 0.05, 0.0973, 0.1225, fc}):
        o5 = math.sqrt(f_ / (dd * pp * math.radians(5)))
        o2 = math.sqrt(f_ / (dd * pp * math.radians(2)))
        kg = pp * GRAVITY_G
        k5 = f_ / (GRAVITY_G * math.radians(5) * pp)      # 需 |P| ≥ fc/(g·Δθ) ⇒ 放大倍数
        k2 = f_ / (GRAVITY_G * math.radians(2) * pp)
        log(f"    {f_:>9.4f}{deg(f_ / k_c):>14.1f}{o5:>10.1f}{o2:>10.1f}"
            f"{('%6.2f/%5.2f' % (2 * math.pi / o5, 2 * math.pi / o2)):>16}"
            f"{deg(f_ / kg):>13.1f}{('%5.1f×/%5.1f×' % (k5, k2)):>15}")
    log(f"    （Ω 上限 = π = {OMEGA_MAX:.4f} rad/s ⇒ 2.00 s/圈; 表里'要≤5°需Ω'都远超上限）")
    if not (SMALL_TRAVEL_MIN < th < SMALL_TRAVEL_MAX):
        log(f"  ⚠ 当前 P 方向使 θ*={deg(th):+.1f}° 落在行程外 ⇒ 即使方法可行也测不到;"
            f" 建议改 P 方向（--p-angle-deg）或只做 --probe-travel 交叉校核")

    args.one_side = False
    args._sim_gravity = (0.0, 0.0)
    args.omega = OMEGA_MAX
    args.omega_used = OMEGA_MAX
    args.method_used = "centrifugal"
    rows = []
    rows.append(_row("A1 离心@π 真实参数", real, args, OMEGA_MAX, reps, (+1, -1),
                     rel_budget=30.0))
    rows.append(_row("B1 单侧单次", real, args, OMEGA_MAX, 1, (+1,), one_side=True,
                     rel_budget=30.0))
    rows.append(_row("B2 单侧多次", real, args, OMEGA_MAX, reps, (+1,), one_side=True,
                     rel_budget=30.0))
    rows.append(_row("B3 两侧交替多次(推荐)", real, args, OMEGA_MAX, reps, (+1, -1),
                     rel_budget=30.0))
    for phi in (30.0, 60.0, 90.0):
        rows.append(_grav_row(f"D1 重力φ={phi:g}°", real, args, [phi], reps, rel_budget=40.0))
    rows.append(_grav_row("D2 重力多倾角 90+60", real, args, [90.0, 60.0], reps,
                          rel_budget=40.0))
    rows.append(_grav_row("E1 手动零点 δ=+5°", real, args, [90.0], reps, manual_deg=5.0,
                          rel_budget=40.0))
    # F. 临时配重（|P|×K）—— 定量出路
    big = dict(real)
    kk = args.sim_payload_k
    big["Px"], big["Py"] = real["Px"] * kk, real["Py"] * kk
    rows.append(_grav_row(f"F1 配重 |P|×{kk:g} 重力φ=90°", big, args, [90.0], reps,
                          rel_budget=40.0))
    rows.append(_row(f"F2 配重 |P|×{kk:g} 离心@π", big, args, OMEGA_MAX, reps, (+1, -1),
                     rel_budget=40.0))

    log("\n" + "=" * 140)
    log("仿真消融（离心行: 误差 = 双向合并 − 水平真值; 重力行: 误差 = **水平等效** − 水平真值）")
    log(f"{'工况':<24}{'plant':>5}{'c':>5}{'Ω':>6}{'轮':>4}{'次':>4}{'有效':>5}{'卡住':>5}"
        f"{'仅+Ω(°)':>10}{'仅−Ω(°)':>10}{'左(°)':>9}{'右(°)':>9}{'合并(°)':>9}"
        f"{'误差(°)':>9}{'±Ω差':>8}{'单侧偏差':>9}{'理论':>9}{'合并σ':>8}{'位移峰值':>9}")
    for r in rows:
        f = lambda v, w=9, p=3: (f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}")
        log(f"{r['case']:<24}{r['plant']:>5}{r['drag']:>5.1f}{r['omega']:>6.2f}"
            f"{r['n_dirs']:>4}{r['repeats']:>4}{r['n_ok']:>5}{r['n_stuck']:>5}"
            f"{f(r['pos_deg'],10)}{f(r['neg_deg'],10)}{f(r['left_deg'])}{f(r['right_deg'])}"
            f"{f(r['est_deg'])}{f(r['err_deg'])}{f(r['omega_gap_deg'],8)}"
            f"{f(r['single_side_deg'])}{f(r['single_side_theory_deg'],9)}"
            f"{f(r['std_deg'],8)}{r['peak_deg']:>9.2f}")
    log("=" * 140)
    for r in rows:
        if r["n_ok"] < r["n_try"] and r["abort_note"]:
            log(f"  [中止] {r['case']}: {r['n_ok']}/{r['n_try']} 有效；首个原因: {r['abort_note']}")
    log("读表要点:")
    log("  · A/B 组（离心 @Ω≤π）: 驱动力 ½μΩ² 远小于静摩擦 ⇒ **推不动**（看卡住列）"
        "⇒ 读数=起点, 误差小是假象; 真实参数下 fc/k=565° ⇒ 要 5° 需 Ω≈33 rad/s（0.19 s/圈）。")
    log("  · D 组（重力法）: 比 Ω=π 的离心法强 9.9×(φ=90°), 但 fc/k=56.8° 仍远超 5°。")
    log(f"  · F 组（临时配重 |P|×{kk:g}）: 定量出路 —— 需要 |P| ≥ fc/(g·sinφ·Δθ);"
        f" 注意配重后测到的是 **P_tot**, 必须 P_payload = P_tot − P_w 才能得到原零点。")

    # 主结果落盘（用真实参数, 重力法多倾角）
    args.method_used = "gravity"
    args.zero_deg = None
    args.rel_budget_deg = 40.0
    sim_travel = ((th + SMALL_TRAVEL_MIN, th + SMALL_TRAVEL_MAX) if args.probe_travel else None)
    rig = SimRig(real, friction="stick", seed=args.seed, travel=sim_travel,
                 accel_limit=args.accel_limit)
    rig.g_psi_deg = 20.0
    main = run_calibration(rig, 0.0, reps, args, seed=args.seed, verbose=False,
                           tilt_list=[90.0, 60.0])
    th_true = th + real["zero_offset"]
    probe = None
    if args.probe_travel:
        probe = probe_travel(rig, Controller(args), args, main["total_mean"],
                             (args.scale if args.scale else 1.0), main["safety"])
    report_and_save(main, args, "仿真台架（真实参数, 重力法 φ=90+60, 两侧交替）", args.out,
                    truth=real, probe=probe,
                    extra=dict(sim_truth=dict(
                        geometry="真实量级参数", **{k: v for k, v in real.items()}),
                        theta_star_level_equiv_true_deg=deg(th_true),
                        est_level_equiv_deg=(deg(main["level_equiv"])
                                             if main["level_equiv"] == main["level_equiv"]
                                             else None),
                        calib_error_deg=(deg(main["level_equiv"] - th_true)
                                         if main["level_equiv"] == main["level_equiv"]
                                         else None),
                        sim_ablation=rows))


# ============================================================================
# 命令行
# ============================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="小 yaw 编码器零位标定（离心平衡点法 · 相对偏置版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--omega", type=float, default=DEFAULT_OMEGA,
                   help=f"离心法大 yaw 角速度 rad/s（**硬上限 {OMEGA_MAX:.4f} = 2 s/圈**）")
    p.add_argument("--method", choices=("manual", "centrifugal", "gravity", "auto"),
                   default="manual",
                   help="★ manual=**人工把 yaw 摆到零点、读编码器**（首选取; 默认）; "
                        "centrifugal/gravity=实验性交叉校核; auto=自动选离心/重力")
    p.add_argument("--manual-tol-deg", type=float, default=MANUAL_TOL_DEG,
                   help="manual: 采样窗口内极差上限（度）—— 超过判'手还在动/没扶稳'")
    p.add_argument("--manual-hold-big", choices=("hold", "free"), default="hold",
                   help="manual: 大 yaw 用低增益速度环守住当前位置(hold, 默认) 或完全松手(free)")
    p.add_argument("--note", default="", help="manual: 记录用的是什么工装/基准（写进 JSON/CSV）")
    p.add_argument("--sim-placement-sd-deg", type=float, default=0.15,
                   help="--sim manual: 模拟'人手每次摆放'的散布（度）")
    p.add_argument("--sim-tremor-deg", type=float, default=0.05,
                   help="--sim manual: 模拟'扶稳时的抖动'峰峰值（度）")
    p.add_argument("--tilt-deg", type=float, default=DEFAULT_TILT_DEG,
                   help="重力法主倾角（度）")
    p.add_argument("--tilt-list", default=DEFAULT_TILT_LIST,
                   help="多倾角平均（逗号分隔, 度）: 不同倾角的粘滞带不同 ⇒ 平均掉粘滞带偏差")
    p.add_argument("--max-acceptable-deg", type=float, default=MAX_ACCEPTABLE_DEG,
                   help="单侧偏差的可接受上限; 超过则判该方法**不可行**并明确报告")
    p.add_argument("--zero-target", choices=("level", "measured"), default="level",
                   help="把哪个位置定义为 0: level=换算到'水平旋转平衡点'(与离心法同一物理点), "
                        "measured=就用本次测到的平衡位置")
    p.add_argument("--zero-deg", type=float, default=None,
                   help="★ 手动零点: 用户直接声明把零点挪动这么多度（覆盖标定建议值）")
    p.add_argument("--force", action="store_true",
                   help="即使判定不可行也继续测（默认不可行时仍会测以便取证, 但结论会标红）")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--mode", choices=("host", "mcu"), default="host")
    p.add_argument("--accel-limit", type=float, default=ACCEL_LIMIT,
                   help="大 yaw 加速段斜坡上限 rad/s²（默认由冲量约束反解, 见 §四）")
    p.add_argument("--spinup-hold", choices=("free", "hold"), default="free",
                   help="加速段: free=自由松手（默认, 需慢斜坡）; hold=相对保持（可快 4 倍）")
    p.add_argument("--bias-deg-lo", type=float, default=deg(BIAS_LO))
    p.add_argument("--bias-deg-hi", type=float, default=deg(BIAS_HI))
    p.add_argument("--bias-torque", type=float, default=BIAS_TORQUE,
                   help="相对偏置/探针力矩上限 N·m（必须 > fc_small）")
    p.add_argument("--bias-sec", type=float, default=BIAS_SEC)
    p.add_argument("--bias-kp", type=float, default=BIAS_KP)
    p.add_argument("--approach-both-sides", dest="one_side", action="store_false",
                   default=False, help="★ 相对偏置两侧交替（**默认**）")
    p.add_argument("--one-side", dest="one_side", action="store_true", default=False,
                   help="对照实验: 偏置固定在同一侧")
    p.add_argument("--rel-budget-deg", type=float, default=REL_BUDGET_DEG,
                   help="相对起始读数的位移预算（度）")
    p.add_argument("--rel-path-budget-deg", type=float, default=REL_PATH_BUDGET_DEG,
                   help="累计路径预算（度）")
    p.add_argument("--stall-sec", type=float, default=STALL_SEC)
    p.add_argument("--stall-deg", type=float, default=STALL_DEG)
    p.add_argument("--rate-abort", type=float, default=RATE_ABORT)
    p.add_argument("--settle-sec", type=float, default=SETTLE_SEC)
    p.add_argument("--settle-timeout", type=float, default=SETTLE_TIMEOUT)
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    p.add_argument("--rate-tol", type=float, default=RATE_TOL)
    p.add_argument("--omega-tol", type=float, default=OMEGA_TOL)
    p.add_argument("--torque-delta", type=float, default=TORQUE_DELTA)
    p.add_argument("--big-kp", type=float, default=BIG_KP)
    p.add_argument("--big-ki", type=float, default=BIG_KI)
    p.add_argument("--max-temp", type=float, default=55.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-only", action="store_true", help="只测平衡点、不写 offset 建议")
    p.add_argument("--probe-travel", action="store_true",
                   help="标定后量两侧机械停止位（行程/scale 自检）")
    p.add_argument("--probe-torque", type=float, default=PROBE_TORQUE)
    p.add_argument("--probe-max-deg", type=float, default=PROBE_MAX_DEG)
    p.add_argument("--probe-sec", type=float, default=PROBE_SEC)
    p.add_argument("--probe-scale-tol-deg", type=float, default=PROBE_SCALE_TOL_DEG)
    p.add_argument("--ack-mcu-limits-bypassed", action="store_true",
                   help="★ 真机必需: 确认已临时放宽/旁路电控小 yaw 限位保护（见前置条件 3）")
    p.add_argument("--out", default=None)
    p.add_argument("--offset-old", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--fc-small", type=float, default=None,
                   help=f"小 yaw 库仑摩擦 N·m（默认真实量级 {REAL['fcSmall']}）")
    p.add_argument("--js", type=float, default=REAL["Js"], help="小 yaw 惯量 kg·m²")
    p.add_argument("--jbig", type=float, default=REAL["Jbig_eff"], help="大 yaw 侧惯量")
    p.add_argument("--fc-big", type=float, default=REAL["fcBig"])
    p.add_argument("--fv-big", type=float, default=REAL["fvBig"])
    p.add_argument("--fv-small", type=float, default=REAL["fvSmall"])
    p.add_argument("--p-angle-deg", type=float, default=REAL["p_angle_deg"],
                   help="P 相对 d 的方向（度）; 默认 30°（避免 θ*=0 这种太巧的工况）")
    p.add_argument("--zero-offset-deg", type=float, default=5.0,
                   help="仿真里植入的已知编码器零位偏移（度）")
    p.add_argument("--sim-payload-k", type=float, default=11.0,
                   help="消融里'挂临时配重把 |P| 放大 K 倍'的那一行")
    p.add_argument("--d-offset", type=float, default=None)
    p.add_argument("--p-moment", type=float, default=None)
    p.add_argument("--sim", action="store_true", help="虚拟台架自检 + 消融（无硬件）")
    p.add_argument("--sim-repeats", type=int, default=3)
    p.add_argument("--sim-tilt-deg", type=float, default=5.0)
    p.add_argument("--sim-drag", type=float, default=0.5,
                   help="消融 ⑩ 的阻力系数 c（会同时跑 c=0 组）")
    p.add_argument("--verbose", action="store_true")
    return p


def run_manual_main(args) -> int:
    """manual 模式入口（真机与 --sim 共用同一段采集逻辑）。"""
    if args.sim:
        real = sim_truth_from_args(args)
        args._manual_true_mapped = real["zero_offset"]      # 真实 mapped 零点 = Z
        log(f"  [--sim manual] 虚拟台架: 真实 mapped 零点 = {deg(real['zero_offset']):+.4f}°，"
            f"摆放散布 σ={args.sim_placement_sd_deg:g}°，扶稳抖动峰峰 {args.sim_tremor_deg:g}°")
        rig = SimRig(real, friction="stick", seed=args.seed, accel_limit=args.accel_limit)
    else:
        rig = HwRig(args.mode)
    exit_code = 0
    try:
        if not rig.wait_ready(15.0):
            log("[ERROR] 15s 内没有收到有效的 MCU/IMU 数据（串口未连接？）")
            try:
                for _ in range(10):
                    rig.send(0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass
            return 2
        run_manual(rig, args, args.seed)
    except KeyboardInterrupt:
        log("\n[Ctrl+C] 立即停止（小 yaw 力矩本来就是 0, 大 yaw 归零）…")
        exit_code = 130
    finally:
        try:
            for _ in range(20):
                rig.send(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            rig.close()
        except Exception:
            pass
        log("已发零力矩并停止")
    return exit_code


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.bias_lo = math.radians(args.bias_deg_lo)
    args.bias_hi = math.radians(args.bias_deg_hi)
    args.bias_tol = BIAS_TOL
    args.stall_hard_deg = STALL_HARD_DEG
    args.manual_tries = MANUAL_TRIES
    # ★ 大 yaw 转速硬上限: 2 s/圈 ⇒ Ω ≤ π
    if args.omega > OMEGA_MAX + 1e-9:
        log(f"[WARN] --omega={args.omega:g} 超过硬上限（2 s/圈 ⇒ Ω ≤ {OMEGA_MAX:.4f} rad/s）"
            f" ⇒ 钳到 {OMEGA_MAX:.4f}")
        args.omega = OMEGA_MAX
    args.omega_used = args.omega
    if isinstance(args.tilt_list, str):
        args.tilt_list = [float(x) for x in args.tilt_list.split(",") if x.strip()]
    if not args.tilt_list:
        args.tilt_list = [args.tilt_deg]
    args.method_used = args.method
    args.d_angle_deg = 0.0
    args._manual_true_mapped = None
    args._sim_gravity = (0.0, 0.0)
    if args.out is None:
        args.out = DEFAULT_OUT_SIM if args.sim else DEFAULT_OUT
    if args.repeats < 1:
        raise SystemExit("[ERROR] --repeats 必须 ≥ 1")
    if args.sim and args.mode != "host":
        log("[INFO] --sim 的虚拟台架没有电控速度内环 ⇒ 强制 --mode=host")
        args.mode = "host"

    geom = {"fc_small": args.fc_small, "d_offset": args.d_offset, "p_moment": args.p_moment}
    if any(v is None for v in geom.values()) and not args.sim:
        try:
            from torque_controller import default_model_params
            mp = default_model_params()
            if geom["fc_small"] is None:
                geom["fc_small"] = float(mp.fcSmall)
            if geom["d_offset"] is None:
                geom["d_offset"] = math.hypot(float(mp.dx), float(mp.dy))
            if geom["p_moment"] is None:
                geom["p_moment"] = math.hypot(float(mp.Px), float(mp.Py))
        except Exception:
            pass
    fea = choose_method(args, geom)
    args._feasibility = {k: (v if not isinstance(v, dict) else v) for k, v in fea.items()}
    if args.method == "auto":
        args.method_used = fea["method"]
    else:
        args.method_used = args.method
    args._geom = geom

    log("=" * 100)
    log("小 yaw 编码器零位标定 — 平衡点法（**相对偏置版**: 标定前不用任何绝对角度）")
    log("  ★ 前置条件: ①标定前绝对角度无意义⇒只用相对偏置; ②首测直接松手不预置;")
    log("             ③电控小 yaw 限位保护以错误零点为基准⇒标定期间必须临时旁路")
    log(f"  ★ 大 yaw 转速硬上限: 2 s/圈 ⇒ Ω ≤ π = {OMEGA_MAX:.4f} rad/s（当前 Ω={args.omega:g}）")
    log(f"  ★ 选法: --method={args.method} ⇒ **{args.method_used}**"
        f"（{'；'.join(fea['reasons'])}）")
    f = lambda v, w=8, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, float) and v == v else f"{'—':>{w}}")
    log(f"     可行性判定(单侧偏差 vs --max-acceptable-deg={args.max_acceptable_deg:g}°):")
    log(f"       离心法 Ω≤π: fc/(Ω²|d||P|) = {f(fea['side_cent'])}°"
        f" ⇒ {'可行' if fea['feasible_cent'] else '**不可行**（推不动/精度不够）'}")
    tl = ", ".join(f"φ={p:g}°→{f(v,7)}°" for p, v in fea["side_grav"].items())
    log(f"       重力法: fc/(|P|·g·sinφ): {tl}"
        f" ⇒ {'可行' if fea['feasible_grav'] else '**不可行**'}")
    if fea.get("k_ratio_90") == fea.get("k_ratio_90"):
        log(f"       刚度对比: k_grav(φ=90°)/k_cent(Ω=π) = {fea['k_ratio_90']:.1f}×"
            f"（φ 阈值 ≈ {deg(math.asin(min(1.0, stiffness(args.omega_used, (geom['d_offset'] or 0), 0, (geom['p_moment'] or 1e-9), 0) / max(1e-12, (geom['p_moment'] or 1e-9) * GRAVITY_G)))):.2f}°）")
    if not (fea["feasible_cent"] or fea["feasible_grav"]):
        dd, pp, fc_ = geom.get("d_offset"), geom.get("p_moment"), geom.get("fc_small")
        log("  ★★ **结论: Ω≤π 与当前几何下都达不到 %.0f° 精度**"
            % args.max_acceptable_deg)
        if dd and pp and fc_:
            import math as _m
            for tgt in (args.max_acceptable_deg, 2.0):
                need_ddP = fc_ / (_m.radians(tgt) * args.omega_used ** 2)
                need_P = fc_ / (GRAVITY_G * _m.sin(_m.radians(max(args.tilt_list))) * _m.radians(tgt))
                log(f"     目标 {tgt:g}°: 离心法需 |d||P| ≥ fc/(Δθ·Ω²) = {need_ddP:.4g} "
                    f"（现 {dd * pp:.4g} ⇒ **{need_ddP / (dd * pp):.1f}×**）; "
                    f"重力法(φ={max(args.tilt_list):g}°)需 |P| ≥ fc/(g·sinφ·Δθ) = {need_P:.4g} "
                    f"（现 {pp:.4g} ⇒ **{need_P / pp:.1f}×**）; 对应 Ω ≥ "
                    f"{_m.sqrt(fc_ / (dd * pp * _m.radians(tgt))):.1f} rad/s "
                    f"（{2 * _m.pi / _m.sqrt(fc_ / (dd * pp * _m.radians(tgt))):.2f} s/圈）")
        log("     出路: ① 减小 fc_small（换脂/预紧/轴承）② **挂已知临时配重把 |P| 放大 K 倍**"
            "（见下）③ 降低精度要求 ④ 改用机械停止位探测(--probe-travel)做交叉校核")
        log("     ★ 配重法注意事项: 加配重后平衡点是**总一阶矩 P_tot = P_payload + P_w** 的平衡点"
            "⇒ 必须用测得的 |P_tot| 与已知配重 P_w **反解 P_payload = P_tot − P_w** 才能得到"
            "'去掉配重后'的零点; 本模式建议只用来 `--report-only` 测**方向**（P 的方向对"
            "配重不敏感的方向分量需按此换算），零点必须按上式换算后确定。")
    elif args.method_used == "gravity" and not fea["feasible_grav"]:
        log("  [WARN] 重力法也未达精度要求 ⇒ 建议 --probe-travel 交叉校核")
    log(f"  行程（**标定后**语义）=[{deg(SMALL_TRAVEL_MIN):+.0f}°, "
        f"{deg(SMALL_TRAVEL_MAX):+.0f}°]; 本次安全 = 相对位移预算 "
        f"{args.rel_budget_deg:g}° + 失速 {args.stall_sec:g}s + 力矩 ≤"
        f"{args.bias_torque:g} N·m + |θ̇|≤{args.rate_abort:g}")
    if args.method_used == "centrifugal":
        log(f"  大 yaw: Ω=±{args.omega:g} rad/s, 速度环={args.mode}, 加速限幅 "
            f"{args.accel_limit:g} rad/s²({args.spinup_hold})")
    else:
        log(f"  大 yaw: **不动（Ω=0）**; 重力法倾角列表 {args.tilt_list}°（多倾角平均消粘滞带）")
        log(f"  加速段冲量校核: 重力法不转大 yaw ⇒ **没有 M12·Ω̇ 甩飞问题、也没有空气阻力**")
    log(f"  小 yaw: 相对偏置 ±{args.bias_deg_lo:g}~{args.bias_deg_hi:g}°, "
        f"{'单侧' if args.one_side else '两侧交替'}, {args.repeats} 次/方向, "
        f"力矩上限 {args.bias_torque:g} N·m")
    fc = args.fc_small
    kk = None
    if args.p_moment and args.d_offset:
        kk = (args.omega ** 2) * args.d_offset * args.p_moment
    if fc and kk:
        need = fc + kk * args.bias_hi
        log(f"  偏置力矩校核: 稳态 Ω 下推离平衡点需要 τ ≥ fc + k·δ = {fc:g} + "
            f"{kk:.3g}×{args.bias_hi:.4f} = {need:.3f} N·m"
            f"（当前上限 {args.bias_torque:g} ⇒ {'够' if args.bias_torque >= need else '**不够, 会推不到目标**'}）")
    else:
        log("  偏置力矩校核: 需要 τ ≥ fc + k·δ（k=Ω²|d||P|）。给 --fc-small/--d-offset/"
            "--p-moment 可自动校核")
    log("=" * 100)

    if args.method == "manual":
        return run_manual_main(args)

    if args.sim:
        sim_ablation(args)
        return 0

    if args.method == "manual":
        log("  ★ manual 模式: 小 yaw **力矩恒 0**（只读编码器）⇒ 不涉及'电控按错误零点清零力矩'"
            "的问题, 因此**不需要** --ack-mcu-limits-bypassed; 工具也不做任何自动运动")
    if not args.ack_mcu_limits_bypassed and args.method != "manual":
        log("[拒绝开跑] 真机标定前必须确认: 已**临时放宽/旁路电控的小 yaw 限位保护**。")
        log("  原因: 电控限位以**当前（错误的）零点**为基准, 标定期间它会误判并把小 yaw 力矩清零")
        log("        （脚本能检测到'下发了力矩但不跟随', 但那已是白跑一趟）。")
        log("  最后防线保留: 低力矩限幅 + 失速检测 + 机械硬限位。")
        log("  确认后加 --ack-mcu-limits-bypassed 重跑; 写回 Δoffset 后请**恢复**限位保护。")
        return EXIT_NEED_ACK

    if not args.force and not (fea["feasible_cent"] or fea["feasible_grav"]) \
            and args.method != "auto":
        log(f"[拒绝开跑] --method={args.method} 在 Ω≤π 下判定为**不可行**；"
            f"确认要硬跑请加 --force（会白跑: 读数=起点, 没有信息量）")
        return 5
    rig = HwRig(args.mode)
    exit_code = 0
    try:
        if not rig.wait_ready(15.0):
            log("[ERROR] 15s 内没有收到有效的 MCU/IMU 数据（串口未连接？）")
            try:
                for _ in range(10):
                    rig.send(0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass
            return 2
        st = rig.read()
        log(f"链路就绪: 小 yaw 读数={deg(st.small_angle):+.2f}°（绝对含义未知）  "
            f"Ω={st.platform_rate:+.3f} rad/s  |gravity_a|={st.gravity_norm:.4f} m/s²  "
            f"温度=({st.temp_big:.0f},{st.temp_small:.0f})℃")
        if st.gravity_norm > GRAVITY_TOL:
            log(f"  [SAFETY] 底盘不水平（|gravity_a|={st.gravity_norm:.3f}）: 平衡点会偏移"
                f"（Ω 偶函数, 消不掉）⇒ **先把底盘放平**")
        log("  提示: 标定期间请勿触碰云台/底盘（大 yaw 会持续较长时间旋转）")
        calib = run_calibration(rig, args.omega, args.repeats, args, seed=args.seed,
                                verbose=args.verbose)
        probe = None
        if args.probe_travel and calib["ok"]:
            probe = probe_travel(rig, Controller(args), args, calib["total_mean"],
                                 (args.scale if args.scale else 1.0), calib["safety"])
        report_and_save(calib, args, f"真机（mode={args.mode}, Ω=±{args.omega:g}）",
                        args.out, probe=probe)
    except KeyboardInterrupt:
        log("\n[Ctrl+C] 立即停止: 力矩归零…")
        exit_code = 130
    finally:
        try:
            for _ in range(20):
                rig.send(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            rig.close()
        except Exception:
            pass
        log("已发零力矩并停止")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
