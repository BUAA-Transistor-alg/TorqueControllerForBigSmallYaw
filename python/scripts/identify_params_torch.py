#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
identify_params_torch.py — 平面二维 8 参模型的 **PyTorch 可导前向仿真** 参数辨识
================================================================================

模型（与 `include/tcbs/mpc/planar_yaw_model.h` 逐项对应，务必保持一致）:

    τ = M(q)·q̈ + h(q, q̇)

    Q(θs) = R(θs)·P                     # P = (Px, Py) 上装一阶矩 m_u·ρ
    M11 = Jbig_eff + Js + 2·(d·Q)       M12 = Js + (d·Q)      M22 = Js
    μ   = 2·(dy·Qx − dx·Qy)             # = ∂M11/∂θs
    h_b = μ·θ̇b·θ̇s + ½μ·θ̇s² − G_b + μ·θ̇s·ω_c + M11·α_c + fric_b
    h_s = −½μ·θ̇b²      − G_s − μ·θ̇b·ω_c − ½μ·ω_c² + M12·α_c + fric_s
    fric_k = fc_k·tanh(λ·θ̇_k) + fv_k·θ̇_k

8 个待辨识参数（顺序固定，与 paramsToVector / regressor 列一致）:
    0 Jbig_eff  1 Js  2 Px  3 Py  4 fc_big  5 fv_big  6 fc_small  7 fv_small

约定（与 docs/calibration.md §4 / 用户要求一致）:
  * **λ 固定 10，不辨识**；几何 d 是实测值（dx, dy 不是辨识参数）。
  * 采集工况: 底盘静止（base_omega = base_alpha = 0）、水平（gravity_a = 0）、pitch 恒 0。
    因此 G_b = G_s = 0、底盘耦合项为 0 —— 本脚本仍按完整公式实现，置零只是工况。
  * 角度按**编码器量化**处理（8192 计数/整圈 ⇒ 步长 2π/8192 ≈ 7.66e-4 rad）。
  * 辨识模型 λ = 10；仿真被控对象(plant) 可用 λ = 100 模拟真实库伦摩擦
    ⇒ 这本身就是**摩擦形状失配**，由 compare_ident_methods.py 量化。

辨识方法 = **输出误差法（output error）**: 用记录的**两轴力矩**作为输入，从记录的初始状态
出发对被控对象做前向仿真（dt = 0.01 s），最小化预测角度/角速度与记录值的差。

================================================================================
★ 默认配方 = **原仓库 TorqueController/python/scripts/param_ident.py 的同款配方**
================================================================================
（原仓库是单 yaw 4 参 `J·dω/dt = τ − τ_c·tanh(λ·ω) − b·ω + τ_d`；本仓库是平面 2 自由度
 8 参模型，所以"同款配方"指的是**同一套训练/损失/积分/参数化/收敛曲线做法**，而不是同几个
 魔数。逐条对齐如下:)

  1. **无任何参数限位**：不再用 sigmoid 软边界 / clamp / 投影 / 惩罚项。
     * 天然为正的 6 个参数（`Jbig_eff, Js, fc_big, fv_big, fc_small, fv_small`）采用
       **原仓库同款 log 参数化**：φ = exp(raw)（正性由参数化隐式保证）；
     * `Px, Py` 可正可负 ⇒ **直接自由参数**（φ = raw）。
  2. **积分 = 半隐式（symplectic）欧拉**，`substeps = 1`，`dt` 取自数据（100 Hz ⇒ 0.01 s）：
     先 ω ← ω + dt·α，再 θ ← θ + dt·ω（与原仓库 `param_ident.py` 逐行同序；
     `--integrator=rk4` 可回到旧的 RK4 精细配方）。
  3. **每次优化步只随机截取 `seg_steps = 10` 步（0.1 s）的片段**，`epochs = 200`；
     每个 epoch 对**每段数据各抽 1 个片段做 1 个 Adam 步** ⇒ 每 epoch 的步数 = 段数，
     总 Adam 步数 = `epochs × 段数`（= 原仓库 `num_epochs × sample 数` 的同构形式）。
     ★ **`epochs = 1000`（与原仓库 num_epochs 同轮数）**。
  4. **损失 = 角度误差 MSE（先 wrap 到 (−π, π]）+ 角速度误差 MSE，两项等权相加**；
     **不用 Huber、不做分窗 mini-batch**（`--loss-mode=huber` 可回到旧配方）。
  5. **Adam，lr = 3e-4**，**无学习率调度**（原仓库没有 scheduler），**无 LBFGS**。
  6. **初值沿用本仓库的 CAD/占位初值** `defaultModelParams()`
     （`Jbig_eff=0.0240, Js=0.0130, Px=0, Py=0, fc_big=0.090, fv_big=0.030, fc_small=0.030,
     fv_small=0.008`）——**不**照搬原仓库那几个单 yaw 魔数（log(0.05)/log(0.5)/log(0.03)/0），
     因为两者是**不同的物理模型**，初值必须来自本模型的 CAD 量级。
  7. **无训练/验证划分、无 holdout**（原仓库也没有）；`val_loss` 只是最后在全批算一次的
     训练损失，供 compare_ident_methods.py 打印用。
  8. 训练中记录每 epoch 的 `loss_history` 与 `param_history`，最后画收敛曲线（见下）。

★ 与原仓库**刻意不同**的一点: 摩擦软符号陡度 **λ = 10**（本仓库约定，见
  `include/tcbs/mpc/planar_yaw_model.h` / FRICTION_LAMBDA），而原仓库单 yaw 版用 **λ = 1e4**。
  不能照搬 1e4：本仓库的**前向仿真/被控对象/MPC 全部用 λ = 10**，辨识模型必须与之一致，
  否则辨识出的 fc 是在拟合一个"更接近硬 sign"的摩擦形状，移植回控制器会造成系统性偏差；
  且 λ = 1e4 时 tanh 在 ω ~ 1e-4 rad/s 内饱和，梯度几乎处处为 0（数值上是硬 sign），
  对本仓库 100 Hz 采样 + 编码器量化（7.7e-4 rad）的数据没有意义。

迭代次数 = **与原仓库同轮数** ⇒ `epochs = 1000`，见 `FitConfig.epochs`（实测 300~500 epoch 即进平台，可自行调小）。

★ 回到旧的"精细配方"（Huber + 分窗 mini-batch + LBFGS + lr=5e-3 + iters=400 + RK4）::

    --legacy-recipe                      # 一键预设
    # 或逐项显式指定:
    --iters=400 --lbfgs-iters=25 --lr=5e-3 --loss-mode=huber \\
    --window-len=150 --windows-per-seg=2 --batch-size=4 --integrator=rk4

收敛曲线（默认保存文件，不依赖显示环境）::

    data/sysid/ident_torch_convergence.png   # loss(log 纵轴) + 8 个参数各自的收敛曲线（3×3）
    data/sysid/ident_torch_traj.png          # 实测 vs 仿真（大 yaw 段 + 小 yaw 段，θ 与 θ̇）
    --plot-out=PREFIX|DIR|xxx.png 改路径/前缀；--no-plot 关闭；--show-plot 交互显示。

用法::

    # 多段一起拟合（.csv 与 .npz 都支持；--data 可重复或用逗号分隔）
    python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv'
    # 只拟合大 yaw（小 yaw 摩擦参数冻结在初值）
    python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv' --fit-axis=big
    # 模型自检（回归矩阵 / numpy vs torch / 前向动力学一致性）
    python3 python/scripts/identify_params_torch.py --selftest

数据格式（与 collect_sysid 约定一致，两种都兼容）:
  * CSV 列: t,theta_big,theta_small,dtheta_big,dtheta_small,tau_big,tau_small,axis,held_target,mcu2_seq
  * npz 键: 同名列数组 + 标量 axis / dt(=0.01) / held_target
    也可用 `theta`(T,2) / `dtheta`(T,2) / `tau`(T,2) 的打包形式。
  缺失的 dtheta 会自动用「中心差分 + 3 点平滑」从角度补算（量化噪声下这是必要的预处理）。
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - 环境缺 torch 时给出明确提示
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# ============================================================================
# 常量
# ============================================================================
PARAM_NAMES = ("Jbig_eff", "Js", "Px", "Py", "fc_big", "fv_big", "fc_small", "fv_small")
PARAM_UNITS = ("kg·m²", "kg·m²", "kg·m", "kg·m", "N·m", "N·m·s/rad", "N·m", "N·m·s/rad")
NPARAM = 8
DT_DEFAULT = 0.01                      # 100 Hz
FRICTION_LAMBDA = 100.0                # ★ 辨识模型固定 λ = 100（不辨识）
                                       #   λ=100 ⇒ |ω|≳1°/s 即饱和，逼近真库仑; 前向仿真须细子步
ENCODER_CPR = 8192                     # 编码器计数/整圈
QUANT_STEP = 2.0 * math.pi / ENCODER_CPR   # ≈ 7.669e-4 rad
AXIS_BIG, AXIS_SMALL = 0, 1

# ★ 无参数限位（与原仓库 `param_ident.py` 一致）：不存在任何上下界 / clamp / 投影。
#   · 天然为正的 6 个参数（Jbig_eff, Js, fc_big, fv_big, fc_small, fv_small）用 **log 参数化**
#     φ = exp(raw) ⇒ 正性隐式保证（原仓库对 J / τ_c / b 就是这么做的）；
#   · Px / Py 可正可负 ⇒ **直接自由参数** φ = raw。
#   顺序与 PARAM_NAMES 一致。旧的 sigmoid 软边界（DEFAULT_BOUNDS / --p-bound）已删除。
POSITIVE_PARAM = np.array([True, True, False, False, True, True, True, True])
POSITIVE_IDX = tuple(int(i) for i in np.nonzero(POSITIVE_PARAM)[0])
FREE_IDX = tuple(int(i) for i in np.nonzero(~POSITIVE_PARAM)[0])


def p_direction(dx: float, dy: float, zero_angle_deg: float = 0.0):
    """★ "零点标定"后 P 必须满足的方向。

    离心平衡（大 yaw 恒速 Ω、小 yaw 松手 τ=0）给出平衡条件 `R(θ*)·P ∥ d`，
    稳定解取径向外侧 ⇒ **P = |P|·R(−θ*)·d̂**，其中 θ* = 平衡点在**当前零点坐标系**里的读数:

      · 零点恰好设在平衡点上（离心/重力标定做过）⇒ θ* = 0 ⇒ P = |P|·d̂（dy=0 时 Py=0）；
      · 零点被手动挪过 δ（或标定有残差）⇒ θ* = δ ⇒ P 相对 d 旋转 −δ。

    返回单位方向 (ux, uy)。
    """
    n = math.hypot(dx, dy)
    if n < 1e-12:
        raise ValueError("P 方向约束需要 |d| > 0（几何偏置为 0 时方向无定义）")
    ux, uy = dx / n, dy / n
    d = math.radians(zero_angle_deg)
    return (ux * math.cos(d) + uy * math.sin(d), -ux * math.sin(d) + uy * math.cos(d))


def default_param_vector() -> np.ndarray:
    """CAD/占位初值（= planar_yaw_params.h 的 defaultModelParams()，Px/Py 取 0）。"""
    return np.array([0.0240, 0.0130, 0.0, 0.0, 0.090, 0.030, 0.030, 0.008], dtype=np.float64)


# ============================================================================
# 参数容器
# ============================================================================
@dataclass
class PlanarParams:
    """平面 8 参模型参数 + 实测几何（几何量不参与辨识）。"""

    # ── 实测几何 / 固定量 ──
    dx: float = 0.030
    dy: float = 0.0
    gravity: float = 9.81
    m_u_known: float = 0.0
    friction_lambda: float = FRICTION_LAMBDA
    tau_offset_big: float = 0.0
    tau_offset_small: float = 0.0
    # ── ★ 8 个待辨识参数 ──
    Jbig_eff: float = 0.0240
    Js: float = 0.0130
    Px: float = 0.0
    Py: float = 0.0
    fc_big: float = 0.090
    fv_big: float = 0.030
    fc_small: float = 0.030
    fv_small: float = 0.008

    # ── 向量化接口（顺序与 PARAM_NAMES 一致）──
    def vector(self) -> np.ndarray:
        return np.array(
            [self.Jbig_eff, self.Js, self.Px, self.Py,
             self.fc_big, self.fv_big, self.fc_small, self.fv_small],
            dtype=np.float64,
        )

    def with_vector(self, phi) -> "PlanarParams":
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != (NPARAM,):
            raise ValueError(f"参数向量长度应为 {NPARAM}，得到 {phi.shape}")
        return replace(
            self,
            Jbig_eff=float(phi[0]), Js=float(phi[1]), Px=float(phi[2]), Py=float(phi[3]),
            fc_big=float(phi[4]), fv_big=float(phi[5]),
            fc_small=float(phi[6]), fv_small=float(phi[7]),
        )

    def geometry_copy(self, **kw) -> "PlanarParams":
        """只改几何/固定量的副本（例如把 λ 换成 plant 的 100）。"""
        return replace(self, **kw)


def _nonzero(v) -> bool:
    """判断重力分量是否非零（支持 float / ndarray / torch 张量）。"""
    if isinstance(v, np.ndarray):
        return bool(np.any(v != 0.0))
    if torch is not None and torch.is_tensor(v):
        with torch.no_grad():
            return bool(torch.any(v != 0).item())
    try:
        return bool(v != 0.0)
    except Exception:
        return True


@dataclass
class Exo:
    """外生量（ModelExo 的 python 版）。

    `gravity_a` 的分量可以是标量，也可以是**逐样本数组**（numpy 形状 [N] / torch 形状 [B]），
    此时会与状态的前导维广播 —— 用于"底盘静态倾斜、大 yaw 转动导致 A 系重力方向随之旋转"。
    """

    gravity_a: tuple = (0.0, 0.0)   # ★ A 系（大 yaw 转子系）重力平面分量 (m/s²)；水平 = (0,0)
    base_omega: float = 0.0         # 底盘绕关节轴角速度（本任务 = 0）
    base_alpha: float = 0.0         # 底盘绕关节轴角加速度（本任务 = 0）
    gravity_on: bool | None = None  # None ⇒ 自动判定（张量不能直接做 if 判断）

    def __post_init__(self):
        if self.gravity_on is None:
            gx, gy = self.gravity_a
            self.gravity_on = bool(_nonzero(gx) or _nonzero(gy))


EXO_ZERO = Exo()


def exo_from_gravity(gx, gy, base_omega: float = 0.0, base_alpha: float = 0.0) -> Exo:
    """按 A 系重力平面分量构造 Exo（自动置 gravity_on ⇒ eom 会自动启用重力项）。"""
    return Exo(gravity_a=(gx, gy), base_omega=base_omega, base_alpha=base_alpha)


# ============================================================================
# numpy 模型（plant / 前向验证 / 回归矩阵；与头文件逐项对应）
# ============================================================================
def planar_derived_np(qs, p: PlanarParams):
    """Q = R(qs)P, M11, M12, μ（支持标量或任意形状数组 qs）。"""
    cs = np.cos(qs)
    sn = np.sin(qs)
    Qx = p.Px * cs - p.Py * sn
    Qy = p.Px * sn + p.Py * cs
    dQ = p.dx * Qx + p.dy * Qy
    mu = 2.0 * (p.dy * Qx - p.dx * Qy)
    M11 = p.Jbig_eff + p.Js + 2.0 * dQ
    M12 = p.Js + dQ
    return Qx, Qy, M11, M12, mu


def friction_np(w, fc, fv, lam):
    return fc * np.tanh(lam * w) + fv * w


def eom_np(q, qd, p: PlanarParams, exo: Exo = EXO_ZERO):
    """M(2×2) 与 h(2)。q/qd 形状 [...,2]；返回 M [...,2,2], h [...,2]。"""
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    Qx, Qy, M11, M12, mu = planar_derived_np(q[..., 1], p)
    M22 = np.full_like(M11, p.Js)
    M = np.stack([np.stack([M11, M12], axis=-1), np.stack([M12, M22], axis=-1)], axis=-2)

    tb = qd[..., 0]
    ts = qd[..., 1]
    wc = exo.base_omega
    ac = exo.base_alpha

    h0 = (mu * tb * ts + 0.5 * mu * ts * ts
          + mu * ts * wc + M11 * ac
          + friction_np(tb, p.fc_big, p.fv_big, p.friction_lambda)
          + p.tau_offset_big)
    h1 = (-0.5 * mu * tb * tb
          + (-mu * tb) * wc - 0.5 * mu * wc * wc + M12 * ac
          + friction_np(ts, p.fc_small, p.fv_small, p.friction_lambda)
          + p.tau_offset_small)
    if exo.gravity_on:
        gx, gy = exo.gravity_a
        Gs = Qx * gy - Qy * gx
        Gb = p.m_u_known * (p.dx * gy - p.dy * gx) + Gs
        h0 = h0 - Gb
        h1 = h1 - Gs
    h = np.stack([h0, h1], axis=-1)
    return M, h


def forward_accel_np(q, qd, tau, p: PlanarParams, exo: Exo = EXO_ZERO):
    """q̈ = M⁻¹(τ − h)。"""
    M, h = eom_np(q, qd, p, exo)
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    inv = 1.0 / det
    r0 = tau[..., 0] - h[..., 0]
    r1 = tau[..., 1] - h[..., 1]
    qdd0 = (M[..., 1, 1] * r0 - M[..., 0, 1] * r1) * inv
    qdd1 = (-M[..., 1, 0] * r0 + M[..., 0, 0] * r1) * inv
    return np.stack([qdd0, qdd1], axis=-1)


def inverse_dynamics_np(q, qd, qdd, p: PlanarParams, exo: Exo = EXO_ZERO):
    """τ = M·q̈ + h（回归矩阵自检 / 残差指标用）。"""
    M, h = eom_np(q, qd, p, exo)
    tau0 = M[..., 0, 0] * qdd[..., 0] + M[..., 0, 1] * qdd[..., 1] + h[..., 0]
    tau1 = M[..., 1, 0] * qdd[..., 0] + M[..., 1, 1] * qdd[..., 1] + h[..., 1]
    return np.stack([tau0, tau1], axis=-1)


def rk4_step_np(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1):
    """RK4 单步（力矩零阶保持），返回 (q_next, qd_next)。"""
    substeps = max(1, int(substeps))
    hh = dt / substeps
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(substeps):
        k1 = forward_accel_np(qa, qda, tau, p, exo)
        k2 = forward_accel_np(qa + 0.5 * hh * qda, qda + 0.5 * hh * k1, tau, p, exo)
        k3 = forward_accel_np(qa + 0.5 * hh * (qda + 0.5 * hh * k1),
                              qda + 0.5 * hh * k2, tau, p, exo)
        k4 = forward_accel_np(qa + hh * (qda + 0.5 * hh * k2),
                              qda + hh * k3, tau, p, exo)
        qa = qa + (hh / 6.0) * (qda + 2.0 * (qda + 0.5 * hh * k1)
                                + 2.0 * (qda + 0.5 * hh * k2) + (qda + hh * k3))
        qda = qda + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return qa, qda


def euler_step_np(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1):
    """★ 半隐式（symplectic）欧拉单步 —— 与原仓库 `param_ident.py` 逐行同序。

        ω ← ω + dt·α(q, ω)      # 先用**当前**状态算加速度
        θ ← θ + dt·ω            # 再用**更新后**的 ω 更新角度

    （原仓库就是 `omega = omega + alpha*dt; theta = theta + omega*dt`，即半隐式欧拉。）
    """
    substeps = max(1, int(substeps))
    hh = dt / substeps
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(substeps):
        acc = forward_accel_np(qa, qda, tau, p, exo)
        qda = qda + hh * acc
        qa = qa + hh * qda
    return qa, qda


def integrate_step_np(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1,
                      integrator: str = "rk4"):
    """按名称分派积分器（"euler" | "rk4"）；未知名称报错而不是静默回退。"""
    name = str(integrator).lower()
    if name in ("euler", "semi-implicit", "semi_implicit"):
        return euler_step_np(q, qd, tau, p, exo, dt, substeps)
    if name == "rk4":
        return rk4_step_np(q, qd, tau, p, exo, dt, substeps)
    raise ValueError(f"未知积分器 {integrator!r}（支持 euler | rk4）")


def regressor_np(q, qd, qdd, p: PlanarParams, exo: Exo = EXO_ZERO):
    """解析回归矩阵 Y（形状 [...,2,8]），满足 τ = Y·φ。

    列顺序: 0 Jbig_eff, 1 Js, 2 Px, 3 Py, 4 fc_big, 5 fv_big, 6 fc_small, 7 fv_small
    （与 include/tcbs/mpc/planar_yaw_model.h::regressor() 完全一致；重力/底盘项为通用写法）
    """
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    qdd = np.asarray(qdd, dtype=np.float64)
    ts = q[..., 1]
    db = qdd[..., 0]
    ds = qdd[..., 1]
    tb = qd[..., 0]
    vs = qd[..., 1]
    cs = np.cos(ts)
    sn = np.sin(ts)
    gx, gy = exo.gravity_a
    wc = exo.base_omega
    ac = exo.base_alpha
    lam = p.friction_lambda

    shape = q.shape[:-1] + (2, NPARAM)
    Y = np.zeros(shape, dtype=np.float64)

    # 0: Jbig_eff（仅 M11；α_c 项来自 M11·α_c）
    Y[..., 0, 0] = db + ac
    # 1: Js（M11 / M12 / M22）
    Y[..., 0, 1] = db + ds + ac
    Y[..., 1, 1] = db + ds + ac
    # 2,3: Px, Py（经 Q=R(θs)P、μ=2(dy·Qx − dx·Qy) 进入 M 与 h）
    dQdPx = p.dx * cs + p.dy * sn
    dQdPy = -p.dx * sn + p.dy * cs
    mudPx = 2.0 * (p.dy * cs - p.dx * sn)
    mudPy = -2.0 * (p.dy * sn + p.dx * cs)
    dGsdPx = cs * gy - sn * gx
    dGsdPy = -sn * gy - cs * gx

    def _p_term(idx, dQdP, dmu, dGs):
        dM11 = 2.0 * dQdP
        dM12 = dQdP
        Y[..., 0, idx] = (dM11 * db + dM12 * ds + dmu * tb * vs + 0.5 * dmu * vs * vs
                          - dGs + dmu * vs * wc + dM11 * ac)
        Y[..., 1, idx] = (dM12 * db - 0.5 * dmu * tb * tb
                          - dGs + (-dmu * tb) * wc - 0.5 * dmu * wc * wc + dM12 * ac)

    _p_term(2, dQdPx, mudPx, dGsdPx)
    _p_term(3, dQdPy, mudPy, dGsdPy)
    # 4..7: 摩擦（只作用于本轴）
    Y[..., 0, 4] = np.tanh(lam * tb)
    Y[..., 0, 5] = tb
    Y[..., 1, 6] = np.tanh(lam * vs)
    Y[..., 1, 7] = vs
    return Y


def simulate_np(p: PlanarParams, q0, qd0, tau_seq, dt, exo: Exo = EXO_ZERO, substeps: int = 1,
                exo_seq=None, integrator: str = "rk4"):
    """numpy 前向仿真（力矩零阶保持）。tau_seq [T,2]；返回 theta [T,2], dtheta [T,2]。

    integrator = "rk4"（默认，旧行为，compare_ident_methods 里的 plant 用）或 "euler"
    （★ 辨识默认配方，与原仓库一致）。
    exo_seq 非 None 时按步使用 exo_seq[i]（静态倾斜下 A 系重力方向随 θ_b 旋转 ⇒ 逐样本重力）。
    """
    tau_seq = np.asarray(tau_seq, dtype=np.float64)
    T = tau_seq.shape[0]
    q = np.array(q0, dtype=np.float64)
    qd = np.array(qd0, dtype=np.float64)
    th = np.zeros((T,) + q.shape, dtype=np.float64)
    dth = np.zeros_like(th)
    for i in range(T):
        th[i] = q
        dth[i] = qd
        q, qd = integrate_step_np(q, qd, tau_seq[i], p,
                                  exo if exo_seq is None else exo_seq[i],
                                  dt, substeps, integrator)
    return th, dth


# ============================================================================
# torch 模型（float64，逐项镜像上面的 numpy 版；用于可导前向仿真）
# ============================================================================
def torch_derived(qs, p: PlanarParams):
    cs = torch.cos(qs)
    sn = torch.sin(qs)
    Qx = p.Px * cs - p.Py * sn
    Qy = p.Px * sn + p.Py * cs
    dQ = p.dx * Qx + p.dy * Qy
    mu = 2.0 * (p.dy * Qx - p.dx * Qy)
    M11 = p.Jbig_eff + p.Js + 2.0 * dQ
    M12 = p.Js + dQ
    return Qx, Qy, M11, M12, mu


def torch_eom(q, qd, p: PlanarParams, exo: Exo = EXO_ZERO):
    """与 eom_np 逐项一致；对「重力=0、底盘静止」的常见工况跳过相应张量运算（纯提速）。

    返回 (M11, M12, h) —— M22 = p.Js 是标量，由 torch_forward_accel 直接使用，
    少建一个 M22 张量、少两次 stack（对 10 万次级前向仿真很重要）。
    """
    cs = torch.cos(q[..., 1])
    sn = torch.sin(q[..., 1])
    Qx = p.Px * cs - p.Py * sn
    Qy = p.Px * sn + p.Py * cs
    dQ = p.dx * Qx + p.dy * Qy
    mu = 2.0 * (p.dy * Qx - p.dx * Qy)
    M11 = (p.Jbig_eff + p.Js) + 2.0 * dQ
    M12 = p.Js + dQ

    tb = qd[..., 0]
    ts = qd[..., 1]
    half_mu = 0.5 * mu
    h0 = mu * ts * (tb + 0.5 * ts)            # = μ·θ̇b·θ̇s + ½μ·θ̇s²
    h1 = -(half_mu * tb) * tb                 # = −½μ·θ̇b²
    if exo.gravity_on:
        gx, gy = exo.gravity_a
        Gs = Qx * gy - Qy * gx
        Gb = p.m_u_known * (p.dx * gy - p.dy * gx) + Gs
        h0 = h0 - Gb
        h1 = h1 - Gs
    wc = exo.base_omega
    ac = exo.base_alpha
    if wc != 0.0:
        h0 = h0 + (mu * ts) * wc
        h1 = h1 + (-mu * tb) * wc - (half_mu * wc) * wc
    if ac != 0.0:
        h0 = h0 + M11 * ac
        h1 = h1 + M12 * ac
    h0 = h0 + p.fc_big * torch.tanh(p.friction_lambda * tb) + p.fv_big * tb + p.tau_offset_big
    h1 = h1 + p.fc_small * torch.tanh(p.friction_lambda * ts) + p.fv_small * ts + p.tau_offset_small
    return M11, M12, torch.stack([h0, h1], dim=-1)


def torch_forward_accel(q, qd, tau, p: PlanarParams, exo: Exo = EXO_ZERO):
    M11, M12, h = torch_eom(q, qd, p, exo)
    Js = p.Js                                  # M22
    det = M11 * Js - M12 * M12
    inv = 1.0 / det
    r0 = tau[..., 0] - h[..., 0]
    r1 = tau[..., 1] - h[..., 1]
    qdd0 = (Js * r0 - M12 * r1) * inv
    qdd1 = (M11 * r1 - M12 * r0) * inv
    return torch.stack([qdd0, qdd1], dim=-1)


def torch_rk4_step(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1):
    hh = dt / max(1, int(substeps))
    for _ in range(max(1, int(substeps))):
        k1 = torch_forward_accel(q, qd, tau, p, exo)
        k2 = torch_forward_accel(q + 0.5 * hh * qd, qd + 0.5 * hh * k1, tau, p, exo)
        k3 = torch_forward_accel(q + 0.5 * hh * (qd + 0.5 * hh * k1),
                                 qd + 0.5 * hh * k2, tau, p, exo)
        k4 = torch_forward_accel(q + hh * (qd + 0.5 * hh * k2), qd + hh * k3, tau, p, exo)
        q = q + (hh / 6.0) * (qd + 2.0 * (qd + 0.5 * hh * k1)
                              + 2.0 * (qd + 0.5 * hh * k2) + (qd + hh * k3))
        qd = qd + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return q, qd


def torch_euler_step(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1):
    """★ 半隐式欧拉单步（可导版，与原仓库 `omega += alpha*dt; theta += omega*dt` 同序）。"""
    n = max(1, int(substeps))
    hh = dt / n
    for _ in range(n):
        acc = torch_forward_accel(q, qd, tau, p, exo)
        qd = qd + hh * acc
        q = q + hh * qd
    return q, qd


def torch_integrate_step(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1,
                         integrator: str = "rk4"):
    """按名称分派可导积分器（"euler" | "rk4"）。"""
    name = str(integrator).lower()
    if name in ("euler", "semi-implicit", "semi_implicit"):
        return torch_euler_step(q, qd, tau, p, exo, dt, substeps)
    if name == "rk4":
        return torch_rk4_step(q, qd, tau, p, exo, dt, substeps)
    raise ValueError(f"未知积分器 {integrator!r}（支持 euler | rk4）")


def torch_rollout(p: PlanarParams, q0, qd0, tau_seq, dt, exo: Exo = EXO_ZERO, substeps: int = 1,
                  grav_seq=None, integrator: str = "rk4"):
    """可导前向仿真。tau_seq [T,B,2], q0/qd0 [B,2]；返回 theta [T,B,2], dtheta [T,B,2]。

    integrator: "rk4"（默认，旧行为）| "euler"（★ 辨识默认配方，与原仓库一致）。
    grav_seq [T,B,2] 非 None 时逐步使用该步的 A 系重力平面分量（静态倾斜 + 大 yaw 转动）。
    """
    T = tau_seq.shape[0]
    q = q0
    qd = qd0
    th = []
    dth = []
    for i in range(T):
        th.append(q)
        dth.append(qd)
        if grav_seq is None:
            ex = exo
        else:
            ex = exo_from_gravity(grav_seq[i, :, 0], grav_seq[i, :, 1],
                                  exo.base_omega, exo.base_alpha)
        q, qd = torch_integrate_step(q, qd, tau_seq[i], p, ex, dt, substeps, integrator)
    return torch.stack(th, dim=0), torch.stack(dth, dim=0)


# ============================================================================
# 角度预处理（量化 → 中心差分 + 3 点平滑）
# ============================================================================
def quantize(theta, step: float = QUANT_STEP):
    """编码器量化（模拟 8192 计数/整圈的读数）。"""
    return np.round(np.asarray(theta, dtype=np.float64) / step) * step


def central_diff(x, dt):
    """二阶精度中心差分（端点用单侧二阶公式）。x [T,...]"""
    x = np.asarray(x, dtype=np.float64)
    d = np.zeros_like(x)
    if x.shape[0] < 3:
        if x.shape[0] == 2:
            d[0] = (x[1] - x[0]) / dt
            d[1] = d[0]
        return d
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (-3.0 * x[0] + 4.0 * x[1] - x[2]) / (2.0 * dt)
    d[-1] = (3.0 * x[-1] - 4.0 * x[-2] + x[-3]) / (2.0 * dt)
    return d


def smooth3(x):
    """3 点滑动平均（端点复制边界）。x [T,...]"""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        return x.copy()
    y = x.copy()
    y[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    y[0] = (x[0] + x[1]) / 2.0
    y[-1] = (x[-1] + x[-2]) / 2.0
    return y


def derivatives_from_angles(theta, dt, smooth: bool = True):
    """量化角 → (dθ, d²θ)：中心差分 + 3 点平滑（与 compare_ident_methods 的预处理一致）。"""
    theta = np.asarray(theta, dtype=np.float64)
    v = central_diff(theta, dt)
    a = central_diff(v, dt)          # 对未平滑的一阶差分再差分
    if smooth:
        v = smooth3(v)
        a = smooth3(a)
    return v, a


# ============================================================================
# 数据段
# ============================================================================
@dataclass
class Segment:
    """一段采集数据（两轴都有记录；axis 标明哪一轴被激励）。"""

    t: np.ndarray                 # [T]
    theta: np.ndarray             # [T,2] 关节角（实测 = 量化后）
    dtheta: np.ndarray            # [T,2]
    tau: np.ndarray               # [T,2]
    axis: int = AXIS_BIG          # 0 = 大 yaw 被激励, 1 = 小 yaw 被激励
    held_target: float = 0.0
    dt: float = DT_DEFAULT
    mcu2_seq: np.ndarray | None = None
    gravity: np.ndarray | None = None      # [T,2] A 系重力平面分量 (m/s²)；None = 水平(0,0)
    ddtheta: np.ndarray | None = None      # [T,2]（可选；oracle 消融用）
    theta_true: np.ndarray | None = None   # [T,2]（仅仿真数据有；评测用）
    dtheta_true: np.ndarray | None = None
    ddtheta_true: np.ndarray | None = None
    source: str = "?"
    tag: str = ""

    @property
    def T(self) -> int:
        return int(self.theta.shape[0])

    @property
    def driven(self) -> int:
        return self.axis

    @property
    def held(self) -> int:
        return 1 - self.axis


def _axis_code(v) -> int:
    """axis 列的容错解析: 0/1、'0'/'1'、'big'/'small' 都接受。"""
    if isinstance(v, (str, bytes, np.str_, np.bytes_)):
        return AXIS_SMALL if str(v).lower().startswith("s") else AXIS_BIG
    try:
        return int(np.asarray(v).reshape(-1)[0])
    except (TypeError, ValueError):
        return AXIS_BIG


def _col(rec, names, n=None):
    for nm in names:
        if nm in rec and rec[nm] is not None:
            return np.asarray(rec[nm], dtype=np.float64)
    raise KeyError(f"缺少列 {names}")


def segment_from_arrays(rec: dict, dt: float, source: str = "?") -> Segment:
    """把「同名列数组」字典转成 Segment（CSV 与 npz 共用）。"""
    if "theta" in rec and np.ndim(rec["theta"]) == 2:
        theta = np.asarray(rec["theta"], dtype=np.float64)[:, :2]
    else:
        tb = _col(rec, ["theta_big", "theta_b"])
        ts = _col(rec, ["theta_small", "theta_s"])
        theta = np.stack([tb, ts], axis=-1)
    if "tau" in rec and np.ndim(rec["tau"]) == 2:
        tau = np.asarray(rec["tau"], dtype=np.float64)[:, :2]
    else:
        tau = np.stack([_col(rec, ["tau_big", "tau_b"]), _col(rec, ["tau_small", "tau_s"])], axis=-1)
    if "dtheta" in rec and np.ndim(rec.get("dtheta")) == 2:
        dtheta = np.asarray(rec["dtheta"], dtype=np.float64)[:, :2]
    elif rec.get("dtheta_big") is not None and rec.get("dtheta_small") is not None:
        dtheta = np.stack([_col(rec, ["dtheta_big"]), _col(rec, ["dtheta_small"])], axis=-1)
    else:
        # 缺角速度列 → 用「中心差分 + 3 点平滑」补算（量化噪声下必须平滑）
        dtheta = np.stack([derivatives_from_angles(theta[:, 0], dt)[0],
                           derivatives_from_angles(theta[:, 1], dt)[0]], axis=-1)
    if not np.all(np.isfinite(dtheta)):
        dtheta = np.nan_to_num(dtheta, nan=0.0, posinf=0.0, neginf=0.0)

    T = theta.shape[0]
    t = _col(rec, ["t"]) if ("t" in rec and rec["t"] is not None) else np.arange(T) * dt
    axis = _axis_code(rec.get("axis", AXIS_BIG))
    held = float(np.asarray(rec.get("held_target", 0.0)).reshape(-1)[0])
    seq = rec.get("mcu2_seq")
    seq = None if seq is None else np.asarray(seq, dtype=np.float64)
    # ── A 系重力平面分量（静态倾斜时非零；水平时全 0）──
    grav = None
    if "gravity" in rec and np.ndim(rec.get("gravity")) == 2:
        grav = np.asarray(rec["gravity"], dtype=np.float64)[:, :2]
    elif rec.get("gravity_ax") is not None and rec.get("gravity_ay") is not None:
        grav = np.stack([np.asarray(rec["gravity_ax"], dtype=np.float64),
                         np.asarray(rec["gravity_ay"], dtype=np.float64)], axis=-1)
    if grav is not None and (grav.shape[0] != theta.shape[0] or not np.any(grav != 0.0)):
        grav = None if not np.any(grav != 0.0) else grav
    dd = rec.get("ddtheta")
    dd = None if dd is None else np.asarray(dd, dtype=np.float64)
    th_true = rec.get("theta_true")
    th_true = None if th_true is None else np.asarray(th_true, dtype=np.float64)
    dth_true = rec.get("dtheta_true")
    dth_true = None if dth_true is None else np.asarray(dth_true, dtype=np.float64)
    dd_true = rec.get("ddtheta_true")
    dd_true = None if dd_true is None else np.asarray(dd_true, dtype=np.float64)
    return Segment(t=t, theta=theta, dtheta=dtheta, tau=tau, axis=axis, held_target=held,
                   dt=dt, mcu2_seq=seq, gravity=grav, ddtheta=dd, theta_true=th_true,
                   dtheta_true=dth_true, ddtheta_true=dd_true, source=source)


def _read_csv(path: str) -> list:
    rows = []
    with open(path, "r", newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        first_fields = [s.strip() for s in first.strip().split(",")]
        try:
            [float(s) for s in first_fields]
            has_header = False
        except ValueError:
            has_header = True
        if has_header:
            rd = csv.DictReader(fh)
            for r in rd:
                rows.append({k.strip(): (float(v) if v not in (None, "") else np.nan)
                             for k, v in r.items() if k is not None})
        else:
            names = ["t", "theta_big", "theta_small", "dtheta_big", "dtheta_small",
                     "tau_big", "tau_small", "axis", "held_target", "mcu2_seq"]
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                vals = [float(s) for s in line.split(",")]
                rows.append({nm: vals[i] for i, nm in enumerate(names) if i < len(vals)})
    if not rows:
        return []
    cols = {k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64) for k in rows[0]}
    for k in ("dtheta_big", "dtheta_small", "mcu2_seq"):
        if k in cols and np.all(np.isnan(cols[k])):
            cols.pop(k)
    return [cols]


def load_segments(patterns, dt_override: float | None = None, verbose: bool = True) -> list:
    """读入若干 csv/npz（glob 或显式路径），返回 Segment 列表。"""
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pat in patterns:
        for p in str(pat).split(","):
            p = p.strip()
            if not p:
                continue
            hit = sorted(globmod.glob(p))
            if not hit and os.path.isfile(p):
                hit = [p]
            files.extend(hit)
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    segs = []
    skipped = []
    for path in files:
        # 单个文件格式不符（列名不匹配/损坏）时只跳过它，不让整次拟合失败
        try:
            if path.endswith(".npz"):
                d = np.load(path, allow_pickle=False)
                rec = {k: d[k] for k in d.files if d[k].ndim >= 1}
                scal = {}
                for k in d.files:                       # 标量元数据（非数值的如 tag 自动忽略）
                    if d[k].ndim == 0:
                        try:
                            scal[k] = float(d[k])
                        except (TypeError, ValueError):
                            pass
                dt = dt_override or float(scal.get("dt", DT_DEFAULT))
                if "axis" in scal:
                    rec["axis"] = scal["axis"]
                if "held_target" in scal:
                    rec["held_target"] = scal["held_target"]
                segs.append(segment_from_arrays(rec, dt, source=path))
            elif path.endswith(".csv"):
                got = 0
                for cols in _read_csv(path):
                    dt = dt_override or DT_DEFAULT
                    if "t" in cols and len(cols["t"]) > 2:
                        dts = np.diff(cols["t"])
                        dts = dts[np.isfinite(dts) & (dts > 1e-6)]
                        if dts.size:
                            dt = dt_override or float(np.median(dts))
                    segs.append(segment_from_arrays(cols, dt, source=path))
                    got += 1
                if got == 0:
                    raise ValueError("空文件")
            else:
                skipped.append((path, "不支持的后缀"))
        except Exception as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
    if verbose:
        for path, why in skipped:
            print(f"[load] 跳过 {path} —— {why}")
        print(f"[load] 共读入 {len(segs)} 段数据（{len(files)} 个文件，跳过 {len(skipped)} 个）")
    return segs


# ============================================================================
# 参数重参数化（★ 原仓库同款：log 参数化，无任何上下界）
# ============================================================================
class ParamSpace:
    """无约束变量 raw ↔ 物理参数 φ 的映射 —— **原仓库 `param_ident.py` 同款**。

    ★ **没有任何上下界 / clamp / 投影 / 惩罚**:

      · 天然为正的参数（`POSITIVE_PARAM` = Jbig_eff, Js, fc_big, fv_big, fc_small, fv_small）:
            φ_j = exp(raw_j)          逆映射 raw_j = log(φ_j)
        ——正性由参数化**隐式**保证（原仓库就是 `J=exp(log_J)`, `τ_c=exp(log_tau_c)`,
          `b=exp(log_b)`），优化中永远不会出现负惯量/负摩擦，但**也没有上限**。
      · `Px, Py`（可正可负）: φ_j = raw_j，直接自由。

    冻结参数（fit_axis 只拟合单轴 / p_constraint）直接用初值常量，不参与优化。

    ★ `p_along_d`（可选，compare_ident_methods.py 用）: 小 yaw 零点已按"离心平衡点"标定后
      模型必然满足 `P = |P|·d̂` ⇒ `Px/Py` 不是两个自由参数，而是**一个自由标量 s = |P|**:
          Px = s·ux,  Py = s·uy,   (ux, uy) = d/|d|
      此时 `Px/Py` 从 free_idx 里移除，raw 向量末尾追加 1 个分量给 s（s 同样**自由、无界**，
      符号不强制为正：物理上应为正，让优化自己决定，可发现"标定取到了不稳定平衡点"）。
    """

    def __init__(self, free_mask: np.ndarray | None = None, p_along_d=None,
                 positive_mask: np.ndarray | None = None):
        self.positive = (POSITIVE_PARAM.copy() if positive_mask is None
                         else np.asarray(positive_mask, dtype=bool).reshape(NPARAM).copy())
        fm = (np.ones(NPARAM, dtype=bool) if free_mask is None
              else np.asarray(free_mask, dtype=bool).reshape(NPARAM).copy())
        if p_along_d is not None:
            u = np.asarray(p_along_d, dtype=np.float64).reshape(2)
            n = float(np.hypot(u[0], u[1]))
            if n < 1e-12:
                raise ValueError("p_along_d 需要 |d| > 0（几何偏置为 0 时方向无定义）")
            self.p_along_d = (u[0] / n, u[1] / n)
            fm[2] = fm[3] = False
        else:
            self.p_along_d = None
        self.free_mask = fm
        self.free_idx = np.nonzero(fm)[0]
        # 自由参数里哪些走 log 参数化
        self.free_positive = self.positive[self.free_idx]

    @property
    def nfree(self) -> int:
        return int(self.free_idx.size) + (0 if self.p_along_d is None else 1)

    @property
    def mode(self) -> str:
        return "along_d" if self.p_along_d is not None else "free/slots"

    def describe(self) -> str:
        """给日志用的说明（哪些参数走 log、哪些自由）。"""
        pos = ",".join(PARAM_NAMES[j] for j in self.free_idx if self.positive[j])
        fre = ",".join(PARAM_NAMES[j] for j in self.free_idx if not self.positive[j])
        parts = []
        if pos:
            parts.append(f"log 参数化: {pos}")
        if fre:
            parts.append(f"自由参数: {fre}")
        if self.p_along_d is not None:
            parts.append(f"P=|P|·d̂(d̂=({self.p_along_d[0]:+.4f},{self.p_along_d[1]:+.4f})) 单标量")
        return "; ".join(parts) if parts else "（全部冻结）"

    def to_raw_init(self, phi0: np.ndarray) -> np.ndarray:
        """φ0 → raw 初值（log 域）。**只对初值取对数**，不给参数加任何界。"""
        phi0 = np.asarray(phi0, dtype=np.float64)
        raw = []
        for j in self.free_idx:
            v = float(phi0[j])
            if self.positive[j]:
                # 初值非正时（不该发生）取一个极小正数做对数的**起点**，而非给参数加界
                raw.append(math.log(v if v > 0.0 else 1e-6))
            else:
                raw.append(v)
        if self.p_along_d is not None:
            ux, uy = self.p_along_d
            raw.append(float(phi0[2]) * ux + float(phi0[3]) * uy)   # CAD 初值常为 0
        return np.array(raw, dtype=np.float64)

    def to_physical(self, raw, phi0: np.ndarray):
        """raw: np 或 torch 张量（长度 = nfree）→ 长度 8 的物理参数（同类型，无 clamp）。"""
        phi0 = np.asarray(phi0, dtype=np.float64)
        n_f = int(self.free_idx.size)
        if isinstance(raw, np.ndarray):
            out = phi0.copy()
            for k, j in enumerate(self.free_idx):
                out[j] = math.exp(float(raw[k])) if self.positive[j] else float(raw[k])
            if self.p_along_d is not None:
                s = float(raw[n_f])
                out[2] = s * self.p_along_d[0]
                out[3] = s * self.p_along_d[1]
            return out
        out = [None] * NPARAM
        for k, j in enumerate(self.free_idx):
            out[j] = torch.exp(raw[k]) if self.positive[j] else raw[k]
        if self.p_along_d is not None:
            s = raw[n_f]
            out[2] = s * self.p_along_d[0]
            out[3] = s * self.p_along_d[1]
        for j in range(NPARAM):
            if out[j] is None:
                out[j] = torch.as_tensor(phi0[j], dtype=raw.dtype, device=raw.device)
        return torch.stack(out)


def params_from_torch(phi_vec, base: PlanarParams) -> PlanarParams:
    return base.with_vector(np.asarray([float(v) for v in phi_vec]))


# ============================================================================
# 拟合器
# ============================================================================
@dataclass
class FitConfig:
    """拟合配置。

    ★★ 默认值 = **原仓库 `param_ident.py` 同款配方**（见文件头 docstring）::

        epochs=1000（= 原仓库 num_epochs 1000）, seg_steps=10（0.1 s 随机片段）,
        loss_mode="mse"（角度 wrap 后 MSE + 角速度 MSE 等权）, lr=3e-4（常数，无 scheduler）,
        integrator="euler"（半隐式欧拉）, substeps=1, lbfgs_iters=0, window_len=0,
        batch_size=0, windows_per_seg=1, iters=0（⇒ 走新配方；>0 才回到旧配方）。

    字段名全部保留（`compare_ident_methods.py` 在用 `iters/lbfgs_iters/lr/huber_delta/
    window_len/batch_size/p_constraint/p_zero_angle_deg/vel_weight` 等），只改了默认值。
    """

    fit_axis: str = "both"          # both | big | small
    # ── ★ 新配方（原仓库同款）──
    epochs: int = 1000              # ★ 与**原仓库同轮数**（num_epochs=1000）；每 epoch 每段 1 个 Adam 步
    seg_steps: int = 10             # ★ 每个优化步随机截取的片段长度（10 步 = 0.1 s @100 Hz）
    loss_mode: str = "mse"          # mse = 角度(wrap)MSE + 角速度 MSE 等权（原仓库）| huber = 旧配方
    integrator: str = "euler"       # euler = 半隐式欧拉（原仓库）| rk4 = 旧精细配方
    lr_schedule: str = "none"       # none = 常数 lr（原仓库）| cosine = 旧配方
    # ── 旧配方（显式给 iters>0 才启用；或 --legacy-recipe 一键预设）──
    iters: int = 0                  # >0 ⇒ 旧配方：精确跑 iters 个 Adam 步（分窗 mini-batch）
    lbfgs_iters: int = 0            # LBFGS 迭代数（0 = 关闭 ⇒ 原仓库没有 LBFGS）
    lbfgs_max_iter: int = 5
    lr: float = 3e-4                # ★ 原仓库 lr=3e-4（旧配方默认 5e-3）
    seed: int = 42
    substeps: int = 1               # 可导仿真每控制步的积分子步
    huber_delta: float = 2.0e-3     # rad（~ 量化步长的 3 倍）；仅 loss_mode="huber" 用
    vel_weight: float = 0.0         # 旧配方的角速度项权重（0 = 不用）；mse 模式**固定等权 1.0**
    vel_huber_delta: float = 0.05   # rad/s
    free_init_vel: bool = False     # 是否把各段初始角速度当作自由参数一起优化
    p_bound: float = 0.05           # ★ 已废弃并被忽略：本脚本不再有任何参数限位（仅为字段兼容保留）
    init_vector: np.ndarray | None = None
    truth_vector: np.ndarray | None = None  # 真值/参考值（画收敛曲线虚线用；None ⇒ 用初值）
    print_every: int = 50
    verbose: bool = True
    dtype: "torch.dtype" = None
    device: str = "cpu"
    max_points: int = 0             # >0 时每段只取前 max_points 个点（加速调试）
    # ── 数据分窗与 mini-batch（**旧配方**的输出误差法可扩展做法）──
    p_constraint: str = "free"      # free | along_d（P=|P|·R(−θ*)·d̂，单参数）| zero（固定 0）
    p_zero_angle_deg: float = 0.0   # ★ 平衡点(θ*)在当前零点坐标系里的读数（度）: 0 = 零点就在平衡点
    fix_p: bool = False             # 兼容旧参数：等价于 p_constraint = "zero"
    window_len: int = 0             # 每个窗口的点数（0 = 整段作一个窗口 ⇒ 新配方不用分窗）
    windows_per_seg: int = 1        # 每段切几个窗口（仅旧配方 + window_len>0 时有效）
    batch_size: int = 0             # 旧配方 Adam 每步随机抽几个窗口（0 = 全部）；LBFGS 一律全批

    def __post_init__(self):
        if self.fix_p:
            self.p_constraint = "zero"
        self.loss_mode = str(self.loss_mode).lower()
        self.integrator = str(self.integrator).lower()
        self.lr_schedule = str(self.lr_schedule).lower()

    # ── 便捷判断/预设 ──
    @property
    def use_legacy_path(self) -> bool:
        """iters > 0 ⇒ 走旧的"精确 iters 步 + 分窗/mini-batch/LBFGS"路径。"""
        return int(self.iters) > 0

    def legacy_recipe(self) -> "FitConfig":
        """把本配置切换成旧的"精细配方"（Huber + 分窗 + LBFGS + lr=5e-3 + iters=400 + RK4）。"""
        self.iters = 400
        self.epochs = 0
        self.seg_steps = 0
        self.lbfgs_iters = 25
        self.lr = 5e-3
        self.loss_mode = "huber"
        self.integrator = "rk4"
        self.lr_schedule = "cosine"
        self.window_len = 150
        self.windows_per_seg = 2
        self.batch_size = 4
        self.vel_weight = 0.0
        return self


@dataclass
class FitResult:
    """拟合结果（★ 新增字段全部带默认值，旧字段名/顺序不变）。"""

    phi: np.ndarray
    phi0: np.ndarray
    loss_history: list
    val_loss: float
    n_iter: int
    seconds: float
    rank_ratio: float = float("nan")
    # ── 收敛曲线用（新配方）──
    param_history: list = None      # 每个 epoch/step 的 8 个参数（list of list）
    epoch_losses: list = None       # 每个 epoch 的平均 loss（= loss_history 的别名，便于画图）
    n_free: int = 0                 # 自由参数个数
    n_steps: int = 0                # 实际 Adam 步数
    recipe: str = ""                # "原仓库配方(epochs×段数)" 或 "旧配方(iters 步)"
    truth: np.ndarray | None = None  # 真值/参考值（画虚线用；无则 None）
    config: dict = None             # 配置摘要（打印/画图用）


def _pack_windows(segs, cfg: FitConfig, axis_sel, dtype, dev):
    """把数据切成等长**窗口**并打包成张量（Adam 每步随机抽几个窗口 ⇒ 同样的算力下
    迭代次数更多、且能覆盖整段轨迹而不是只截前面一段）。

    window_len = 0 时每个数据段就是一个窗口（等价于"整段拟合"）。
    返回 dict: tau/theta/dtheta [L,W,2], mask [L,W], q0/qd0 [W,2], w_axis [W,2]
    以及每个窗口的来源信息。
    """
    dt = segs[0].dt
    for s in segs:
        if abs(s.dt - dt) > 1e-12:
            raise ValueError("所有数据段的 dt 必须一致")

    # 每个窗口的 (段, 起点)；window_len=0 ⇒ 整段，windows_per_seg 个等间隔窗口
    specs = []
    for si, s in enumerate(segs):
        n = s.T if cfg.max_points <= 0 else min(s.T, cfg.max_points)
        if cfg.window_len <= 0 or cfg.window_len >= n:
            starts = [0]
            L = n
        else:
            L = int(cfg.window_len)
            k = max(1, int(cfg.windows_per_seg))
            if k == 1:
                starts = [0]
            else:
                starts = sorted({int(round(v)) for v in np.linspace(0, n - L, k)})
        for st in starts:
            specs.append((si, st, L))
    L = max(sp[2] for sp in specs)
    W = len(specs)

    tau = np.zeros((L, W, 2))
    theta = np.zeros((L, W, 2))
    dtheta = np.zeros((L, W, 2))
    grav = np.zeros((L, W, 2))
    mask = np.zeros((L, W))
    q0 = np.zeros((W, 2))
    qd0 = np.zeros((W, 2))
    w_axis = np.zeros((W, 2))
    for wi, (si, st, l) in enumerate(specs):
        s = segs[si]
        sl = slice(st, st + l)
        tau[:l, wi] = s.tau[sl]
        theta[:l, wi] = s.theta[sl]
        dtheta[:l, wi] = s.dtheta[sl]
        if s.gravity is not None:
            grav[:l, wi] = s.gravity[sl]
        mask[:l, wi] = 1.0
        q0[wi] = s.theta[st]
        qd0[wi] = s.dtheta[st]
        for a in axis_sel:
            w_axis[wi, a] = 1.0 / len(axis_sel)

    t = lambda x: torch.tensor(x, dtype=dtype, device=dev)          # noqa: E731
    return {"tau": t(tau), "theta": t(theta), "dtheta": t(dtheta), "mask": t(mask),
            "q0": t(q0), "qd0": t(qd0), "w_axis": t(w_axis), "grav": t(grav),
            "has_gravity": bool(np.any(grav != 0.0)),
            "dt": dt, "L": L, "W": W,
            "lens": [int(sp[2]) for sp in specs],          # 每个窗口的**有效**长度（未 padding）
            "specs": [("seg%d" % sp[0], sp[1], sp[2]) for sp in specs],
            "seg_of_window": [sp[0] for sp in specs]}


def _config_summary(cfg: FitConfig, space: "ParamSpace", W: int, dt: float,
                    n_free: int) -> dict:
    """配置摘要（打印 + 存进 FitResult，供收敛曲线标题/报告使用）。"""
    if cfg.use_legacy_path:
        recipe = f"旧精细配方：Adam {cfg.iters} 步(lr={cfg.lr:g}, {cfg.lr_schedule})"
        if cfg.lbfgs_iters > 0:
            recipe += f" + LBFGS {cfg.lbfgs_iters} 步"
        recipe += f"，{cfg.loss_mode} 损失，{cfg.integrator.upper()} 积分"
    else:
        recipe = (f"原仓库配方：epochs={cfg.epochs} × 段数{W} 个 Adam 步"
                  f"（epochs = 原仓库 1000 的 1/5），每次 {cfg.seg_steps} 步(0.1 s)随机片段，"
                  f"lr={cfg.lr:g}（常数），损失 = 角度wrap MSE + 角速度 MSE(等权)，"
                  f"{cfg.integrator.upper()} 积分，无限位(log 参数化)")
    return {"recipe": recipe, "epochs": int(cfg.epochs), "iters": int(cfg.iters),
            "seg_steps": int(cfg.seg_steps), "lr": float(cfg.lr), "loss_mode": cfg.loss_mode,
            "integrator": cfg.integrator, "substeps": int(cfg.substeps),
            "lr_schedule": cfg.lr_schedule, "lbfgs_iters": int(cfg.lbfgs_iters),
            "fit_axis": cfg.fit_axis, "p_constraint": cfg.p_constraint,
            "n_free": int(n_free), "n_sample": int(W), "dt": float(dt),
            "limits": "无（无上下界 / 无 clamp / 无投影 / 无惩罚项）",
            "param_space": space.describe(),
            "init_source": ("默认 CAD/占位初值 defaultModelParams()" if cfg.init_vector is None
                            else "用户 --init-vector")}


def fit_params_torch(segs, cfg: FitConfig, base: PlanarParams, exo: Exo = EXO_ZERO) -> FitResult:
    """★ 输出误差法拟合：用记录力矩做可导前向仿真，最小化预测误差。

    **默认（新）配方 = 原仓库 `param_ident.py` 同款**（详见文件头 docstring）::

        for epoch in range(epochs):                  # ★ epochs=200 = 原 1000 的 1/5
            for sample in samples:                   # = 每个数据段
                s = randint(0, T - seg_steps)        # ★ 随机截取 10 步（0.1 s）片段
                θ_sim, ω_sim = rollout(φ, τ[s:s+10], θ[s], ω[s])   # ★ 半隐式欧拉, dt 取自数据
                err = wrap(θ_sim − θ_meas)           # ★ 先 wrap 到 (−π, π]
                loss = mean(err²) + mean((ω_sim − ω_meas)²)        # ★ 两项等权 MSE
                opt.zero_grad(); loss.backward(); opt.step()       # ★ Adam, lr=3e-4（常数）

    **无任何参数限位**：天然为正的 6 个参数走 log 参数化（φ = exp(raw)），Px/Py 自由；
    没有 clamp / box 约束 / 投影 / 惩罚项。每个 epoch 记录平均 loss 与当前 8 个参数
    （`loss_history` / `param_history`），供画收敛曲线。

    回到旧的"精细配方"（精确 iters 个 Adam 步 + 分窗 mini-batch + Huber + LBFGS + RK4）:
    设 `cfg.iters > 0`（或 `cfg.legacy_recipe()` / CLI `--legacy-recipe`）⇒ 走下面的旧分支。
    """
    if torch is None:
        raise RuntimeError(f"需要 torch: {_TORCH_IMPORT_ERROR}")
    if not segs:
        raise ValueError("没有可用的数据段")
    dtype = cfg.dtype or torch.float64
    dev = torch.device(cfg.device)
    t_start = time.time()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 自由参数掩码 + P 的处理方式（★ 只有"冻结/方向"，没有任何"限位"）──
    free = np.ones(NPARAM, dtype=bool)
    if cfg.fit_axis == "big":          # 只拟合大 yaw：小 yaw 摩擦不可辨识 → 冻结
        free[6] = free[7] = False
    elif cfg.fit_axis == "small":      # 只拟合小 yaw：Jbig_eff 不进小 yaw 方程 → 冻结
        free[0] = free[4] = free[5] = False
    elif cfg.fit_axis != "both":
        raise ValueError("fit_axis 必须是 both / big / small")
    mode = str(cfg.p_constraint).replace("-", "_")
    if mode not in ("free", "along_d", "zero"):
        raise ValueError("p_constraint 必须是 free / along_d / zero")
    p_along = None
    if mode == "zero":
        free[2] = free[3] = False      # Px = Py ≡ 0（初值即 0）
    elif mode == "along_d":
        # ★ P = |P|·R(−θ*)·d̂（θ* = p_zero_angle_deg；θ*=0 时退化为 Py=0）
        p_along = p_direction(base.dx, base.dy, cfg.p_zero_angle_deg)
        print(f"[torch] P 方向约束: θ* = {cfg.p_zero_angle_deg:+.2f}° ⇒ "
              f"P ∝ ({p_along[0]:+.5f}, {p_along[1]:+.5f})")
    space = ParamSpace(free, p_along_d=p_along)      # ★ 只吃 free_mask，不存在 bounds

    # ★ 初值: 本仓库的 CAD/占位初值 defaultModelParams()（**不**照搬原仓库单 yaw 的魔数）；
    #   也不做任何 clip —— 正参数取对数时若初值非正，只用 1e-6 作对数**起点**（不是限位）。
    phi0 = (default_param_vector() if cfg.init_vector is None
            else np.asarray(cfg.init_vector, dtype=np.float64).copy())

    # 只对参与拟合的轴计误差
    axis_sel = {"big": [0], "small": [1], "both": [0, 1]}[cfg.fit_axis]

    # ★ 新配方 = 每段一个 sample（不分窗、不 mini-batch）；显式给了分窗设置时提示被忽略
    pack_cfg = cfg
    if not cfg.use_legacy_path and (cfg.window_len > 0 or cfg.windows_per_seg > 1
                                    or cfg.batch_size > 0):
        if cfg.verbose:
            print("[torch] ★ 新配方按「整段 = 一个 sample」采样，忽略 "
                  f"window_len={cfg.window_len}, windows_per_seg={cfg.windows_per_seg}, "
                  f"batch_size={cfg.batch_size}（要旧配方请用 --iters>0 或 --legacy-recipe）")
        pack_cfg = replace(cfg, window_len=0, windows_per_seg=1, batch_size=0)
    batch = _pack_windows(segs, pack_cfg, axis_sel, dtype, dev)
    tau_t, th_t, dth_t = batch["tau"], batch["theta"], batch["dtheta"]
    mask_t, q0_all, qd0_all, w_axis_all = (batch["mask"], batch["q0"], batch["qd0"],
                                           batch["w_axis"])
    dt, L, W = batch["dt"], batch["L"], batch["W"]
    lens = batch["lens"]                 # 每个 sample 的**有效**长度（未 padding）

    raw = torch.tensor(space.to_raw_init(phi0), dtype=dtype, device=dev, requires_grad=True)
    v0_free = (torch.zeros(W, 2, dtype=dtype, device=dev, requires_grad=True)
               if cfg.free_init_vel else None)

    huber = torch.nn.functional.huber_loss
    integrator = cfg.integrator
    loss_mode = cfg.loss_mode

    def build_model_params():
        phi = space.to_physical(raw, phi0)
        p = PlanarParams(
            dx=base.dx, dy=base.dy, gravity=base.gravity, m_u_known=base.m_u_known,
            friction_lambda=base.friction_lambda,
            tau_offset_big=base.tau_offset_big, tau_offset_small=base.tau_offset_small,
            Jbig_eff=phi[0], Js=phi[1], Px=phi[2], Py=phi[3],
            fc_big=phi[4], fv_big=phi[5], fc_small=phi[6], fv_small=phi[7])
        return p, phi

    def _weighted(term, ax_w, mask):
        """按"轴权重（+可选有效点 mask）"归约成标量。

        `ax_w [B,2]` 的每一行只对**参与拟合的轴**给非零权重且和为 1
        （fit_axis=big ⇒ (1,0)；both ⇒ (0.5,0.5)）⇒ 归约后 = 所选轴上的**平均**。
        mask=None（新配方的 10 步片段，无 padding）⇒ 按点数平均；
        否则按有效点数平均（旧配方的分窗要屏蔽 padding）。
        """
        if mask is None:
            return (term * ax_w.unsqueeze(0)).sum() / term.shape[0]
        w = mask.unsqueeze(-1) * ax_w.unsqueeze(0)
        return (term * w).sum() / mask.sum()

    def _pair_loss(th_pred, dth_pred, th_true, dth_true, mask=None, ax_w=None):
        """★ 原仓库同款损失: 角度误差 MSE（**先 wrap 到 (−π,π]**）+ 角速度误差 MSE，等权相加。

        loss_mode="huber" 时退回旧配方（Huber + 可选 vel_weight），默认不用。
        """
        err = th_pred - th_true
        err = torch.atan2(torch.sin(err), torch.cos(err))       # ★ 先 wrap 再平方
        verr = dth_pred - dth_true
        if loss_mode == "mse":
            pos = _weighted(err ** 2, ax_w, mask)
            vel = _weighted(verr ** 2, ax_w, mask)
            return pos + vel                                    # ★ 两项**等权**相加
        pos_l = huber(err, torch.zeros_like(err), delta=cfg.huber_delta, reduction="none")
        loss = _weighted(pos_l, ax_w, mask)
        if cfg.vel_weight > 0.0:
            vel_l = huber(verr, torch.zeros_like(verr), delta=cfg.vel_huber_delta,
                          reduction="none")
            loss = loss + cfg.vel_weight * _weighted(vel_l, ax_w, mask)
        return loss

    def _slice_loss(w: int, s: int, n: int):
        """★ 新配方的一个优化步: 第 w 个 sample 的 [s, s+n) 片段做一次可导前向仿真。

        起点状态用**实测** θ/ω（与原仓库一致），片段内用记录的力矩作输入。
        """
        p, phi = build_model_params()
        sl = slice(int(s), int(s) + int(n))
        tau_b = tau_t[sl, w:w + 1]                     # [n,1,2]
        th_b = th_t[sl, w:w + 1]
        dth_b = dth_t[sl, w:w + 1]
        q0_b = th_t[int(s), w:w + 1]
        qd0_b = dth_t[int(s), w:w + 1]
        if v0_free is not None:
            qd0_b = qd0_b + v0_free[w:w + 1]
        grav_b = batch["grav"][sl, w:w + 1] if batch["has_gravity"] else None
        th_pred, dth_pred = torch_rollout(p, q0_b, qd0_b, tau_b, dt, exo, cfg.substeps,
                                          grav_seq=grav_b, integrator=integrator)
        return _pair_loss(th_pred, dth_pred, th_b, dth_b, mask=None,
                          ax_w=w_axis_all[w:w + 1])

    def _window_loss(record_params: bool = False, idx=None):
        """旧配方（也用于新配方的全段 val_loss / 画图）: 整窗/整段一起前向仿真。"""
        p, phi = build_model_params()
        grav_all = batch["grav"] if batch["has_gravity"] else None
        if idx is None:
            tau_b, th_b, dth_b, m_b = tau_t, th_t, dth_t, mask_t
            q0_b, qd0_b, wa_b = q0_all, qd0_all, w_axis_all
            grav_b = grav_all
            v0_b = v0_free
        else:
            tau_b, th_b, dth_b, m_b = (tau_t[:, idx], th_t[:, idx], dth_t[:, idx],
                                       mask_t[:, idx])
            q0_b, qd0_b, wa_b = q0_all[idx], qd0_all[idx], w_axis_all[idx]
            grav_b = None if grav_all is None else grav_all[:, idx]
            v0_b = None if v0_free is None else v0_free[idx]
        qd_start = qd0_b if v0_b is None else (qd0_b + v0_b)
        th_pred, dth_pred = torch_rollout(p, q0_b, qd_start, tau_b, dt, exo, cfg.substeps,
                                          grav_seq=grav_b, integrator=integrator)
        loss = _pair_loss(th_pred, dth_pred, th_b, dth_b, mask=m_b, ax_w=wa_b)
        return (loss, phi) if record_params else loss

    params = [raw] + ([v0_free] if v0_free is not None else [])
    loss_hist, param_hist = [], []
    n_steps = 0
    n_free = space.nfree
    summary = _config_summary(cfg, space, W, dt, n_free)

    if cfg.verbose:
        _grav = "带静态倾斜重力" if batch["has_gravity"] else "水平(重力=0)"
        print(f"[torch] {summary['recipe']}")
        print(f"[torch] 数据: {len(segs)} 段 → {W} 个 sample（最长 {L} 点 = {L*dt:.2f} s，"
              f"{_grav}）；拟合轴={cfg.fit_axis}，自由参数={n_free}；参数化: "
              f"{summary['param_space']}；参数限位: {summary['limits']}")
        print(f"[torch] 初值 φ0 = " + _fmt_vec(phi0) + f"（来源: {summary['init_source']}）")
        if loss_mode != "mse" or cfg.vel_weight not in (0.0, 1.0):
            print(f"[torch] 提示: loss_mode={loss_mode}；mse 模式下角速度项**固定等权**(1.0)，"
                  f"vel_weight={cfg.vel_weight} 仅在 huber 模式生效")
        if not cfg.use_legacy_path and cfg.lbfgs_iters > 0:
            print(f"[torch] 提示: 新配方不含 LBFGS（原仓库没有），lbfgs_iters="
                  f"{cfg.lbfgs_iters} 被忽略；需要 LBFGS 请用 --iters>0 或 --legacy-recipe")

    if not cfg.use_legacy_path:
        # ════════════════════════════════════════════════════════════════════
        # ★ 新配方（原仓库同款）: epochs × 段数 个「单片段」Adam 步
        # ════════════════════════════════════════════════════════════════════
        epochs = int(cfg.epochs)
        seg_steps = int(cfg.seg_steps)
        if epochs > 0:
            if seg_steps <= 0:
                raise ValueError("seg_steps 必须 > 0（原仓库配方 = 10 步 = 0.1 s）")
            # 与原仓库 `if L < SEG_STEPS: continue` 一致：太短的段直接跳过
            valid = [w for w in range(W) if lens[w] >= seg_steps]
            if not valid:
                raise ValueError(f"没有长度 ≥ seg_steps={seg_steps} 的数据段")
            if len(valid) < W and cfg.verbose:
                print(f"[torch] {W - len(valid)} 段短于 seg_steps={seg_steps}，已跳过")
            opt = torch.optim.Adam(params, lr=cfg.lr)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
                     if cfg.lr_schedule == "cosine" else None)     # ★ 默认无 scheduler
            rng = np.random.RandomState(cfg.seed)
            print_every = max(1, int(cfg.print_every))
            for ep in range(epochs):
                ep_loss, n_used = 0.0, 0
                for w in valid:                      # ★ 每段各抽 1 个片段做 1 个 Adam 步
                    s = int(rng.randint(0, lens[w] - seg_steps + 1))
                    opt.zero_grad(set_to_none=True)
                    loss = _slice_loss(w, s, seg_steps)
                    loss.backward()
                    opt.step()
                    if sched is not None:
                        sched.step()
                    ep_loss += float(loss.detach())
                    n_used += 1
                    n_steps += 1
                if n_used == 0:
                    continue
                ep_loss /= n_used
                loss_hist.append(ep_loss)            # ★ 每 epoch 的平均 loss（原仓库同款）
                with torch.no_grad():
                    _, phi_now = build_model_params()
                param_hist.append([float(v) for v in phi_now])
                # 前 5 / 后 5 个 epoch 一定打印（其余按 print_every）
                if cfg.verbose and (ep < 5 or ep >= epochs - 5 or ep == epochs - 1
                                    or ep % print_every == 0):
                    print(f"[torch] epoch {ep:5d}/{epochs}  loss={ep_loss:.6e}  "
                          + _fmt_vec(phi_now))
        elif cfg.verbose:
            print("[torch] epochs<=0 ⇒ 不优化，直接输出初值（只做数据/画图冒烟）")

        with torch.no_grad():
            final_loss, phi_final = _window_loss(record_params=True)
        recipe = ("原仓库配方(epochs×段数 个单片段 Adam 步)" if epochs > 0
                  else "未优化(epochs<=0)")
    else:
        # ════════════════════════════════════════════════════════════════════
        # 旧精细配方（iters>0 才走这里）: 分窗 mini-batch Adam（+ 可选 LBFGS）
        # ════════════════════════════════════════════════════════════════════
        opt = torch.optim.Adam(params, lr=cfg.lr)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.iters))
                 if cfg.lr_schedule == "cosine" else None)
        n_batch = W if cfg.batch_size <= 0 else min(W, int(cfg.batch_size))
        if cfg.verbose:
            print(f"[torch] 旧配方: 窗口数 W={W}（每窗 {L} 点 = {L*dt:.2f} s）, "
                  f"Adam 批={n_batch}/{W} 窗")
        for it in range(cfg.iters):
            opt.zero_grad(set_to_none=True)
            if n_batch < W:
                idx = torch.from_numpy(np.random.choice(W, size=n_batch, replace=False)).to(dev)
            else:
                idx = None
            loss = _window_loss(idx=idx)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            loss_hist.append(float(loss.detach()))
            n_steps += 1
            with torch.no_grad():
                _, phi_now = _window_loss(record_params=True)
            param_hist.append([float(v) for v in phi_now])
            if cfg.verbose and (it % max(1, cfg.print_every) == 0 or it == cfg.iters - 1):
                print(f"[torch] adam {it:5d}  loss={loss_hist[-1]:.6e}  " + _fmt_vec(phi_now))

        # ── LBFGS 精调（全批：所有窗口；默认关闭）──
        if cfg.lbfgs_iters > 0:
            opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_max_iter,
                                     history_size=10, line_search_fn="strong_wolfe")

            def closure():
                opt2.zero_grad(set_to_none=True)
                l = _window_loss()
                l.backward()
                return l

            for it in range(cfg.lbfgs_iters):
                l = opt2.step(closure)
                loss_hist.append(float(l.detach()))
                n_steps += 1
                with torch.no_grad():
                    _, phi_now = _window_loss(record_params=True)
                param_hist.append([float(v) for v in phi_now])
                if cfg.verbose:
                    print(f"[torch] lbfgs{it:5d}  loss={float(l.detach()):.6e}  "
                          + _fmt_vec(phi_now))

        with torch.no_grad():
            final_loss, phi_final = _window_loss(record_params=True)
        recipe = ("旧精细配方(%d 步 Adam%s%s)"
                  % (cfg.iters,
                     "" if cfg.lbfgs_iters <= 0 else f" + {cfg.lbfgs_iters} 步 LBFGS",
                     "" if sched is None else " + cosine"))

    phi_np = np.array([float(v) for v in phi_final])
    truth = None if cfg.truth_vector is None else np.asarray(cfg.truth_vector, dtype=np.float64)
    return FitResult(phi=phi_np, phi0=phi0, loss_history=loss_hist,
                     val_loss=float(final_loss), n_iter=n_steps,
                     seconds=time.time() - t_start,
                     param_history=param_hist, epoch_losses=list(loss_hist),
                     n_free=n_free, n_steps=n_steps, recipe=recipe, truth=truth,
                     config=summary)


def _fmt_vec(phi) -> str:
    vals = [float(v) for v in phi]
    return ("[" + " ".join(f"{v:+.5f}" for v in vals) + "]")


def loss_trend(loss_history) -> dict:
    """收敛趋势摘要（不要求逐点单调，只看整体是否下降）。"""
    h = [float(v) for v in (loss_history or [])]
    if not h:
        return {"n": 0}
    k = max(1, min(10, len(h) // 5 or 1))
    head = float(np.mean(h[:k]))
    tail = float(np.mean(h[-k:]))
    return {"n": len(h), "head": head, "tail": tail,
            "ratio": (tail / head) if head > 0 else float("nan"),
            "decreased": bool(tail < head),
            "min": float(np.min(h)), "final": float(h[-1])}


# ============================================================================
# 自检（模型一致性）
# ============================================================================
def model_self_test(verbose: bool = True) -> bool:
    """检查: (1) Y·φ == ID(φ)；(2) Y == ∂ID/∂φ（数值偏导）；(3) torch 与 numpy 一致。"""
    rng = np.random.default_rng(0)
    ok = True
    p = PlanarParams(dx=0.037, dy=-0.011, gravity=9.81, m_u_known=0.25,
                     friction_lambda=10.0).with_vector(
        np.array([0.0243, 0.0132, 0.0042, -0.0018, 0.092, 0.031, 0.028, 0.0085]))
    exo = Exo(gravity_a=(0.31, -0.17), base_omega=0.42, base_alpha=-0.9)
    q = rng.normal(size=(5, 2)) * 0.6
    qd = rng.normal(size=(5, 2)) * 2.0
    qdd = rng.normal(size=(5, 2)) * 5.0

    Y = regressor_np(q, qd, qdd, p, exo)
    tau_id = inverse_dynamics_np(q, qd, qdd, p, exo)
    # ID 对 φ 是**仿射**的: ID = Y·φ + ID(φ=0)。
    # ID(φ=0) 里只剩不参与辨识的固定量（这里只有 m_u_known 的重力项；本任务 m_u_known=0 且 g=0）
    tau_affine = inverse_dynamics_np(q, qd, qdd, p.with_vector(np.zeros(NPARAM)), exo)
    tau_lin = np.einsum("...ij,j->...i", Y, p.vector()) + tau_affine
    e1 = np.max(np.abs(tau_lin - tau_id))
    # 数值偏导（中心差分）
    Ynum = np.zeros_like(Y)
    for j in range(NPARAM):
        for sgn in (+1, -1):
            v = p.vector()
            h = 1e-6 * max(1.0, abs(v[j]))
            v[j] += sgn * h
            Ynum[..., :, j] += sgn * inverse_dynamics_np(q, qd, qdd, p.with_vector(v), exo) / (2 * h)
    e2 = np.max(np.abs(Ynum - Y))
    # torch vs numpy
    if torch is not None:
        pt = p
        qt = torch.tensor(q, dtype=torch.float64)
        qdt = torch.tensor(qd, dtype=torch.float64)
        qddt = torch.tensor(qdd, dtype=torch.float64)
        taut = torch.tensor(tau_id, dtype=torch.float64)
        e3 = float(torch.max(torch.abs(torch_forward_accel(qt, qdt, taut, pt, exo)
                                      - torch.tensor(forward_accel_np(q, qd, tau_id, p, exo),
                                                     dtype=torch.float64))))
        q0 = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        qd0 = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
        tau_seq = torch.tensor(rng.normal(size=(7, 1, 2)) * 0.3, dtype=torch.float64)
        th_t, dth_t = torch_rollout(pt, q0, qd0, tau_seq, 0.01, exo, 4)
        th_n, dth_n = simulate_np(pt, q0[0].numpy(), qd0[0].numpy(),
                                  tau_seq[:, 0, :].numpy(), 0.01, exo, 4)
        e4 = float(np.max(np.abs(th_t[:, 0, :].numpy() - th_n)))
    else:
        e3 = e4 = float("nan")
    # (5) 重力通道确实生效: 倾斜重力 vs 零重力的 τ 必须不同，且量级符合 P·g
    tau_tilt = inverse_dynamics_np(q, qd, qdd, p, exo)
    tau_level = inverse_dynamics_np(q, qd, qdd, p, Exo())
    e5 = float(np.max(np.abs(tau_tilt - tau_level)))
    # (6) 逐样本重力（数组）与逐步循环等价；torch 的 grav_seq 与 numpy 的 exo_seq 一致
    gxs = rng.normal(size=q.shape[0]) * 0.4
    gys = rng.normal(size=q.shape[0]) * 0.4
    exo_arr = exo_from_gravity(gxs, gys, exo.base_omega, exo.base_alpha)
    e6 = float(np.max(np.abs(inverse_dynamics_np(q, qd, qdd, p, exo_arr)
                             - np.stack([inverse_dynamics_np(q[i:i + 1], qd[i:i + 1],
                                                            qdd[i:i + 1], p,
                                                            exo_from_gravity(gxs[i], gys[i],
                                                                             exo.base_omega,
                                                                             exo.base_alpha))[0]
                                        for i in range(q.shape[0])], axis=0))))
    # (7) 逐样本重力 + 前向仿真: numpy(exo_seq) 与 torch(grav_seq) 必须完全一致
    e7 = float("nan")
    if torch is not None:
        tau_seq = rng.normal(size=(q.shape[0], 2)) * 0.2
        seq = [exo_from_gravity(float(gxs[i]), float(gys[i])) for i in range(q.shape[0])]
        th_np, _ = simulate_np(p, q[0], qd[0], tau_seq, 0.01, EXO_ZERO, 2, exo_seq=seq)
        gseq = torch.tensor(np.stack([gxs, gys], axis=-1), dtype=torch.float64)[:, None, :]
        th_t, _ = torch_rollout(p, torch.tensor(q[0:1], dtype=torch.float64),
                                torch.tensor(qd[0:1], dtype=torch.float64),
                                torch.tensor(tau_seq[:, None, :], dtype=torch.float64),
                                0.01, EXO_ZERO, 2, grav_seq=gseq)
        e7 = float(np.max(np.abs(th_t[:, 0, :].numpy() - th_np)))
    # (8) P 约束映射: along_d ⇒ Px/Py 严格沿 d（含 dy≠0）；(9) log/自由参数化往返一致
    #     ★ 新 ParamSpace 没有任何 bounds，所以这里只查「映射是否精确、是否可逆」。
    e8 = float("nan")
    e9 = float("nan")
    e10 = float("nan")
    for (ddx, ddy) in ((0.03, 0.0), (0.021, -0.017)):
        sp = ParamSpace(np.ones(NPARAM, dtype=bool), p_along_d=(ddx, ddy))
        r0 = sp.to_raw_init(p.vector())
        ph = sp.to_physical(r0, p.vector())
        n = math.hypot(ddx, ddy)
        e8 = max(e8, abs(ph[2] * (ddy / n) - ph[3] * (ddx / n))) if np.isfinite(e8) else 0.0
        # 反向: 直接给 s 赋值（新映射里 s 就是自由 raw 分量，不再有 sigmoid）
        raw = r0.copy()
        raw[-1] = 1.5
        ph2 = sp.to_physical(raw, p.vector())
        s = 1.5
        e9 = min(e9 if np.isfinite(e9) else 1e9,
                 max(abs(ph2[2] - s * ddx / n), abs(ph2[3] - s * ddy / n)))
    # (10) 无限位 + log 参数化: raw = log(φ0) 应精确还原 φ0（正参数），Px/Py 原样通过
    sp_free = ParamSpace()
    e10 = float(np.max(np.abs(sp_free.to_physical(sp_free.to_raw_init(p.vector()),
                                                 p.vector()) - p.vector())))
    # (11) log 参数化确实**没有界**: raw 取 ±50 ⇒ φ = exp(±50) 极端但有限，且不出现负值
    pos_ext = sp_free.to_physical(np.array([50.0, -50.0, 0.0, 0.0, 50.0, -50.0, 50.0, -50.0]),
                                  p.vector())
    e11 = 0.0 if (np.all(pos_ext[[0, 1, 4, 5, 6, 7]] > 0.0)
                  and np.all(np.isfinite(pos_ext))) else 1.0
    # (12) 新配方积分器 euler: torch 与 numpy 必须逐位一致（同序半隐式欧拉）
    e12 = float("nan")
    if torch is not None:
        taus = rng.normal(size=(6, 2)) * 0.2
        th_e, dth_e = simulate_np(p, q[0], qd[0], taus, 0.01, exo, 1, integrator="euler")
        gseq2 = torch.zeros(6, 1, 2, dtype=torch.float64)
        th_te, dth_te = torch_rollout(p, torch.tensor(q[0:1], dtype=torch.float64),
                                      torch.tensor(qd[0:1], dtype=torch.float64),
                                      torch.tensor(taus[:, None, :], dtype=torch.float64),
                                      0.01, exo, 1, grav_seq=None, integrator="euler")
        e12 = float(np.max(np.abs(th_te[:, 0, :].numpy() - th_e)))
        del gseq2
    if verbose:
        print("[selftest] max|Y·φ − ID|          =", f"{e1:.3e}")
        print("[selftest] max|Y_analytic − Y_num|=", f"{e2:.3e}")
        print("[selftest] max|torch − numpy| accel=", f"{e3:.3e}")
        print("[selftest] max|torch − numpy| rollout θ =", f"{e4:.3e}")
        print("[selftest] 重力项通道 |τ_tilt − τ_level| max =", f"{e5:.3e}", "(应 ≈ P·g 量级)")
        print("[selftest] 逐样本重力一致性 max err =", f"{e6:.3e}")
        print("[selftest] 倾斜前向仿真 torch vs numpy =", f"{e7:.3e}")
        print("[selftest] P 沿 d 约束 (Px·dy = Py·dx) max err =", f"{e8:.3e}")
        print("[selftest] P 沿 d 约束 (|P| 缩放一致性) max err =", f"{e9:.3e}")
        print("[selftest] log/自由参数化往返 max err =", f"{e10:.3e}", "(无限位)")
        print("[selftest] 无限位检查 raw=±50 ⇒ 正参数仍 >0 且有限 :", "OK" if e11 == 0.0 else "FAIL")
        print("[selftest] euler torch vs numpy 一致 max err =", f"{e12:.3e}")
    ok = (e1 < 1e-9 and e2 < 1e-5 and e5 > 1e-4 and e6 < 1e-12 and e8 < 1e-12
          and e9 < 1e-12 and e10 < 1e-12 and e11 == 0.0
          and (torch is None or (e3 < 1e-10 and e4 < 1e-9 and e7 < 1e-9 and e12 < 1e-12)))
    if verbose:
        print("[selftest]", "PASS" if ok else "FAIL")
    return ok


# ============================================================================
# 收敛曲线 / 轨迹对比（matplotlib，默认写文件；--show-plot 才交互显示）
# ============================================================================
def resolve_plot_paths(plot_out: str | None, default_prefix: str = "data/sysid/ident_torch"):
    """--plot-out 的解析规则（返回 (收敛曲线, 轨迹对比) 两个 png 路径）。

      · 未给 ⇒ 默认 `<default_prefix>_convergence.png` / `<default_prefix>_traj.png`
      · 以 .png 结尾 ⇒ 该路径作为收敛曲线，轨迹图为 `<stem>_traj.png`
      · 目录（以 / 结尾或已存在的目录）⇒ `<dir>/ident_torch_convergence.png` 等
      · 其它 ⇒ 当作**前缀** ⇒ `<前缀>_convergence.png` / `<前缀>_traj.png`
    """
    if not plot_out:
        return default_prefix + "_convergence.png", default_prefix + "_traj.png"
    p = str(plot_out)
    if p.lower().endswith(".png"):
        return p, p[:-4] + "_traj.png"
    if p.endswith(os.sep) or os.path.isdir(p):
        base = os.path.join(p, "ident_torch")
        return base + "_convergence.png", base + "_traj.png"
    return p + "_convergence.png", p + "_traj.png"


_CJK_FONTS = ("Noto Sans CJK JP", "Noto Sans CJK SC", "Source Han Sans JP", "Microsoft YaHei",
              "WenQuanYi Zen Hei", "SimHei", "Droid Sans Fallback", "AR PL UMing CN",
              "PingFang SC", "Heiti SC", "Arial Unicode MS")


def _setup_font():
    """挑一个可用的中文字体（找不到就退回默认，并提示图上中文可能显示为方框）。"""
    import matplotlib
    from matplotlib import font_manager
    try:
        avail = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:                                       # pragma: no cover
        return None
    for name in _CJK_FONTS:
        if name in avail:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    print("[plot][warn] 未找到中文字体（Noto Sans CJK / SimHei / YaHei 等），"
          "图中中文可能显示为方框")
    return None


def _lazy_pyplot(show_plot: bool):
    """按需导入 matplotlib.pyplot；★ 非显示环境统一 Agg（不在这里偷偷改全局后端）。"""
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()
    return plt


def plot_convergence(res: FitResult, out_path: str, show_plot: bool = False,
                     title_note: str = "") -> str:
    """★ 图 1: loss（log 纵轴）+ 8 个参数各自的收敛曲线（3×3 共 9 格）。

    每个参数子图上用**虚线**标出参考值：有真值（`--truth-params`）画真值，否则画初值 φ0。
    """
    plt = _lazy_pyplot(show_plot)
    hist = np.asarray(res.param_history if res.param_history else [res.phi], dtype=np.float64)
    if hist.ndim != 2:
        hist = hist.reshape(1, -1)
    losses = np.asarray(res.loss_history, dtype=np.float64)
    steps = np.arange(hist.shape[0])
    xlabel = "epoch" if not str(res.recipe).startswith("旧") else "Adam/LBFGS 步"

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(f"参数辨识收敛曲线 —— {res.recipe}\n{title_note}".strip(), fontsize=11)

    ax = axes[0, 0]
    ax.plot(np.arange(losses.size), losses, lw=1.2, color="C3")
    if losses.size:
        ax.plot(np.arange(losses.size), losses, ".", ms=2, color="C3")
    ax.set_yscale("log")                     # ★ loss 用 log 纵轴（原仓库同款）
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    ax.set_title(f"Loss（log 纵轴, 末值={losses[-1]:.3e}）" if losses.size else "Loss")
    ax.grid(True, which="both", alpha=0.3)

    ref = res.truth if res.truth is not None else res.phi0
    ref_label = "真值" if res.truth is not None else "初值"
    for j, nm in enumerate(PARAM_NAMES):
        a = axes[(j + 1) // 3, (j + 1) % 3]
        a.plot(steps, hist[:, j], lw=1.2, color=f"C{j}")
        a.axhline(float(ref[j]), ls="--", lw=1.0, color="k", alpha=0.7,
                  label=f"{ref_label}={float(ref[j]):.4g}")
        a.set_xlabel(xlabel)
        a.set_title(f"{nm}  [{PARAM_UNITS[j]}]")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out_path, show_plot)
    return out_path


def angle_rmse_deg(a, b):
    """wrap 到 (−π, π] 后的角度 RMSE（**度**）。"""
    d = np.arctan2(np.sin(np.asarray(a) - np.asarray(b)), np.cos(np.asarray(a) - np.asarray(b)))
    return float(np.degrees(np.sqrt(np.mean(d ** 2))))


def vel_rmse(a, b):
    """角速度 RMSE（**rad/s**）。"""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def plot_trajectory(res: FitResult, segs, base: PlanarParams, exo: Exo = EXO_ZERO,
                    out_path: str = "data/sysid/ident_torch_traj.png", show_plot: bool = False,
                    integrator: str = "euler") -> str:
    """★ 图 2: 实测 vs 仿真（至少两段: 一段大 yaw 激励、一段小 yaw 激励）。

    行 = 数据段（大 yaw 段 / 小 yaw 段），列 = 角度 θ / 角速度 θ̇；
    每个子图画**两轴**的实测(实线)与仿真(虚线)，标题里标注两轴的 RMSE。
    """
    plt = _lazy_pyplot(show_plot)
    phi = np.asarray(res.phi, dtype=np.float64)
    p_model = base.with_vector(phi)

    # 选段：优先「大 yaw 被激励」+「小 yaw 被激励」各一段，缺口时用现有段补齐
    def _pick(axis):
        for s in segs:
            if int(s.axis) == axis and s.T > 10:
                return s
        return None

    picks = [_pick(AXIS_BIG), _pick(AXIS_SMALL)]
    used = [s for s in picks if s is not None]
    if len(used) < 2:                       # 只有单轴数据 ⇒ 用前两段（或同一段两次）
        cand = [s for s in segs if s.T > 10]
        used = (cand[:2] if len(cand) >= 2 else (cand * 2)[:2])
    if not used:
        raise ValueError("没有可画的数据段")

    nrow = len(used)
    fig, axes = plt.subplots(nrow, 2, figsize=(15, 3.6 * nrow), squeeze=False)
    fig.suptitle("实测 vs 仿真（参数辨识后，同一段力矩输入的前向仿真）", fontsize=11)

    for i, seg in enumerate(used):
        # 用记录力矩从**该段实测初值**前向仿真（与辨识/评测一致的积分器）
        q0 = np.asarray(seg.theta[0], dtype=np.float64)
        qd0 = np.asarray(seg.dtheta[0], dtype=np.float64)
        th_sim, dth_sim = simulate_np(p_model, q0, qd0, seg.tau, seg.dt, exo,
                                      substeps=1, integrator=integrator)
        t = np.arange(seg.T) * seg.dt
        axis_name = "大 yaw 激励段" if int(seg.axis) == AXIS_BIG else "小 yaw 激励段"
        for col, (meas, sim, lab, unit) in enumerate((
                (seg.theta, th_sim, "θ", "rad"),
                (seg.dtheta, dth_sim, "θ̇", "rad/s"))):
            a = axes[i][col]
            ann = []
            for k, (ax_name, ax_c) in enumerate((("big", "C0"), ("small", "C1"))):
                a.plot(t, meas[:, k], color=ax_c, lw=1.1, alpha=0.85,
                       label=f"实测 {ax_name}")
                a.plot(t, sim[:, k], color=ax_c, lw=1.1, ls="--", alpha=0.9,
                       label=f"仿真 {ax_name}")
                # ★ 标注与**该子图的量**对应的 RMSE：角度用度，角速度用 rad/s
                if col == 0:
                    ann.append(f"{ax_name}: RMSE={angle_rmse_deg(sim[:, k], meas[:, k]):.3f}°")
                else:
                    ann.append(f"{ax_name}: RMSE={vel_rmse(sim[:, k], meas[:, k]):.4f} rad/s")
            a.set_title(f"{axis_name}（{os.path.basename(seg.source)[:28]}）  " + " | ".join(ann),
                        fontsize=9)
            a.set_xlabel("t [s]")
            a.set_ylabel(f"{lab} [{unit}]")
            a.grid(True, alpha=0.3)
            a.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, out_path, show_plot)
    return out_path


def _save_fig(fig, path: str, show_plot: bool):
    """保存 PNG（父目录自动创建）；只有 --show-plot 才 plt.show()。"""
    import matplotlib.pyplot as plt
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"[plot] 已保存 {path}  ({os.path.getsize(path) / 1024.0:.1f} KB)")


# ============================================================================
# CLI
# ============================================================================
def _build_argparser():
    ap = argparse.ArgumentParser(
        description="平面 8 参模型 —— PyTorch 可导前向仿真参数辨识（输出误差法）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", action="append", default=None,
                    help="数据 glob（可重复/逗号分隔）；默认 data/sysid/*.csv 和 *.npz")
    ap.add_argument("--fit-axis", choices=["both", "big", "small"], default="both",
                    help="both=两轴一起拟合；big/small=只拟合该轴（另一轴摩擦参数冻结）")
    g = ap.add_argument_group(
        "★ 默认配方（= 原仓库 TorqueController/python/scripts/param_ident.py 同款）",
        "epochs=200（= 原仓库 num_epochs=1000 的 1/5）；每 epoch 对**每个数据段**随机截取 "
        "seg_steps=10 步（0.1 s）片段做 1 个 Adam 步 ⇒ 总步数 = epochs × 段数；"
        "损失 = 角度误差 MSE（先 wrap 到 (−π,π]）+ 角速度误差 MSE，**两项等权**；"
        "Adam lr=3e-4 **常数**（无 scheduler）、**无 LBFGS**；积分 = 半隐式欧拉，substeps=1，"
        "dt 取自数据（100 Hz ⇒ 0.01 s）；**参数无任何限位**（正参数 log 参数化 φ=exp(raw)，"
        "Px/Py 自由）；初值沿用本仓库 CAD 占位值 defaultModelParams()。")
    g.add_argument("--epochs", type=int, default=FitConfig.epochs,
                   help="★ 训练 epoch 数；**1000 = 与原仓库 num_epochs 相同**"
                        "（每 epoch 的 Adam 步数 = 数据段数；0 = 不优化只画图）")
    g.add_argument("--seg-steps", type=int, default=FitConfig.seg_steps,
                   help="★ 每个优化步随机截取的片段长度（原仓库 SEG_STEPS=10 步 = 0.1 s）")
    g.add_argument("--loss-mode", choices=["mse", "huber"], default=FitConfig.loss_mode,
                   help="★ mse = 角度wrap MSE + 角速度 MSE 等权（原仓库配方，默认）；"
                        "huber = 旧配方（Huber，可用 --vel-weight 加权）")
    g.add_argument("--integrator", choices=["euler", "rk4"], default=FitConfig.integrator,
                   help="★ euler = 半隐式（symplectic）欧拉（原仓库配方，默认）；rk4 = 旧配方")
    g.add_argument("--lr-schedule", choices=["none", "cosine"], default=FitConfig.lr_schedule,
                   help="★ none = 常数学习率（原仓库没有 scheduler，默认）；cosine = 旧配方")
    ap.add_argument("--legacy-recipe", action="store_true",
                    help="★ 一键回到旧的精细配方: iters=400 + lr=5e-3 + Huber + 分窗"
                         "(window_len=150, 2 窗/段, batch=4) + LBFGS(25) + RK4 + cosine")
    ap.add_argument("--iters", type=int, default=FitConfig.iters,
                    help="旧配方的 Adam 步数；**0（默认）= 走新配方（epochs×段数 个步）**；"
                         ">0 = 精确跑 iters 个整窗/分窗 mini-batch Adam 步（旧配方）")
    ap.add_argument("--lbfgs-iters", type=int, default=FitConfig.lbfgs_iters,
                    help="LBFGS 迭代数（0=关闭 ⇒ 原仓库配方没有 LBFGS）")
    ap.add_argument("--lr", type=float, default=FitConfig.lr,
                    help="Adam 学习率（★ 新配方默认 3e-4，与原仓库一致；旧配方用 5e-3）")
    ap.add_argument("--seed", type=int, default=FitConfig.seed)
    ap.add_argument("--substeps", type=int, default=FitConfig.substeps,
                    help="可导前向仿真每个控制步的积分子步数（原仓库配方 = 1）")
    ap.add_argument("--huber-delta", type=float, default=FitConfig.huber_delta,
                    help="角度 Huber 阈值 (rad)（仅 --loss-mode=huber 时生效）")
    ap.add_argument("--vel-weight", type=float, default=FitConfig.vel_weight,
                    help="角速度误差项权重（仅 --loss-mode=huber 有效；mse 模式固定等权 1.0）")
    ap.add_argument("--free-init-vel", action="store_true",
                    help="把每段初始角速度也作为自由参数优化")
    ap.add_argument("--p-bound", type=float, default=FitConfig.p_bound,
                    help="★ 已废弃并被忽略（本脚本不再有任何参数限位；仅为 CLI 兼容保留）")
    ap.add_argument("--init-vector", type=str, default=None,
                    help="逗号分隔的 8 个初值（默认用 CAD 占位值 defaultModelParams()）")
    ap.add_argument("--truth-params", type=str, default=None,
                    help="逗号分隔的 8 个真值/参考值（收敛曲线上的虚线；默认画初值 φ0）")
    ap.add_argument("--dx", type=float, default=0.030, help="实测几何 dx (m)")
    ap.add_argument("--dy", type=float, default=0.0, help="实测几何 dy (m)")
    ap.add_argument("--model-lambda", type=float, default=FRICTION_LAMBDA,
                    help="★ 辨识模型的摩擦软符号陡度 λ；默认 10 = 本仓库约定（与前向仿真/"
                         "MPC/planar_yaw_model.h 一致）。**原仓库单 yaw 版用 1e4**，本仓库不能"
                         "照搬（原因见文件头 docstring）；想复现 1e4 可传 --model-lambda=1e4 消融")
    ap.add_argument("--dt", type=float, default=None, help="覆盖 dt（默认取数据里的）")
    ap.add_argument("--max-points", type=int, default=0, help="每段只用前 N 点（加速调试）")
    ap.add_argument("--window-len", type=int, default=FitConfig.window_len,
                    help="每个窗口的点数（0=整段一个窗口；**仅旧配方**用，新配方忽略）")
    ap.add_argument("--windows-per-seg", type=int, default=FitConfig.windows_per_seg,
                    help="每段切几个窗口（仅旧配方用，新配方忽略）")
    ap.add_argument("--batch-size", type=int, default=FitConfig.batch_size,
                    help="Adam 每步随机抽几个窗口（0=全批；仅旧配方用，新配方忽略）")
    ap.add_argument("--fix-p", action="store_true",
                    help="把 Px/Py 固定为 0（等价 --p-constraint=zero）")
    ap.add_argument("--p-constraint", choices=["free", "along_d", "along-d", "zero"],
                    default="free",
                    help="Px/Py 的处理: free=各自独立；along_d=零点标定后 P=|P|·R(−θ*)·d̂（单参数）；"
                         "zero=固定为 0")
    ap.add_argument("--p-zero-angle", type=float, default=0.0,
                    help="平衡点 θ* 在当前零点坐标系里的读数（度）。0 = 零点恰好设在平衡点上；"
                         "手动挪过零点就填实际读数。仅 --p-constraint=along_d 时有效")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 线程数（0=默认；小张量下 1 通常最快）")
    ap.add_argument("--print-every", type=int, default=FitConfig.print_every)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None,
                    help="可选：把辨识结果写成文本（默认只打印，不建文件）")
    pg = ap.add_argument_group("收敛曲线 / 轨迹对比（matplotlib）")
    pg.add_argument("--plot-out", type=str, default=None,
                    help="输出路径/前缀（默认 data/sysid/ident_torch）：以 .png 结尾 ⇒ 收敛曲线用"
                         "该文件、轨迹用 <名>_traj.png；给目录 ⇒ 目录下 ident_torch_*.png；"
                         "否则当前缀 ⇒ <前缀>_convergence.png / <前缀>_traj.png")
    pg.add_argument("--no-plot", action="store_true", help="不画图、不写 PNG")
    pg.add_argument("--show-plot", action="store_true",
                    help="交互显示（默认用 Agg 后端只写文件，无显示环境下不报错）")
    ap.add_argument("--selftest", action="store_true", help="只跑模型自检")
    return ap


def _parse_vec8(s, what: str) -> np.ndarray:
    """解析逗号分隔的 8 个数（--init-vector / --truth-params）。"""
    v = np.array([float(x) for x in str(s).replace(";", ",").split(",") if x.strip() != ""],
                 dtype=np.float64)
    if v.size != NPARAM:
        raise SystemExit(f"[error] {what} 需要 {NPARAM} 个数，收到 {v.size} 个")
    return v


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    base = PlanarParams(dx=args.dx, dy=args.dy, friction_lambda=args.model_lambda)

    if args.selftest:
        return 0 if model_self_test() else 1

    if torch is None:
        print(f"[error] 需要 torch: {_TORCH_IMPORT_ERROR}", file=sys.stderr)
        return 2

    patterns = args.data or ["data/sysid/*.csv", "data/sysid/*.npz"]
    segs = load_segments(patterns, dt_override=args.dt)
    if not segs:
        print("[error] 没有读到数据；请用 --data=<glob> 指定（或先跑采集脚本）",
              file=sys.stderr)
        return 2

    cfg = FitConfig(fit_axis=args.fit_axis, iters=args.iters, lbfgs_iters=args.lbfgs_iters,
                    lr=args.lr, seed=args.seed, substeps=args.substeps,
                    huber_delta=args.huber_delta, vel_weight=args.vel_weight,
                    free_init_vel=args.free_init_vel, p_bound=args.p_bound,
                    epochs=args.epochs, seg_steps=args.seg_steps, loss_mode=args.loss_mode,
                    integrator=args.integrator, lr_schedule=args.lr_schedule,
                    init_vector=(None if args.init_vector is None
                                 else _parse_vec8(args.init_vector, "--init-vector")),
                    truth_vector=(None if args.truth_params is None
                                  else _parse_vec8(args.truth_params, "--truth-params")),
                    print_every=args.print_every, device=args.device,
                    max_points=args.max_points, window_len=args.window_len,
                    windows_per_seg=args.windows_per_seg, batch_size=args.batch_size,
                    p_constraint=args.p_constraint, fix_p=args.fix_p,
                    p_zero_angle_deg=args.p_zero_angle)
    if args.legacy_recipe:
        cfg.legacy_recipe()
        print("[cfg] ★ --legacy-recipe: 已切到旧精细配方（iters=400, lr=5e-3, Huber, 分窗, "
              "LBFGS=25, RK4, cosine）")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    res = fit_params_torch(segs, cfg, base)

    print("\n" + "=" * 78)
    print(f"辨识结果（输出误差法, 段数={len(segs)}, 轴={cfg.fit_axis}）  用时 {res.seconds:.1f}s")
    print(f"配方: {res.recipe}；Adam/总步数={res.n_steps}；参数限位={res.config['limits']}")
    print(f"{'#':>2} {'参数':<10} {'初值':>12} {'估计':>12} {'变化':>12}  单位")
    for j, nm in enumerate(PARAM_NAMES):
        print(f"{j:>2} {nm:<10} {res.phi0[j]:>12.6f} {res.phi[j]:>12.6f} "
              f"{res.phi[j] - res.phi0[j]:>+12.6f}  {PARAM_UNITS[j]}")
    print("=" * 78)

    # ── 收敛摘要: 前 5 / 后 5 个 loss（用户要求）──
    h = res.loss_history
    if h:
        n5 = min(5, len(h))
        print(f"loss 曲线: 共 {len(h)} 点；前 {n5} 个 = "
              + ", ".join(f"{v:.6e}" for v in h[:n5]))
        print(f"           后 {n5} 个 = " + ", ".join(f"{v:.6e}" for v in h[-n5:]))
        tr = loss_trend(h)
        print(f"            前 {n5} 均值={tr['head']:.6e} → 后 {n5} 均值={tr['tail']:.6e}"
              f"（比值 {tr['ratio']:.4f}，最小值 {tr['min']:.6e}）"
              f"  ⇒ {'下降 ✓' if tr['decreased'] else '未下降 ✗'}")
    print("可粘贴到 include/tcbs/mpc/planar_yaw_params.h:")
    print(f"  p.Jbig_eff = {res.phi[0]:.6f};  p.Js = {res.phi[1]:.6f};")
    print(f"  p.Px = {res.phi[2]:.6f};  p.Py = {res.phi[3]:.6f};")
    print(f"  p.fcBig = {res.phi[4]:.6f};  p.fvBig = {res.phi[5]:.6f};")
    print(f"  p.fcSmall = {res.phi[6]:.6f};  p.fvSmall = {res.phi[7]:.6f};")

    # ── 收敛曲线 / 轨迹对比（默认写 PNG；无显示环境也不报错）──
    if not args.no_plot:
        conv_png, traj_png = resolve_plot_paths(args.plot_out)
        note = (f"段数={len(segs)}, 轴={cfg.fit_axis}, 步数={res.n_steps}, "
                f"lr={cfg.lr:g}, 损失={cfg.loss_mode}, 积分={cfg.integrator}")
        try:
            plot_convergence(res, conv_png, show_plot=args.show_plot, title_note=note)
            plot_trajectory(res, segs, base, integrator=cfg.integrator,
                            out_path=traj_png, show_plot=args.show_plot)
        except Exception as exc:                    # 画图失败不应让辨识结果丢失
            print(f"[plot][warn] 画图失败（辨识结果仍然有效）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("# identify_params_torch.py 输出（输出误差法）\n")
            fh.write(f"# 配方: {res.recipe}\n")
            for nm, v in zip(PARAM_NAMES, res.phi):
                fh.write(f"{nm} = {v:.9f}\n")
        print(f"[out] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
