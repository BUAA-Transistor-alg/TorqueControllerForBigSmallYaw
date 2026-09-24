#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""② 可微仿真模型 —— **不含任何可学习参数** 的 RK4 可导前向仿真 + numpy 参考实现。

模型（与 `include/tcbs/mpc/planar_yaw_model.h` 的 `eomBacklash` 逐项对应，务必保持一致）:

    q = (θ_motor, θ_platform, θ_small)      # q[0] 电机、q[1] 大 yaw 云台、q[2] 小 yaw
    u = (τ_cmd, 0, τ_small)                 # 只有电机被大 yaw 力矩驱动

    Δ  = θ_motor − θ_platform − β
    τ_t = k·[ dz(Δ) + γ·Δ ] + c·Δ̇            # 传动扭矩（电机受到 −τ_t）
    dz(Δ) = relu_ε(Δ − δ/2) − relu_ε(−Δ − δ/2)      # 平滑死区，relu_ε(x)=½(x+√(x²+ε²))
    M    = blkdiag(J_motor, M2)             # M2 = 云台/小 yaw 子块（与 2-DOF 完全同一组式）
    h[0] = τ_t + fc_motor·tanh(λ·θ̇_motor) + fv_motor·θ̇_motor + τ_off_motor
    h[1] = h_b − τ_t      h[2] = h_s
    q̈    = M⁻¹(u − h)

云台/小 yaw 子块（= 2-DOF 那套"平面耦合"模型）:

    Q(θs) = R(θs)·P                     # P = (Px, Py) 上装一阶矩 m_u·ρ
    M11 = Jbig_eff + Js + 2·(d·Q)       M12 = Js + (d·Q)      M22 = Js
    μ   = 2·(dy·Qx − dx·Qy)             # = ∂M11/∂θs
    h_b = μ·θ̇p·θ̇s + ½μ·θ̇s² − G_b + μ·θ̇s·ω_c + M11·α_c + fric_b
    h_s = −½μ·θ̇p²      − G_s − μ·θ̇p·ω_c − ½μ·ω_c² + M12·α_c + fric_s
    fric_k = fc_k·tanh(λ·θ̇_k) + fv_k·θ̇_k

★ 可学习参数**不是**这个类的成员: 每次 ``forward`` 从第 1 个参数按 batch 传进来
（物理量，已经由 ``train`` 里的 :class:`~identify_params.params.ParamGroups` 从 raw 映射好）。
类里只有"固定参数"（几何/λ/ε/偏置 + 被冻结的那些参数，来自 ``base``）与
"步长 / 细化子步 / 积分器"。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .params import (
    EXO_ZERO,
    NPARAM,
    PlanarParams,
    ParamGroups,
    Exo,
    beta_of,
)

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# ============================================================================
# ★ forward 的输入通道布局（可微仿真模型的张量契约）
# ============================================================================
# 参数 2: seq_const [batch, NCONST] —— **整条序列共用**的逐 batch 常量
SEQ_CONST_NAMES = ("q0_motor", "q0_platform", "q0_small",
                   "qd0_motor", "qd0_platform", "qd0_small",
                   "base_omega", "base_alpha")
NCONST = len(SEQ_CONST_NAMES)          # 8
IDX_Q0 = slice(0, 3)
IDX_QD0 = slice(3, 6)
IDX_BASE_OMEGA = 6
IDX_BASE_ALPHA = 7

# 参数 3: seq_var [batch, seq_len, NVAR] —— **每个时间点**都可能不同的量
#   ★ 两路力矩（τ_cmd, τ_small）放在**头两个**通道（用户约定），其后是逐点外生量。
SEQ_VAR_NAMES = ("tau_big", "tau_small", "grav_x", "grav_y", "beta")
NVAR = len(SEQ_VAR_NAMES)              # 5
IDX_TAU_BIG = 0
IDX_TAU_SMALL = 1
IDX_GRAV_X = 2
IDX_GRAV_Y = 3
IDX_BETA = 4


# ============================================================================
# numpy 参考实现（plant / 前向验证 / 回归矩阵 / 评测；与头文件逐项对应）
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
        # ★ 大 yaw 侧: 已知上装质量那份 (m_u_known·d) + 待辨识的偏心 Pb，两者相加不重复计数
        Gb = ((p.Pbx + p.m_u_known * p.dx) * gy
              - (p.Pby + p.m_u_known * p.dy) * gx) + Gs
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
    """★ 半隐式（symplectic）欧拉单步（与旧脚本 `param_ident.py` 逐行同序）。

        ω ← ω + dt·α(q, ω)      # 先用**当前**状态算加速度
        θ ← θ + dt·ω            # 再用**更新后**的 ω 更新角度
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
    """解析回归矩阵 Y（形状 [...,2,NPARAM=18]），满足 τ = Y·φ。

    列顺序: 0 Jbig_eff, 1 Js, 2 Px, 3 Py, 4 fc_big, 5 fv_big, 6 fc_small, 7 fv_small,
             8..15 = 背隙/电机侧（在 3-DOF 的 `eomBacklash` 里，不属于本 2-DOF 子块 ⇒ 恒 0），
             16 Pbx, 17 Pby（只出现在大 yaw 行）。
    （与 include/tcbs/mpc/planar_yaw_model.h::regressor() 完全一致）
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
    # 16,17: Pbx, Pby（列号 = PARAM_NAMES 的下标）——只进大 yaw（云台）行
    Y[..., 0, 16] = -gy
    Y[..., 0, 17] = gx
    # 4..7: 摩擦（只作用于本轴）
    Y[..., 0, 4] = np.tanh(lam * tb)
    Y[..., 0, 5] = tb
    Y[..., 1, 6] = np.tanh(lam * vs)
    Y[..., 1, 7] = vs
    return Y


def simulate_np(p: PlanarParams, q0, qd0, tau_seq, dt, exo: Exo = EXO_ZERO, substeps: int = 1,
                exo_seq=None, integrator: str = "rk4"):
    """**2-DOF** 前向仿真（力矩零阶保持）。tau_seq [T,2]；返回 theta [T,2], dtheta [T,2]。

    ⚠ 这是"平面 2-DOF 子块"的仿真（没有电机/背隙状态），主要用于自检与回归矩阵对照；
    真实数据（有背隙）请用 `simulate_backlash_np`（3-DOF）。
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
# ★ 3-DOF（含大 yaw 背隙）模型 —— numpy 版（与 C++ eomBacklash 逐项镜像）
#   q = (θ_motor, θ_platform, θ_small)；u = (τ_cmd, 0, τ_small)
# ============================================================================
def smooth_relu_np(x, eps):
    """relu_ε(x) = ½(x + √(x²+ε²))（C++ `smoothRelu` 的 numpy 版）。"""
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


def backlash_torque_np(Delta, DDelta, p: PlanarParams):
    """τ_t = k·[dz(Δ) + γ·Δ] + c·Δ̇（C++ `backlashTorque` 的 numpy 版）。"""
    h = 0.5 * p.backlash_delta
    eps = p.backlash_smooth_eps
    return (p.backlash_k * (smooth_relu_np(Delta - h, eps) - smooth_relu_np(-Delta - h, eps)
                            + p.backlash_through * Delta)
            + p.backlash_c * DDelta)


def motor_friction_np(w, p: PlanarParams):
    return p.fc_motor * np.tanh(p.friction_lambda * w) + p.fv_motor * w


def eom_backlash_np(q, qd, p: PlanarParams, exo: Exo = EXO_ZERO):
    """3-DOF 的 ``(M3, h3)``；M3 = blkdiag(J_motor, M2)（块对角 ⇒ 求逆只需标量 + 2×2 逆）。"""
    M2, h2 = eom_np(q[..., 1:3], qd[..., 1:3], p, exo)
    D = q[..., 0] - q[..., 1] - beta_of(p, exo)
    Dd = qd[..., 0] - qd[..., 1]
    tt = backlash_torque_np(D, Dd, p)
    h = np.stack([tt + motor_friction_np(qd[..., 0], p) + p.tau_offset_motor,
                  h2[..., 0] - tt,
                  h2[..., 1]], axis=-1)
    M = np.zeros(np.shape(q)[:-1] + (3, 3), dtype=np.float64)
    M[..., 0, 0] = p.Jmotor
    M[..., 1:, 1:] = M2
    return M, h


def _motor_input_np(tau):
    """(τ_cmd, τ_small) → u = (τ_cmd, 0, τ_small)。"""
    tau = np.asarray(tau, dtype=np.float64)
    zero = np.zeros_like(tau[..., 0])
    return np.stack([tau[..., 0], zero, tau[..., 1]], axis=-1)


def forward_accel_backlash_np(q, qd, tau, p: PlanarParams, exo: Exo = EXO_ZERO):
    """q̈ = M⁻¹(u − h)（3-DOF）。"""
    M, h = eom_backlash_np(q, qd, p, exo)
    r = _motor_input_np(tau) - h
    M11, M12, M22 = M[..., 1, 1], M[..., 1, 2], M[..., 2, 2]
    det2 = M11 * M22 - M12 * M12
    return np.stack([r[..., 0] / M[..., 0, 0],
                     (M22 * r[..., 1] - M12 * r[..., 2]) / det2,
                     (-M12 * r[..., 1] + M11 * r[..., 2]) / det2], axis=-1)


def inverse_dynamics_backlash_np(q, qd, qdd, p: PlanarParams, exo: Exo = EXO_ZERO):
    """τ_gen = M·q̈ + h（3-DOF；用于一致性自检）。"""
    M, h = eom_backlash_np(q, qd, p, exo)
    return np.einsum("...ij,...j->...i", M, qdd) + h


def rk4_step_backlash_np(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1):
    """RK4 单步（3-DOF，力矩零阶保持）。"""
    substeps = max(1, int(substeps))
    hh = dt / substeps
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(substeps):
        k1 = forward_accel_backlash_np(qa, qda, tau, p, exo)
        k2 = forward_accel_backlash_np(qa + 0.5 * hh * qda, qda + 0.5 * hh * k1, tau, p, exo)
        k3 = forward_accel_backlash_np(qa + 0.5 * hh * (qda + 0.5 * hh * k1),
                                       qda + 0.5 * hh * k2, tau, p, exo)
        k4 = forward_accel_backlash_np(qa + hh * (qda + 0.5 * hh * k2),
                                       qda + hh * k3, tau, p, exo)
        qa = qa + (hh / 6.0) * (qda + 2.0 * (qda + 0.5 * hh * k1)
                                + 2.0 * (qda + 0.5 * hh * k2) + (qda + hh * k3))
        qda = qda + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return qa, qda


def euler_step_backlash_np(q, qd, tau, p: PlanarParams, exo: Exo, dt, substeps: int = 1):
    """半隐式（symplectic）欧拉单步（3-DOF）: ω ← ω + h·α; θ ← θ + h·ω。"""
    substeps = max(1, int(substeps))
    hh = dt / substeps
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(substeps):
        qda = qda + hh * forward_accel_backlash_np(qa, qda, tau, p, exo)
        qa = qa + hh * qda
    return qa, qda


def integrate_step_backlash_np(q, qd, tau, p: PlanarParams, exo: Exo, dt,
                               substeps: int = 1, integrator: str = "rk4"):
    name = str(integrator).lower()
    if name in ("euler", "semi-implicit", "semi_implicit"):
        return euler_step_backlash_np(q, qd, tau, p, exo, dt, substeps)
    if name == "rk4":
        return rk4_step_backlash_np(q, qd, tau, p, exo, dt, substeps)
    raise ValueError(f"未知积分器 {integrator!r}（支持 euler | rk4）")


def simulate_backlash_np(p: PlanarParams, q0, qd0, tau_seq, dt, exo: Exo = EXO_ZERO,
                         substeps: int = 4, exo_seq=None, integrator: str = "rk4",
                         beta_seq=None):
    """**3-DOF** 前向仿真（力矩零阶保持）。tau_seq [T,2]；返回 theta [T,3], dtheta [T,3]。

    与 C++ `integrateStepBacklash` 同一组方程。``beta_seq`` 非 None 时**逐步**替换
    ``exo.backlash_beta``（= 数据里那列在线估计的 β）。
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
        ex = exo if exo_seq is None else exo_seq[i]
        if beta_seq is not None:
            ex = replace(ex, backlash_beta=beta_seq[i])
        q, qd = integrate_step_backlash_np(q, qd, tau_seq[i], p, ex, dt, substeps, integrator)
    return th, dth


def rollout_backlash_batched_np(p: PlanarParams, seq_const, seq_var,
                                dt: float, substeps: int = 4, integrator: str = "rk4"):
    """**按 batch 向量化**的 3-DOF 前向（numpy；与 `simulate_backlash_np` 同逻辑）。

    输入用的是与可微模型**同一张量契约**:
        seq_const [B,8]   q0(3) + qd0(3) + base_omega + base_alpha
        seq_var   [B,T,5] tau_big, tau_small, grav_x, grav_y, β
    返回 ``(theta [B,T,3], dtheta [B,T,3])``（第 i 行 = 积分**前**的状态，与逐段版一致）。

    为什么单独写: 逐段版（`simulate_backlash_np`）是 Python 标量循环，喂优化器太慢
    （实测比本函数慢 40~120×）。本函数直接复用同一个 numpy 版 `rk4_step_backlash_np`
    对 ``[B,3]`` 状态整批推进 —— 不引入第二份模型公式。
    """
    sc = np.asarray(seq_const, dtype=np.float64)
    sv = np.asarray(seq_var, dtype=np.float64)
    if sc.ndim != 2 or sc.shape[1] != NCONST:
        raise ValueError(f"seq_const 形状应为 [B,{NCONST}]，得到 {sc.shape}")
    if sv.ndim != 3 or sv.shape[2] != NVAR:
        raise ValueError(f"seq_var 形状应为 [B,T,{NVAR}]，得到 {sv.shape}")
    B, T = int(sv.shape[0]), int(sv.shape[1])
    q = np.array(sc[:, IDX_Q0], dtype=np.float64)
    qd = np.array(sc[:, IDX_QD0], dtype=np.float64)
    wc = sc[:, IDX_BASE_OMEGA]
    ac = sc[:, IDX_BASE_ALPHA]
    th = np.empty((B, T, 3), dtype=np.float64)
    dth = np.empty((B, T, 3), dtype=np.float64)
    for t in range(T):
        th[:, t] = q
        dth[:, t] = qd
        exo = Exo(gravity_a=(sv[:, t, IDX_GRAV_X], sv[:, t, IDX_GRAV_Y]), gravity_on=True,
                  base_omega=wc, base_alpha=ac, backlash_beta=sv[:, t, IDX_BETA])
        q, qd = integrate_step_backlash_np(q, qd, sv[:, t, IDX_TAU_BIG:IDX_TAU_SMALL + 1], p,
                                           exo, dt, substeps, integrator)
    return th, dth



#   参数字段允许是**标量**（固定参数）或 **[B] 张量**（可学习参数按 batch 广播）。
# ============================================================================
@dataclass
class StepExo:
    """单个时间点的外生量（可导前向仿真内部用）。"""

    gx: object = 0.0
    gy: object = 0.0
    gravity_on: bool = False
    wc: object = 0.0
    ac: object = 0.0
    wc_on: bool = False
    ac_on: bool = False
    beta: object = 0.0          # 已解析好的 β（标量或 [B]），不再回退


def torch_smooth_relu(x, eps: float):
    return 0.5 * (x + torch.sqrt(x * x + eps * eps))


def torch_backlash_torque(D, Dd, p: PlanarParams):
    """τ_t = k·[dz(Δ) + γ·Δ] + c·Δ̇（平滑死区；ε = p.backlash_smooth_eps）。"""
    h = 0.5 * p.backlash_delta
    eps = p.backlash_smooth_eps
    dz = torch_smooth_relu(D - h, eps) - torch_smooth_relu(-D - h, eps)
    return p.backlash_k * (dz + p.backlash_through * D) + p.backlash_c * Dd


def torch_platform_eom(q, qd, p: PlanarParams, step: StepExo):
    """2-DOF 子块的 (M11, M12, h[...,2])；q/qd 形状 [...,2]。

    对「重力=0、底盘静止」的常见工况跳过相应张量运算（纯提速）。
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

    if step.gravity_on:
        Gs = Qx * step.gy - Qy * step.gx
        # ★ 大 yaw 侧: 已知上装质量那份 (m_u_known·d) + 待辨识的偏心 Pb，两者相加不重复计数
        Gb = ((p.Pbx + p.m_u_known * p.dx) * step.gy
              - (p.Pby + p.m_u_known * p.dy) * step.gx) + Gs
        h0 = h0 - Gb
        h1 = h1 - Gs
    if step.wc_on:
        h0 = h0 + (mu * ts) * step.wc
        h1 = h1 + (-mu * tb) * step.wc - (half_mu * step.wc) * step.wc
    if step.ac_on:
        h0 = h0 + M11 * step.ac
        h1 = h1 + M12 * step.ac
    h0 = h0 + p.fc_big * torch.tanh(p.friction_lambda * tb) + p.fv_big * tb + p.tau_offset_big
    h1 = h1 + p.fc_small * torch.tanh(p.friction_lambda * ts) + p.fv_small * ts + p.tau_offset_small
    return M11, M12, torch.stack([h0, h1], dim=-1)


def torch_forward_accel_backlash(q, qd, tau_big, tau_small, p: PlanarParams, step: StepExo):
    """q̈ = M⁻¹(u − h)（3-DOF，可导）。τ 逐 batch 传入。"""
    M11, M12, h2 = torch_platform_eom(q[..., 1:3], qd[..., 1:3], p, step)
    tt = torch_backlash_torque(q[..., 0] - q[..., 1] - step.beta,
                               qd[..., 0] - qd[..., 1], p)
    h0 = (tt + p.fc_motor * torch.tanh(p.friction_lambda * qd[..., 0])
          + p.fv_motor * qd[..., 0] + p.tau_offset_motor)
    h1 = h2[..., 0] - tt
    h2s = h2[..., 1]
    Js = p.Js
    det2 = M11 * Js - M12 * M12
    r0 = tau_big - h0
    r1 = -h1
    r2 = tau_small - h2s
    return torch.stack([r0 / p.Jmotor,
                        (Js * r1 - M12 * r2) / det2,
                        (M11 * r2 - M12 * r1) / det2], dim=-1)


def torch_rk4_step_backlash(q, qd, tau_big, tau_small, p: PlanarParams, step: StepExo,
                            dt, substeps: int = 1):
    """RK4 单步（3-DOF，力矩零阶保持，可导）。"""
    hh = dt / max(1, int(substeps))
    for _ in range(max(1, int(substeps))):
        k1 = torch_forward_accel_backlash(q, qd, tau_big, tau_small, p, step)
        k2 = torch_forward_accel_backlash(q + 0.5 * hh * qd, qd + 0.5 * hh * k1,
                                          tau_big, tau_small, p, step)
        k3 = torch_forward_accel_backlash(q + 0.5 * hh * (qd + 0.5 * hh * k1),
                                          qd + 0.5 * hh * k2, tau_big, tau_small, p, step)
        k4 = torch_forward_accel_backlash(q + hh * (qd + 0.5 * hh * k2), qd + hh * k3,
                                          tau_big, tau_small, p, step)
        q = q + (hh / 6.0) * (qd + 2.0 * (qd + 0.5 * hh * k1)
                              + 2.0 * (qd + 0.5 * hh * k2) + (qd + hh * k3))
        qd = qd + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return q, qd


def torch_euler_step_backlash(q, qd, tau_big, tau_small, p: PlanarParams, step: StepExo,
                              dt, substeps: int = 1):
    """半隐式（symplectic）欧拉单步（3-DOF，可导）。"""
    n = max(1, int(substeps))
    hh = dt / n
    for _ in range(n):
        qd = qd + hh * torch_forward_accel_backlash(q, qd, tau_big, tau_small, p, step)
        q = q + hh * qd
    return q, qd


def torch_integrate_step_backlash(q, qd, tau_big, tau_small, p: PlanarParams, step: StepExo,
                                  dt, substeps: int = 1, integrator: str = "rk4"):
    """按名称分派可导积分器（"euler" | "rk4"）。"""
    name = str(integrator).lower()
    if name in ("euler", "semi-implicit", "semi_implicit"):
        return torch_euler_step_backlash(q, qd, tau_big, tau_small, p, step, dt, substeps)
    if name == "rk4":
        return torch_rk4_step_backlash(q, qd, tau_big, tau_small, p, step, dt, substeps)
    raise ValueError(f"未知积分器 {integrator!r}（支持 euler | rk4）")


# ============================================================================
# ★ 可微仿真模型（不含可学习参数）
# ============================================================================
# 环境缺 torch 时仍允许 import 本模块（numpy 参考实现照常可用），只有实例化才报错。
_ModuleBase = torch.nn.Module if torch is not None else object


class DifferentiableSimulator(_ModuleBase):
    """3-DOF（含大 yaw 背隙）**可导**前向仿真模型。

    构造时传入（★ 用户约定）: **步长 dt、细化子步 substeps**，以及积分器（保留 euler 消融）、
    固定参数所在的 ``base`` 与描述可学习参数列顺序的 :class:`ParamGroups`。
    可学习参数**不是**成员: 每次 ``forward`` 由调用方按 batch 传进来。

    ``forward(params, seq_const, seq_var)`` 的 3 个参数（★ 用户约定的张量契约）::

        params     [batch, param_num]       物理量（已从 raw 映射好），第 k 列 = learnable_names[k]
        seq_const  [batch, NCONST=8]        整条序列共用的逐 batch 常量:
                                            q0(3) + qd0(3) + base_omega + base_alpha
        seq_var    [batch, seq_len, NVAR=5] 每个时间点都可能不同的量（★ 力矩在头两个）:
                                            τ_big, τ_small, grav_x, grav_y, β

    返回 ``(pos, vel)`` 两个元组，每个元组内是 3 个 ``[batch, seq_len]`` 张量
    （电机 / 云台 / 小 yaw 的位置与速度）。

    ★ 逐点外生量（重力方向、背隙中心 β）都从 ``seq_var`` 读:
      · ``gravity_on`` 为 ``None`` 时按 ``seq_var`` 里是否有非零重力自动判定；
      · ``beta_from_input=True`` 时用 ``seq_var[..., IDX_BETA]``（= 数据里在线估计的那列，
        与 MPC 运行期一致）；否则用模型参数 ``p.backlash_beta``（可学习或冻结）。
    """

    SEQ_CONST_NAMES = SEQ_CONST_NAMES
    SEQ_VAR_NAMES = SEQ_VAR_NAMES
    NCONST = NCONST
    NVAR = NVAR

    def __init__(self, dt: float, substeps: int = 4, integrator: str = "rk4",
                 layout: ParamGroups | None = None, base: PlanarParams | None = None,
                 beta_from_input: bool = False, gravity_on: bool | None = None):
        super().__init__()
        if torch is None:                       # pragma: no cover
            raise RuntimeError(f"需要 torch: {_TORCH_IMPORT_ERROR}")
        name = str(integrator).lower()
        if name not in ("rk4", "euler", "semi-implicit", "semi_implicit"):
            raise ValueError(f"未知积分器 {integrator!r}（支持 euler | rk4）")
        if base is None:
            base = layout.base if layout is not None else PlanarParams()
        self.dt = float(dt)
        self.substeps = max(1, int(substeps))
        self.integrator = name
        self.base = base
        self.layout = layout if layout is not None else ParamGroups(base)
        self.beta_from_input = bool(beta_from_input)
        self.gravity_on = gravity_on

    # ── 打包辅助（把布局约定集中在一处，训练逻辑不必记通道号）──
    @staticmethod
    def pack_seq_const(q0, qd0, base_omega=0.0, base_alpha=0.0):
        """[B,3]+[B,3]+标量 → [B,8]。"""
        b = q0.shape[0]
        wc = torch.as_tensor(base_omega, dtype=q0.dtype, device=q0.device).expand(b)
        ac = torch.as_tensor(base_alpha, dtype=q0.dtype, device=q0.device).expand(b)
        return torch.cat([q0, qd0, wc[:, None], ac[:, None]], dim=1)

    @staticmethod
    def pack_seq_var(tau_big, tau_small, grav=None, beta=None):
        """[B,T]+[B,T]+[B,T,2]+[B,T] → [B,T,5]。

        ``grav`` 是 [B,T,2] 的 (gx, gy)，与 ``seq_var`` 里的通道号不同 ⇒ 这里用自己的
        二维下标，避免把 ``IDX_GRAV_X`` 误当成 grav 的列号。
        """
        b, t = tau_big.shape
        z = torch.zeros(b, t, dtype=tau_big.dtype, device=tau_big.device)
        gx = z if grav is None else grav[..., 0]
        gy = z if grav is None else grav[..., 1]
        be = z if beta is None else beta
        return torch.stack([tau_big, tau_small, gx, gy, be], dim=-1)

    # ── 主入口 ──
    def forward(self, params, seq_const, seq_var):
        if seq_const.shape[-1] != NCONST:
            raise ValueError(f"seq_const 末维应为 {NCONST}（{SEQ_CONST_NAMES}），"
                             f"得到 {tuple(seq_const.shape)}")
        if seq_var.shape[-1] != NVAR:
            raise ValueError(f"seq_var 末维应为 {NVAR}（{SEQ_VAR_NAMES}），"
                             f"得到 {tuple(seq_var.shape)}")
        p = self.layout.build_params(params, self.base)

        B, T = int(seq_var.shape[0]), int(seq_var.shape[1])
        if getattr(params, "dim", None) is not None and params.dim() == 2 and params.shape[0] != B:
            raise ValueError(f"params 的第 0 维（{params.shape[0]}）应为 batch={B}")
        q = seq_const[:, IDX_Q0]
        qd = seq_const[:, IDX_QD0]
        wc_all = seq_const[:, IDX_BASE_OMEGA]
        ac_all = seq_const[:, IDX_BASE_ALPHA]
        wc_on = bool(torch.any(wc_all != 0).item())
        ac_on = bool(torch.any(ac_all != 0).item())
        if self.gravity_on is None:
            grav_on = bool(torch.any(seq_var[..., IDX_GRAV_X:IDX_GRAV_Y + 1] != 0).item())
        else:
            grav_on = bool(self.gravity_on)
        tau_big = seq_var[..., IDX_TAU_BIG]
        tau_small = seq_var[..., IDX_TAU_SMALL]
        beta_all = seq_var[..., IDX_BETA] if self.beta_from_input else p.backlash_beta

        pos = [[] for _ in range(3)]
        vel = [[] for _ in range(3)]
        for t in range(T):
            for k in range(3):
                pos[k].append(q[:, k])
                vel[k].append(qd[:, k])
            step = StepExo(
                gx=seq_var[:, t, IDX_GRAV_X] if grav_on else 0.0,
                gy=seq_var[:, t, IDX_GRAV_Y] if grav_on else 0.0,
                gravity_on=grav_on,
                wc=wc_all if wc_on else 0.0,
                ac=ac_all if ac_on else 0.0,
                wc_on=wc_on,
                ac_on=ac_on,
                beta=beta_all[:, t] if self.beta_from_input else beta_all,
            )
            q, qd = torch_integrate_step_backlash(q, qd, tau_big[:, t], tau_small[:, t], p, step,
                                                  self.dt, self.substeps, self.integrator)
        return (tuple(torch.stack(s, dim=1) for s in pos),
                tuple(torch.stack(s, dim=1) for s in vel))

    # ── 明确记录 batch（``B`` 只用于形状检查，不改变语义）──
    def extra_repr(self) -> str:
        return (f"dt={self.dt:g}, substeps={self.substeps}, integrator={self.integrator}, "
                f"n_learnable={self.layout.n_learnable}, beta_from_input={self.beta_from_input}")
