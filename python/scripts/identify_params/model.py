#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""② 可微仿真模型 —— **两刚体平面拉格朗日模型的缩合参数版**（辨识用；主模型不动）。

状态 ``q = (θ_b, θ_s)`` = 两轴**相对关节角**（b = 云台侧，s = 小 yaw 侧）;
输入 ``u = (T_b, T_s)`` = 两轴控制力矩；外生量 ``(g_ax, g_ay)`` = **A 系**重力平面分量、
``ω_c`` = 底盘角速度。``α_c ≡ 0``、没有背隙/电机自由度。

方程（``D = (dx, dy)`` 已知，``R`` 为绕 z 的旋转）::

    Q      = R(θ_s)·(X_s, Y_s)
    μK     = D_x·Q_x + D_y·Q_y          μK' = D_y·Q_x − D_x·Q_y
    J_b    = I_b + |X_b|²               J_s = I_s + (X_s²+Y_s²)/μ
    A      = J_b + μ|D|²   B = J_s
    Δ      = A·B − (μK)²   ≥ J_s·J_b + μ|D|²·I_s > 0     ★构造保证
    M      = [[A+B+2μK, B+μK], [B+μK, B]]

    G_s    = X_s(g_ax·sinθ_s − g_ay·cosθ_s) + Y_s(g_ax·cosθ_s + g_ay·sinθ_s)
    G_b    = −(X_b+μdx)·g_ay + (Y_b+μdy)·g_ax + G_s
    Q_b    = T_b − f_bc·tanh(λ·θ̇_b) − f_bv·θ̇_b        （Q_s 同理）
    R_b    = Q_b − G_b − μK'·θ̇_s·[2(ω_c+θ̇_b) + θ̇_s]
    R_s    = Q_s − G_s + μK'·(ω_c+θ̇_b)²
    θ̈_b    = [B·R_b − (B+μK)·R_s]/Δ
    θ̈_s    = [(A+B+2μK)·R_s − (B+μK)·R_b]/Δ

★ 重力用 **A 系逐样本**写法（= 数据列 `gravity_ax/ay`），与"世界系 g + 绝对角 ψ_b/ψ_s"
  的写法**严格等价**（已验证到 1e-17），但不需要 θ_c 通道、也没有世界系 g 的反算误差。

张量契约::

    seq_const [B,5]   = q0_b, q0_s, qd0_b, qd0_s, base_omega
    seq_var   [B,T,4] = tau_b, tau_s, grav_ax, grav_ay
    forward(params[B,11], seq_const, seq_var) -> (theta[B,T,2], dtheta[B,T,2])
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .params import EXO_ZERO, Exo, PlanarParams, ParamGroups

# ============================================================================
# 张量契约
# ============================================================================
SEQ_CONST_NAMES = ("q0_b", "q0_s", "qd0_b", "qd0_s", "base_omega")
NCONST = len(SEQ_CONST_NAMES)                     # 5
IDX_Q0 = slice(0, 2)
IDX_QD0 = slice(2, 4)
IDX_BASE_OMEGA = 4

SEQ_VAR_NAMES = ("tau_b", "tau_s", "grav_ax", "grav_ay")
NVAR = len(SEQ_VAR_NAMES)                         # 4
IDX_TAU_B = 0
IDX_TAU_S = 1
IDX_GRAV_X = 2
IDX_GRAV_Y = 3

NHORIZON_CH = 2                                   # 状态/损失通道数


# ============================================================================
# 派生量 / 加速度（numpy；支持前导批维）
# ============================================================================
def derived_np(qs, p: PlanarParams):
    """``(A, B, muK, muKp, Delta)``；``qs`` = θ_s（标量或任意形状数组）。"""
    cs = np.cos(qs)
    sn = np.sin(qs)
    Qx = p.X_s * cs - p.Y_s * sn
    Qy = p.X_s * sn + p.Y_s * cs
    muK = p.dx * Qx + p.dy * Qy
    muKp = p.dy * Qx - p.dx * Qy
    A, B = p.A, p.B
    return A, B, muK, muKp, A * B - muK * muK


def gravity_np(qs, g_ax, g_ay, p: PlanarParams):
    """**A 系**重力 → ``(G_b, G_s)``（只依赖 θ_s，不含 ψ_b/θ_c）。"""
    ss, cs = np.sin(qs), np.cos(qs)
    G_s = p.X_s * (g_ax * ss - g_ay * cs) + p.Y_s * (g_ax * cs + g_ay * ss)
    G_b = -(p.X_b + p.mu * p.dx) * g_ay + (p.Y_b + p.mu * p.dy) * g_ax + G_s
    return G_b, G_s


def accel_np(q, qd, u, g_ax, g_ay, wc, p: PlanarParams):
    """``q̈``；``q``/``qd`` 形状 ``[...,2]``，``u`` = ``(T_b, T_s)``，``wc`` = ω_c。"""
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    qs = q[..., 1]
    vb, vs = qd[..., 0], qd[..., 1]

    A, B, muK, muKp, Delta = derived_np(qs, p)
    G_b, G_s = gravity_np(qs, g_ax, g_ay, p)
    lam = p.friction_lambda
    Q_b = u[..., 0] - p.f_bc * np.tanh(lam * vb) - p.f_bv * vb
    Q_s = u[..., 1] - p.f_sc * np.tanh(lam * vs) - p.f_sv * vs
    R_b = Q_b - G_b - muKp * vs * (2.0 * (wc + vb) + vs)
    R_s = Q_s - G_s + muKp * (wc + vb) ** 2
    BpK = B + muK
    return np.stack([(B * R_b - BpK * R_s) / Delta,
                     ((A + B + 2.0 * muK) * R_s - BpK * R_b) / Delta], axis=-1)


def rk4_step_np(q, qd, u, g_ax, g_ay, wc, p: PlanarParams, dt, substeps: int = 4):
    """RK4 单步（力矩零阶保持）。"""
    ns = max(1, int(substeps))
    hh = dt / ns
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(ns):
        k1 = accel_np(qa, qda, u, g_ax, g_ay, wc, p)
        k2 = accel_np(qa + 0.5 * hh * qda, qda + 0.5 * hh * k1, u, g_ax, g_ay, wc, p)
        k3 = accel_np(qa + 0.5 * hh * (qda + 0.5 * hh * k1), qda + 0.5 * hh * k2,
                      u, g_ax, g_ay, wc, p)
        k4 = accel_np(qa + hh * (qda + 0.5 * hh * k2), qda + hh * k3, u, g_ax, g_ay, wc, p)
        qa = qa + hh * qda + (hh * hh / 6.0) * (k1 + k2 + k3)
        qda = qda + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return qa, qda


def euler_step_np(q, qd, u, g_ax, g_ay, wc, p: PlanarParams, dt, substeps: int = 1):
    """前向 Euler 单步（只用于对照/自检）。"""
    ns = max(1, int(substeps))
    hh = dt / ns
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(ns):
        a = accel_np(qa, qda, u, g_ax, g_ay, wc, p)
        qa = qa + hh * qda
        qda = qda + hh * a
    return qa, qda


# ============================================================================
# 批量前向（numpy）—— 与 torch/C++ 同一张量契约
# ============================================================================
def rollout_batched_np(p: PlanarParams, seq_const, seq_var, dt: float,
                       substeps: int = 4, integrator: str = "rk4"):
    """``(theta [B,T,2], dtheta [B,T,2])``（第 t 行 = 积分**前**状态）。"""
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
    step = rk4_step_np if integrator == "rk4" else euler_step_np
    th = np.empty((B, T, 2), dtype=np.float64)
    dth = np.empty((B, T, 2), dtype=np.float64)
    for t in range(T):
        th[:, t] = q
        dth[:, t] = qd
        q, qd = step(q, qd, sv[:, t, IDX_TAU_B:IDX_TAU_S + 1], sv[:, t, IDX_GRAV_X],
                     sv[:, t, IDX_GRAV_Y], wc, p, dt, substeps)
    return th, dth


def simulate_np(p: PlanarParams, q0, qd0, tau, dt: float, exo: Exo = EXO_ZERO,
                substeps: int = 4, exo_seq=None, integrator: str = "rk4"):
    """单段标量推演（画图/单段评测用）。

    ``tau [T,2]``；``exo`` 给常量重力/ω_c，``exo_seq``（长度 T 的 Exo 列表）给逐样本值。
    """
    tau = np.asarray(tau, dtype=np.float64)
    T = int(tau.shape[0])
    q = np.array(q0, dtype=np.float64)
    qd = np.array(qd0, dtype=np.float64)
    th = np.empty((T, 2), dtype=np.float64)
    dth = np.empty((T, 2), dtype=np.float64)
    step = rk4_step_np if integrator == "rk4" else euler_step_np
    for t in range(T):
        th[t] = q
        dth[t] = qd
        e = exo if exo_seq is None else exo_seq[t]
        gx, gy = e.gravity_a
        q, qd = step(q, qd, tau[t], gx, gy, e.base_omega, p, dt, substeps)
    return th, dth


# ============================================================================
# 可微仿真（torch）—— 模型不持有任何可学习参数
# ============================================================================
class DifferentiableSimulator:
    """RK4 可微前向；``forward(params, seq_const, seq_var)``。

    ``params`` 可以是 ``[P]``（所有 batch 共用）或 ``[B,P]``（逐 batch）；
    ``layout.build_params`` 把可学习值展开成 :class:`PlanarParams`。
    """

    SEQ_CONST_NAMES = SEQ_CONST_NAMES
    SEQ_VAR_NAMES = SEQ_VAR_NAMES
    NCONST, NVAR = NCONST, NVAR

    def __init__(self, dt: float, substeps: int = 4, integrator: str = "rk4",
                 layout: ParamGroups | None = None, base: PlanarParams | None = None):
        if torch is None:
            raise RuntimeError(f"需要 torch: {_TORCH_IMPORT_ERROR}")
        self.dt = float(dt)
        self.substeps = max(1, int(substeps))
        self.integrator = str(integrator)
        self.layout = layout
        self.base = PlanarParams() if base is None else base

    # ── 打包 ──
    @staticmethod
    def pack_seq_const(q0, qd0, base_omega):
        """``[B,2]+[B,2]+[B]|标量 → [B,5]``。"""
        b = int(q0.shape[0])
        if torch.is_tensor(base_omega):
            w = base_omega.reshape(b, 1).expand(b, 1)
        else:
            w = torch.full((b, 1), float(base_omega), dtype=q0.dtype, device=q0.device)
        return torch.cat([q0, qd0, w], dim=1)

    @staticmethod
    def pack_seq_var(tau_b, tau_s, grav_ax=None, grav_ay=None):
        """``[B,T]+[B,T]+[B,T]+[B,T] → [B,T,4]``。"""
        b, t = tau_b.shape
        z = torch.zeros(b, t, dtype=tau_b.dtype, device=tau_b.device)
        return torch.stack([tau_b, tau_s,
                            z if grav_ax is None else grav_ax,
                            z if grav_ay is None else grav_ay], dim=-1)

    # ── 单步加速度 ──
    def _accel(self, q, qd, u, gx, gy, wc, p: PlanarParams):
        qs = q[..., 1]
        vb, vs = qd[..., 0], qd[..., 1]
        cs, sn = torch.cos(qs), torch.sin(qs)
        Qx = p.X_s * cs - p.Y_s * sn
        Qy = p.X_s * sn + p.Y_s * cs
        muK = p.dx * Qx + p.dy * Qy
        muKp = p.dy * Qx - p.dx * Qy
        A = p.J_b + p.mu * (p.dx * p.dx + p.dy * p.dy)
        B = p.J_s
        Delta = A * B - muK * muK
        G_s = p.X_s * (gx * sn - gy * cs) + p.Y_s * (gx * cs + gy * sn)
        G_b = -(p.X_b + p.mu * p.dx) * gy + (p.Y_b + p.mu * p.dy) * gx + G_s
        lam = p.friction_lambda
        Q_b = u[..., 0] - p.f_bc * torch.tanh(lam * vb) - p.f_bv * vb
        Q_s = u[..., 1] - p.f_sc * torch.tanh(lam * vs) - p.f_sv * vs
        R_b = Q_b - G_b - muKp * vs * (2.0 * (wc + vb) + vs)
        R_s = Q_s - G_s + muKp * (wc + vb) ** 2
        BpK = B + muK
        return torch.stack([(B * R_b - BpK * R_s) / Delta,
                            ((A + B + 2.0 * muK) * R_s - BpK * R_b) / Delta], dim=-1)

    # ── 主入口 ──
    def forward(self, params, seq_const, seq_var):
        if seq_const.shape[-1] != NCONST:
            raise ValueError(f"seq_const 末维应为 {NCONST}（{SEQ_CONST_NAMES}）")
        if seq_var.shape[-1] != NVAR:
            raise ValueError(f"seq_var 末维应为 {NVAR}（{SEQ_VAR_NAMES}）")
        p = (self.layout.build_params(params, self.base)
             if self.layout is not None else params)
        T = int(seq_var.shape[1])
        q = seq_const[:, IDX_Q0]
        qd = seq_const[:, IDX_QD0]
        wc = seq_const[:, IDX_BASE_OMEGA]
        th = []
        dth = []
        hh = self.dt / self.substeps
        for t in range(T):
            th.append(q)
            dth.append(qd)
            u = seq_var[:, t, IDX_TAU_B:IDX_TAU_S + 1]
            gx = seq_var[:, t, IDX_GRAV_X]
            gy = seq_var[:, t, IDX_GRAV_Y]
            if self.integrator == "euler":
                a = self._accel(q, qd, u, gx, gy, wc, p)
                q = q + hh * qd
                qd = qd + hh * a
                continue
            for _ in range(self.substeps):
                k1 = self._accel(q, qd, u, gx, gy, wc, p)
                k2 = self._accel(q + 0.5 * hh * qd, qd + 0.5 * hh * k1, u, gx, gy, wc, p)
                k3 = self._accel(q + 0.5 * hh * (qd + 0.5 * hh * k1),
                                 qd + 0.5 * hh * k2, u, gx, gy, wc, p)
                k4 = self._accel(q + hh * (qd + 0.5 * hh * k2), qd + hh * k3,
                                 u, gx, gy, wc, p)
                q = q + hh * qd + (hh * hh / 6.0) * (k1 + k2 + k3)
                qd = qd + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.stack(th, dim=1), torch.stack(dth, dim=1)

    __call__ = forward          # `sim(params, seq_const, seq_var)` 与旧接口一致


__all__ = [
    "SEQ_CONST_NAMES", "NCONST", "IDX_Q0", "IDX_QD0", "IDX_BASE_OMEGA",
    "SEQ_VAR_NAMES", "NVAR", "IDX_TAU_B", "IDX_TAU_S", "IDX_GRAV_X", "IDX_GRAV_Y",
    "NHORIZON_CH", "derived_np", "gravity_np", "accel_np", "rk4_step_np", "euler_step_np",
    "rollout_batched_np", "simulate_np", "DifferentiableSimulator",
]
