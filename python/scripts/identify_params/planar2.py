#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两刚体平面拉格朗日模型 —— **缩合参数版**（辨识用；主模型不动）。

来源: 另一会话给出的模型总结（只搬模型本体，不采纳其"怎么辨识"的建议）。

────────────────────────────────────────────────────────────────────────────
一、模型（与总结里的公式一一对应）
────────────────────────────────────────────────────────────────────────────
广义坐标是**相对关节角**: ``q = (θ_b, θ_s)``，其中

    ψ_b = θ_c + θ_b        （b 的绝对方位角）
    ψ_s = ψ_b + θ_s        （s 的绝对方位角）

``θ_c(t) = θ_c0 + ω_c·t`` 已知、``ω_c`` 为常数 ⇒ ``α_c = 0``（本模型不含 α_c 项；
若以后要用非零 α_c，需要把旧 3-DOF 模型里的 ``M11·α_c`` / ``M12·α_c`` 补回来）。
``g = (g_x, g_y)`` 是**世界系**重力平面分量（每条数据段一个常量）。

缩合参数（``D = (D_x, D_y)`` **已知**、不辨识）:

    X_b = m_b·P_bx,  Y_b = m_b·P_by          （b 侧一阶矩）
    X_s = m_s·P_sx,  Y_s = m_s·P_sy          （s 侧一阶矩）
    I_b, I_s                                 （各自**绕质心**的转动惯量，>0）
    μ   = m_s
    f_bc, f_bv, f_sc, f_sv                   （库仑/粘滞摩擦）

★ 用 ``(I_b, I_s)`` 而**不是** ``(J_b, J_s)`` 当自由参数（两者等价，个数相同）:

    J_b = I_b + |P_b|²      （m_b 已固定为 1）
    J_s = I_s + (X_s²+Y_s²)/μ

  好处: ``I_b, I_s, μ > 0`` ⇒ ``Δ ≥ J_s·J_b + μ|D|²·I_s > 0`` **由构造保证**
  （不再需要靠搜索盒/滑动条去挡住 ``I_s < 0`` 那条 NaN 通路）。

派生量:

    A     = J_b + μ·|D|²
    B     = J_s
    μK    = D·R(θ_s)·(X_s, Y_s) = D_x·Q_x + D_y·Q_y,  Q = R(θ_s)·(X_s, Y_s)
    μK'   = d(μK)/dθ_s = D_y·Q_x − D_x·Q_y
    Δ     = A·B − (μK)²
    M     = [[A+B+2μK, B+μK], [B+μK, B]]

    G_b = (X_b+μD_x)(g_x sinψ_b − g_y cosψ_b) + (Y_b+μD_y)(g_x cosψ_b + g_y sinψ_b) + G_s
    G_s = X_s(g_x sinψ_s − g_y cosψ_s) + Y_s(g_x cosψ_s + g_y sinψ_s)

    Q_b = T_b − f_bc·tanh(100·θ̇_b) − f_bv·θ̇_b
    Q_s = T_s − f_sc·tanh(100·θ̇_s) − f_sv·θ̇_s
    R_b = Q_b − G_b − μK'·θ̇_s·[2(ω_c+θ̇_b) + θ̇_s]
    R_s = Q_s − G_s + μK'·(ω_c+θ̇_b)²

    θ̈_b = [B·R_b − (B+μK)·R_s] / Δ
    θ̈_s = [(A+B+2μK)·R_s − (B+μK)·R_b] / Δ

────────────────────────────────────────────────────────────────────────────
二、与 ``model.py``（3-DOF 含背隙）里"云台+小 yaw"子块的关系（★ 已逐项验证）
────────────────────────────────────────────────────────────────────────────
把旧模型里的 (θ_b, θ_s) 取成 (θ_platform, θ_small)、把电机那一维去掉（电机刚性锁定、
背隙不建模）、并把旧模型的平台行输入换成 T_b，则**两者完全等价**，只要：

    A ↔ Jbig_eff          B ↔ Js          μK ↔ dQ = D·R(θ_s)·P
    (X_b+μD_x, Y_b+μD_y) ↔ (Pbx, Pby)     (X_s, Y_s) ↔ (Px, Py)
    f_bc/f_bv ↔ fc_big/fv_big             f_sc/f_sv ↔ fc_small/fv_small

（注意符号约定: 本模型的 G_b/G_s 与旧模型的 Gb/Gs **反号**，因为旧模型写成
 ``M·q̈ + h = u``、本模型写成 ``M·q̈ = R``。）

★ **μ 是规范自由度**: μ 只以 ``μ·|D|²``、``X_b+μD_x``、``Y_b+μD_y`` 三种组合进入模型
  （``μK`` 里 μ 已约掉）⇒ ``(μ, J_b, X_b, Y_b)`` 沿 ``μ→μ+δ, J_b→J_b−δ|D|²,
  X_b→X_b−δD_x, Y_b→Y_b−δD_y`` 完全不改变动力学。所以 μ **不可由动力学唯一确定**；
  要唯一化必须固定 μ（或固定 J_b/X_b/Y_b 之一）。本文件把 μ 留作普通参数，
  由使用方决定是否固定（见 ``FIXABLE``）。

⚠ **``Δ > 0`` 由构造保证**（前提就是 ``I_b, I_s, μ > 0``）:

    Δ = A·B − (μK)² = (J_b + μ|D|²)·J_s − (D·R(θ_s)(X_s,Y_s))²
      ≥ J_s·J_b + μ|D|²·I_s            （用了 (D·Q)² ≤ |D|²·|Q|²）

  RHS 三项全部 > 0 ⇒ Δ > 0，任何 θ_s 都不会出现 Δ≤0。反过来若像原始缩合参数那样把
  ``J_s`` 当自由正参数、不约束 ``I_s = J_s − (X_s²+Y_s²)/μ ≥ 0``，则实测 2148/4000
  的随机样本会在某些 θ_s 上出现 Δ<0（= 旧 3-DOF 模型 ``det2<0`` 那条 NaN 通路）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

FRICTION_LAMBDA = 100.0        # tanh 软符号陡度（与主模型一致，固定不辨识）


# ============================================================================
# 参数表（唯一来源: default 字段）
# ============================================================================
@dataclass(frozen=True)
class Spec:
    name: str
    unit: str
    positive: bool          # True = 正数（log 参数化）; False = 全体实数
    default: float
    note: str = ""


#  ★ 默认值的选择: 让本模型**逐位复现**当前 `params.py` 默认参数下旧模型
#    "云台+小yaw 子块"的动力学；μ 只能在这个交叠区间里取（见文件头"μ 是规范自由度"
#    与 "Δ>0 必须 I_s≥0"）:
#        0.0078²/0.00057 = 0.1068 ≤ μ ≤ 0.5449  (= Jbig_eff/|D|² 与 I_b≥0 的共同区间)
#    取 μ = 0.2（两个惯性都还明显为正），于是:
#      J_b = |X_b|²+I_b = 4.9e-5+1.641e-3 = 1.69e-3；A = J_b + μ|D|² = 2.67e-3 = 旧 Jbig_eff ✓
#      (X_b+μD_x, Y_b+μD_y) = (0, -0.007+0.014) = (0, 0.007) = 旧 (Pbx, Pby) ✓
#      (X_s, Y_s) = (0.00025, -0.0078) = 旧 (Px, Py) ✓,  J_s = 0.00057 = 旧 Js ✓
#      I_s = 2.65e-4 > 0, I_b = 1.64e-3 > 0 ✓
PARAM_SPECS2: tuple[Spec, ...] = (
    Spec("X_b", "kg·m", False, 0.0, "m_b·P_bx（b 侧一阶矩 x）"),
    Spec("Y_b", "kg·m", False, -0.007, "m_b·P_by"),
    Spec("X_s", "kg·m", False, 0.00025, "m_s·P_sx（s 侧一阶矩 x）"),
    Spec("Y_s", "kg·m", False, -0.0078, "m_s·P_sy"),
    Spec("I_b", "kg·m²", True, 0.001641, "b 绕**质心**转动惯量（J_b = |X_b|²+I_b）"),
    Spec("I_s", "kg·m²", True, 0.0002654875, "s 绕**质心**转动惯量（J_s = |X_s|²/μ+I_s）"),
    Spec("mu", "kg", True, 0.2, "★ m_s（规范自由度: 只以 μ|D|²、X_b+μD_x 等组合进入动力学）"),
    Spec("f_bc", "N·m", True, 0.00065, "b 侧库仑摩擦"),
    Spec("f_bv", "N·m·s/rad", True, 0.0255, "b 侧粘滞摩擦"),
    Spec("f_sc", "N·m", True, 0.0042, "s 侧库仑摩擦"),
    Spec("f_sv", "N·m·s/rad", True, 0.24, "s 侧粘滞摩擦"),
)

PARAM_NAMES2 = tuple(s.name for s in PARAM_SPECS2)
NPARAM2 = len(PARAM_SPECS2)                       # 11
PARAM_INDEX2 = {s.name: i for i, s in enumerate(PARAM_SPECS2)}
POSITIVE2 = tuple(s.name for s in PARAM_SPECS2 if s.positive)
REAL2 = tuple(s.name for s in PARAM_SPECS2 if not s.positive)
FIXABLE = ("mu",)          # 建议固定/或与 J_b,X_b,Y_b 三选一固定（规范自由度）


def default_vector2() -> np.ndarray:
    return np.array([s.default for s in PARAM_SPECS2], dtype=np.float64)


# ============================================================================
# 参数容器（含已知量 dx/dy = D、λ）
# ============================================================================
@dataclass
class Planar2Params:
    # ── 已知几何（不辨识）──
    dx: float = 0.0            # D_x
    dy: float = 0.07           # D_y
    friction_lambda: float = FRICTION_LAMBDA
    # ── 11 个缩合参数（默认 = 参数表）──
    X_b: float = PARAM_SPECS2[0].default
    Y_b: float = PARAM_SPECS2[1].default
    X_s: float = PARAM_SPECS2[2].default
    Y_s: float = PARAM_SPECS2[3].default
    I_b: float = PARAM_SPECS2[4].default
    I_s: float = PARAM_SPECS2[5].default
    mu: float = PARAM_SPECS2[6].default
    f_bc: float = PARAM_SPECS2[7].default
    f_bv: float = PARAM_SPECS2[8].default
    f_sc: float = PARAM_SPECS2[9].default
    f_sv: float = PARAM_SPECS2[10].default

    # ── 便捷视图 ──
    @property
    def D2(self):
        return self.dx * self.dx + self.dy * self.dy

    @property
    def J_b(self):
        """J_b = I_b + m_b|P_b|²（m_b = 1）—— 派生量，不是自由参数。"""
        return self.I_b + self.X_b * self.X_b + self.Y_b * self.Y_b

    @property
    def J_s(self):
        """J_s = I_s + m_s|P_s|² = I_s + (X_s²+Y_s²)/μ —— 派生量。"""
        return self.I_s + (self.X_s * self.X_s + self.Y_s * self.Y_s) / self.mu

    @property
    def A(self):
        return self.J_b + self.mu * self.D2

    @property
    def B(self):
        return self.J_s

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES2], dtype=np.float64)

    def with_vector(self, phi) -> "Planar2Params":
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != (NPARAM2,):
            raise ValueError(f"参数向量长度应为 {NPARAM2}，得到 {phi.shape}")
        return replace(self, **{n: float(phi[i]) for i, n in enumerate(PARAM_NAMES2)})

    def Delta_min(self):
        """Δ 的下界 = J_s·J_b + μ|D|²·I_s（> 0 ⇒ 恒正定）。"""
        return self.J_s * self.J_b + self.mu * self.D2 * self.I_s


def params2_from_torch(phi_vec, base: Planar2Params) -> Planar2Params:
    """长度 11 的向量（np 或 torch，保持可导）→ Planar2Params。"""
    return replace(base, **{n: phi_vec[i] for i, n in enumerate(PARAM_NAMES2)})


# ============================================================================
# 派生量 / 右端（numpy；支持前导批维）
# ============================================================================
def derived_np(qs, p: Planar2Params):
    """返回 ``(A, B, muK, muKp, Delta)``；``qs`` = θ_s（标量或任意形状数组）。"""
    cs = np.cos(qs)
    sn = np.sin(qs)
    Qx = p.X_s * cs - p.Y_s * sn
    Qy = p.X_s * sn + p.Y_s * cs
    muK = p.dx * Qx + p.dy * Qy
    muKp = p.dy * Qx - p.dx * Qy
    A = p.A
    B = p.B
    Delta = A * B - muK * muK
    return A, B, muK, muKp, Delta


def gravity_np(psi_b, psi_s, gx, gy, p: Planar2Params):
    """``(G_b, G_s)``（世界系重力 (gx, gy)、绝对角 ψ_b/ψ_s）。"""
    Xb = p.X_b + p.mu * p.dx
    Yb = p.Y_b + p.mu * p.dy
    sb, cb = np.sin(psi_b), np.cos(psi_b)
    ss, cs = np.sin(psi_s), np.cos(psi_s)
    G_s = p.X_s * (gx * ss - gy * cs) + p.Y_s * (gx * cs + gy * ss)
    G_b = Xb * (gx * sb - gy * cb) + Yb * (gx * cb + gy * sb) + G_s
    return G_b, G_s


def accel_np(q, qd, u, gx, gy, psi_off, wc, p: Planar2Params):
    """``q̈``；``q``/``qd`` 形状 ``[...,2]``，``u`` = ``(T_b, T_s)``。

    ``psi_off`` = θ_c（逐样本或标量，rad），``wc`` = ω_c（逐样本或标量）。
    """
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    qb, qs = q[..., 0], q[..., 1]
    vb, vs = qd[..., 0], qd[..., 1]
    psi_b = psi_off + qb
    psi_s = psi_b + qs

    A, B, muK, muKp, Delta = derived_np(qs, p)
    G_b, G_s = gravity_np(psi_b, psi_s, gx, gy, p)
    lam = p.friction_lambda
    Q_b = u[..., 0] - p.f_bc * np.tanh(lam * vb) - p.f_bv * vb
    Q_s = u[..., 1] - p.f_sc * np.tanh(lam * vs) - p.f_sv * vs
    R_b = Q_b - G_b - muKp * vs * (2.0 * (wc + vb) + vs)
    R_s = Q_s - G_s + muKp * (wc + vb) ** 2
    BpK = B + muK
    qdd_b = (B * R_b - BpK * R_s) / Delta
    qdd_s = ((A + B + 2.0 * muK) * R_s - BpK * R_b) / Delta
    return np.stack([qdd_b, qdd_s], axis=-1)


def rk4_step_np(q, qd, u, gx, gy, psi_off, wc, p: Planar2Params, dt, substeps: int = 4):
    """RK4 单步（力矩零阶保持）。"""
    hh = dt / max(1, int(substeps))
    qa = np.array(q, dtype=np.float64)
    qda = np.array(qd, dtype=np.float64)
    for _ in range(max(1, int(substeps))):
        k1 = accel_np(qa, qda, u, gx, gy, psi_off, wc, p)
        k2 = accel_np(qa + 0.5 * hh * qda, qda + 0.5 * hh * k1, u, gx, gy, psi_off, wc, p)
        k3 = accel_np(qa + 0.5 * hh * (qda + 0.5 * hh * k1),
                      qda + 0.5 * hh * k2, u, gx, gy, psi_off, wc, p)
        k4 = accel_np(qa + hh * (qda + 0.5 * hh * k2), qda + hh * k3,
                      u, gx, gy, psi_off, wc, p)
        qa = qa + hh * qda + (hh * hh / 6.0) * (k1 + k2 + k3)
        qda = qda + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return qa, qda


def rollout_np(p: Planar2Params, q0, qd0, u_seq, gx_seq, gy_seq, psi_off_seq, wc_seq,
               dt, substeps: int = 4):
    """整段开环推演。

    ``q0/qd0`` ``[B,2]``；``u_seq`` ``[B,T,2]``；``gx_seq/gy_seq/psi_off_seq/wc_seq``
    ``[B,T]`` 或标量。返回 ``(theta [B,T,2], dtheta [B,T,2])``（第 i 行 = 积分前状态）。
    """
    u_seq = np.asarray(u_seq, dtype=np.float64)
    Bb, T = int(u_seq.shape[0]), int(u_seq.shape[1])
    th = np.empty((Bb, T, 2), dtype=np.float64)
    dth = np.empty((Bb, T, 2), dtype=np.float64)
    q = np.array(q0, dtype=np.float64)
    qd = np.array(qd0, dtype=np.float64)

    def _bc(x):
        a = np.asarray(x, dtype=np.float64)
        return np.broadcast_to(a, (Bb, T)) if a.ndim <= 1 else a

    gx, gy = _bc(gx_seq), _bc(gy_seq)
    po, wc = _bc(psi_off_seq), _bc(wc_seq)
    for t in range(T):
        th[:, t] = q
        dth[:, t] = qd
        q, qd = rk4_step_np(q, qd, u_seq[:, t], gx[:, t], gy[:, t], po[:, t], wc[:, t],
                            p, dt, substeps)
    return th, dth


# ============================================================================
# torch 版（可微）
# ============================================================================
def _t64(x):
    """张量原样返回；Python 标量 / numpy 数组一律转 **float64** 张量。

    ★ 必须显式指定 dtype: `torch.tensor(0.37)` 会推断成 **float32**（值变成
      0.370000004768…），经 `muKp·(ω_c+θ̇)²` 这类项放大后能带来 ~1e-11 的相对误差。
    """
    if torch is None:
        raise RuntimeError("需要 torch")
    if torch.is_tensor(x):
        # float32 也一起提上来: 本模块是 float64 设计，静默用 float32 会带来 ~1e-11 的误差
        return x if x.dtype == torch.float64 else x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def derived_torch(qs, p: Planar2Params):
    cs = torch.cos(qs)
    sn = torch.sin(qs)
    Qx = p.X_s * cs - p.Y_s * sn
    Qy = p.X_s * sn + p.Y_s * cs
    muK = p.dx * Qx + p.dy * Qy
    muKp = p.dy * Qx - p.dx * Qy
    A = p.J_b + p.mu * (p.dx * p.dx + p.dy * p.dy)
    B = p.J_s
    return A, B, muK, muKp, A * B - muK * muK


def gravity_torch(psi_b, psi_s, gx, gy, p: Planar2Params):
    Xb = p.X_b + p.mu * p.dx
    Yb = p.Y_b + p.mu * p.dy
    sb, cb = torch.sin(psi_b), torch.cos(psi_b)
    ss, cs = torch.sin(psi_s), torch.cos(psi_s)
    G_s = p.X_s * (gx * ss - gy * cs) + p.Y_s * (gx * cs + gy * ss)
    G_b = Xb * (gx * sb - gy * cb) + Yb * (gx * cb + gy * sb) + G_s
    return G_b, G_s


def accel_torch(q, qd, u, gx, gy, psi_off, wc, p: Planar2Params):
    # 标量参数一律 float64（见 _t64 的注释）；q/qd/u 由调用方保证 dtype
    gx, gy, psi_off, wc = _t64(gx), _t64(gy), _t64(psi_off), _t64(wc)
    qb, qs = q[..., 0], q[..., 1]
    vb, vs = qd[..., 0], qd[..., 1]
    psi_b = psi_off + qb
    psi_s = psi_b + qs
    A, B, muK, muKp, Delta = derived_torch(qs, p)
    G_b, G_s = gravity_torch(psi_b, psi_s, gx, gy, p)
    lam = p.friction_lambda
    Q_b = u[..., 0] - p.f_bc * torch.tanh(lam * vb) - p.f_bv * vb
    Q_s = u[..., 1] - p.f_sc * torch.tanh(lam * vs) - p.f_sv * vs
    R_b = Q_b - G_b - muKp * vs * (2.0 * (wc + vb) + vs)
    R_s = Q_s - G_s + muKp * (wc + vb) ** 2
    BpK = B + muK
    return torch.stack([(B * R_b - BpK * R_s) / Delta,
                        ((A + B + 2.0 * muK) * R_s - BpK * R_b) / Delta], dim=-1)


def rk4_step_torch(q, qd, u, gx, gy, psi_off, wc, p: Planar2Params, dt, substeps: int = 4):
    hh = dt / max(1, int(substeps))
    qa, qda = q, qd
    for _ in range(max(1, int(substeps))):
        k1 = accel_torch(qa, qda, u, gx, gy, psi_off, wc, p)
        k2 = accel_torch(qa + 0.5 * hh * qda, qda + 0.5 * hh * k1, u, gx, gy, psi_off, wc, p)
        k3 = accel_torch(qa + 0.5 * hh * (qda + 0.5 * hh * k1),
                         qda + 0.5 * hh * k2, u, gx, gy, psi_off, wc, p)
        k4 = accel_torch(qa + hh * (qda + 0.5 * hh * k2), qda + hh * k3,
                         u, gx, gy, psi_off, wc, p)
        qa = qa + hh * qda + (hh * hh / 6.0) * (k1 + k2 + k3)
        qda = qda + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return qa, qda


# ============================================================================
# 与世界系 / A 系重力的换算
# ============================================================================
def world_gravity_from_Aframe(g_ax, g_ay, psi_b):
    """A 系（随 b 转）重力 → 世界系重力: ``g_w = R(ψ_b)·g_A``。

    数据里记的是 A 系逐样本 ``gravity_ax/ay``；本模型要的是世界系常量 (g_x,g_y)。
    同一条段内算出来的 g_w 应当近似恒定（底盘静置）—— 可直接当一致性自检。
    """
    cb = np.cos(psi_b)
    sb = np.sin(psi_b)
    return cb * g_ax - sb * g_ay, sb * g_ax + cb * g_ay


def gravity_from_Aframe(psi_b, qs, g_ax, g_ay, p: Planar2Params):
    """直接用 **A 系** 重力 + 相对角算 ``(G_b, G_s)``（等价写法，省掉 θ_c）。

    代入 ``g_w = R(ψ_b)·g_A`` 后 ``ψ_b`` 项塌缩成常数、``ψ_s`` 项只留 θ_s:
        G_s = X_s(g_ax·sinθ_s − g_ay·cosθ_s) + Y_s(g_ax·cosθ_s + g_ay·sinθ_s)
        G_b = −(X_b+μD_x)·g_ay + (Y_b+μD_y)·g_ax + G_s
    """
    Xb = p.X_b + p.mu * p.dx
    Yb = p.Y_b + p.mu * p.dy
    ss, cs = np.sin(qs), np.cos(qs)
    G_s = p.X_s * (g_ax * ss - g_ay * cs) + p.Y_s * (g_ax * cs + g_ay * ss)
    G_b = -Xb * g_ay + Yb * g_ax + G_s
    return G_b, G_s


def describe2() -> str:
    return ("两刚体平面模型（缩合参数，D 已知）: " +
            ", ".join(f"{s.name}{'⁺' if s.positive else ''}" for s in PARAM_SPECS2) +
            f"；已知 D=({0.0},{0.07}), λ={FRICTION_LAMBDA:g}；共 {NPARAM2} 参")


__all__ = [
    "FRICTION_LAMBDA", "Spec", "PARAM_SPECS2", "PARAM_NAMES2", "NPARAM2", "PARAM_INDEX2",
    "POSITIVE2", "REAL2", "FIXABLE", "default_vector2", "Planar2Params", "params2_from_torch",
    "derived_np", "gravity_np", "accel_np", "rk4_step_np", "rollout_np",
    "derived_torch", "gravity_torch", "accel_torch", "rk4_step_torch",
    "world_gravity_from_Aframe", "gravity_from_Aframe", "describe2", "math", "_t64",
]
