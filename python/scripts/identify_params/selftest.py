#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型自检 —— 2-DOF 缩合参数模型（``python3 -m identify_params --selftest``）。

检查项:
  (1) 派生量与正定性: Δ ≥ J_s·J_b + μ|D|²·I_s > 0（任意参数 / 任意 θ_s）
  (2) 内嵌参考实现的**回归对拍**: 与旧 3-DOF 模型 "(云台, 小 yaw) 子块" 逐点一致
  (3) numpy RK4 ↔ torch 可微前向一致
  (4) torch 梯度: 11 个参数全部非零（不能有"漏掉的参数恒无梯度"）
  (5)(6) 两组映射往返 + 无限位（raw=±50 仍有限、正数仍 >0）
  (7) 批量契约: ``rollout_batched_np`` 与逐段 ``simulate_np`` 一致
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .model import (DifferentiableSimulator, accel_np, derived_np, rollout_batched_np,
                    simulate_np)
from .params import (NPARAM, POSITIVE_IDX, Exo, ParamGroups, PlanarParams,
                     default_param_vector)


# ============================================================================
# 内嵌参考实现: 旧 3-DOF 模型 (θ_motor, θ_platform, θ_small) 的 "云台+小 yaw" 子块
#   M q̈ = u − h,  u = (T_b, T_s)  （完整原实现见 git 历史 `model.py`）
# ============================================================================
def _old_subblock_accel(q, qd, u, g_ax, g_ay, wc, Jbig_eff, Js, P, Pb, d, m_u=0.0):
    qs = q[..., 1]
    Qx = P[0] * np.cos(qs) - P[1] * np.sin(qs)
    Qy = P[0] * np.sin(qs) + P[1] * np.cos(qs)
    dQ = d[0] * Qx + d[1] * Qy
    mu = 2.0 * (d[1] * Qx - d[0] * Qy)
    M11 = Jbig_eff + Js + 2.0 * dQ
    M12 = Js + dQ
    M22 = Js
    tb, ts = qd[..., 0], qd[..., 1]
    Gs = Qx * g_ay - Qy * g_ax
    Gb = (Pb[0] + m_u * d[0]) * g_ay - (Pb[1] + m_u * d[1]) * g_ax + Gs
    h0 = mu * tb * ts + 0.5 * mu * ts * ts - Gb + mu * ts * wc
    h1 = -0.5 * mu * tb * tb - Gs - mu * tb * wc - 0.5 * mu * wc * wc
    r0, r1 = u[..., 0] - h0, u[..., 1] - h1
    det = M11 * M22 - M12 * M12
    return np.stack([(M22 * r0 - M12 * r1) / det, (-M12 * r0 + M11 * r1) / det], axis=-1)


def _to_new(Jbig_eff, Js, P, Pb, d, mu):
    """旧 3-DOF 子块参数 → 新的 11 个缩合参数（给定 μ 这个规范代表）。"""
    Xb, Yb = Pb[0] - mu * d[0], Pb[1] - mu * d[1]
    Jb = Jbig_eff - mu * (d[0] ** 2 + d[1] ** 2)
    Xs, Ys = P
    return np.array([Xb, Yb, Xs, Ys,
                     Jb - (Xb * Xb + Yb * Yb),                    # I_b
                     Js - (Xs * Xs + Ys * Ys) / mu,               # I_s
                     mu, 0.0, 0.0, 0.0, 0.0])                     # μ + 摩擦(置 0)


def model_self_test(verbose: bool = True) -> bool:
    """跑全部一致性检查；返回 True/False（``--selftest`` 的退出码靠它）。"""
    ok = True
    rng = np.random.default_rng(0)
    D = np.array([0.0, 0.07])
    base = PlanarParams(dx=float(D[0]), dy=float(D[1]))

    def chk(name, val, tol):
        nonlocal ok
        good = bool(np.isfinite(val) and val < tol)
        ok = ok and good
        if verbose:
            print(f"  {name:<44} {val:10.3e}  (阈值 {tol:.0e})  {'OK' if good else 'FAIL'}")

    if verbose:
        print(f"[selftest] 2-DOF 缩合模型: {NPARAM} 参（{len(POSITIVE_IDX)} 正 / "
              f"{NPARAM - len(POSITIVE_IDX)} 实数），D 已知、无背隙/电机自由度")

    # ── (1) 正定性 ──
    worst, worst_margin = np.inf, np.inf
    for _ in range(4000):
        ph = np.concatenate([rng.normal(0, 0.05, 4), np.abs(rng.normal(size=7)) + 1e-9])
        p = base.with_vector(ph)
        dmin = float(derived_np(rng.uniform(-np.pi, np.pi, 16), p)[4].min())
        worst = min(worst, dmin)
        worst_margin = min(worst_margin, dmin - float(p.Delta_min()))
    if verbose:
        print(f"  {'(1) 4000 组随机参数 min Δ':<44} {worst:10.3e}  (>0 构造保证)  "
              f"{'OK' if worst > 0 else 'FAIL'}")
    ok = ok and worst > 0

    # ── (2) 与旧 3-DOF 子块参考实现对拍 ──
    Jbig, Js = 0.00267, 0.00057
    P, Pb = np.array([0.00025, -0.0078]), np.array([0.0, 0.007])
    n = 300
    q = np.stack([rng.normal(0, 1.0, n), rng.normal(0, 1.5, n)], -1)
    qd = rng.normal(0, 1.0, (n, 2))
    u = rng.normal(0, 0.3, (n, 2))
    g_ax, g_ay = rng.normal(0, 1.2, n), rng.normal(0, 1.2, n)
    wc = 0.37
    ref = _old_subblock_accel(q, qd, u, g_ax, g_ay, wc, Jbig, Js, P, Pb, D)
    e2 = 0.0
    for mu in (0.05, 0.2, 0.5):
        p = base.with_vector(_to_new(Jbig, Js, P, Pb, D, mu))
        e2 = max(e2, float(np.abs(accel_np(q, qd, u, g_ax, g_ay, wc, p) - ref).max()))
    chk("(2) 新模型 vs 旧 3-DOF 子块 max|Δq̈|", e2, 1e-11)

    # ── (3)(4)(5)(6) torch ──
    if torch is None:
        if verbose:
            print("  (3)-(6) 跳过: 环境无 torch")
    else:
        phi = default_param_vector()
        lay = ParamGroups(base)
        leaf = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
        Tn = 40
        sc = DifferentiableSimulator.pack_seq_const(
            torch.tensor(rng.normal(0, 0.3, (4, 2))), torch.tensor(rng.normal(0, 0.3, (4, 2))), 0.21)
        tau = torch.tensor(rng.normal(0, 0.2, (4, Tn, 2)))
        sv = DifferentiableSimulator.pack_seq_var(
            tau[..., 0], tau[..., 1], torch.tensor(rng.normal(0, 1.0, (4, Tn))),
            torch.tensor(rng.normal(0, 1.0, (4, Tn))))
        sim = DifferentiableSimulator(dt=0.01, substeps=4, layout=lay, base=base)
        th_t, dth_t = sim(leaf, sc, sv)
        th_n, dth_n = rollout_batched_np(base.with_vector(phi), sc.numpy(), sv.numpy(), 0.01, 4)
        chk("(3) torch vs numpy 整段前向 max|Δθ|",
            max(float(np.abs(th_t.detach().numpy() - th_n).max()),
                float(np.abs(dth_t.detach().numpy() - dth_n).max())), 1e-11)

        (th_t.sum() + dth_t.sum()).backward()
        g = leaf.grad.detach().numpy()
        nz = int(np.sum(np.abs(g) > 0.0))
        ok = ok and (nz == NPARAM)
        if verbose:
            print(f"  {'(4) 11 个参数梯度非零':<44} {nz:10d}/{NPARAM}  "
                  f"|∂|max={np.abs(g).max():.3e}  {'OK' if nz == NPARAM else 'FAIL'}")

        e5 = float(np.max(np.abs(lay.full_vector(
            lay.to_physical(lay.to_raw_init(phi))) - phi)))
        chk("(5) raw↔physical↔full_vector 往返 max|Δφ|", e5, 1e-12)
        raw_ext = np.array([-50.0 if not lay.is_positive(nm) else 50.0
                            for nm in lay.learnable_names])
        pe = lay.full_vector(lay.to_physical(raw_ext))
        good6 = bool(np.all(np.isfinite(pe)) and np.all(pe[list(POSITIVE_IDX)] > 0.0))
        ok = ok and good6
        if verbose:
            print(f"  {'(6) 无限位 (raw=±50): 有限且正数>0':<44} {'—':>10}  "
                  f"{'OK' if good6 else 'FAIL'}")

    # ── (7) 批量契约 == 逐段（逐位） ──
    p = base.with_vector(default_param_vector())
    ex = Exo(gravity_a=(0.7, -1.1), base_omega=0.13)
    q0s = rng.normal(0, 0.4, (3, 2))
    qd0s = rng.normal(0, 0.4, (3, 2))
    taus = rng.normal(0, 0.2, (3, 30, 2))
    sc = np.concatenate([q0s, qd0s, np.full((3, 1), 0.13)], axis=1)
    sv = np.zeros((3, 30, 4))
    sv[:, :, 0:2] = taus
    sv[:, :, 2], sv[:, :, 3] = 0.7, -1.1
    th_b, dth_b = rollout_batched_np(p, sc, sv, 0.01, 4)
    e7 = 0.0
    for b in range(3):
        th_s, dth_s = simulate_np(p, q0s[b], qd0s[b], taus[b], 0.01, ex, 4,
                                  exo_seq=[ex] * 30)
        e7 = max(e7, float(np.abs(th_b[b] - th_s).max()),
                 float(np.abs(dth_b[b] - dth_s).max()))
    chk("(7) 批量 vs 逐段 max|Δ|", e7, 1e-12)

    if verbose:
        print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if model_self_test() else 1)
