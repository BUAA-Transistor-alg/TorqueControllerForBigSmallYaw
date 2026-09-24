#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``fast_sim``（手写 C++）精度自检 —— 与 numpy 参考实现逐点比对 + 并行/分块不变性。

跑法::

    cd python/scripts && python3 -m identify_params.fast_sim.selftest

判据:
  · 每个场景与 :func:`~identify_params.model.simulate_backlash_np`（逐段精确参考）的
    最大绝对误差应 ~1e-14（唯一残差来源是 tanh 实现差异）；阈值 1e-9 已足够宽松；
  · 不同 ``nthreads`` / ``block`` 必须给出**逐位一致**的结果（并行只按 batch 切分）；
  · 损失（CMA-ES 真正优化的量，经 :class:`~identify_params.loss.WindowObjective`）
    在 cpp 后端与 numpy 后端下相差 < 1e-10（相对）。
"""

from __future__ import annotations

import numpy as np

from ..model import DifferentiableSimulator, simulate_backlash_np
from ..params import EXO_ZERO, Exo, ParamGroups, PlanarParams, default_param_vector
from .wrapper import FastSimulator, build_info

TOL_ABS = 1e-9          # 与 numpy 参考的最大绝对误差阈值（实测 ~1e-14）
TOL_INVARIANT = 0.0     # 线程/分块之间要求逐位一致


def _params(overrides: dict | None = None) -> np.ndarray:
    phi = default_param_vector().copy()
    for k, v in (overrides or {}).items():
        phi[k] = v
    return phi


def _case(B, T, phi, gravity=True, beta=True, base_omega=0.0, base_alpha=0.0, seed=0):
    """造一组输入（与可微模型同一张量契约）。"""
    rng = np.random.default_rng(seed)
    base = PlanarParams(dx=0.037, dy=-0.011, m_u_known=0.25,
                        friction_lambda=100.0).with_vector(phi)
    layout = ParamGroups(base)
    sc = np.zeros((B, 8))
    sc[:, 0:3] = rng.normal(size=(B, 3)) * 0.25
    sc[:, 3:6] = rng.normal(size=(B, 3)) * 0.40
    sc[:, 6] = base_omega
    sc[:, 7] = base_alpha
    sv = np.zeros((B, T, 5))
    sv[..., 0] = rng.normal(size=(B, T)) * 0.25          # τ_big
    sv[..., 1] = rng.normal(size=(B, T)) * 0.25          # τ_small
    sv[..., 2] = rng.normal(size=(B, T)) * 0.2 if gravity else 0.0
    sv[..., 3] = rng.normal(size=(B, T)) * 0.2 if gravity else 0.0
    sv[..., 4] = rng.normal(size=(B, T)) * 0.02 if beta else 0.0
    return base, layout, sc, sv


def _numpy_ref(base, sc, sv, dt, substeps):
    """逐段调用 simulate_backlash_np（评测用的那个参考实现）。"""
    B, T = sv.shape[0], sv.shape[1]
    th = np.empty((B, T, 3))
    dth = np.empty((B, T, 3))
    for b in range(B):
        exo_seq = [Exo(gravity_a=(float(sv[b, t, 2]), float(sv[b, t, 3])),
                       base_omega=float(sc[b, 6]), base_alpha=float(sc[b, 7]))
                   for t in range(T)]
        th[b], dth[b] = simulate_backlash_np(
            base, sc[b, 0:3], sc[b, 3:6], sv[b, :, 0:2], dt, EXO_ZERO, substeps,
            exo_seq=exo_seq, beta_seq=sv[b, :, 4], integrator="rk4")
    return th, dth


def _cpp(base, sc, sv, phi, dt, substeps, nthreads, block):
    sim = FastSimulator(base, dt=dt, substeps=substeps, nthreads=nthreads, block=block)
    p18 = np.repeat(base.vector()[None, :], sv.shape[0], axis=0)
    return sim.rollout(p18, sc, sv)


SCENARIOS = [
    # name,           B,  T,   dt,    substeps, params 覆盖,                gravity, beta, wc,   ac
    ("基础 rk4",       8,  51,  0.01,  4, None,                              True,   True,  0.0,  0.0),
    ("水平(重力=0)",    8,  51,  0.01,  4, None,                              False,  True,  0.0,  0.0),
    ("β≡0",            8,  51,  0.01,  4, None,                              True,   False, 0.0,  0.0),
    ("底盘 ω/α 非零",   8,  51,  0.01,  4, None,                              True,   True,  0.42, -0.9),
    ("死区内小扰动",     8,  81,  0.01,  4, {8: 0.6, 9: 800.0, 10: 5.0},       True,   True,  0.0,  0.0),
    ("极硬接触 k=5000",  8,  51,  0.01,  4, {9: 5000.0, 8: 0.004},            True,   True,  0.0,  0.0),
    ("substeps=1 长轨迹", 4, 301, 0.01,  1, None,                             True,   True,  0.0,  0.0),
    ("小 dt / 多子步",   4,  61,  0.002, 8, None,                             True,   True,  0.0,  0.0),
    ("单条 B=1",        1,  51,  0.01,  4, None,                              True,   True,  0.0,  0.0),
    ("分块边界 B=33",   33,  41,  0.01,  4, None,                             True,   True,  0.0,  0.0),
]


def fast_sim_self_test(verbose: bool = True) -> bool:
    """跑全部精度场景；返回 True/False。"""
    ok = True
    if verbose:
        print(f"[fast_sim selftest] {build_info()}")
        print(f"[fast_sim selftest] 阈值: |cpp−numpy|max < {TOL_ABS:g}；"
              f"线程/分块差 = {TOL_INVARIANT:g}（逐位）")
        print(f"  {'场景':<18} {'B×T':>9} {'|θ|max':>10} {'|θ̇|max':>10}  "
              f"{'线程/分块不变':>12}  判定")
    for (name, B, T, dt, sub, ov, grav, beta, wc, ac) in SCENARIOS:
        phi = _params(ov)
        base, layout, sc, sv = _case(B, T, phi, grav, beta, wc, ac)
        th_r, dth_r = _numpy_ref(base, sc, sv, dt, sub)
        th_c, dth_c = _cpp(base, sc, sv, phi, dt, sub, nthreads=1, block=32)
        e_th = float(np.abs(th_c - th_r).max())
        e_dt = float(np.abs(dth_c - dth_r).max())
        # 线程/分块不变性（逐位）
        inv = True
        for nt in (2, 0):
            for blk in (1, 7, 64, 256):
                th2, dth2 = _cpp(base, sc, sv, phi, dt, sub, nthreads=nt, block=blk)
                if not (np.array_equal(th2, th_c) and np.array_equal(dth2, dth_c)):
                    inv = False
        good = (e_th < TOL_ABS and e_dt < TOL_ABS and inv)
        ok = ok and good
        if verbose:
            print(f"  {name:<18} {f'{B}×{T}':>9} {e_th:10.2e} {e_dt:10.2e}  "
                  f"{'✓' if inv else '✗':>12}  {'PASS' if good else 'FAIL'}")
    # ── 与 torch 模型对比（若可用）──
    try:
        import torch  # noqa: F401
        from ..loss import rollout_states
        phi = _params()
        base, layout, sc, sv = _case(6, 61, phi, True, True)
        d = DifferentiableSimulator(dt=0.01, substeps=4, integrator="rk4", layout=layout,
                                    base=base, beta_from_input=True)
        fs = FastSimulator(base, dt=0.01, substeps=4, nthreads=1, block=32)
        p18 = np.repeat(base.vector()[None, :], 6, axis=0)
        th_c, dth_c = fs.rollout(p18, sc, sv)
        with torch.no_grad():
            sc_t = torch.tensor(sc)
            sv_t = torch.tensor(sv)
            params_t = layout.to_physical(
                torch.tensor(layout.to_raw_init(phi))).unsqueeze(0).expand(6, -1)
            th_t, dth_t = rollout_states(d, params_t, sc_t, sv_t)
        e_t = float(np.abs(th_c - th_t.numpy()).max())
        good = e_t < 1e-9
        ok = ok and good
        if verbose:
            print(f"  {'vs torch 模型':<18} {f'{6}×{61}':>9} {e_t:10.2e} {'—':>12}  "
                  f"{'PASS' if good else 'FAIL'}")
        # ── 损失一致性（cpp vs numpy 后端，CMA-ES 真正优化的量）──
        from ..data import Segment
        from ..loss import WindowObjective as WO
        from ..model import rollout_backlash_batched_np
        from ..train import FitConfig, build_fit_context
        rng = np.random.default_rng(3)
        Tn = 120
        seg = Segment(t=np.arange(Tn) * 0.01,
                      theta=np.cumsum(rng.normal(size=(Tn, 3)) * 0.002, axis=0),
                      dtheta=rng.normal(size=(Tn, 3)) * 0.05,
                      tau=rng.normal(size=(Tn, 2)) * 0.2,
                      beta=rng.normal(size=Tn) * 0.01, source="synthetic")
        cfg = FitConfig(window_len=60, windows_per_seg=2)
        ctx = build_fit_context([seg], cfg, base, honor_windows=True)
        seq_const = DifferentiableSimulator.pack_seq_const(
            ctx.q0_all, ctx.qd0_all, 0.0, 0.0)
        sc_c = np.ascontiguousarray(seq_const.numpy())
        sv_c = np.ascontiguousarray(ctx.seq_var.numpy())
        x = ctx.layout.to_raw_init(phi)
        p18 = np.empty((ctx.W, 18), dtype=np.float64)
        kw = dict(loss_mode="mse")

        def _full(raw):
            phys = ctx.layout.to_physical(torch.tensor(np.asarray(raw)))
            return ctx.layout.full_vector(np.asarray(phys.detach().cpu().numpy()))

        def f_np(raw):
            return rollout_backlash_batched_np(base.with_vector(_full(raw)), sc_c, sv_c,
                                               ctx.dt, 4)

        def f_cpp(raw):
            p18[:] = _full(raw)
            return fs.rollout(p18, sc_c, sv_c)

        o_np = WO(f_np, ctx.th_t, ctx.dth_t, mask=ctx.mask_t, ax_w=ctx.w_axis_all, loss_kw=kw)
        o_cp = WO(f_cpp, ctx.th_t, ctx.dth_t, mask=ctx.mask_t, ax_w=ctx.w_axis_all, loss_kw=kw)
        v_np, v_cp = o_np(x), o_cp(x)
        rel = abs(v_cp - v_np) / max(1e-30, abs(v_np))
        good = rel < 1e-10
        ok = ok and good
        if verbose:
            print(f"  {'损失 cpp vs numpy':<18} {'—':>9} rel={rel:8.2e} {'—':>12}  "
                  f"{'PASS' if good else 'FAIL'}")
    except ImportError:  # pragma: no cover
        if verbose:
            print("  （缺 torch，跳过 torch / 损失一致性检查）")
    if verbose:
        print("[fast_sim selftest]", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if fast_sim_self_test() else 1)
