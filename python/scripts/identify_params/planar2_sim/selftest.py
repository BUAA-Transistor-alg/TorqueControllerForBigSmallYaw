#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``planar2_sim``（手写 C++ 2-DOF）精度自检 —— 与 numpy 参考逐点比对 + 并行/分块不变性。

跑法::

    cd python/scripts && python3 -m identify_params.planar2_sim.selftest
    # 仓库根目录（identify_params 包不在根目录，需要 PYTHONPATH）:
    PYTHONPATH=python/scripts python3 -m identify_params.planar2_sim.selftest
    # 也可以直接当脚本跑（内部自己 sys.path 找 python/scripts）:
    python3 python/scripts/identify_params/planar2_sim/selftest.py

判据:
  · 每个场景与 **A 系 numpy 参考** 的最大绝对误差 < 1e-9（实测 ~1e-16）；
  · 不同 ``nthreads``(1,2,0) × ``block``(1,7,64,256) 必须给出**逐位一致**的结果；
  · 用 :func:`~identify_params.planar2.accel_np` 对随机状态逐点对拍加速度，误差 < 1e-12。

★★★★ 与任务说明的一处矛盾（以实现 planar2.py 为准）★★★★
  任务说明称 "``rollout_np`` 有 A 系重力路径，``psi_off`` 直接传 0 即可"。**这不成立**:
  ``rollout_np`` / ``accel_np`` 只有 ``gravity_np``（**世界系**重力 + ψ_b）一个入口，
  传 ``psi_off=0`` 时 ``G_b`` 仍含 ``q_b`` 项，与 :func:`~identify_params.planar2.gravity_from_Aframe`
  的 A 系写法并不相等。两者只在逐点 ``g_w = R(ψ_b)·g_A`` 时严格等价（planar2.py 文件头已证），
  而 ``rollout_np`` 在一个外层步内把 **世界系** 重力当常量、``q_b`` 却在步内演化，
  A 系内核则把 ``g_A`` 当常量（``G_A`` 与 ``q_b`` 无关）—— 两者相差一个 O(dt) 的"帧采样"项
  （基础场景实测 Δθ ≈ 4.6e-5，随 dt 收缩，**不是**实现误差）。

  所以本自检的"逐场景精度"用一个**精确的 A 系 numpy 参考**（:func:`_rollout_Aframe_np`）:
  每个 RK4 子步都用 ``planar2.accel_np``，只把输入的 ``g_A`` 按当前子步的
  ``ψ_b = θ_c + q_b`` 换成世界系（换算后 ``gravity_np`` 与 A 系写法恒等，与 ``q_b`` 无关）。
  另外单独给出 ``rollout_np`` 能精确对拍的情形（重力=0）与一条 O(dt) 诊断。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

if __package__ in (None, ""):        # 直接 python3 .../selftest.py（仓库根目录也能跑）
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _SCRIPTS = os.path.dirname(os.path.dirname(_HERE))     # python/scripts
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    from identify_params.planar2 import (                  # noqa: E402
        PARAM_INDEX2,
        Planar2Params,
        accel_np,
        default_vector2,
        rollout_np,
    )
    from identify_params.planar2_sim.wrapper import (      # noqa: E402
        DEFAULT_BLOCK,
        FastSimulator,
        accel,
        build_info,
    )
else:                                # python3 -m identify_params.planar2_sim.selftest
    from ..planar2 import (                                # noqa: E402
        PARAM_INDEX2,
        Planar2Params,
        accel_np,
        default_vector2,
        rollout_np,
    )
    from .wrapper import (                                 # noqa: E402
        DEFAULT_BLOCK,
        FastSimulator,
        accel,
        build_info,
    )

TOL_ABS = 1e-9          # 与 numpy 参考的最大绝对误差阈值（实测 ~1e-16）
TOL_ACCEL = 1e-12       # 逐点加速度对拍阈值
TOL_INVARIANT = 0.0     # 线程/分块之间要求逐位一致


# ============================================================================
# 场景表
# ============================================================================
#   name, B, T, dt, substeps, gravity_amp(A 系), wc, theta_c, seed, overrides
SCENARIOS = [
    ("基础",            8,  51, 0.01,    4, 0.30, 0.0,  0.0,  0, None),
    ("重力=0",          8,  51, 0.01,    4, 0.00, 0.0,  0.0,  1, None),
    ("ω_c 非零",        8,  51, 0.01,    4, 0.30, 0.42, 0.37, 2, None),
    ("小 dt / 多子步",   4,  61, 0.002,   8, 0.30, 0.0,  0.0,  3, None),
    ("单条 B=1",        1,  51, 0.01,    4, 0.30, 0.0,  0.0,  4, None),
    ("分块边界 B=33",    33, 41, 0.01,    4, 0.30, 0.0,  0.0,  5, None),
    ("多块 B=513",      513, 11, 0.01,    4, 0.30, 0.0,  0.0,  8, None),
    ("极小子惯量",       8,  51, 0.001,   4, 0.30, 0.0,  0.0,  6, {"I_b": 1e-7, "I_s": 1e-7}),
    ("大摩擦",          8,  51, 1e-4,    4, 0.30, 0.0,  0.0,  7,
     {"f_bc": 8.0, "f_bv": 0.1, "f_sc": 8.0, "f_sv": 0.1}),
]


def _phi(overrides: dict | None = None) -> np.ndarray:
    """默认 11 参向量 + 按名字覆盖。"""
    phi = default_vector2().copy()
    for k, v in (overrides or {}).items():
        phi[PARAM_INDEX2[k]] = float(v)
    return phi


def _accel_Aframe_np(q, qd, u, ga_x, ga_y, wc, theta_c, p):
    """A 系重力的 ``q̈``：把 ``g_A`` 按 ``g_w = R(ψ_b)·g_A``（ψ_b = θ_c + q_b）换算后调
    :func:`~identify_params.planar2.accel_np`。

    换算后 ``accel_np`` 内部的 ``gravity_np`` 与 ``gravity_from_Aframe`` **逐点恒等**
    （结果与 q_b 无关），所以这就是 A 系模型的权威 numpy 右端。
    """
    psi_b = theta_c + q[:, 0]
    cb, sb = np.cos(psi_b), np.sin(psi_b)
    gwx = cb * ga_x - sb * ga_y
    gwy = sb * ga_x + cb * ga_y
    return accel_np(q, qd, u, gwx, gwy, theta_c, wc, p)


def _rollout_Aframe_np(p, q0, qd0, u_seq, ga_seq, wc, theta_c, dt, substeps):
    """**A 系重力路径**的 numpy 参考（planar2 没有直接入口，用 ``accel_np`` +
    ``rk4_step_np`` 的更新顺序组装；每步都调 plan2 的 ``accel_np``）。"""
    B, T = int(u_seq.shape[0]), int(u_seq.shape[1])
    th = np.empty((B, T, 2), dtype=np.float64)
    dth = np.empty((B, T, 2), dtype=np.float64)
    q = np.array(q0, dtype=np.float64)
    qd = np.array(qd0, dtype=np.float64)
    wc_arr = np.full(B, float(wc))
    hh = dt / max(1, int(substeps))
    for t in range(T):
        th[:, t] = q
        dth[:, t] = qd
        u = u_seq[:, t]
        gax = ga_seq[:, t, 0]
        gay = ga_seq[:, t, 1]
        for _ in range(max(1, int(substeps))):
            k1 = _accel_Aframe_np(q, qd, u, gax, gay, wc_arr, theta_c, p)
            k2 = _accel_Aframe_np(q + 0.5 * hh * qd, qd + 0.5 * hh * k1,
                                  u, gax, gay, wc_arr, theta_c, p)
            k3 = _accel_Aframe_np(q + 0.5 * hh * (qd + 0.5 * hh * k1), qd + 0.5 * hh * k2,
                                  u, gax, gay, wc_arr, theta_c, p)
            k4 = _accel_Aframe_np(q + hh * (qd + 0.5 * hh * k2), qd + hh * k3,
                                  u, gax, gay, wc_arr, theta_c, p)
            q = q + hh * qd + (hh * hh / 6.0) * (k1 + k2 + k3)
            qd = qd + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return th, dth


def _make_case(B, T, dt, substeps, gamp, wc, theta_c, seed, phi):
    """造一组输入 + 精确 A 系 numpy 参考轨迹。

    ``seq_var`` 的 g_ax/g_ay 用**逐样本随机**的 A 系重力（时间变化，顺便验证 gather/时间索引）。
    """
    rng = np.random.default_rng(seed)
    p = Planar2Params().with_vector(phi)
    q0 = rng.normal(size=(B, 2)) * 0.25
    qd0 = rng.normal(size=(B, 2)) * 0.40
    u = rng.normal(size=(B, T, 2)) * 0.25
    ga = rng.normal(size=(B, T, 2)) * float(gamp)

    th_ref, dth_ref = _rollout_Aframe_np(p, q0, qd0, u, ga, wc, theta_c, dt, substeps)

    seq_var = np.stack([u[..., 0], u[..., 1], ga[..., 0], ga[..., 1]], axis=-1)   # [B,T,4]
    seq_const = np.zeros((B, 5), dtype=np.float64)
    seq_const[:, 0:2] = q0
    seq_const[:, 2:4] = qd0
    seq_const[:, 4] = float(wc)
    params_Bx11 = np.repeat(phi[None, :], B, axis=0)
    return p, params_Bx11, seq_const, seq_var, th_ref, dth_ref, (q0, qd0, u, ga)


# ============================================================================
# 各项检查
# ============================================================================
def _invariance_ok(p, params_Bx11, sc, sv, dt, sub, ref):
    """线程数 (1,2,0) × 分块 (1,7,64,256) 必须与 ref **逐位一致**。"""
    th0, dth0 = ref
    for nt in (1, 2, 0):
        for blk in (1, 7, 64, 256):
            sim = FastSimulator(p, dt=dt, substeps=sub, nthreads=nt, block=blk)
            th, dth = sim.rollout(params_Bx11, sc, sv)
            if not (np.array_equal(th, th0) and np.array_equal(dth, dth0)):
                return False, (nt, blk, float(np.abs(th - th0).max()),
                               float(np.abs(dth - dth0).max()))
    return True, None


def _accel_pointwise_check(seed: int = 11, n: int = 512) -> float:
    """随机 (q, q̇, τ, g_A, ω_c) 逐点对拍 ``planar2.accel_np``（最直接的公式校验）。"""
    rng = np.random.default_rng(seed)
    phi = default_vector2()
    p = Planar2Params().with_vector(phi)
    theta_c = 0.37
    q = np.stack([rng.normal(size=n) * 1.0, rng.normal(size=n) * 2.0], axis=1)
    qd = rng.normal(size=(n, 2)) * 1.5
    u = rng.normal(size=(n, 2)) * 0.5
    ga = rng.normal(size=(n, 2)) * 0.4
    wc = rng.normal(size=n) * 0.7

    ref = _accel_Aframe_np(q, qd, u, ga[:, 0], ga[:, 1], wc, theta_c, p)   # [n,2]
    state = np.concatenate([q, qd], axis=1)
    uu = np.concatenate([u, ga], axis=1)
    params_Bx11 = np.repeat(phi[None, :], n, axis=0)
    got = accel(params_Bx11, state, uu, wc)
    return float(np.abs(got - ref).max())


def _rollout_np_zero_gravity_check(B=8, T=51, dt=0.01, sub=4, seed=21) -> float:
    """``rollout_np`` 能**精确**对拍的情形: 世界系重力 ≡ 0 ⇔ A 系重力 ≡ 0。"""
    rng = np.random.default_rng(seed)
    phi = default_vector2()
    p = Planar2Params().with_vector(phi)
    q0 = rng.normal(size=(B, 2)) * 0.25
    qd0 = rng.normal(size=(B, 2)) * 0.40
    u = rng.normal(size=(B, T, 2)) * 0.25
    zeros = np.zeros((B, T))
    th_r, dth_r = rollout_np(p, q0, qd0, u, zeros, zeros, 0.0, 0.0, dt, sub)

    sc = np.zeros((B, 5))
    sc[:, 0:2] = q0
    sc[:, 2:4] = qd0
    sv = np.concatenate([u, np.zeros((B, T, 2))], axis=-1)
    params_Bx11 = np.repeat(phi[None, :], B, axis=0)
    th_c, dth_c = FastSimulator(p, dt=dt, substeps=sub, nthreads=1,
                                block=DEFAULT_BLOCK).rollout(params_Bx11, sc, sv)
    return max(float(np.abs(th_c - th_r).max()), float(np.abs(dth_c - dth_r).max()))


def _rollout_np_world_diag(B=8, T=51, dt=0.01, sub=4, seed=0) -> tuple[float, float]:
    """**诊断（非判据）**: 任务说明建议的"两趟"对拍 —— 常量世界系重力 → rollout_np 参考，
    再用参考轨迹把 g_w 反算成 A 系 g_A 喂 C++。二者相差的是 O(dt) 帧采样项。"""
    rng = np.random.default_rng(seed)
    phi = default_vector2()
    p = Planar2Params().with_vector(phi)
    q0 = rng.normal(size=(B, 2)) * 0.25
    qd0 = rng.normal(size=(B, 2)) * 0.40
    u = rng.normal(size=(B, T, 2)) * 0.25
    gw = rng.normal(size=(B, 2)) * 0.30
    gx = np.repeat(gw[:, 0:1], T, axis=1)
    gy = np.repeat(gw[:, 1:2], T, axis=1)
    th_r, dth_r = rollout_np(p, q0, qd0, u, gx, gy, 0.0, 0.0, dt, sub)

    psi_b = th_r[..., 0]
    cb, sb = np.cos(psi_b), np.sin(psi_b)
    gax = cb * gw[:, 0:1] + sb * gw[:, 1:2]
    gay = -sb * gw[:, 0:1] + cb * gw[:, 1:2]
    sc = np.zeros((B, 5))
    sc[:, 0:2] = q0
    sc[:, 2:4] = qd0
    sv = np.stack([u[..., 0], u[..., 1], gax, gay], axis=-1)
    params_Bx11 = np.repeat(phi[None, :], B, axis=0)
    th_c, dth_c = FastSimulator(p, dt=dt, substeps=sub, nthreads=1,
                                block=DEFAULT_BLOCK).rollout(params_Bx11, sc, sv)
    return float(np.abs(th_c - th_r).max()), float(np.abs(dth_c - dth_r).max())


# ============================================================================
# 主自检
# ============================================================================
def planar2_self_test(verbose: bool = True) -> bool:
    ok = True
    if verbose:
        print(f"[planar2_sim selftest] {build_info()}")
        print("[planar2_sim selftest] 参考 = 精确 A 系 numpy 驱动（每子步调 planar2.accel_np）")
        print(f"[planar2_sim selftest] 阈值: |cpp−numpy|max < {TOL_ABS:g}；"
              f"accel 对拍 < {TOL_ACCEL:g}；线程/分块差 = {TOL_INVARIANT:g}（逐位）")
        print(f"  {'场景':<16} {'B×T':>8} {'|Δθ|max':>10} {'|Δθ̇|max':>10}  "
              f"{'线程/分块不变':>12}  判定")

    for (name, B, T, dt, sub, gamp, wc, tc, seed, ov) in SCENARIOS:
        phi = _phi(ov)
        p, params_Bx11, sc, sv, th_ref, dth_ref, _ = _make_case(
            B, T, dt, sub, gamp, wc, tc, seed, phi)
        sim = FastSimulator(p, dt=dt, substeps=sub, nthreads=1, block=DEFAULT_BLOCK)
        th_c, dth_c = sim.rollout(params_Bx11, sc, sv)
        e_th = float(np.abs(th_c - th_ref).max())
        e_dt = float(np.abs(dth_c - dth_ref).max())
        inv, worst = _invariance_ok(p, params_Bx11, sc, sv, dt, sub, (th_c, dth_c))
        good = (e_th < TOL_ABS and e_dt < TOL_ABS and inv)
        ok = ok and good
        if verbose:
            print(f"  {name:<16} {f'{B}×{T}':>8} {e_th:10.2e} {e_dt:10.2e}  "
                  f"{'✓' if inv else '✗':>12}  {'PASS' if good else 'FAIL'}")
            if not inv and worst is not None:
                print(f"      线程/分块不一致: nthreads={worst[0]} block={worst[1]} "
                      f"|Δθ|={worst[2]:.3e} |Δθ̇|={worst[3]:.3e}")

    # ── vs rollout_np（唯一能精确对拍的情形: 重力=0）──
    e0 = _rollout_np_zero_gravity_check()
    good = e0 < TOL_ABS
    ok = ok and good
    if verbose:
        print(f"  {'vs rollout_np(g=0)':<16} {'8×51':>8} {e0:10.2e} {'':>10}  "
              f"{'—':>12}  {'PASS' if good else 'FAIL'}")

    # ── 逐点 accel 对拍（公式校验，最直接）──
    e_acc = _accel_pointwise_check(seed=11)
    good = e_acc < TOL_ACCEL
    ok = ok and good
    if verbose:
        print(f"  {'accel 逐点对拍':<16} {'512':>8} {e_acc:10.2e} {'':>10}  "
              f"{'—':>12}  {'PASS' if good else 'FAIL'}")

    if verbose:
        d_th, d_dt = _rollout_np_world_diag()
        print(f"  [诊断] 基础 vs rollout_np(常量世界系重力, 两趟): "
              f"Δθ={d_th:.2e} Δθ̇={d_dt:.2e}")
        print("         ↑ O(dt) 帧采样差（世界系重力在步内随 q_b 变 / A 系内核 g_A 步内恒定），"
              "非实现误差；见文件头说明")
        _bench()
        print("[planar2_sim selftest]", "PASS" if ok else "FAIL")
    return ok


# ============================================================================
# 性能
# ============================================================================
def _bench(B: int = 256, T: int = 101, sub: int = 4, reps: int = 5) -> None:
    rng = np.random.default_rng(123)
    phi = default_vector2()
    p = Planar2Params().with_vector(phi)
    params = np.repeat(phi[None, :], B, axis=0)
    q0 = rng.normal(size=(B, 2)) * 0.2
    qd0 = rng.normal(size=(B, 2)) * 0.4
    u = rng.normal(size=(B, T, 2)) * 0.3
    ga = rng.normal(size=(B, T, 2)) * 0.2
    sc = np.zeros((B, 5))
    sc[:, 0:2] = q0
    sc[:, 2:4] = qd0
    sv = np.concatenate([u, ga], axis=-1)
    print(f"[planar2_sim selftest] 性能: B={B}, T={T}, substeps={sub}, "
          f"RK4 子步/item={T*sub}")
    for nt in (1, 0):
        sim = FastSimulator(p, dt=0.01, substeps=sub, nthreads=nt, block=DEFAULT_BLOCK)
        sim.rollout(params, sc, sv)                     # warmup
        t0 = time.perf_counter()
        for _ in range(reps):
            sim.rollout(params, sc, sv)
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"    nthreads={nt:<2d}  {ms:8.3f} ms/次  "
              f"({B*T*sub / (ms*1e-3) / 1e6:7.1f} M RK4-子步/s)")


if __name__ == "__main__":
    sys.exit(0 if planar2_self_test() else 1)
