#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型一致性自检（回归矩阵 / 参数分组映射 / torch 与 numpy / 梯度 / 背隙可观测性）。

检查清单:
  (1) Y·φ == ID(φ)；(2) Y == ∂ID/∂φ（数值偏导）；
  (5) 重力通道确实生效；(6) 逐样本重力一致性；
  (8)(9) P 沿 d 约束（``ParamGroups(p_along_d=...)``）；
  (10)(11) 两组参数映射往返 + log 参数化**没有界**；
  (13) 3-DOF 装配与 2-DOF 子块严格一致；(14) 正/逆动力学往返；
  (15) 可微仿真模型 vs numpy 参考（rk4 / euler 两种积分器）；
  (16) 全部可学习参数（18 个）的梯度有限且非零；(17) β 确实影响轨迹。

★ 旧脚本里"2-DOF 的 torch 版"（``torch_forward_accel`` / ``torch_rollout``）已被
:class:`~identify_params.model.DifferentiableSimulator` 取代（3-DOF 模块就是唯一模型），
所以对应的 2-DOF torch vs numpy 两条检查并入 (15) 的 3-DOF 对照。
"""

from __future__ import annotations

import math

import numpy as np

from .model import (
    DifferentiableSimulator,
    backlash_torque_np,
    forward_accel_backlash_np,
    inverse_dynamics_backlash_np,
    inverse_dynamics_np,
    motor_friction_np,
    regressor_np,
    simulate_backlash_np,
)
from .params import (
    NPARAM,
    P_ALONG_D_NAME,
    POSITIVE_IDX,
    PlanarParams,
    ParamGroups,
    Exo,
    EXO_ZERO,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def model_self_test(verbose: bool = True) -> bool:
    """跑全部一致性检查；返回 True/False（``--selftest`` 的退出码靠它）。"""
    rng = np.random.default_rng(0)
    p = PlanarParams(dx=0.037, dy=-0.011, gravity=9.81, m_u_known=0.25,
                     friction_lambda=10.0).with_vector(
        np.array([0.0243, 0.0132, 0.0042, -0.0018, 0.092, 0.031, 0.028, 0.0085,
                  0.0701, 320.0, 1.7, 0.0031, 0.0075, 0.041, 0.012, 0.0042,
                  0.0061, -0.0034]))      # ← 18 参（末两个 = Pbx/Pby）
    exo = Exo(gravity_a=(0.31, -0.17), base_omega=0.42, base_alpha=-0.9)
    q = rng.normal(size=(5, 2)) * 0.6
    qd = rng.normal(size=(5, 2)) * 2.0
    qdd = rng.normal(size=(5, 2)) * 5.0

    # ── (1) 回归矩阵 Y·φ + ID(φ=0) == ID(φ) ──
    Y = regressor_np(q, qd, qdd, p, exo)
    tau_id = inverse_dynamics_np(q, qd, qdd, p, exo)
    tau_affine = inverse_dynamics_np(q, qd, qdd, p.with_vector(np.zeros(NPARAM)), exo)
    tau_lin = np.einsum("...ij,j->...i", Y, p.vector()) + tau_affine
    e1 = float(np.max(np.abs(tau_lin - tau_id)))
    # ── (2) Y == ∂ID/∂φ（中心差分）──
    Ynum = np.zeros_like(Y)
    for j in range(NPARAM):
        for sgn in (+1, -1):
            v = p.vector()
            h = 1e-6 * max(1.0, abs(v[j]))
            v[j] += sgn * h
            Ynum[..., :, j] += sgn * inverse_dynamics_np(q, qd, qdd, p.with_vector(v), exo) / (2 * h)
    e2 = float(np.max(np.abs(Ynum - Y)))
    # ── (5) 重力通道确实生效 ──
    tau_tilt = inverse_dynamics_np(q, qd, qdd, p, exo)
    tau_level = inverse_dynamics_np(q, qd, qdd, p, Exo())
    e5 = float(np.max(np.abs(tau_tilt - tau_level)))
    # ── (6) 逐样本重力（数组）与逐步循环等价 ──
    gxs = rng.normal(size=q.shape[0]) * 0.4
    gys = rng.normal(size=q.shape[0]) * 0.4
    exo_arr = Exo(gravity_a=(gxs, gys), base_omega=exo.base_omega, base_alpha=exo.base_alpha)
    e6 = float(np.max(np.abs(inverse_dynamics_np(q, qd, qdd, p, exo_arr)
                             - np.stack([inverse_dynamics_np(q[i:i + 1], qd[i:i + 1],
                                                            qdd[i:i + 1], p,
                                                            Exo(gravity_a=(gxs[i], gys[i]),
                                                                base_omega=exo.base_omega,
                                                                base_alpha=exo.base_alpha))[0]
                                        for i in range(q.shape[0])], axis=0))))

    # ── (8)(9) P 方向约束: along_d ⇒ Px/Py 严格沿 d ──
    e8 = float("nan")
    e9 = float("inf")
    n_ok = 0
    for (ddx, ddy) in ((0.03, 0.0), (0.021, -0.017)):
        n = math.hypot(ddx, ddy)
        lay_p = ParamGroups(p, p_along_d=(ddx, ddy))
        r0 = lay_p.to_raw_init(p.vector())
        bp = lay_p.build_params(lay_p.to_physical(r0))
        e8 = (max(e8, abs(float(bp.Px) * (ddy / n) - float(bp.Py) * (ddx / n)))
              if np.isfinite(e8) else abs(float(bp.Px) * (ddy / n) - float(bp.Py) * (ddx / n)))
        raw = r0.copy()
        raw[lay_p.index(P_ALONG_D_NAME)] = 1.5
        bp2 = lay_p.build_params(lay_p.to_physical(raw))
        e9 = min(e9, max(abs(float(bp2.Px) - 1.5 * ddx / n), abs(float(bp2.Py) - 1.5 * ddy / n)))
        if "Px" not in lay_p.learnable_names and "Py" not in lay_p.learnable_names:
            n_ok += 1
    e9 = float(e9)

    # ── (10)(11) 两组映射往返 + 无限位 ──
    lay = ParamGroups(p)                      # 全部 18 个都可学习
    e10 = float(np.max(np.abs(lay.full_vector(lay.to_physical(lay.to_raw_init(p.vector())))
                              - p.vector())))
    raw_ext = np.zeros(lay.n_learnable)
    for k, nm in enumerate(lay.learnable_names):
        raw_ext[k] = 50.0
    if lay.n_learnable > 1:
        raw_ext[1] = -50.0            # 再取一个极小值（exp(−50)）
    pos_ext = lay.full_vector(lay.to_physical(raw_ext))
    e11 = 0.0 if (np.all(pos_ext[list(POSITIVE_IDX)] > 0.0)
                  and np.all(np.isfinite(pos_ext))) else 1.0

    # ── (13) 3-DOF 装配 == 2-DOF 子块 + τ_t ──
    q3 = rng.normal(size=(5, 3)) * 0.5
    qd3 = rng.normal(size=(5, 3)) * 1.5
    qdd3 = rng.normal(size=(5, 3)) * 4.0
    tau3_in = rng.normal(size=(5, 2)) * 0.4
    tau3 = inverse_dynamics_backlash_np(q3, qd3, qdd3, p, exo)
    tau2 = inverse_dynamics_np(q3[:, 1:3], qd3[:, 1:3], qdd3[:, 1:3], p, exo)
    tt = backlash_torque_np(q3[:, 0] - q3[:, 1] - p.backlash_beta, qd3[:, 0] - qd3[:, 1], p)
    e13 = max(
        float(np.max(np.abs(tau3[:, 1] - (tau2[:, 0] - tt)))),
        float(np.max(np.abs(tau3[:, 2] - tau2[:, 1]))),
        float(np.max(np.abs(tau3[:, 0] - (p.Jmotor * qdd3[:, 0] + tt
                                          + motor_friction_np(qd3[:, 0], p)
                                          + p.tau_offset_motor)))))
    # ── (14) 正/逆动力学往返 ──
    qdd_f = forward_accel_backlash_np(q3, qd3, tau3_in, p, exo)
    tau_rt = inverse_dynamics_backlash_np(q3, qd3, qdd_f, p, exo)
    u_expect = np.stack([tau3_in[:, 0], np.zeros(5), tau3_in[:, 1]], axis=-1)
    e14 = float(np.max(np.abs(tau_rt - u_expect)))

    # ── (15) 可微仿真模型 vs numpy 参考（rk4 / euler）──
    e15 = float("nan")
    if torch is not None:
        base_all = p.with_vector(p.vector())
        lay_all = ParamGroups(base_all)
        ph = lay_all.to_physical(torch.tensor(lay_all.to_raw_init(p.vector()),
                                              dtype=torch.float64))
        params1 = ph.unsqueeze(0)
        sc = DifferentiableSimulator.pack_seq_const(
            torch.tensor(q3[0:1], dtype=torch.float64),
            torch.tensor(qd3[0:1], dtype=torch.float64),
            exo.base_omega, exo.base_alpha)
        grav_seq = np.tile(np.array(exo.gravity_a, dtype=np.float64), (q3.shape[0], 1))
        beta_seq = np.full(q3.shape[0], p.backlash_beta)
        sv = DifferentiableSimulator.pack_seq_var(
            torch.tensor(tau3_in[None, :, 0], dtype=torch.float64),
            torch.tensor(tau3_in[None, :, 1], dtype=torch.float64),
            torch.tensor(grav_seq[None], dtype=torch.float64),
            torch.tensor(beta_seq[None], dtype=torch.float64))
        for integ, sub in (("rk4", 4), ("euler", 4)):
            sim = DifferentiableSimulator(dt=0.01, substeps=sub, integrator=integ,
                                          layout=lay_all, base=base_all, beta_from_input=True)
            pos, vel = sim(params1, sc, sv)
            th_t = np.stack([x[0].detach().numpy() for x in pos], axis=-1)
            dth_t = np.stack([x[0].detach().numpy() for x in vel], axis=-1)
            exo_seq = [Exo(gravity_a=(float(grav_seq[i, 0]), float(grav_seq[i, 1])),
                           base_omega=exo.base_omega, base_alpha=exo.base_alpha)
                       for i in range(q3.shape[0])]
            th_n, dth_n = simulate_backlash_np(p, q3[0], qd3[0], tau3_in, 0.01, EXO_ZERO, sub,
                                               exo_seq=exo_seq, beta_seq=beta_seq,
                                               integrator=integ)
            cur = max(float(np.max(np.abs(th_t - th_n))), float(np.max(np.abs(dth_t - dth_n))))
            e15 = cur if not np.isfinite(e15) else max(e15, cur)
    # ── (16) 全部可学习参数的梯度有限且非零 ──
    e16 = float("nan")
    grad_min = grad_max = float("nan")
    if torch is not None:
        raw_t = torch.tensor(lay_all.to_raw_init(p.vector()), dtype=torch.float64,
                             requires_grad=True)
        # ★ 显式开重力再查梯度: Pbx/Pby **只**通过重力项进入模型
        sim_g = DifferentiableSimulator(dt=0.01, substeps=1, integrator="euler",
                                       layout=lay_all, base=base_all,
                                       beta_from_input=False, gravity_on=True)
        sv_g = DifferentiableSimulator.pack_seq_var(
            torch.tensor(tau3_in[None, :, 0], dtype=torch.float64),
            torch.tensor(tau3_in[None, :, 1], dtype=torch.float64),
            torch.tensor(grav_seq[None], dtype=torch.float64), None)
        pos_g, vel_g = sim_g(lay_all.to_physical(raw_t).unsqueeze(0), sc, sv_g)
        loss = sum((x ** 2).mean() for x in pos_g) + sum((x ** 2).mean() for x in vel_g)
        g = torch.autograd.grad(loss, raw_t)[0]
        gv = np.abs(g.detach().numpy())
        grad_min, grad_max = float(np.min(gv)), float(np.max(gv))
        e16 = 0.0 if (np.all(np.isfinite(gv)) and np.all(gv > 1e-12)) else 1.0
    # ── (17) β 必须真的影响轨迹 ──
    th_a, _ = simulate_backlash_np(p, q3[0], qd3[0], tau3_in, 0.01, exo, 4, integrator="euler")
    th_b, _ = simulate_backlash_np(p.with_vector(p.vector() + np.eye(NPARAM)[15] * 0.02),
                                   q3[0], qd3[0], tau3_in, 0.01, exo, 4, integrator="euler")
    e17 = float(np.max(np.abs(th_a - th_b)))

    if verbose:
        print("[selftest] max|Y·φ − ID|          =", f"{e1:.3e}")
        print("[selftest] max|Y_analytic − Y_num|=", f"{e2:.3e}")
        print("[selftest] 重力项通道 |τ_tilt − τ_level| max =", f"{e5:.3e}", "(应 ≈ P·g 量级)")
        print("[selftest] 逐样本重力一致性 max err =", f"{e6:.3e}")
        print("[selftest] P 沿 d 约束 (Px·dy = Py·dx) max err =", f"{e8:.3e}")
        print("[selftest] P 沿 d 约束 (|P| 缩放一致性) max err =", f"{e9:.3e}")
        print("[selftest] 两组映射往返(实数/正数) max err =", f"{e10:.3e}", "(无限位)")
        print("[selftest] 无限位检查 raw=±50 ⇒ 正参数仍 >0 且有限 :", "OK" if e11 == 0.0 else "FAIL")
        print("[selftest] Px/Py 在 along_d 下已移出可学习组 :", "OK" if n_ok == 2 else "FAIL")
        print("[selftest] 3-DOF 装配 vs 2-DOF 子块 + τ_t max err =", f"{e13:.3e}")
        print("[selftest] 3-DOF 正/逆动力学往返 max err =", f"{e14:.3e}")
        print("[selftest] 可微模型 vs numpy(rk4/euler) max err =", f"{e15:.3e}")
        print("[selftest] 18 参梯度 |∂loss/∂raw| ∈ [%.3e, %.3e] :" % (grad_min, grad_max),
              "OK" if e16 == 0.0 else "FAIL")
        print("[selftest] β 影响轨迹 max|Δθ| =", f"{e17:.3e}", "(应 > 0)")
    ok = (e1 < 1e-9 and e2 < 1e-5 and e5 > 1e-4 and e6 < 1e-12 and e8 < 1e-12
          and e9 < 1e-12 and e10 < 1e-12 and e11 == 0.0 and n_ok == 2
          and e13 < 1e-12 and e14 < 1e-10 and e17 > 1e-6
          and (torch is None or (e15 < 1e-9 and e16 == 0.0)))
    if verbose:
        print("[selftest]", "PASS" if ok else "FAIL")
    return ok
