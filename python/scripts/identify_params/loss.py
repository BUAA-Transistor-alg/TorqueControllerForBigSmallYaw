#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""损失计算 —— **只有这一处定义**（Adam 训练器 `train.py` 与 CMA-ES 优化器 `cmaes_fit.py` 共用）。

损失 = 角度误差 MSE（★ **不 wrap**：角度在加载时已连续化，模型必须对**圈数**负责）
       + 角速度误差 MSE，**两项等权**（``loss_mode="mse"``，默认）；
    或旧配方的 Huber（``loss_mode="huber"``，可用 ``vel_weight`` 加权）。

归约口径:  ``ax_w``（``[B,C]``，每行只对参与拟合的通道给非零权重且和为 1）⇒ 结果 =
**所选通道上的平均**；给了 ``mask`` 就按有效点平均（分窗 padding 用）。

本模块只依赖**已有的 torch 模型**（:class:`~identify_params.model.DifferentiableSimulator`）,
不做任何参数化/优化器相关的事 —— 参数化在 :mod:`~identify_params.params`，优化在
:mod:`~identify_params.train` / :mod:`~identify_params.cmaes_fit`。
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

# 默认损失超参（与 train.FitConfig 的默认值保持一致）
HUBER_DELTA = 2.0e-3        # rad（~ 量化步长的 3 倍）
VEL_HUBER_DELTA = 0.05      # rad/s
BAD_LOSS = 1.0e30           # 目标值非有限（NaN/Inf）时返回的哨兵（无梯度优化器不能吃 NaN）


# ============================================================================
# 损失本体
# ============================================================================
def weighted_reduce(term, ax_w, mask=None):
    """``term [B,T,C]`` × ``ax_w [B,C]`` → 标量（所选通道上的平均）。

    ``mask=None`` ⇒ 按 ``B×T`` 个点平均；否则按 ``mask`` 的有效点数平均。
    """
    if mask is None:
        return (term * ax_w[:, None, :]).sum() / (term.shape[0] * term.shape[1])
    w = mask[:, :, None] * ax_w[:, None, :]
    return (term * w).sum() / mask.sum()


def pointwise_loss(th_pred, dth_pred, th_true, dth_true, *, loss_mode="mse",
                   huber_delta=HUBER_DELTA, vel_weight=0.0,
                   vel_huber_delta=VEL_HUBER_DELTA):
    """逐点损失项 ``[B,T,C]``（**未做轴/掩码归约**）。

    ``loss_mode="mse"``（默认）: ``(θ_err)² + (ω_err)²``，两项等权。
    ``loss_mode="huber"``（旧配方）: ``huber(θ_err) + vel_weight·huber(ω_err)``。
    """
    err = th_pred - th_true              # ★ 不 wrap（模型对圈数负责）
    verr = dth_pred - dth_true
    if loss_mode == "mse":
        return err ** 2 + verr ** 2
    pl = torch.nn.functional.huber_loss(err, torch.zeros_like(err),
                                        delta=huber_delta, reduction="none")
    if vel_weight <= 0.0:
        return pl
    vl = torch.nn.functional.huber_loss(verr, torch.zeros_like(verr),
                                        delta=vel_huber_delta, reduction="none")
    return pl + vel_weight * vl


def pair_loss(th_pred, dth_pred, th_true, dth_true, mask=None, ax_w=None, *,
              loss_mode="mse", huber_delta=HUBER_DELTA, vel_weight=0.0,
              vel_huber_delta=VEL_HUBER_DELTA, reduce="mean"):
    """归约后的损失。

    ``reduce="mean"``       → 标量（单片段 / 整窗 / CMA-ES 目标都用它）；
    ``reduce="per_sample"`` → ``[B]`` 逐样本损失（全批配方自己再 ``.mean()`` 用）。
    """
    pt = pointwise_loss(th_pred, dth_pred, th_true, dth_true, loss_mode=loss_mode,
                        huber_delta=huber_delta, vel_weight=vel_weight,
                        vel_huber_delta=vel_huber_delta)
    if reduce == "per_sample":
        return (pt * ax_w[:, None, :]).sum(dim=(1, 2))
    return weighted_reduce(pt, ax_w, mask)


# ============================================================================
# 「前向 + 损失」一步到位（前向用的就是已有的 torch 模型）
# ============================================================================
def rollout_states(sim, params, seq_const, seq_var):
    """跑一次可微前向，返回 ``(theta[B,T,C], dtheta[B,T,C])``（C = 2: 云台/小 yaw）。

    ★ ``sim`` 就是 :class:`~identify_params.model.DifferentiableSimulator`（不含可学习参数）,
    调用方按 batch 传 ``params``（物理量）。模型直接返回已堆好的张量。
    """
    return sim(params, seq_const, seq_var)


def sequence_loss(sim, params, seq_const, seq_var, theta, dtheta, mask=None, ax_w=None,
                  **loss_kw):
    """前向 + 损失一步到位；返回 ``(loss, th_pred, dth_pred)``。"""
    th_pred, dth_pred = rollout_states(sim, params, seq_const, seq_var)
    loss = pair_loss(th_pred, dth_pred, theta, dtheta, mask=mask, ax_w=ax_w, **loss_kw)
    return loss, th_pred, dth_pred


class WindowObjective:
    """把 **raw 参数向量** → **确定性**窗口损失（给 CMA-ES 这类无梯度优化器调用）。

    ``forward_fn(raw_np) -> (theta_pred, dtheta_pred)``：由调用方按"用哪个前向"注入
      · torch 模型 : ``layout.to_physical`` → ``DifferentiableSimulator`` → stack
      · numpy 批量 : :func:`~identify_params.model.rollout_batched_np`
      · 手写 C++   : :class:`~identify_params.planar2_sim.FastSimulator`
    返回的张量/数组都会走 **同一份损失实现**（``pair_loss``）—— 损失只有这一处。

    ★ 内部 ``torch.no_grad()``：不建图、不反向 —— 用无梯度优化器时**梯度全程关闭**。
    评价口径是**确定性的**（固定窗口 + mask，没有随机片段），否则目标带噪声、CMA-ES 会退化。
    """

    def __init__(self, forward_fn, theta, dtheta, mask=None, ax_w=None, loss_kw=None):
        self.forward_fn = forward_fn
        self.theta = theta
        self.dtheta = dtheta
        self.mask = mask
        self.ax_w = ax_w
        self.loss_kw = dict(loss_kw or {})
        self.n_eval = 0
        self.best_raw = None
        self.best_f = float("inf")

    def loss_of(self, raw) -> float:
        """一个 raw 向量的确定性损失（**不建梯度图**）。"""
        with torch.no_grad():
            th_pred, dth_pred = self.forward_fn(np.asarray(raw, dtype=np.float64))
            th_pred = torch.as_tensor(th_pred)
            dth_pred = torch.as_tensor(dth_pred)
            loss = pair_loss(th_pred, dth_pred, self.theta, self.dtheta,
                             mask=self.mask, ax_w=self.ax_w, **self.loss_kw)
            return float(loss)

    def __call__(self, raw) -> float:
        """目标函数入口（无梯度优化器调用这个）。非有限值折成哨兵，避免污染优化器。"""
        f = self.loss_of(raw)
        self.n_eval += 1
        if not math.isfinite(f):
            return BAD_LOSS
        if f < self.best_f:
            self.best_f = f
            self.best_raw = np.asarray(raw, dtype=np.float64).copy()
        return f
