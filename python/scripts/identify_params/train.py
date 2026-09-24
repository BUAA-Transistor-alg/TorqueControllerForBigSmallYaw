#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""③ 训练逻辑 —— 输出误差法（output error）参数辨识。

★ 默认配方 = 原仓库 ``TorqueController/python/scripts/param_ident.py`` 的同款配方:

  1. **无任何参数限位**：没有 sigmoid 软边界 / clamp / 投影 / 惩罚项。
     可学习参数按**可取值范围**分成两组（见 :mod:`~identify_params.params`）:
     全体实数直接自由，正数用 log 参数化 φ = exp(raw)。
  2. **积分 = RK4，substeps = 4**，dt 取自数据（100 Hz ⇒ 0.01 s）。
  3. **每次优化步只随机截取 seg_steps = 10 步（0.1 s）的片段**，epochs = 1000；
     每个 epoch 对每段数据各抽 1 个片段做 1 个 Adam 步。
  4. **损失 = 角度误差 MSE（先 wrap 到 (−π, π]）+ 角速度误差 MSE，两项等权相加**。
  5. **Adam，lr = 3e-4**，**无学习率调度**，**无 LBFGS**。
  6. 初值 = ``default_param_vector()``（**只由参数表 `PARAM_SPECS` 的 `default` 派生**，
     全包唯一来源；要换起点改参数表，或用 `--init-vector` 整体替换）。
  7. 无训练/验证划分；``val_loss`` 只是最后在全批算一次的训练损失。
  8. 训练中记录每 epoch 的 loss 与 18 个参数，供收敛曲线使用。

可微仿真模型见 :class:`~identify_params.model.DifferentiableSimulator`（不含可学习参数）;
可学习参数 → 物理参数的映射由 :class:`~identify_params.params.ParamGroups` 完成。
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from .data import ask_skip_segment, state_arrays
from .loss import pair_loss
from .model import DifferentiableSimulator, simulate_backlash_np
from .params import (
    AXIS_BIG,
    AXIS_CHANNELS,
    AXIS_SMALL,
    EXO_ZERO,
    EXTRA_PARAM_NAMES,
    NCORE,
    NPARAM,
    PARAM_NAMES,
    PARAM_UNITS,
    PlanarParams,
    ParamGroups,
    Exo,
    default_param_vector,
    exo_from_gravity,
    p_direction,
    resolve_fixed_names,
)

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# ============================================================================
# 拟合配置 / 结果
# ============================================================================
@dataclass
class FitConfig:
    """拟合配置。

    ★★ 默认值 = **原仓库 `param_ident.py` 同款配方**::

        epochs=1000, seg_steps=10（0.1 s 随机片段）,
        loss_mode="mse"（角度 wrap 后 MSE + 角速度 MSE 等权）, lr=3e-4（常数，无 scheduler）,
        integrator="rk4", substeps=4, lbfgs_iters=0, window_len=0, batch_size=0,
        windows_per_seg=1, iters=0（⇒ 走新配方；>0 才回到旧配方）。
    """

    fit_axis: str = "both"          # both | big | small
    # ── ★ 新配方（原仓库同款）──
    epochs: int = 1000              # ★ 与**原仓库同轮数**；每 epoch 每段 1 个 Adam 步
    seg_steps: int = 10             # ★ 每个优化步随机截取的片段长度（10 步 = 0.1 s @100 Hz）
    loss_mode: str = "mse"          # mse = 角度(wrap)MSE + 角速度 MSE 等权 | huber = 旧配方
    integrator: str = "rk4"         # ★ rk4（与 C++ MPC 的 integrateStepBacklash 同款）
    lr_schedule: str = "none"       # none = 常数 lr（原仓库）| cosine = 从第 0 步就开始余弦退火
    cos_decay_steps: int = 0        # ★ **最后 n 步**用余弦把 lr 从 `lr` 衰减到 0；0 = 关闭
    # ★ 长跑断点: 每 N 个 epoch 把当前 φ 写一份到 `<checkpoint_prefix>.epXXXXXX`。
    checkpoint_every: int = 0       # 0 = 关闭
    checkpoint_prefix: str = ""     # 通常 = --out（不给就不写）
    # ── 旧配方（显式给 iters>0 才启用；或 --legacy-recipe 一键预设）──
    iters: int = 0                  # >0 ⇒ 旧配方：精确跑 iters 个 Adam 步（分窗 mini-batch）
    lbfgs_iters: int = 0            # LBFGS 迭代数（0 = 关闭 ⇒ 原仓库没有 LBFGS）
    lbfgs_max_iter: int = 5
    lr: float = 3e-4                # ★ 原仓库 lr=3e-4（旧配方默认 5e-3）
    seed: int = 42
    substeps: int = 4               # ★ 背隙接触模态 ω≈190 rad/s ⇒ 10 ms 一步需细分
    huber_delta: float = 2.0e-3     # rad（~ 量化步长的 3 倍）；仅 loss_mode="huber" 用
    vel_weight: float = 0.0         # 旧配方的角速度项权重（0 = 不用）；mse 模式固定等权 1.0
    vel_huber_delta: float = 0.05   # rad/s
    free_init_vel: bool = False     # 是否把各段初始角速度当作自由参数一起优化
    freeze_params: tuple = ()       # ★ 额外固定（= 移出两组可学习参数）的参数下标（消融用）
    # ★★ **默认固定背隙直通项 γ**（= 参数 11）在初值上（默认 0.002）。
    #   它的定位只是"死区内的梯度引导"；一旦放开拟合，它会被优化器拿来**替模型填"刚性接触"
    #   的台阶**（实测 γ 从 0.002 涨到 0.27~0.39、δ 被撑到 0.25 rad，真值 0.087）。
    #   要复现"γ 自由"的消融：`--no-freeze-backlash-through`。
    freeze_backlash_through: bool = True
    # ── 背隙中心 β 的来源（★ β 是**必需数据**，不再是可学习参数；--beta-mode=fit 已移除）──
    #   auto  : 与拟合帧匹配 —— state_mode=est 用估计帧 `backlash_center`；
    #           state_mode=true（该段有 theta_true）用真值帧 `beta_true`
    #   column: 强制估计帧 `backlash_center`
    #   true  : 强制真值帧 `beta_true`（只用于仿真数据）
    #   缺列 ⇒ **直接报错**（不再退回"β 固定初值"）
    beta_mode: str = "auto"
    # ★ 状态目标来源: est（默认）= 记录/估计值；true = 仿真真值（仅仿真数据）
    state_mode: str = "est"
    # ★ 全批加速: 每个 epoch 仍对**每段**随机抽一个 seg_steps 片段，但把 W 段拼成一个
    #   batch 做**一次** Adam 步（损失 = 各段损失的均值）。
    batch_segments: bool = False
    # ★ 每 N 个 epoch 在**留出集**上算一次开环 RMSE（学习曲线用；0 = 关）。
    eval_every: int = 0
    eval_segs: list | None = None
    # 由 fit_params_torch 回填: β 是否来自数据列（★ 现在恒为 True —— β 是必需数据）
    use_beta_column: bool = True
    p_bound: float = 0.05           # ★ 已废弃并被忽略（不再有任何参数限位；仅为字段兼容保留）
    init_vector: np.ndarray | None = None
    truth_vector: np.ndarray | None = None  # 真值/参考值（画收敛曲线虚线用）
    print_every: int = 50
    verbose: bool = True
    dtype: "torch.dtype" = None
    device: str = "cpu"
    max_points: int = 0             # >0 时每段只取前 max_points 个点（加速调试）
    # ── 数据分窗与 mini-batch（**旧配方**用）──
    p_constraint: str = "free"      # free | along_d（P=|P|·R(−θ*)·d̂，单标量）| zero（固定 0）
    p_zero_angle_deg: float = 0.0   # ★ 平衡点(θ*)在当前零点坐标系里的读数（度）
    fix_p: bool = False             # 兼容旧参数：等价于 p_constraint = "zero"
    window_len: int = 0             # 每个窗口的点数（0 = 整段作一个窗口 ⇒ 新配方不用分窗）
    windows_per_seg: int = 1        # 每段切几个窗口（仅旧配方 + window_len>0 时有效）
    batch_size: int = 0             # 旧配方 Adam 每步随机抽几个窗口（0 = 全部）

    def __post_init__(self):
        if self.fix_p:
            self.p_constraint = "zero"
        self.loss_mode = str(self.loss_mode).lower()
        self.integrator = str(self.integrator).lower()
        self.lr_schedule = str(self.lr_schedule).lower()
        self.beta_mode = str(self.beta_mode).lower()
        self.cos_decay_steps = max(0, int(self.cos_decay_steps))
        if self.cos_decay_steps > 0 and self.lr_schedule not in ("none", ""):
            print(f"[torch] [WARN] 同时给了 --cos-decay-steps={self.cos_decay_steps} 与 "
                  f"--lr-schedule={self.lr_schedule} ⇒ 以 cos-decay 为准（忽略 lr_schedule）")
            self.lr_schedule = "none"

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
    """拟合结果。"""

    phi: np.ndarray
    phi0: np.ndarray
    loss_history: list
    val_loss: float
    n_iter: int
    seconds: float
    rank_ratio: float = float("nan")
    # ── 收敛曲线用（新配方）──
    param_history: list = None      # 每 step/epoch 的 18 个参数（list of list）
    epoch_losses: list = None       # 每个 epoch 的平均 loss（= loss_history 的别名）
    n_free: int = 0                 # 可学习参数个数（两组之和）
    n_steps: int = 0                # 实际 Adam 步数
    recipe: str = ""                # 配方描述
    truth: np.ndarray | None = None  # 真值/参考值（画虚线用）
    eval_hist: list = None           # [(epoch, {通道 RMSE})]（留出集学习曲线）
    config: dict = None             # 配置摘要（打印/画图用）


# ============================================================================
# 数据打包（batch-first: [W, L, ...]）
# ============================================================================
def beta_source(seg, state_mode: str = "est", beta_mode: str = "auto"):
    """按**拟合帧**选 β 源（★ 用户约定：β 必须来自与拟合目标同一帧的数据列）。

      · ``auto``  : 与拟合帧匹配 —— `state_mode=est` 用估计帧 `backlash_center`；
                    `state_mode=true`（该段有 `theta_true`）用真值帧 `beta_true`
      · ``column``: 强制估计帧 `backlash_center`
      · ``true``  : 强制真值帧 `beta_true`

    返回 ``(array | None, 说明)``；``None`` 表示该段在所选帧下没有 β 数据。
    （`--beta-mode=fit` 已移除：β 不再作为可学习参数、也不再退回固定初值。）
    """
    fit_true = (str(state_mode) == "true" and seg.theta_true is not None)
    mode = str(beta_mode)
    if mode == "true":
        if not fit_true:
            raise ValueError(
                f"--beta-mode=true 用的是真值帧 β，但 {seg.source} 没有 `theta_true`"
                f"（拟合帧 = est）⇒ 请改用 --beta-mode=auto/column，"
                f"或对带真值列的仿真数据用 --state-mode=true")
        return seg.beta_true, "beta_true(真值帧)"
    if mode == "column":
        if fit_true:
            raise ValueError(
                f"--beta-mode=column 用的是估计帧 `backlash_center`，但拟合帧 = true"
                f"（{seg.source} 有 `theta_true`）⇒ 请改用 --beta-mode=true/auto")
        return seg.beta, "backlash_center(估计帧)"
    return ((seg.beta_true, "beta_true(真值帧)") if fit_true
            else (seg.beta, "backlash_center(估计帧)"))


def filter_beta_segments(segs, state_mode: str = "est", beta_mode: str = "auto",
                         what: str = "训练集") -> list:
    """★ β 是**必需数据**：把在所选帧下没有 β 的段用同一套 ``[Y/n]`` 提示**跳过**
    （不再是直接报错；Y=跳过，默认；n=退出）。跨帧的非法组合仍由 `beta_source` 报错。"""
    if not segs:
        return list(segs or [])
    out, dropped = [], 0
    for s in segs:
        arr, tag = beta_source(s, state_mode, beta_mode)
        if arr is None:
            ask_skip_segment(s, f"缺少 β 数据（{tag} 列）")
            dropped += 1
            continue
        out.append(s)
    if dropped:
        print(f"[beta] {what}: 跳过 {dropped}/{len(segs)} 段（缺 β）")
    return out


def _pack_windows(segs, cfg: FitConfig, chan_sel, dtype, dev):
    """把数据切成等长**窗口**并打包成张量。

    ★ batch-first: 第 0 维 = 窗口（= 可微仿真模型的 batch），第 1 维 = 时间。
    状态是 3-DOF: theta/dtheta [W,L,3]（电机 / 云台 / 小 yaw）；tau [W,L,2]。
    `chan_sel` = 参与损失的状态通道（big ⇒ (0,1)、small ⇒ (2,)、both ⇒ (0,1,2)）。
    返回 dict: tau/theta/dtheta, mask [W,L], q0/qd0 [W,3], w_axis [W,3] 以及来源信息。
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

    tau = np.zeros((W, L, 2))
    theta = np.zeros((W, L, 3))
    dtheta = np.zeros((W, L, 3))
    grav = np.zeros((W, L, 2))
    beta = np.zeros((W, L))
    mask = np.zeros((W, L))
    q0 = np.zeros((W, 3))
    qd0 = np.zeros((W, 3))
    w_axis = np.zeros((W, 3))
    for wi, (si, st, l) in enumerate(specs):
        s = segs[si]
        sl = slice(st, st + l)
        th_s, dth_s = state_arrays(s, cfg.state_mode)
        tau[wi, :l] = s.tau[sl]
        theta[wi, :l] = th_s[sl]
        dtheta[wi, :l] = dth_s[sl]
        if s.gravity is not None:
            grav[wi, :l] = s.gravity[sl]
        bsrc = beta_source(s, cfg.state_mode, cfg.beta_mode)[0]
        if bsrc is not None:
            beta[wi, :l] = bsrc[sl]
        mask[wi, :l] = 1.0
        q0[wi] = th_s[st]
        qd0[wi] = dth_s[st]
        for c in chan_sel:
            w_axis[wi, c] = 1.0 / len(chan_sel)

    t = lambda x: torch.tensor(x, dtype=dtype, device=dev)          # noqa: E731
    return {"tau": t(tau), "theta": t(theta), "dtheta": t(dtheta), "mask": t(mask),
            "q0": t(q0), "qd0": t(qd0), "w_axis": t(w_axis), "grav": t(grav),
            "beta": t(beta),
            "has_gravity": bool(np.any(grav != 0.0)),
            "has_beta": bool(np.any(beta != 0.0)),
            "dt": dt, "L": L, "W": W,
            "lens": [int(sp[2]) for sp in specs],          # 每个窗口的**有效**长度（未 padding）
            "specs": [("seg%d" % sp[0], sp[1], sp[2]) for sp in specs],
            "seg_of_window": [sp[0] for sp in specs]}


# ============================================================================
# 输出 / 学习率 / 摘要
# ============================================================================
def write_params_file(path: str, phi, recipe: str = "") -> None:
    """把 18 参写成 `--out` 那种 `名字 = 值` 文本（checkpoint 与最终输出共用同一格式）。

    ★ 父目录不存在会自动创建（否则长跑会在**最后一步**写结果时才发现目录不存在而报错）。
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# identify_params（输出误差法）输出\n")
        if recipe:
            fh.write(f"# 配方: {recipe}\n")
        for nm, v in zip(PARAM_NAMES, np.asarray(phi, dtype=np.float64)):
            fh.write(f"{nm} = {float(v):.9f}\n")


def lr_at_step(step: int, total: int, base_lr: float, decay_steps: int) -> float:
    """分段学习率: 前 `total − decay_steps` 步 = `base_lr`（常数），
    最后 `decay_steps` 步按**余弦**从 `base_lr` 衰减到 0::

        p = (step − (total − n)) / n
        lr = base_lr · ½(1 + cos(π·p))          # p=0 → base_lr, p=1 → 0

    `decay_steps ≤ 0` ⇒ 恒为 `base_lr`（= 关闭）；`decay_steps ≥ total` ⇒ 全程余弦。
    """
    n = int(decay_steps)
    if n <= 0:
        return float(base_lr)
    total = max(1, int(total))
    start = total - n
    if step < start:
        return float(base_lr)
    p = min(1.0, max(0.0, (step - start) / float(n)))
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * p))


def _config_summary(cfg: FitConfig, layout: "ParamGroups", W: int, dt: float,
                    n_free: int) -> dict:
    """配置摘要（打印 + 存进 FitResult，供收敛曲线标题/报告使用）。"""
    if cfg.use_legacy_path:
        recipe = f"旧精细配方：Adam {cfg.iters} 步(lr={cfg.lr:g}, {cfg.lr_schedule})"
        if cfg.lbfgs_iters > 0:
            recipe += f" + LBFGS {cfg.lbfgs_iters} 步"
        recipe += f"，{cfg.loss_mode} 损失，{cfg.integrator.upper()} 积分"
    else:
        _lr_desc = (f"lr={cfg.lr:g}（常数）"
                    + (f"，最后 {cfg.cos_decay_steps} epoch 余弦衰减到 0"
                       if cfg.cos_decay_steps > 0 else ""))
        recipe = (f"原仓库配方：epochs={cfg.epochs} × 段数{W} 个 Adam 步"
                  f"（= 原仓库 num_epochs 同轮数），每次 {cfg.seg_steps} 步(0.1 s)随机片段，"
                  f"{_lr_desc}，损失 = 角度 MSE(**不 wrap**) + 角速度 MSE(等权)，"
                  f"3-DOF {cfg.integrator.upper()} 积分(substeps={cfg.substeps})，"
                  f"无限位(两组: 全体实数 / 正数)")
    return {"recipe": recipe, "epochs": int(cfg.epochs), "iters": int(cfg.iters),
            "use_beta_column": bool(cfg.use_beta_column),
            "freeze_backlash_through": bool(cfg.freeze_backlash_through),
            "seg_steps": int(cfg.seg_steps), "lr": float(cfg.lr), "loss_mode": cfg.loss_mode,
            "integrator": cfg.integrator, "substeps": int(cfg.substeps),
            "lr_schedule": cfg.lr_schedule, "lbfgs_iters": int(cfg.lbfgs_iters),
            "cos_decay_steps": int(cfg.cos_decay_steps),
            "fit_axis": cfg.fit_axis, "p_constraint": cfg.p_constraint,
            "n_free": int(n_free), "n_sample": int(W), "dt": float(dt),
            "limits": "无（无上下界 / 无 clamp / 无投影 / 无惩罚项）",
            "param_space": layout.describe(),
            "fixed_params": list(layout.fixed_names),
            "init_source": ("参数表 PARAM_SPECS 的初值（全包唯一来源）"
                            if cfg.init_vector is None else "用户 --init-vector（整体替换）")}


def format_param_table(phi, phi0) -> list:
    """参数报告表（初值/估计/变化，按物理三组分段）—— Adam CLI 与 CMA-ES CLI 共用。"""
    phi = np.asarray(phi, dtype=np.float64)
    phi0 = np.asarray(phi0, dtype=np.float64)
    out = [f"{'#':>2} {'参数':<16} {'初值':>12} {'估计':>12} {'变化':>12}  单位",
           f"{'':>2} ---- 平面 8 参（云台/小 yaw 子块）----"]
    for j in range(NCORE):
        out.append(f"{j:>2} {PARAM_NAMES[j]:<16} {phi0[j]:>12.6f} {phi[j]:>12.6f} "
                   f"{phi[j] - phi0[j]:>+12.6f}  {PARAM_UNITS[j]}")
    out.append(f"{'':>2} ---- ★ 大 yaw 背隙 / 电机侧 ----")
    for j in range(NCORE, NCORE + len(EXTRA_PARAM_NAMES)):
        extra = f"   (= {math.degrees(phi[j]):.3f}°)" if j == 8 else ""
        out.append(f"{j:>2} {PARAM_NAMES[j]:<16} {phi0[j]:>12.6f} {phi[j]:>12.6f} "
                   f"{phi[j] - phi0[j]:>+12.6f}  {PARAM_UNITS[j]}{extra}")
    out.append(f"{'':>2} ---- ★ 大 yaw 侧一阶矩 Pb（只在倾斜 + 大 yaw 转动时可辨识）----")
    for j in range(NCORE + len(EXTRA_PARAM_NAMES), NPARAM):
        out.append(f"{j:>2} {PARAM_NAMES[j]:<16} {phi0[j]:>12.6f} {phi[j]:>12.6f} "
                   f"{phi[j] - phi0[j]:>+12.6f}  {PARAM_UNITS[j]}")
    return out


def format_header_snippet(phi) -> list:
    """可直接粘进 `include/tcbs/mpc/planar_yaw_params.h` 的几行。"""
    p = np.asarray(phi, dtype=np.float64)
    return [
        f"  p.Jbig_eff = {p[0]:.6f};  p.Js = {p[1]:.6f};",
        f"  p.Px = {p[2]:.6f};  p.Py = {p[3]:.6f};",
        f"  p.fcBig = {p[4]:.6f};  p.fvBig = {p[5]:.6f};",
        f"  p.fcSmall = {p[6]:.6f};  p.fvSmall = {p[7]:.6f};",
        f"  p.backlash_delta = {p[8]:.6f};  p.backlash_k = {p[9]:.4f};",
        f"  p.backlash_c = {p[10]:.4f};  p.backlash_through = {p[11]:.6f};",
        f"  p.Jmotor = {p[12]:.6f};  p.fcMotor = {p[13]:.6f};",
        f"  p.fvMotor = {p[14]:.6f};  // β 由估计器在线给（离线拟合值 {p[15]:+.6f} 仅供参考）",
        f"  p.Pbx = {p[16]:.6f};  p.Pby = {p[17]:.6f};"
        f"  // 大 yaw 侧一阶矩（只有倾斜数据才可辨识）",
    ]


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
# 拟合上下文（★ Adam 训练器与 CMA-ES 优化器**共用**的一份准备逻辑）
# ============================================================================
@dataclass
class FitContext:
    """一次拟合所需的全部数据/模型对象（数据打包 + β + 固定参数 + 可微模型）。

    ``train.fit_params_torch``（Adam）与 ``cmaes_fit``（CMA-ES）都从这里拿输入，
    保证两条优化路径用的是**同一份**打包/连续化/β/参数化/模型语义。
    """

    segs: list
    batch: dict
    dtype: object
    dev: object
    layout: "ParamGroups"
    base_phi: PlanarParams
    phi0: np.ndarray
    sim: "DifferentiableSimulator"
    dt: float
    L: int
    W: int
    lens: list
    chan_sel: tuple
    tau_t: object
    th_t: object
    dth_t: object
    mask_t: object
    q0_all: object
    qd0_all: object
    w_axis_all: object
    grav_all: object
    beta_all: object
    seq_var: object
    use_beta_col: bool
    beta_tag: str
    raw: object
    v0_free: object
    n_free: int
    summary: dict


def build_fit_context(segs, cfg: FitConfig, base: PlanarParams, exo: Exo = EXO_ZERO,
                      dtype=None, dev=None, seed: bool = True,
                      what: str = "训练集", honor_windows: bool = False) -> FitContext:
    """把数据段 + 配置 + 几何 → 可直接喂给优化器的上下文（Adam / CMA-ES 共用）。

    做这几件事（顺序与原 `fit_params_torch` 完全一致，数值不变）:
      ① 只对参与拟合的通道计误差（fit_axis）；② β 必需校验/提示跳过；
      ③ 数据打包成 batch-first 窗口张量；④ 初值（唯一来源 = `PARAM_SPECS`）；
      ⑤ P 方向约束 → 固定参数 → `ParamGroups`；⑥ seq_const/seq_var 与可微模型。
    """
    if torch is None:
        raise RuntimeError(f"需要 torch: {_TORCH_IMPORT_ERROR}")
    if not segs:
        raise ValueError("没有可用的数据段")
    dtype = dtype or (cfg.dtype or torch.float64)
    dev = torch.device(dev or cfg.device)
    if seed:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # ① 只对参与拟合的状态通道计误差（big ⇒ 电机+云台；small ⇒ 小 yaw；both ⇒ 3 个通道）
    chan_sel = ((0, 1, 2) if cfg.fit_axis == "both"
                else AXIS_CHANNELS[AXIS_BIG if cfg.fit_axis == "big" else AXIS_SMALL])

    # ② Adam 新配方 = 每段一个 sample（不分窗、不 mini-batch）；显式给了分窗设置时提示被忽略。
    #    ★ 无梯度优化器（CMA-ES）用**固定窗口**当目标 ⇒ honor_windows=True 时**不**忽略窗口设置。
    pack_cfg = cfg
    if (not honor_windows and not cfg.use_legacy_path
            and (cfg.window_len > 0 or cfg.windows_per_seg > 1
                 or (cfg.batch_size > 0 and not cfg.batch_segments))):
        if cfg.verbose:
            print("[torch] ★ 新配方按「整段 = 一个 sample」采样，忽略 "
                  f"window_len={cfg.window_len}, windows_per_seg={cfg.windows_per_seg}, "
                  f"batch_size={cfg.batch_size}（要旧配方请用 --iters>0 或 --legacy-recipe）")
        pack_cfg = replace(cfg, window_len=0, windows_per_seg=1, batch_size=0)

    # ── ★ β: **必需数据**，永远从所选帧的数据列来（--beta-mode=fit 已移除）──
    #   缺 β 的段用 [Y/n] 提示**跳过**（与 β 连续性校验同一套），不再直接报错。
    if cfg.beta_mode not in ("auto", "column", "true"):
        raise ValueError("beta_mode 必须是 auto / column / true")
    segs = filter_beta_segments(segs, cfg.state_mode, cfg.beta_mode, what)
    if not segs:
        raise ValueError(f"没有可用的数据段（{what}全部因缺 β 被跳过）")
    if cfg.eval_segs:
        cfg.eval_segs = filter_beta_segments(cfg.eval_segs, cfg.state_mode, cfg.beta_mode,
                                             "留出集")
    beta_tag = beta_source(segs[0], cfg.state_mode, cfg.beta_mode)[1]
    use_beta_col = True
    cfg.use_beta_column = True

    batch = _pack_windows(segs, pack_cfg, chan_sel, dtype, dev)
    dt, L, W = batch["dt"], batch["L"], batch["W"]
    lens = batch["lens"]

    # ③ 初值（★ 唯一来源 = 参数表 PARAM_SPECS；`--init-vector` 是整体替换）
    phi0 = (default_param_vector() if cfg.init_vector is None
            else np.asarray(cfg.init_vector, dtype=np.float64).copy())

    # ④ P 的方向约束（可选）: P = |P|·R(−θ*)·d̂ ⇒ Px/Py 退化成 real 组末尾一个派生标量
    mode = str(cfg.p_constraint).replace("-", "_")
    if mode not in ("free", "along_d", "zero"):
        raise ValueError("p_constraint 必须是 free / along_d / zero")
    p_along = None
    if mode == "along_d":
        p_along = p_direction(base.dx, base.dy, cfg.p_zero_angle_deg)
        print(f"[torch] P 方向约束: θ* = {cfg.p_zero_angle_deg:+.2f}° ⇒ "
              f"P ∝ ({p_along[0]:+.5f}, {p_along[1]:+.5f})")

    # ⑤ 固定参数 = 冻结的那些（★ 不属于"全体实数/正数"任何一组）
    fixed = resolve_fixed_names(cfg.fit_axis, cfg.freeze_params,
                                cfg.freeze_backlash_through, cfg.p_constraint, use_beta_col)
    base_phi = base.with_vector(phi0)          # 固定参数取初值
    layout = ParamGroups(base_phi, fixed_names=fixed, p_along_d=p_along)
    if layout.fixed_names and cfg.verbose:
        print("[torch] ★ 固定参数（不参与优化，恒保持初值）: "
              + ", ".join(f"{n}={layout.fixed_value(n):.6g}" for n in layout.fixed_names))

    raw = torch.tensor(layout.to_raw_init(phi0), dtype=dtype, device=dev, requires_grad=True)
    # ★ 每段的初始角速度自由量: 3-DOF ⇒ [W,3]（旧脚本这里是 [W,2]，状态从 2-DOF 扩到 3-DOF
    #   之后没同步，导致 `--free-init-vel` 一开就崩；这里修正为 3 个通道）。
    v0_free = (torch.zeros(W, 3, dtype=dtype, device=dev, requires_grad=True)
               if cfg.free_init_vel else None)

    tau_t, th_t, dth_t = batch["tau"], batch["theta"], batch["dtheta"]
    mask_t, q0_all, qd0_all, w_axis_all = (batch["mask"], batch["q0"], batch["qd0"],
                                           batch["w_axis"])
    grav_all = batch["grav"] if batch["has_gravity"] else None
    beta_all = batch["beta"] if use_beta_col else None

    # ⑥ seq_const / seq_var 打包成模型的输入契约（力矩在 seq_var 头两个通道）
    seq_var = DifferentiableSimulator.pack_seq_var(tau_t[..., 0], tau_t[..., 1],
                                                   grav_all, beta_all)
    sim = DifferentiableSimulator(dt=dt, substeps=cfg.substeps, integrator=cfg.integrator,
                                  layout=layout, base=base_phi, beta_from_input=use_beta_col)
    n_free = layout.n_learnable
    summary = _config_summary(cfg, layout, W, dt, n_free)
    return FitContext(
        segs=segs, batch=batch, dtype=dtype, dev=dev,
        layout=layout, base_phi=base_phi, phi0=phi0, sim=sim,
        dt=dt, L=L, W=W, lens=lens, chan_sel=chan_sel,
        tau_t=tau_t, th_t=th_t, dth_t=dth_t, mask_t=mask_t,
        q0_all=q0_all, qd0_all=qd0_all, w_axis_all=w_axis_all,
        grav_all=grav_all, beta_all=beta_all, seq_var=seq_var,
        use_beta_col=use_beta_col, beta_tag=beta_tag, raw=raw, v0_free=v0_free,
        n_free=n_free, summary=summary)


# ============================================================================
# 拟合器
# ============================================================================
def fit_params_torch(segs, cfg: FitConfig, base: PlanarParams, exo: Exo = EXO_ZERO) -> FitResult:
    """★ 输出误差法拟合：用记录力矩做可导前向仿真，最小化预测误差。

    **默认（新）配方 = 原仓库同款**::

        for epoch in range(epochs):
            for sample in samples:                   # = 每个数据段
                s = randint(0, T - seg_steps)        # ★ 随机截取 10 步（0.1 s）片段
                pos, vel = sim(φ, seq_const, seq_var)  # ★ RK4, dt 取自数据
                err = wrap(θ_sim − θ_meas)
                loss = mean(err²) + mean((ω_sim − ω_meas)²)
                opt.zero_grad(); loss.backward(); opt.step()

    **无任何参数限位**：可学习参数分成两组，正数组走 log 参数化（φ = exp(raw)），
    实数组直接自由；没有 clamp / box 约束 / 投影 / 惩罚项。被冻结的参数（含默认固定的 γ）
    在 :class:`~identify_params.params.ParamGroups` 里就是**固定参数**，不进可学习向量。
    """
    if torch is None:
        raise RuntimeError(f"需要 torch: {_TORCH_IMPORT_ERROR}")
    if not segs:
        raise ValueError("没有可用的数据段")
    t_start = time.time()

    # ★ 准备逻辑（打包 / β / 固定参数 / 模型）与 CMA-ES 优化器**共用同一份**
    ctx = build_fit_context(segs, cfg, base, exo)
    segs, batch, layout, base_phi, phi0 = ctx.segs, ctx.batch, ctx.layout, ctx.base_phi, ctx.phi0
    sim, dt, L, W, lens = ctx.sim, ctx.dt, ctx.L, ctx.W, ctx.lens
    raw, v0_free = ctx.raw, ctx.v0_free
    dev = ctx.dev
    th_t, dth_t = ctx.th_t, ctx.dth_t
    mask_t, w_axis_all = ctx.mask_t, ctx.w_axis_all
    q0_all, qd0_all = ctx.q0_all, ctx.qd0_all
    seq_var, beta_tag = ctx.seq_var, ctx.beta_tag
    n_free, summary = ctx.n_free, ctx.summary

    loss_mode = cfg.loss_mode

    # ── 物理参数 / 张量辅助 ──
    def current_phi():
        """raw → 可学习物理参数 [P]（可导）。"""
        return layout.to_physical(raw)

    def current_full() -> np.ndarray:
        """raw → 完整 18 维参数（np，固定参数已填入）——存档/评测/画图用。"""
        with torch.no_grad():
            v = current_phi().detach().cpu().numpy()
        return layout.full_vector(np.asarray(v, dtype=np.float64))

    def params_b(nb: int):
        return current_phi().unsqueeze(0).expand(nb, -1)

    def seq_const_windows(idx=None):
        """整窗/整段的 seq_const；``free_init_vel`` 时把每窗初始角速度加上去。"""
        if idx is None:
            q0, qd0, v0 = q0_all, qd0_all, v0_free
        else:
            q0, qd0 = q0_all[idx], qd0_all[idx]
            v0 = None if v0_free is None else v0_free[idx]
        if v0 is not None:
            qd0 = qd0 + v0
        return DifferentiableSimulator.pack_seq_const(q0, qd0, exo.base_omega, exo.base_alpha)

    def seq_const_at(wid, starts):
        """片段起点的 seq_const（起点状态用**实测** θ/ω，与原仓库一致）。"""
        q0 = th_t[wid, starts]
        qd0 = dth_t[wid, starts]
        if v0_free is not None:
            qd0 = qd0 + v0_free[wid]
        return DifferentiableSimulator.pack_seq_const(q0, qd0, exo.base_omega, exo.base_alpha)

    def _rollout_windows(idx=None):
        """整窗/整段前向仿真 → [B,L,3] 的位置与速度。"""
        sv = seq_var if idx is None else seq_var[idx]
        pos, vel = sim(params_b(int(sv.shape[0])), seq_const_windows(idx), sv)
        return torch.stack(pos, dim=-1), torch.stack(vel, dim=-1)

    def _rollout_slices(wid, starts, n: int):
        """从 ``starts`` 起的 n 步片段（可跨多个窗口）→ [B,n,3]。"""
        ii = starts[:, None] + torch.arange(n, device=dev)[None, :]
        sv = seq_var[wid[:, None], ii]                    # [B,n,5]
        pos, vel = sim(params_b(int(wid.numel())), seq_const_at(wid, starts), sv)
        return torch.stack(pos, dim=-1), torch.stack(vel, dim=-1)

    # ★ 损失本体只在 `loss.py` 定义一次（角度不 wrap、两项等权 / Huber）；
    #   这里只负责"取哪一段数据、怎么归约"。
    _loss_kw = dict(loss_mode=loss_mode, huber_delta=cfg.huber_delta,
                    vel_weight=cfg.vel_weight, vel_huber_delta=cfg.vel_huber_delta)

    def _pair_loss(th_pred, dth_pred, th_true, dth_true, mask=None, ax_w=None,
                   reduce="mean"):
        return pair_loss(th_pred, dth_pred, th_true, dth_true, mask=mask, ax_w=ax_w,
                         reduce=reduce, **_loss_kw)

    def _slice_loss(w: int, s: int, n: int):
        """★ 新配方的一个优化步: 第 w 个 sample 的 [s, s+n) 片段做一次可导前向仿真。"""
        wid = torch.tensor([int(w)], dtype=torch.long, device=dev)
        st = torch.tensor([int(s)], dtype=torch.long, device=dev)
        th_pred, dth_pred = _rollout_slices(wid, st, int(n))
        ii = st[:, None] + torch.arange(int(n), device=dev)[None, :]
        th_b = th_t[wid[:, None], ii]
        dth_b = dth_t[wid[:, None], ii]
        return _pair_loss(th_pred, dth_pred, th_b, dth_b, mask=None, ax_w=w_axis_all[wid])

    def _window_loss(idx=None):
        """旧配方（也用于新配方的全段 val_loss / 画图）: 整窗/整段一起前向仿真。"""
        th_pred, dth_pred = _rollout_windows(idx)
        if idx is None:
            th_b, dth_b, m_b, wa = th_t, dth_t, mask_t, w_axis_all
        else:
            th_b, dth_b, m_b, wa = th_t[idx], dth_t[idx], mask_t[idx], w_axis_all[idx]
        return _pair_loss(th_pred, dth_pred, th_b, dth_b, mask=m_b, ax_w=wa)

    params = [raw] + ([v0_free] if v0_free is not None else [])
    loss_hist, param_hist = [], []
    eval_hist = []
    n_steps = 0

    def _record_eval(ep_idx: int) -> None:
        """留出集上的开环 RMSE（学习曲线）——只在 `--eval-every` 打开且有留出数据时调用。"""
        if cfg.eval_every <= 0 or not cfg.eval_segs:
            return
        if (ep_idx + 1) % int(cfg.eval_every) != 0:
            return
        rm = channel_rmse(cfg.eval_segs, current_full(), base_phi,
                          integrator=cfg.integrator, substeps=cfg.substeps,
                          state_mode=cfg.state_mode, beta_mode=cfg.beta_mode)
        eval_hist.append((int(ep_idx) + 1, rm))
        if cfg.verbose:
            print(f"[eval@{ep_idx + 1:6d}] 留出集窗口 RMSE[°] 电机/云台/小yaw = "
                  f"{rm['motor_deg']:.3f}/{rm['platform_deg']:.3f}/{rm['small_deg']:.3f}"
                  f"  角速度 = {rm['motor_rate']:.4f}/{rm['platform_rate']:.4f}/"
                  f"{rm['small_rate']:.4f}")

    if cfg.verbose:
        _grav = "带静态倾斜重力" if batch["has_gravity"] else "水平(重力=0)"
        print(f"[torch] {summary['recipe']}")
        print(f"[torch] 数据: {len(segs)} 段 → {W} 个 sample（最长 {L} 点 = {L*dt:.2f} s，"
              f"{_grav}）；拟合轴={cfg.fit_axis}，可学习参数={n_free}；参数分组: "
              f"{summary['param_space']}；参数限位: {summary['limits']}")
        print(f"[torch] β 来源（必需数据）: {beta_tag}；损失**不 wrap**（模型对圈数负责）")
        print("[torch] 初值 φ0 = " + _fmt_vec(phi0) + f"（来源: {summary['init_source']}）")
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
            cos_n = int(cfg.cos_decay_steps)
            sched = (None if cos_n > 0 else
                     (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
                      if cfg.lr_schedule == "cosine" else None))
            if cfg.verbose and cos_n > 0:
                print(f"[torch] ★ lr 调度: 前 {max(0, epochs - cos_n)} epoch 常数 lr="
                      f"{cfg.lr:g}，最后 {min(cos_n, epochs)} epoch 余弦衰减到 0")
            rng = np.random.RandomState(cfg.seed)
            print_every = max(1, int(cfg.print_every))
            _ck_n = int(cfg.checkpoint_every)
            _ck_pre = str(cfg.checkpoint_prefix or "")
            if _ck_n > 0 and not _ck_pre:
                print("[torch] [WARN] --checkpoint-every 给了但没给 --out ⇒ 不做断点保存")
            if _ck_n > 0 and _ck_pre:
                print(f"[torch] ★ 断点: 每 {_ck_n} epoch 写一份 "
                      f"{_ck_pre}.epXXXXXX（共约 {epochs // max(1, _ck_n)} 份）")

            def _maybe_checkpoint(ep: int) -> None:
                if _ck_n <= 0 or not _ck_pre or ep <= 0 or ep % _ck_n != 0:
                    return
                try:
                    write_params_file(f"{_ck_pre}.ep{ep:06d}", current_full(),
                                      recipe=f"checkpoint @ epoch {ep}/{epochs}")
                except Exception as exc:
                    print(f"[torch] [warn] checkpoint 写入失败: {exc}", file=sys.stderr)

            def _set_cos_lr(ep: int) -> None:
                if cos_n > 0:
                    _lr = lr_at_step(ep, epochs, cfg.lr, cos_n)
                    for _g2 in opt.param_groups:
                        _g2["lr"] = _lr

            if cfg.batch_segments:
                # ── ★ 全批: 每 epoch 对每段各抽 1 个 seg_steps 片段 → 拼成一个 batch
                #    做**一次** Adam 步（损失 = 各段损失的均值 = 原配方 W 个梯度的平均）──
                vw = torch.as_tensor(np.asarray(valid), dtype=torch.long, device=dev)
                nv = int(vw.numel())
                ar = torch.arange(nv, device=dev)
                lens_t = torch.as_tensor(np.asarray(lens, dtype=np.int64), device=dev)
                # `--batch-size=B` ⇒ 每 epoch 把段随机分成 ⌈N/B⌉ 组、每组一次 Adam 步
                bs = nv if cfg.batch_size <= 0 else min(nv, int(cfg.batch_size))
                n_group = int(np.ceil(nv / bs))
                if cfg.verbose:
                    print(f"[torch] ★ --batch-segments: 每 epoch {n_group} 次 Adam 步"
                          f"（每步 batch={bs}/{nv} 段 × {seg_steps} 步；损失 = 组内各段损失均值）")
                for ep in range(epochs):
                    _maybe_checkpoint(ep)
                    _set_cos_lr(ep)
                    ep_loss = 0.0
                    for _g in range(n_group):
                        sel = (ar if n_group == 1
                               else torch.from_numpy(rng.permutation(nv)[:bs]).to(dev))
                        wid = vw[sel]
                        nb = int(sel.numel())
                        # 起点上界: 还要再多取 1 个点当"状态 0" ⇒ lens − seg_steps − 1
                        hi = (lens_t[wid] - seg_steps - 1).clamp(min=0)
                        st = torch.floor(torch.rand(nb, device=dev)
                                         * (hi + 1).double()).long()
                        th_pred, dth_pred = _rollout_slices(wid, st, seg_steps + 1)
                        ii = st[:, None] + torch.arange(seg_steps + 1, device=dev)[None, :]
                        b_th = th_t[wid[:, None], ii]
                        b_dth = dth_t[wid[:, None], ii]
                        # ★ 损失本体在 loss.py（不 wrap、两项等权 / Huber）；这里按样本归约
                        per = _pair_loss(th_pred, dth_pred, b_th, b_dth,
                                         ax_w=w_axis_all[wid], reduce="per_sample")
                        loss = per.mean() / float(seg_steps + 1)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        if sched is not None:
                            sched.step()
                        ep_loss += float(loss.detach())
                        n_steps += 1
                    ep_loss /= max(1, n_group)
                    loss_hist.append(ep_loss)
                    param_hist.append(current_full().tolist())
                    if cfg.verbose and (ep < 5 or ep >= epochs - 5 or ep == epochs - 1
                                        or ep % print_every == 0):
                        print(f"[torch] epoch {ep:5d}/{epochs}  loss={ep_loss:.6e}  "
                              + _fmt_vec(current_full()))
                    _record_eval(ep)
            # ── 默认路径: 每 epoch 对每段各抽 1 个 seg_steps 片段做 1 个 Adam 步 ──
            if not cfg.batch_segments:
                for ep in range(epochs):
                    _maybe_checkpoint(ep)
                    _set_cos_lr(ep)
                    ep_loss, n_used = 0.0, 0
                    for w in valid:
                        s0 = int(rng.randint(0, lens[w] - seg_steps + 1))
                        opt.zero_grad(set_to_none=True)
                        loss = _slice_loss(w, s0, seg_steps)
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
                    loss_hist.append(ep_loss)
                    param_hist.append(current_full().tolist())
                    if cfg.verbose and (ep < 5 or ep >= epochs - 5 or ep == epochs - 1
                                        or ep % print_every == 0):
                        print(f"[torch] epoch {ep:5d}/{epochs}  loss={ep_loss:.6e}  "
                              + _fmt_vec(current_full()))
                    _record_eval(ep)

        elif cfg.verbose:
            print("[torch] epochs<=0 ⇒ 不优化，直接输出初值（只做数据/画图冒烟）")

        with torch.no_grad():
            final_loss = _window_loss()
        phi_final = current_full()
        recipe = ("原仓库配方(epochs×段数 个单片段 Adam 步)" if epochs > 0
                  else "未优化(epochs<=0)")
        if cfg.batch_segments and epochs > 0:
            recipe = (f"★ 全批配方(epochs={epochs} 个**全批** Adam 步, 每步 batch=全部段)"
                      f"—— 同一损失，只是把原配方的'每段一次更新'合并成'每 epoch 一次'")
    else:
        # ════════════════════════════════════════════════════════════════════
        # 旧精细配方（iters>0 才走这里）: 分窗 mini-batch Adam（+ 可选 LBFGS）
        # ════════════════════════════════════════════════════════════════════
        opt = torch.optim.Adam(params, lr=cfg.lr)
        cos_n = int(cfg.cos_decay_steps)
        sched = (None if cos_n > 0 else
                 (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.iters))
                  if cfg.lr_schedule == "cosine" else None))
        n_batch = W if cfg.batch_size <= 0 else min(W, int(cfg.batch_size))
        if cfg.verbose:
            print(f"[torch] 旧配方: 窗口数 W={W}（每窗 {L} 点 = {L*dt:.2f} s）, "
                  f"Adam 批={n_batch}/{W} 窗")
        for it in range(cfg.iters):
            if cos_n > 0:
                _lr = lr_at_step(it, cfg.iters, cfg.lr, cos_n)
                for _g2 in opt.param_groups:
                    _g2["lr"] = _lr
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
            param_hist.append(current_full().tolist())
            if cfg.verbose and (it % max(1, cfg.print_every) == 0 or it == cfg.iters - 1):
                print(f"[torch] adam {it:5d}  loss={loss_hist[-1]:.6e}  "
                      + _fmt_vec(current_full()))

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
                param_hist.append(current_full().tolist())
                if cfg.verbose:
                    print(f"[torch] lbfgs{it:5d}  loss={float(l.detach()):.6e}  "
                          + _fmt_vec(current_full()))

        with torch.no_grad():
            final_loss = _window_loss()
        phi_final = current_full()
        recipe = ("旧精细配方(%d 步 Adam%s%s)"
                  % (cfg.iters,
                     "" if cfg.lbfgs_iters <= 0 else f" + {cfg.lbfgs_iters} 步 LBFGS",
                     "" if sched is None else " + cosine"))

    truth = None if cfg.truth_vector is None else np.asarray(cfg.truth_vector, dtype=np.float64)
    return FitResult(phi=phi_final, phi0=phi0, loss_history=loss_hist,
                     val_loss=float(final_loss), n_iter=n_steps,
                     seconds=time.time() - t_start,
                     param_history=param_hist, epoch_losses=list(loss_hist),
                     n_free=n_free, n_steps=n_steps, recipe=recipe, truth=truth,
                     config=summary, eval_hist=eval_hist)


# ============================================================================
# 全批开环前向仿真误差（比 loss 好读；口径 = 整段/窗口 + 同一积分器）
# ============================================================================
def channel_rmse(segs, phi, base: PlanarParams, integrator: str = "rk4",
                 substeps: int = 4, window: int = 10, state_mode: str = "est",
                 beta_mode: str = "auto") -> dict:
    """用给定参数做开环前向仿真（numpy 参考实现），返回三通道的角度/角速度 RMSE（两套口径）。

    · ``*_deg`` / ``*_rate``（**窗口口径**）: 每 ``window`` 步（默认 10 步 = 0.1 s）从记录
      初值重新起跑一次，覆盖整段所有起点。这是"模型有多准"的**主指标**；
    · ``*_full_deg`` / ``*_full_rate``（**整段口径**）: 从段首一路开环跑到底。它同时含
      **模型误差的累积**与**初值偏差**，只能当参考。

    角度误差**不 wrap**（与损失口径一致：加载时已连续化，模型必须对圈数负责），用度；
    角速度用 rad/s。β 按 `state_mode` 对应的帧取（`beta_source`）。
    """
    p = base.with_vector(np.asarray(phi, dtype=np.float64))
    names = ("motor", "platform", "small")
    se_w = np.zeros(3); sv_w = np.zeros(3); n_w = 0
    se_f = np.zeros(3); sv_f = np.zeros(3); n_f = 0
    w = max(1, int(window))
    for s in segs:
        th_m, dth_m = state_arrays(s, state_mode)
        seq = None
        if s.gravity is not None:
            seq = [exo_from_gravity(float(s.gravity[i, 0]), float(s.gravity[i, 1]))
                   for i in range(s.T)]
        bs = beta_source(s, state_mode, beta_mode)[0]
        # 窗口口径
        for st in range(0, max(1, s.T - w), w):
            sl = slice(st, st + w + 1)
            th, dth = simulate_backlash_np(
                p, th_m[st], dth_m[st], s.tau[sl], s.dt, EXO_ZERO, substeps,
                exo_seq=(None if seq is None else seq[sl]),
                beta_seq=(None if bs is None else bs[sl]), integrator=integrator)
            d = th - th_m[sl]                                   # ★ 不 wrap（对圈数负责）
            se_w += np.sum(d * d, axis=0)
            sv_w += np.sum((dth - dth_m[sl]) ** 2, axis=0)
            n_w += int(th.shape[0])
        # 整段口径
        th, dth = simulate_backlash_np(p, th_m[0], dth_m[0], s.tau, s.dt,
                                       EXO_ZERO, substeps, exo_seq=seq, beta_seq=bs,
                                       integrator=integrator)
        d = th - th_m                                           # ★ 不 wrap
        se_f += np.sum(d * d, axis=0)
        sv_f += np.sum((dth - dth_m) ** 2, axis=0)
        n_f += int(th.shape[0])
    out = {}
    for k, nm in enumerate(names):
        out[nm + "_deg"] = float(np.degrees(np.sqrt(se_w[k] / max(1, n_w))))
        out[nm + "_rate"] = float(np.sqrt(sv_w[k] / max(1, n_w)))
        out[nm + "_full_deg"] = float(np.degrees(np.sqrt(se_f[k] / max(1, n_f))))
        out[nm + "_full_rate"] = float(np.sqrt(sv_f[k] / max(1, n_f)))
    return out


def _fmt_rmse(rm: dict) -> str:
    return ("窗口(0.1s) RMSE: 角度[°] 电机/云台/小yaw = "
            f"{rm['motor_deg']:.3f} / {rm['platform_deg']:.3f} / {rm['small_deg']:.3f}; "
            "角速度[rad/s] = "
            f"{rm['motor_rate']:.4f} / {rm['platform_rate']:.4f} / {rm['small_rate']:.4f}")


def _fmt_rmse_full(rm: dict) -> str:
    return ("整段开环 RMSE: 角度[°] 电机/云台/小yaw = "
            f"{rm['motor_full_deg']:.3f} / {rm['platform_full_deg']:.3f} / "
            f"{rm['small_full_deg']:.3f}")
