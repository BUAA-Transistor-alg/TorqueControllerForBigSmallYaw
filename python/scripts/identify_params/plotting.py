#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收敛曲线 / 轨迹对比 / 学习曲线（matplotlib，默认写文件；--show-plot 才交互显示）。"""

from __future__ import annotations

import math
import os

import numpy as np

from .model import simulate_backlash_np
from .params import (
    AXIS_BIG,
    AXIS_SMALL,
    EXO_ZERO,
    NPARAM,
    PARAM_NAMES,
    PARAM_UNITS,
    PlanarParams,
    Exo,
)
from .train import FitResult, beta_source


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
    """★ 图 1: loss（log 纵轴）+ 18 个参数各自的收敛曲线。

    每个参数子图上用**虚线**标出参考值：有真值（`--truth-params`）画真值，否则画初值 φ0。
    """
    plt = _lazy_pyplot(show_plot)
    hist = np.asarray(res.param_history if res.param_history else [res.phi], dtype=np.float64)
    if hist.ndim != 2:
        hist = hist.reshape(1, -1)
    losses = np.asarray(res.loss_history, dtype=np.float64)
    steps = np.arange(hist.shape[0])
    xlabel = "epoch" if not str(res.recipe).startswith("旧") else "Adam/LBFGS 步"

    n_panel = 1 + NPARAM                     # loss + 18 个参数
    ncol = 4
    nrow = int(math.ceil(n_panel / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.6 * nrow), squeeze=False)
    axes = axes.reshape(-1)
    fig.suptitle(f"参数辨识收敛曲线 —— {res.recipe}\n{title_note}".strip(), fontsize=11)

    ax = axes[0]
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
        a = axes[j + 1]
        a.plot(steps, hist[:, j], lw=1.2, color=f"C{j}")
        a.axhline(float(ref[j]), ls="--", lw=1.0, color="k", alpha=0.7,
                  label=f"{ref_label}={float(ref[j]):.4g}")
        a.set_xlabel(xlabel)
        a.set_title(f"{nm}  [{PARAM_UNITS[j]}]")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7, loc="best")
    for k in range(n_panel, axes.size):      # 关掉多余的子图
        axes[k].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out_path, show_plot)
    return out_path


def angle_rmse_deg(a, b):
    """角度 RMSE（**度**）。★ **不 wrap**：加载时已连续化，模型必须对圈数负责。"""
    d = np.asarray(a) - np.asarray(b)
    return float(np.degrees(np.sqrt(np.mean(d ** 2))))


def vel_rmse(a, b):
    """角速度 RMSE（**rad/s**）。"""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def plot_trajectory(res: FitResult, segs, base: PlanarParams, exo: Exo = EXO_ZERO,
                    out_path: str = "data/sysid/ident_torch_traj.png", show_plot: bool = False,
                    integrator: str = "rk4", state_mode: str = "est",
                    beta_mode: str = "auto") -> str:
    """★ 图 2: 实测 vs 模型前向（至少两段: 一段大 yaw 激励、一段小 yaw 激励）。

    行 = 数据段，列 = 角度 θ / 角速度 θ̇；
    每个子图画**三个通道**（电机 / 云台 / 小 yaw）的实测(实线)与仿真(虚线)，标题里标注 RMSE。
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
    has_beta = any(s.beta_true is not None for s in used)      # 仿真数据才画 β 那一列
    ncol = 3 if has_beta else 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    fig.suptitle("实测（仿真环境） vs 模型前向（参数辨识后，同一段力矩输入）", fontsize=11)

    for i, seg in enumerate(used):
        # 用记录力矩从**该段实测初值**前向仿真（与辨识/评测一致的积分器与 β 帧）
        q0 = np.asarray(seg.theta[0], dtype=np.float64)
        qd0 = np.asarray(seg.dtheta[0], dtype=np.float64)
        beta_seq = beta_source(seg, state_mode, beta_mode)[0]
        th_sim, dth_sim = simulate_backlash_np(p_model, q0, qd0, seg.tau, seg.dt, exo,
                                               substeps=4, integrator=integrator,
                                               beta_seq=beta_seq)
        t = np.arange(seg.T) * seg.dt
        axis_name = "大 yaw 激励段" if int(seg.axis) == AXIS_BIG else "小 yaw 激励段"
        for col, (meas, sim, lab, unit) in enumerate((
                (seg.theta, th_sim, "θ", "rad"),
                (seg.dtheta, dth_sim, "θ̇", "rad/s"))):
            a = axes[i][col]
            ann = []
            for k, (ax_name, ax_c) in enumerate((("电机", "C0"), ("云台", "C3"),
                                                 ("小 yaw", "C1"))):
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
        # ── 第 3 列: 背隙中心 β（真值 / 在线估计 / 观测极差中心）──
        if has_beta:
            a = axes[i][2]
            tt = t
            if seg.beta_true is not None:
                a.plot(tt, seg.beta_true, "k-", lw=1.3, label="β 真值（仿真环境）")
            if seg.beta is not None:
                a.plot(tt, seg.beta, "C3--", lw=1.3, label="β 在线估计（控制器用）")
                D = seg.theta[:, 0] - seg.theta[:, 1]
                obs = 0.5 * (D.max() + D.min())
                a.axhline(obs, color="C0", ls=":", lw=1.2,
                          label=f"本段 Δ 极差中心（观测上限）={obs:+.4f}")
            a.set_title(f"背隙中心 β（{axis_name}）: 只在穿越死区的段里可观测", fontsize=9)
            a.set_xlabel("t [s]")
            a.set_ylabel("β [rad]")
            a.grid(True, alpha=0.3)
            a.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, out_path, show_plot)
    return out_path


def plot_learning(res: FitResult, out_path: str, show_plot: bool = False) -> str:
    """★ 图 3: **留出集**开环 RMSE vs epoch（学习曲线）——回答"多训练有没有用"。"""
    if not res.eval_hist:
        raise ValueError("没有 eval_hist（要用 --eval-every>0 且给了 --val-data）")
    plt = _lazy_pyplot(show_plot)
    ep = np.array([e for e, _ in res.eval_hist], dtype=float)
    names = (("motor", "电机"), ("platform", "云台"), ("small", "小 yaw"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.suptitle("留出集开环误差 vs epoch（窗口 0.1 s）—— 学习曲线", fontsize=11)
    for ax, key, unit in ((axes[0], "_deg", "角度 RMSE [°]"),
                          (axes[1], "_rate", "角速度 RMSE [rad/s]")):
        for k, (en, cn) in enumerate(names):
            ax.plot(ep, [rm[en + key] for _, rm in res.eval_hist],
                    lw=1.4, color=f"C{k}", marker=".", ms=3, label=cn)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(unit)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
