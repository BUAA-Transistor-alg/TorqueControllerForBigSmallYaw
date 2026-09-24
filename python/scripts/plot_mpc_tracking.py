#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_mpc_tracking.py — 把 `tcbs_mpc_param_eval --dump=<前缀>` 导出的闭环轨迹画成图。

图的内容（每个场景一行）:
  · **目标轨迹 vs 控制轨迹**: 总瞄准角 `ref_aim`（目标）与 `aim = θ_platform + θ_small`（实际），
    以及大 yaw 平台角 `platform` 与其目标 `ref_big`；
  · 小 yaw 关节角 `small`；
  · 两轴力矩 `tau_big / tau_small`；
  · 背隙中心 β: 仿真真值 `beta_true` vs 控制器用的在线估计 `beta_hat`。

用法::

    ./build/tcbs_mpc_param_eval --plant-rigid --plant-beta-random=0.026 \
        --plant-beta-drift=0.0044 --phi=<拟合参数> --phi2=<背隙8参> --dump=_tmp/traj
    python3 python/scripts/plot_mpc_tracking.py --csv='_tmp/traj_*.csv' \
        --out=data/archive/.../fit/mpc_tracking.png
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_params.plotting import _lazy_pyplot, _save_fig          # noqa: E402

SCENARIO_NAMES = ("阶跃 0.6 rad", "大阶跃 1.2 rad", "正弦 0.3 rad @0.5Hz")


def _load(path):
    d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    if d is None or d.dtype.names is None or d.size == 0:
        raise ValueError(f"读不到轨迹: {path}")
    return {k: np.atleast_1d(d[k]).astype(float) for k in d.dtype.names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="闭环轨迹（目标 vs 控制）画图")
    ap.add_argument("--csv", required=True, help="轨迹 CSV glob（可逗号分隔多个）")
    ap.add_argument("--out", default="data/sysid/mpc_tracking.png", help="输出 PNG")
    ap.add_argument("--title", default="", help="图标题前缀（例如参数来源）")
    ap.add_argument("--show-plot", action="store_true", help="交互显示")
    a = ap.parse_args(argv)

    files = []
    for pat in a.csv.split(","):
        hit = sorted(globmod.glob(pat.strip()))
        if not hit and os.path.isfile(pat.strip()):
            hit = [pat.strip()]
        files.extend(hit)
    files = sorted(set(files))
    if not files:
        print(f"没有找到轨迹 CSV: {a.csv}", file=sys.stderr)
        return 1

    plt = _lazy_pyplot(a.show_plot)
    n = len(files)
    fig, axes = plt.subplots(n, 3, figsize=(17, 3.1 * n), squeeze=False)
    fig.suptitle(f"MPC 闭环: 目标轨迹 vs 控制轨迹 {a.title}".strip(), fontsize=11)
    print(f"{'场景':<16} {'最大误差(rad)':>12} {'RMS(rad)':>10} "
          f"{'β 估计误差(rad)':>14}")
    for i, f in enumerate(files):
        d = _load(f)
        t = d["t"]
        name = SCENARIO_NAMES[i] if i < len(SCENARIO_NAMES) else os.path.basename(f)
        err = d["ref_aim"] - d["aim"]
        print(f"{name:<16} {np.max(np.abs(err)):>12.4f} "
              f"{np.sqrt(np.mean(err ** 2)):>10.4f} "
              f"{np.max(np.abs(d['beta_true'] - d['beta_hat'])):>14.4f}")

        ax = axes[i][0]
        ax.plot(t, d["ref_aim"], "k--", lw=1.4, label="目标（总瞄准角 ref）")
        ax.plot(t, d["aim"], "C0", lw=1.3, label="实际（θ_平台+θ_小yaw）")
        ax.plot(t, d["ref_big"], "C2:", lw=1.2, label="大 yaw 目标 ref_big")
        ax.plot(t, d["platform"], "C2", lw=1.0, alpha=0.8, label="大 yaw 平台角")
        ax.set_title(f"{name}: 目标 vs 控制（max|e|={np.max(np.abs(err)):.4f}, "
                     f"RMS={np.sqrt(np.mean(err ** 2)):.4f}）", fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("角度 [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        ax = axes[i][1]
        ax.plot(t, d["small"], "C1", lw=1.2, label="小 yaw 关节角 θ_small")
        ax.plot(t, d["tau_big"], "C0", lw=1.0, label="τ_big（大 yaw 电机）")
        ax.plot(t, d["tau_small"], "C3", lw=1.0, label="τ_small")
        ax.set_title("小 yaw 关节角 / 两轴力矩", fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("rad / N·m")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        ax = axes[i][2]
        ax.plot(t, d["beta_true"], "k-", lw=1.3, label="β 真值（仿真环境）")
        ax.plot(t, d["beta_hat"], "C3--", lw=1.3, label="β 在线估计（控制器用）")
        ax.set_title(f"背隙中心 β: 真值 vs 在线估计（max|Δβ|="
                     f"{np.max(np.abs(d['beta_true'] - d['beta_hat'])):.4f} rad）", fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("β [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, a.out, a.show_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
