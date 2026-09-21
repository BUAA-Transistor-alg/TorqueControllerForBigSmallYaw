#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_openloop_compare.py — **开环**：用识别出来的参数做前向仿真 vs 实际采集曲线。

与 `identify_params_torch.py --plot-out=...` 生成的 `ident_traj.png` 的区别：
  · 这里可以一次画**多段**（默认 6 段，按"激励幅度最大"挑选，大小 yaw 各一半）；
  · 每段一行、三列：θ 叠加（实测实线 / 开环仿真虚线）+ θ 误差曲线 + θ̇ 叠加；
  · 每段标题里给出**窗口 0.1 s** 与**整段 3 s** 开环 RMSE，并打印成表。

"实测"= 采集脚本落盘的那几列（`theta_big_motor / theta_big_platform / theta_small`）；
"开环"= 只喂**记录的力矩**（`tau_big/tau_small`）与**该段实测初值**，之后不再有任何反馈。

用法::

    python3 python/scripts/plot_openloop_compare.py \
        --data='data/archive/.../fit4x/data/val/*.csv' \
        --params=data/archive/.../fit4x/params_A400.txt \
        --n=6 --out=data/archive/.../fit4x/openloop_vs_measured.png

    # 也可以用 --init-vector 直接给 18 个数（逗号分隔）
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_params_torch import (AXIS_BIG, AXIS_SMALL, EXO_ZERO, PARAM_NAMES,   # noqa: E402
                                   PlanarParams, _lazy_pyplot, _save_fig, channel_rmse,
                                   exo_from_gravity, load_segments, simulate_backlash_np,
                                   state_arrays)

CORE = ("Jbig_eff", "Js", "Px", "Py", "fc_big", "fv_big", "fc_small", "fv_small")
EXTRA = ("backlash_delta", "backlash_k", "backlash_c", "backlash_through",
         "Jmotor", "fc_motor", "fv_motor", "backlash_beta")
PB = ("Pbx", "Pby")          # ★ 大 yaw 侧一阶矩（追加在末尾；只随大 yaw 转）


def parse_params(path):
    """从 `--out` 写的参数文件里取 18 个数（按 PARAM_NAMES 顺序）。"""
    vals = {}
    for ln in open(path, errors="replace"):
        m = re.match(r"\s*([A-Za-z_]+)\s*=\s*([-+0-9.eE]+)", ln)
        if m:
            vals[m.group(1)] = float(m.group(2))
    order = CORE + EXTRA + PB
    missing = [k for k in order if k not in vals]
    if missing:
        raise SystemExit(f"[error] {path} 里缺少这些参数: {missing}")
    return np.array([vals[k] for k in order], dtype=np.float64)


def pick_segments(segs, n):
    """按激励幅度（平台/关节角相对自身均值的最大偏离）挑段，大小 yaw 各一半。"""
    def amp(s):
        th, _ = state_arrays(s, "est")
        return float(np.max(np.abs(th[:, 1] - np.mean(th[:, 1]))))
    big = sorted([s for s in segs if s.axis == AXIS_BIG], key=amp, reverse=True)
    sml = sorted([s for s in segs if s.axis == AXIS_SMALL], key=amp, reverse=True)
    out = []
    for i in range(max(1, n // 2)):
        if i < len(big):
            out.append(big[i])
        if i < len(sml):
            out.append(sml[i])
    return out[:max(1, n)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="开环仿真 vs 实际采集 曲线对比")
    ap.add_argument("--data", required=True, help="数据 glob（逗号分隔）")
    ap.add_argument("--params", default=None, help="辨识输出参数文件（--out 写的那个）")
    ap.add_argument("--init-vector", default=None, help="或直接给 18 个数（逗号分隔）")
    ap.add_argument("--n", type=int, default=6, help="画几段（默认 6，大小 yaw 各一半）")
    ap.add_argument("--substeps", type=int, default=2, help="开环仿真每控制步的子步（与训练一致）")
    ap.add_argument("--integrator", choices=["rk4", "euler"], default="rk4")
    ap.add_argument("--beta-mode", choices=["auto", "fit", "column", "true"], default="auto",
                    help="与训练一致：auto = 数据里有 backlash_center 列就逐样本用它")
    ap.add_argument("--state-mode", choices=["est", "true"], default="est")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.07)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--out", required=True, help="输出 PNG")
    ap.add_argument("--show-plot", action="store_true")
    a = ap.parse_args(argv)

    if bool(a.params) == bool(a.init_vector):
        raise SystemExit("[error] 必须且只能给一个: --params 或 --init-vector")
    phi = (parse_params(a.params) if a.params
           else np.array([float(x) for x in a.init_vector.replace(";", ",").split(",") if x.strip()],
                         dtype=np.float64))
    if phi.size != len(PARAM_NAMES):
        raise SystemExit(f"[error] 参数应为 {len(PARAM_NAMES)} 个，收到 {phi.size}")

    base = PlanarParams(dx=a.dx, dy=a.dy)
    p_model = base.with_vector(phi)
    segs = load_segments([a.data], dt_override=a.dt, verbose=False)
    if not segs:
        raise SystemExit("[error] 没读到数据")
    use_beta = any(s.beta is not None for s in segs) and a.beta_mode in ("auto", "column")
    sel = pick_segments(segs, a.n)

    print(f"参数: " + " ".join(f"{k}={v:+.5f}" for k, v in zip(PARAM_NAMES, phi)))
    print(f"逐样本 β: {'用数据列 backlash_center' if use_beta else '用常数 backlash_beta'}"
          f"；状态目标: {a.state_mode}；积分 {a.integrator}/substeps={a.substeps}")

    rows = []
    for s in sel:
        th_m, dth_m = state_arrays(s, a.state_mode)
        seq = None
        if s.gravity is not None:
            seq = [exo_from_gravity(float(s.gravity[i, 0]), float(s.gravity[i, 1]))
                   for i in range(s.T)]
        bs = s.beta if use_beta else None
        th, dth = simulate_backlash_np(p_model, th_m[0], dth_m[0], s.tau, s.dt, EXO_ZERO,
                                       a.substeps, exo_seq=seq, beta_seq=bs,
                                       integrator=a.integrator)
        win = channel_rmse([s], phi, base, integrator=a.integrator, substeps=a.substeps,
                           use_beta=use_beta, state_mode=a.state_mode)
        rows.append((s, th_m, dth_m, th, dth, win))

    print(f"\n{'段':<28} {'轴':<6} {'窗口 0.1s RMSE 电机/云台/小yaw':<32} "
          f"{'整段 RMSE 电机/云台/小yaw'}")
    for s, th_m, _d, th, _dd, win in rows:
        ax = "大yaw" if s.axis == AXIS_BIG else "小yaw"
        print(f"{os.path.basename(s.source)[:28]:<28} {ax:<6} "
              f"{win['motor_deg']:6.3f}/{win['platform_deg']:6.3f}/{win['small_deg']:6.3f}°"
              f"{'':<12} {win['motor_full_deg']:6.2f}/{win['platform_full_deg']:6.2f}/"
              f"{win['small_full_deg']:6.2f}°")

    plt = _lazy_pyplot(a.show_plot)
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, 3, figsize=(16.5, 3.2 * nrow), squeeze=False)
    fig.suptitle("开环仿真（识别参数） vs 实际采集曲线　—— 只喂记录的力矩与本段实测初值",
                 fontsize=11)
    chans = (("电机", "C0"), ("云台", "C3"), ("小 yaw", "C1"))
    for i, (s, th_m, dth_m, th, dth, win) in enumerate(rows):
        t = np.arange(s.T) * s.dt
        ax = axes[i][0]
        for k, (nm, c) in enumerate(chans):
            ax.plot(t, th_m[:, k], color=c, lw=1.2, label=f"实测 {nm}")
            ax.plot(t, th[:, k], color=c, lw=1.1, ls="--", alpha=0.9, label=f"开环 {nm}")
        ax.set_title(f"{'大yaw' if s.axis == AXIS_BIG else '小yaw'}激励段 "
                     f"（{os.path.basename(s.source)[:26]}）　窗口 RMSE="
                     + "/".join(f"{win[k]:.3f}" for k in
                                ("motor_deg", "platform_deg", "small_deg")) + "°",
                     fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("θ [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=3, loc="best")

        ax = axes[i][1]
        for k, (_nm, c) in enumerate(chans):
            ax.plot(t, th[:, k] - th_m[:, k], color=c, lw=1.1, label=f"误差 {_nm}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("开环误差 θ_sim − θ_实测", fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("误差 [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=3, loc="best")

        ax = axes[i][2]
        for k, (nm, c) in enumerate(chans):
            ax.plot(t, dth_m[:, k], color=c, lw=1.2, label=f"实测 {nm}")
            ax.plot(t, dth[:, k], color=c, lw=1.1, ls="--", alpha=0.9, label=f"开环 {nm}")
        ax.set_title("角速度 θ̇（rad/s）", fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("θ̇ [rad/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=3, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, a.out, a.show_plot)

    # 顺带把"角度 RMSE"和"角速度 RMSE"分别汇总一遍（便于贴报告）
    print("\n汇总（角度 ° / 角速度 rad/s，窗口）：")
    for k in ("motor", "platform", "small"):
        v = np.array([[r[5][k + "_deg"], r[5][k + "_rate"]] for r in rows])
        print(f"  {k:<9} 平均 {v[:, 0].mean():7.3f}°  /  {v[:, 1].mean():7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
