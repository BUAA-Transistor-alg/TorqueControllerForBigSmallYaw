#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_ident_compare.py — 把多份辨识日志画在一起（同图对比不同设置/轮数）。

日志就是 `identify_params_torch.py` 的标准输出（终端/重定向文件都行），
脚本按 `[torch] epoch   N/M  loss=... [+a +b ...]` 逐行解析。

用途举例:
  · 1000 epoch vs 10000 epoch 的参数轨迹；
  · "在线 β" vs "β 用真值" vs "状态用真值" 三组对照（看 δ/k/γ 收敛到哪里）。

用法::

    python3 python/scripts/plot_ident_compare.py \\
        --log=a.log,b.log,c.log --names="在线β,β真值,状态真值" \\
        --out=data/archive/.../fit10k/compare.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_params_torch import (PARAM_NAMES, PARAM_UNITS, _save_fig,   # noqa: E402
                                   _lazy_pyplot)

_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)\s+loss=([-\d.eE+]+)\s+\[([^\]]*)\]")


def parse_log(path):
    """→ (epochs [n], losses [n], params [n,16])"""
    ep, ls, ps = [], [], []
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            m = _EPOCH_RE.search(ln)
            if not m:
                continue
            vals = [float(v) for v in m.group(4).split()]
            if len(vals) != len(PARAM_NAMES):
                continue
            ep.append(int(m.group(1)))
            ls.append(float(m.group(3)))
            ps.append(vals)
    if not ep:
        raise ValueError(f"{path}: 没解析到任何 epoch 行")
    return np.array(ep), np.array(ls), np.array(ps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多份辨识日志对比画图")
    ap.add_argument("--log", required=True, help="日志路径（逗号分隔）")
    ap.add_argument("--names", default="", help="每份日志的图例名（逗号分隔；默认用文件名）")
    ap.add_argument("--out", required=True, help="输出 PNG")
    ap.add_argument("--which", default="backlash_delta,backlash_k,backlash_through,Jbig_eff,"
                                       "fc_big,fv_big,Jmotor",
                    help="要画哪些参数（逗号分隔的参数名）")
    ap.add_argument("--show-plot", action="store_true")
    a = ap.parse_args(argv)

    logs = [x.strip() for x in a.log.split(",") if x.strip()]
    names = ([x.strip() for x in a.names.split(",")]
             if a.names else [os.path.basename(x) for x in logs])
    while len(names) < len(logs):
        names.append(os.path.basename(logs[len(names)]))
    which = [x.strip() for x in a.which.split(",") if x.strip()]

    runs = []
    for p, nm in zip(logs, names):
        ep, ls, ps = parse_log(p)
        runs.append((nm, ep, ls, ps))
        print(f"{nm:>14s}: {len(ep)} 点, epoch {ep[0]}..{ep[-1]}  "
              f"loss {ls[0]:.4e}→{ls[-1]:.4e}")

    n = len(which) + 1
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    plt = _lazy_pyplot(a.show_plot)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 2.9 * nrow), squeeze=False)
    axes = axes.reshape(-1)
    fig.suptitle("辨识日志对比（同一批数据、不同设置/轮数）", fontsize=11)

    ax = axes[0]
    for nm, ep, ls, _ps in runs:
        ax.plot(ep, ls, lw=1.0, label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss（单片段，含噪）")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    for k, pname in enumerate(which):
        if pname not in PARAM_NAMES:
            continue
        j = PARAM_NAMES.index(pname)
        ax = axes[k + 1]
        for nm, ep, _ls, ps in runs:
            ax.plot(ep, ps[:, j], lw=1.1, label=nm)
        ax.set_xlabel("epoch")
        ax.set_title(f"{pname}  [{PARAM_UNITS[j]}]", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for k in range(n, axes.size):
        axes[k].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, a.out, a.show_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
