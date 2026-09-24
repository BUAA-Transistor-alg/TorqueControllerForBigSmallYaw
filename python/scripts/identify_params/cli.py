#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口。

两种等价调用方式::

    # ① 模块方式（推荐）
    cd python/scripts && python3 -m identify_params --data='../../data/sysid/*.npz'
    #    或在仓库根目录:
    PYTHONPATH=python/scripts python3 -m identify_params --data='data/sysid/*.npz'

    # ② 直接按文件路径运行（本文件里的引导会把父目录加进 sys.path）
    python3 python/scripts/identify_params/cli.py --data='data/sysid/*.npz'

常用示例::

    --fit-axis=big          # 只拟合大 yaw（小 yaw 摩擦固定）
    --selftest              # 模型自检（回归矩阵 / 分组映射 / torch vs numpy / 梯度）
"""

from __future__ import annotations

# ★ 允许**直接按文件路径**运行本文件（`python3 .../identify_params/cli.py`）:
#   此时 `__package__` 为空，下面的相对 import（`from .data import ...`）会抛
#   "attempted relative import with no known parent package"。这里把自己所在包的父目录
#   加进 sys.path 并把 `__package__` 指成包名，相对 import 即可正常解析。
if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "identify_params"

import argparse
import sys
import time

import numpy as np

from .data import HOLD_KEEP_SEC, load_segments, seg_fingerprint, truncate_hold_segments
from .params import FRICTION_LAMBDA, NPARAM, PlanarParams
from .plotting import plot_convergence, plot_learning, plot_trajectory, resolve_plot_paths
from .selftest import model_self_test
from .train import (
    FitConfig,
    _fmt_rmse,
    _fmt_rmse_full,
    channel_rmse,
    filter_beta_segments,
    fit_params_torch,
    format_header_snippet,
    format_param_table,
    loss_trend,
    write_params_file,
)

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _build_argparser():
    ap = argparse.ArgumentParser(
        description="平面 3-DOF（含大 yaw 背隙）18 参模型 —— PyTorch 可导前向仿真参数辨识（输出误差法）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", action="append", default=None,
                    help="数据 glob（可重复/逗号分隔）；默认 data/sysid/*.npz"
                         "（采集脚本默认只写 npz）。老 csv 数据仍可读: --data='.../*.csv'；"
                         "同名 npz+csv 同时命中时只读 npz 那一份")
    ap.add_argument("--fit-axis", choices=["both", "big", "small"], default="both",
                    help="both=三个状态通道（电机/云台/小 yaw）一起拟合；"
                         "big=只算电机+云台通道（小 yaw 摩擦固定）；"
                         "small=只算小 yaw 通道（大 yaw 惯量/摩擦 + 电机/背隙 8 参全固定）")
    g = ap.add_argument_group(
        "★ 默认配方（= 原仓库 TorqueController/python/scripts/param_ident.py 同款）",
        "epochs=1000；每 epoch 对**每个数据段**随机截取 seg_steps=10 步（0.1 s）片段做 1 个 "
        "Adam 步 ⇒ 总步数 = epochs × 段数；损失 = 角度误差 MSE（先 wrap 到 (−π,π]）+ 角速度 "
        "误差 MSE，**两项等权**；Adam lr=3e-4 **常数**、**无 LBFGS**；积分 = RK4，substeps=4；"
        "dt 取自数据；**参数无任何限位**（可学习参数按可取值范围分两组: 全体实数直接自由、"
        "正数 log 参数化；被冻结的参数=固定参数，不进任何一组）。")
    g.add_argument("--epochs", type=int, default=FitConfig.epochs,
                   help="★ 训练 epoch 数；**1000 = 与原仓库 num_epochs 相同**"
                        "（每 epoch 的 Adam 步数 = 数据段数；0 = 不优化只画图）")
    g.add_argument("--seg-steps", type=int, default=FitConfig.seg_steps,
                   help="★ 每个优化步随机截取的片段长度（原仓库 SEG_STEPS=10 步 = 0.1 s）")
    g.add_argument("--loss-mode", choices=["mse", "huber"], default=FitConfig.loss_mode,
                   help="★ mse = 角度wrap MSE + 角速度 MSE 等权（原仓库配方，默认）；"
                        "huber = 旧配方（Huber，可用 --vel-weight 加权）")
    g.add_argument("--integrator", choices=["euler", "rk4"], default=FitConfig.integrator,
                   help="★ 默认 rk4（3-DOF 背隙接触模态很硬: 同子步数下 euler 的数值角速度"
                        "误差可达 0.13 rad/s，会污染摩擦/k 的拟合）；euler = 原仓库同款消融")
    g.add_argument("--checkpoint-every", type=int, default=FitConfig.checkpoint_every,
                   help="★ 长跑断点: 每 N 个 epoch 把当前 18 参写一份到 `<--out>.epXXXXXX`"
                        "（格式与 --out 相同，可直接当 --params / --init-vector 用）。"
                        "0 = 关闭（默认）。**长跑强烈建议开**。")
    g.add_argument("--cos-decay-steps", type=int, default=FitConfig.cos_decay_steps,
                   help="★ **最后 N 步**用余弦把学习率从 `--lr` 衰减到 0（前面保持常数）。"
                        "典型: `--epochs=2000 --lr=1e-2 --cos-decay-steps=1000`。0 = 关闭。")
    g.add_argument("--lr-schedule", choices=["none", "cosine"], default=FitConfig.lr_schedule,
                   help="★ none = 常数学习率（原仓库没有 scheduler，默认）；cosine = 旧配方")
    ap.add_argument("--legacy-recipe", action="store_true",
                    help="★ 一键回到旧的精细配方: iters=400 + lr=5e-3 + Huber + 分窗"
                         "(window_len=150, 2 窗/段, batch=4) + LBFGS(25) + RK4 + cosine")
    ap.add_argument("--iters", type=int, default=FitConfig.iters,
                    help="旧配方的 Adam 步数；**0（默认）= 走新配方**；>0 = 精确跑 iters 个"
                         "整窗/分窗 mini-batch Adam 步（旧配方）")
    ap.add_argument("--lbfgs-iters", type=int, default=FitConfig.lbfgs_iters,
                    help="LBFGS 迭代数（0=关闭 ⇒ 原仓库配方没有 LBFGS）")
    ap.add_argument("--lr", type=float, default=FitConfig.lr,
                    help="Adam 学习率（★ 新配方默认 3e-4，与原仓库一致；旧配方用 5e-3）")
    ap.add_argument("--seed", type=int, default=FitConfig.seed)
    ap.add_argument("--substeps", type=int, default=FitConfig.substeps,
                    help="可导前向仿真每个控制步的积分子步数（★ 3-DOF 默认 4: 背隙接触模态 "
                         "ω=√(k/μ_red)≈190 rad/s，10 ms 一步已到显式稳定边界）")
    ap.add_argument("--huber-delta", type=float, default=FitConfig.huber_delta,
                    help="角度 Huber 阈值 (rad)（仅 --loss-mode=huber 时生效）")
    ap.add_argument("--vel-weight", type=float, default=FitConfig.vel_weight,
                    help="角速度误差项权重（仅 --loss-mode=huber 有效；mse 模式固定等权 1.0）")
    ap.add_argument("--free-init-vel", action="store_true",
                    help="把每段初始角速度也作为自由参数优化")
    ap.add_argument("--p-bound", type=float, default=FitConfig.p_bound,
                    help="★ 已废弃并被忽略（本脚本不再有任何参数限位；仅为 CLI 兼容保留）")
    ap.add_argument("--init-vector", type=str, default=None,
                    help=f"逗号分隔的 {NPARAM} 个初值，**整体替换**默认初值。"
                         f"不给就用参数表 `PARAM_SPECS` 里的初值（初值只有那一处定义；"
                         f"CLI 不再提供按参数的初值开关）")
    ap.add_argument("--truth-params", type=str, default=None,
                    help=f"逗号分隔的 {NPARAM} 个真值/参考值（收敛曲线上的虚线；默认画初值 φ0）")
    ap.add_argument("--dx", type=float, default=0.0,
                    help="实测几何 dx (m)，默认 0（两轴横向无偏置）")
    ap.add_argument("--dy", type=float, default=0.07,
                    help="实测几何 dy (m)，默认 0.07（小 yaw 轴在大 yaw 轴前方 0.07 m）")
    ap.add_argument("--model-lambda", type=float, default=FRICTION_LAMBDA,
                    help="★ 辨识模型的摩擦软符号陡度 λ；默认 100 = 本仓库约定（与前向仿真/"
                         "MPC/planar_yaw_model.h 一致）。**原仓库单 yaw 版用 1e4**，本仓库不能"
                         "照搬；想复现 1e4 可传 --model-lambda=1e4 消融")
    ap.add_argument("--dt", type=float, default=None, help="覆盖 dt（默认取数据里的）")
    ap.add_argument("--max-points", type=int, default=0, help="每段只用前 N 点（加速调试）")
    ap.add_argument("--window-len", type=int, default=FitConfig.window_len,
                    help="每个窗口的点数（0=整段一个窗口；**仅旧配方**用，新配方忽略）")
    ap.add_argument("--windows-per-seg", type=int, default=FitConfig.windows_per_seg,
                    help="每段切几个窗口（仅旧配方用，新配方忽略）")
    ap.add_argument("--batch-size", type=int, default=FitConfig.batch_size,
                    help="Adam 每步随机抽几个窗口（0=全批；仅旧配方用，新配方忽略）")
    ap.add_argument("--fix-p", action="store_true",
                    help="把 Px/Py 固定为 0（等价 --p-constraint=zero）")
    ap.add_argument("--p-constraint", choices=["free", "along_d", "along-d", "zero"],
                    default="free",
                    help="Px/Py 的处理: free=各自独立；along_d=零点标定后 P=|P|·R(−θ*)·d̂"
                         "（两组之外的一个派生标量）；zero=固定为 0")
    ap.add_argument("--p-zero-angle", type=float, default=0.0,
                    help="平衡点 θ* 在当前零点坐标系里的读数（度）。仅 along_d 时有效")
    # ★ 初值只有一处管理（参数表 PARAM_SPECS 的 default）；CLI 只提供"整体替换"的 --init-vector，
    #   不再有按参数的初值开关（--backlash-* / --j*-motor 已删除）。
    ap.add_argument("--freeze-backlash-through", action=argparse.BooleanOptionalAction,
                    default=FitConfig.freeze_backlash_through,
                    help="★ **默认开**: 把背隙直通项 γ（参数 11）固定（移出可学习组）在初值上。"
                         "它的定位只是死区内的梯度引导；放开拟合会让它替模型去'填'刚性接触的"
                         "台阶（γ→0.3、δ→0.25 rad，参数失去物理意义）。"
                         "要用 --no-freeze-backlash-through 复现'γ 自由'的消融")
    ap.add_argument("--beta-mode", choices=["auto", "column", "true"],
                    default=FitConfig.beta_mode,
                    help="★ 背隙死区中心 β 的来源（**必需数据**；`fit` 已移除）: "
                         "auto（默认）= 与拟合帧匹配（est 帧用 `backlash_center`、true 帧用 "
                         "`beta_true`）；column = 强制估计帧 `backlash_center`；"
                         "true = 强制真值帧 `beta_true`（只用于仿真数据）。"
                         "加载时会先把 β 以 2π 为单位对齐、使 Δ=θm−θp−β 落进 (−π,π]，"
                         "再校验其相邻差（>π 会询问是否跳过该段）；缺列的段直接报错")
    ap.add_argument("--batch-segments", action="store_true",
                    help="★ 全批加速: 每 epoch 仍对每段随机抽 1 个 seg_steps 片段，但把所有段"
                         "拼成一个 batch 做**一次** Adam 步（损失 = 各段损失均值）。"
                         "段数多（几十~上百）时逐段更新慢得跑不动，用它")
    ap.add_argument("--eval-every", type=int, default=FitConfig.eval_every,
                    help="★ 每 N 个 epoch 在**留出集**（--val-data）上算一次开环 RMSE，"
                         "最后画成学习曲线（0 = 关，默认）")
    ap.add_argument("--state-mode", choices=["est", "true"], default=FitConfig.state_mode,
                    help="★ 状态目标来源: est（默认）= 记录/估计值；true = 仿真真值列"
                         "`theta_true_*`（**只有 dry-run 数据有**）——上限对照")
    ap.add_argument("--eval-max-segs", type=int, default=0,
                    help="★ **只限制评测/学习曲线用的段数**（默认 0 = 用全部）：训练集不受影响。"
                         "全量开环评测是大头（240 段 ≈ 3 min、一行都不打印）；给 40 ⇒ 约 30 s；"
                         "抽样是确定性的（按 --seed）")
    ap.add_argument("--hold-max-sec", type=float, default=HOLD_KEEP_SEC,
                    help="★ **保持段**（文件名带 `_hold` 后缀）只取前 N 秒（默认 3.0）；"
                         "**普通收集段不截断**；0 = 不截断（老行为）")
    ap.add_argument("--val-data", type=str, default=None,
                    help="留出（测试）数据 glob: 只用于**评估与画图**，不参与训练")
    ap.add_argument("--freeze-params", type=str, default=None,
                    help="额外固定的参数下标（逗号分隔，编号见输出表）⇒ 移出两组可学习参数。"
                         "典型用法: ① 无倾角且两轴从不同时激励时 Px/Py 没有可观测量 ⇒ "
                         "--freeze-params=2,3；② 把背隙直通项钉成固定小值 ⇒ --freeze-params=11")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 线程数（0=默认；小张量下 1 通常最快）")
    ap.add_argument("--print-every", type=int, default=FitConfig.print_every)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None,
                    help="可选：把辨识结果写成文本（默认只打印，不建文件）")
    pg = ap.add_argument_group("收敛曲线 / 轨迹对比（matplotlib）")
    pg.add_argument("--plot-out", type=str, default=None,
                    help="输出路径/前缀（默认 data/sysid/ident_torch）：以 .png 结尾 ⇒ 收敛曲线用"
                         "该文件、轨迹用 <名>_traj.png；给目录 ⇒ 目录下 ident_torch_*.png；"
                         "否则当前缀 ⇒ <前缀>_convergence.png / <前缀>_traj.png")
    pg.add_argument("--no-plot", action="store_true", help="不画图、不写 PNG")
    pg.add_argument("--show-plot", action="store_true",
                    help="交互显示（默认用 Agg 后端只写文件，无显示环境下不报错）")
    ap.add_argument("--selftest", action="store_true", help="只跑模型自检")
    return ap


def _parse_vecN(s, what: str, n: int = NPARAM) -> np.ndarray:
    """解析逗号分隔的 n 个数（--init-vector / --truth-params）。"""
    v = np.array([float(x) for x in str(s).replace(";", ",").split(",") if x.strip() != ""],
                 dtype=np.float64)
    if v.size != n:
        raise SystemExit(f"[error] {what} 需要 {n} 个数，收到 {v.size} 个")
    return v


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    base = PlanarParams(dx=args.dx, dy=args.dy, friction_lambda=args.model_lambda)

    if args.selftest:
        return 0 if model_self_test() else 1

    if torch is None:
        print(f"[error] 需要 torch: {_TORCH_IMPORT_ERROR}", file=sys.stderr)
        return 2

    patterns = args.data or ["data/sysid/*.npz"]
    # ★ 辨识初值只有一处管理（`params.PARAM_SPECS` 的 `default`，经 `default_param_vector()`）:
    #   这里**不**组装/覆盖初值；`init_vector=None` ⇒ 训练入口自己取参数表的初值。
    #   要自定义就给 `--init-vector`（18 个数，整体替换）。
    segs = load_segments(patterns, dt_override=args.dt)
    segs = truncate_hold_segments(segs, args.hold_max_sec)
    if not segs:
        print("[error] 没有读到数据；请用 --data=<glob> 指定（或先跑采集脚本）",
              file=sys.stderr)
        return 2
    # ── 留出（测试）数据: 只评估/画图，不训练 ──
    val_segs = None
    if args.val_data:
        val_segs = load_segments([args.val_data], dt_override=args.dt)
        val_segs = truncate_hold_segments(val_segs, args.hold_max_sec)
        if not val_segs:
            print(f"[error] --val-data={args.val_data} 没读到数据", file=sys.stderr)
            return 2
    # ── ★ 评测段数上限（--eval-max-segs，只影响评测/画图/学习曲线；确定性子集）──
    def _cap_eval(lst):
        n_cap = int(getattr(args, "eval_max_segs", 0) or 0)
        if not lst or n_cap <= 0 or len(lst) <= n_cap:
            return lst
        pick = np.sort(np.random.RandomState(int(args.seed)).choice(len(lst), size=n_cap,
                                                                   replace=False))
        print(f"[eval] --eval-max-segs={n_cap}: 评测集用 {n_cap}/{len(lst)} 段"
              f"（确定性子集，训练集不变）", flush=True)
        return [lst[int(i)] for i in pick]
    val_segs = _cap_eval(val_segs)

    # ── ★ β 是必需数据: 缺 β 的段用 [Y/n] 提示跳过（与 β 连续性校验同一套；非交互默认跳过）──
    segs = filter_beta_segments(segs, args.state_mode, args.beta_mode, "训练集")
    if not segs:
        print("[error] 没有可用的数据段（所有段都因缺 β 被跳过）", file=sys.stderr)
        return 2
    val_segs = (filter_beta_segments(val_segs, args.state_mode, args.beta_mode, "留出集")
                if val_segs else val_segs)
    if args.val_data and not val_segs:
        print("[error] 留出集的段都因缺 β 被跳过", file=sys.stderr)
        return 2

    # ── ★ 训练/留出重合检查（防止"留出集"其实是训练集的一部分）──
    if val_segs:
        tr_fp = {seg_fingerprint(sg) for sg in segs}
        dup = sum(1 for sg in val_segs if seg_fingerprint(sg) in tr_fp)
        if dup:
            print(f"[warn] ★★ 留出集里有 {dup}/{len(val_segs)} 段与训练集**完全相同**！"
                  "原因几乎肯定是采集脚本的 `--seed` 相同（默认 42，采集是确定性的）"
                  "⇒ 这个留出集只能当「拟合误差」看，**不能**当泛化误差；"
                  "请用不同的 `--seed` 重采留出集（训练与留出的 seed 必须不同）。")
        else:
            print(f"[ok] 留出集与训练集无重合（{len(val_segs)} 段，指纹比对）")

    # ── 处理后的 β（必需数据；真实来源由 fit 里的 beta_source 按拟合帧决定）──
    _frame_true = (str(args.state_mode) == "true")
    _bsrc = [(sg.beta_true if _frame_true else sg.beta) for sg in segs]
    _bsrc = [b for b in _bsrc if b is not None]
    n_beta = len(_bsrc)
    if n_beta:
        bvals = np.concatenate(_bsrc)
        print(f"[beta] {n_beta}/{len(segs)} 段有 β（拟合帧 = {args.state_mode}，"
              f"mode={args.beta_mode}）: 处理后范围 "
              f"[{bvals.min():+.4f}, {bvals.max():+.4f}] rad")
    # Δ = θm − θp − β 必须落在 (−π,π]，否则去 wrap 的 loss 会炸
    _dn = float(np.abs(np.concatenate([sg.theta[:, 0] - sg.theta[:, 1] - sg.beta
                                       for sg in segs if sg.beta is not None])).max()) \
        if any(sg.beta is not None for sg in segs) else float("nan")
    if np.isfinite(_dn):
        print(f"[beta] 处理后 |Δ=θm−θp−β| 全局最大 = {_dn:.4f} rad（须 ≤ π：模型看到的是物理量级）")

    cfg = FitConfig(fit_axis=args.fit_axis, iters=args.iters, lbfgs_iters=args.lbfgs_iters,
                    lr=args.lr, seed=args.seed, substeps=args.substeps,
                    huber_delta=args.huber_delta, vel_weight=args.vel_weight,
                    free_init_vel=args.free_init_vel, p_bound=args.p_bound,
                    epochs=args.epochs, seg_steps=args.seg_steps, loss_mode=args.loss_mode,
                    integrator=args.integrator, lr_schedule=args.lr_schedule,
                    cos_decay_steps=args.cos_decay_steps,
                    checkpoint_every=args.checkpoint_every, checkpoint_prefix=(args.out or ""),
                    init_vector=(None if args.init_vector is None
                                 else _parse_vecN(args.init_vector, "--init-vector")),
                    truth_vector=(None if args.truth_params is None
                                  else _parse_vecN(args.truth_params, "--truth-params")),
                    freeze_params=(() if not args.freeze_params
                                   else tuple(int(x) for x in
                                              str(args.freeze_params).split(",") if x != "")),
                    print_every=args.print_every, device=args.device,
                    max_points=args.max_points, window_len=args.window_len,
                    windows_per_seg=args.windows_per_seg, batch_size=args.batch_size,
                    p_constraint=args.p_constraint, fix_p=args.fix_p,
                    p_zero_angle_deg=args.p_zero_angle,
                    beta_mode=args.beta_mode, batch_segments=args.batch_segments,
                    freeze_backlash_through=args.freeze_backlash_through,
                    state_mode=args.state_mode, eval_every=args.eval_every,
                    eval_segs=val_segs)
    if args.legacy_recipe:
        cfg.legacy_recipe()
        print("[cfg] ★ --legacy-recipe: 已切到旧精细配方（iters=400, lr=5e-3, Huber, 分窗, "
              "LBFGS=25, RK4, cosine）")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    res = fit_params_torch(segs, cfg, base)

    print("\n" + "=" * 78)
    print(f"辨识结果（输出误差法, 段数={len(segs)}, 轴={cfg.fit_axis}）  用时 {res.seconds:.1f}s")
    print(f"配方: {res.recipe}；Adam/总步数={res.n_steps}；参数限位={res.config['limits']}")
    print(f"可学习参数分组: {res.config['param_space']}")
    for _ln in format_param_table(res.phi, res.phi0):
        print(_ln)
    print("=" * 78)
    # ── ★ 全批开环前向仿真误差（比 loss 好读；口径 = 整段 + 同一起点 + 同一积分器）──
    eval_segs = val_segs if val_segs is not None else segs
    eval_segs = _cap_eval(eval_segs)
    tag = "留出集" if val_segs is not None else "训练集"
    print(f"[eval] 开环评测 {len(eval_segs)} 段（窗口 0.1 s + 整段）× 2 遍（估计参数 / 初值对照）…"
          f"  240 段约 3 min —— **这里会静默一会儿，不是卡死**；"
          f"想快就用 --eval-max-segs", flush=True)
    _t0 = time.time()
    rm = channel_rmse(eval_segs, res.phi, base, integrator=cfg.integrator, substeps=cfg.substeps,
                      state_mode=cfg.state_mode, beta_mode=cfg.beta_mode)
    _t1 = time.time()
    rm0 = channel_rmse(eval_segs, res.phi0, base, integrator=cfg.integrator,
                       substeps=cfg.substeps, state_mode=cfg.state_mode,
                       beta_mode=cfg.beta_mode)
    print(f"[eval] 完成: 估计参数 {_t1 - _t0:.1f}s + 初值对照 {time.time() - _t1:.1f}s",
          flush=True)
    print(f"全批前向仿真 val_loss = {res.val_loss:.6e}")
    print(f"  ★ 估计参数（{tag}）: {_fmt_rmse(rm)}")
    print(f"               {_fmt_rmse_full(rm)}")
    print(f"    （对照）初值: {_fmt_rmse(rm0)}")

    # ── 收敛摘要: 前 5 / 后 5 个 loss ──
    h = res.loss_history
    if h:
        n5 = min(5, len(h))
        print(f"loss 曲线: 共 {len(h)} 点；前 {n5} 个 = "
              + ", ".join(f"{v:.6e}" for v in h[:n5]))
        print(f"           后 {n5} 个 = " + ", ".join(f"{v:.6e}" for v in h[-n5:]))
        tr = loss_trend(h)
        print(f"            前 {n5} 均值={tr['head']:.6e} → 后 {n5} 均值={tr['tail']:.6e}"
              f"（比值 {tr['ratio']:.4f}，最小值 {tr['min']:.6e}）"
              f"  ⇒ {'下降 ✓' if tr['decreased'] else '未下降 ✗'}")
    print("可粘贴到 include/tcbs/mpc/planar_yaw_params.h:")
    for _ln in format_header_snippet(res.phi):
        print(_ln)

    # ── 留出集学习曲线（--eval-every）──
    if res.eval_hist:
        print(f"留出集学习曲线（{len(res.eval_hist)} 个点，epoch "
              f"{res.eval_hist[0][0]}..{res.eval_hist[-1][0]}）:")
        step = max(1, len(res.eval_hist) // 10)
        for e, r in res.eval_hist[::step] + ([res.eval_hist[-1]]
                                             if (len(res.eval_hist) - 1) % step else []):
            print(f"  epoch {e:6d}: 角度 RMSE[°] {r['motor_deg']:.3f}/{r['platform_deg']:.3f}"
                  f"/{r['small_deg']:.3f}   角速度 {r['motor_rate']:.4f}/"
                  f"{r['platform_rate']:.4f}/{r['small_rate']:.4f}")

    # ── 收敛曲线 / 轨迹对比（默认写 PNG；无显示环境也不报错）──
    if not args.no_plot:
        conv_png, traj_png = resolve_plot_paths(args.plot_out)
        note = (f"段数={len(segs)}, 轴={cfg.fit_axis}, 步数={res.n_steps}, "
                f"lr={cfg.lr:g}, 损失={cfg.loss_mode}, 积分={cfg.integrator}")
        try:
            plot_convergence(res, conv_png, show_plot=args.show_plot, title_note=note)
            plot_trajectory(res, eval_segs, base, integrator=cfg.integrator,
                            out_path=traj_png, show_plot=args.show_plot,
                            state_mode=cfg.state_mode, beta_mode=cfg.beta_mode)
            if res.eval_hist:
                learn_png = conv_png[:-4] + "_learning.png"
                plot_learning(res, learn_png, show_plot=args.show_plot)
        except Exception as exc:                    # 画图失败不应让辨识结果丢失
            print(f"[plot][warn] 画图失败（辨识结果仍然有效）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    if args.out:
        write_params_file(args.out, res.phi, recipe=res.recipe)
        print(f"[out] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
