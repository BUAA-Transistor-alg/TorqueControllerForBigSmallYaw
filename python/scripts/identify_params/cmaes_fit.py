#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 **CMA-ES**（`cmaes` 包）做参数辨识 —— 无梯度优化，前向可选手写 C++ / numpy / torch。

设计要点
--------
* **损失只有一处实现**：:mod:`identify_params.loss`（角度 MSE 不 wrap + 角速度 MSE 等权）。
* **前向**（``--forward``）:
    · ``cpp``   手写 C++（分块向量化 + std::thread + ctypes，见 ``planar2_sim/``）—— 最快；
    · ``numpy`` :func:`~identify_params.model.rollout_batched_np`（按 batch 向量化）；
    · ``torch`` :已有的可微模型 :class:`~identify_params.model.DifferentiableSimulator`；
    · ``auto``  依次尝试 cpp → numpy → torch。
* **无梯度**：全程 ``torch.set_grad_enabled(False)``，目标函数内部再套一层 ``torch.no_grad()``。
* **目标确定性**：固定窗口 + mask（没有随机片段）—— 否则目标带噪声、CMA-ES 会退化。
* 数据打包 / β 必需校验 / 固定参数 / 参数化（实数 + 正数 log）与 Adam 训练器**共用**
  :func:`~identify_params.train.build_fit_context`，所以两条路径语义完全一致。

需要安装::

    pip install cmaes

用法::

    cd python/scripts && python3 -m identify_params.cmaes_fit \
        --data='../../data/cars/Sentry1/sysid/*.npz' --generations=2000 \
        --window-len=100 --windows-per-seg=1 --max-segs=200 \
        --forward=cpp --threads=8 --out=../../data/cars/Sentry1/ident/params_cmaes.txt
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# ★ 允许直接按文件路径运行（与 cli.py 同一套引导）
if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "identify_params"

from .data import HOLD_KEEP_SEC, load_segments, truncate_hold_segments
from .loss import WindowObjective
from .model import rollout_batched_np
from .params import (EXO_ZERO, FRICTION_LAMBDA, NPARAM, TAU_SIGN_BIG_DEFAULT,
                     TAU_SIGN_SMALL_DEFAULT, PlanarParams, set_tau_sign,
                     tau_sign_desc)
from .plotting import plot_convergence, plot_trajectory, resolve_plot_paths
from .train import (
    FitConfig,
    FitResult,
    _fmt_rmse,
    _fmt_rmse_full,
    build_fit_context,
    channel_rmse,
    format_header_snippet,
    format_param_table,
    write_params_file,
)

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# ============================================================================
# 前向后端
# ============================================================================
def _make_forward(name: str, ctx, exo, base_phi: PlanarParams, args):
    """``forward_fn(raw_np) -> (θ[B,T,2], θ̇[B,T,2])``；raw = 可学习参数的 raw 空间取值。"""
    layout = ctx.layout
    W = ctx.W
    # seq_const/seq_var 已在 build_fit_context 里按新契约打包好（含逐段 ω_c）
    sc_np = np.ascontiguousarray(ctx.seq_const.cpu().numpy(), dtype=np.float64)
    sv_np = np.ascontiguousarray(ctx.seq_var.cpu().numpy(), dtype=np.float64)

    def _phys(raw):
        """raw → 完整 11 维物理参数（辨识参数顺序 = C++/numpy 的参数顺序）。"""
        with torch.no_grad():
            phys = layout.to_physical(torch.as_tensor(np.asarray(raw, dtype=np.float64)))
        return layout.full_vector(np.asarray(phys.detach().cpu().numpy(), dtype=np.float64))

    order = ["cpp", "numpy", "torch"] if name == "auto" else [name]
    reasons = []
    for backend in order:
        if backend == "cpp":
            try:
                from .planar2_sim import FastSimulator
                fs = FastSimulator(base_phi, dt=ctx.dt, substeps=args.substeps,
                                   integrator=args.integrator, nthreads=args.nthreads,
                                   block=args.block)
                p_buf = np.empty((W, NPARAM), dtype=np.float64)

                def fwd_cpp(raw, _fs=fs, _buf=p_buf, _sc=sc_np, _sv=sv_np):
                    _buf[:] = _phys(raw)
                    return _fs.rollout(_buf, _sc, _sv)
                return fwd_cpp, (f"手写 C++ planar2_sim（block={fs.block}, nthreads="
                                 f"{fs.nthreads or '默认'}, {fs.substeps} 子步）")
            except Exception as exc:
                reasons.append(f"cpp 不可用（{type(exc).__name__}: {exc}）")
                continue
        if backend == "numpy":
            def fwd_np(raw, _sc=sc_np, _sv=sv_np):
                # numpy 版一批共用一组参数 ⇒ 用候选的物理参数构造 PlanarParams
                return rollout_batched_np(base_phi.with_vector(_phys(raw)),
                                          _sc, _sv, ctx.dt, args.substeps, args.integrator)
            return fwd_np, "numpy 批量向量化（rollout_batched_np）"
        if backend == "torch":
            # ★ 实测: 这个前向是「T 次小张量 op」的 Python 循环，多线程（≥8）反而**严重变慢**
            #   （18 线程时 W=1024 从 ~0.36 s 涨到 ~4.7 s）⇒ 默认钉 1 线程，要改显式 --threads=N。
            if args.threads <= 0:
                torch.set_num_threads(1)
            else:
                torch.set_num_threads(int(args.threads))
            sim = ctx.sim

            def fwd_torch(raw):
                with torch.no_grad():
                    phys = layout.to_physical(
                        torch.as_tensor(np.asarray(raw, dtype=np.float64)))
                    return sim(phys.unsqueeze(0).expand(W, -1), ctx.seq_const, ctx.seq_var)
            return fwd_torch, f"已有 torch 可微模型（no_grad；{args.integrator}）"
        reasons.append(f"未知后端 {backend!r}")
    raise SystemExit("[error] 没有可用的前向后端: " + "; ".join(reasons))


# ============================================================================
# CLI
# ============================================================================
def _build_argparser():
    ap = argparse.ArgumentParser(
        description="CMA-ES 参数辨识（无梯度；前向可用手写 C++ / numpy / torch）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", action="append", default=None,
                    help="数据 glob（可重复/逗号分隔）；默认 data/sysid/*.npz")
    ap.add_argument("--val-data", type=str, default=None, help="留出数据（只评估/画图）")
    ap.add_argument("--hold-max-sec", type=float, default=HOLD_KEEP_SEC,
                    help="保持段只取前 N 秒（0 = 不截断）")
    ap.add_argument("--dt", type=float, default=None, help="覆盖 dt（默认取数据里的）")
    ap.add_argument("--max-points", type=int, default=0, help="每段只用前 N 点")
    ap.add_argument("--max-segs", type=int, default=0,
                    help="★ 目标函数只用 N 段（按 --seed 确定性抽样）。**默认 0 = 用全量数据**"
                         "（稳定 loss，不要再抽片段；抽样会让目标不稳、易被退化解钻空子）")
    ap.add_argument("--eval-max-segs", type=int, default=0,
                    help="最终评测/画图用 N 段（0 = 全部；不影响优化目标）")
    # ── 目标函数的窗口口径（确定性）──
    ap.add_argument("--window-len", type=int, default=0,
                    help="★ 目标里每个窗口的点数。**默认 0 = 整段 = 全量数据**"
                         "（确定性、无随机片段）；>0 才切成固定窗口")
    ap.add_argument("--windows-per-seg", type=int, default=1,
                    help="每段取几个**等间隔**窗口（仅在 --window-len>0 时有意义；"
                         "=1 取段首。默认配合 --window-len=0 用整段）")
    # ── 与 Adam 训练器一致的选项 ──
    ap.add_argument("--fit-axis", choices=["both", "big", "small"], default="both")
    ap.add_argument("--freeze-params", type=str, default=None,
                    help="额外固定的参数下标（逗号分隔）")
    ap.add_argument("--fix-mu", action="store_true",
                    help="★ 把 μ（规范自由度）钉在初值上；默认不固定")
    ap.add_argument("--p-constraint", choices=["free", "along_d", "along-d", "zero"],
                    default="free")
    ap.add_argument("--p-zero-angle", type=float, default=0.0)
    ap.add_argument("--state-mode", choices=["est", "true"], default="est")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--integrator", choices=["rk4", "euler"], default="rk4",
                    help="★ 注意: cpp 后端只实现了 rk4")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.07)
    ap.add_argument("--model-lambda", type=float, default=FRICTION_LAMBDA)
    ap.add_argument("--init-vector", type=str, default=None,
                    help=f"逗号分隔的 {NPARAM} 个初值（= CMA-ES 的初始均值）；"
                         f"不给就用参数表 PARAM_SPECS 的初值")
    ap.add_argument("--truth-params", type=str, default=None,
                    help=f"逗号分隔的 {NPARAM} 个真值（画收敛曲线虚线用）")
    # ── CMA-ES 超参 ──
    ap.add_argument("--generations", type=int, default=2000, help="★ 最大代数")
    ap.add_argument("--popsize", type=int, default=0,
                    help="种群大小（0 = cmaes 自动；重启时按 2^n 翻倍，上限 256）")
    ap.add_argument("--sigma", type=float, default=0.2,
                    help="★ 初始步长，**在 raw（内部）空间**，不是物理量："
                         "① 正参数 φ=exp(raw) ⇒ 1σ ≈ 物理值 ×exp(σ)（0.2→±22%%，0.3→+35%%，"
                         "与初值大小无关，尺度不变）；"
                         "② 实参数 φ=raw ⇒ 1σ = **绝对** ±σ（0.3 ⇒ Px/Py/β/Pb ±0.3）。"
                         "注意两组的「同一个 σ」含义不同：初值 0.1 的实参数在 σ=0.3 下相对步长 ±300%%")
    ap.add_argument("--tol-fun", type=float, default=0.0,
                    help="目标值早停阈值（best ≤ 它就停；0 = 关）。注意 cmaes.CMA 自带的"
                         " tol-fun/tol-x 在 0.13.1 里**不可通过构造参数设置**，它自己的 "
                         "`should_stop()` 仍按内部默认值生效")
    ap.add_argument("--lr-adapt", action="store_true", help="用 LRA-CMA（cmaes>=0.11）")
    ap.add_argument("--bounds-lo", type=float, default=1e-4,
                    help="★ 搜索盒下界（**物理量**，作用在正参数上）")
    ap.add_argument("--bounds-hi", type=float, default=500.0,
                    help="★ 搜索盒上界（**物理量**，作用在正参数上）")
    ap.add_argument("--real-bound", type=float, default=10.0,
                    help="★ 实参数（Px/Py/β/Pb）的对称盒: |φ| ≤ 本值")
    ap.add_argument("--restart-patience", type=int, default=300,
                    help="★ **放宽的收敛判据**（cmaes 自带的 _tolfun=1e-12/_tolx=3e-13 太严、基本"
                         "永不触发）: 连续 N 代相对改善 < --restart-delta 就结束本轮（准备重启）。"
                         "0 = 关（只信库自带判据）")
    ap.add_argument("--restart-delta", type=float, default=1e-3,
                    help="★ 上面那个「相对改善」的阈值: (ref_best − best)/|ref_best| 小于它就当作"
                         "没进步")
    ap.add_argument("--restarts", type=int, default=0,
                    help="★ IPOP-style 重启次数（默认 0）。每一轮 popsize ×2、sigma 重置、"
                         "从**当前最优**出发 —— 用来跳出局部最优（小 yaw 卡住时很有用）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau-sign-big", type=float, choices=(1.0, -1.0),
                    default=TAU_SIGN_BIG_DEFAULT,
                    help="★ τ_cmd（大 yaw 电机通道）符号: +1=原样（默认），-1=取反。"
                         "只作用于辨识环境（加载时施加），不影响主工程/控制器")
    ap.add_argument("--tau-sign-small", type=float, choices=(1.0, -1.0),
                    default=TAU_SIGN_SMALL_DEFAULT,
                    help="★ τ_small（小 yaw 通道）符号: -1=取反（**默认**），+1=原样。"
                         "想回到『两路都原样』就显式给 --tau-sign-small=1")
    # ── 前向/并行 ──
    ap.add_argument("--forward", choices=["cpp", "numpy", "torch", "auto"], default="cpp",
                    help="★ 用哪个前向。**默认 cpp**（手写 C++，满线程）；numpy/torch 是"
                         "对照/回退；auto = cpp → numpy → torch 依次尝试（可跨机器无 g++ 时用）")
    ap.add_argument("--nthreads", type=int, default=0,
                    help="cpp 后端的线程数（★ 默认 0 = **满线程** = hardware_concurrency）")
    ap.add_argument("--block", type=int, default=32,
                    help="cpp 后端的分块宽度（分块向量化；1..256）")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 线程数（0 = torch 后端自动用 **1** 线程：实测这个前向在多线程下"
                         "反而显著变慢；cpp 后端用 --nthreads）")
    # ── 输出 ──
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="每 N 代把当前最优写成 `<--out>.genXXXXXX`（0 = 关）")
    ap.add_argument("--out", type=str, default=None, help="参数结果文件")
    pg = ap.add_argument_group("收敛曲线 / 轨迹对比（matplotlib）")
    pg.add_argument("--plot-out", type=str, default=None)
    pg.add_argument("--no-plot", action="store_true")
    pg.add_argument("--show-plot", action="store_true")
    return ap


def _parse_vecN(s, what: str, n: int = NPARAM) -> np.ndarray:
    v = np.array([float(x) for x in str(s).replace(";", ",").split(",") if x.strip() != ""],
                 dtype=np.float64)
    if v.size != n:
        raise SystemExit(f"[error] {what} 需要 {n} 个数，收到 {v.size} 个")
    return v


def _full_of(ctx, raw) -> np.ndarray:
    """raw → 完整 18 维物理参数（PARAM_NAMES 顺序）。"""
    with torch.no_grad():
        phys = ctx.layout.to_physical(torch.as_tensor(np.asarray(raw, dtype=np.float64)))
    return ctx.layout.full_vector(np.asarray(phys.detach().cpu().numpy(), dtype=np.float64))


def _raw_bounds(layout, lo, hi, real_bound):
    """物理搜索盒 → raw 空间 [P,2]（正参数取 log；实参数取 ±real_bound）。"""
    lo, hi = float(lo), float(hi)
    if lo <= 0.0 or hi <= lo:
        return None
    b = np.empty((layout.n_learnable, 2), dtype=np.float64)
    for k, name in enumerate(layout.learnable_names):
        if layout.is_positive(name):
            b[k, 0], b[k, 1] = math.log(lo), math.log(hi)
        else:
            b[k, 0], b[k, 1] = -float(real_bound), float(real_bound)
    return b


def _subsample_segments(segs, n_max: int, seed: int, what: str = "目标"):
    """确定性抽段（控制目标函数开销）。"""
    if n_max <= 0 or len(segs) <= n_max:
        return segs
    pick = np.sort(np.random.RandomState(int(seed)).choice(len(segs), size=int(n_max),
                                                           replace=False))
    print(f"[cmaes] {what}: 只用 {n_max}/{len(segs)} 段（确定性抽样，seed={seed}）")
    return [segs[int(i)] for i in pick]


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    # ★ 必须在读数据之前生效（符号在加载时施加）
    set_tau_sign(big=args.tau_sign_big, small=args.tau_sign_small)
    if torch is None:
        print(f"[error] 需要 torch: {_TORCH_IMPORT_ERROR}", file=sys.stderr)
        return 2
    try:
        import cmaes  # noqa: F401
    except ImportError:
        print("[error] 需要 cmaes:  pip install cmaes", file=sys.stderr)
        return 2

    base = PlanarParams(dx=args.dx, dy=args.dy, friction_lambda=args.model_lambda)
    cfg = FitConfig(
        fit_axis=args.fit_axis, state_mode=args.state_mode, fix_mu=args.fix_mu,
        substeps=args.substeps, integrator=args.integrator,
        freeze_params=(() if not args.freeze_params
                       else tuple(int(x) for x in str(args.freeze_params).split(",") if x != "")),
        p_constraint=args.p_constraint, p_zero_angle_deg=args.p_zero_angle,
        init_vector=(None if args.init_vector is None
                     else _parse_vecN(args.init_vector, "--init-vector")),
        truth_vector=(None if args.truth_params is None
                      else _parse_vecN(args.truth_params, "--truth-params")),
        seed=args.seed, device="cpu",
        # ★ 目标口径 = 固定窗口（确定性）
        window_len=int(args.window_len), windows_per_seg=int(args.windows_per_seg),
        max_points=int(args.max_points), epochs=0, iters=0)
    if args.threads > 0 and args.forward in ("torch", "auto"):
        torch.set_num_threads(args.threads)

    patterns = args.data or ["data/sysid/*.npz"]
    segs = truncate_hold_segments(load_segments(patterns, dt_override=args.dt),
                                 args.hold_max_sec)
    if not segs:
        print("[error] 没有读到数据", file=sys.stderr)
        return 2
    val_segs = None
    if args.val_data:
        val_segs = truncate_hold_segments(load_segments([args.val_data], dt_override=args.dt),
                                         args.hold_max_sec)
    # 2-DOF 模型不使用 β（背隙不建模）
    if not segs:
        print("[error] 没有可用的数据段（都因缺 β 被跳过）", file=sys.stderr)
        return 2
    if val_segs:
        pass  # 留出集同样不按 β 过滤

    t_start = time.time()
    # ★ 目标只用抽样的段；最终评测仍用全部（可由 --eval-max-segs 再限）
    obj_segs = _subsample_segments(segs, args.max_segs, args.seed, "目标")
    ctx = build_fit_context(obj_segs, cfg, base, what="目标", honor_windows=True)
    fwd, fwd_desc = _make_forward(args.forward, ctx, EXO_ZERO, ctx.base_phi, args)
    # 注意: build_fit_context 里的 exo 恒为 EXO_ZERO（base_omega/alpha=0），这里保持一致
    obj = WindowObjective(fwd, ctx.th_t, ctx.dth_t, mask=ctx.mask_t, ax_w=ctx.w_axis_all,
                          loss_kw=dict(loss_mode="mse", huber_delta=cfg.huber_delta,
                                       vel_weight=0.0, vel_huber_delta=cfg.vel_huber_delta))

    x0 = ctx.layout.to_raw_init(ctx.phi0)
    print(f"[cmaes] 前向: {fwd_desc}")
    print(f"[cmaes] 目标: {ctx.W} 个窗口 × {ctx.L} 步（window_len={args.window_len}"
          f"{' = 整段/全量' if int(args.window_len) == 0 else ''}, "
          f"windows_per_seg={args.windows_per_seg}, "
          f"mask={'有（长度不一按有效点平均）' if ctx.mask_t is not None else '无'}）；"
          f"可学习参数 {ctx.n_free} 个；损失**不 wrap**")
    print("[cmaes] 初值 φ0 = " + np.array2string(ctx.phi0, precision=5))

    # ── ★ 无梯度：全程关闭 autograd ──
    torch.set_grad_enabled(False)
    f0 = obj(x0)
    print(f"[cmaes] 初值目标 f0 = {f0:.6e}", flush=True)

    # ── ★ 搜索盒（物理量）: 正参数 ∈ [lo, hi]（log 空间），实参数 |φ| ≤ real_bound ──
    bounds = _raw_bounds(ctx.layout, args.bounds_lo, args.bounds_hi, args.real_bound)
    if bounds is not None:
        print(f"[cmaes] 搜索盒: 正参数 φ ∈ [{args.bounds_lo:g}, {args.bounds_hi:g}]"
              f"（raw ∈ [{math.log(args.bounds_lo):+.3f}, {math.log(args.bounds_hi):+.3f}]）"
              f"；实参数 |φ| ≤ {args.real_bound:g}", flush=True)

    CMA = cmaes.CMA
    total_gen = int(args.generations)
    n_restart, gen, evals, stop_all = 0, 0, 0, False
    best_raw = np.asarray(x0, dtype=np.float64).copy()
    best_f = float(f0)
    loss_hist, param_hist = [best_f], [_full_of(ctx, x0)]

    while not stop_all and gen < total_gen:
        psize = (0 if int(args.popsize) <= 0 else min(256, int(args.popsize) * (2 ** n_restart)))
        if n_restart == 0:
            mean0 = np.asarray(x0, dtype=np.float64).copy()
        else:
            mean0 = best_raw.copy()
            print(f"[cmaes] ★ 第 {n_restart} 次重启（IPOP-style）: 从当前最优出发、"
                  f"sigma 重置为 {args.sigma:g}、popsize → {psize or '自动'}", flush=True)
        kw = dict(mean=mean0, sigma=float(args.sigma), seed=int(args.seed) + n_restart)
        if bounds is not None:
            kw["bounds"] = bounds
        if psize:
            kw["population_size"] = psize
        if args.lr_adapt:
            kw["lr_adapt"] = True
        try:
            opt = CMA(**kw)
        except TypeError:                       # 老版本 cmaes 不认 lr_adapt
            kw.pop("lr_adapt", None)
            opt = CMA(**kw)
        gen_this = 0
        ref_f, ref_gen = best_f, gen            # 最近一次"显著改善"的位置
        while gen < total_gen:
            gen += 1
            gen_this += 1
            sols = []
            for _ in range(int(opt.population_size)):
                x = np.asarray(opt.ask(), dtype=np.float64)
                sols.append((x, obj(x)))
            opt.tell(sols)
            evals += len(sols)
            gen_best = min(f for _, f in sols)
            if gen_best < best_f:
                best_f = float(gen_best)
                best_raw = np.asarray(
                    sols[int(np.argmin([f for _, f in sols]))][0], dtype=np.float64)
            loss_hist.append(best_f)
            param_hist.append(_full_of(ctx, best_raw))
            if args.print_every > 0 and (gen % args.print_every == 0 or gen == 1):
                gen_mean = float(np.mean([f for _, f in sols]))
                print(f"[cmaes] gen {gen:5d}/{total_gen} (重启 {n_restart})  best={best_f:.6e}  "
                      f"gen_mean={gen_mean:.6e}  evals={evals}", flush=True)
            if args.checkpoint_every and args.out and gen % int(args.checkpoint_every) == 0:
                write_params_file(f"{args.out}.gen{gen:06d}", _full_of(ctx, best_raw),
                                  recipe=f"CMA-ES checkpoint @ gen {gen}；{tau_sign_desc()}")
            # ── ★ 放宽的收敛判据: 连续 patience 代相对改善 < delta ⇒ 本轮结束 ──
            if (ref_f - best_f) / max(abs(ref_f), 1e-30) >= float(args.restart_delta):
                ref_f, ref_gen = best_f, gen
            if (int(args.restart_patience) > 0
                    and gen - ref_gen >= int(args.restart_patience)):
                print(f"[cmaes] 连续 {gen - ref_gen} 代相对改善 < {args.restart_delta:g}"
                      f"（best={best_f:.6e}）⇒ 本轮结束，准备重启", flush=True)
                break
            if args.tol_fun and best_f <= float(args.tol_fun):
                print(f"[cmaes] best={best_f:.6e} ≤ --tol-fun={args.tol_fun:g} ⇒ 第 {gen} 代提前结束")
                stop_all = True
                break
            stop = getattr(opt, "should_stop", None)
            if callable(stop) and stop():
                print(f"[cmaes] 本轮第 {gen_this} 代: cmaes 自身判据判定收敛（best={best_f:.6e}）",
                      flush=True)
                break
        if stop_all or gen >= total_gen:
            break
        if n_restart >= int(args.restarts) or gen_this < 2:
            break
        n_restart += 1

    phi = _full_of(ctx, best_raw)
    seconds = time.time() - t_start
    recipe = (f"CMA-ES（cmaes, sigma={args.sigma:g}, 重启 {n_restart} 次, "
              f"generations={gen}, evals={evals}, 盒=[{args.bounds_lo:g},{args.bounds_hi:g}]"
              f"/|real|≤{args.real_bound:g}, forward={fwd_desc.split('（')[0]}，"
              f"{tau_sign_desc()}）")
    res = FitResult(phi=phi, phi0=ctx.phi0, loss_history=loss_hist, val_loss=float(best_f),
                    n_iter=int(evals), seconds=seconds, param_history=param_hist,
                    epoch_losses=list(loss_hist), n_free=ctx.n_free, n_steps=int(evals),
                    recipe=recipe,
                    truth=(None if cfg.truth_vector is None
                           else np.asarray(cfg.truth_vector, dtype=np.float64)),
                    config=dict(ctx.summary, recipe=recipe))

    print("\n" + "=" * 78)
    print(f"CMA-ES 辨识结果（段数={len(ctx.segs)}/{len(segs)}[目标/全部], "
          f"窗口={ctx.W}×{ctx.L}）  用时 {seconds:.1f}s  评估 {evals} 次  "
          f"重启 {n_restart} 次")
    print(f"配方: {recipe}")
    for _ln in format_param_table(res.phi, res.phi0):
        print(_ln)
    print("=" * 78)

    # ── 最终开环评测（与 Adam 训练器同一口径: numpy 逐段 + 不 wrap）──
    eval_segs = val_segs if val_segs is not None else segs
    eval_segs = _subsample_segments(eval_segs, args.eval_max_segs, args.seed, "评测")
    tag = "留出集" if val_segs is not None else "训练集"
    rm = channel_rmse(eval_segs, res.phi, base, integrator=cfg.integrator,
                      substeps=cfg.substeps, state_mode=cfg.state_mode)
    rm0 = channel_rmse(eval_segs, res.phi0, base, integrator=cfg.integrator,
                       substeps=cfg.substeps, state_mode=cfg.state_mode)
    print(f"目标值 f(best) = {best_f:.6e}   f(初值) = {f0:.6e}")
    print(f"  ★ 估计参数（{tag}）: {_fmt_rmse(rm)}")
    print(f"               {_fmt_rmse_full(rm)}")
    print(f"    （对照）初值: {_fmt_rmse(rm0)}")
    print("可粘贴到 include/tcbs/mpc/planar_yaw_params.h:")
    for _ln in format_header_snippet(res.phi):
        print(_ln)

    if not args.no_plot:
        conv_png, traj_png = resolve_plot_paths(args.plot_out)
        try:
            plot_convergence(res, conv_png, show_plot=args.show_plot,
                             title_note=f"CMA-ES, {ctx.W} 窗口 × {ctx.L} 步, forward={args.forward}")
            plot_trajectory(res, eval_segs, base, integrator=cfg.integrator,
                            out_path=traj_png, show_plot=args.show_plot,
                            state_mode=cfg.state_mode)
        except Exception as exc:
            print(f"[plot][warn] 画图失败（结果仍有效）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    if args.out:
        write_params_file(args.out, res.phi, recipe=recipe)
        print(f"[out] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
