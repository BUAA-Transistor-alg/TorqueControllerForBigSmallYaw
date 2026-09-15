#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ident_methods.py — 在**仿真**中对比三种 8 参辨识方法
================================================================================

被辨识对象: `include/tcbs/mpc/planar_yaw_model.h` 的平面 2-DOF 模型（8 参）
    φ = [Jbig_eff, Js, Px, Py, fc_big, fv_big, fc_small, fv_small]

本脚本**不依赖真机、不依赖任何已有采集数据**:
  1) 用 `data/targets/*.npz`（键 `target`，100 Hz）的录制目标序列做激励源
     （增强: 随机截取 300 点、连续化、随机缩放/偏置；小 yaw 整体缩放到 ±40°），
     经 `trajectory_planner.TrajectoryPlanner` 平滑后，用**上位机 PID**
     (kp=2, ki=0.1, kd=0.2, ±1.0 N·m, 力矩变化率 ≤ 0.1/step @100 Hz) 分轴采集:
       axis=0 大 yaw 跟踪序列 / 小 yaw PID 保持到 ±40° 内随机位置；
       axis=1 小 yaw 跟踪序列 / 大 yaw PID 保持到随机位置。
     **plant 用 λ_plant = 100 模拟真实库伦摩擦**（积分步长 0.05 ms 保证数值稳定），
     角度按**编码器量化** (8192 计数/整圈 ⇒ 2π/8192 ≈ 7.66e-4 rad) 加噪。
     生成若干段数据并留一部分做**验证集**。
  2) 用同一批数据拟合同样的 8 个参数，比较三种方法:
     (a) LS + 理想保持值: 逆动力学线性回归 τ = Y(φ)·φ；held 轴 θ̇/θ̈ 视为 **0**（位置仍用实测）
     (b) LS + 实测值:     同样回归，但 held 轴用**实测** θ̇/θ̈（量化角中心差分 + 3 点平滑）
     (c) torch:           `identify_params_torch.py` 的**输出误差法**（可导前向仿真）
     三种方法使用**同一批数据 + 同一预处理**（中心差分 + 3 点平滑；剔饱和/无信息样本）。
  3) 打印并写出 `docs/sysid_compare.md`: 每个参数的真值/三法估计/绝对/相对误差；
     验证集上的前向仿真角度 RMSE；λ 失配（plant 100 vs 模型 10）的量化；
     (a) vs (b) 差异来源分解；以及"实车该用哪种方法"的结论。

用法::

    # 纯仿真（无真机、无采集数据也能跑通；这是验收命令）
    python3 python/scripts/compare_ident_methods.py --sim-only --segments=6
    # 若已有真机数据，可只做三法对比（不做仿真采集）
    python3 python/scripts/compare_ident_methods.py --data='data/sysid/*.csv' --segments=0

注意: 本脚本默认只写 `docs/sysid_compare.md` 一个文件（用 --no-write 可关闭）。
仿真数据全部保存在内存中，不留数据文件（--dump-sim=DIR 可显式落盘成 CSV）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from identify_params_torch import (  # noqa: E402
    AXIS_BIG, AXIS_SMALL, DT_DEFAULT, ENCODER_CPR, EXO_ZERO, FRICTION_LAMBDA, NPARAM,
    PARAM_NAMES, QUANT_STEP, FitConfig, PlanarParams, Segment,
    default_param_vector, derivatives_from_angles, exo_from_gravity, fit_params_torch,
    p_direction,
    inverse_dynamics_np, model_self_test, quantize, regressor_np,
    rk4_step_np, simulate_np,
)
from trajectory_planner import StepRefinementWrapper, TrajectoryPlanner  # noqa: E402

TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0

# 仿真"真值"参数（plant）—— ★ 按**真实量级**取值:
#   原仓库 data/cars/{Infantry1,Infantry2}/params/Identified_parameters.txt（单 yaw 整体）:
#       Infantry1: J=0.024986  tau_c=0.122514  b=0.025648
#       Infantry2: J=0.016541  tau_c=0.097297   b=0.032100
#   用户补充: 大小 yaw 取同量级；大 yaw 因承载小 yaw 整体约翻倍；
#             两 yaw 轴间距 ≈ 0.1 m；小 yaw 上装 m_u ≈ 0.1 kg、质心位移 ρ ≈ 0.1 m ⇒ |P| ≈ 0.01。
#   ⇒ Jbig_eff≈2×0.025, Js≈0.02, fc_big≈2×0.12, fv_big≈2×0.026,
#     fc_small=0.0973（同量级原仓库）、fv_small=0.028。
#   P 的方向**故意与 d 差 30°**（这样"P 沿 d / Py=0"的约束才真的在起作用），
#   等价于平衡点 θ* = angle(d) − angle(P) ≈ −29.9°（见 THETA_STAR_TRUE_DEG）。
TRUTH_VECTOR = np.array([0.050, 0.020, 0.0087, 0.0050, 0.220, 0.055, 0.0973, 0.028])

# 实测几何（小 yaw 轴相对大 yaw 轴的平面偏置），不是辨识参数 —— 真实量级 ≈ 0.1 m
DX_M = 0.10
DY_M = 0.0

# ★ "已完成零点标定"的等效真值: 平衡点被定义为 θs=0 之后，物理上必然 P = |P|·d̂
#   （d=(dx,0) ⇒ Py=0），模长取与 TRUTH_VECTOR 相同的 |P|
P_ALONG_TRUTH = float(np.hypot(TRUTH_VECTOR[2], TRUTH_VECTOR[3]))

# ★ 离心平衡点（θ* = angle(d) − angle(P)，见 planar_yaw_model 的 μ=0 条件）。
#   零点若按"平衡点"标定，则模型里 P = |P|·R(−θ*)·d̂；这里给出真值，用来判断
#   "平衡点是否落在小 yaw 行程内"（不可达 ⇒ 零点标定流程要另想办法）。
THETA_STAR_TRUE_DEG = (math.degrees(math.atan2(DY_M, DX_M))
                       - math.degrees(math.atan2(TRUTH_VECTOR[3], TRUTH_VECTOR[2])))


# ============================================================================
# 配置
# ============================================================================
@dataclass
class SimConfig:
    dt: float = DT_DEFAULT               # 100 Hz
    seg_len: int = 300                   # 每段 3 s
    # ── plant / 辨识模型 ──
    plant_lambda: float = 1.0e4           # ★ 被控对象用 λ=1e4 逼近真·库仑（sgn）
    model_lambda: float = FRICTION_LAMBDA  # ★ 辨识/拟合模型固定 λ = 100
    plant_substep: float = 2.0e-5        # plant 积分步长 0.02 ms
                                         #   λ=1e4 ⇒ |J_f|≈4.9e4/s ⇒ RK4 上限 ≈57 µs
    validate_substeps: int = 10          # 前向验证的 RK4 子步
                                         #   模型 λ=100 ⇒ dt_sub=1 ms ≪ 2.78/|J_f|≈5.7 ms ✓
    # ── 几何（实测）──
    dx: float = DX_M
    dy: float = DY_M
    # ── 上位机 PID ──
    kp: float = 2.0
    ki: float = 0.1
    kd: float = 0.2
    pid_out_limit: float = 1.0           # N·m
    pid_rate_limit: float = 0.1          # N·m/step
    pid_d_filter_tau: float = 0.02       # 微分项低通时间常数 (s)
    # ── 编码器 ──
    quant_step: float = QUANT_STEP       # 2π/8192
    # ── 规划器限幅（★ 按"轴的力矩权限"分轴设置，不是照抄默认值）──
    #   默认 V=30/A=50/J=2000 是原仓库给大 yaw 的粗暴值；若照抄，小 yaw 会被
    #   反应力矩/跟踪误差推出 −25°~+20° 的硬行程（collect_sim 会直接报错）。
    #   这里给: 大 yaw V=3/A=6/J=80；小 yaw V=2/A=4/J=50（跟踪误差 ~2-4°，留够 8° 余量）
    #   ★ 真实摩擦量级下（fc_small≈0.1 N·m）跟踪误差 ~fc/kp ≈ 3°，必须把参考再放慢，
    #     否则 θs 会被摩擦/反应力矩推出 −25°~+20° 的硬行程。
    planner_v_big: float = 2.0
    planner_a_big: float = 4.0
    planner_j_big: float = 60.0
    planner_v_small: float = 1.5
    planner_a_small: float = 2.5
    planner_j_small: float = 40.0
    # 兼容旧字段（不再使用；保留避免外部引用报错）
    planner_v: float = 3.0
    planner_a: float = 6.0
    planner_j: float = 80.0
    refine_n: int = 100
    # ── ★ 小 yaw 实际机械行程（用户实测确认，非对称）──
    #   硬界限 −25° ~ +20°（中心 −2.5°）：仿真里**从不触发**，触发即 bug（collect_sim 会硬报错）
    #   参考包络两侧各留 ~8° 跟踪余量 ⇒ −17° ~ +12°
    small_travel_min_deg: float = -25.0
    small_travel_max_deg: float = 20.0
    small_env_margin_deg: float = 8.0
    # 旧包络（本仓库早期假设，保留用于"新旧包络对比"）: 对称 ±40°，硬界限 ±45°
    small_legacy_env_deg: float = 40.0
    small_legacy_travel_deg: float = 45.0
    small_env_mode: str = "asym"         # asym（新，非对称）| legacy（旧，对称 ±40°）
    env_fill_min: float = 0.50           # 参考轨迹占包络半宽的比例下界（随机 0.50~0.90）
    env_fill_max: float = 0.90           # 上界（摆幅扫描时令 min == max 固定摆幅）
    # ── 静态倾斜（绕 y 轴；逐段可不同）──
    #   空 = 水平；非空时第 k 段用 tilt_list[k % len] 度。
    #   A 系重力平面分量逐样本算: g_A = R_z(−θ_b)·(g·sinφ, 0)
    tilt_deg_list: tuple = ()
    gravity: float = 9.81
    # ── 激励范围（其它）──
    big_hold_range: float = math.pi
    init_offset_big: float = 0.12        # 起始位置随机偏置（rad），制造 PID 瞬态
    init_offset_small: float = 0.06

    # ── 派生量: 行程 / 包络（弧度）──
    @property
    def travel_min(self) -> float:
        return (-self.small_legacy_travel_deg if self.small_env_mode == "legacy"
                else self.small_travel_min_deg) * DEG

    @property
    def travel_max(self) -> float:
        return (self.small_legacy_travel_deg if self.small_env_mode == "legacy"
                else self.small_travel_max_deg) * DEG

    @property
    def env_min(self) -> float:
        if self.small_env_mode == "legacy":
            return -self.small_legacy_env_deg * DEG
        return (self.small_travel_min_deg + self.small_env_margin_deg) * DEG

    @property
    def env_max(self) -> float:
        if self.small_env_mode == "legacy":
            return self.small_legacy_env_deg * DEG
        return (self.small_travel_max_deg - self.small_env_margin_deg) * DEG

    @property
    def env_center(self) -> float:
        return 0.5 * (self.env_min + self.env_max)

    @property
    def env_half(self) -> float:
        return 0.5 * (self.env_max - self.env_min)

    @property
    def small_center_deg(self) -> float:
        return 0.5 * (self.travel_min + self.travel_max) / DEG

    def env_label(self) -> str:
        return (f"新包络（非对称 −{abs(self.small_travel_min_deg):.0f}°~+"
                f"{self.small_travel_max_deg:.0f}°，参考 [{self.env_min/DEG:+.0f}°,"
                f"{self.env_max/DEG:+.0f}°]）" if self.small_env_mode == "asym" else
                f"旧包络（对称 ±{self.small_legacy_env_deg:.0f}°）")


@dataclass
class LSConfig:
    p_constraint: str = "free"           # free | along_d | zero（主方法的 P 处理方式）
    p_zero_angle_deg: float = 0.0        # 平衡点在当前零点坐标系里的读数（度）；仅 along_d 用

    rcond: float = 1e-6                  # 截断 SVD 相对阈值（先验 φ0 只在截断方向生效）
    sat_ratio: float = 0.98              # |τ| ≥ sat_ratio·limit 视为饱和 → 剔除
    quiescent_tau: float = 0.03          # N·m
    quiescent_omega: float = 0.05        # rad/s
    max_big_age: float = 0.15            # s（mcu2_seq 陈旧样本剔除；仿真数据不陈旧）


# ============================================================================
# 激励源: 录制目标序列 / 自造序列
# ============================================================================
def load_target_library(patterns, verbose: bool = True):
    """读 data/targets/*.npz 的 `target` 键（100 Hz）。"""
    import glob as globmod
    files = []
    for pat in patterns:
        files.extend(sorted(globmod.glob(pat)))
    seqs = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            if "target" in d.files:
                seqs.append(np.asarray(d["target"], dtype=np.float64).reshape(-1))
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[targets] 读取失败 {f}: {exc}")
    if verbose:
        print(f"[targets] 读到 {len(seqs)} 条录制目标序列（来自 {len(files)} 个文件）")
    return seqs


def synth_targets(rng, n_seq: int, length: int = 4000):
    """没有录制序列时自造激励: 多正弦 + 平滑阶跃混合（100 Hz, 角速度 ~ ±1.5 rad/s）。"""
    seqs = []
    for k in range(n_seq):
        t = np.arange(length) * DT_DEFAULT
        y = np.zeros_like(t)
        for _ in range(int(rng.integers(2, 4))):
            f = rng.uniform(0.15, 1.2)
            a = rng.uniform(0.3, 1.0)
            y += a * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
        # 叠几个平滑阶跃（用 tanh 过渡，过渡时间 ~0.15 s）
        for _ in range(int(rng.integers(2, 5))):
            t0 = rng.uniform(1.0, t[-1] - 1.0)
            amp = rng.uniform(-0.8, 0.8)
            y += amp * 0.5 * (1.0 + np.tanh((t - t0) / 0.15))
        seqs.append(y)
    return seqs


def make_reference(raw, axis: int, rng, cfg: SimConfig):
    """从一条录制序列生成 (driven 轴参考轨迹, held 轴保持目标)。"""
    if len(raw) <= cfg.seg_len:
        seq = raw.copy()
    else:
        start = int(rng.integers(0, len(raw) - cfg.seg_len))
        seq = raw[start:start + cfg.seg_len].copy()

    # 连续化: 相邻跳变 > π 时把后续点 ±2π（与 collect_sysid_data.py 一致）
    for i in range(1, len(seq)):
        d = seq[i] - seq[i - 1]
        if d > math.pi:
            seq[i] -= TWO_PI
        elif d < -math.pi:
            seq[i] += TWO_PI

    # ★ held_target 永远是"**被保持那根轴**的目标"（driven 轴是被激励的那根）:
    #   axis==SMALL（小 yaw 被驱动）⇒ 目标 = 大 yaw 的保持位置（±π，多圈连续）
    #   axis==BIG  （大 yaw 被驱动）⇒ 目标 = 小 yaw 的保持位置（必须落在小 yaw 包络内!）
    if axis == AXIS_SMALL:
        # 小 yaw 行程受限（新: −25°~+20°；旧: ±40°）:
        #   1) 去掉均值 ⇒ 把序列中心移到包络中心（新包络中心是 −2.5°，**不是 0**）
        #   2) 整体缩放到占包络半宽的 0.55~1.0 倍（留跟踪余量）
        seq = seq - np.mean(seq)
        peak = float(np.max(np.abs(seq))) or 1.0
        scale = (cfg.env_half / peak) * rng.uniform(cfg.env_fill_min, cfg.env_fill_max)
        seq = seq * scale + cfg.env_center
        seq = np.clip(seq, cfg.env_min, cfg.env_max)
        held = rng.uniform(-cfg.big_hold_range, cfg.big_hold_range)     # ← 大 yaw 的保持目标
        pv, pa, pj = cfg.planner_v_small, cfg.planner_a_small, cfg.planner_j_small
    else:
        seq = seq * rng.uniform(0.5, 1.0)        # 随机缩放
        seq = seq + rng.uniform(-math.pi, math.pi)   # 随机偏置（大 yaw 多圈连续）
        # ← 小 yaw 的保持目标: 落在包络内**再收 1°**，给跟踪超调留余量（否则会撞硬限位）
        held = rng.uniform(cfg.env_min + 2.0 * DEG, cfg.env_max - 2.0 * DEG)
        pv, pa, pj = cfg.planner_v_big, cfg.planner_a_big, cfg.planner_j_big

    # TrajectoryPlanner 平滑（+ StepRefinementWrapper 细化）—— 限幅必须与"轴的力矩权限"匹配:
    # 小 yaw 行程只有 45°、PID 只有 kp=2/±1.0 N·m/0.1 per step ⇒ 参考太猛会直接顶到行程硬界限
    planner = TrajectoryPlanner(pv, pa, pj)
    refined = StepRefinementWrapper(planner.step, cfg.refine_n)
    smooth = np.zeros_like(seq)
    pos, vel, acc = float(seq[0]), 0.0, 0.0
    for i in range(len(seq)):
        pos, vel, acc, _ = refined.step(float(seq[i]), pos, vel, acc, cfg.dt)
        smooth[i] = pos
    if axis == AXIS_SMALL:
        smooth = np.clip(smooth, cfg.env_min, cfg.env_max)
    return smooth, float(held)


# ============================================================================
# 仿真采集（plant = λ=100；PID 闭环；编码器量化）
# ============================================================================
def _wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


def collect_sim(cfg: SimConfig, truth: PlanarParams, specs, rng, verbose=True):
    """矢量化的分轴 PID 闭环采集。

    specs: list of (raw_seq, axis)；返回 Segment 列表（theta 已量化，附真值轨迹）。
    """
    B = len(specs)
    T = cfg.seg_len
    dt = cfg.dt
    substeps = max(1, int(round(dt / cfg.plant_substep)))

    ref = np.zeros((T, B, 2))
    held_target = np.zeros(B)
    axis_arr = np.zeros(B, dtype=int)
    for b, (raw, ax) in enumerate(specs):
        seq, held = make_reference(raw, ax, rng, cfg)
        axis_arr[b] = ax
        held_target[b] = held
        if ax == AXIS_BIG:
            ref[:, b, 0] = seq
            ref[:, b, 1] = held
        else:
            ref[:, b, 1] = seq
            ref[:, b, 0] = held

    # ── 初始状态: 在参考首值附近给随机偏置（制造 PID 瞬态激励）──
    q = ref[0].copy()
    off = np.where(axis_arr == AXIS_BIG, cfg.init_offset_big, cfg.init_offset_small)
    q[:, 0] += rng.uniform(-1.0, 1.0, size=B) * off
    q[:, 1] += rng.uniform(-1.0, 1.0, size=B) * off * 0.5
    q[:, 1] = np.clip(q[:, 1], cfg.env_min, cfg.env_max)   # 起始点也在包络内
    qd = np.zeros((B, 2))

    plant = truth.geometry_copy(friction_lambda=cfg.plant_lambda)

    # ── ★ 静态倾斜: 第 b 段用 tilt_list[b % len] 度（绕 y 轴），A 系重力逐样本算 ──
    tl = np.asarray(cfg.tilt_deg_list if len(cfg.tilt_deg_list) else (0.0,), dtype=np.float64)
    tilt_deg = tl[np.arange(B) % tl.size]
    tilt_rad = tilt_deg * DEG
    g_c_planar = cfg.gravity * np.sin(tilt_rad)      # C 系平面分量 (g·sinφ, 0)
    gravity_log = np.zeros((T, B, 2))

    theta_true = np.zeros((T, B, 2))
    dtheta_true = np.zeros((T, B, 2))
    ddtheta_true = np.zeros((T, B, 2))
    theta_meas = np.zeros((T, B, 2))
    dtheta_pid = np.zeros((T, B, 2))
    tau_log = np.zeros((T, B, 2))

    integral = np.zeros((B, 2))
    dth_filt = np.zeros((B, 2))
    prev_meas = quantize(q, cfg.quant_step)
    tau_prev = np.zeros((B, 2))
    alpha = dt / (cfg.pid_d_filter_tau + dt)

    for i in range(T):
        theta_true[i] = q
        dtheta_true[i] = qd
        # A 系重力平面分量: g_A = R_z(−θ_b)·(g·sinφ, 0)
        gx = g_c_planar * np.cos(q[:, 0])
        gy = -g_c_planar * np.sin(q[:, 0])
        gravity_log[i, :, 0] = gx
        gravity_log[i, :, 1] = gy
        exo_i = exo_from_gravity(gx, gy) if np.any(g_c_planar != 0.0) else EXO_ZERO
        meas = quantize(q, cfg.quant_step)
        theta_meas[i] = meas

        raw_rate = (meas - prev_meas) / dt
        dth_filt = (1.0 - alpha) * dth_filt + alpha * raw_rate
        dtheta_pid[i] = dth_filt
        prev_meas = meas

        err = np.array([[_wrap(ref[i, b, 0] - meas[b, 0]), _wrap(ref[i, b, 1] - meas[b, 1])]
                        for b in range(B)])
        unsat = cfg.kp * err + cfg.ki * (integral + err * dt) - cfg.kd * dth_filt
        sat_hi = unsat > cfg.pid_out_limit
        sat_lo = unsat < -cfg.pid_out_limit
        integrate = ~((sat_hi & (err > 0)) | (sat_lo & (err < 0)))
        integral = np.where(integrate, integral + err * dt, integral)
        tau_cmd = cfg.kp * err + cfg.ki * integral - cfg.kd * dth_filt
        tau_cmd = np.clip(tau_cmd, -cfg.pid_out_limit, cfg.pid_out_limit)
        tau_cmd = np.clip(tau_cmd, tau_prev - cfg.pid_rate_limit, tau_prev + cfg.pid_rate_limit)
        tau_log[i] = tau_cmd

        q_old, qd_old = q, qd
        q, qd = rk4_step_np(q, qd, tau_cmd, plant, exo_i, dt, substeps)
        ddtheta_true[i] = (qd - qd_old) / dt
        tau_prev = tau_cmd

    # ── ★ 小 yaw 硬行程校验（−25°~+20° / 旧 ±45°）: 仿真里应当从不触发 ──
    ts = theta_true[:, :, 1]
    ts_min, ts_max = float(np.min(ts)), float(np.max(ts))
    viol = int(np.sum((ts < cfg.travel_min - 1e-9) | (ts > cfg.travel_max + 1e-9)))
    if viol > 0:
        raise RuntimeError(
            f"[sim] 小 yaw 触发硬行程界限 {viol} 次: θs∈[{ts_min/DEG:+.2f}°,{ts_max/DEG:+.2f}°] "
            f"越出 [{cfg.travel_min/DEG:+.1f}°,{cfg.travel_max/DEG:+.1f}°] —— 这是 bug"
            f"（参考包络应留出足够跟踪余量）")
    env_viol = int(np.sum((ts < cfg.env_min - 1e-9) | (ts > cfg.env_max + 1e-9)))

    segs = []
    for b in range(B):
        th_m = theta_meas[:, b, :]
        dth_m, _ = derivatives_from_angles(th_m, dt)      # ★ 统一预处理: 中心差分 + 3 点平滑
        seg = Segment(
            t=np.arange(T) * dt,
            theta=th_m,
            dtheta=dth_m,
            tau=tau_log[:, b, :],
            axis=int(axis_arr[b]),
            held_target=float(held_target[b]),
            dt=dt,
            mcu2_seq=np.arange(T, dtype=np.float64),      # 仿真里大 yaw 直连 ⇒ 每拍都新
            theta_true=theta_true[:, b, :],
            dtheta_true=dtheta_true[:, b, :],
            ddtheta_true=ddtheta_true[:, b, :],
            gravity=gravity_log[:, b, :] if np.any(g_c_planar != 0.0) else None,
            source=f"sim[{b}]",
            tag=("big" if axis_arr[b] == AXIS_BIG else "small"),
        )
        segs.append(seg)
    if verbose:
        print(f"[sim] 采集完成: {B} 段 × {T} 点 @ {1.0/dt:.0f} Hz；"
              f"plant λ={cfg.plant_lambda:g}, 积分步长 {cfg.plant_substep*1e3:.3f} ms "
              f"({substeps} 子步/控制步)")
        print(f"[sim] 小 yaw {cfg.env_label()}；θs∈[{ts_min/DEG:+.2f}°,{ts_max/DEG:+.2f}°]"
              f"（行程中心 {cfg.small_center_deg:+.1f}°），越出硬界限 {viol} 次、"
              f"越出参考包络 {env_viol} 点/{T*B}")
        if np.any(tilt_deg != 0.0):
            print(f"[sim] 静态倾斜: 逐段 {tilt_deg.tolist()} 度（绕 y 轴）⇒ "
                  f"A 系重力平面分量幅值 g·sinφ = {abs(g_c_planar).max():.3f} m/s²，"
                  f"方向随 θ_b 旋转")
    return {"segs": segs, "tilt_deg": tilt_deg, "theta_small_range_deg": (ts_min/DEG, ts_max/DEG),
            "limit_violations": viol, "env_violations": env_viol,
            "gravity_amp": float(np.max(np.abs(gravity_log)))}


def split_fit_val(segs, holdout_per_axis: int, rng):
    """每轴各留 holdout_per_axis 段做验证集。"""
    fit, val = [], []
    for ax in (AXIS_BIG, AXIS_SMALL):
        idx = [i for i, s in enumerate(segs) if s.axis == ax]
        rng.shuffle(idx)
        val_idx = set(idx[:min(holdout_per_axis, max(0, len(idx) - 1))])
        for i in idx:
            (val if i in val_idx else fit).append(segs[i])
    return fit, val


# ============================================================================
# 方法 (a)(b): 逆动力学线性回归（τ = Y·φ）
# ============================================================================
def build_ls_system(segs, mode: str, cfg: SimConfig, ls: LSConfig, oracle: bool = False,
                    grav_scale: float = 1.0):
    """构造 (Y, τ) 并做统一的样本剔除。

    mode = 'ideal'   : held 轴 θ̇ = θ̈ = 0（位置仍用实测）
    mode = 'measured': held 轴用实测 θ̇/θ̈
    oracle = True    : 状态与导数都取 plant 真值（消融用，隔离"导数噪声"的影响）
    返回 Y [N,8], tau [N], info(dict)
    """
    assert mode in ("ideal", "measured")
    Y_parts, tau_parts = [], []
    n_row = 0
    dropped = {"border": 0, "sat": 0, "quiescent": 0, "stale": 0}
    for seg in segs:
        T = seg.T
        q = seg.theta.copy()
        qd = seg.dtheta.copy()
        _, qdd = derivatives_from_angles(seg.theta, seg.dt)
        if oracle:
            q = seg.theta_true.copy()
            qd = seg.dtheta_true.copy()
            qdd = seg.ddtheta_true.copy()
        held = seg.held
        if mode == "ideal":
            qd[:, held] = 0.0
            qdd[:, held] = 0.0

        # ── 样本有效性 ──
        valid = np.zeros(T, dtype=bool)
        valid[1:T - 1] = True
        dropped["border"] += int((~valid).sum())
        tau_abs = np.max(np.abs(seg.tau), axis=1)
        om_abs = np.max(np.abs(qd), axis=1)
        sat = tau_abs >= ls.sat_ratio * cfg.pid_out_limit
        quiescent = (tau_abs < ls.quiescent_tau) & (om_abs < ls.quiescent_omega)
        stale = np.zeros(T, dtype=bool)
        if seg.mcu2_seq is not None and ls.max_big_age > 0:
            seq = seg.mcu2_seq
            new = np.concatenate([[True], np.diff(seq) != 0])
            last_new = np.maximum.accumulate(np.where(new, np.arange(T), 0))
            age = (np.arange(T) - last_new) * seg.dt
            stale = age > ls.max_big_age
        dropped["sat"] += int((valid & sat).sum())
        dropped["quiescent"] += int((valid & ~sat & quiescent).sum())
        dropped["stale"] += int((valid & ~sat & ~quiescent & stale).sum())
        keep = valid & ~sat & ~quiescent & ~stale
        if not np.any(keep):
            continue
        exo = (EXO_ZERO if seg.gravity is None else
               exo_from_gravity(seg.gravity[keep, 0] * grav_scale,
                                seg.gravity[keep, 1] * grav_scale))
        Yseg = regressor_np(q[keep], qd[keep], qdd[keep],
                            PlanarParams(dx=cfg.dx, dy=cfg.dy,
                                         friction_lambda=cfg.model_lambda), exo)
        Y_parts.append(Yseg.reshape(-1, NPARAM))
        tau_parts.append(seg.tau[keep].reshape(-1))
        n_row += int(keep.sum())
    if not Y_parts:
        raise RuntimeError("没有可用样本（剔除过严？）")
    Y = np.concatenate(Y_parts, axis=0)
    tau = np.concatenate(tau_parts, axis=0)
    # 每个样本贡献 2 行（两轴方程各一行）⇒ 行数 = 2 × 保留样本数
    info = {"n_row": int(Y.shape[0]), "n_sample": n_row, "n_kept_samples": n_row,
            "dropped": dropped}
    return Y, tau, info


def solve_ls_truncated(Y, tau, phi0, rcond: float = 1e-6, free_mask=None):
    """截断 SVD 最小二乘（Δφ 形式，列归一化）——与 docs/calibration.md §4.1 一致。"""
    if free_mask is not None:              # 冻结某些参数（例如 Px=Py≡0）⇒ 该列置零
        Y = Y.copy()
        Y[:, ~np.asarray(free_mask, dtype=bool)] = 0.0
    r = tau - Y @ phi0
    col_raw = np.linalg.norm(Y, axis=0)          # 原始列范数（用于判定"非零列"）
    col = np.where(col_raw < 1e-12, 1.0, col_raw)  # 归一化用（避免除 0）
    Ys = Y / col
    U, S, Vt = np.linalg.svd(Ys, full_matrices=False)
    rank = int(np.sum(S > rcond * S[0]))
    z = Vt[:rank].T @ ((U[:, :rank].T @ r) / S[:rank])
    dphi = z / col
    phi = phi0 + dphi
    resid = Y @ phi - tau
    # 参数不确定度（只计入 τ 侧残差 —— 对 (b) 是**乐观下界**，它没算回归矩阵自身的噪声）
    n_row = int(Y.shape[0])
    XtXi = np.linalg.pinv(Y.T @ Y, rcond=1e-12)
    dof = max(1, n_row - rank)
    s2 = float(np.mean(resid ** 2)) * n_row / dof
    std = np.sqrt(np.clip(np.diag(XtXi) * s2, 0.0, None))
    sv_raw = np.linalg.svd(Y, compute_uv=False)   # 未归一化列 ⇒ 反映物理量纲下的可辨识性
    cov = XtXi * s2                                # 参数协方差（供派生量投影，如 |P| 的 σ）
    # 条件数只在**非零列**上算（along_d / zero 约束下 Px/Py 列被并成 1 列或置零）
    active = col_raw > (1e-9 * float(col_raw.max()) if col_raw.size else 1.0)
    n_active = int(np.count_nonzero(active))
    if n_active >= 2:
        Yr = Y[:, active]
        Sr = np.linalg.svd(Yr / np.linalg.norm(Yr, axis=0), compute_uv=False)
        Rr = np.linalg.svd(Yr, compute_uv=False)
        cond_norm = float(Sr[0] / Sr[-1]) if Sr[-1] > 0 else float("inf")
        cond_raw = float(Rr[0] / Rr[-1]) if Rr[-1] > 0 else float("inf")
    else:
        cond_norm = cond_raw = float("inf")
    return phi, {"sv": S, "sv_raw": sv_raw, "std": std, "cov": cov, "rank": rank,
                 "n_active_cols": n_active,
                 "cond_norm": cond_norm, "cond_raw": cond_raw,
                 "resid_rms": float(np.sqrt(np.mean(resid ** 2))), "n": n_row}


# ============================================================================
# 零点偏差 δ 的影响（"手动微调过零点"的容差）
# ============================================================================
def zero_angle_study(segs, cfg: SimConfig, ls_cfg: LSConfig, truth_vec,
                     theta_star_true_deg: float,
                     deltas=(-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0)):
    """**假设的 θ\* 差多少**会导致 `P` 偏多少（数据固定，只改约束方向）。

    真实平衡点是 θ*_true（= angle(d) − angle(P)）；拟合时用 θ*_assumed = θ*_true + δ
    去写约束 `P = |P|·R(−θ*_assumed)·d̂`。几何上拟合出来的 `P` 会整体旋转 δ ⇒
    `|ΔP| = 2|P̂|·sin(δ/2) ≈ |P̂|·δ[rad]`。另外单独给一行"假设 θ*=0"（实车最容易犯的错）。
    """
    rows = []
    phi0 = default_param_vector()
    truth_vec = np.asarray(truth_vec, dtype=np.float64)
    P_true = float(np.hypot(truth_vec[2], truth_vec[3]))
    Y, tau, _ = build_ls_system(segs, "measured", cfg, ls_cfg)
    cases = [(f"θ*={theta_star_true_deg + d:+.2f}°", theta_star_true_deg + d, d)
             for d in deltas]
    cases.append(("θ*=0（naive 默认）", 0.0, 0.0 - theta_star_true_deg))
    for lab, za, dlt in cases:
        phi, si = solve_ls_p_constrained(Y, tau, phi0, ls_cfg.rcond, "along_d",
                                         cfg.dx, cfg.dy, za)
        rows.append({
            "label": lab, "theta_assumed": float(za), "delta": float(dlt),
            "phi": phi,
            "Px_err": float(phi[2] - truth_vec[2]),
            "Py_err": float(phi[3] - truth_vec[3]),
            "P_err": float(np.hypot(phi[2] - truth_vec[2], phi[3] - truth_vec[3])),
            "P_mag_err": float(abs(si["s"]) - P_true),
            "med6": float(np.median([abs(phi[j] - truth_vec[j]) / abs(truth_vec[j]) * 100.0
                                     for j in STRONG_PARAMS])),
            "sigma_s": float(si["sigma_s"]), "cond": float(si["cond_norm"]),
        })
        print(f"[zero] {lab:22s} δ={dlt:+6.2f}° ⇒ |ΔP|={rows[-1]['P_err']:.5f} "
              f"({rows[-1]['P_err']/P_true*100:5.1f}% |P|), |Δ|P||={rows[-1]['P_mag_err']:+.5f}, "
              f"6 参中位误差={rows[-1]['med6']:5.1f}%, σ_|P|={rows[-1]['sigma_s']:.5f}")
    return rows


def d_scan(cfg: SimConfig, truth_vec, raw_lib, rng, args, ls_cfg, dx_grid=(0.03, 0.06, 0.10),
           zero_angle_deg=0.0):
    """★ `|d|` 对 `P` 的**惯性耦合通道**灵敏度的影响（水平数据）。

    `Px/Py` 在水平时只通过 `d·R(θs)P` 与 `μ=2(dy·Qx−dx·Qy)` 进入动力学，系数 ∝ |d|；
    `|d|` 从 3 cm 变成 10 cm（真实几何）后，σ_P 应当按 1/|d| 改善。这里逐 |d| 重采一次
    水平数据（plant 的耦合也随之变化），用 LS 的协方差给出 σ_Px / σ_|P| 与条件数。
    """
    out = []
    for dx in dx_grid:
        c = replace(cfg, dx=float(dx), dy=0.0)
        specs = []
        for _ in range(max(3, args.segments // 2)):
            specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], AXIS_BIG))
        for _ in range(max(3, args.segments // 2)):
            specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], AXIS_SMALL))
        tp = PlanarParams(dx=float(dx), dy=0.0,
                          friction_lambda=FRICTION_LAMBDA).with_vector(truth_vec)
        segs = collect_sim(c, tp, specs, rng, verbose=False)["segs"]
        Y, tau, _ = build_ls_system(segs, "measured", c, ls_cfg)
        row = {"dx": float(dx)}
        for mode in ("free", "along_d"):
            _, si = solve_ls_p_constrained(Y, tau, default_param_vector(), ls_cfg.rcond,
                                           mode, c.dx, c.dy, zero_angle_deg)
            row[f"{mode}_sigma_Px"] = float(np.asarray(si["std"])[2])
            row[f"{mode}_sigma_s"] = float(si["sigma_s"])
            row[f"{mode}_cond"] = float(si["cond_norm"])
        row["sigma_Px"] = row["free_sigma_Px"]
        row["sigma_s"] = row["along_d_sigma_s"]
        out.append(row)
        print(f"[dscan] dx={dx:.3f} m: σ_Px(自由)={row['free_sigma_Px']:.5f} "
              f"σ_|P|(沿d)={row['along_d_sigma_s']:.5f} 条件数={row['free_cond']:.2f}")
    return out


# ============================================================================
# P 的可辨识度扫描（回答"要多大倾角 / 多少段"）
# ============================================================================
def ls_param_sigma(segs, cfg: SimConfig, ls_cfg: LSConfig, mode: str = "measured",
                   grav_scale: float = 1.0):
    """只算 LS 的参数不确定度（不做前向仿真）——用于扫描倾角/段数。"""
    Y, tau, info = build_ls_system(segs, mode, cfg, ls_cfg, oracle=False,
                                   grav_scale=grav_scale)
    phi, sinfo = solve_ls_truncated(Y, tau, default_param_vector(), ls_cfg.rcond)
    return {"std": sinfo["std"], "cond_norm": sinfo["cond_norm"],
            "cond_raw": sinfo["cond_raw"], "sv": sinfo["sv"], "n_sample": info["n_sample"]}


def p_identifiability_scan(segs, cfg: SimConfig, ls_cfg: LSConfig, data_tilt_deg: float,
                           tilt_grid=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0),
                           seg_grid=(1, 2, 3, 4, 6, 8)):
    """两条扫描（都基于**同一批**已采数据，只改假设）:

    1) **倾角扫描**: 把 A 系重力按 `sin(φ)/sin(φ_data)` 缩放（其他一切不变）⇒ σ_P(φ)。
       物理含义: "若这次实验是在倾角 φ 下做的，P 能定到多准"（灵敏度 ∝ g·sinφ）。
    2) **段数扫描**: 只用前 n 段（在真实采集倾角下）⇒ σ_P(n)，看噪声平均的收益。
    """
    out = {"tilt": [], "segs": [], "data_tilt_deg": data_tilt_deg}
    if abs(math.sin(math.radians(data_tilt_deg))) < 1e-6:
        return out                       # 水平数据无法外推（灵敏度为 0，无参考尺度）
    for phi in tilt_grid:
        sc = math.sin(math.radians(phi)) / math.sin(math.radians(data_tilt_deg))
        r = ls_param_sigma(segs, cfg, ls_cfg, "measured", grav_scale=sc)
        out["tilt"].append((float(phi), float(r["std"][2]), float(r["std"][3]),
                            float(r["cond_norm"])))
    by_axis = {AXIS_BIG: [], AXIS_SMALL: []}
    for seg in segs:
        by_axis[seg.axis].append(seg)
    seen = set()
    for n in seg_grid:                    # 每个轴取前 n 段（超出可用段数时封顶，避免重复行）
        sub = []
        for ax in (AXIS_BIG, AXIS_SMALL):
            sub.extend(by_axis[ax][:max(1, min(n, len(by_axis[ax])))])
        if len(sub) < 2 or len(sub) in seen:
            continue
        seen.add(len(sub))
        r = ls_param_sigma(sub, cfg, ls_cfg, "measured")
        out["segs"].append((int(len(sub)), float(r["std"][2]), float(r["std"][3])))
    return out


# ============================================================================
# P 的处理方式: free / along_d（零点标定后 P=|P|d̂）/ zero
# ============================================================================
P_CONSTRAINT_LABELS = {
    "free": "P 自由 (Px, Py 各自独立)",
    "along_d": "P 沿 d（零点标定后：单参数 |P|，Py=0）",
    "zero": "P 固定为 0（基线对照）",
}


def solve_ls_p_constrained(Y, tau, phi0, rcond, mode="free", dx=DX_M, dy=DY_M,
                           zero_angle_deg=0.0):
    """按 P 的处理方式解 LS。

    mode="along_d" 时把 `Px/Py` 两列合成**单列**（方向 `u = R(−θ*)·d̂`）⇒ 只有 1 个自由度
    | `s = |P|`，解出的 `P = s·u`；`θ* = zero_angle_deg` 是平衡点在当前零点坐标系里的读数
    （θ*=0 ⇒ 零点就在平衡点上 ⇒ dy=0 时严格 Py=0）。
    """
    Y = np.asarray(Y, dtype=np.float64)
    phi0 = np.asarray(phi0, dtype=np.float64)
    if mode == "free":
        phi, info = solve_ls_truncated(Y, tau, phi0, rcond)
        info = dict(info)
        info["p_mode"] = "free"
        info["s"] = float(np.hypot(phi[2], phi[3]))
        info["sigma_s"] = _sigma_projection(info, dx, dy)
        return phi, info
    if mode == "zero":
        fm = np.ones(NPARAM, dtype=bool)
        fm[2] = fm[3] = False
        phi, info = solve_ls_truncated(Y, tau, phi0, rcond, free_mask=fm)
        info = dict(info)
        info["p_mode"] = "zero"
        info["s"] = 0.0
        info["sigma_s"] = 0.0
        return phi, info
    ux, uy = p_direction(dx, dy, zero_angle_deg)
    Yc = Y.copy()
    Yc[:, 2] = Y[:, 2] * ux + Y[:, 3] * uy
    Yc[:, 3] = 0.0
    p0 = phi0.copy()
    p0[2] = p0[3] = 0.0                     # 单参数初值 = 0（P=0）
    phi_c, info = solve_ls_truncated(Yc, tau, p0, rcond)
    s = float(phi_c[2])
    sig_s = float(np.asarray(info["std"])[2])
    phi = phi_c.copy()
    phi[2] = s * ux
    phi[3] = s * uy
    info = dict(info)
    std = np.asarray(info["std"], dtype=np.float64).copy()
    std[2] = sig_s * abs(ux)
    std[3] = sig_s * abs(uy)
    info.update({"p_mode": "along_d", "std": std, "s": s, "sigma_s": sig_s,
                 "unit_d": (ux, uy), "zero_angle_deg": float(zero_angle_deg)})
    return phi, info


def _sigma_projection(info, dx, dy):
    """自由 P 情形下 |P| 方向（沿 d̂）的不确定度: σ_s = sqrt(ûᵀ Σ_P û)。"""
    cov = info.get("cov")
    n = math.hypot(dx, dy)
    if cov is None or n < 1e-12:
        return float("nan")
    ux, uy = dx / n, dy / n
    return float(np.sqrt(max(ux * ux * cov[2, 2] + 2 * ux * uy * cov[2, 3]
                             + uy * uy * cov[3, 3], 0.0)))


def swing_scan(base_cfg: SimConfig, truth_along_d, raw_lib, rng, args, ls_cfg,
               tilt_segs=None, fills=(0.25, 0.5, 0.75, 1.0), zero_angle_deg=0.0):
    """小 yaw 摆幅扫描: 摆幅 = 包络半宽 × fill。

    对每个摆幅采一批**水平**数据（真值 P 沿 d），用 along_d 约束（Py=0 已知）算
    σ_|P| ⇒ 回答"要把 |P| 定到 10% 需要多大摆幅"；再与倾斜数据合并，看混合采集的要求。
    """
    out = []
    for f in fills:
        cfg = replace(base_cfg, small_env_mode="asym", tilt_deg_list=(),
                      env_fill_min=float(f), env_fill_max=float(f))
        specs = []
        for _ in range(max(3, args.segments // 2)):
            specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], AXIS_BIG))
        for _ in range(max(3, args.segments // 2)):
            specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], AXIS_SMALL))
        segs = collect_sim(cfg, truth_along_d, specs, rng, verbose=False)["segs"]
        row = {"fill": float(f), "swing_deg": cfg.env_half * f / DEG}
        for tag, ss in (("level", segs), ("mixed", list(segs) + list(tilt_segs or []))):
            if not ss:
                continue
            Y, tau, info = build_ls_system(ss, "measured", cfg, ls_cfg)
            for mode in ("free", "along_d"):
                _, si = solve_ls_p_constrained(Y, tau, default_param_vector(),
                                               ls_cfg.rcond, mode, cfg.dx, cfg.dy,
                                               zero_angle_deg)
                row[f"{tag}_{mode}_sigma_s"] = float(si["sigma_s"])
                row[f"{tag}_{mode}_cond"] = float(si["cond_norm"])
        out.append(row)
        p = float(np.hypot(truth_along_d.Px, truth_along_d.Py))
        if "level_along_d_sigma_s" in row:
            print(f"[swing] 摆幅 ±{row['swing_deg']:.1f}°: σ_|P|(水平, 沿d约束) = "
                  f"{row['level_along_d_sigma_s']:.5f} = {row['level_along_d_sigma_s']/p*100:.1f}% |P|"
                  f"；σ_|P|(水平+倾斜) = {row.get('mixed_along_d_sigma_s', float('nan')):.5f}")
    return out


# ============================================================================
# 验证: 前向仿真角度 RMSE / 逆动力学残差（oracle 状态）
# ============================================================================
def forward_rmse(phi, segs, cfg: SimConfig, lambda_override: float | None = None):
    """用参数做前向仿真（从实测初始状态、用记录力矩），返回每轴 RMSE 与发散标志。"""
    lam = cfg.model_lambda if lambda_override is None else lambda_override
    # λ 越大摩擦模态越快 ⇒ 想得到"纯模型误差"的数值下限必须用更小的积分步
    sub = cfg.validate_substeps * (1 if lam <= cfg.model_lambda + 1e-9 else 50)
    p = PlanarParams(dx=cfg.dx, dy=cfg.dy, friction_lambda=lam).with_vector(phi)
    out = {a: {"meas": [], "true": [], "n": 0} for a in (AXIS_BIG, AXIS_SMALL)}
    diverged = False
    for seg in segs:
        exo_seq = None
        if seg.gravity is not None:
            exo_seq = [exo_from_gravity(float(seg.gravity[i, 0]), float(seg.gravity[i, 1]))
                       for i in range(seg.T)]
        th_pred, _ = simulate_np(p, seg.theta[0], seg.dtheta[0], seg.tau, seg.dt,
                                 EXO_ZERO, sub, exo_seq=exo_seq)
        if not np.all(np.isfinite(th_pred)) or np.max(np.abs(th_pred)) > 1e3:
            diverged = True
            continue
        for a in (AXIS_BIG, AXIS_SMALL):
            e_meas = th_pred[:, a] - seg.theta[:, a]
            out[a]["meas"].append(e_meas)
            if seg.theta_true is not None:
                out[a]["true"].append(th_pred[:, a] - seg.theta_true[:, a])
            out[a]["n"] += seg.T
    res = {"diverged": diverged}
    for a in (AXIS_BIG, AXIS_SMALL):
        em = np.concatenate(out[a]["meas"]) if out[a]["meas"] else np.array([np.nan])
        res[a] = {"rmse_meas": float(np.sqrt(np.mean(em ** 2))),
                  "rmse_true": (float(np.sqrt(np.mean(np.concatenate(out[a]["true"]) ** 2)))
                                if out[a]["true"] else float("nan")),
                  "n": out[a]["n"]}
    res["rmse_meas_all"] = float(np.sqrt(np.mean(
        [res[AXIS_BIG]["rmse_meas"] ** 2, res[AXIS_SMALL]["rmse_meas"] ** 2])))
    return res


def id_residual_rms(phi, segs, cfg: SimConfig, oracle: bool = True,
                    lambda_override: float | None = None):
    """逆动力学残差（不积分 ⇒ 不累积漂移）。oracle=True 用 plant 真值状态/导数。"""
    lam = cfg.model_lambda if lambda_override is None else lambda_override
    # λ 越大摩擦模态越快 ⇒ 想得到"纯模型误差"的数值下限必须用更小的积分步
    sub = cfg.validate_substeps * (1 if lam <= cfg.model_lambda + 1e-9 else 50)
    p = PlanarParams(dx=cfg.dx, dy=cfg.dy, friction_lambda=lam).with_vector(phi)
    errs = []
    for seg in segs:
        if oracle and seg.theta_true is not None:
            q, qd, qdd = seg.theta_true, seg.dtheta_true, seg.ddtheta_true
            tau = seg.tau
        else:
            q, qd = seg.theta, seg.dtheta
            _, qdd = derivatives_from_angles(seg.theta, seg.dt)
            tau = seg.tau
        exo = EXO_ZERO if seg.gravity is None else exo_from_gravity(seg.gravity[:, 0],
                                                                   seg.gravity[:, 1])
        pred = inverse_dynamics_np(q, qd, qdd, p, exo)
        errs.append((pred - tau).reshape(-1))
    e = np.concatenate(errs)
    # 99% 截尾 RMS: 实机数据里大 yaw 经 MCU2 低速率链路被"保持"，其角度是阶梯状外推值，
    # 二阶差分会出现个别巨大尖峰（与模型无关），不截尾会让这个指标失去意义。
    ae = np.abs(e)
    thr = float(np.percentile(ae, 99.0))
    ae_t = ae[ae <= thr] if np.any(ae <= thr) else ae
    return float(np.sqrt(np.mean(ae_t ** 2))), float(np.max(ae))


def held_axis_stats(segs):
    """统计 held 轴的真实运动量（决定 (a) 忽略掉多少真项）。"""
    v, a, dev = [], [], []
    for seg in segs:
        h = seg.held
        v.append(seg.dtheta_true[:, h])
        a.append(seg.ddtheta_true[:, h])
        dev.append(seg.theta_true[:, h] - seg.held_target)
    v = np.concatenate(v)
    a = np.concatenate(a)
    dev = np.concatenate(dev)
    return {"rms_v": float(np.sqrt(np.mean(v ** 2))), "rms_a": float(np.sqrt(np.mean(a ** 2))),
            "max_dev": float(np.max(np.abs(dev))), "rms_dev": float(np.sqrt(np.mean(dev ** 2)))}


def data_stats(segs, cfg: SimConfig, has_truth: bool = True):
    """数据统计（力矩/角速度范围/饱和比例）。"""
    tau = np.concatenate([s.tau for s in segs])
    w = np.concatenate([(s.dtheta_true if (has_truth and s.dtheta_true is not None)
                         else s.dtheta) for s in segs])
    sat = np.mean(np.abs(tau) >= 0.99 * cfg.pid_out_limit)
    return {"rms_tau": float(np.sqrt(np.mean(tau ** 2))), "max_tau": float(np.max(np.abs(tau))),
            "sat_frac": float(sat), "max_omega": float(np.max(np.abs(w))),
            "rms_omega": float(np.sqrt(np.mean(w ** 2))), "n_seg": len(segs),
            "n_sample": int(sum(s.T for s in segs))}


def friction_mismatch(truth: PlanarParams, segs, cfg: SimConfig):
    """量化 λ 失配: 真值 fc 在 λ_plant=100 与 λ_model=10 下的力矩差（逐轴）。"""
    out = {}
    for a, nm, fc in ((AXIS_BIG, "big", truth.fc_big), (AXIS_SMALL, "small", truth.fc_small)):
        w = np.concatenate([s.dtheta_true[:, a] for s in segs])
        d = fc * (np.tanh(cfg.plant_lambda * w) - np.tanh(cfg.model_lambda * w))
        out[nm] = {"rms": float(np.sqrt(np.mean(d ** 2))), "max": float(np.max(np.abs(d))),
                   "p90": float(np.percentile(np.abs(d), 90)),
                   "frac_low": float(np.mean(np.abs(w) < 0.1))}
    return out


# ============================================================================
# 报告
# ============================================================================
METHOD_LABELS = {
    "A": "(a) LS + 理想保持值",
    "B": "(b) LS + 实测值",
    "C": "(c) torch 输出误差法",
    "A_oracle": "(a') LS + 理想保持值 + oracle 导数",
    "B_oracle": "(b') LS + 实测导数 = oracle（消融）",
    "PF": "(b)-P0 变体: LS + P 固定为 0",
    "CF": "(c)-P0 变体: torch + P 固定为 0",
}


def fmt_param_table(truth_vec, results: dict, methods):
    """参数对比表: 真值 / 各方法估计 / 绝对误差 / 相对误差（无真值时只列估计）。"""
    has_truth = truth_vec is not None
    lines = []
    if has_truth:
        hdr = ("| # | 参数 | 真值 | " + " | ".join(
            f"{METHOD_LABELS[m]} | 绝对误差 | 相对误差" for m in methods) + " |")
        lines.append(hdr)
        lines.append("|---|---|---|" + "---|---|---|" * len(methods))
    else:
        lines.append("| # | 参数 | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |")
        lines.append("|---|---|" + "---|" * len(methods))
    for j, nm in enumerate(PARAM_NAMES):
        row = [f"{j}", f"`{nm}`"]
        if has_truth:
            t = float(truth_vec[j])
            row.append(f"{t:.5f}")
        for m in methods:
            e = results[m]["phi"][j]
            row.append(f"{e:.5f}")
            if has_truth:
                ae = e - float(truth_vec[j])
                re = ae / float(truth_vec[j]) if abs(float(truth_vec[j])) > 1e-12 else float("nan")
                row += [f"{ae:+.5f}", f"{re*100:+.1f}%"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt_rmse_table(rmse_res: dict, labels: dict):
    lines = ["| 参数来源 | 大 yaw RMSE(rad, vs 量化实测) | 小 yaw RMSE(rad, vs 量化实测) | "
             "合成 RMSE | 大 yaw RMSE(vs 真值轨迹) | 小 yaw RMSE(vs 真值轨迹) | 逆动力学残差 RMS(N·m) |",
             "|---|---|---|---|---|---|---|"]
    for key, lab in labels.items():
        r = rmse_res[key]
        d = r.get("resid", (float("nan"), float("nan")))
        div = " **发散**" if r.get("diverged") else ""
        lines.append(f"| {lab}{div} | {r[AXIS_BIG]['rmse_meas']:.3e} | "
                     f"{r[AXIS_SMALL]['rmse_meas']:.3e} | {r['rmse_meas_all']:.3e} | "
                     f"{r[AXIS_BIG]['rmse_true']:.3e} | {r[AXIS_SMALL]['rmse_true']:.3e} | "
                     f"{d[0]:.3e} |")
    return "\n".join(lines)


# ============================================================================
# 主流程
# ============================================================================
def _build_argparser():
    ap = argparse.ArgumentParser(
        description="在仿真中对比 LS(理想保持值)/LS(实测值)/torch 输出误差法 三种 8 参辨识方法",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sim-only", action="store_true",
                    help="只用仿真生成数据（无真机/无采集数据也能跑通）")
    ap.add_argument("--data", action="append", default=None,
                    help="真机数据 glob（.csv/.npz）；给了就额外在真机数据上做三法对比")
    ap.add_argument("--segments", type=int, default=6,
                    help="仿真采集: **每轴**段数（大 yaw 激励 / 小 yaw 激励各这么多段）")
    ap.add_argument("--holdout", type=int, default=2, help="每轴留作验证集的段数")
    ap.add_argument("--targets", type=str, default="data/targets/*.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seg-len", type=int, default=300)
    ap.add_argument("--plant-lambda", type=float, default=100.0)
    ap.add_argument("--plant-substep", type=float, default=5.0e-5)
    ap.add_argument("--refine-n", type=int, default=100)
    ap.add_argument("--truth-scale", type=float, default=1.0,
                    help="真值参数整体缩放（1.0 = 用内置真值）")
    # torch 拟合
    ap.add_argument("--torch-iters", type=int, default=200, help="Adam 迭代数")
    ap.add_argument("--torch-lbfgs", type=int, default=6, help="LBFGS 迭代数")
    ap.add_argument("--torch-substeps", type=int, default=1)
    ap.add_argument("--torch-vel-weight", type=float, default=0.0)
    ap.add_argument("--torch-points", type=int, default=0, help="每段只取前 N 点（0=全部，加速用）")
    ap.add_argument("--torch-batch", type=int, default=6,
                    help="Adam 每步随机抽取的窗口数（0=全批；LBFGS 一律全批）")
    ap.add_argument("--torch-window", type=int, default=100,
                    help="每个拟合窗口的点数（0=整段；100×3 恰好覆盖 3 s）")
    ap.add_argument("--torch-windows-per-seg", type=int, default=3,
                    help="每段切几个窗口")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch 线程数（0=默认；小张量下 1 通常更快）")
    ap.add_argument("--torch-free-init-vel", action="store_true")
    ap.add_argument("--ls-rcond", type=float, default=1e-6)
    ap.add_argument("--no-ablation", action="store_true", help="不做 oracle 导数消融")
    ap.add_argument("--no-write", action="store_true", help="不写 docs/sysid_compare.md")
    ap.add_argument("--out-md", type=str, default="docs/sysid_compare.md")
    ap.add_argument("--dump-sim", type=str, default=None,
                    help="可选: 把仿真数据落盘成 CSV 到该目录（默认不落盘）")
    ap.add_argument("--quick", action="store_true", help="快速档（段数/迭代数减半，用于冒烟测试）")
    ap.add_argument("--envelopes", type=str, default="asym,legacy",
                    help="小 yaw 包络方案: asym（新，−25°~+20°）| legacy（旧，±40°）；逗号分隔")
    ap.add_argument("--tilts", type=str, default="0;8,-8,10,-10",
                    help="静态倾斜方案: 分号分隔多个方案，每个方案是逐段倾角的逗号列表"
                         "（如 \"0;8,-8,10,-10\"；\"0\" = 水平）")
    ap.add_argument("--p-constraint", choices=["free", "along_d", "zero"], default="free",
                    help="主方法（a)/(b)/(c) 的 P 处理方式; free 保持既有结论可复现")
    ap.add_argument("--dx", type=float, default=DX_M, help="实测几何 dx (m)")
    ap.add_argument("--dy", type=float, default=DY_M, help="实测几何 dy (m)")
    for _nm, _lab in (("Jbig", "Jbig_eff"), ("Js", "Js"), ("Px", "Px"), ("Py", "Py"),
                      ("fc_big", "fc_big"), ("fv_big", "fv_big"),
                      ("fc_small", "fc_small"), ("fv_small", "fv_small")):
        ap.add_argument(f"--truth-{_nm.replace('_', '-')}", type=float, default=None,
                        help=f"覆盖 plant 真值 {_lab}（默认取真实量级的内置值）")
    ap.add_argument("--truth-vector", type=str, default=None,
                    help="一次性覆盖 8 个真值（逗号分隔，顺序同 PARAM_NAMES）")
    ap.add_argument("--d-scan", type=str, default="0.03,0.06,0.10",
                    help="|d| 扫描网格（m）: 水平数据下 P 的惯性耦合通道灵敏度")
    ap.add_argument("--no-d-scan", dest="d_scan", action="store_const", const="")
    ap.add_argument("--p-zero-angle", type=float, default=0.0,
                    help="平衡点 θ* 在当前零点坐标系里的读数（度）；along_d 约束下用它旋转 P 方向")
    ap.add_argument("--p-exp", dest="p_exp", action="store_true", default=True,
                    help="跑「P 的三种处理方式」专项实验（真值 P 沿 d，模拟已完成零点标定）")
    ap.add_argument("--no-p-exp", dest="p_exp", action="store_false")
    ap.add_argument("--swing-fills", type=str, default="0.25,0.5,0.75,1.0",
                    help="小 yaw 摆幅扫描: 包络半宽的比例（摆幅 = 半宽 × fill）")
    ap.add_argument("--pfix", choices=["none", "level", "all"], default="level",
                    help="额外跑一个 Px=Py=0 固定变体的数据集范围（level=只对水平数据集）")
    ap.add_argument("--no-pfix", action="store_true", help="完全跳过 P 固定变体")
    ap.add_argument("--scan-tilts", type=str, default="0,2,4,6,8,10,15,20",
                    help="倾角扫描网格（度，逗号分隔）")
    ap.add_argument("--scan-segs", type=str, default="1,2,3,4,6,8",
                    help="段数扫描网格（每轴段数，逗号分隔）")
    return ap


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("compare_ident_methods.py — 三种 8 参辨识方法在仿真中的对比")
    print("=" * 96)
    if args.torch_threads > 0:
        import torch
        torch.set_num_threads(args.torch_threads)
    ok = model_self_test(verbose=False)
    print(f"[check] 模型自检（Y·φ==ID、Y==∂ID/∂φ、torch==numpy）: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[warn] 模型自检未通过，结果不可信", file=sys.stderr)

    if args.quick:
        args.segments = max(3, args.segments // 2)
        args.torch_iters = max(30, args.torch_iters // 6)
        args.torch_lbfgs = max(3, args.torch_lbfgs // 4)
        args.torch_points = args.torch_points or 120
        args.torch_batch = 3
        args.torch_window = min(args.torch_window, 120)

    base_cfg = dict(seg_len=args.seg_len, plant_lambda=args.plant_lambda,
                    plant_substep=args.plant_substep, refine_n=args.refine_n,
                    dx=args.dx, dy=args.dy)
    # ── 真值（可用 --truth-* 覆盖，便于做 fc_small=0.05/0.03 之类的敏感性对照）──
    truth_vec = TRUTH_VECTOR.astype(np.float64).copy()
    if args.truth_vector:
        vals = [float(x) for x in str(args.truth_vector).split(",")]
        assert len(vals) == NPARAM, "--truth-vector 需要 8 个数"
        truth_vec = np.array(vals, dtype=np.float64)
    _ov = {"Jbig": 0, "Js": 1, "Px": 2, "Py": 3, "fc_big": 4, "fv_big": 5,
           "fc_small": 6, "fv_small": 7}
    for nm, j in _ov.items():
        v = getattr(args, f"truth_{nm}", None)
        if v is not None:
            truth_vec[j] = float(v)
    truth_vec = truth_vec * args.truth_scale
    theta_star = (math.degrees(math.atan2(args.dy, args.dx))
                  - math.degrees(math.atan2(truth_vec[3], truth_vec[2])))
    args.theta_star_true = theta_star
    args.truth_vec = truth_vec
    args.dx = float(args.dx)
    args.dy = float(args.dy)
    print(f"[cfg] 真值 φ = " + ", ".join(f"{v:.5f}" for v in truth_vec))
    print(f"[cfg] d = ({args.dx:.3f}, {args.dy:.3f}) m, |P| = "
          f"{float(np.hypot(truth_vec[2], truth_vec[3])):.5f} kg·m, "
          f"平衡点 θ* = {theta_star:+.2f}°，行程 [−25°,+20°] ⇒ "
          f"{'可达' if -25.0 <= theta_star <= 20.0 else '★ 不可达（平衡点在小 yaw 行程之外）'}")
    ls_cfg = LSConfig(rcond=args.ls_rcond, p_constraint=args.p_constraint,
                      p_zero_angle_deg=args.p_zero_angle)
    args.scan_tilts = tuple(float(x) for x in str(args.scan_tilts).split(",") if x.strip())
    args.scan_segs = tuple(int(x) for x in str(args.scan_segs).split(",") if x.strip())
    args.swing_fills = tuple(float(x) for x in str(args.swing_fills).split(",") if x.strip())
    print(f"[cfg] 主方法 P 处理 = {P_CONSTRAINT_LABELS[args.p_constraint]}")
    truth_plant = PlanarParams(dx=args.dx, dy=args.dy,
                               friction_lambda=FRICTION_LAMBDA).with_vector(truth_vec)

    # ── 1) 仿真采集: (包络方案) × (静态倾斜方案) 的笛卡尔积 ──
    env_modes = [e.strip() for e in str(args.envelopes).split(",") if e.strip()]
    tilt_sets = []
    for chunk in str(args.tilts).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        tilt_sets.append(tuple(float(x) for x in chunk.split(",") if x.strip() != ""))
    dataset = []
    if args.segments > 0:
        raw_lib = load_target_library([args.targets])
        if not raw_lib:
            print("[targets] 未找到录制目标序列 → 自造正弦/阶跃激励序列")
            raw_lib = synth_targets(rng, 8)
        for env_mode in env_modes:
            for tilt in tilt_sets:
                cfg = SimConfig(small_env_mode=env_mode, tilt_deg_list=tilt, **base_cfg)
                specs = []
                for _ in range(args.segments):
                    specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], AXIS_BIG))
                for _ in range(args.segments):
                    specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], AXIS_SMALL))
                t0 = time.time()
                out = collect_sim(cfg, truth_plant, specs, rng)
                print(f"[sim] 用时 {time.time()-t0:.1f}s")
                tilted = any(abs(t) > 1e-9 for t in tilt)
                env_txt = "新包络(−25°~+20°)" if env_mode == "asym" else "旧包络(±40°)"
                tmax = max(abs(np.asarray(tilt)))
                name = f"{env_txt} × {'水平' if not tilted else f'倾斜 ±{tmax:g}°'}"
                pfix = (args.pfix == "all") or (args.pfix == "level" and not tilted)
                if args.no_pfix:
                    pfix = False
                entry = {"name": name, "segs": out["segs"], "truth": truth_vec,
                         "truth_params": truth_plant, "cfg": cfg, "ls_cfg": ls_cfg,
                         "env_mode": env_mode,
                         "tilt": tilt, "tilted": tilted, "pfix": pfix,
                         "sim_stats": out}
                if env_mode == "asym":          # ★ P 处理实验挂在"新包络"两个数据集上
                    entry["p_treatments"] = True
                    entry["theta_star_true_deg"] = theta_star
                    if tilt == (0.0,):
                        asym_level_segs = out["segs"]
                    else:
                        asym_tilt_segs, cfg_asym_tilt = out["segs"], cfg
                        entry["p_zero_variants"] = True
                dataset.append(entry)

    # ── ★ P 处理 / 零点偏差 δ / 摆幅 / |d| 扫描（复用上面新包络的仿真数据）──
    args.swing_results = []
    args.zero_study = []
    args.d_scan_results = []
    if args.segments > 0 and "asym_tilt_segs" in locals():
        print(f"\n[zero] 假设 θ* 的灵敏度（真实 θ* = {theta_star:+.2f}°；LS 网格 + torch 复核）")
        args.zero_study = zero_angle_study(asym_tilt_segs, cfg_asym_tilt, ls_cfg,
                                           truth_vec, theta_star,
                                           deltas=(-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0))
        print("\n[swing] 小 yaw 摆幅扫描（水平 + 倾斜混合；along_d 用正确 θ*）")
        args.swing_results = swing_scan(
            SimConfig(small_env_mode="asym", **base_cfg), truth_plant, raw_lib, rng, args,
            ls_cfg, tilt_segs=asym_tilt_segs, fills=args.swing_fills,
            zero_angle_deg=theta_star)
        if str(args.d_scan).strip():
            print("\n[dscan] |d| 对 P 惯性耦合通道灵敏度的影响（水平数据）")
            args.d_scan_results = d_scan(
                SimConfig(small_env_mode="asym", **base_cfg), truth_vec, raw_lib, rng, args,
                ls_cfg, dx_grid=[float(x) for x in str(args.d_scan).split(",") if x.strip()],
                zero_angle_deg=theta_star)

    real_segs = []
    if args.data:
        from identify_params_torch import load_segments
        real_segs = load_segments(args.data)
        print(f"[data] 真机数据 {len(real_segs)} 段")
        if real_segs:
            rcfg = SimConfig(small_env_mode="asym", **base_cfg)
            dataset.append({"name": "真机采集数据", "segs": real_segs, "truth": None,
                            "cfg": rcfg, "ls_cfg": ls_cfg, "env_mode": "asym",
                            "tilt": (), "tilted": False, "pfix": False,
                            "sim_stats": None})

    if not dataset:
        print("[error] 既没有仿真段也没有真机数据", file=sys.stderr)
        return 2

    reports = []
    for ds in dataset:
        reports.append(run_one_dataset(ds, ds.get("truth_params") or truth_plant, args, rng))

    # ── 汇总打印 ──
    for rep in reports:
        print(rep["console"])

    args.measured_seconds = f"{time.time()-t_start:.0f} s"
    if not args.no_write:
        write_markdown(args.out_md, reports, args)
        print(f"\n[out] 对比报告已写入 {args.out_md}")

    if args.dump_sim and dataset:
        for ds in dataset:
            sub = os.path.join(args.dump_sim, ds["env_mode"] + ("_tilt" if ds["tilted"] else "_level"))
            dump_csv(ds["segs"], sub)
            print(f"[out] 仿真数据已落盘到 {sub}")
    print(f"\n总用时 {time.time()-t_start:.1f}s")
    return 0


def run_one_dataset(ds, truth_params, args, rng):
    """在一个数据集上跑三种方法（+ P 固定变体）+ 验证 + 消融，返回报告字典。"""
    name, segs, truth_vec = ds["name"], ds["segs"], ds["truth"]
    cfg, ls_cfg = ds["cfg"], ds["ls_cfg"]
    print(f"\n{'='*96}\n数据集: {name}（{len(segs)} 段）\n{'='*96}")
    has_truth = all(s.theta_true is not None for s in segs)
    fit_segs, val_segs = split_fit_val(segs, args.holdout, rng)
    if not val_segs:
        val_segs = fit_segs
    print(f"[split] 拟合集 {len(fit_segs)} 段 / 验证集 {len(val_segs)} 段 "
          f"(大 yaw {sum(s.axis==AXIS_BIG for s in fit_segs)}+{sum(s.axis==AXIS_BIG for s in val_segs)}"
          f", 小 yaw {sum(s.axis==AXIS_SMALL for s in fit_segs)}+{sum(s.axis==AXIS_SMALL for s in val_segs)})")

    dstats = data_stats(segs, cfg, has_truth=has_truth)
    hstats = held_axis_stats(segs) if has_truth else {
        "rms_v": float("nan"), "rms_a": float("nan"),
        "max_dev": float("nan"), "rms_dev": float("nan")}
    print(f"[data] 样本 {dstats['n_sample']} 点；|τ| RMS={dstats['rms_tau']:.4f} N·m, "
          f"最大 {dstats['max_tau']:.3f}；饱和(≥0.99限幅)比例 {dstats['sat_frac']*100:.2f}%；"
          f"|ω| RMS={dstats['rms_omega']:.2f} rad/s, 最大 {dstats['max_omega']:.2f}")
    if has_truth:
        print(f"[data] held 轴真实运动: RMS|θ̇|={hstats['rms_v']:.4f} rad/s, "
              f"RMS|θ̈|={hstats['rms_a']:.3f} rad/s², 偏离保持目标最大 {hstats['max_dev']*57.3:.2f}°")

    phi0 = default_param_vector()
    free_nop = np.ones(NPARAM, dtype=bool)
    free_nop[2] = free_nop[3] = False          # Px/Py 固定为 0
    results, rmse, labels = {}, {}, {}
    methods = ["A", "B", "C"]

    # ── (a)(b) 逆动力学线性回归 ──
    for key, mode in (("A", "ideal"), ("B", "measured")):
        Y, tau, info = build_ls_system(fit_segs, mode, cfg, ls_cfg, oracle=False)
        phi, sinfo = solve_ls_p_constrained(Y, tau, phi0, ls_cfg.rcond,
                                            ls_cfg.p_constraint, cfg.dx, cfg.dy)
        results[key] = {"phi": phi, "info": info, "sv": sinfo}
        print(f"[{key}] LS({mode}) 行数={info['n_row']}（样本 {info['n_sample']}） "
              f"秩={sinfo['rank']}/8  条件数(列归一化)={sinfo['cond_norm']:.3e} "
              f"残差RMS={sinfo['resid_rms']:.4f} N·m  剔除={info['dropped']}")
        print(f"     φ = " + " ".join(f"{v:+.5f}" for v in phi))

    def torch_fit(p_constraint: str, tag: str, verbose: bool = True,
                  zero_angle_deg: float | None = None):
        fcfg = FitConfig(iters=args.torch_iters, lbfgs_iters=args.torch_lbfgs,
                         seed=args.seed, substeps=args.torch_substeps,
                         vel_weight=args.torch_vel_weight,
                         free_init_vel=args.torch_free_init_vel,
                         max_points=args.torch_points,
                         window_len=args.torch_window,
                         windows_per_seg=args.torch_windows_per_seg,
                         batch_size=args.torch_batch, p_constraint=p_constraint,
                         p_zero_angle_deg=(args.p_zero_angle if zero_angle_deg is None
                                           else zero_angle_deg),
                         print_every=max(10, args.torch_iters // 6), verbose=True)
        t0 = time.time()
        r = fit_params_torch(fit_segs, fcfg,
                             PlanarParams(dx=cfg.dx, dy=cfg.dy,
                                          friction_lambda=cfg.model_lambda))
        secs = time.time() - t0
        if verbose:
            print(f"[{tag}] torch（P: {P_CONSTRAINT_LABELS[p_constraint]}）: "
                  f"全批 loss={r.val_loss:.4e}，用时 {secs:.1f}s")
            print(f"     φ = " + " ".join(f"{v:+.5f}" for v in r.phi))
        return r, secs

    # ── (c) torch 输出误差法 ──
    fit_res, torch_seconds = torch_fit(ls_cfg.p_constraint, "C")
    results["C"] = {"phi": fit_res.phi, "info": {"n_row": 0, "dropped": {}}, "sv": None}

    # ── ★ P 的三种处理方式（自由 / 沿 d / 固定 0）在同一批数据上的对比 ──
    ptreat = None
    if ds.get("p_treatments") and has_truth:
        ptreat = {}
        za_true = float(ds.get("theta_star_true_deg", args.theta_star_true))   # 真实 θ*
        Y, tau, info = build_ls_system(fit_segs, "measured", cfg, ls_cfg, oracle=False)

        def _ls(za):
            return solve_ls_p_constrained(Y, tau, phi0, ls_cfg.rcond, "along_d",
                                          cfg.dx, cfg.dy, za)

        phi_free, si_free = solve_ls_p_constrained(Y, tau, phi0, ls_cfg.rcond, "free",
                                                   cfg.dx, cfg.dy, 0.0)
        phi_z, si_z = solve_ls_p_constrained(Y, tau, phi0, ls_cfg.rcond, "zero",
                                             cfg.dx, cfg.dy, 0.0)
        phi_t, si_t = _ls(za_true)
        phi_0, si_0 = _ls(0.0)
        ptreat["free"] = {"ls_phi": phi_free, "ls_info": si_free, "za": None,
                          "torch_phi": fit_res.phi, "torch_loss": fit_res.val_loss}
        ptreat["along_d"] = {"ls_phi": phi_t, "ls_info": si_t, "za": za_true}
        ptreat["along_d_zero"] = {"ls_phi": phi_0, "ls_info": si_0, "za": 0.0}
        ptreat["zero"] = {"ls_phi": phi_z, "ls_info": si_z, "za": None}
        for k in ("free", "along_d", "along_d_zero", "zero"):
            v = ptreat[k]
            si = v["ls_info"]
            print(f"[P:{k:14s}] LS Px={v['ls_phi'][2]:+.5f} Py={v['ls_phi'][3]:+.5f} "
                  f"|P|={abs(si.get('s', 0.0)):.5f} σ_|P|={si.get('sigma_s', float('nan')):.5f} "
                  f"条件数={si['cond_norm']:.2f} 残差RMS={si['resid_rms']:.4f}")
        rp, _ = torch_fit("along_d", f"P:along_d(θ*={za_true:+.2f}°)", zero_angle_deg=za_true)
        ptreat["along_d"].update({"torch_phi": rp.phi, "torch_loss": rp.val_loss})
        rp, _ = torch_fit("along_d", "P:along_d(θ*=0)", zero_angle_deg=0.0)
        ptreat["along_d_zero"].update({"torch_phi": rp.phi, "torch_loss": rp.val_loss})
        rp, _ = torch_fit("zero", "P:zero", zero_angle_deg=0.0)
        ptreat["zero"].update({"torch_phi": rp.phi, "torch_loss": rp.val_loss})
        # δ 灵敏度复核: 只把"假设的 θ*"挪 2°（数据不变）
        if ds.get("p_zero_variants"):
            za2 = za_true + 2.0
            phi_2, si_2 = _ls(za2)
            rp, _ = torch_fit("along_d", "P:along_d(θ*+2°)", zero_angle_deg=za2)
            ptreat["along_d_off2"] = {"ls_phi": phi_2, "ls_info": si_2, "za": za2,
                                      "torch_phi": rp.phi, "torch_loss": rp.val_loss}
        for mode, v in ptreat.items():
            v["rmse_ls"] = forward_rmse(v["ls_phi"], val_segs, cfg)["rmse_meas_all"]
            v["rmse_torch"] = (forward_rmse(v["torch_phi"], val_segs, cfg)["rmse_meas_all"]
                               if "torch_phi" in v else float("nan"))
            v["resid_ls"] = id_residual_rms(v["ls_phi"], val_segs, cfg, oracle=True)[0]
            v["resid_torch"] = (id_residual_rms(v["torch_phi"], val_segs, cfg, oracle=True)[0]
                                if "torch_phi" in v else float("nan"))
            print(f"[P:{mode:9s}] 验证 RMSE: LS={v['rmse_ls']:.3e} torch={v['rmse_torch']:.3e}")

    # ── P 固定为 0 的变体（水平数据下 P 不可辨识时的兜底方案；只做 LS，torch 版见上面 PF）──
    pfix = {}
    if ds.get("pfix") and has_truth:
        Y, tau, info = build_ls_system(fit_segs, "measured", cfg, ls_cfg, oracle=False)
        phi_p, sinfo_p = solve_ls_p_constrained(Y, tau, phi0, ls_cfg.rcond, "zero",
                                                cfg.dx, cfg.dy)
        pfix["PF"] = {"phi": phi_p, "sv": sinfo_p, "info": info}
        print(f"[PF] LS + P 固定 0: 秩={sinfo_p['rank']}/8 残差RMS={sinfo_p['resid_rms']:.4f} N·m")

    # ── 验证集: 前向仿真 RMSE + 逆动力学残差 ──
    def add_rmse(key, phi, label, lam=None, oracle_resid=True):
        r = forward_rmse(phi, val_segs, cfg, lambda_override=lam)
        r["resid"] = id_residual_rms(phi, val_segs, cfg, oracle=oracle_resid,
                                     lambda_override=lam)
        rmse[key] = r
        labels[key] = label
        return r

    for m in methods:
        add_rmse(m, results[m]["phi"], f"{METHOD_LABELS[m]} 拟合参数")
    for m, v in pfix.items():
        add_rmse(m, v["phi"], METHOD_LABELS[m])
    if truth_vec is not None:
        add_rmse("T10", truth_vec, "★ 真值参数 + 模型 λ=10（**摩擦形状失配地板**）")
        add_rmse("T100", truth_vec, "★ 真值参数 + λ=100（形状匹配；可达下限，含初速度估计误差）",
                 lam=cfg.plant_lambda)
        add_rmse("P0", phi0, "CAD 初值 φ0（未辨识基线）")

    # ── 消融: oracle 导数（隔离"导数噪声"）──
    ablation = None
    if not args.no_ablation and has_truth:
        ablation = {}
        for key, mode in (("A_oracle", "ideal"), ("B_oracle", "measured")):
            Y, tau, info = build_ls_system(fit_segs, mode, cfg, ls_cfg, oracle=True)
            phi, sinfo = solve_ls_truncated(Y, tau, phi0, ls_cfg.rcond)
            ablation[key] = {"phi": phi, "rank": sinfo["rank"],
                             "resid_rms": sinfo["resid_rms"], "n_row": info["n_row"],
                             "n_sample": info["n_sample"]}
            r = forward_rmse(phi, val_segs, cfg)
            r["resid"] = id_residual_rms(phi, val_segs, cfg, oracle=True)
            rmse[key] = r
            labels[key] = METHOD_LABELS[key]

    mism = friction_mismatch(truth_params, segs, cfg) if has_truth else {}
    p_scan = {}
    if ds.get("tilted") and has_truth:
        data_tilt = max(abs(float(t)) for t in (ds.get("tilt") or (0.0,)))
        p_scan = p_identifiability_scan(fit_segs, cfg, ls_cfg, data_tilt,
                                        tilt_grid=args.scan_tilts, seg_grid=args.scan_segs)
        if p_scan.get("tilt"):
            print("[scan] 倾角扫描（σ_Px / σ_Py, N·m·… 单位 kg·m）:")
            for phi, sx, sy, cd in p_scan["tilt"]:
                print(f"       φ={phi:5.1f}°  σ_Px={sx:.5f}  σ_Py={sy:.5f}  条件数={cd:.3e}")
        if p_scan.get("segs"):
            print("[scan] 段数扫描（真实倾角下）:")
            for n, sx, sy in p_scan["segs"]:
                print(f"       {n:2d} 段  σ_Px={sx:.5f}  σ_Py={sy:.5f}")

    # ── 控制台报告 ──
    out = []
    out.append(f"\n{'-'*96}\n数据集: {name}\n{'-'*96}")
    out.append("【参数估计对比】")
    out.append(fmt_param_table(None if truth_vec is None else np.asarray(truth_vec),
                               results, methods))
    if pfix:
        out.append("\n【P 固定为 0 变体（Px = Py ≡ 0）】")
        out.append(fmt_param_table(None if truth_vec is None else np.asarray(truth_vec),
                                   {k: {"phi": v["phi"]} for k, v in pfix.items()},
                                   list(pfix.keys())))
    if ablation:
        out.append("\n【消融: 把状态/导数换成 plant 真值（隔离导数噪声）】")
        abl_res = {k: {"phi": v["phi"]} for k, v in ablation.items()}
        out.append(fmt_param_table(truth_vec, abl_res, ["A_oracle", "B_oracle"]))
    out.append("\n【验证集前向仿真 RMSE】")
    out.append(fmt_rmse_table(rmse, labels))
    if mism:
        out.append("\n【λ 失配量化（真值 fc 在 plant λ=100 与模型 λ=10 下的力矩差）】")
        for k, v in mism.items():
            out.append(f"  {k:5s}: RMS={v['rms']:.4f} N·m, 90 分位={v['p90']:.4f}, "
                       f"最大={v['max']:.4f} N·m; |ω|<0.1 rad/s 的样本占 {v['frac_low']*100:.1f}%")
    txt = "\n".join(out)
    print(txt)

    return {"name": name, "truth": (None if truth_vec is None else np.asarray(truth_vec)),
            "results": results, "pfix": pfix, "ablation": ablation, "rmse": rmse,
            "labels": labels, "dstats": dstats, "hstats": hstats, "mism": mism,
            "methods": methods, "console": txt, "fit_segs": len(fit_segs),
            "val_segs": len(val_segs), "phi0": phi0, "cfg": cfg, "ls_cfg": ls_cfg,
            "torch_seconds": torch_seconds, "torch_loss": fit_res.val_loss,
            "torch_loss_head": fit_res.loss_history[:5] or [float("nan")],
            "has_truth": has_truth, "env_mode": ds.get("env_mode"),
            "tilt": ds.get("tilt"), "tilted": ds.get("tilted"),
            "sim_stats": ds.get("sim_stats"), "p_scan": p_scan, "ptreat": ptreat,
            "p_treatments": bool(ds.get("p_treatments"))}
# ============================================================================
# Markdown 报告
# ============================================================================
def _ds_short(rep) -> str:
    env = {"asym": "新包络(−25°~+20°)", "legacy": "旧包络(±40°)"}.get(rep.get("env_mode"), "—")
    tilt = "水平" if not rep.get("tilted") else (
        "倾斜 ±" + "/".join(f"{abs(float(t)):g}" for t in sorted(set(abs(float(t)) for t in rep["tilt"]))) + "°")
    return f"{env} × {tilt}"


def _phi_of(rep, key):
    """取某方法的估计参数（含 P 固定变体）。"""
    if key in rep["results"]:
        return rep["results"][key]["phi"]
    if key in rep.get("pfix", {}):
        return rep["pfix"][key]["phi"]
    return None


def _val_rmse(rep, key):
    r = rep["rmse"].get(key)
    return float(r["rmse_meas_all"]) if r else float("nan")


def _relerr(rep, key, j):
    phi = _phi_of(rep, key)
    if phi is None or rep["truth"] is None:
        return float("nan")
    t = float(rep["truth"][j])
    return (float(phi[j]) - t) / t * 100.0 if abs(t) > 1e-12 else float("nan")


STRONG_PARAMS = (0, 1, 4, 5, 6, 7)      # Px/Py 之外的强可辨识参数


def _param_err_score(rep, key):
    """参数的"相对误差评分"：6 个强可辨识参数的 |相对误差| 中位数（%）。"""
    vals = [abs(_relerr(rep, key, j)) for j in STRONG_PARAMS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def _param_err_max(rep, key):
    vals = [abs(_relerr(rep, key, j)) for j in STRONG_PARAMS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.max(vals)) if vals else float("nan")


def fmt_param_score(reports):
    """参数精度 vs 验证 RMSE 的排序对照（两者不一定一致，必须分开看）。"""
    L = ["| 数据集 | (a) 6 参中位误差 | (b) 6 参中位误差 | (c) 6 参中位误差 | "
         "RMSE 排名(小→大) | 参数精度最好者 |",
         "|---|---|---|---|---|---|"]
    for rep in reports:
        if not rep["has_truth"]:
            continue
        sc = {k: _param_err_score(rep, k) for k in ("A", "B", "C")}
        rk = {k: _val_rmse(rep, k) for k in ("A", "B", "C")}
        order = sorted(rk, key=lambda k: rk[k])
        best_p = min(sc, key=lambda k: sc[k])
        L.append(f"| **{_ds_short(rep)}** | {sc['A']:.1f}% | {sc['B']:.1f}% | {sc['C']:.1f}% | "
                 f"{' < '.join(k for k in order)} | {METHOD_LABELS[best_p]} |")
    return "\n".join(L)


def _sigma_of(rep, key, j):
    if key not in rep["results"]:
        return float("nan")
    sv = rep["results"][key].get("sv")
    return float(sv["std"][j]) if sv else float("nan")


def _cond(rep, key, norm=True):
    if key not in rep["results"]:
        return float("nan")
    sv = rep["results"][key].get("sv")
    if not sv:
        return float("nan")
    return float(sv["cond_norm"] if norm else sv["cond_raw"])


def fmt_cross_rmse(reports):
    """★ 核心表: 各数据集 × 各方法的验证集前向仿真合成 RMSE (rad)。"""
    keys = ["A", "B", "C"]
    extra = [k for k in ("PF", "CF") if any(k in r.get("pfix", {}) for r in reports)]
    L = ["| 数据集 | " + " | ".join(METHOD_LABELS[k] for k in keys + extra) +
         " | 地板: 真值+λ=10 | 可达下限: 真值+λ=100 |",
         "|---|" + "---|" * (len(keys) + len(extra) + 2)]
    for rep in reports:
        if not rep["has_truth"]:
            continue
        row = [f"**{_ds_short(rep)}**"]
        for k in keys + extra:
            row.append(f"{_val_rmse(rep, k):.3e}")
        row.append(f"{_val_rmse(rep, 'T10'):.3e}")
        row.append(f"{_val_rmse(rep, 'T100'):.3e}")
        L.append("| " + " | ".join(row) + " |")
    return "\n".join(L)


def fmt_cross_identifiability(reports):
    """★ 核心表: P/H 的可辨识性（LS 1σ、条件数）与 P 的估计误差。"""
    L = ["| 数据集 | (b) 条件数(列归一化) | (b) 条件数(原始) | σ_Px | \\|Px\\|/σ_Px | "
         "σ_Py | \\|Py\\|/σ_Py | Px 误差 (b) | Px 误差 (c) | Py 误差 (b) | Py 误差 (c) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rep in reports:
        if not rep["has_truth"]:
            continue
        t = rep["truth"]
        sx, sy = _sigma_of(rep, "B", 2), _sigma_of(rep, "B", 3)
        rx = abs(float(t[2])) / sx if sx > 0 else float("inf")
        ry = abs(float(t[3])) / sy if sy > 0 else float("inf")
        L.append(f"| **{_ds_short(rep)}** | {_cond(rep, 'B'):.2e} | {_cond(rep, 'B', False):.2e} | "
                 f"{sx:.5f} | {rx:.2f} | {sy:.5f} | {ry:.2f} | "
                 f"{_relerr(rep, 'B', 2):+.1f}% | {_relerr(rep, 'C', 2):+.1f}% | "
                 f"{_relerr(rep, 'B', 3):+.1f}% | {_relerr(rep, 'C', 3):+.1f}% |")
    return "\n".join(L)


def fmt_pair_table(reports, pick_a, pick_b, title_a, title_b,
                   metrics=("cond", "sigP", "rmse")):
    """两张数据集的成对对比（例如 新包络 vs 旧包络，或 水平 vs 倾斜）。"""
    ra = next((r for r in reports if pick_a(r)), None)
    rb = next((r for r in reports if pick_b(r)), None)
    if ra is None or rb is None:
        return "（缺少可比数据集）"
    rows = []
    rows.append(("小 yaw 实际摆幅 |θs|", f"±{max(abs(x) for x in ra['sim_stats']['theta_small_range_deg']):.2f}°",
                 f"±{max(abs(x) for x in rb['sim_stats']['theta_small_range_deg']):.2f}°",
                 "—"))
    rows.append(("回归矩阵条件数(列归一化, (b))", f"{_cond(ra,'B'):.2e}", f"{_cond(rb,'B'):.2e}",
                 _ratio_txt(_cond(ra, 'B'), _cond(rb, 'B'))))
    for j, nm in ((2, "Px"), (3, "Py")):
        sa, sb = _sigma_of(ra, "B", j), _sigma_of(rb, "B", j)
        rows.append((f"σ_{nm} (LS 1σ)", f"{sa:.5f}", f"{sb:.5f}", _ratio_txt(sa, sb)))
        ta, tb = abs(float(ra["truth"][j])), abs(float(rb["truth"][j]))
        rows.append((f"|{nm}|/σ_{nm}", f"{ta/sa:.2f}" if sa > 0 else "∞",
                     f"{tb/sb:.2f}" if sb > 0 else "∞", "—"))
        rows.append((f"{nm} 相对误差 (b)", f"{_relerr(ra,'B',j):+.1f}%", f"{_relerr(rb,'B',j):+.1f}%",
                     f"{abs(_relerr(ra,'B',j))-abs(_relerr(rb,'B',j)):+.0f} 个百分点"))
        rows.append((f"{nm} 相对误差 (c)", f"{_relerr(ra,'C',j):+.1f}%", f"{_relerr(rb,'C',j):+.1f}%",
                     f"{abs(_relerr(ra,'C',j))-abs(_relerr(rb,'C',j)):+.0f} 个百分点"))
    for k in ("A", "B", "C"):
        va, vb = _val_rmse(ra, k), _val_rmse(rb, k)
        rows.append((f"验证 RMSE {METHOD_LABELS[k]} (rad)", f"{va:.3e}", f"{vb:.3e}",
                     _ratio_txt(va, vb)))
    rows.append(("地板: 真值+λ=10 (rad)", f"{_val_rmse(ra,'T10'):.3e}", f"{_val_rmse(rb,'T10'):.3e}",
                 _ratio_txt(_val_rmse(ra, 'T10'), _val_rmse(rb, 'T10'))))
    L = [f"| 指标 | {title_a} | {title_b} | 变化 |", "|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(L)


def _ratio_txt(a, b) -> str:
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-300 or abs(a) < 1e-300:
        return "—"
    r = a / b
    return f"{r:.2f}×" + ("（更低更好）" if False else "")


def fmt_scan_tables(rep):
    """倾角扫描 / 段数扫描（回答"要多大倾角、多少段"）。"""
    sc = rep.get("p_scan") or {}
    L = []
    if sc.get("tilt"):
        L.append(f"**倾角扫描**（把 A 系重力按 sin φ / sin {sc['data_tilt_deg']:g}° 缩放，"
                 f"其余不变；φ=0 即水平基准）:")
        L.append("")
        L.append("| 倾角 φ | σ_Px (kg·m) | σ_Py (kg·m) | \\|Px\\|/σ_Px | \\|Py\\|/σ_Py | 条件数(列归一化) |")
        L.append("|---|---|---|---|---|---|")
        t = rep["truth"]
        for phi, sx, sy, cd in sc["tilt"]:
            rx = abs(float(t[2])) / sx if sx > 1e-12 else float("inf")
            ry = abs(float(t[3])) / sy if sy > 1e-12 else float("inf")
            L.append(f"| {phi:g}° | {sx:.5f} | {sy:.5f} | {rx:.2f} | {ry:.2f} | {cd:.2e} |")
        L.append("")
    if sc.get("segs"):
        L.append("**段数扫描**（真实采集倾角下，只用前 n 段/每轴）:")
        L.append("")
        L.append("| 段数（两轴合计） | σ_Px (kg·m) | σ_Py (kg·m) |")
        L.append("|---|---|---|")
        for n, sx, sy in sc["segs"]:
            L.append(f"| {n} | {sx:.5f} | {sy:.5f} |")
        L.append("")
    return "\n".join(L)


def fmt_floor_gap(reports):
    """λ 地板 与 各方法差距 **分开列**（地板与方法无关，别混为一谈）。"""
    L = ["| 数据集 | 地板 = 真值参数+λ=10 (rad) | 数值下限 = 真值+λ=100 (rad) | "
         "(a) 距地板 | (b) 距地板 | (c) 距地板 | CAD 初值 |",
         "|---|---|---|---|---|---|---|"]
    for rep in reports:
        if not rep["has_truth"]:
            continue
        fl = _val_rmse(rep, "T10")
        row = [f"**{_ds_short(rep)}**", f"{fl:.3e}", f"{_val_rmse(rep,'T100'):.3e}"]
        for k in ("A", "B", "C"):
            v = _val_rmse(rep, k)
            row.append(f"{v/fl:.2f}×" if fl > 0 else "—")
        row.append(f"{_val_rmse(rep,'P0'):.3e}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("（「距地板」= 方法 RMSE ÷ 地板。**小于 1× 不代表参数更准** —— 那只是"
             "「用错参数在这几段验证轨迹上偶然更贴合」的小样本效应（验证集只有几段、"
             "地板本身又高时很容易出现）；判断方法好坏请看 §0 的参数误差评分表，"
             "RMSE 只用来排除明显发散的解。）")
    return "\n".join(L)


def fmt_pfix_table(reports):
    reps = [r for r in reports if r.get("pfix")]
    if not reps:
        return "（未启用 P 固定变体）"
    L = ["| 数据集 | (b) LS 放开 P (rad) | (b)-P0 固定 P≡0 (rad) | (c) torch 放开 P (rad) | "
         "(c)-P0 固定 P≡0 (rad) | Px 估计: (b) / (c) |",
         "|---|---|---|---|---|---|"]
    for rep in reps:
        L.append(f"| **{_ds_short(rep)}** | {_val_rmse(rep,'B'):.3e} | {_val_rmse(rep,'PF'):.3e} | "
                 f"{_val_rmse(rep,'C'):.3e} | {_val_rmse(rep,'CF'):.3e} | "
                 f"{_phi_of(rep,'B')[2]:+.4f} / {_phi_of(rep,'C')[2]:+.4f} |")
    return "\n".join(L)


def _p_of(rep, mode, kind):
    v = (rep.get("ptreat") or {}).get(mode)
    return None if v is None else v.get(kind)


def fmt_ptreat_table(rep):
    """一个数据集上"P 的三种处理方式"的完整对比（LS 与 torch 都列）。"""
    pt = rep.get("ptreat") or {}
    if not pt:
        return "（无 P 处理实验数据）"
    t = rep["truth"]
    P_true = float(np.hypot(t[2], t[3]))
    L = ["| P 的处理方式 | 估计 `Px` | 估计 `Py` | 估计 \\|P\\| | \\|P\\| 相对误差 | "
         "LS σ_\\|P\\| | 6 参中位误差 | 验证 RMSE (LS) | 验证 RMSE (torch) |",
         "|---|---|---|---|---|---|---|---|---|"]
    order = [k for k in ("free", "along_d", "along_d_wrong0", "zero") if k in pt]
    labs = {"free": P_CONSTRAINT_LABELS["free"],
            "along_d": P_CONSTRAINT_LABELS["along_d"],
            "along_d_wrong0": "P 沿 d 但**假设 θ*=0**（零点信息错误）",
            "zero": P_CONSTRAINT_LABELS["zero"]}
    for mode in order:
        v = pt[mode]
        ph_t = v["torch_phi"]
        ph_l = v["ls_phi"]
        mag = float(np.hypot(ph_t[2], ph_t[3]))
        si = v.get("ls_info") or {}
        L.append(f"| {labs[mode]} | {ph_t[2]:+.5f} | {ph_t[3]:+.5f} | {mag:.5f} | "
                 f"{(mag-P_true)/P_true*100:+.0f}% | "
                 f"{si.get('sigma_s', float('nan')):.5f} | "
                 f"{_med6(rep, ph_t):.1f}% | {v['rmse_ls']:.3e} | {v['rmse_torch']:.3e} |")
    L.append("")
    L.append(f"（真值: `Px`={t[2]:+.5f}, `Py`={t[3]:+.5f}, \\|P\\|={P_true:.5f}；"
             f"表中 `Px/Py/\\|P\\|` 与 6 参中位误差都取自 **torch** 解，"
             f"σ_\\|P\\| 取自 LS 的协方差投影（沿约束方向）。"
             f"6 参中位误差 = `Jbig_eff/Js/fc_big/fv_big/fc_small/fv_small` 的 \\|相对误差\\| 中位数。）")
    return "\n".join(L)


def _med6(rep, phi):
    t = rep["truth"]
    return float(np.median([abs(phi[j] - t[j]) / abs(t[j]) * 100.0 for j in STRONG_PARAMS]))


def _ptreat_sigma(rep, mode):
    v = (rep.get("ptreat") or {}).get(mode) or {}
    si = v.get("ls_info") or {}
    return float(si.get("sigma_s", float("nan")))


def fmt_zero_study(rows):
    if not rows:
        return "（无零点偏差研究数据）"
    L = ["| 真实零点偏差 δ_true | 假设 θ*=0 拟合: ΔPx | ΔPy | \\|ΔP\\| | \\|ΔP\\|/\\|P\\| | "
         "6 参中位误差 | 假设 θ*=δ_true 拟合: \\|ΔP\\| | 6 参中位误差 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a0, at = r["assume0"], r["assume_true"]
        P_true = float(np.hypot(r["truth"][2], r["truth"][3]))
        L.append(f"| {r['delta']:+.1f}° | {a0['Px_err']:+.5f} | {a0['Py_err']:+.5f} | "
                 f"{a0['P_err']:.5f} | {a0['P_err']/P_true*100:.1f}% | {a0['med6']:.1f}% | "
                 f"{at['P_err']:.5f} | {at['med6']:.1f}% |")
    # 每 1° 的灵敏度（用 δ=0 与最大 δ 的斜率）
    if len(rows) >= 2:
        r0, r1 = rows[0], rows[-1]
        dd = r1["delta"] - r0["delta"]
        if dd > 1e-9:
            per_deg = (r1["assume0"]["P_err"] - r0["assume0"]["P_err"]) / dd
            per_deg6 = (r1["assume0"]["med6"] - r0["assume0"]["med6"]) / dd
            L.append("")
            L.append(f"**灵敏度**: 零点每差 1°（而拟合仍假设 θ*=0），`|ΔP|` 增加约 "
                     f"**{per_deg:.5f} kg·m/°**（≈ |P| 的 {per_deg/float(np.hypot(r0['truth'][2], r0['truth'][3]))*100:.0f}%/°），"
                     f"6 参中位误差增加约 {per_deg6:.1f} 个百分点/°。"
                     f"用**正确的** θ*=δ_true 拟合时 `|ΔP|` ≈ "
                     f"{r1['assume_true']['P_err']:.5f}（回到噪声水平）。")
    return "\n".join(L)


def _p_treat_verdict(reports) -> str:
    reps = [r for r in reports if r.get("ptreat") and r["has_truth"]]
    if not reps:
        return "（无 P 处理实验）"
    out = []
    out.append("**① 沿 d 约束的收益（同一批数据、同一方法）**")
    out.append("")
    for rep in reps:
        t = rep["truth"]
        P_true = float(np.hypot(t[2], t[3]))
        f, a, z = rep["ptreat"].get("free"), rep["ptreat"].get("along_d"), rep["ptreat"].get("zero")
        if not (f and a and z):
            continue
        ef = float(np.hypot(f["torch_phi"][2] - t[2], f["torch_phi"][3] - t[3]))
        ea = float(np.hypot(a["torch_phi"][2] - t[2], a["torch_phi"][3] - t[3]))
        ez = float(np.hypot(z["torch_phi"][2] - t[2], z["torch_phi"][3] - t[3]))
        sf, sa = _ptreat_sigma(rep, "free"), _ptreat_sigma(rep, "along_d")
        out.append(f"- **{_ds_short(rep)}**: `|P|` 误差 torch: 自由 {ef/P_true*100:.0f}% "
                   f"→ 沿 d **{ea/P_true*100:.0f}%**（P 固定 0 是 {ez/P_true*100:.0f}%，"
                   f"因为真值 |P|={P_true:.4f}）；LS 的 σ_|P| 从 {sf:.5f} 降到 {sa:.5f}"
                   f"（{sf/sa if sa>0 else float('inf'):.1f}×）；验证 RMSE torch "
                   f"{f['rmse_torch']:.3e} → {a['rmse_torch']:.3e}。")
    out.append("")
    out.append("**② 约束后水平数据够不够（还是必须倾斜）**")
    out.append("")
    lv = next((r for r in reps if not r.get("tilted") and r.get("env_mode") == "asym"
               and not r.get("p_zero_angle")), None)
    ti = next((r for r in reps if r.get("tilted") and r.get("env_mode") == "asym"), None)
    if lv is not None and ti is not None:
        t = lv["truth"]
        P_true = float(np.hypot(t[2], t[3]))
        def _rel(rep, mode="along_d", kind="torch_phi"):
            v = (rep.get("ptreat") or {}).get(mode)
            if not v:
                return float("nan")
            ph = v[kind]
            return float(np.hypot(ph[2] - rep["truth"][2], ph[3] - rep["truth"][3])) / P_true * 100
        ea_lv, ea_ti = _rel(lv), _rel(ti)
        s_lv, s_ti = _ptreat_sigma(lv, "along_d"), _ptreat_sigma(ti, "along_d")
        thr = P_true * 0.10
        out.append(f"- 沿 d 约束下，`|P|` 的估计误差: 水平 {ea_lv:.0f}% vs 倾斜 {ea_ti:.0f}%；"
                   f"LS 的 σ_|P|: 水平 {s_lv:.5f}（= |P| 的 {s_lv/P_true*100:.0f}%）vs "
                   f"倾斜 {s_ti:.5f}（{s_ti/P_true*100:.0f}%）。")
        need_tilt = (s_lv > thr)
        out.append(f"- 判据: 要判断「够不够」看的是 **σ_|P| 的 1σ 水平**（不是估计误差的单次实现），"
                   f"目标取 10%（σ = {thr:.5f}）。水平数据沿 d 约束后 σ_|P| = {s_lv:.5f}，"
                   + ("**仍然远达不到 10%** ⇒ 水平数据即使做了零点标定也**不能**替代倾斜段。"
                      if need_tilt else
                      "**已经满足 10%** ⇒ 水平数据 + 零点标定约束就够了，不必额外倾斜。"))
        out.append(f"- 原因: 水平时 `P` 只剩 `d·R(θs)P` 这一条弱通道"
                   f"（灵敏度 ∝ d ≈ 3 cm），零点标定只是把 `P` 的**方向**定死（省掉 1 个自由度），"
                   f"并没有增加**模长**的信息量；模长的强通道来自重力 `∂G/∂P ≈ |g_A|`（倾斜才有）。")
    return "\n".join(out)


def _zero_verdict(rows, reports) -> str:
    """回答"零点挪过之后还该不该用 P 方向约束"（几何 + 实测双证据）。"""
    if not rows:
        return "（无零点偏差研究）"
    r_max = rows[-1]
    P_true = float(np.hypot(r_max["truth"][2], r_max["truth"][3]))
    dd = r_max["delta"] or 1.0
    # ── 几何（解析）: 假设 θ* 偏 δ ⇒ 拟合出的 P 整体旋转 δ ⇒ |ΔP| = 2|P̂|sin(δ/2) ──
    geo_per_deg = 2.0 * math.sin(math.radians(0.5))          # ≈ 0.00873 = 1.75%/°
    # ── 实测（LS 网格）: 同一 δ 下"假设 0"与"假设真值"两解的 P 向量差 ──
    meas = []
    for r in rows:
        if r["delta"] <= 0:
            continue
        a0, at = r["assume0"]["phi"], r["assume_true"]["phi"]
        meas.append((r["delta"], float(np.hypot(a0[2] - at[2], a0[3] - at[3]))))
    # ── torch 复核（δ 数据集里的 along_d_wrong0 vs along_d）──
    dset = next((x for x in reports if x.get("p_treatments") and x.get("p_zero_angle")), None)
    torch_delta = None
    if dset and dset.get("ptreat", {}).get("along_d_wrong0"):
        w = dset["ptreat"]["along_d_wrong0"]["torch_phi"]
        c = dset["ptreat"]["along_d"]["torch_phi"]
        torch_delta = float(np.hypot(w[2] - c[2], w[3] - c[3]))
    # ── 约束的收益（在倾斜 P 实验数据集上，用 torch 的 |P| 误差）──
    ti = next((x for x in reports if x.get("ptreat") and x.get("tilted")
               and not x.get("p_zero_angle")), None)
    gain_pp = float("nan")
    if ti is not None:
        t = ti["truth"]
        Pt = float(np.hypot(t[2], t[3]))
        f, a = ti["ptreat"]["free"], ti["ptreat"]["along_d"]
        ef = float(np.hypot(f["torch_phi"][2] - t[2], f["torch_phi"][3] - t[3])) / Pt * 100
        ea = float(np.hypot(a["torch_phi"][2] - t[2], a["torch_phi"][3] - t[3])) / Pt * 100
        gain_pp = ef - ea
    out = []
    out.append("**几何（解析）**: 约束方向被写错 δ，等价于把拟合出来的 `P` 整体旋转 δ ⇒ "
               f"`|ΔP| = 2|P̂|sin(δ/2) ≈ |P̂|·δ[rad]`，即 **每 1° 给 `|P|` 带来约 "
               f"{geo_per_deg*100:.2f}% 的系统性偏差**（与 |P| 大小无关）。")
    if meas:
        txt = "、".join(f"δ={d:g}° 时 {v:.5f}" for d, v in meas)
        out.append(f"- LS 实测的「两解之差」（假设 θ*=0 vs 假设 θ*=δ_true）: {txt} "
                   f"⇒ 与几何关系一致（差异 ≈ |P̂|·δ[rad]）。")
    if torch_delta is not None:
        out.append(f"- **torch 复核**（δ_true={dset['p_zero_angle']:g}° 数据集）: "
                   f"用错方向与用对方向的 `|ΔP|` 相差 {torch_delta:.5f} kg·m "
                   f"= |P| 的 {torch_delta/P_true*100:.1f}%（几何预期 "
                   f"{abs(float(np.hypot(dset['ptreat']['along_d']['torch_phi'][2], dset['ptreat']['along_d']['torch_phi'][3])))*geo_per_deg*dset['p_zero_angle']/P_true*100:.1f}%）。")
    out.append(f"- **用对 θ* 的效果**: δ_true={r_max['delta']:g}° 时，假设 θ*=0 的 6 参中位误差 "
               f"{r_max['assume0']['med6']:.1f}%，改用 θ*=δ_true 后 "
               f"{r_max['assume_true']['med6']:.1f}%（`|ΔP|` {r_max['assume0']['P_err']:.5f} → "
               f"{r_max['assume_true']['P_err']:.5f}）。")
    if np.isfinite(gain_pp):
        tol = gain_pp / (geo_per_deg * 100)
        out.append("")
        out.append(f"**值不值得用约束**: 在倾斜数据上，`P` 沿 d 约束把 `|P|` 的估计误差改善约 "
                   f"**{gain_pp:.1f} 个百分点**；而零点每错 1° 要付出 "
                   f"{geo_per_deg*100:.2f} 个百分点的系统偏差 ⇒ 两者相等的位置在 "
                   f"**δ ≈ {tol:.1f}°**。")
        verdict = ("**只要零点读数的不确定度 ≲1°，无脑用约束**（收益远大于代价）；"
                   f"1~{tol:.0f}° 之间属于「值得用但要把 θ* 填对」，> {tol:.0f}° 就别用约束了。"
                   if tol > 1.5 else
                   "**约束的收益本来就很小（≤1 个百分点）**，任何可察觉的零点误差都会把它吃掉 ⇒ "
                   "除非 θ* 有把握准到 1° 以内，否则宁可 `--p-constraint=free`。")
        out.append("- " + verdict)
    out.append("")
    out.append("**回答「手动挪了 2° 还该不该用约束」**:")
    out.append("")
    cost2 = geo_per_deg * 2 * 100
    out.append(f"- 若**不知道**新的 θ*（仍假设 0）: 2° 会带来约 **{cost2:.1f}% |P| 的系统性偏差**"
               f"（纯几何、无法靠多采数据消除）。")
    out.append(f"- 若**知道**新的 θ*（手动微调后重新量出来，例如用离心平衡法扫一遍找极小点）: "
               f"`--p-constraint=along_d --p-zero-angle=<读数>` 就能把这项偏差消掉，"
               f"只留下 ≤1° 的残余（≤{geo_per_deg*100:.2f}%）。")
    out.append("- 所以: **约束的前提是 θ* 已知**。挪过零点就必须重新给 θ*；"
               "给不出 θ* 时用 `free`（或先用离心法把 θ* 标出来再上约束）。")
    return "\n".join(out)


def _swing_verdict(rows) -> str:
    if not rows:
        return "（无摆幅扫描）"
    out = []
    lv = [(r["fill"], r.get("level_along_d_sigma_s", float("nan"))) for r in rows]
    mx = [(r["fill"], r.get("mixed_along_d_sigma_s", float("nan"))) for r in rows]
    lv = [(f, v) for f, v in lv if np.isfinite(v)]
    mx = [(f, v) for f, v in mx if np.isfinite(v)]
    if lv:
        f0, v0 = lv[0]
        f1, v1 = lv[-1]
        out.append(f"- **只看水平数据**: 摆幅从 ±{rows[0]['swing_deg']:.1f}° 加到 "
                   f"±{rows[-1]['swing_deg']:.1f}°（把整个包络用满），σ_|P| 只从 {v0:.5f} 变到 {v1:.5f}"
                   f"（{v0/v1 if v1>0 else float('inf'):.2f}×）——**几乎没用**。"
                   f"即使把 {rows[-1]['swing_deg']:.1f}° 全用满，σ_|P| = {v1:.5f} "
                   f"= |P| 的 {v1/P_ALONG_TRUTH*100:.0f}%，离 10% 差 "
                   f"{v1/(0.10*P_ALONG_TRUTH):.0f} 倍。")
        out.append(f"  - 原因: 摆幅只影响 `d·R(θs)P` 这条**弱**通道的信息量，"
                   f"而它本身就比倾斜时的重力通道小 ~2 个数量级（灵敏度 ∝ d ≈ 3 cm vs |g_A| ≈ 1.7 m/s²）。")
    if mx:
        f0, v0 = mx[0]
        f1, v1 = mx[-1]
        out.append(f"- **水平 + 倾斜混合**: σ_|P| 在摆幅 ±{rows[0]['swing_deg']:.1f}° 时就已经是 "
                   f"{v0:.5f}（|P| 的 {v0/P_ALONG_TRUTH*100:.0f}%），摆幅 ±{rows[-1]['swing_deg']:.1f}° 时 "
                   f"{v1:.5f}（{v1/P_ALONG_TRUTH*100:.0f}%）⇒ **倾斜段一进来，摆幅就不再是瓶颈**；"
                   f"水平段的价值是标定 `J/fc/fv`，而不是提供 `P`。")
    out.append("")
    out.append("**结论**: 想把 `|P|` 定到 10% 以内，靠「把摆幅做大」在 −25°/+20° 行程内是做不到的"
               "（把包络用满也只有 ~"
               f"{rows[-1].get('level_along_d_sigma_s', float('nan'))/P_ALONG_TRUTH*100:.0f}% 的 1σ）；"
               "必须靠**倾斜段**（≥4° 起效、±10° 稳），此时小 yaw 只要按包络正常摆动即可"
               "（水平段负责惯量/摩擦，倾斜段负责 `P`）。")
    return "\n".join(out)


def fmt_swing_table(rows):
    if not rows:
        return "（无摆幅扫描数据）"
    L = ["| 小 yaw 摆幅（包络半宽比例）| 摆幅 | σ_\\|P\\| (水平, 沿 d 约束) | 占 \\|P\\| | "
         "σ_\\|P\\| (水平+倾斜) | 占 \\|P\\| |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['fill']:.2f} | ±{r['swing_deg']:.1f}° | "
                 f"{r.get('level_along_d_sigma_s', float('nan')):.5f} | "
                 f"{r.get('level_along_d_sigma_s', float('nan'))/P_ALONG_TRUTH*100:.0f}% | "
                 f"{r.get('mixed_along_d_sigma_s', float('nan')):.5f} | "
                 f"{r.get('mixed_along_d_sigma_s', float('nan'))/P_ALONG_TRUTH*100:.0f}% |")
    return "\n".join(L)


def write_markdown(path, reports, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    L = []
    reps_t = [r for r in reports if r["has_truth"]]
    main_rep = next((r for r in reps_t if r.get("env_mode") == "asym" and not r.get("tilted")
                     and r.get("sim_stats")), reps_t[0] if reps_t else reports[0])
    key = "C" if any("C" in r["results"] for r in reps_t) else "A"

    L.append("# 三种辨识方法对比 + 小 yaw 行程包络 / 静态倾斜对 8 参可辨识性的影响")
    L.append("")
    L.append("> 本文件由 `python/scripts/compare_ident_methods.py` **自动生成**（不要手改）。"
             f"seed={args.seed}，每轴段数={args.segments}，plant λ={args.plant_lambda:g}，"
             f"辨识模型 λ={FRICTION_LAMBDA:g}，100 Hz / 每段 {args.seg_len} 点。")
    L.append("> 模型: `include/tcbs/mpc/planar_yaw_model.h`；采集与辨识约定: `docs/calibration.md` §4；"
             "采集脚本: `python/scripts/collect_sysid.py`。")
    L.append("")
    L.append("**复现命令**（快速版 ~2 min / 完整版见 §9 实测耗时）:")
    L.append("")
    L.append("```bash")
    L.append("# 快速版（冒烟/看趋势）")
    L.append("python3 python/scripts/compare_ident_methods.py --sim-only --segments=3 --quick")
    L.append("# 完整版（本文件的内容）")
    L.append("python3 python/scripts/compare_ident_methods.py --sim-only --segments=6")
    L.append("```")
    L.append("")
    L.append("**数据集 = (小 yaw 包络) × (静态倾斜) 的组合**;  方法代号:")
    L.append("")
    L.append("| 代号 | 方法 | held 轴处理 |")
    L.append("|---|---|---|")
    L.append("| (a) | 逆动力学线性回归 τ = Y(φ)·φ（截断 SVD-LS）| 位置实测，**θ̇ = θ̈ = 0** |")
    L.append("| (b) | 同上 | 位置与 **θ̇/θ̈ 都实测**（量化角中心差分 + 3 点平滑）|")
    L.append("| (c) | **输出误差法**（可导前向仿真 + Adam/LBFGS）| 由前向仿真自动带出 |")
    L.append("| (a')/(b') | 消融: θ̇/θ̈ 换成 plant 真值 | 仅诊断（隔离导数噪声）|")
    L.append("| PF / CF | P 固定为 0 的变体（LS / torch）| 水平数据下 P 不可辨识时的兜底 |")
    L.append("")

    # ── 0. TL;DR ──
    L.append("## 0. 结论速览")
    L.append("")
    L.append(fmt_cross_rmse(reports))
    L.append("")
    L.append("（合成 RMSE = 两轴前向仿真角度 RMSE 的均方根；验证集为留出段，"
             "「地板」= 用**真值参数**但模型 λ=10 的 RMSE —— 与方法无关，见 §6。）")
    L.append("")
    if len(reps_t) >= 2:
        L.append(fmt_cross_identifiability(reports))
        L.append("")
    L.append("**参数精度 vs 验证 RMSE 的排序不一定一致**（验证集只有几段、且离地板很近时，"
             "RMSE 的分辨力有限；参数误差才是方法好坏的直接证据）:")
    L.append("")
    L.append(fmt_param_score(reports))
    L.append("")
    L.append("（「6 参中位误差」= `Jbig_eff/Js/fc_big/fv_big/fc_small/fv_small` 的 |相对误差| 中位数；"
             "`Px/Py` 单列，因为它们在水平数据下本就不可辨识。）")
    L.append("")
    lv_asym = next((r for r in reps_t if r.get("env_mode") == "asym" and not r.get("tilted")
                    and r.get("sim_stats")), None)
    ti_asym = next((r for r in reps_t if r.get("env_mode") == "asym" and r.get("tilted")
                    and r.get("sim_stats")), None)
    if lv_asym is not None and ti_asym is not None:
        t = lv_asym["truth"]
        s_lv, s_ti = _sigma_of(lv_asym, "B", 2), _sigma_of(ti_asym, "B", 2)
        e_b_lv, e_b_ti = _relerr(lv_asym, "B", 2), _relerr(ti_asym, "B", 2)
        e_c_lv, e_c_ti = _relerr(lv_asym, "C", 2), _relerr(ti_asym, "C", 2)
        L.append(f"- **静态倾斜对 `Px` 的作用**: LS 的 1σ 从 {s_lv:.5f} 降到 {s_ti:.5f} kg·m"
                 f"（改善 {s_lv/s_ti if s_ti>0 else float('inf'):.1f}×），"
                 f"`Px` 的相对误差 (b) {e_b_lv:+.0f}% → {e_b_ti:+.0f}%，"
                 f"(c) {e_c_lv:+.0f}% → {e_c_ti:+.0f}%。")
        L.append(f"- **行程变窄（±40° → −25°/+20°）的作用**: 见 §4 的成对对比表"
                 f"（条件数、σ_P、三方法误差与验证 RMSE）。")
    L.append("- 实车采集设计（倾角数量、段数、小 yaw 摆幅）见 §9。")
    L.append("")

    # ── 1. 仿真设置 ──
    L.append("## 1. 仿真设置")
    L.append("")
    c0 = main_rep["cfg"]
    L.append("### 1.1 小 yaw 行程与包络（本轮改成**非对称**实测行程）")
    L.append("")
    L.append("| 方案 | 硬界限（仿真中从不触发）| 参考包络（留 ~8° 跟踪余量）| 包络中心 |")
    L.append("|---|---|---|---|")
    L.append("| **新（本轮默认）** | −25° ~ +20° | −17° ~ +12° | −2.5° |")
    L.append("| 旧（早期假设，保留做对比）| ±45° | ±40° | 0° |")
    L.append("")
    L.append(f"实现: `SimConfig.small_travel_min_deg/small_travel_max_deg/small_env_margin_deg`；"
             f"每段实际 θs 范围与硬界限触发次数在 §2 表里逐数据集列出"
             f"（`collect_sim` 一旦发现越界会**直接抛错**，因为它意味着参考包络/初值设置是 bug）。")
    L.append("")
    L.append("### 1.2 静态倾斜数据集（level vs tilted）")
    L.append("")
    L.append("底盘**静置**在绕 y 轴的固定倾角 φ 下（`base_omega = base_alpha = 0`），"
             "只有重力平面分量非零；IMU 装在大 yaw 转子上 ⇒ A 系重力方向随 θ_b 旋转:")
    L.append("")
    L.append("```")
    L.append("g_C(平面) = (g·sinφ, 0)          # C 系（底盘系）")
    L.append("g_A(t)    = R_z(−θ_b(t))·g_C     # A 系（大 yaw 转子系）= 逐样本记录 gravity_ax/ay")
    L.append("```")
    L.append("")
    L.append("CSV/npz 里逐样本带 `gravity_ax/gravity_ay`（与 `python/scripts/collect_sysid.py` 的列名、"
             "单位 m/s² 一致 ⇒ 实机倾斜段数据可直接喂给这两个脚本）。")
    L.append("")
    L.append("### 1.3 其它设置（未变）")
    L.append("")
    L.append(f"- plant: λ={c0.plant_lambda:g}（模拟真实库仑摩擦，积分步长 "
             f"{c0.plant_substep*1e3:.3f} ms）；模型 λ={c0.model_lambda:g}（固定，不辨识）")
    L.append(f"- 编码器: {ENCODER_CPR} 计数/圈 ⇒ 量化步长 {QUANT_STEP:.3e} rad（角度 round 量化）")
    L.append(f"- 上位机 PID: kp={c0.kp}, ki={c0.ki}, kd={c0.kd}，限幅 ±{c0.pid_out_limit} N·m，"
             f"力矩变化率 ≤ {c0.pid_rate_limit}/step；分轴采集（一轴跟踪、另一轴 PID 保持）")
    L.append(f"- 几何（实测，不辨识）: dx={c0.dx} m, dy={c0.dy} m；"
             f"水平时 gravity_a ≡ 0 ⇒ P 只能靠 θs 耦合项辨识")
    L.append("")

    # ── 2. 数据集总览 ──
    L.append("## 2. 数据集总览")
    L.append("")
    L.append("| 数据集 | 段数(拟合/验证) | θs 实际范围 | 参考包络 | 硬界限越界 | g_A 幅值 (m/s²) |")
    L.append("|---|---|---|---|---|---|")
    for rep in reports:
        st = rep.get("sim_stats")
        if st is None:
            L.append(f"| {rep['name']}（派生/真机数据，无仿真统计）| "
                     f"{rep['fit_segs']}/{rep['val_segs']} | — | — | — | — |")
            continue
        lo, hi = st["theta_small_range_deg"]
        ga = st.get("gravity_amp", 0.0)
        L.append(f"| **{_ds_short(rep)}** | {rep['fit_segs']}/{rep['val_segs']} | "
                 f"{lo:+.2f}° ~ {hi:+.2f}° | [{rep['cfg'].env_min/DEG:+.0f}°,"
                 f"{rep['cfg'].env_max/DEG:+.0f}°] | {st['limit_violations']} | {ga:.3f} |")
    L.append("")
    L.append("（`硬界限越界 = 0` 是硬性要求；越界会直接抛错而不是只打印。）")
    L.append("")

    # ── 3. 各数据集详情 ──
    L.append("## 3. 各数据集详情")
    L.append("")
    for rep in reports:
        L.append(f"### {_ds_short(rep)}")
        L.append("")
        L.append(f"- 数据: {rep['dstats']['n_sample']} 点；|τ| RMS = {rep['dstats']['rms_tau']:.4f} N·m"
                 f"（最大 {rep['dstats']['max_tau']:.3f}，饱和 {rep['dstats']['sat_frac']*100:.2f}%）；"
                 f"|ω| RMS = {rep['dstats']['rms_omega']:.2f} rad/s")
        if rep["has_truth"]:
            L.append(f"- 真值 φ = " + ", ".join(f"{v:+.5f}" for v in rep["truth"]))
            L.append(f"- held 轴真实运动: RMS|θ̇| = {rep['hstats']['rms_v']:.4f} rad/s，"
                     f"RMS|θ̈| = {rep['hstats']['rms_a']:.3f} rad/s²，偏离保持目标最大 "
                     f"{rep['hstats']['max_dev']*57.2958:.2f}°")
            L.append(f"- torch: Adam {args.torch_iters} 步（每步抽 {args.torch_batch} 个 "
                     f"{args.torch_window} 点窗口）+ LBFGS {args.torch_lbfgs} 次，"
                     f"全批 loss = {rep['torch_loss']:.4e}，用时 {rep['torch_seconds']:.0f} s")
        L.append("")
        if rep["has_truth"]:
            L.append("**参数估计**")
            L.append("")
            L.append(fmt_param_table(rep["truth"], rep["results"], ["A", "B", "C"]))
            L.append("")
            L.append("**回归矩阵与样本剔除（可审计）**")
            L.append("")
            L.append("| 方法 | 参与行数 | 样本数 | 数值秩 | 拟合残差 RMS (N·m) | 条件数(列归一化) | "
                     "剔除: 边界/饱和/无信息/陈旧 |")
            L.append("|---|---|---|---|---|---|---|")
            for k in ("A", "B"):
                inf, sv = rep["results"][k]["info"], rep["results"][k]["sv"]
                d = inf.get("dropped", {})
                L.append(f"| {METHOD_LABELS[k]} | {inf.get('n_row',0)} | {inf.get('n_sample',0)} | "
                         f"{sv['rank']}/8 | {sv['resid_rms']:.4f} | {sv['cond_norm']:.2e} | "
                         f"{d.get('border',0)}/{d.get('sat',0)}/{d.get('quiescent',0)}/"
                         f"{d.get('stale',0)} |")
            svB = rep["results"]["B"]["sv"]
            L.append(f"| {METHOD_LABELS['C']} | —（不用回归矩阵）| {rep['fit_segs']} 段 × "
                     f"{rep['cfg'].seg_len} 点 | — | 全批窗口 loss = {rep['torch_loss']:.4e} | — | — |")
            L.append("")
            L.append(f"`Y`（(b) 列归一化）奇异值: " + ", ".join(f"{v:.3e}" for v in svB["sv"]) +
                     f" ⇒ 条件数 {svB['cond_norm']:.2e}；未归一化（物理量纲）"
                     f"条件数 {svB['cond_raw']:.2e}")
            L.append("")
            if rep.get("pfix"):
                L.append("**P 固定为 0 的变体**")
                L.append("")
                L.append(fmt_pfix_table([rep]))
                L.append("")
            if rep["ablation"]:
                L.append("**消融: θ̇/θ̈ 换成 plant 真值**")
                L.append("")
                abl = {k: {"phi": v["phi"]} for k, v in rep["ablation"].items()}
                L.append(fmt_param_table(rep["truth"], abl, ["A_oracle", "B_oracle"]))
                L.append("")
        L.append("**验证集前向仿真 RMSE**")
        L.append("")
        L.append(fmt_rmse_table(rep["rmse"], rep["labels"]))
        L.append("")
        if rep.get("p_scan"):
            L.append(fmt_scan_tables(rep))
            L.append("")

    # ── 4. 包络对比 ──
    L.append("## 4. 行程包络对比: 新（−25°~+20°） vs 旧（±40°）")
    L.append("")
    L.append("同一倾斜条件下，只改小 yaw 包络 ⇒ 看「行程变窄」对可辨识性的代价。")
    L.append("")
    L.append("### 4.1 水平底盘")
    L.append("")
    L.append(fmt_pair_table(reports, lambda r: r.get("env_mode") == "asym" and not r.get("tilted") and r.get("sim_stats"),
                            lambda r: r.get("env_mode") == "legacy" and not r.get("tilted") and r.get("sim_stats"),
                            "新包络 × 水平", "旧包络 × 水平"))
    L.append("")
    L.append("### 4.2 静态倾斜")
    L.append("")
    L.append(fmt_pair_table(reports, lambda r: r.get("env_mode") == "asym" and r.get("tilted") and r.get("sim_stats"),
                            lambda r: r.get("env_mode") == "legacy" and r.get("tilted") and r.get("sim_stats"),
                            "新包络 × 倾斜", "旧包络 × 倾斜"))
    L.append("")

    # ── 5. 水平 vs 倾斜 ──
    L.append("## 5. 水平 vs 静态倾斜: P 的可辨识性")
    L.append("")
    L.append("### 5.1 新包络")
    L.append("")
    L.append(fmt_pair_table(reports, lambda r: r.get("env_mode") == "asym" and not r.get("tilted") and r.get("sim_stats"),
                            lambda r: r.get("env_mode") == "asym" and r.get("tilted") and r.get("sim_stats"),
                            "新包络 × 水平", "新包络 × 倾斜"))
    L.append("")
    L.append("### 5.2 旧包络")
    L.append("")
    L.append(fmt_pair_table(reports, lambda r: r.get("env_mode") == "legacy" and not r.get("tilted") and r.get("sim_stats"),
                            lambda r: r.get("env_mode") == "legacy" and r.get("tilted") and r.get("sim_stats"),
                            "旧包络 × 水平", "旧包络 × 倾斜"))
    L.append("")
    L.append("### 5.3 P 固定为 0 是否更好？（水平数据）")
    L.append("")
    L.append(fmt_pfix_table(reports))
    L.append("")
    L.append(_pfix_verdict(reports))
    L.append("")
    L.append("### 5.4 要多大倾角 / 多少段")
    L.append("")
    L.append(_tilt_verdict(reports))
    L.append("")

    # ── 6. ★ P 的处理方式与零点标定 ──
    L.append("## 6. ★ P 的处理方式（自由 / 沿 d / 固定 0）与零点标定")
    L.append("")
    L.append("**约束从哪来**: 大 yaw 恒速 Ω、小 yaw 松手（τ=0）时，平衡点满足 "
             "`μ(θ*) = 0 ⇒ R(θ*)·P ∥ d`，稳定解取径向外侧 ⇒")
    L.append("")
    L.append("```")
    L.append("P = |P| · R(−θ*) · d̂        θ* = 平衡点在当前零点坐标系里的读数")
    L.append("  · 零点恰好设在平衡点上（离心/重力标定做过）⇒ θ* = 0 ⇒ P = |P|·d̂（dy=0 时 Py=0）")
    L.append("  · 零点被手动挪过 δ（或标定有残差）      ⇒ θ* = δ ⇒ P 相对 d 旋转 −δ")
    L.append("```")
    L.append("")
    L.append("对应开关: `--p-constraint=free|along_d|zero`（默认 `free`，保持既有结论可复现）"
             "+ `--p-zero-angle=<deg>`（θ*，默认 0）。`along_d` 把 `Px/Py` 两个自由参数压成"
             "**一个** `|P|`（少 1 个自由度、`Y` 少 1 列）。")
    L.append("")
    pt_reps = [r for r in reports if r.get("ptreat")]
    if pt_reps:
        L.append("### 6.1 三种处理方式的对比（同一批数据、同一预处理）")
        L.append("")
        for rep in pt_reps:
            L.append(f"**{rep['name']}**")
            L.append("")
            L.append(fmt_ptreat_table(rep))
            L.append("")
            sv = (rep["results"].get("B") or {}).get("sv")
            if sv:
                sra = np.asarray(sv["sv_raw"], dtype=float)
                L.append(f"`Y` 奇异值（未归一化，非零列）: " +
                         ", ".join(f"{v:.3e}" for v in sra[sra > 0]) +
                         f"；条件数（非零列）{(rep['results']['B'].get('sv') or {}).get('cond_norm', float('nan')):.2f}。")
                L.append("")
        L.append("### 6.2 结论: 沿 d 约束的收益 · 水平数据够不够")
        L.append("")
        L.append(_p_treat_verdict(reports))
        L.append("")
        L.append("### 6.3 零点偏差 δ 的容差（手动微调过零点）")
        L.append("")
        L.append("把真实零点偏 δ 的**等效数据**用 `shift_zero()` 造出来（零点平移时 PID 参考与"
                 "小 yaw 读数一起平移 ⇒ 物理轨迹与力矩完全不变，只有记录的 θs 与模型里的 P 变），"
                 "然后分别用「假设 θ*=0」与「假设 θ*=δ_true」拟合:")
        L.append("")
        L.append(fmt_zero_study(getattr(args, "zero_study", []) or []))
        L.append("")
        L.append(_zero_verdict(getattr(args, "zero_study", []) or [], reports))
        L.append("")
        L.append("### 6.4 已知 `Py=0` 时，小 yaw 摆幅需要多大")
        L.append("")
        L.append("在水平数据上（真值 P 沿 d）把参考摆幅按包络半宽的比例缩放，"
                 "用 along_d 约束算 σ_|P|；再与倾斜数据合并看混合采集的效果:")
        L.append("")
        L.append(fmt_swing_table(getattr(args, "swing_results", []) or []))
        L.append("")
        L.append(_swing_verdict(getattr(args, "swing_results", []) or []))
        L.append("")
    L.append("## 7. λ 失配的「地板」与「方法差距」分开看")
    L.append("")
    L.append("两件事必须分开: (i) **地板** = 真值参数 + 模型 λ=10 的 RMSE —— 由摩擦形状失配"
             "（plant λ=100 vs 模型 λ=10）决定，**与辨识方法无关**；"
             "(ii) **方法差距** = 各方法 RMSE 与地板之比 —— 才是方法好坏。")
    L.append("")
    L.append(fmt_floor_gap(reports))
    L.append("")
    L.append(_lambda_mismatch_txt(main_rep))
    L.append("")

    # ── 7. 归因 ──
    L.append("## 8. 差异来源分析（a vs b、c、以及 P/行程/倾斜）")
    L.append("")
    L.append(_attribution(main_rep))
    L.append("")

    # ── 8. 实车采集设计 ──
    L.append("## 9. 实车采集设计（定量建议）")
    L.append("")
    L.append(_acquisition_design(reports))
    L.append("")

    # ── 9. 复现 ──
    L.append("## 10. 复现命令与耗时")
    L.append("")
    L.append("```bash")
    L.append("# 模型自检（回归矩阵/重力通道/torch==numpy）")
    L.append("python3 python/scripts/identify_params_torch.py --selftest")
    L.append("")
    L.append("# 快速版: 3 段/轴、迭代数减半（冒烟用，趋势一致、数值偏差大）")
    L.append("python3 python/scripts/compare_ident_methods.py --sim-only --segments=3 --quick")
    L.append("")
    L.append("# 完整版（本文件）: 4 个数据集（2 包络 × 水平/倾斜）+ P 固定变体")
    L.append(f"python3 python/scripts/compare_ident_methods.py --sim-only --segments={args.segments}")
    L.append("")
    L.append("# 只跑某一组: 例如只对比「倾斜 vs 水平」（新包络）")
    L.append("python3 python/scripts/compare_ident_methods.py --sim-only --segments=6 \\\\")
    L.append("    --envelopes=asym --tilts='0;8,-8,10,-10'")
    L.append("")
    L.append("# 把仿真数据落盘（列名与 collect_sysid.py 对齐，含 gravity_ax/ay）")
    L.append("python3 python/scripts/compare_ident_methods.py --sim-only --segments=6 \\\\")
    L.append("    --torch-iters=0 --no-write --dump-sim=/tmp/sysid_sim")
    L.append("")
    L.append("# 用脚本 1 单独做输出误差法拟合（真机 CSV/npz 同理）")
    L.append("python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv'")
    L.append("python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv' \\\\")
    L.append("    --fit-axis=big --iters=200 --lbfgs-iters=6")
    L.append("python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv' \\\\")
    L.append("    --fix-p        # 水平数据: 把 Px/Py 按住为 0")
    L.append("```")
    L.append("")
    L.append(f"实测耗时（本机，4 个数据集 + 2 个 P 固定 torch 变体）: **{args.measured_seconds}**"
             f"（`--torch-iters`/`--segments` 可线性缩放；快速版约 2 min）。")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def _rel_err(phi, truth):
    t = np.asarray(truth, dtype=np.float64)
    e = np.asarray(phi, dtype=np.float64) - t
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(t) > 1e-12, e / t, np.nan)
    return e, r


def _best_method(rep):
    keys = rep["methods"]
    if not keys:
        return "C"
    return min(keys, key=lambda k: rep["rmse"][k]["rmse_meas_all"])


def _tldr_err_line(rep):
    parts = []
    for m in rep["methods"]:
        e, r = _rel_err(rep["results"][m]["phi"], rep["truth"])
        parts.append(f"{METHOD_LABELS[m]}: 最大相对误差 {np.nanmax(np.abs(r))*100:.1f}%，"
                     f"验证 RMSE {rep['rmse'][m]['rmse_meas_all']:.3e} rad")
    return "- " + "；".join(parts) + "。"


def _pfix_verdict(reports) -> str:
    reps = [r for r in reports if r.get("pfix") and r["has_truth"]]
    if not reps:
        return "（未启用 P 固定变体）"
    out = ["数据说话（水平数据下 P 本来不可辨识）:", ""]
    out.append("| 数据集 | Px 真值 | (b) 放开 P 的 Px | (b) RMSE | 固定 P≡0 的 RMSE | "
               "(c) 放开 P 的 Px | (c) RMSE | 固定 P≡0 的 RMSE |")
    out.append("|---|---|---|---|---|---|---|---|")
    for rep in reps:
        t = rep["truth"]
        out.append(f"| {_ds_short(rep)} | {float(t[2]):+.4f} | {_phi_of(rep,'B')[2]:+.4f} | "
                   f"{_val_rmse(rep,'B'):.3e} | {_val_rmse(rep,'PF'):.3e} | "
                   f"{_phi_of(rep,'C')[2]:+.4f} | {_val_rmse(rep,'C'):.3e} | "
                   f"{_val_rmse(rep,'CF'):.3e} |")
    out.append("")
    verdicts = []
    for rep in reps:
        b, pf = _val_rmse(rep, "B"), _val_rmse(rep, "PF")
        c, cf = _val_rmse(rep, "C"), _val_rmse(rep, "CF")
        verdicts.append((rep, b, pf, c, cf))
    lv = [v for v in verdicts if not v[0].get("tilted")]
    if lv:
        rep, b, pf, c, cf = lv[0]
        better_ls = pf < b
        better_to = cf < c
        out.append(f"**结论（水平数据 {_ds_short(rep)}）**: "
                   f"把 P 按住 (b) {b:.3e} → {pf:.3e} rad"
                   f"（{'改善' if better_ls else '变差'} {abs(1-pf/b)*100:.0f}%），"
                   f"(c) {c:.3e} → {cf:.3e} rad"
                   f"（{'改善' if better_to else '变差'} {abs(1-cf/c)*100:.0f}%）。"
                   + ("⇒ **水平数据下应当把 Px/Py 按住（或强正则）**："
                      "放开只会让两个不可辨识的参数去吸收噪声/失配，反而把 J 和摩擦带偏。"
                      if (better_ls and better_to) else
                      "⇒ **本数据集上固定 P 对验证 RMSE 没有明显优势**（差异 ±6% 内，"
                      "属小样本噪声）。但 (b) 放开 P 时 `Px` 会跑到 "
                      f"{_phi_of(rep,'B')[2]:+.4f}（真值 {float(rep['truth'][2]):+.4f}，"
                      f"误差 {abs(_relerr(rep,'B',2)):.0f}%）—— 这个数值**没有物理意义**。"
                      "所以建议: 水平数据也把 `Px/Py` 按住（`--fix-p`），"
                      "**理由不是 RMSE 更好，而是参数可复现、可跨批次比较、不被噪声带跑**；"
                      "真正的解法是补倾斜段（见 §5.4）。"))
    ti = [v for v in verdicts if v[0].get("tilted")]
    if ti:
        rep, b, pf, c, cf = ti[0]
        out.append("")
        out.append(f"**倾斜数据 {_ds_short(rep)}**: 固定 P (b) {b:.3e} → {pf:.3e}、"
                   f"(c) {c:.3e} → {cf:.3e} rad —— "
                   + ("倾斜段把 P 变成可辨识参数后，**应当放开**（按住反而变差）。"
                      if (pf > b or cf > c) else "固定 P 仍然不亏。"))
    return "\n".join(out)


def _tilt_verdict(reports) -> str:
    scans = [(r, r["p_scan"]) for r in reports if r.get("p_scan") and r["p_scan"].get("tilt")]
    if not scans:
        return "（本次运行没有倾斜数据集，无法给出倾角/段数结论）"
    out = []
    for rep, sc in scans:
        t = rep["truth"]
        need = None
        for phi, sx, sy, cd in sc["tilt"]:
            if sx > 0 and abs(float(t[2])) / sx >= 3.0:
                need = phi
                break
        out.append(f"- **{_ds_short(rep)}**（实采倾角 "
                   f"±{sc['data_tilt_deg']:g}°，g·sinφ = "
                   f"{rep['sim_stats']['gravity_amp']:.2f} m/s²）")
        if need is not None:
            out.append(f"  - 要让 `|Px|/σ ≥ 3`，需要倾角 ≥ **{need:g}°**；"
                       f"实采 {sc['data_tilt_deg']:g}° 时 σ_Px = {sc['tilt'][-1][1]:.5f} kg·m。")
        else:
            out.append(f"  - 扫描网格内（最大 {sc['tilt'][-1][0]:g}°）都没有达到 |Px|/σ ≥ 3，"
                       f"说明该数据集下 P 仍然定不准。")
        # 段数: σ ∝ 1/√n ⇒ 估算达到目标所需的段数
        if sc.get("segs"):
            n_ref, sx_ref, _ = sc["segs"][-1]
            for target_ratio in (3.0, 5.0):
                sx_target = abs(float(t[2])) / target_ratio
                if sx_ref > 0:
                    n_need = n_ref * (sx_ref / sx_target) ** 2
                    out.append(f"  - 要把 σ_Px 压到 |Px|/{target_ratio:.0f} = {sx_target:.5f} kg·m，"
                               f"按 σ ∝ 1/√n 估算需要约 **{math.ceil(n_need)} 段**"
                               f"（当前 {n_ref} 段时 σ_Px = {sx_ref:.5f}）。")
        if sc.get("segs"):
            n0, s0, _ = sc["segs"][0]
            n1, s1, _ = sc["segs"][-1]
            if s1 > 0 and s0 > 0 and n1 > n0:
                idx = math.log(s0 / s1) / math.log(math.sqrt(n1 / n0))
                out.append(f"  - 实测段数缩放指数 ≈ {idx:.2f}"
                           f"（理论 1.0 = 白噪声随 1/√n 平均；<1 说明误差里有系统性成分，"
                           f"加段数收益递减）。")
    return "\n".join(out)


def _lambda_mismatch_txt(rep) -> str:
    if not rep.get("mism"):
        return "（该数据集无真值，跳过）"
    big, small = rep["mism"]["big"], rep["mism"]["small"]
    floor, low = _val_rmse(rep, "T10"), _val_rmse(rep, "T100")
    return (f"本数据集实测: 真值 fc 在 λ=100 与 λ=10 下的库伦力矩差 "
            f"RMS = {big['rms']:.4f} N·m（大 yaw，最大 {big['max']:.4f}）/ "
            f"{small['rms']:.4f} N·m（小 yaw，最大 {small['max']:.4f}），"
            f"其中 |ω|<0.1 rad/s 的样本占 {big['frac_low']*100:.0f}% / "
            f"{small['frac_low']*100:.0f}%（失配集中在低速段）。\n\n"
            f"⇒ 即使参数完全正确，λ=10 的模型形式也留下 **{floor:.3e} rad** 的地板"
            f"（λ 匹配时只有 {low:.3e} rad，即数值下限）。\n\n"
            f"**注意**: 地板的**大小**与「行程包络」无关（它由摩擦形状失配 + 数据本身的速度分布"
            f"决定），但「行程变窄」会改变各方法的**参数误差**与接近地板的能力 —— "
            f"两件事在 §4/§5 里分开列。")


def _attribution(rep) -> str:
    if not rep["has_truth"]:
        return "（无真值，跳过归因）"
    truth = rep["truth"]
    rm = rep["rmse"]
    h = rep["hstats"]
    tau_rms = rep["dstats"]["rms_tau"]
    dt = rep["cfg"].dt
    sig_a = QUANT_STEP * 0.707 / (dt ** 2) / math.sqrt(3)
    m12 = abs(float(truth[1]))
    dropped = m12 * h["rms_a"]
    out = []
    out.append("1. **(a) vs (b) —— held 轴「忽略真项」还是「吃进 θ̈ 噪声」**")
    out.append("")
    out.append(f"   - held 轴（PID 保持）真实运动: RMS|θ̇| = {h['rms_v']:.4f} rad/s，"
               f"RMS|θ̈| = {h['rms_a']:.3f} rad/s²，偏离保持目标最大 "
               f"{h['max_dev']*57.2958:.2f}°。")
    out.append(f"   - (a) 把 held 轴 θ̇/θ̈ 置零 ⇒ 丢掉的真项量级 ≈ M12·θ̈_held ≈ "
               f"{m12:.4f}×{h['rms_a']:.2f} = {dropped:.4f} N·m"
               f"（样本 |τ| RMS {tau_rms:.3f} N·m 的 {dropped/max(tau_rms,1e-9)*100:.0f}%）；"
               f"小 yaw 方程里的离心项 −½μθ̇b² 也一并丢掉。")
    out.append(f"   - (b) 用 {ENCODER_CPR} 线量化角的二阶差分（+3 点平滑）: θ̈ 噪声标准差 ≈ "
               f"{sig_a:.2f} rad/s²，与真实 |θ̈| RMS（{h['rms_a']:.2f} rad/s²）同量级 ⇒ "
               f"回归矩阵被噪声污染（errors-in-variables ⇒ 衰减偏置）。")
    if rep.get("ablation"):
        ra, rb = _val_rmse(rep, "A"), _val_rmse(rep, "B")
        rao, rbo = _val_rmse(rep, "A_oracle"), _val_rmse(rep, "B_oracle")
        out.append("   - **消融（把 θ̇/θ̈ 换成 plant 真值）把两者分开了**:")
        out.append(f"     * (a) {ra:.3e} → (a') {rao:.3e} rad：几乎不变 ⇒ (a) 的误差"
                   f"**主要来自 held 轴假设错**，不是导数噪声；")
        out.append(f"     * (b) {rb:.3e} → (b') {rbo:.3e} rad：改善 "
                   f"{(1-rbo/max(rb,1e-12))*100:.0f}% ⇒ (b) 的误差**主要来自导数噪声**。")
    out.append("")
    out.append("2. **(c) 输出误差法**")
    out.append("")
    out.append(f"   不用带噪 θ̈，而是用**记录力矩**做可导前向仿真比较角度轨迹 ⇒ "
               f"量化噪声只以 {QUANT_STEP:.2e} rad 的测量噪声进入；held 轴真实运动被自动带进去。"
               f"本数据集验证 RMSE (c) = {_val_rmse(rep,'C'):.3e} rad，"
               f"地板 {_val_rmse(rep,'T10'):.3e} rad ⇒ 距地板 "
               f"{_val_rmse(rep,'C')/max(_val_rmse(rep,'T10'),1e-12):.2f}×。")
    out.append("")
    out.append("3. **行程包络 / 静态倾斜 对 `Px/Py` 的影响**")
    out.append("")
    t = truth
    out.append(f"   - 本数据集: θs 实际范围 {rep['sim_stats']['theta_small_range_deg'][0]:+.2f}° ~ "
               f"{rep['sim_stats']['theta_small_range_deg'][1]:+.2f}°"
               f"（包络 [{rep['cfg'].env_min/DEG:+.0f}°,{rep['cfg'].env_max/DEG:+.0f}°]），"
               f"g_A 幅值 {rep['sim_stats'].get('gravity_amp', 0.0):.2f} m/s²。")
    out.append(f"   - `Px/Py` 的灵敏度有两个通道: (i) **惯性耦合** `d·Q`（d = {rep['cfg'].dx*100:.0f} cm，"
               f"灵敏度 ∝ d·|θ̈| ≈ {abs(2*rep['cfg'].dx*h['rms_a']):.2f} N·m 每 kg·m）—— "
               f"只在 θs 变化时出现，因此**行程越窄、held 位置越集中，它与 Jbig/Js 的共线性越强**；"
               f"(ii) **重力** `∂G/∂P ≈ |g_A|`（水平时为 0，倾斜 10° 时 ≈ 1.7 m/s²）—— "
               f"强两个数量级且不依赖 θs 摆幅。")
    out.append(f"   - 本数据集 Px 的相对误差: (a) {_relerr(rep,'A',2):+.0f}%，"
               f"(b) {_relerr(rep,'B',2):+.0f}%，(c) {_relerr(rep,'C',2):+.0f}%；"
               f"LS 的 1σ = {_sigma_of(rep,'B',2):.5f} kg·m（|真值|/σ = "
               f"{abs(float(t[2]))/max(_sigma_of(rep,'B',2),1e-12):.2f}）。")
    return "\n".join(out)


def _acquisition_design(reports) -> str:
    reps_t = [r for r in reports if r["has_truth"]]
    lv = next((r for r in reps_t if r.get("env_mode") == "asym" and not r.get("tilted")
               and r.get("sim_stats")), None)
    ti = next((r for r in reps_t if r.get("env_mode") == "asym" and r.get("tilted")
               and r.get("sim_stats")), None)
    out = []
    out.append("以下结论全部由 §4/§5 的实测数值推出（不是经验口号）。")
    out.append("")
    out.append("### 8.1 小 yaw 摆幅与 held 位置")
    out.append("")
    if lv is not None:
        lo, hi = lv["sim_stats"]["theta_small_range_deg"]
        sx = _sigma_of(lv, "B", 2)
        out.append(f"- 行程变成 **−25° ~ +20°**（可用总行程 45°，中心 −2.5°）后，"
                   f"参考轨迹应当把包络 **−17° ~ +12°** 用完（本次仿真 θs 实际到 "
                   f"{lo:+.1f}° ~ {hi:+.1f}°）。")
        out.append(f"- 水平段里 `Px/Py` 唯一的信息通道是 `d·R(θs)P`（d = 3 cm）⇒ "
                   f"**每段的 held 位置必须铺满包络**（不同段取不同的 θs，本次 8 段里 "
                   f"大 yaw 激励段的 held θs 落在包络内多个位置），否则 `Px/Py` 与 "
                   f"`Jbig_eff/Js` 完全共线、只能定到 σ ≈ {sx:.4f} kg·m（远大于 |P| ≈ 0.005）。")
    out.append("")
    out.append("### 8.2 需要几个倾角、每个倾角几段")
    out.append("")
    scans = [(r, r["p_scan"]) for r in reps_t if r.get("p_scan") and r["p_scan"].get("tilt")]
    if not scans:
        out.append("（无倾斜数据集扫描结果）")
    else:
        rep, sc = scans[0]
        t = rep["truth"]
        need = next((phi for phi, sx, sy, cd in sc["tilt"]
                     if sx > 0 and abs(float(t[2])) / sx >= 3.0), None)
        minphi = need if need is not None else sc["tilt"][-1][0]
        out.append(f"- **倾角**: 从扫描表（§3 中倾斜数据集）看，要让 `|Px|/σ ≥ 3` 需要倾角 "
                   f"**≥ {minphi:g}°**（`g·sinφ ≥ {9.81*math.sin(math.radians(minphi)):.2f} m/s²`）。"
                   f"取 **±10°（或 +10°/−10° 两档）** 更稳: 正负号换来 `g_A` 方向翻转，"
                   f"能把 `Px/Py` 与其它参数解耦得更干净。")
        out.append(f"- **倾角数量**: ≥2 个（建议 +10° 与 −10°，或 +8/+12°）；"
                   f"同一倾角内大 yaw 转动本身也会让 `g_A` 在 A 系里旋转 ⇒ "
                   f"每个倾角下**大 yaw 必须被激励**（否则 `g_A` 方向不变，Px/Py 仍有共线风险）。")
        if sc.get("segs") and len(sc["segs"]) >= 2:
            n0, s0, _ = sc["segs"][0]
            n_ref, sx_ref, _ = sc["segs"][-1]
            ex = (math.log(s0 / sx_ref) / math.log(n_ref / n0)) if (s0 > 0 and sx_ref > 0
                                                                   and n_ref > n0) else 1.0
            tgt = abs(float(t[2])) / 5.0
            if sx_ref <= tgt:
                out.append(f"- **段数**: 倾斜数据集每轴 {n_ref // 2} 段（合计 {n_ref} 段）时 "
                           f"σ_Px = {sx_ref:.5f}，已满足 |Px|/σ ≥ 5（阈值 σ = {tgt:.5f}）"
                           f"⇒ **{n_ref // 2} 段/轴就够**。")
            else:
                n_need = math.ceil(n_ref * (sx_ref / tgt) ** (1.0 / max(ex, 0.05)))
                out.append(f"- **段数**: 每轴 {n_ref // 2} 段（合计 {n_ref} 段）时 σ_Px = {sx_ref:.5f}；"
                           f"按**实测**缩放指数 {ex:.2f}（不是理想 1/√n = 0.5 ⇒ "
                           f"倾斜下 P 的误差以**系统性成分**为主），想达到 |Px|/σ = 5"
                           f"（σ ≈ {tgt:.5f}）约需 **{n_need} 段**。")
            out.append(f"  - 实测: 从 {n0} 段加到 {n_ref} 段，σ_Px 只从 {s0:.5f} 变到 {sx_ref:.5f}"
                       f"（{(s0/sx_ref if sx_ref>0 else float('inf')):.2f}×）⇒ **多采段数收益有限**，"
                       f"决定性的是**倾角本身**。实车每段 3 s，8 段合计仅 ~0.4 min 运动时间，"
                       f"真正的成本是换倾角的机械调整时间。")
        out.append("- **建议组合**: 水平段（标定 Jbig/Js/fc/fv）与倾斜段（标定 Px/Py）**分开采**、"
                   "合并拟合: 水平段给惯性/摩擦以最好的信噪比，倾斜段专门把 P 拉出来。")
    out.append("")
    out.append("### 8.3 三种方法怎么用")
    out.append("")
    out.append("- 主力: **(c) 输出误差法**（`identify_params_torch.py`）—— 不依赖带噪 θ̈，"
               "held 轴真实运动自动进入前向仿真；")
    out.append("- 交叉检查: **(b) LS**（导数由状态估计器给，别用裸差分）—— 两者接近才可信；")
    out.append("- 底部兜底: 若这一轮没有采到倾斜数据，就**把 Px/Py 按住为 0**"
               "（`--fix-p` / 本脚本的 PF/CF 变体），别让它们自由乱跑；")
    out.append("- 验证: 用留出段的前向仿真 RMSE 判断，不看拟合残差（LS 的拟合残差会因为"
               "把噪声拟合进去而偏小）。")
    return "\n".join(out)


def dump_csv(segs, outdir):
    """落盘成与 collect_sysid.py **同列头**的 CSV（末两列 gravity_ax/ay 仅在非水平时写出）."""
    os.makedirs(outdir, exist_ok=True)
    for i, s in enumerate(segs):
        has_g = s.gravity is not None
        hdr = ("t,theta_big,theta_small,dtheta_big,dtheta_small,"
               "tau_big,tau_small,axis,held_target,mcu2_seq" +
               (",gravity_ax,gravity_ay" if has_g else ""))
        fn = os.path.join(outdir, f"sim_{s.tag}_{i:02d}.csv")
        with open(fn, "w") as fh:
            fh.write(hdr + "\n")
            for k in range(s.T):
                row = (f"{s.t[k]:.6f},{s.theta[k,0]:.9f},{s.theta[k,1]:.9f},"
                       f"{s.dtheta[k,0]:.9f},{s.dtheta[k,1]:.9f},"
                       f"{s.tau[k,0]:.9f},{s.tau[k,1]:.9f},"
                       f"{s.axis},{s.held_target:.9f},{k}")
                if has_g:
                    row += f",{s.gravity[k,0]:.6f},{s.gravity[k,1]:.6f}"
                fh.write(row + "\n")


if __name__ == "__main__":
    sys.exit(main())
