#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据段加载与角度预处理（CSV / npz）。

数据格式（与 ``collect_sysid`` 约定一致）:
  * CSV 列: ``t, theta_big, theta_small, dtheta_big, dtheta_small, tau_big, tau_small, axis,
    held_target, …``，**必须**还有 ``theta_big_motor, theta_big_platform``（★ 3-DOF 辨识必需）
  * npz 键: 同名列数组 + 标量 axis / dt(=0.01) / held_target；
    也可用 ``theta``(T,3) / ``dtheta``(T,3) / ``tau``(T,2) 的打包形式。
  * ★★ **角速度一律取"实际记录"的通道，绝不用差分从角度列重算**（用户要求）:
        · 电机侧   ``dtheta_big_motor``
        · 云台侧   ``dtheta_big_platform`` → 旧语义 ``dtheta_big``（= 云台侧）
        · 小 yaw   ``dtheta_small`` → ``small_joint_rate_est``
    缺任一列 ⇒ 该段**直接跳过并报明原因**（不再有"中心差分 + 平滑"的兜底）。
  * ★★ **角度在加载时做连续化，β 做圈数对齐**（用户约定，见 `continuous_angles` / `align_beta`）:
        · θ_motor / θ_platform / θ_small: 首值 wrap 到 (−π,π]，之后逐点相对前值 unwrap；
        · β（`backlash_center`）: 以 2π 为单位挪到使 Δ = θm − θp − β 落进 (−π,π] 的位置；
        · `theta_true` / `beta_true`（仿真真值帧）套用同一套；
        · 处理完后校验 β 的相邻差，> π 就询问 [Y/n]（Y=跳过该序列，非交互默认 Y）。
      ⇒ 去掉了 loss 里的 wrap，模型输出必须对**圈数**负责。
  * ★ **保持段（文件名带 `_hold`）只取前 3 s**（``--hold-max-sec``，默认 3.0）。
"""

from __future__ import annotations

import csv
import glob as globmod
import os
from dataclasses import dataclass, fields as _dataclass_fields, replace

import numpy as np

from .params import AXIS_BIG, AXIS_SMALL, DT_DEFAULT, QUANT_STEP


# ============================================================================
# 编码器量化（工具；**不做任何差分**）
# ============================================================================
def quantize(theta, step: float = QUANT_STEP):
    """编码器量化（模拟 8192 计数/整圈的读数）。"""
    return np.round(np.asarray(theta, dtype=np.float64) / step) * step


# ============================================================================
# ★★ 时间序列连续化 + β 圈数对齐（加载时对所有序列一次做完；用户约定）
# ============================================================================
def wrap_to_pi(x):
    """折到 (−π, π]（与损失/评测口径一致的 wrap）。"""
    x = np.asarray(x, dtype=np.float64)
    return np.arctan2(np.sin(x), np.cos(x))


def continuous_angles(theta):
    """**首值 wrap 到 (−π, π]，之后逐点相对前一个值 unwrap**（θ [T] 或 [T,K]）。

    等价于「先把首值折进 (−π,π]，再 np.unwrap」——np.unwrap 本身不改首值、逐点把差折进
    (−π,π] 再加回去，所以两者逐位一致（已逐点比对）。返回新数组，不改入参。
    """
    x = np.array(theta, dtype=np.float64, copy=True)
    if x.shape[0] == 0:
        return x
    x[0] = wrap_to_pi(x[0])
    return np.unwrap(x, axis=0)


def align_beta(beta_raw, theta_new, theta_raw):
    """把**记录**的 β 以 2π 为单位挪到使 ``Δ = θm − θp − β`` 落进 (−π, π] 的位置。

        k[i]    = round( (θm_new[i] − θp_new[i] − β_raw[i]) / 2π )
        β_new[i]= β_raw[i] + 2π·k[i]
        Δ_new[i]= Δ_raw[i] 折到 (−π, π]     （模 2π 不变，代表元取在 (−π,π]）

    ★ 不是"Δ 严格不变"：记录里的 β 偶尔会落在相邻的 2π 分支上（实测 78/1860 段
      |Δ_raw| > π），这里把它修回物理量级。返回 ``(β_new, k)``。
    """
    beta_raw = np.asarray(beta_raw, dtype=np.float64).reshape(-1)
    theta_new = np.asarray(theta_new, dtype=np.float64)
    D = theta_new[:, 0] - theta_new[:, 1] - beta_raw
    k = np.round(D / (2.0 * np.pi))
    return beta_raw + 2.0 * np.pi * k, k


def _beta_has_jump(beta, tol: float = np.pi):
    """处理后的 β 是否有相邻两点相差超过 π；返回 (有, 位置, 跳变量)。"""
    if beta is None or len(beta) < 2:
        return False, -1, 0.0
    d = np.abs(np.diff(np.asarray(beta, dtype=np.float64)))
    i = int(np.argmax(d))
    return bool(d[i] > tol), i, float(d[i])


def ask_skip_segment(seg, reason: str) -> bool:
    """询问是否跳过这条序列（缺 β / β 不连续共用）。**非交互（非 tty / EOF）默认 Y（跳过）**。

    Y（默认）⇒ 返回 True（跳过该序列）；n ⇒ 直接退出程序。
    """
    import sys
    msg = (f"[beta] 序列 {os.path.basename(str(seg.source))} {reason} —— 跳过这条序列？[Y/n] ")
    try:
        if not sys.stdin.isatty():
            print(msg + "（非交互 ⇒ 默认 Y：跳过）")
            return True
        ans = input(msg).strip().lower()
    except EOFError:
        print(msg + "（读到 EOF ⇒ 默认 Y：跳过）")
        return True
    if ans in ("", "y", "yes"):
        return True
    print("[beta] 选择退出。")
    raise SystemExit(2)


def check_beta_continuity(segs, verbose: bool = True):
    """校验**处理后**的 β 序列：相邻两点相差 > π ⇒ 询问 [Y/n]（Y=跳过该序列，默认）。

    ★ 没有任何 β 数据（既无 `backlash_center` 也无 `beta_true`）的序列，
    **用同一套 [Y/n] 提示跳过**（不再是"直接报错"）。
    """
    if not segs:
        return segs
    out, n_bad = [], 0
    for seg in segs:
        if seg.beta is None and seg.beta_true is None:
            n_bad += 1
            ask_skip_segment(seg, "没有任何 β 数据（既无 `backlash_center` 也无 `beta_true` 列）")
            continue
        bad = False
        for which, arr in (("β(backlash_center)", seg.beta), ("β_true", seg.beta_true)):
            has_jump, idx, jump = _beta_has_jump(arr)
            if has_jump:
                bad = True
                ask_skip_segment(seg, f"的{which}在 index {idx} 处相邻两点相差 "
                                      f"{jump:.4f} rad > π（处理后仍不连续）")   # n ⇒ SystemExit
        if not bad:
            out.append(seg)
        else:
            n_bad += 1
    if verbose and n_bad:
        print(f"[beta] 校验: {n_bad} 段不合格（缺 β / β 不连续），已跳过 "
              f"{len(segs) - len(out)}/{len(segs)} 段")
    return out


# ============================================================================
# 数据段
# ============================================================================
@dataclass
class Segment:
    """一段采集数据（两轴都有记录；axis 标明哪一轴被激励）。

    ★ 状态是 **3-DOF**: ``theta[:, 0] = θ_motor``（电机编码器，延时补偿后）、
      ``theta[:, 1] = θ_platform``（云台侧，IMU 反解）、``theta[:, 2] = θ_small``。
      ``tau[:, 0] = τ_cmd``（发给电机的力矩）、``tau[:, 1] = τ_small``。
    """

    t: np.ndarray                 # [T]
    theta: np.ndarray             # [T,3] 关节角（电机 / 云台 / 小 yaw）
    dtheta: np.ndarray            # [T,3] ★ **实际记录**的角速度（绝不来自差分）
    tau: np.ndarray               # [T,2] (τ_cmd_big, τ_small)
    axis: int = AXIS_BIG          # 0 = 大 yaw 被激励, 1 = 小 yaw 被激励
    held_target: float = 0.0
    dt: float = DT_DEFAULT
    mcu2_seq: np.ndarray | None = None
    gravity: np.ndarray | None = None      # [T,2] A 系重力平面分量 (m/s²)；None = 水平(0,0)
    # ★ [T] 背隙死区中心 β（**在线**估计值）= 数据列 `backlash_center`；None = 没有该列
    beta: np.ndarray | None = None
    beta_true: np.ndarray | None = None    # [T] 仿真真值 β（仅诊断/画图；实机恒 None/0）
    ddtheta: np.ndarray | None = None      # [T,3]（可选；oracle 消融用）
    theta_true: np.ndarray | None = None   # [T,3]（仅仿真数据有；评测用）
    dtheta_true: np.ndarray | None = None
    ddtheta_true: np.ndarray | None = None
    source: str = "?"
    tag: str = ""

    @property
    def T(self) -> int:
        return int(self.theta.shape[0])

    @property
    def driven(self) -> int:
        return self.axis

    @property
    def held(self) -> int:
        return 1 - self.axis


def _axis_code(v) -> int:
    """axis 列的容错解析: 0/1、'0'/'1'、'big'/'small' 都接受。"""
    if isinstance(v, (str, bytes, np.str_, np.bytes_)):
        return AXIS_SMALL if str(v).lower().startswith("s") else AXIS_BIG
    try:
        return int(np.asarray(v).reshape(-1)[0])
    except (TypeError, ValueError):
        return AXIS_BIG


def _col(rec, names, n=None):
    for nm in names:
        if nm in rec and rec[nm] is not None:
            return np.asarray(rec[nm], dtype=np.float64)
    raise KeyError(f"缺少列 {names}")


# ★ 3-DOF 辨识需要的列（电机侧 + 云台侧都要有）:
#   · `theta_big_motor`    = 电机侧（编码器 + 一阶延时补偿）—— 电机状态的真值
#   · `theta_big_platform` = 云台侧（IMU 反解 θ_p = ψ_platform − ψ_chassis）—— 云台状态
#   两者之差 Δ 就是传动形变（背隙）；**δ/k/c/J_motor 直接由模型拟合**。
_MOTOR_COLS = ("theta_big_motor", "theta_big", "theta_b")
_PLATFORM_COLS = ("theta_big_platform", "theta_platform")
_SMALL_COLS = ("theta_small", "theta_s")

# ★★ 角速度**只认实际记录列**（用户要求：不用差分代替记录值）。与三路角度一一对应:
#   motor    ← dtheta_big_motor（MCU 编码器侧；**不能**用 dtheta_big，后者旧语义 = 云台侧）
#   platform ← dtheta_big_platform，旧语义 dtheta_big = 云台侧（同源，可作兜底）
#   small    ← dtheta_small（= collect_sysid 里的 small_joint_rate）→ small_joint_rate_est
_MOTOR_RATE_COLS = ("dtheta_big_motor",)
_PLATFORM_RATE_COLS = ("dtheta_big_platform", "dtheta_big")
_SMALL_RATE_COLS = ("dtheta_small", "small_joint_rate_est")


def _rate_col(rec, names, what: str, source: str):
    """取**记录**的角速度列；缺失时给出明确原因（而不是退回差分重算）。"""
    for nm in names:
        if nm in rec and rec[nm] is not None:
            return np.asarray(rec[nm], dtype=np.float64).reshape(-1)
    raise KeyError(
        f"{source}: 缺少**记录**的{what}角速度列 {names} —— 本包不再用「中心差分 + 平滑」从"
        f"角度列重算角速度（用户要求: 拟合只用实际记录值）⇒ 该段无法参与拟合，请重采或改用"
        f"collect_sysid.py 写全量列的数据")



# ★ 保持段（`collect_sysid.py --record-hold` 落盘的、文件名带 `_hold` 后缀的那些段）里的
#   "静止保持"部分：到位+稳定判据满足之后就一直几乎不动了，后面的点是纯浪费算力，而且
#   长时间静止段会主导 loss（把参数往"零速摩擦"方向拉）⇒ 默认**只取前 3 s**。
HOLD_NAME_SUFFIX = "_hold"
HOLD_KEEP_SEC = 3.0


def is_hold_segment(seg: "Segment") -> bool:
    """按文件名后缀判定"静止保持段"（`collect_sysid.py` 的 HOLD_SUFFIX 约定）。"""
    return HOLD_NAME_SUFFIX in os.path.basename(str(seg.source))


def truncate_hold_segments(segs, max_sec: float = HOLD_KEEP_SEC, verbose: bool = True):
    """把**保持段**截断到前 ``max_sec`` 秒（按点数 = round(max_sec/dt)），普通段原样返回。

    只切"逐样本数组"，标量元数据（axis/held_target/dt/source/tag）不动。
    在**载入后立刻**做 ⇒ 拟合、留出评估、画图用的都是同一批截断后的数据。
    """
    if max_sec is None or max_sec <= 0:
        return segs
    out, n_cut, pts_kept, pts_drop = [], 0, 0, 0
    fields = [f.name for f in _dataclass_fields(Segment)]
    for seg in segs:
        T = int(seg.T)
        n = int(round(float(max_sec) / seg.dt))
        if not is_hold_segment(seg) or T <= n:
            out.append(seg)
            continue
        sl = slice(0, n)
        kw = {}
        for name in fields:
            v = getattr(seg, name)
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
                kw[name] = v[sl]
        out.append(replace(seg, **kw))
        n_cut += 1
        pts_kept += n
        pts_drop += T - n
    if verbose and n_cut:
        tot = pts_kept + pts_drop
        print(f"[hold] 保持段截断: {n_cut} 段 → 各取前 {max_sec:g} s（{pts_kept} 点），"
              f"丢 {pts_drop} 点（占这些段的 {100.0 * pts_drop / max(1, tot):.0f}%）"
              f"；普通段不截断")
    return out


def seg_fingerprint(seg: "Segment") -> str:
    """一段数据的指纹（只取"动力学内容": τ 与三通道角度）——用于检出**训练/留出重合**。

    ⚠ 这个检查是必须的: `collect_sysid.py` 对 `--seed` 是**确定性**的（默认 42），
    所以"另跑一次采留出集"会生成与训练集**逐位相同**的数据。指纹刻意不看 `t`。
    """
    import hashlib
    h = hashlib.md5()
    for a in (seg.tau, seg.theta):
        h.update(np.ascontiguousarray(np.round(np.asarray(a, dtype=np.float64), 6)).tobytes())
    return h.hexdigest()


def state_arrays(seg: "Segment", state_mode: str = "est"):
    """按 ``state_mode`` 取"状态目标": est = 记录/估计值（控制器真正看到的），
    true = 仿真真值（仅 dry-run 数据有；**上限对照**）。返回 (theta, dtheta)。

    ★ 角速度同样**只用记录值**（``dtheta_true``）；缺列直接报错，不再用差分从角度重算。
    """
    if state_mode == "true" and seg.theta_true is not None:
        th = np.asarray(seg.theta_true, dtype=np.float64)
        if th.ndim == 2 and th.shape[1] >= 3:
            if seg.dtheta_true is None:
                raise ValueError(
                    f"{seg.source}: `--state-mode=true` 需要**记录**的 `dtheta_true` 列；"
                    f"本包不再用差分从 `theta_true` 重算角速度")
            return th[:, :3], np.asarray(seg.dtheta_true, dtype=np.float64)[:, :3]
    return seg.theta, seg.dtheta


def segment_from_arrays(rec: dict, dt: float, source: str = "?") -> Segment:
    """把「同名列数组」字典转成 Segment（CSV 与 npz 共用）。"""
    if "theta" in rec and np.ndim(rec["theta"]) == 2:
        theta = np.asarray(rec["theta"], dtype=np.float64)
        if theta.shape[1] < 3:
            raise KeyError(
                f"{source}: 打包的 theta 只有 {theta.shape[1]} 列；3-DOF 辨识需要 3 列"
                "（电机 / 云台 / 小 yaw）")
        theta = theta[:, :3]
    else:
        tb = _col(rec, list(_MOTOR_COLS))          # 电机侧
        try:
            tp = _col(rec, list(_PLATFORM_COLS))   # ★ 云台侧（3-DOF 必需）
        except KeyError as exc:
            raise KeyError(
                f"缺少**云台侧**角度列 {_PLATFORM_COLS}（θ_p = platform_azimuth − "
                f"chassis_azimuth）。3-DOF 背隙辨识必须同时有电机侧与云台侧两列 ⇒ "
                f"12 列老数据需要重采（collect_sysid.py 已写全量列）") from exc
        ts = _col(rec, list(_SMALL_COLS))
        theta = np.stack([tb, tp, ts], axis=-1)
    if "tau" in rec and np.ndim(rec["tau"]) == 2:
        tau = np.asarray(rec["tau"], dtype=np.float64)[:, :2]
    else:
        tau = np.stack([_col(rec, ["tau_big", "tau_b"]), _col(rec, ["tau_small", "tau_s"])], axis=-1)
    # ★★ 角速度: 优先打包的 `dtheta`(T,3)（也是记录值），否则按三路显式记录列取。
    #   **绝不**用中心差分从角度列重算（用户要求：拟合只用实际记录值）。
    if "dtheta" in rec and np.ndim(rec.get("dtheta")) == 2:
        dtheta = np.asarray(rec["dtheta"], dtype=np.float64)
        if dtheta.shape[1] < 3:
            raise KeyError(
                f"{source}: 打包的 `dtheta` 只有 {dtheta.shape[1]} 列；3-DOF 需要 3 列"
                "（电机 / 云台 / 小 yaw）")
        dtheta = dtheta[:, :3]
    else:
        dtheta = np.stack([_rate_col(rec, _MOTOR_RATE_COLS, "电机侧", source),
                           _rate_col(rec, _PLATFORM_RATE_COLS, "云台侧", source),
                           _rate_col(rec, _SMALL_RATE_COLS, "小 yaw", source)], axis=-1)
    if dtheta.shape[0] != theta.shape[0]:
        raise ValueError(f"{source}: 记录角速度点数 {dtheta.shape[0]} 与角度点数 "
                         f"{theta.shape[0]} 不一致")
    if not np.all(np.isfinite(dtheta)):
        raise ValueError(f"{source}: 记录角速度列里有非有限值（NaN/Inf）")

    T = theta.shape[0]
    t = _col(rec, ["t"]) if ("t" in rec and rec["t"] is not None) else np.arange(T) * dt
    axis = _axis_code(rec.get("axis", AXIS_BIG))
    held = float(np.asarray(rec.get("held_target", 0.0)).reshape(-1)[0])
    seq = rec.get("mcu2_seq")
    seq = None if seq is None else np.asarray(seq, dtype=np.float64)
    # ── A 系重力平面分量（静态倾斜时非零；水平时全 0）──
    grav = None
    if "gravity" in rec and np.ndim(rec.get("gravity")) == 2:
        grav = np.asarray(rec["gravity"], dtype=np.float64)[:, :2]
    elif rec.get("gravity_ax") is not None and rec.get("gravity_ay") is not None:
        grav = np.stack([np.asarray(rec["gravity_ax"], dtype=np.float64),
                         np.asarray(rec["gravity_ay"], dtype=np.float64)], axis=-1)
    if grav is not None and (grav.shape[0] != theta.shape[0] or not np.any(grav != 0.0)):
        grav = None if not np.any(grav != 0.0) else grav
    # ── ★ 背隙中心 β（在线估计列）──
    beta = None
    if rec.get("beta") is not None:
        beta = np.asarray(rec["beta"], dtype=np.float64)
    elif rec.get("backlash_center") is not None:
        beta = np.asarray(rec["backlash_center"], dtype=np.float64)
    if beta is not None:
        beta = beta.reshape(-1)
        if beta.shape[0] != theta.shape[0] or not np.all(np.isfinite(beta)):
            beta = None
        elif not np.any(beta != 0.0):
            beta = None            # 全 0 = 没有可用信息（老数据/实机恒 0）
    bt = rec.get("backlash_beta_true")
    beta_true = None if bt is None else np.asarray(bt, dtype=np.float64).reshape(-1)
    dd = rec.get("ddtheta")
    dd = None if dd is None else np.asarray(dd, dtype=np.float64)
    # ★ 仿真真值状态（[T,3]）: npz 里是打包好的 `theta_true`；CSV 里是 6 个 `*_true_*` 列
    th_true = rec.get("theta_true")
    if th_true is None and rec.get("theta_true_motor") is not None:
        th_true = np.stack([_col(rec, ["theta_true_motor"]),
                            _col(rec, ["theta_true_platform"]),
                            _col(rec, ["theta_true_small"])], axis=-1)
    th_true = None if th_true is None else np.asarray(th_true, dtype=np.float64)
    if th_true is not None and not np.any(th_true != 0.0):
        th_true = None                      # 实机数据这 6 列恒 0 = 未知
    dth_true = rec.get("dtheta_true")
    if dth_true is None and rec.get("dtheta_true_motor") is not None:
        dth_true = np.stack([_col(rec, ["dtheta_true_motor"]),
                             _col(rec, ["dtheta_true_platform"]),
                             _col(rec, ["dtheta_true_small"])], axis=-1)
    dth_true = None if dth_true is None else np.asarray(dth_true, dtype=np.float64)
    if dth_true is not None and not np.any(dth_true != 0.0):
        dth_true = None
    dd_true = rec.get("ddtheta_true")
    dd_true = None if dd_true is None else np.asarray(dd_true, dtype=np.float64)

    # ── ★★ 连续化 + β 圈数对齐（用户约定；在**加载时**一次做完，拟合/评测/画图共用）──
    #   est 帧: 三路角度先 wrap 首值再逐点 unwrap；β 用同一帧的位移以 2π 为单位对齐。
    theta_raw = theta
    theta = continuous_angles(theta)
    if beta is not None:
        beta, _ = align_beta(beta, theta, theta_raw)
    #   true 帧（仅仿真数据有）: 同一套处理，保持 θ_true 与 β_true 自洽。
    if th_true is not None:
        th_true_raw = th_true
        th_true = continuous_angles(th_true)
        if beta_true is not None:
            beta_true, _ = align_beta(beta_true, th_true, th_true_raw)

    return Segment(t=t, theta=theta, dtheta=dtheta, tau=tau, axis=axis, held_target=held,
                   dt=dt, mcu2_seq=seq, gravity=grav, beta=beta, ddtheta=dd,
                   theta_true=th_true, dtheta_true=dth_true, ddtheta_true=dd_true,
                   source=source, beta_true=beta_true)


def _read_csv(path: str) -> list:
    rows = []
    with open(path, "r", newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        first_fields = [s.strip() for s in first.strip().split(",")]
        try:
            [float(s) for s in first_fields]
            has_header = False
        except ValueError:
            has_header = True
        if has_header:
            rd = csv.DictReader(fh)
            for r in rd:
                rows.append({k.strip(): (float(v) if v not in (None, "") else np.nan)
                             for k, v in r.items() if k is not None})
        else:
            names = ["t", "theta_big", "theta_small", "dtheta_big", "dtheta_small",
                     "tau_big", "tau_small", "axis", "held_target", "mcu2_seq"]
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                vals = [float(s) for s in line.split(",")]
                rows.append({nm: vals[i] for i, nm in enumerate(names) if i < len(vals)})
    if not rows:
        return []
    cols = {k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64) for k in rows[0]}
    for k in ("dtheta_big", "dtheta_small", "mcu2_seq"):
        if k in cols and np.all(np.isnan(cols[k])):
            cols.pop(k)
    return [cols]


def load_segments(patterns, dt_override: float | None = None, verbose: bool = True) -> list:
    """读入若干 **npz**（推荐，采集脚本默认格式）/ csv（老数据）（glob 或显式路径）。

    ★ 同一段的多种格式只读一份（优先 npz > csv）: 老目录里 ``x.csv`` 与 ``x.npz`` 常常成对
      存在，若都读进来会把同一段数据加载两次 —— 既污染留出集统计，也白费算力。
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pat in patterns:
        for p in str(pat).split(","):
            p = p.strip()
            if not p:
                continue
            hit = sorted(globmod.glob(p))
            if not hit and os.path.isfile(p):
                hit = [p]
            files.extend(hit)
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    # ── ★ 同一段的多种格式去重: 优先 npz（列是 csv 的超集），其次 csv ──
    _pref = {".npz": 0, ".csv": 1}
    by_stem: dict = {}
    order: list = []
    dropped: list = []
    for f in files:
        stem, ext = os.path.splitext(f)
        cur = by_stem.get(stem)
        if cur is None:
            by_stem[stem] = f
            order.append(stem)
        elif _pref.get(ext, 9) < _pref.get(os.path.splitext(cur)[1], 9):
            by_stem[stem] = f
            dropped.append(cur)
        else:
            dropped.append(f)
    if dropped and verbose:
        print(f"[load] 同一段有 {len(dropped)} 个重复格式，只读优先的那份"
              f"（npz > csv）: " + ", ".join(os.path.basename(d) for d in dropped[:3])
              + (" …" if len(dropped) > 3 else ""))
    files = [by_stem[s] for s in order]

    segs = []
    skipped = []
    for path in files:
        # 单个文件格式不符（列名不匹配/损坏）时只跳过它，不让整次拟合失败
        try:
            if path.endswith(".npz"):
                d = np.load(path, allow_pickle=False)
                rec = {k: d[k] for k in d.files if d[k].ndim >= 1}
                scal = {}
                for k in d.files:                       # 标量元数据（非数值的如 tag 自动忽略）
                    if d[k].ndim == 0:
                        try:
                            scal[k] = float(d[k])
                        except (TypeError, ValueError):
                            pass
                dt = dt_override or float(scal.get("dt", DT_DEFAULT))
                if "axis" in scal:
                    rec["axis"] = scal["axis"]
                if "held_target" in scal:
                    rec["held_target"] = scal["held_target"]
                segs.append(segment_from_arrays(rec, dt, source=path))
            elif path.endswith(".csv"):
                got = 0
                for cols in _read_csv(path):
                    dt = dt_override or DT_DEFAULT
                    # ★ 采样周期优先取**记录**的 `dt` 列；只有老 CSV 没有该列时，才用记录的
                    #   时间戳 `t` 的间隔推（这是"时间基"、不是被拟合的状态量，不涉及用差分
                    #   替代记录值）。
                    if "dt" in cols and len(cols["dt"]):
                        _dv = cols["dt"][np.isfinite(cols["dt"]) & (cols["dt"] > 1e-6)]
                        if _dv.size:
                            dt = dt_override or float(np.median(_dv))
                    if dt_override is None and "dt" not in cols and "t" in cols and len(cols["t"]) > 2:
                        dts = np.diff(cols["t"])
                        dts = dts[np.isfinite(dts) & (dts > 1e-6)]
                        if dts.size:
                            dt = float(np.median(dts))
                    segs.append(segment_from_arrays(cols, dt, source=path))
                    got += 1
                if got == 0:
                    raise ValueError("空文件")
            else:
                skipped.append((path, "不支持的后缀"))
        except Exception as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
    if verbose:
        for path, why in skipped:
            print(f"[load] 跳过 {path} —— {why}")
        print(f"[load] 共读入 {len(segs)} 段数据（{len(files)} 个文件，跳过 {len(skipped)} 个）")
    # ── ★ 处理后的 β 连续性校验（相邻两点相差 > π 就询问 [Y/n]；非交互默认跳过）──
    return check_beta_continuity(segs, verbose=verbose)
