"""背隙标定（大 yaw）: 从 collect_sysid.py 记录的数据里直接估 **宽度 δ** 与 **中心 β**。

★ 定位（与 identify_params_torch.py 的关系）:
    参数辨识（`identify_params_torch.py`）已经把 **δ/k/c/γ/J_motor/电机摩擦/β 直接作为模型
    参数一起拟合**（3-DOF，无任何特殊处理）。本脚本**不参与**那条拟合路径，只提供两件事:
      ① 一个**独立的初值/交叉校验**（`--backlash-delta=auto` 的粗估更粗，见下）；
      ② 诊断: 若这里估出的 δ 与拟合值差很多，说明数据里两列有一列不对（或链路时延没标定）。
    所以它是"量一下背隙有多宽"的尺子，不是辨识的前置步骤。

原理（不需要 3-DOF 拟合）:
    Δ_raw = theta_big_motor − theta_big_platform
    两条曲线都来自同一次采集:
      · `theta_big_motor`    = MCU 编码器（电机侧，带链路延迟/保持）
      · `theta_big_platform` = IMU 反解（云台侧，θ_p = platform_azimuth − chassis_azimuth）
    云台双向旋转 ⇒ 一个窗口内**两侧翼都会被访问到**:
      · 极差 max(Δ_raw) − min(Δ_raw) 就是传动能"空走"的总范围 ⇒ **δ（宽度）**
      · 中点 (max+min)/2 就是死区中心 ⇒ **β**（注意: 中心随电机/云台共同旋转而漂移，
        且云台角由 IMU 推出会有漂移 ⇒ 这只作为离线参考，运行期由估计器在线给）

用法:
    python3 python/scripts/calibrate_backlash.py --data='data/cars/Sentry1/sysid/*.npz'
    python3 python/scripts/calibrate_backlash.py --data='.../*.npz' --json-out=... --skip-first 5

注意:
    · 需要**新格式**的 CSV（含 `theta_big_motor` / `theta_big_platform` 两列）；
    · 用分位数（默认 0.1%/99.9%）取极值，抗个别坏帧；
    ★ **必须做延迟补偿**（默认开，`--no-delay-comp` 关）:
      注意两个列的区别:
        · `theta_big_motor`      = 估计器**已做延时补偿**的电机角（加过 θ̇·transport_delay）
        · `theta_big_motor_meas` = **原始**滞后测量（未补偿）
      ⇒ `--delay-comp`（默认开）用**原始列**做一阶补偿:
          Δ_c = theta_big_motor_meas + θ̇_platform·big_enc_age − theta_big_platform
        关掉它则直接用 `theta_big_motor`（已补偿，但补偿量由估计器的 transport_delay 决定）。
      两者都只在 **|θ̇_platform| 较小** 的样本上取极值（`--rate-max`，默认 0.5 rad/s），
      把"延迟×角速度"排除掉，剩下的才是真正的"空走"范围。
    · 偏置/符号问题会表现为"δ 明显偏大" ⇒ 本脚本同时打印两列的统计量。
"""
import argparse
import glob as globmod
import json
import math
import os
import sys

import numpy as np


def _load(path):
    """读一段数据（**npz（采集脚本默认格式）或 csv（老数据）**），只取标定需要的列。

    返回 ``(m, mm, p, age, rate)`` 或 ``None``（缺电机侧/云台侧列 ⇒ 跳过该文件）。
    """
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}

        def get(name):
            v = arrs.get(name)
            return None if v is None else np.atleast_1d(np.asarray(v, dtype=float))
    else:
        d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
        if d is None or d.dtype.names is None:
            return None

        def get(name):
            if name not in d.dtype.names:
                return None
            return np.atleast_1d(np.asarray(d[name], dtype=float))

    m = get("theta_big_motor")
    p = get("theta_big_platform")
    if m is None or p is None:
        return None
    mm = get("theta_big_motor_meas")
    if mm is None:
        mm = m
    age = get("big_enc_age")
    if age is None:
        age = np.zeros_like(m)
    rate = get("dtheta_big_platform")
    if rate is None:
        rate = np.zeros_like(m)
    ok = np.isfinite(m) & np.isfinite(p) & np.isfinite(age) & np.isfinite(rate)
    age = np.clip(np.where(age < 0.0, 0.0, age), 0.0, 0.5)   # 负年龄 = 未知 ⇒ 不补偿
    if not np.any(ok):
        return None
    return m[ok], mm[ok], p[ok], age[ok], rate[ok]


def main():
    ap = argparse.ArgumentParser(description="大 yaw 背隙 δ / 中心 β 标定（从记录数据直接估）")
    ap.add_argument("--data", required=True,
                    help="数据 glob（**npz（默认格式）/ csv**，可逗号分隔多个）")
    ap.add_argument("--q-lo", type=float, default=0.1, help="下分位（%%），默认 0.1")
    ap.add_argument("--q-hi", type=float, default=99.9, help="上分位（%%），默认 99.9")
    ap.add_argument("--skip-first", type=int, default=1,
                    help="跳过最开始的 N 个文件（首段的起始位姿任意，默认 1）")
    ap.add_argument("--no-delay-comp", action="store_true",
                    help="关闭传输延迟补偿（不推荐；会把 延迟×角速度 算进 δ）")
    ap.add_argument("--rate-max", type=float, default=0.5,
                    help="只在这些样本上取极值: |θ̇_platform| ≤ 该值 (rad/s)，默认 0.5")
    ap.add_argument("--age-max", type=float, default=0.005,
                    help="★ 只用**刚刷新**的电机样本: big_enc_age ≤ 该值 (s)，默认 5 ms。"
                         "这是分离背隙与链路延迟/保持的**关键**条件（值保持期间电机角是旧的，"
                         "平台却在动 ⇒ 差值里含 平台位移，会虚增 δ）")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    files = []
    for pat in a.data.split(","):
        hit = sorted(globmod.glob(pat))
        if not hit and os.path.isfile(pat):
            hit = [pat]
        files.extend(hit)
    files = sorted(set(files))[a.skip_first:]
    if not files:
        print("没有可读的数据（npz/csv；需要新格式: 含 theta_big_motor / theta_big_platform）")
        return 1

    lows, highs, spans, centers, ns, pool = [], [], [], [], [], []
    used = skipped = 0
    for f in files:
        r = _load(f)
        if r is None:
            skipped += 1
            continue
        m, mm, p, age, rate = r
        if a.no_delay_comp:
            D = m - p                     # 已补偿列，直接用
        else:
            D = mm + rate * age - p       # 原始列 + 一阶延迟补偿
        # 两个条件一起做成掩码（分两步筛会让数组长度不一致）
        mask = (np.abs(rate) <= a.rate_max) & (age <= a.age_max)
        D = D[mask]
        if len(D) < 20:
            skipped += 1
            continue
        lo, hi = np.percentile(D, a.q_lo), np.percentile(D, a.q_hi)
        lows.append(lo); highs.append(hi)
        spans.append(hi - lo); centers.append(0.5 * (hi + lo))
        pool.append(D)
        ns.append(len(D)); used += 1
    if used == 0:
        print("没有可用的段（列名不对？还是老格式的数据？）")
        return 1

    spans = np.array(spans); centers = np.array(centers)
    # 用"所有段极值"的并集给一个整体估计（比逐段中位数更贴物理: 每段只走了部分行程）
    d_all = spans
    delta_med = float(np.median(d_all))
    delta_max = float(np.max(d_all))
    beta_med = float(np.median(centers))

    print(f"段数 {used}（跳过 {skipped}）  每段可用点数中位 {int(np.median(ns))}"
          f"  （已按 |θ̇|≤{a.rate_max} rad/s 且 age≤{a.age_max*1e3:.1f} ms 过滤）")
    if np.median(ns) < 50:
        print("  ⚠ 过滤后点数太少 —— δ 会被链路延迟污染；优先把 MCU1↔MCU2 链路提速")
    print("\n=== 逐段 Δ_raw = theta_big_motor − theta_big_platform 的极差 ===")
    # ★ **合并所有段**的过滤后 Δ 再取极差 —— 单段可能只走了一侧翼（中位数会偏低），
    #   合并后"两侧翼都被访问到"的概率最高 ⇒ 这才是 δ 的最佳估计。
    D_pool = np.concatenate(pool) if pool else np.zeros(1)
    delta_pool = float(np.percentile(D_pool, a.q_hi) - np.percentile(D_pool, a.q_lo))
    print(f"  δ（逐段极差）: 中位 {math.degrees(delta_med):.3f}°  最大 {math.degrees(delta_max):.3f}°"
          f"  最小 {math.degrees(d_all.min()):.3f}°")
    print(f"  δ（★ 全段合并, 推荐值）= {math.degrees(delta_pool):.3f}°  （{delta_pool:.6f} rad）")
    print(f"  中心 β   : 中位 {math.degrees(beta_med):+.3f}°  "
          f"（标准差 {math.degrees(centers.std()):.3f}° ← 大说明 IMU 漂移/多圈解卷绕有问题）")
    print(f"\n  ⇒ 建议: --backlash-delta = {delta_pool:.6f} rad   ({math.degrees(delta_pool):.3f}°)"
          f"（作为 torch 3-DOF 拟合的初值；拟合会自己再修它）")
    print(f"     （离线参考的 β = {beta_med:+.6f} rad；运行期用估计器的 backlash_center 在线值）")
    if delta_med < math.radians(0.2):
        print("\n  ⚠ δ 估出来接近 0: 要么这台车背隙真的很小，要么这两列有一列不对"
              "（例如数据是旧格式、或 theta_big_platform 没被记录/恒为 0）")
    out = dict(n_segments=used, n_skipped=skipped,
               delta_rad=delta_pool, delta_rad_median=delta_med,
               delta_max_rad=delta_max, delta_deg=math.degrees(delta_pool),
               beta_rad=beta_med, beta_std_rad=float(centers.std()),
               q_lo=a.q_lo, q_hi=a.q_hi, skip_first=a.skip_first, files=len(files))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\n[save] {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
