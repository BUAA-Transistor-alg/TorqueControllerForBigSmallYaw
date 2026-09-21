"""18 个参数: 被控对象真值 vs 水平/倾斜两组数据的辨识值 + 绝对/相对误差。

真值来源: 被控对象配置（`--sim-no-backlash --sim-no-backlash-k=2500 --sim-com-seed=7`，
水平/倾斜共享同一组随机重心），其中 P/Pb 的真值从 npz 的 `*_true` 标量读。
用法: python3 data/archive/20260920_sim_com_tilt/table_all_params.py [beta-mode]
      beta-mode ∈ {auto, fit}（默认 auto = 运行期口径）
"""
import os
import sys
import glob
import math
import numpy as np

A = os.path.dirname(os.path.abspath(__file__))
MODE = (sys.argv[1] if len(sys.argv) > 1 else "auto").lower()
ARMS = {"水平": f"level_{'auto' if MODE == 'auto' else 'betafit'}",
        "倾斜": f"tilt_{'auto' if MODE == 'auto' else 'betafit'}"}

NAMES = ["Jbig_eff", "Js", "Px", "Py", "fc_big", "fv_big", "fc_small", "fv_small",
         "backlash_delta", "backlash_k", "backlash_c", "backlash_through",
         "Jmotor", "fc_motor", "fv_motor", "backlash_beta", "Pbx", "Pby"]
UNITS = ["kg·m²", "kg·m²", "kg·m", "kg·m", "N·m", "N·m·s/rad", "N·m", "N·m·s/rad",
         "rad", "N·m/rad", "N·m·s/rad", "—", "kg·m²", "N·m", "N·m·s/rad", "rad",
         "kg·m", "kg·m"]
# 备注: 该参数在这次拟合里是什么角色
NOTE = {
    "backlash_through": "★ 冻结(γ 固定 0.002, 不辨识)",
    "backlash_beta": "☆ 不拟合(β 用数据里的在线列)",
    "backlash_delta": "拟合(真值 0 ⇒ 相对误差无意义)",
}


def read_truth():
    f = sorted(glob.glob(f"{A}/level_train/*.npz"))[0]
    d = np.load(f, allow_pickle=False)
    return dict(
        Jbig_eff=0.050, Js=0.020,
        Px=float(d["px_true"]), Py=float(d["py_true"]),
        fc_big=0.220, fv_big=0.055, fc_small=0.0973, fv_small=0.028,
        backlash_delta=0.0, backlash_k=2500.0, backlash_c=2.0, backlash_through=0.002,
        Jmotor=0.006, fc_motor=0.030, fv_motor=0.010, backlash_beta=0.0,
        Pbx=float(d["pbx_true"]), Pby=float(d["pby_true"]))


def parse_params(path):
    v = {}
    for line in open(path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, val = line.split("=", 1)
            try:
                v[k.strip()] = float(val.strip())
            except ValueError:
                pass
    return v


T = read_truth()
idv = {}
for tag, arm in ARMS.items():
    p = f"{A}/{arm}/params.txt"
    if not os.path.exists(p):
        raise SystemExit(f"[error] 缺少 {p}（先跑 fits.sh）")
    idv[tag] = parse_params(p)

print(f"β 口径: --beta-mode={MODE}   （状态: --state-mode=est；3000 epoch；刚性大 yaw）")
print(f"真值: 被控对象 = `--sim-no-backlash --sim-no-backlash-k=2500 --sim-com-seed=7`；"
      f"水平 |g_A|=0，倾斜 10° |g_A|=1.703 m/s²\n")

hdr = (f"{'参数':<17}{'单位':<12}{'真值':>13}{'水平辨识':>13}{'倾斜辨识':>13}"
       f"{'水平绝对误差':>14}{'倾斜绝对误差':>14}{'水平相对':>11}{'倾斜相对':>11}  备注")
print(hdr)
print("-" * len(hdr))
rows = {}
for i, n in enumerate(NAMES):
    t = T[n]
    a = idv["水平"].get(n, float("nan"))
    b = idv["倾斜"].get(n, float("nan"))
    da, db = a - t, b - t
    ra = f"{da / abs(t) * 100:+.2f}%" if abs(t) > 1e-12 else "—"
    rb = f"{db / abs(t) * 100:+.2f}%" if abs(t) > 1e-12 else "—"
    note = NOTE.get(n, "")
    print(f"{n:<17}{UNITS[i]:<12}{t:>13.6f}{a:>13.6f}{b:>13.6f}"
          f"{da:>+14.6f}{db:>+14.6f}{ra:>11}{rb:>11}  {note}")
    rows[n] = (t, a, b, da, db)

print("\n矢量参数（一阶矩）按「幅值 + 方向」看:")
for tag, key in (("P", ("Px", "Py")), ("Pb", ("Pbx", "Pby"))):
    tv = np.array([T[key[0]], T[key[1]]])
    av = np.array([idv["水平"][key[0]], idv["水平"][key[1]]])
    bv = np.array([idv["倾斜"][key[0]], idv["倾斜"][key[1]]])
    ang = lambda x: (math.degrees(math.acos(max(-1, min(1, float(x @ tv / (np.linalg.norm(x) * np.linalg.norm(tv)))))))
                     if np.linalg.norm(x) > 1e-12 else float("nan"))
    print(f"  |{tag}| 真值 {np.linalg.norm(tv):.5f} | 水平 {np.linalg.norm(av):.5f}"
          f"（{np.linalg.norm(av)/np.linalg.norm(tv)*100:5.1f}%）方向差 {ang(av):5.1f}°"
          f" | 倾斜 {np.linalg.norm(bv):.5f}（{np.linalg.norm(bv)/np.linalg.norm(tv)*100:5.1f}%）"
          f"方向差 {ang(bv):5.1f}°")

print("\n哪个采样更接近真值（按 |相对误差|，矢量参数按方向差 + 幅值相对误差之和）:")
win = {"水平": 0, "倾斜": 0, "平": 0}
for i, n in enumerate(NAMES):
    t, a, b, da, db = rows[n]
    if n in ("backlash_beta", "backlash_through"):
        continue
    ea = abs(da) if abs(t) > 1e-12 else abs(da) / 0.001      # 真值 0 时用绝对误差比
    eb = abs(db) if abs(t) > 1e-12 else abs(db) / 0.001
    w = "水平" if ea < eb * 0.999 else ("倾斜" if eb < ea * 0.999 else "平")
    win[w] += 1
print("  " + "  ".join(f"{k}: {v}" for k, v in win.items() if v))
