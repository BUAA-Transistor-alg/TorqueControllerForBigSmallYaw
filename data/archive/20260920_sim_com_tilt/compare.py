"""水平 vs 倾斜 两组数据、4 个辨识臂的参数误差对比（重点: 重心 P / Pb）。"""
import os, re, glob, math
import numpy as np

A = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Jbig_eff","Js","Px","Py","fc_big","fv_big","fc_small","fv_small",
         "backlash_delta","backlash_k","backlash_c","backlash_through",
         "Jmotor","fc_motor","fv_motor","backlash_beta","Pbx","Pby"]
ARMS = [("level_auto","水平 auto"), ("tilt_auto","倾斜 auto"),
        ("level_betafit","水平 βfit"), ("tilt_betafit","倾斜 βfit")]


def truth():
    f = sorted(glob.glob(f"{A}/level_train/*.npz"))[0]
    d = np.load(f, allow_pickle=False)
    return dict(Px=float(d["px_true"]), Py=float(d["py_true"]),
                Pbx=float(d["pbx_true"]), Pby=float(d["pby_true"]))


def parse_params(p):
    v = {}
    for line in open(p):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, val = line.split("=", 1)
            try: v[k.strip()] = float(val.strip())
            except ValueError: pass
    return v


def parse_log(p):
    t = open(p, errors="ignore").read(); o = {}
    m = re.search(r"估计参数（留出集）: 窗口\(0\.1s\) RMSE: 角度\[°\][^=]*=\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", t)
    if m: o["win"] = [float(x) for x in m.groups()]
    m = re.search(r"整段开环 RMSE: 角度\[°\][^=]*=\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", t)
    if m: o["full"] = [float(x) for x in m.groups()]
    m = re.search(r"前 5 均值=([\d.eE+-]+) → 后 5 均值=([\d.eE+-]+)", t)
    if m: o["loss"] = (float(m.group(1)), float(m.group(2)))
    return o


T = truth()
print(f"重心真值（--sim-com-seed=7）: P = ({T['Px']:+.5f}, {T['Py']:+.5f}) |P|={math.hypot(T['Px'],T['Py']):.5f} kg·m")
print(f"                              Pb = ({T['Pbx']:+.5f}, {T['Pby']:+.5f}) |Pb|={math.hypot(T['Pbx'],T['Pby']):.5f} kg·m\n")
hdr = f"{'臂':<14}{'Px 误差':>10}{'Py 误差':>10}{'|P| 相对':>10}{'P 方向误差':>12}" \
      f"{'Pbx 误差':>10}{'Pby 误差':>10}{'|Pb| 相对':>11}{'Pb 方向误差':>13}" \
      f"{'留出窗口 电/台/小 [°]':>24}{'整段开环':>22}"
print(hdr); print("-"*len(hdr))
rows = {}
for arm, tag in ARMS:
    pp = f"{A}/{arm}/params.txt"
    if not os.path.exists(pp):
        print(f"{tag:<14} (还没有 {arm}/params.txt)"); continue
    v = parse_params(pp); lg = parse_log(f"{A}/{arm}/run.log")
    P = np.array([v["Px"], v["Py"]]); Pb = np.array([v["Pbx"], v["Pby"]])
    Pt = np.array([T["Px"], T["Py"]]); Pbt = np.array([T["Pbx"], T["Pby"]])
    ang = lambda a, b: math.degrees(math.acos(max(-1, min(1, float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))))))) if np.linalg.norm(a)>0 and np.linalg.norm(b)>0 else float("nan")
    win = "/".join(f"{x:.3f}" for x in lg.get("win", [float('nan')]*3))
    full = "/".join(f"{x:.2f}" for x in lg.get("full", [float('nan')]*3))
    print(f"{tag:<14}{P[0]-Pt[0]:>+10.5f}{P[1]-Pt[1]:>+10.5f}"
          f"{np.linalg.norm(P)/np.linalg.norm(Pt):>10.2f}{ang(P,Pt):>12.1f}"
          f"{Pb[0]-Pbt[0]:>+10.5f}{Pb[1]-Pbt[1]:>+10.5f}"
          f"{np.linalg.norm(Pb)/np.linalg.norm(Pbt):>11.2f}{ang(Pb,Pbt):>13.1f}"
          f"{win:>24}{full:>22}")
    rows[arm] = v

print("\n机械量（应保持不变/接近真值；真值 = 初值）:")
mech = ["Jbig_eff","Js","fc_big","fv_big","fc_small","fv_small","backlash_delta","backlash_k",
        "backlash_c","Jmotor","fc_motor","fv_motor"]
TR = dict(Jbig_eff=0.050, Js=0.020, fc_big=0.220, fv_big=0.055, fc_small=0.0973, fv_small=0.028,
          backlash_delta=0.0, backlash_k=2500.0, backlash_c=2.0, Jmotor=0.006, fc_motor=0.030, fv_motor=0.010)
print(f"{'臂':<14}" + "".join(f"{n:>13}" for n in mech))
print(f"{'真值':<14}" + "".join(f"{TR[n]:>13.5f}" for n in mech))
for arm, tag in ARMS:
    if arm in rows:
        print(f"{tag:<14}" + "".join(f"{rows[arm][n]:>13.5f}" for n in mech))
