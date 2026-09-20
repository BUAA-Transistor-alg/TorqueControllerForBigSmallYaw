"""把 5 个对照臂的结果汇总成一张表（无背隙仿真 · δ 可辨识性）。"""
import os, re, sys, glob
import numpy as np

A = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Jbig_eff", "Js", "Px", "Py", "fc_big", "fv_big", "fc_small", "fv_small",
         "backlash_delta", "backlash_k", "backlash_c", "backlash_through",
         "Jmotor", "fc_motor", "fv_motor", "backlash_beta"]
ARMS = ["A_bogus_delta", "B_tiny_delta", "C_mid_delta", "D_current_defaults", "E_true_state",
        "F_bogus_true_state", "G_bogus_betafit", "H_tiny_betafit", "G10k_bogus_betafit"]
# 备注（β 来源 / 状态来源）:
#   A~D: β=在线列, 状态=估计       E: β=在线列, 状态=真值
#   F:   β=在线列, 状态=真值       G/H/G10k: β=拟合全局常数(--beta-mode=fit), 状态=真值


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
    return np.array([v[n] for n in NAMES])


def parse_log(path):
    txt = open(path, errors="ignore").read()
    out = {}
    m = re.search(r"初值 φ0 = \[([^\]]+)\]", txt)
    if m:
        out["init"] = np.array([float(x) for x in m.group(1).split()])
    m = re.search(r"估计参数（留出集）: 窗口\(0\.1s\) RMSE: 角度\[°\][^=]*=\s*"
                  r"([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+); 角速度\[rad/s\]\s*=\s*"
                  r"([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", txt)
    if m:
        out["val_deg"] = [float(x) for x in m.groups()[:3]]
        out["val_rate"] = [float(x) for x in m.groups()[3:]]
    m = re.search(r"前 5 均值=([\d.eE+-]+) → 后 5 均值=([\d.eE+-]+)", txt)
    if m:
        out["loss_head"], out["loss_tail"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"共读入 (\d+) 段", txt)
    if m:
        out["n_seg"] = int(m.group(1))
    m = re.search(r"整段开环 RMSE: 角度\[°\][^=]*=\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", txt)
    if m:
        out["full_deg"] = [float(x) for x in m.groups()]
    return out


print(f"{'臂':<20}{'δ初值':>9}{'δ拟合':>9}{'k拟合':>10}{'c拟合':>8}"
      f"{'留出角度 电/台/小 [°]':>26}{'整段开环 电/台/小 [°]':>26}{'loss 首→末':>20}")
print("-" * 122)
for a in ARMS:
    pr, lg = f"{A}/{a}/params.txt", f"{A}/{a}/run.log"
    if not os.path.exists(pr):
        print(f"{a:<20}  (还没有 params.txt)"); continue
    phi = parse_params(pr)
    info = parse_log(lg)
    i0 = info.get("init")
    d0 = f"{i0[8]:.4f}" if i0 is not None else "?"
    vd = info.get("val_deg"); fd = info.get("full_deg")
    vds = "/".join(f"{x:.3f}" for x in vd) if vd else "?"
    fds = "/".join(f"{x:.2f}" for x in fd) if fd else "?"
    ls = (f"{info['loss_head']:.4e}→{info['loss_tail']:.4e}"
          if "loss_head" in info else "?")
    print(f"{a:<20}{d0:>9}{phi[8]:>9.4f}{phi[9]:>10.1f}{phi[10]:>8.3f}{vds:>26}{fds:>26}{ls:>20}")

print("\n平面/电机那 11 个参数（留出集拟合后 vs 仿真真值）")
TRUTH = np.array([0.050, 0.020, 0.00866, 0.005, 0.220, 0.055, 0.0973, 0.028,
                  np.nan, np.nan, np.nan, 0.002, 0.006, 0.030, 0.010, 0.0])
print(f"{'臂':<20}" + "".join(f"{NAMES[i]:>11}" for i in [0, 1, 4, 5, 6, 7, 12, 13, 14]))
print(f"{'真值':<20}" + "".join(f"{TRUTH[i]:>11.5f}" for i in [0, 1, 4, 5, 6, 7, 12, 13, 14]))
for a in ARMS:
    pr = f"{A}/{a}/params.txt"
    if not os.path.exists(pr):
        continue
    phi = parse_params(pr)
    print(f"{a:<20}" + "".join(f"{phi[i]:>11.5f}" for i in [0, 1, 4, 5, 6, 7, 12, 13, 14]))
