"""把各次辨识结果放到**同一个留出集**（按时间最后 20%）上排名。

被评参数集: 共同初值 / ident18(10k,3e-4) / ident18_cos(2k,1e-2+cos) / cos40k(40k,1e-2+cos20k)
            / lr3e4_200k（最终）+ 它的若干断点。
"""
import sys, glob, os, numpy as np
sys.path.insert(0, 'python/scripts')
import identify_params_torch as ip

D = 'data/cars/Sentry1'
NAMES = ip.PARAM_NAMES
ALL = sorted(glob.glob(f'{D}/sysid/*.npz'))
N = len(ALL); NV = N // 5
VAL_FILES = ALL[N - NV:]
TRAIN_FILES = ALL[:N - NV]
print(f"数据: {N} 个 npz；留出（最后 20%）= {len(VAL_FILES)} 个 = {os.path.basename(VAL_FILES[0])} … "
      f"{os.path.basename(VAL_FILES[-1])}")

val = ip.truncate_hold_segments(ip.load_segments([",".join(VAL_FILES)], verbose=False), 3.0, verbose=False)
tr = ip.truncate_hold_segments(ip.load_segments([",".join(TRAIN_FILES)], verbose=False), 3.0, verbose=False)
tilt = lambda ss: [s for s in ss if s.gravity is not None and float(np.mean(np.hypot(s.gravity[:, 0], s.gravity[:, 1]))) >= 1.0]
print(f"留出集: {len(val)} 段（其中斜坡 {len(tilt(val))} 段）；训练集 {len(tr)} 段\n")

INIT = np.array([0.045614, 0.008116, 0.021348, -0.007430, 0.096245, 0.237374, 0.033434, 0.048466,
                 0.0873, 200.0, 2.0, 0.002, 0.006, 0.030, 0.010, 0.0, 0.0, 0.0])


def parse(path):
    v = {}
    for ln in open(path):
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, x = ln.split("=", 1)
            try: v[k.strip()] = float(x.strip())
            except ValueError: pass
    return np.array([v[n] for n in NAMES])


SETS = [("共同初值", INIT),
        ("ident18 (10k @3e-4)", parse(f'{D}/ident18/params.txt')),
        ("ident18_cos (2k @1e-2+cos)", parse(f'{D}/ident18_cos/params.txt')),
        ("cos40k (40k @1e-2+cos20k)", parse(f'{D}/ident18_cos40k/params.txt')),
        ("lr3e4_200k 最终", parse(f'{D}/ident18_lr3e4_200k/params.txt'))]
for e in (20000, 50000, 100000, 150000, 190000):
    p = f'{D}/ident18_lr3e4_200k/params.txt.ep{e:06d}'
    if os.path.exists(p): SETS.append((f"  200k 的 ep{e//1000}k", parse(p)))

base = ip.PlanarParams(dx=0.0, dy=0.07)
hdr = (f"{'参数集':<26}{'留出窗口 电/台/小 [°]':>26}{'留出整段 电/台/小 [°]':>26}"
       f"{'留出·斜坡 电/台/小':>24}{'三个角度均值':>12}")
print(hdr); print("-" * len(hdr))
rows = []
for tag, phi in SETS:
    r = ip.channel_rmse(val, phi, base, integrator="rk4", substeps=2, use_beta=True, state_mode="est")
    rt = ip.channel_rmse(tilt(val), phi, base, integrator="rk4", substeps=2, use_beta=True, state_mode="est")
    w = (r["motor_deg"] + r["platform_deg"] + r["small_deg"]) / 3
    rows.append((w, tag, phi, r, rt))
for w, tag, phi, r, rt in sorted(rows):
    print(f"{tag:<26}"
          f"{r['motor_deg']:>10.3f}/{r['platform_deg']:>6.3f}/{r['small_deg']:>6.3f}"
          f"{r['motor_full_deg']:>10.2f}/{r['platform_full_deg']:>6.2f}/{r['small_full_deg']:>6.2f}"
          f"{rt['motor_deg']:>9.3f}/{rt['platform_deg']:>6.3f}/{rt['small_deg']:>6.3f}"
          f"{w:>12.4f}")
print("\n★ 排名第一:", sorted(rows)[0][1])
