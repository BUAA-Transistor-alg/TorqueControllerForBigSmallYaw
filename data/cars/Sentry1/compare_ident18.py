"""对比 data/cars/Sentry1/{ident18, ident18_cos} 两组参数（同一批 240 段数据、同一评测口径）。

· 两组都是"全批配方"训练：ident18 = 10000 epoch、lr=3e-4 常数；ident18_cos = 2000 epoch、
  lr=1e-2 + 最后 1000 epoch 余弦衰减到 0。**都在这 240 段上训练**（没给 --val-data）
  ⇒ 下面全是**训练集**口径，两组可比，但不是泛化误差。
· 指标: `channel_rmse` 窗口(0.1 s)口径（主）+ 整段开环口径（参考）；按"平放/斜坡"分组再算一遍
  （斜坡段才有重力信息 ⇒ P/Pb 由它们决定）。
"""
import sys, glob, math, numpy as np
sys.path.insert(0, 'python/scripts')
import identify_params_torch as ip

D = 'data/cars/Sentry1'
NAMES = ip.PARAM_NAMES
SETS = [("ident18 (10k, lr3e-4)", f"{D}/ident18/params.txt"),
        ("ident18_cos (2k, lr1e-2+cos)", f"{D}/ident18_cos/params.txt"),
        ("cos40k (40k, lr1e-2+cos20k)", f"{D}/ident18_cos40k/params.txt"),
        # ★ 两组运行**共同的初值** = 改默认值之前的那组（ident18 与 ident18_cos 都从它出发）
        ("共同初值(改动前默认)", "INIT")]


def parse(path):
    v = {}
    for ln in open(path):
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, x = ln.split("=", 1)
            try:
                v[k.strip()] = float(x.strip())
            except ValueError:
                pass
    return np.array([v[n] for n in NAMES])


INIT = np.array([0.045614, 0.008116, 0.021348, -0.007430, 0.096245, 0.237374,
                 0.033434, 0.048466, 0.096463, 157.8279, 2.591145, 0.002,
                 0.005455, 0.004139, 0.030332, 0.0, 0.0, 0.0])
phi = {tag: (INIT if p == "INIT" else parse(p)) for tag, p in SETS}
segs = ip.truncate_hold_segments(ip.load_segments([f'{D}/sysid/*.npz'], verbose=False), 3.0, verbose=False)
# 按倾斜分组: |g_A| 均值 ≥ 1.0 m/s²（≈6°）算斜坡
tilt, level = [], []
for s in segs:
    gm = float(np.mean(np.hypot(s.gravity[:, 0], s.gravity[:, 1]))) if s.gravity is not None else 0.0
    (tilt if gm >= 1.0 else level).append(s)
print(f"数据: {len(segs)} 段 = 斜坡 {len(tilt)} + 平放 {len(level)}（保持段已截 3 s）\n")

base = ip.PlanarParams(dx=0.0, dy=0.07)
res = {}
for tag, _ in SETS:
    res[tag] = {}
    for name, ss in (("全部", segs), ("斜坡", tilt), ("平放", level)):
        res[tag][name] = ip.channel_rmse(ss, phi[tag], base, integrator="rk4", substeps=2,
                                         use_beta=True, state_mode="est")
    print(f"{tag:<30} 全部 {len(segs)} 段: 窗口 {res[tag]['全部']['motor_deg']:.3f}/"
          f"{res[tag]['全部']['platform_deg']:.3f}/{res[tag]['全部']['small_deg']:.3f}°  "
          f"整段 {res[tag]['全部']['motor_full_deg']:.2f}/{res[tag]['全部']['platform_full_deg']:.2f}/"
          f"{res[tag]['全部']['small_full_deg']:.2f}°", flush=True)

print("\n分组（窗口口径，角度 [°] 电机/云台/小yaw）:")
hdr = f"{'组':<28}{'全部':>26}{'斜坡':>26}{'平放':>26}"
print(hdr); print("-"*len(hdr))
for tag, _ in SETS:
    row = f"{tag:<28}"
    for name in ("全部", "斜坡", "平放"):
        r = res[tag][name]
        row += f"{r['motor_deg']:>8.3f}/{r['platform_deg']:>6.3f}/{r['small_deg']:>6.3f}"
    print(row)

print("\n18 个参数对比（Δ = cos − ident18；相对 = Δ/|ident18|）:")
print(f"{'参数':<18}{'ident18':>13}{'ident18_cos':>13}{'Δ':>13}{'相对':>10}")
for i, n in enumerate(NAMES):
    a, b = phi["ident18 (10k, lr3e-4)"][i], phi["ident18_cos (2k, lr1e-2+cos)"][i]
    d = b - a
    rel = f"{d/abs(a)*100:+.1f}%" if abs(a) > 1e-12 else "—"
    print(f"{n:<18}{a:>13.6f}{b:>13.6f}{d:>+13.6f}{rel:>10}")
