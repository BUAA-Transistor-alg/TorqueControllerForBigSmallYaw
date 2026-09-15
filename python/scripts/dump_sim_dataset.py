"""临时脚本: 用 compare_ident_methods 的同一套闭环仿真采集 100 段并落盘（用完即删）。

与 compare_ident_methods.main() 的数据构造完全一致:
  分轴 50+50 段、录制目标序列做激励、上位机 PID、plant λ=100 / 0.05 ms 子步、
  角度按 2π/8192 量化、dtheta 由量化角中心差分 + 3 点平滑、逐样本记录 gravity_ax/ay（水平 ⇒ 0）。
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_ident_methods as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "data", "sysid", "sim100")   # 可用 --out 覆盖
N_PER_AXIS = 50           # 每轴 50 段 ⇒ 合计 100 段
SEED = 42

def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--plant-lambda", type=float, default=1.0e4)
    ap.add_argument("--n-per-axis", type=int, default=N_PER_AXIS)
    ap.add_argument("--tilt-deg", type=float, default=0.0,
                    help="底盘固定倾角（度，绕 y 轴）: 全部段用同一个倾角 ⇒ 各段 held 大 yaw 方位角"
                         "不同 ⇒ A 系里 g_A 方向铺开，用来辨识 P")
    ap.add_argument("--plant-substep", type=float, default=2.0e-5,
                    help="plant 积分步长（λ=1e4 用 2e-5；λ=1e5 需 ≤5e-6）")
    a = ap.parse_args()
    OUT = a.out
    rng = np.random.default_rng(SEED)
    raw_lib = cm.load_target_library(["data/targets/*.npz"])
    if not raw_lib:
        raw_lib = cm.synth_targets(rng, 8)

    truth = cm.PlanarParams(dx=cm.DX_M, dy=cm.DY_M,
                            friction_lambda=cm.FRICTION_LAMBDA).with_vector(cm.TRUTH_VECTOR)
    cfg = cm.SimConfig(small_env_mode="asym", plant_lambda=a.plant_lambda,
                       plant_substep=a.plant_substep,
                       tilt_deg_list=(a.tilt_deg,) if a.tilt_deg != 0.0 else ())

    specs = []
    for _ in range(a.n_per_axis):
        specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], 0))   # 0 = 大 yaw 被激励
    for _ in range(a.n_per_axis):
        specs.append((raw_lib[int(rng.integers(0, len(raw_lib)))], 1))   # 1 = 小 yaw 被激励

    t0 = time.time()
    out = cm.collect_sim(cfg, truth, specs, rng, verbose=True)
    segs = out["segs"]
    os.makedirs(OUT, exist_ok=True)
    cm.dump_csv(segs, OUT)
    n_big = sum(1 for s in segs if s.axis == 0)
    print(f"[dump] {len(segs)} 段（big {n_big} / small {len(segs)-n_big}）"
          f" × {cfg.seg_len} 点 @ {1.0/cfg.dt:.0f} Hz → {OUT}/  用时 {time.time()-t0:.1f}s")
    print(f"[dump] 真值 φ = {np.array2string(cm.TRUTH_VECTOR, precision=4)}")
    print(f"[dump] 几何 dx={cm.DX_M} dy={cm.DY_M}; 小 yaw θs 范围 "
          f"{out['theta_small_range_deg'][0]:+.2f}° ~ {out['theta_small_range_deg'][1]:+.2f}°; "
          f"g_A 幅值 {out['gravity_amp']:.3f} m/s²")
    print(f"[dump] θ* (平衡点) = {cm.THETA_STAR_TRUE_DEG:+.3f}°")

if __name__ == "__main__":
    main()
