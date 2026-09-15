"""临时脚本: 在**同 10 段留出集**上，用同一条前向仿真通道比较
   先验 / 真值 / LS(measured) / LS(ideal) / LS(drop) / torch 的参数质量（用完即删）。"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_ident_methods as cm
import identify_params_torch as ip

TRUTH = np.array([0.050, 0.020, 0.0087, 0.0050, 0.220, 0.055, 0.0973, 0.028])
NAMES = list(ip.PARAM_NAMES)

def parse_ls(path):
    """同时兼容 C++ LS 工具（p.fcBig = ...;）与 torch 脚本（fc_big = ...）两种格式。"""
    t = open(path).read()
    d = {}
    alias = {"fcBig": "fc_big", "fvBig": "fv_big", "fcSmall": "fc_small", "fvSmall": "fv_small"}
    for k, v in re.findall(r"[p\.\s]?(Jbig_eff|Js|Px|Py|fcBig|fvBig|fcSmall|fvSmall|fc_big|fv_big|fc_small|fv_small)\s*=\s*([-\d.eE+]+)", t):
        d[alias.get(k, k)] = float(v)
    return np.array([d["Jbig_eff"], d["Js"], d["Px"], d["Py"],
                     d["fc_big"], d["fv_big"], d["fc_small"], d["fv_small"]])

def main():
    import argparse
    ap = argparse.ArgumentParser(description="在同一留出集上统一评参（模型 λ=100）")
    ap.add_argument("--dataset", default=os.environ.get("EVAL_DATASET", "data/archive/20260915_sim_sysid/datasets/sim100_tilt10_lam1e5"))
    ap.add_argument("--cand", action="append", default=[],
                    help="label=path（可重复；path 为 LS/torch 输出的 txt）")
    ap.add_argument("--lambda-model", type=float, default=100.0)
    a = ap.parse_args()
    ds_dir = a.dataset
    hold = sorted(glob.glob(os.path.join(ds_dir, "*.csv")))
    val = [f for f in hold if re.search(r"_(4[5-9]|9[5-9])\.csv$", f)]
    try:
        segs = ip.load_segments(val, verbose=False)
    except TypeError:
        segs = ip.load_segments(val)
    cfg = cm.SimConfig(small_env_mode="asym")
    prior = ip.default_param_vector()

    cands = [("先验(CAD 占位)", prior, 100.0),
             ("★ 真值(地板)", TRUTH, 100.0)]
    for spec in a.cand:
        lab, path = spec.rsplit("=", 1)
        cands.append((lab, parse_ls(path), 100.0))

    print(f"数据集: {ds_dir}  留出集: {len(val)} 段 ({len(segs)} 载入)  λ_model={a.lambda_model:g}; 前向仿真驱动=记录力矩\n")
    hdr = f"{'方法':<28}{'Jbig':>9}{'Js':>9}{'Px':>10}{'Py':>10}{'fc_b':>8}{'fv_b':>8}{'fc_s':>8}{'fv_s':>8} | {'RMSE大':>8}{'RMSE小':>8}"
    print(hdr); print("-" * len(hdr))
    for name, phi, lam in cands:
        rel = (np.asarray(phi) - TRUTH) / TRUTH * 100
        # 统一用辨识模型 λ=10（即实际下发控制器的模型）；λ=100 的"数值下限"单独一条
        r = cm.forward_rmse(np.asarray(phi), segs, cfg, lambda_override=a.lambda_model)
        print(f"{name:<28}" + "".join(f"{v:>9.4f}" if i < 2 else f"{v:>10.4f}"
                                      for i, v in enumerate(np.asarray(phi))) +
              f" | {r[0]['rmse_meas']:>8.5f}{r[1]['rmse_meas']:>8.5f}", flush=True)
        print(f"{'   → 相对误差 %':<28}" + "".join(f"{v:>9.1f}" if i < 2 else f"{v:>10.1f}"
                                                  for i, v in enumerate(rel)))

def floor_lambda100(segs, cfg):
    """真值参数 + λ=100 的数值下限（需要更小积分步，很慢 ⇒ 单独算）。"""
    from dataclasses import replace
    cfg2 = replace(cfg, validate_substeps=2)      # λ=100 ⇒ 内部再 ×50 = 100 子步
    r = cm.forward_rmse(TRUTH, segs, cfg2, lambda_override=100.0)
    print(f"{'★ 真值 + λ=100 (数值下限)':<28}" + " " * 72 +
          f" | {r[0]['rmse_meas']:>8.5f}{r[1]['rmse_meas']:>8.5f}", flush=True)


if __name__ == "__main__":
    main()
    if "--floor" in sys.argv:
        import glob, re
        val = [f for f in sorted(glob.glob("data/archive/20260915_sim_sysid/datasets/sim100/*.csv"))
               if re.search(r"_(4[5-9]|9[5-9])\.csv$", f)]
        floor_lambda100(ip.load_segments(val, verbose=False), cm.SimConfig(small_env_mode="asym"))
