# 归档：仿真采集数据与辨识结果（2026-09-15）

> **这是历史归档，不要在这里写新数据。** 新的实机采集数据请写到 `data/sysid/`；
> 新一次辨识的结果写到 `data/sysid/` 或另建 `data/archive/<日期>_<用途>/`。
> 本目录是为了"不干扰新数据"而把 2026-09-15 那轮**仿真验证**的全部产物集中存放。

## 这些数据是什么

用 `python/scripts/dump_sim_dataset.py`（内部调用 `compare_ident_methods.collect_sim`）做的
**闭环仿真采集**：录制目标序列做激励 + 上位机 PID、**分轴**（一轴激励、另一轴 PID 保持）、
角度按 2π/8192 量化、`dtheta` 由量化角中心差分 + 3 点平滑、100 Hz / 每段 300 点 / 每轴 50 段（共 100 段）。
列头与实机 `collect_sysid.py` 的 CSV 一致（倾斜数据集多两列 `gravity_ax,gravity_ay`）。

真值参数（用户给定的真实量级）：

| 参数 | `Jbig_eff` | `Js` | `Px` | `Py` | `fc_big` | `fv_big` | `fc_small` | `fv_small` |
|---|---|---|---|---|---|---|---|---|
| 值 | 0.050 | 0.020 | 0.0087 | 0.0050 | 0.220 | 0.0550 | 0.0973 | 0.0280 |

几何 `dx = 0.10 m`、`dy = 0`；小 yaw 行程 `[−25°, +20°]`；`θ*（离心平衡点）= −29.9°`（在行程之外）。

## 目录

| 路径 | 被控对象 λ | 底盘 | 用途 |
|---|---|---|---|
| `datasets/sim100/` | 100 | 水平 | 最早一轮：plant λ=100 vs 模型 λ=10 的"λ 失配"演示（LS 的 `fv_small` 被估成负数） |
| `datasets/sim100_lam10/` | 10 | 水平 | λ 匹配对照（隔离"摩擦形状失配"的影响） |
| `datasets/sim100_lam1e4/` | 1e4 | 水平 | 主要一轮：模型 λ=100/1000 × 三种拟合方法 |
| `datasets/sim100_lam1e5/` | 1e5 | 水平 | 极端理想库仑（≈sign）下的水平对照 |
| `datasets/sim100_tilt10_lam1e5/` | 1e5 | **固定倾斜 10°（全部段同一倾角）** | ★ 证明"固定一个倾角即可让 `P` 可辨识" |
| `ls_runs/*.txt` | — | — | C++ 线性最小二乘（`tools/identify_params`）各模式/各 λ 的辨识输出 |
| `torch/ident_torch_*.{txt,log,png}` | — | — | torch 输出误差法的参数、训练日志、收敛曲线与轨迹对比图 |
| `dump_lam1e5.log` | — | — | λ=1e5 两组数据的采集日志（含子步/耗时/θs 范围/越限检查） |

采集命令（可复现）：

```bash
python3 python/scripts/dump_sim_dataset.py --plant-lambda=1e5 --plant-substep=2e-6 --tilt-deg=10 \
        --out=data/archive/20260915_sim_sysid/datasets/sim100_tilt10_lam1e5        # ≈290 s
python3 python/scripts/dump_sim_dataset.py --plant-lambda=1e4 \
        --out=data/archive/20260915_sim_sysid/datasets/sim100_lam1e4              # ≈40 s
```

## 结论摘要（详细表格见 `docs/sysid_ls_vs_torch.md`）

1. **torch 输出误差法 > LS**：惯量/摩擦 1~6% vs LS 的 16~72%（LS 的 θ̈ 来自量化角二阶差分，噪声与真值同量级）。
2. **水平数据下 `Px/Py` 不可辨识**（LS −1200%~−2100%，torch 也只能压到 ±25%）；
   改成**固定一个 10° 倾角贯穿全程**后 torch 的 `Px` 误差从 −96.6% 降到 **+21.7%** ⇒ 倾斜段是必需的。
3. **模型 λ=100 足够**：与 λ=1000 相比参数几乎相同、跟踪完全等价，但 λ=1000 需要 32 个积分子步
   （最坏 12~20 ms，超出 10 ms 控制周期）⇒ 在线用 λ=100 + `substeps=4`。
4. **plant λ=1e4→1e5 对结论没有影响**（`fv_small` 一直偏低 ~31~38%，是"模型形状 vs 真库仑"的失配被吸进粘滞项；
   `fc` 仍准到 1~4%）。

## 注意

- `datasets/*` 每套 100 个 CSV（~3.2 MB），五套合计 ~16 MB —— 归档进 git 是为了留证，不必再解压处理；
  要重新拟合直接用 CSV 路径即可，例如：
  `./build/tcbs_identify_params <csv...> --held=measured --dx=0.1 --dy=0 --lambda=100`
- 这些数据的"真值参数"是为了验证**方法**而设的仿真量，**不能**直接当作实车参数使用。
