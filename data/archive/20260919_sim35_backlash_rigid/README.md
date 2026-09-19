# 20260919 — 「刚性接触 + 死区完全自由 + β 随机/漂移」仿真数据集（100 段训练 + 8 段留出）

> 这是用户要求的**仿真环境 2**（`collect_sysid.py --dry-run --sim-rigid`）采出来的**测试数据**:
> 背隙**接触面完全刚性**（k = ∞，撞击为完全非弹性冲击）、**死区内完全自由**（τ_t ≡ 0，
> 连阻尼都没有）、且**每条数据的死区中心 β 单独随机**并带**微弱随时间漂移**。
> 用途: 把"3-DOF + 平滑死区"这套建模/辨识/控制链路逼到最不利情形，看三类曲线怎么变。
> 环境实现与结论见 **`docs/backlash_model.md`**（§2.1.1 β 可观测性、§2.4 环境说明、§7 验证结果）。

## 1. 数据

| 项 | 值 |
|---|---|
| 生成命令（训练100段） | `python3 python/scripts/collect_sysid.py --dry-run --sim-rigid --segments=100 --duration-sec=3 --no-record-hold --out=<out>` |
| 留出（测试） | 同命令 `--segments=8`（另一次运行 ⇒ 另一批随机 β0 与激励） |
| 段数 / 点数 | 训练 **100 段**、留出 **8 段**；每段 300 点（3 s @100 Hz） |
| 被控对象参数 | `Jbig_eff=0.050, Js=0.020, Px=0.00866, Py=0.005, fc_big=0.220, fv_big=0.055, fc_small=0.0973, fv_small=0.028`（λ=100，积分步长 0.05 ms） |
| 电机侧 | `J_motor=0.006, fcMotor=0.030, fvMotor=0.010` |
| 背隙 | `δ=0.0873 rad (5.0°)`；接触 `k=∞`（刚性约束）、`c=0`、`γ=0`；**β0 ~ U(−0.3δ, +0.3δ) 每段重抽**，漂移 `±0.05δ / 30 s` |
| 几何 | `dx=0, dy=0.07`（仓库默认实测几何）；底盘静止 |

数据里多出 8 列（实机恒 0）: `backlash_center`（在线估计 β）、`backlash_beta_true`（真值）、
`theta_true_*` / `dtheta_true_*`（真值状态）。

## 2. 训练与评估（`fit/`）

| 文件 | 内容 |
|---|---|
| `fit_train100_1000ep.log` | **主结果**: 100 段 × 1000 epoch（全批，`--substeps=2`），在线 β + 估计状态 |
| `ident_convergence.png` | 收敛曲线（loss + 16 个参数各自曲线，虚线=初值） |
| `ident_traj.png` | ★ **模型 vs 仿真环境** 运动曲线（**留出集**，三通道 θ/θ̇ + β 真值/在线/观测中心） |
| `mpc_tracking.png` | ★ **控制轨迹 vs 目标轨迹**（刚性被控对象 + 随机漂移 β + 在线 β 估计） |
| `fit_beta_true.log` | 诊断: β 用**真值**（上限对照） |
| `fit_state_true.log` | 诊断: 状态用**真值**（上限对照） |

**10000 epoch 加练**（`fit10k/`，同一批数据、同一协议/种子，`--eval-every=250` 记录留出学习曲线）:

> ⚠ **`fit10k/` 的"留出集"数字是污染过的**：采集脚本对 `--seed` 确定性，"另跑一次"采出来的
> 留出集与训练集逐位相同（8/8 重合）⇒ 那里的"留出误差"只能当**训练集拟合误差**读。
> 已修正并重做的版本见 **`fit4x/`**（400 段训练 + 32 段真留出，含 γ 冻结/自由、100/400 段对照）。

| 文件 | 内容 |
|---|---|
| `fit10k/ident_convergence.png` | 10000 epoch 的收敛曲线（loss + 16 参） |
| `fit10k/ident_learning.png` | ★ **留出集开环误差 vs epoch**（学习曲线: 多训练到底有没有用） |
| `fit10k/ident_traj.png` | 模型 vs 环境（留出集三通道 + β） |
| `fit10k/mpc_tracking_fitted.png` | ★ 控制 vs 目标（10000 epoch 拟合参数） |
| `fit10k/compare_runs.png` | 三组 10000 epoch 运行的参数轨迹对比（在线 β / β 真值 / 状态真值） |
| `fit10k/compare_1k_vs_10k.png` | 1000 vs 10000 epoch 的参数轨迹对比 |
| `fit10k/main.log` / `beta.log` / `state.log` | 三次运行的完整日志 |
| `fit10k/mpc_eval.log` | 10000 epoch 参数的闭环评估（E1~E4） |

复现:

```bash
# ① 采集（约 35 min）
python3 python/scripts/collect_sysid.py --dry-run --sim-rigid --segments=100 \
        --duration-sec=3 --no-record-hold --out=/tmp/rigid_train
python3 python/scripts/collect_sysid.py --dry-run --sim-rigid --segments=8 \
        --duration-sec=3 --no-record-hold --out=/tmp/rigid_val
# ② 训练（100 段 × 1000 epoch；全批，约 50 min）
python3 python/scripts/identify_params_torch.py --data='/tmp/rigid_train/*.csv' \
        --val-data='/tmp/rigid_val/*.csv' --epochs=1000 --batch-segments \
        --substeps=2 --threads=4 --beta-mode=auto --state-mode=est \
        --plot-out=.../fit/ident --out=.../fit/params.txt
# ②' 加练到 10000 epoch（同协议/种子，带留出学习曲线），三条日志一起对比
python3 python/scripts/identify_params_torch.py --data='/tmp/rigid_train/*.npz' \
        --val-data='/tmp/rigid_val/*.csv' --epochs=10000 --batch-segments \
        --substeps=2 --threads=4 --eval-every=250 --print-every=250 \
        --beta-mode=auto --state-mode=est --plot-out=.../fit10k/ident
python3 python/scripts/plot_ident_compare.py --log='main.log,beta.log,state.log' \
        --names='在线β,β真值,状态真值' --out=.../fit10k/compare_runs.png

# ③ 闭环（刚性被控对象 + 随机漂移 β，控制器用在线 β）
./build/tcbs_mpc_param_eval --plant-rigid --plant-beta-random=0.0262 \
        --plant-beta-drift=0.0044 --plant-beta-period=30 --plant-seed=7 \
        --phi=<拟合8参> --phi2=<拟合背隙8参> --dump=.../fit/mpc
python3 python/scripts/plot_mpc_tracking.py --csv='.../fit/mpc_*.csv' \
        --out=.../fit/mpc_tracking.png
```

## 3. 结论摘要（完整版见 `docs/backlash_model.md` §7.5）

训练: 100 段 × **1000 epoch**（全批，`--substeps=2`，§2 命令），用时 **185 s**。

1. **收敛曲线**（`ident_convergence.png`）: loss 从 ~0.10 降到 ~0.06（每 epoch 只抽 0.1 s 片段，
   所以曲线本身噪声大，看趋势即可）；16 个参数里 `δ/k/c/Jmotor/电机摩擦` 很快稳定，
   平面那 8 个（尤其 `fv_big`、`Px/Py`）在 1000 步内还在缓慢移动 —— 它们在本数据集里
   要么不可辨识（`P`）、要么被接触模型的结构误差吸收（`fv_big`）。
2. **模型 vs 仿真环境**（`ident_traj.png`，**留出集 8 段**）: 窗口 0.1 s 角度 RMSE
   电机/云台/小 yaw = **0.945° / 0.187° / 0.543°**（初值 1.054/0.224/0.890）；
   整段开环会漂到 6°/6°/11°（3 s 开环，接触事件时序误差累积，属预期）。
   第三列是 β: 真值 / 在线估计 / 本段 Δ 极差中心 —— 能直接看出"只有穿越死区的段才可观测"。
3. **控制 vs 目标**（`mpc_tracking_fitted.png`，被控对象 = 刚性接触 + 随机漂移 β）:
   RMS **0.0302 / 0.1654 / 0.0071 rad**（阶跃 0.6 / 大阶跃 1.2 / 正弦），
   对照"控制器用准确参数"的 0.0261/0.1578/0.0069 ⇒ **只差 +17%/+5%/+3%，失败数 0**。
   β 换成真值或直接取 0 对闭环影响 ≤2%。
4. **上限对照**: β 用真值时 `δ` 从 +21% 收敛到 **+1.6%** ⇒ **δ 的高估主要来自 β 的在线估计误差**
   （其根源是电机角估计误差 ≈25 mrad，见 §2.1.1）；拟合残差几乎不变，说明模型是用
   "更大的 δ + 更大的 γ"把 β 偏置吸收掉了。
