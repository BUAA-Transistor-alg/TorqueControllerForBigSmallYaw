# 20260919 — 大 yaw 背隙 3-DOF 模型的仿真验证数据

> 这一批是**纯 dry-run 仿真**数据（`collect_sysid.py --dry-run`，被控对象 = C++ 同方程 +
> 背隙），用途是**验证 3-DOF 模型与 16 参 torch 辨识这条新链路**，不是实车标定数据。
> 结论与完整说明见 **`docs/backlash_model.md` §7**。

## 1. 数据（`data/`）

| 项 | 值 |
|---|---|
| 生成命令 | `python3 python/scripts/collect_sysid.py --dry-run --segments=6 --duration-sec=2 --record-hold --out=<out>` |
| 段数 | **11**（6 激励：3 大 yaw + 3 小 yaw；5 静止保持段 `_hold`，首段不记） |
| 点数 | 200 点（2 s）激励段 + 934~1394 点（9~14 s）保持段；100 Hz |
| 被控对象参数 | `Jbig_eff=0.050, Js=0.020, Px=0.00866, Py=0.005, fc_big=0.220, fv_big=0.055, fc_small=0.0973, fv_small=0.028`（λ=100，积分步长 0.05 ms） |
| 背隙参数（真值） | `δ=0.0873 rad (5.0°), k=200 N·m/rad, c=2.0, γ=0.002, J_motor=0.006, fcMotor=0.030, fvMotor=0.010, β=0` |
| 几何 | `dx=0, dy=0.07`（仓库默认实测几何） |
| 底盘 | 静止（采集规范要求；链路语义仍按"与大 yaw 同包、有延迟+值保持"记录） |

## 2. 拟合（`fit/`）

| 文件 | 内容 |
|---|---|
| `fit_cad200.log` | 从 **CAD 占位初值**出发、`--epochs=200 --substeps=2 --threads=2`（391 s）的完整输出 |
| `fit_truth.log` | 从**真值**出发、默认配方 `--epochs=1000`（本机约 2 h ⇒ **日志记到 epoch 200 就停了**，后面的 epoch 没有跑；耗时 ~11 min/200ep） |
| `ident_convergence.png` / `ident_traj.png` | 用**真值参数**（epochs=0）画的收敛/轨迹对比图（4×5 面板 = loss + 16 参；轨迹图三通道） |

复现（几何用仓库默认，不需要 `--dx/--dy`）：

```bash
# ① 重建数据（约 4 min）
python3 python/scripts/collect_sysid.py --dry-run --segments=6 --duration-sec=2 \
        --record-hold --out=/tmp/bs_data
# ② 从真值出发（验证"真值是不动点"；想跑满 1000 epoch 约 2 h）
TRUTH=0.050,0.020,0.00866,0.005,0.220,0.0550,0.0973,0.0280,0.0873,200,2.0,0.002,0.006,0.030,0.010,0.0
python3 python/scripts/identify_params_torch.py --data='/tmp/bs_data/*.csv' \
        --epochs=1000 --init-vector=$TRUTH --truth-params=$TRUTH --threads=1
# ③ 只用窗口 RMSE 复盘（不含优化）
python3 python/scripts/identify_params_torch.py --data='/tmp/bs_data/*.csv' \
        --epochs=0 --init-vector=$TRUTH
# ④ 闭环（模型 vs 被控对象）
./build/tcbs_mpc_param_eval --phi=$TRUTH_8 --phi2=$TRUTH_EXTRA_8
```

> 注：这批数据里有 5 个 `_hold` 保持段（800~1394 点）。辨识脚本**默认只取每个保持段的前 3 s**
> （`--hold-max-sec`，0 = 不截断）：后段几乎是静止（std 比前 3 s 小 10~800 倍），
> 而开头正是大角度阶跃。本文的数字是当年旧版（不截断）跑出来的。

## 3. 结论摘要（详见 docs/backlash_model.md §7）

1. 辨识脚本的 numpy/torch 3-DOF 模型与 C++ `eomBacklash` 在"同输入同初值"下一致到
   **7e-4 rad / 3.9e-3 rad/s**（RK4/4 子步）；
2. **准确参数**闭环：RMS = 0.0277 / 0.1636 / 0.0066 rad（阶跃 / 大阶跃 / 正弦），失败数 0；
3. **拟合参数**（真值初值 200 epoch）闭环：0.0269 / 0.1631 / 0.0066 —— 与准确参数同量级；
4. **从 CAD 占位初值**跑 200 epoch：背隙 8 参基本到位（δ −1.2%、k ±0.02%、c +23%、
   J_motor +24%），平面参数只走了一半 ⇒ 闭环 0.0373 / 0.1699 / 0.0131（仍可用）；
5. 该数据集里 **`Px/Py` 不可辨识**（无倾角 + 分轴激励 ⇒ 两轴不同时动 ⇒ 耦合项为 0），
   会被噪声带走 ⇒ 拟合时建议 `--freeze-params=2,3`。

> ⚠ 这批数据是用**修好 `PlanarYawPlant._h()` 之后**的代码生成的。修之前它把"电机角速度"
> 当成了"云台角速度"（`tb, ts = qd[0], qd[1]` 应为 `qd[1], qd[2]`）⇒ 老批次的 dry-run 数据
> 不能用（`docs/backlash_model.md` §7.1 有记录）。
