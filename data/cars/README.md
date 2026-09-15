# data/cars — 每台车的标定结果（沿用原仓库 TorqueController 的目录约定）

每标定一台新云台/整车，就在本目录下建一个车名文件夹（例如 `Infantry3`），把**该车的**标定结果放进去：

```
data/cars/<车名>/
  LinearParams.txt                   # McuDataPreprocessor::LinearParams 的全部映射参数（scale/offset/torque_scale）
  params/Identified_parameters.txt   # 8 个动力学参数（含 λ、实测几何 dx/dy、各参数 σ）
  params/Figure_1.png                # 收敛曲线 / 留出段验证图（torch 的 ident_torch_*_convergence.png 可改名放这）
  sysid_samples/*.npz                # 该车采集的原始样本（collect_sysid.py 的输出）
  notes.md（可选）                   # 标定日期、温度、倾角、人工零点用的基准/工装、异常记录
```

规则：

1. **一车一目录**，不要覆盖别的车；文件名保持上面这套（脚本与文档都按这些名字找）。
2. `data/sysid/` 是**工作区**（正在采集/分析的临时数据），默认不入库；整理好后把该车的样本
   挪到 `data/cars/<车名>/sysid_samples/` 再提交。
3. 本轮**仿真**验证数据不在 `cars/` 下 —— 在 `data/archive/20260915_sim_sysid/`，
   因为那是"验证方法"用的、带仿真真值的数据，不能当作实车参数。
4. 标什么、怎么标、精度要求：见根 `README.md` §0 清单与 `docs/calibration.md`。

（本文件只为说明约定，`data/cars/` 下暂无实车数据。）
