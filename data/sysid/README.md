# data/sysid — 新的（实机）辨识与标定数据放这里

- 采集：`python3 python/scripts/collect_sysid.py --tag=big --segments=6`（分轴、录制序列+增强+PID、100 Hz）
- ★ **仿真环境 2**（只用于采集**测试数据**）：`--dry-run --sim-rigid` = 背隙接触面完全刚性 +
  死区完全自由 + β 每条数据随机/微弱漂移，见 `docs/backlash_model.md` §2.4
- 小 yaw 零点：跑 `./build/tcbs_test_serial`（力矩恒 0），人工摆到机械零点，读 `yaw_small_angle`，
  取负写进 `recv_small_yaw_offset`（`send_small_yaw_offset` 取相反符号）
- ★ **数据格式要求（3-DOF 背隙辨识）**: CSV/npz 必须同时含**电机侧** `theta_big_motor` 与
  **云台侧** `theta_big_platform` 两列（`collect_sysid.py` 现在会写全量列）；
  只有 12 列的老数据没有云台侧列 ⇒ 会被跳过并提示重采。
- 辨识（唯一路径）：`python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv' --epochs=1000`
  现在拟合 **16 参**（平面 8 参 + 背隙/电机侧 8 参）；δ 的独立校验用
  `python3 python/scripts/calibrate_backlash.py --data='data/sysid/*.csv'`（只做初值/交叉校验）。
  细节见 `docs/backlash_model.md`。
- 2026-09-15 那轮**仿真**验证数据已归档到 `data/archive/20260915_sim_sysid/`，不要混用
  （那批用的是**旧占位几何 `(0.10, 0)`**，复现要显式 `--dx=0.1 --dy=0`）。
- **几何改动后的复核**（`d = (0, 0.07)` 实测值）见 `data/archive/20260916_geom_0_007/`；
  实机采集/辨识直接用**仓库默认几何**即可，不需要给 `--dx/--dy`。
- 辨识结果写到 `--out=`（默认 `data/sysid/identified_params.txt`）；真车标定值按
  `data/cars/<车名>/` 归档。
