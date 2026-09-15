# data/sysid — 新的（实机）辨识与标定数据放这里

- 采集：`python3 python/scripts/collect_sysid.py --tag=big --segments=6`（分轴、录制序列+增强+PID、100 Hz）
- 小 yaw 零点标定：`python3 python/scripts/calibrate_small_zero.py --method=manual --repeats=3`
- 辨识：`./build/tcbs_identify_params data/sysid/*.csv --held=measured --dx=<实测> --lambda=100`
  与 `python3 python/scripts/identify_params_torch.py --data='data/sysid/*.csv' --dx=<实测>`
- 2026-09-15 那轮**仿真**验证数据已归档到 `data/archive/20260915_sim_sysid/`，不要混用。
