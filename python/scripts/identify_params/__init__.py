# -*- coding: utf-8 -*-
"""identify_params —— 平面 3-DOF（含大 yaw 背隙）模型的 PyTorch 可导前向仿真参数辨识。

由已删除的 ``python/scripts/identify_params_torch.py`` 拆分而来（该单文件版本已不存在），
按职责分层:

  · :mod:`~identify_params.params`   —— ① **默认参数配置**（**辨识初值的唯一来源**）:
       参数表（名字/单位/**可取值范围**/**初值**）、模型参数容器 ``PlanarParams``、
       外生量 ``Exo``，以及把可学习参数分成 **全体实数 / 正数** 两组的 ``ParamGroups``。
       被冻结的参数在这里就变成**固定参数**，不进任何一组。
  · :mod:`~identify_params.model`    —— ② **可微仿真模型**:
       ``DifferentiableSimulator``（**不含任何可学习参数**，RK4 可导积分，返回每个时间点的
       位置/速度）+ numpy 参考实现（前向验证 / 回归矩阵 / 评测）。
  · :mod:`~identify_params.train`    —— ③ **训练逻辑**:
       ``FitConfig`` / ``FitResult`` / 输出误差法主循环 / 损失 / 学习率 / 评测 RMSE。
  · :mod:`~identify_params.data`     —— 数据段加载（CSV / npz）、**连续化 + β 圈数对齐**。
  · :mod:`~identify_params.plotting` —— 收敛曲线 / 轨迹对比 / 学习曲线。
  · :mod:`~identify_params.selftest` —— 模型一致性自检。
  · :mod:`~identify_params.cli`      —— 命令行入口（``python -m identify_params``）。
"""

from .data import (  # noqa: F401
    Segment,
    align_beta,
    ask_skip_segment,
    check_beta_continuity,
    continuous_angles,
    load_segments,
    segment_from_arrays,
    state_arrays,
    truncate_hold_segments,
    wrap_to_pi,
)
from .model import (  # noqa: F401
    DifferentiableSimulator,
    eom_backlash_np,
    forward_accel_backlash_np,
    inverse_dynamics_backlash_np,
    regressor_np,
    simulate_backlash_np,
    simulate_np,
)
from .params import (  # noqa: F401
    AXIS_BIG,
    AXIS_SMALL,
    NPARAM,
    PARAM_NAMES,
    PARAM_SPECS,
    POSITIVE_PARAM_NAMES,
    REAL_PARAM_NAMES,
    Exo,
    ParamGroups,
    ParamSpec,
    PlanarParams,
    default_param_vector,
    resolve_fixed_names,
)
from .train import FitConfig, FitResult, channel_rmse, fit_params_torch  # noqa: F401

__all__ = [
    # ① 默认参数配置
    "ParamSpec", "PARAM_SPECS", "PARAM_NAMES", "NPARAM",
    "REAL_PARAM_NAMES", "POSITIVE_PARAM_NAMES",
    "PlanarParams", "Exo", "ParamGroups", "resolve_fixed_names", "default_param_vector",
    "AXIS_BIG", "AXIS_SMALL",
    # ② 可微仿真模型
    "DifferentiableSimulator", "simulate_backlash_np", "simulate_np",
    "eom_backlash_np", "forward_accel_backlash_np", "inverse_dynamics_backlash_np",
    "regressor_np",
    # ③ 训练逻辑
    "FitConfig", "FitResult", "fit_params_torch", "channel_rmse",
    # 数据
    "Segment", "load_segments", "segment_from_arrays", "truncate_hold_segments",
    "state_arrays",
    # 连续化 / β 圈数对齐
    "wrap_to_pi", "continuous_angles", "align_beta", "check_beta_continuity",
    "ask_skip_segment",
]

