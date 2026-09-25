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
  · :mod:`~identify_params.loss`     —— **损失（只有这一处定义）**:
       ``pair_loss`` / ``pointwise_loss`` / ``weighted_reduce`` / ``WindowObjective``
       （前向 + 损失一步到位；给无梯度优化器用，内部 ``no_grad``）。
  · :mod:`~identify_params.train`    —— ③ **训练逻辑（Adam）**:
       ``FitConfig`` / ``FitResult`` / ``build_fit_context``（与 CMA-ES 共用的准备逻辑）
       / 输出误差法主循环 / 学习率 / 评测 RMSE。
  · :mod:`~identify_params.cmaes_fit` —— **CMA-ES 优化器**（无梯度；前向可选
       ``cpp``/``numpy``/``torch``，见 ``python3 -m identify_params.cmaes_fit --help``）。
  · :mod:`~identify_params.manual_tune` —— **手动标定 GUI**（PyQt5：左 3×3 实测 vs 仿真曲线，
       右 文件翻页 + 参数滑块；正数对数调节）。跑: ``python3 -m identify_params.manual_tune``。
  · :mod:`~identify_params.planar2_sim` —— **手写 C++ 快速前向**（分块向量化 + ``std::thread``
       + ctypes；独立子目录，不参与主工程构建；**按需编译**，所以不在包导入时加载）。
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
    NCONST,
    NVAR,
    SEQ_CONST_NAMES,
    SEQ_VAR_NAMES,
    DifferentiableSimulator,
    accel_np,
    derived_np,
    gravity_np,
    rk4_step_np,
    rollout_batched_np,
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
    TAU_SIGN_BIG_DEFAULT,
    TAU_SIGN_SMALL_DEFAULT,
    Exo,
    ParamGroups,
    ParamSpec,
    PlanarParams,
    apply_tau_sign,
    default_param_vector,
    reset_tau_sign,
    resolve_fixed_names,
    set_tau_sign,
    set_tau_sign_big,
    set_tau_sign_small,
    tau_sign,
    tau_sign_big,
    tau_sign_desc,
    tau_sign_small,
)
from .loss import (  # noqa: F401
    WindowObjective,
    pair_loss,
    pointwise_loss,
    rollout_states,
    sequence_loss,
    weighted_reduce,
)
from .train import (  # noqa: F401
    FitConfig,
    FitContext,
    FitResult,
    build_fit_context,
    channel_rmse,
    fit_params_torch,
    format_header_snippet,
    format_param_table,
)

__all__ = [
    # ① 参数配置（2-DOF 缩合参数）
    "ParamSpec", "PARAM_SPECS", "PARAM_NAMES", "NPARAM",
    "REAL_PARAM_NAMES", "POSITIVE_PARAM_NAMES",
    "PlanarParams", "Exo", "ParamGroups", "resolve_fixed_names", "default_param_vector",
    # ★ 控制力矩符号（两路独立；只作用于辨识环境）
    "TAU_SIGN_BIG_DEFAULT", "TAU_SIGN_SMALL_DEFAULT", "apply_tau_sign",
    "set_tau_sign", "set_tau_sign_big", "set_tau_sign_small",
    "tau_sign", "tau_sign_big", "tau_sign_small", "tau_sign_desc", "reset_tau_sign",
    "AXIS_BIG", "AXIS_SMALL",
    # ② 可微仿真模型（numpy + torch）
    "DifferentiableSimulator", "rollout_batched_np", "simulate_np",
    "accel_np", "derived_np", "gravity_np", "rk4_step_np",
    "NCONST", "NVAR", "SEQ_CONST_NAMES", "SEQ_VAR_NAMES",
    # ③ 损失（唯一定义处）与训练逻辑
    "pair_loss", "pointwise_loss", "weighted_reduce", "rollout_states", "sequence_loss",
    "WindowObjective",
    "FitConfig", "FitResult", "FitContext", "build_fit_context",
    "fit_params_torch", "channel_rmse", "format_param_table", "format_header_snippet",
    # 数据
    "Segment", "load_segments", "segment_from_arrays", "truncate_hold_segments",
    "state_arrays",
    # 连续化 / 角度处理
    "wrap_to_pi", "continuous_angles", "align_beta", "check_beta_continuity",
    "ask_skip_segment",
]

