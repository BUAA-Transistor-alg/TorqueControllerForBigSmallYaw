# -*- coding: utf-8 -*-
"""``fast_sim`` —— 手写 C++ 高效率前向仿真（分块向量化 + std::thread + ctypes）。

★ **完全独立于主工程**: 本目录不被 `CMakeLists.txt` 引用、不 include 任何项目头文件，
  只用 C++17 标准库；编译由 :func:`build`（= 本目录的 ``build.sh``）完成。

用法::

    from identify_params.fast_sim import FastSimulator, build, build_info, available
    build()                                    # 或让它首次使用时自动编译
    sim = FastSimulator(base, dt=0.01, substeps=4, nthreads=0, block=64)
    theta, dtheta = sim.rollout(params18, seq_const, seq_var)

数值上与 :func:`~identify_params.model.simulate_backlash_np` 一致到机器精度
（实测 θ 多为**完全相同**、角速度 ≤2.5e-15；损失与 numpy 后端相同），且不同线程数/分块
宽度**逐位一致**。自检::

    python3 -m identify_params.fast_sim.selftest

只支持 rk4；euler 消融请用 torch/numpy 版。
"""

from .wrapper import (  # noqa: F401
    FastSimulator,
    available,
    build,
    build_info,
    hardware_threads,
)

__all__ = ["FastSimulator", "build", "build_info", "available", "hardware_threads"]
