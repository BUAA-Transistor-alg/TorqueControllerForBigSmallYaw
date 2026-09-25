# -*- coding: utf-8 -*-
"""``planar2_sim`` —— 手写 C++ **2-DOF**（大/小 yaw，A 系重力）高效率前向仿真。

★ **完全独立于主工程**: 本目录不被 `CMakeLists.txt` 引用、不 include 任何项目头文件，
  只用 C++17 标准库；编译由 :func:`build`（= 本目录的 ``build.sh``）完成。
  数学 = :mod:`identify_params.planar2` 的 numpy 参考实现（``gravity_from_Aframe`` 写法）。

用法::

    from identify_params.planar2_sim import FastSimulator, build, build_info, available
    build()                                    # 或让它首次使用时自动编译
    sim = FastSimulator(base, dt=0.01, substeps=4, nthreads=0, block=32)
    theta, dtheta = sim.rollout(params_Bx11, seq_const, seq_var)

数值上与 :func:`~identify_params.planar2.rollout_np` 一致到 ~1e-13，且不同线程数/分块
宽度**逐位一致**。自检::

    cd python/scripts && python3 -m identify_params.planar2_sim.selftest
    # 或在仓库根目录: PYTHONPATH=python/scripts python3 -m identify_params.planar2_sim.selftest

只支持 rk4。已知常量编译进 .so: ``D=(0,0.07)``、``lambda=100``。
"""

from .wrapper import (  # noqa: F401
    DEFAULT_BLOCK,
    DX,
    DY,
    FRICTION_LAMBDA,
    MAX_BLOCK,
    NPARAM,
    FastSimulator,
    accel,
    available,
    build,
    build_info,
    hardware_threads,
)

__all__ = [
    "FastSimulator", "accel", "build", "build_info", "available", "hardware_threads",
    "DX", "DY", "FRICTION_LAMBDA", "NPARAM", "MAX_BLOCK", "DEFAULT_BLOCK",
]
