#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``fast_sim`` 的 ctypes 包装 —— 手写 C++ 前向仿真（分块向量化 + std::thread）。

    from identify_params.fast_sim import FastSimulator, build, available
    sim = FastSimulator(base, dt=0.01, substeps=4, nthreads=0, block=64)
    theta, dtheta = sim.rollout(params18, seq_const, seq_var)   # 都是 [B,...] numpy

接口与 :class:`~identify_params.model.DifferentiableSimulator` **同一张量契约**:
    params18   [B,18]  物理参数（顺序 = PARAM_NAMES；[15]=β 忽略，β 由 seq_var 逐点给）
    seq_const  [B,8]   q0(3) + qd0(3) + base_omega + base_alpha
    seq_var    [B,T,5] tau_big, tau_small, grav_x, grav_y, β
    返回       (θ [B,T,3], θ̇ [B,T,3])

★ 数值上与 :func:`~identify_params.model.simulate_backlash_np`（numpy 参考实现）
  一致到 ~1e-14（残差只来自 tanh 的实现差异）。同一组输入下不同 ``nthreads``/``block``
  的结果**逐位一致**（并行只按 batch 切分，没有跨条目归约）。
★ 只支持 ``integrator="rk4"``（C++ 里就写了 RK4；euler 消融用 numpy/torch 版）。
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

from ..params import PlanarParams

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "fast_sim.cpp")
_BUILD_SH = os.path.join(_HERE, "build.sh")
_LIB = os.path.join(_HERE, "_build", "libfast_sim.so")

_F64 = ctypes.c_double
_I32 = ctypes.c_int


def _lib_needs_build() -> bool:
    if not os.path.isfile(_LIB):
        return True
    so_t = os.path.getmtime(_LIB)
    # 源码或编译脚本任一更新都要重编
    return so_t < max(os.path.getmtime(_SRC), os.path.getmtime(_BUILD_SH))


def build(force: bool = False, quiet: bool = False) -> str:
    """编译共享库（**独立于主工程**：只调 build.sh，不用 CMake）。返回 .so 路径。"""
    if not force and not _lib_needs_build():
        return _LIB
    if not os.path.isfile(_BUILD_SH):
        raise RuntimeError(f"找不到 {_BUILD_SH}")
    if not quiet:
        print(f"[fast_sim] 编译 {os.path.relpath(_SRC)} → {os.path.relpath(_LIB)}")
    r = subprocess.run(["bash", _BUILD_SH], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(_LIB):
        raise RuntimeError(
            "fast_sim 编译失败（需要 g++ / C++17）\n"
            f"命令: bash {_BUILD_SH}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    return _LIB


def available() -> bool:
    """共享库是否已存在且不比源码旧（不触发编译）。"""
    return os.path.isfile(_LIB) and not _lib_needs_build()


_LIB_HANDLE = None
_BINDINGS = None


def _load():
    """加载 .so 并绑定 C 函数（默认缺库就自动编译；``FAST_SIM_NO_AUTOBUILD=1`` 可关）。"""
    global _LIB_HANDLE, _BINDINGS
    if _BINDINGS is not None:
        return _BINDINGS
    if not os.path.isfile(_LIB):
        if os.environ.get("FAST_SIM_NO_AUTOBUILD", "0") == "1":
            raise RuntimeError(
                f"fast_sim 共享库不存在: {_LIB}\n先跑: bash {_BUILD_SH}")
        build()
    lib = ctypes.CDLL(_LIB)
    lib.fast_sim_rollout.restype = _I32
    lib.fast_sim_rollout.argtypes = [
        _I32, _I32, _I32, _F64,                       # B T substeps dt
        ctypes.POINTER(_F64), ctypes.POINTER(_F64), ctypes.POINTER(_F64),   # params/seq_const/seq_var
        _F64, _F64, _F64, _F64, _F64,                 # dx dy m_u_known lambda eps
        _F64, _F64, _F64,                             # off_big off_small off_motor
        ctypes.POINTER(_F64), ctypes.POINTER(_F64),   # theta_out dtheta_out
        _I32, _I32,                                   # nthreads block
    ]
    lib.fast_sim_build_info.restype = ctypes.c_char_p
    lib.fast_sim_hardware_threads.restype = _I32
    _LIB_HANDLE, _BINDINGS = lib, (lib, lib.fast_sim_build_info, lib.fast_sim_hardware_threads)
    return _BINDINGS


def build_info() -> str:
    lib, info, _ = _load()
    return info().decode("utf-8", "replace")


def hardware_threads() -> int:
    lib, _, hw = _load()
    return int(hw())


def _as_f64(arr, shape, name: str) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} 形状应为 {shape}，得到 {a.shape}")
    return a


class FastSimulator:
    """手写 C++ 前向仿真的 Python 门面（分块向量化 + std::thread 多线程）。

    构造: ``FastSimulator(base, dt, substeps=4, integrator="rk4", nthreads=0, block=64)``
      · ``base``      : :class:`~identify_params.params.PlanarParams`（提供**固定参数**:
        dx/dy/m_u_known/friction_lambda/backlash_smooth_eps/tau_offset_*）；
      · ``nthreads``  : ≤0 ⇒ 用 ``std::thread::hardware_concurrency()``；
      · ``block``     : 分块向量化的块宽（默认 64，上限 256）。
    与 torch 版的区别: 不持有可学习参数（每次 ``rollout`` 传入），且**不做梯度**。
    """

    def __init__(self, base: PlanarParams, dt: float, substeps: int = 4,
                 integrator: str = "rk4", nthreads: int = 0, block: int = 64,
                 lib_path: str | None = None):
        name = str(integrator).lower()
        if name not in ("rk4",):
            raise ValueError(f"fast_sim 只实现了 rk4（收到 {integrator!r}）；"
                             f"euler 消融请用 model.DifferentiableSimulator / simulate_backlash_np")
        self.base = base
        self.dt = float(dt)
        self.substeps = max(1, int(substeps))
        self.integrator = "rk4"
        self.nthreads = int(nthreads)
        self.block = max(1, min(256, int(block)))
        if lib_path:                                  # 允许指定别的 .so（调试用）
            lib = ctypes.CDLL(lib_path)
            lib.fast_sim_rollout.restype = _I32
            self._lib = lib
        else:
            self._lib, _, _ = _load()
        self.n_eval = 0

    # ── 主入口 ──
    def rollout(self, params, seq_const, seq_var):
        """一次批量前向：``params [B,18]``、``seq_const [B,8]``、``seq_var [B,T,5]``。

        返回 ``(theta [B,T,3], dtheta [B,T,3])``（numpy float64）。
        """
        seq_var = np.ascontiguousarray(seq_var, dtype=np.float64)
        if seq_var.ndim != 3 or seq_var.shape[2] != 5:
            raise ValueError(f"seq_var 形状应为 [B,T,5]，得到 {seq_var.shape}")
        B, T = int(seq_var.shape[0]), int(seq_var.shape[1])
        p = _as_f64(params, (B, 18), "params")
        sc = _as_f64(seq_const, (B, 8), "seq_const")
        th = np.empty((B, T, 3), dtype=np.float64)
        dth = np.empty((B, T, 3), dtype=np.float64)
        rc = self._lib.fast_sim_rollout(
            B, T, self.substeps, self.dt,
            p.ctypes.data_as(ctypes.POINTER(_F64)),
            sc.ctypes.data_as(ctypes.POINTER(_F64)),
            seq_var.ctypes.data_as(ctypes.POINTER(_F64)),
            float(self.base.dx), float(self.base.dy), float(self.base.m_u_known),
            float(self.base.friction_lambda), float(self.base.backlash_smooth_eps),
            float(self.base.tau_offset_big), float(self.base.tau_offset_small),
            float(self.base.tau_offset_motor),
            th.ctypes.data_as(ctypes.POINTER(_F64)),
            dth.ctypes.data_as(ctypes.POINTER(_F64)),
            self.nthreads, self.block)
        if rc != 0:
            raise RuntimeError(f"fast_sim_rollout 返回 {rc}（参数非法: B/T/substeps/dt 或空指针）")
        self.n_eval += 1
        return th, dth

    # 与 torch 模型同样的调用姿势：sim(params, seq_const, seq_var)
    __call__ = rollout

    def extra_repr(self) -> str:
        return (f"dt={self.dt:g}, substeps={self.substeps}, nthreads={self.nthreads}, "
                f"block={self.block}")
