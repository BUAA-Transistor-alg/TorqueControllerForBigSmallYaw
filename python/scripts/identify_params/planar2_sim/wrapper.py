#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``planar2_sim`` 的 ctypes 包装 —— 手写 C++ **2-DOF** 前向仿真（分块向量化 + std::thread）。

    from identify_params.planar2_sim import FastSimulator, build, available
    sim = FastSimulator(base, dt=0.01, substeps=4, nthreads=0, block=32)
    theta, dtheta = sim.rollout(params_Bx11, seq_const, seq_var)

张量契约（与 :func:`~identify_params.planar2.rollout_np` 的 A 系重力路径对齐）::

    params     [B,11]  物理参数（顺序 = PARAM_NAMES2，**非 log/raw 空间**）
    seq_const  [B,5]   q0_b q0_s qd0_b qd0_s base_omega
    seq_var    [B,T,4] tau_b tau_s g_ax g_ay   ← **A 系**（随 b 转）逐样本重力分量
    返回       (θ [B,T,2], θ̇ [B,T,2])（第 t 行 = **积分前**状态）

★ 编译期常量（**不是**参数）: ``D = (0.0, 0.07)``、``lambda = 100.0``。
  构造时传入的 ``base`` 只用于兼容旧签名；若它带了 ``dx``/``dy``/``friction_lambda``
  且与编译期常量不一致，会直接报错（而不是静默算错物理）。
★ 与 :mod:`identify_params.planar2` 的 numpy 参考一致到 ~1e-13（残差只来自
  libm 与 numpy 的 sin/cos/tanh 实现差异）。同一组输入下不同 ``nthreads``/``block``
  的结果**逐位一致**（并行只按 batch 切连续区间，没有跨条目归约）。
★ 只支持 ``integrator="rk4"``。
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "planar2_sim.cpp")
_BUILD_SH = os.path.join(_HERE, "build.sh")
_LIB = os.path.join(_HERE, "_build", "libplanar2_sim.so")

_F64 = ctypes.c_double
_I32 = ctypes.c_int

# 与 planar2_sim.cpp / planar2.py 的已知常量严格一致（编译进 .so，不可运行时改）
DX = 0.0
DY = 0.07
FRICTION_LAMBDA = 100.0
NPARAM = 11
NCONST = 5
NVAR = 4
MAX_BLOCK = 256
DEFAULT_BLOCK = 32


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
        print(f"[planar2_sim] 编译 {os.path.relpath(_SRC)} → {os.path.relpath(_LIB)}")
    r = subprocess.run(["bash", _BUILD_SH], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(_LIB):
        raise RuntimeError(
            "planar2_sim 编译失败（需要 g++ / C++17）\n"
            f"命令: bash {_BUILD_SH}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    return _LIB


def available() -> bool:
    """共享库是否已存在且不比源码旧（不触发编译）。"""
    return os.path.isfile(_LIB) and not _lib_needs_build()


_LIB_HANDLE = None
_BINDINGS = None


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """给一个已加载的 .so 绑好本模块用到的函数签名。"""
    lib.planar2_rollout.restype = _I32
    lib.planar2_rollout.argtypes = [
        ctypes.POINTER(_F64),                      # params
        _I32,                                      # B
        ctypes.POINTER(_F64),                      # seq_const
        ctypes.POINTER(_F64),                      # seq_var
        _I32,                                      # T
        _F64, _I32, _I32, _I32,                    # dt substeps nthreads block
        ctypes.POINTER(_F64), ctypes.POINTER(_F64),  # theta dtheta
    ]
    lib.planar2_accel.restype = _I32
    lib.planar2_accel.argtypes = [
        ctypes.POINTER(_F64),                      # params
        _I32,                                      # B
        ctypes.POINTER(_F64),                      # state [B,4]
        ctypes.POINTER(_F64),                      # u     [B,4]
        ctypes.POINTER(_F64),                      # base_omega [B]
        ctypes.POINTER(_F64),                      # qdd   [B,2]
    ]
    lib.planar2_build_info.restype = ctypes.c_char_p
    lib.planar2_hardware_threads.restype = _I32
    return lib


def _load():
    """加载 .so 并绑定 C 函数（默认缺库**或源码更新**就自动编译；
    ``PLANAR2_SIM_NO_AUTOBUILD=1`` 可关）。"""
    global _LIB_HANDLE, _BINDINGS
    if _BINDINGS is not None:
        return _BINDINGS
    if _lib_needs_build():
        if not os.path.isfile(_LIB) and os.environ.get("PLANAR2_SIM_NO_AUTOBUILD", "0") == "1":
            raise RuntimeError(
                f"planar2_sim 共享库不存在: {_LIB}\n先跑: bash {_BUILD_SH}")
        build()
    lib = _bind(ctypes.CDLL(_LIB))
    _LIB_HANDLE, _BINDINGS = lib, (lib, lib.planar2_build_info, lib.planar2_hardware_threads)
    return _BINDINGS


def build_info() -> str:
    _, info, _ = _load()
    return info().decode("utf-8", "replace")


def hardware_threads() -> int:
    _, _, hw = _load()
    return int(hw())


def _as_f64(arr, shape, name: str) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} 形状应为 {shape}，得到 {a.shape}")
    return a


def _check_base(base) -> None:
    """``base`` 只为兼容签名；一旦它声明了与编译期常量不同的 dx/dy/lambda 就报错。"""
    if base is None:
        return
    for attr, want, label in (("dx", DX, "dx"), ("dy", DY, "dy"),
                              ("friction_lambda", FRICTION_LAMBDA, "friction_lambda")):
        got = getattr(base, attr, None)
        if got is None:
            continue
        if float(got) != float(want):
            raise ValueError(
                f"planar2_sim 把 {label}={want:g} 编译进了 .so（D=(0,0.07), lambda=100），"
                f"但 base.{attr}={float(got):g}；本内核不支持运行时改这些常量")


def accel(params, state, u, base_omega, lib_path: str | None = None) -> np.ndarray:
    """逐点加速度 ``q̈ = accel_np``（batch，不积分）—— 自检对拍用。

    ``params [B,11]``、``state [B,4] = (q_b,q_s,v_b,v_s)``、
    ``u [B,4] = (tau_b,tau_s,g_ax,g_ay)``（A 系重力）、``base_omega [B]``。
    返回 ``qdd [B,2]``。
    """
    st = np.ascontiguousarray(state, dtype=np.float64)
    if st.ndim != 2 or st.shape[1] != 4:
        raise ValueError(f"state 形状应为 [B,4]，得到 {st.shape}")
    B = int(st.shape[0])
    p = _as_f64(params, (B, NPARAM), "params")
    uu = _as_f64(u, (B, NVAR), "u")
    wc = np.ascontiguousarray(base_omega, dtype=np.float64).reshape(-1)
    if wc.shape != (B,):
        raise ValueError(f"base_omega 形状应为 [{B}]，得到 {wc.shape}")
    out = np.empty((B, 2), dtype=np.float64)
    lib = ctypes.CDLL(lib_path) if lib_path else _load()[0]
    if lib_path:
        _bind(lib)
    rc = lib.planar2_accel(
        p.ctypes.data_as(ctypes.POINTER(_F64)), B,
        st.ctypes.data_as(ctypes.POINTER(_F64)),
        uu.ctypes.data_as(ctypes.POINTER(_F64)),
        wc.ctypes.data_as(ctypes.POINTER(_F64)),
        out.ctypes.data_as(ctypes.POINTER(_F64)))
    if rc != 0:
        raise RuntimeError(f"planar2_accel 返回 {rc}（B≤0 或空指针）")
    return out


class FastSimulator:
    """手写 C++ **2-DOF** 前向仿真的 Python 门面（分块向量化 + std::thread 多线程）。

    构造: ``FastSimulator(base, dt, substeps=4, integrator="rk4", nthreads=0, block=32)``
      · ``base``      : 只为兼容旧签名（:class:`~identify_params.planar2.Planar2Params`）；
        提供 ``dx/dy/friction_lambda`` 时会与编译期常量做一致性检查；
      · ``nthreads``  : ≤0 ⇒ 用 ``std::thread::hardware_concurrency()``；
      · ``block``     : 分块向量化的块宽（默认 32，上限 256）。
    与 torch 版的区别: 不持有可学习参数（每次 ``rollout`` 传入），且**不做梯度**。
    """

    def __init__(self, base=None, dt: float = 0.01, substeps: int = 4,
                 integrator: str = "rk4", nthreads: int = 0, block: int = DEFAULT_BLOCK,
                 lib_path: str | None = None):
        name = str(integrator).lower()
        if name not in ("rk4",):
            raise ValueError(f"planar2_sim 只实现了 rk4（收到 {integrator!r}）；"
                             f"euler 消融请用 numpy 参考实现")
        _check_base(base)
        self.base = base
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError(f"dt 必须 > 0，得到 {self.dt}")
        self.substeps = max(1, int(substeps))
        self.integrator = "rk4"
        self.nthreads = int(nthreads)
        self.block = max(1, min(MAX_BLOCK, int(block)))
        if lib_path:                                  # 允许指定别的 .so（调试用）
            self._lib = _bind(ctypes.CDLL(lib_path))
        else:
            self._lib = _load()[0]
        self.n_eval = 0

    # ── 主入口 ──
    def rollout(self, params, seq_const, seq_var):
        """一次批量前向：``params [B,11]``、``seq_const [B,5]``、``seq_var [B,T,4]``。

        返回 ``(theta [B,T,2], dtheta [B,T,2])``（numpy float64）。
        """
        seq_var = np.ascontiguousarray(seq_var, dtype=np.float64)
        if seq_var.ndim != 3 or seq_var.shape[2] != NVAR:
            raise ValueError(f"seq_var 形状应为 [B,T,{NVAR}]，得到 {seq_var.shape}")
        B, T = int(seq_var.shape[0]), int(seq_var.shape[1])
        p = _as_f64(params, (B, NPARAM), "params")
        sc = _as_f64(seq_const, (B, NCONST), "seq_const")
        th = np.empty((B, T, 2), dtype=np.float64)
        dth = np.empty((B, T, 2), dtype=np.float64)
        rc = self._lib.planar2_rollout(
            p.ctypes.data_as(ctypes.POINTER(_F64)), B,
            sc.ctypes.data_as(ctypes.POINTER(_F64)),
            seq_var.ctypes.data_as(ctypes.POINTER(_F64)),
            T, self.dt, self.substeps, self.nthreads, self.block,
            th.ctypes.data_as(ctypes.POINTER(_F64)),
            dth.ctypes.data_as(ctypes.POINTER(_F64)))
        if rc != 0:
            raise RuntimeError(f"planar2_rollout 返回 {rc}（参数非法: B/T/substeps/dt 或空指针）")
        self.n_eval += 1
        return th, dth

    # 与 numpy/torch 模型同样的调用姿势：sim(params, seq_const, seq_var)
    __call__ = rollout

    def extra_repr(self) -> str:
        return (f"dt={self.dt:g}, substeps={self.substeps}, nthreads={self.nthreads}, "
                f"block={self.block}")

    def __repr__(self) -> str:
        return f"FastSimulator({self.extra_repr()})"
