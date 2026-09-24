#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 默认参数配置 —— 参数表 + 两组可学习参数 + 模型参数容器。

★ **辨识初值只在这里定义一次**：每个参数的初值 = `ParamSpec.default`（见 `PARAM_SPECS`）。
  `PlanarParams` 的 18 个参数字段默认值与 `default_param_vector()` 都由该表派生；
  训练入口直接调用 `default_param_vector()`；CLI 只提供 `--init-vector`（整体替换）。

可学习参数按**可取值范围**分成两组（★ 用户约定，不再按"核心 8 参 / 背隙 8 参 / Pb 2 参"
那种物理功能分组）:

  · ``real``     —— **全体实数**: φ = raw（直接自由，可正可负）
  · ``positive`` —— **正数**:     φ = exp(raw)（正性由参数化隐式保证，**无上下界**）

固定参数（几何 dx/dy、摩擦软符号陡度 λ、平滑死区 ε、力矩偏置、以及**被冻结的辨识参数**）
**不属于任何一组**，其取值留在 :class:`PlanarParams` 里，通过 :class:`ParamGroups` 的
``base`` 带进模型。可微仿真模型本身**不持有**任何可学习参数，见 ``model.py``。

参数顺序与 ``include/tcbs/mpc/planar_yaw_model.h`` / ``planar_yaw_params.h`` 保持一致:

    0 Jbig_eff  1 Js        2 Px      3 Py       4 fc_big   5 fv_big
    6 fc_small  7 fv_small
    8 backlash_delta(δ)     9 backlash_k     10 backlash_c    11 backlash_through(γ)
    12 Jmotor   13 fc_motor 14 fv_motor      15 backlash_beta(β)
    16 Pbx      17 Pby
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - 环境缺 torch 时给出明确提示
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# ============================================================================
# 常量
# ============================================================================
DT_DEFAULT = 0.01                      # 100 Hz
FRICTION_LAMBDA = 100.0                # ★ 辨识模型固定 λ = 100（不辨识）
ENCODER_CPR = 8192                     # 编码器计数/整圈
QUANT_STEP = 2.0 * math.pi / ENCODER_CPR   # ≈ 7.669e-4 rad
AXIS_BIG, AXIS_SMALL = 0, 1
# ── 状态通道 ↔ 被激励轴（3-DOF: q = 电机, 云台, 小 yaw）──
AXIS_CHANNELS = {AXIS_BIG: (0, 1), AXIS_SMALL: (2,)}

# ★ 无参数限位（与原仓库 `param_ident.py` 一致）：不存在任何上下界 / clamp / 投影。
#   · 正数（可取值范围 = (0, +∞)）用 **log 参数化** φ = exp(raw) ⇒ 正性隐式保证；
#   · 全体实数直接自由 φ = raw。
# 取值范围只由下面这张表的 `positive` 字段决定，冻结与否是另一件事（见 ParamGroups）。


# ============================================================================
# 参数表（默认参数配置）
# ============================================================================
@dataclass(frozen=True)
class ParamSpec:
    """一个辨识参数的名字 / 单位 / **可取值范围** / **初值** / 物理含义。"""

    name: str
    unit: str
    positive: bool          # True = 可取值范围是正数（log 参数化）；False = 全体实数
    default: float          # ★★ **辨识初值的唯一来源**（见文件头"初值只有一处"）
    note: str = ""


# ════════════════════════════════════════════════════════════════════════════
# ★★ 辨识初值（`default` 字段）在这里**只定义一次**，是全包唯一来源:
#      · `PlanarParams` 的 18 个参数字段默认值  → 由本表派生（`_spec_default`）
#      · `default_param_vector()`              → 由本表派生
#      · 训练入口 `fit_params_torch` 的初值     → `default_param_vector()`
#    要改初值就只改这张表；`--init-vector` 是"整体替换"的显式用户输入，不是另一份默认值。
# ════════════════════════════════════════════════════════════════════════════
PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("Jbig_eff", "kg·m²", True, 0.1, "大 yaw 等效惯量（含上装）"),
    ParamSpec("Js", "kg·m²", True, 0.1, "小 yaw 惯量"),
    ParamSpec("Px", "kg·m", False, 0.1, "上装一阶矩 x（可正可负）"),
    ParamSpec("Py", "kg·m", False, 0.1, "上装一阶矩 y（可正可负）"),
    ParamSpec("fc_big", "N·m", True, 0.1, "大 yaw 库仑摩擦"),
    ParamSpec("fv_big", "N·m·s/rad", True, 0.1, "大 yaw 粘滞摩擦"),
    ParamSpec("fc_small", "N·m", True, 0.1, "小 yaw 库仑摩擦"),
    ParamSpec("fv_small", "N·m·s/rad", True, 0.1, "小 yaw 粘滞摩擦"),
    ParamSpec("backlash_delta", "rad", True, 0.1, "δ: 背隙宽度"),
    ParamSpec("backlash_k", "N·m/rad", True, 200.0, "k: 接触刚度"),
    ParamSpec("backlash_c", "N·m·s/rad", True, 0.1, "c: 接触阻尼"),
    ParamSpec("backlash_through", "—", False, 0.002, "γ: 死区直通线性项（梯度引导，**默认固定**）"),
    ParamSpec("Jmotor", "kg·m²", True, 0.1, "电机侧等效惯量"),
    ParamSpec("fc_motor", "N·m", True, 0.1, "电机侧库仑摩擦"),
    ParamSpec("fv_motor", "N·m·s/rad", True, 0.1, "电机侧粘滞摩擦"),
    ParamSpec("backlash_beta", "rad", False, 0.0, "β: 死区中心偏置（可正可负；★ 必为数据列，不参与拟合）"),
    ParamSpec("Pbx", "kg·m", False, 0.1, "大 yaw 侧一阶矩 x（只随大 yaw 转的偏心）"),
    ParamSpec("Pby", "kg·m", False, 0.1, "大 yaw 侧一阶矩 y"),
)

PARAM_NAMES = tuple(s.name for s in PARAM_SPECS)
PARAM_UNITS = tuple(s.unit for s in PARAM_SPECS)
PARAM_INDEX = {s.name: i for i, s in enumerate(PARAM_SPECS)}
NPARAM = len(PARAM_SPECS)                       # 18


def spec_of(name: str) -> ParamSpec:
    """按名字取参数表条目（未知名字直接报错，避免静默拼错）。"""
    try:
        return PARAM_SPECS[PARAM_INDEX[name]]
    except KeyError as exc:
        raise KeyError(f"未知参数名 {name!r}；可选: {', '.join(PARAM_NAMES)}") from exc


def _spec_default(name: str) -> float:
    """参数表里的**初值**（唯一来源；供 `PlanarParams` 字段默认值派生）。"""
    return spec_of(name).default

# ── 兼容旧分组名（打印/文档/旧脚本对照用；**参数分组以可取值范围为准**）──
CORE_PARAM_NAMES = PARAM_NAMES[:8]
NCORE = len(CORE_PARAM_NAMES)
EXTRA_PARAM_NAMES = PARAM_NAMES[8:16]
PB_PARAM_NAMES = PARAM_NAMES[16:18]
NPB = len(PB_PARAM_NAMES)

# ── ★ 两组: 按可取值范围 ──
POSITIVE_PARAM_NAMES = tuple(s.name for s in PARAM_SPECS if s.positive)
REAL_PARAM_NAMES = tuple(s.name for s in PARAM_SPECS if not s.positive)
POSITIVE_PARAM = np.array([s.positive for s in PARAM_SPECS], dtype=bool)
POSITIVE_IDX = tuple(int(i) for i in np.nonzero(POSITIVE_PARAM)[0])
FREE_IDX = tuple(int(i) for i in np.nonzero(~POSITIVE_PARAM)[0])

# ── ★ 方向约束下 Px/Py 退化成的那个**派生实数标量**（见 ParamGroups(p_along_d=...)）──
P_ALONG_D_NAME = "P_along_d"

assert POSITIVE_PARAM.size == NPARAM


# ============================================================================
# 模型参数容器（18 个辨识参数 + 实测几何/固定量）
# ============================================================================
@dataclass
class PlanarParams:
    """平面 3-DOF（含大 yaw 背隙）模型参数 + 实测几何（几何量不参与辨识）。"""

    # ── 实测几何 / 固定量 ──
    # ★ 默认 = 本构型实测几何 (dx, dy) = (0, 0.07) m（与 C++ defaultModelParams()/ModelParams
    #   一致）: 两轴在 x（右）方向无偏置、小 yaw 轴在大 yaw 轴**前方** 0.07 m。
    #   d 只以耦合项进入模型 ⇒ d 错 k 倍, 辨识出的 |P| 就错 1/k 倍, 换机械务必改这里或 --dx/--dy。
    dx: float = 0.0
    dy: float = 0.07
    gravity: float = 9.81
    m_u_known: float = 0.0
    friction_lambda: float = FRICTION_LAMBDA
    tau_offset_big: float = 0.0
    tau_offset_small: float = 0.0
    # ── ★ 18 个辨识参数的字段默认值 = **参数表 PARAM_SPECS 的 `default`**（唯一来源）──
    # ── 8 个核心辨识参数（平面 2-DOF 子块）──
    Jbig_eff: float = _spec_default("Jbig_eff")
    Js: float = _spec_default("Js")
    Px: float = _spec_default("Px")
    Py: float = _spec_default("Py")
    fc_big: float = _spec_default("fc_big")
    fv_big: float = _spec_default("fv_big")
    fc_small: float = _spec_default("fc_small")
    fv_small: float = _spec_default("fv_small")
    # ── 8 个背隙/电机侧辨识参数 ──
    backlash_delta: float = _spec_default("backlash_delta")      # δ: 背隙宽度（rad）
    backlash_k: float = _spec_default("backlash_k")              # k: 接触刚度 (N·m/rad)
    backlash_c: float = _spec_default("backlash_c")              # c: 接触阻尼 (N·m·s/rad)
    backlash_through: float = _spec_default("backlash_through")  # γ: 死区直通项（**默认固定**）
    Jmotor: float = _spec_default("Jmotor")                      # 电机侧等效惯量 (kg·m²)
    fc_motor: float = _spec_default("fc_motor")                  # 电机侧库仑摩擦 (N·m)
    fv_motor: float = _spec_default("fv_motor")                  # 电机侧粘滞摩擦 (N·m·s/rad)
    backlash_beta: float = _spec_default("backlash_beta")        # β: 死区中心（Δ = θ_m−θ_p−β）
    # ── 2 个大 yaw 侧一阶矩（kg·m）: Pbx/Pby ──
    #   Gb = (Pbx + m_u_known·dx)·gy − (Pby + m_u_known·dy)·gx + Gs
    Pbx: float = _spec_default("Pbx")
    Pby: float = _spec_default("Pby")
    # ── 固定量（**不**辨识）──
    backlash_smooth_eps: float = 1.0e-4  # 平滑死区 ε（= C++ ModelParams 默认）
    tau_offset_motor: float = 0.0        # 电机侧力矩偏置（默认关）

    # ── 向量化接口（顺序与 PARAM_NAMES 一致）──
    def vector(self) -> np.ndarray:
        return np.array([getattr(self, nm) for nm in PARAM_NAMES], dtype=np.float64)

    def with_vector(self, phi) -> "PlanarParams":
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != (NPARAM,):
            raise ValueError(f"参数向量长度应为 {NPARAM}，得到 {phi.shape}")
        return replace(self, **{nm: float(phi[i]) for i, nm in enumerate(PARAM_NAMES)})

    def geometry_copy(self, **kw) -> "PlanarParams":
        """只改几何/固定量的副本（例如把 λ 换成 plant 的 100）。"""
        return replace(self, **kw)


def params_from_torch(phi_vec, base: PlanarParams) -> PlanarParams:
    """用长度 18 的参数向量构造 PlanarParams。

    参数可以是 numpy 数组**或 torch 张量**（后者保持可导 —— 训练路径就靠它）。
    ⚠ 这里必须把 **全部 18 个**字段都从 `phi_vec` 取: 漏掉哪个，那个参数在训练里就
      永远是 `base` 的**常量**、梯度恒 0（Pbx/Pby 曾经就这样被漏掉，靠自检抓到）。
    """
    return replace(base, **{nm: phi_vec[i] for i, nm in enumerate(PARAM_NAMES)})


def default_param_vector() -> np.ndarray:
    """辨识初值（长度 18，顺序 = ``PARAM_NAMES``）。

    ★ **由参数表 `PARAM_SPECS` 的 `default` 派生**——全包初值只有那一处定义；
    要改初值/换一组起点，只改 `PARAM_SPECS`，不要在这里或调用处另写一份字面量。
    """
    return np.array([s.default for s in PARAM_SPECS], dtype=np.float64)


def default_geometry() -> PlanarParams:
    """几何/固定量的默认值（18 个辨识参数取参数表 `PARAM_SPECS` 的初值）。"""
    return PlanarParams()


# ============================================================================
# 外生量
# ============================================================================
def _nonzero(v) -> bool:
    """判断重力分量是否非零（支持 float / ndarray / torch 张量）。"""
    if isinstance(v, np.ndarray):
        return bool(np.any(v != 0.0))
    if torch is not None and torch.is_tensor(v):
        with torch.no_grad():
            return bool(torch.any(v != 0).item())
    try:
        return bool(v != 0.0)
    except Exception:
        return True


@dataclass
class Exo:
    """外生量（ModelExo 的 python 版）。

    `gravity_a` 的分量可以是标量，也可以是**逐样本数组**（numpy 形状 [N] / torch 形状 [B]），
    此时会与状态的前导维广播 —— 用于"底盘静态倾斜、大 yaw 转动导致 A 系重力方向随之旋转"。
    """

    gravity_a: tuple = (0.0, 0.0)   # ★ A 系（大 yaw 转子系）重力平面分量 (m/s²)；水平 = (0,0)
    base_omega: float = 0.0         # 底盘绕关节轴角速度（本任务 = 0）
    base_alpha: float = 0.0         # 底盘绕关节轴角加速度（本任务 = 0）
    gravity_on: bool | None = None  # None ⇒ 自动判定（张量不能直接做 if 判断）
    # ★ 背隙死区中心 β：可以是标量，也可以是**逐样本**数组（np [T,B] / torch [B] 广播）。
    #   ``None`` ⇒ 用模型参数 `PlanarParams.backlash_beta`。
    backlash_beta: float | None = None

    def __post_init__(self):
        if self.gravity_on is None:
            gx, gy = self.gravity_a
            self.gravity_on = bool(_nonzero(gx) or _nonzero(gy))


EXO_ZERO = Exo()


def exo_from_gravity(gx, gy, base_omega: float = 0.0, base_alpha: float = 0.0) -> Exo:
    """按 A 系重力平面分量构造 Exo（自动置 gravity_on ⇒ eom 会自动启用重力项）。"""
    return Exo(gravity_a=(gx, gy), base_omega=base_omega, base_alpha=base_alpha)


def beta_of(p: PlanarParams, exo: Exo):
    """背隙死区中心 β: 优先用 Exo 里的（可逐样本），否则用模型参数。"""
    return p.backlash_beta if exo.backlash_beta is None else exo.backlash_beta


def p_direction(dx: float, dy: float, zero_angle_deg: float = 0.0):
    """★ "零点标定"后 P 必须满足的方向。

    离心平衡（大 yaw 恒速 Ω、小 yaw 松手 τ=0）给出平衡条件 `R(θ*)·P ∥ d`，
    稳定解取径向外侧 ⇒ **P = |P|·R(−θ*)·d̂**，其中 θ* = 平衡点在**当前零点坐标系**里的读数。

    返回单位方向 (ux, uy)。
    """
    n = math.hypot(dx, dy)
    if n < 1e-12:
        raise ValueError("P 方向约束需要 |d| > 0（几何偏置为 0 时方向无定义）")
    ux, uy = dx / n, dy / n
    d = math.radians(zero_angle_deg)
    return (ux * math.cos(d) + uy * math.sin(d), -ux * math.sin(d) + uy * math.cos(d))


# ============================================================================
# ★ 可学习参数分组 + 重参数化（两组: 全体实数 / 正数；固定参数不属于任何一组）
# ============================================================================
class ParamGroups:
    """可学习参数按**可取值范围**分成两组，并把冻结参数排除成"固定参数"。

    · ``real_names``     —— 全体实数: φ_j = raw_j（直接自由）
    · ``positive_names`` —— 正数:     φ_j = exp(raw_j)（正性隐式保证，无上下界）

    ``learnable_names = real_names + positive_names``（★ 可微仿真模型第 1 个参数的列顺序）。
    固定参数（``fixed_names``，含几何/λ/ε/偏置与被冻结的辨识参数）**不在**可学习向量里，
    它们的取值由 ``base`` 提供。**没有任何上下界 / clamp / 投影 / 惩罚**。

    ★ ``p_along_d=(ux, uy)``（可选）: 小 yaw 零点按"离心平衡点"标定后模型必然满足
    ``P = |P|·d̂`` ⇒ Px/Py 不是一个自由向量而是一个自由标量 s = |P|。此时 Px/Py 从
    ``real_names`` 里移除，而在 real 组**末尾**追加一个派生实数槽 ``P_along_d``
    （s 同样自由、无界、不强制为正：物理上应为正，让优化自己决定）。
    """

    def __init__(self, base: PlanarParams | None = None,
                 fixed_names=(), p_along_d=None):
        self.base = PlanarParams() if base is None else base
        fixed = {str(n) for n in fixed_names}
        unknown = fixed - set(PARAM_NAMES)
        if unknown:
            raise KeyError(f"未知固定参数名: {sorted(unknown)}")
        if p_along_d is not None:
            u = np.asarray(p_along_d, dtype=np.float64).reshape(2)
            n = float(np.hypot(u[0], u[1]))
            if n < 1e-12:
                raise ValueError("p_along_d 需要 |d| > 0（几何偏置为 0 时方向无定义）")
            self.p_along_d = (u[0] / n, u[1] / n)
            fixed |= {"Px", "Py"}
        else:
            self.p_along_d = None
        self.fixed_names = tuple(n for n in PARAM_NAMES if n in fixed)
        self.real_names = tuple(n for n in REAL_PARAM_NAMES if n not in fixed)
        if self.p_along_d is not None:
            self.real_names = self.real_names + (P_ALONG_D_NAME,)
        self.positive_names = tuple(n for n in POSITIVE_PARAM_NAMES if n not in fixed)
        self.learnable_names = self.real_names + self.positive_names
        self._pos = set(self.positive_names)

    # ── 基本属性 ──
    @property
    def n_learnable(self) -> int:
        return len(self.learnable_names)

    @property
    def mode(self) -> str:
        return "along_d" if self.p_along_d is not None else "free"

    def index(self, name: str) -> int:
        """可学习向量里的列号。"""
        try:
            return self.learnable_names.index(name)
        except ValueError as exc:
            raise KeyError(f"{name!r} 不在可学习参数里（固定参数为: "
                           f"{', '.join(self.fixed_names) or '无'}）") from exc

    def is_positive(self, name: str) -> bool:
        return name in self._pos

    def fixed_value(self, name: str) -> float:
        return float(getattr(self.base, name))

    # ── raw ↔ 物理参数 ──
    def to_raw_init(self, phi0: np.ndarray) -> np.ndarray:
        """φ0(18) → raw 初值（正数组取 log）。**只对初值取对数**，不给参数加任何界。"""
        phi0 = np.asarray(phi0, dtype=np.float64)
        out = []
        for nm in self.learnable_names:
            v = float(phi0[PARAM_INDEX[nm]]) if nm != P_ALONG_D_NAME else self._p_along_init(phi0)
            if nm in self._pos:
                # 初值非正时（不该发生）取一个极小正数做对数的**起点**，而非给参数加界
                out.append(math.log(v if v > 0.0 else 1e-6))
            else:
                out.append(v)
        return np.array(out, dtype=np.float64)

    def _p_along_init(self, phi0: np.ndarray) -> float:
        ux, uy = self.p_along_d
        return float(phi0[PARAM_INDEX["Px"]]) * ux + float(phi0[PARAM_INDEX["Py"]]) * uy

    def to_physical(self, raw):
        """raw（np 或 torch，长度 = n_learnable）→ 长度 n_learnable 的物理参数（同类型）。"""
        if isinstance(raw, np.ndarray):
            out = np.empty(self.n_learnable, dtype=np.float64)
            for k, nm in enumerate(self.learnable_names):
                v = float(raw[k])
                out[k] = math.exp(v) if nm in self._pos else v
            return out
        out = [None] * self.n_learnable
        for k, nm in enumerate(self.learnable_names):
            out[k] = torch.exp(raw[k]) if nm in self._pos else raw[k]
        return torch.stack(out)

    def build_params(self, values, base: PlanarParams | None = None) -> PlanarParams:
        """物理参数值（长度 P，或 [B,P]）→ 完整 PlanarParams（固定参数取 base）。"""
        base = self.base if base is None else base
        two_d = (getattr(values, "ndim", None) == 2
                 or (hasattr(values, "dim") and values.dim() == 2))
        kw = {}
        for k, nm in enumerate(self.learnable_names):
            v = values[:, k] if two_d else values[k]
            if nm == P_ALONG_D_NAME:
                ux, uy = self.p_along_d
                kw["Px"] = v * ux
                kw["Py"] = v * uy
            else:
                kw[nm] = v
        return replace(base, **kw)

    def full_vector(self, values) -> np.ndarray:
        """可学习物理参数（np，长度 P）→ 完整 18 维向量（固定参数取 base）。"""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        out = self.base.vector()
        for k, nm in enumerate(self.learnable_names):
            if nm == P_ALONG_D_NAME:
                ux, uy = self.p_along_d
                out[PARAM_INDEX["Px"]] = v[k] * ux
                out[PARAM_INDEX["Py"]] = v[k] * uy
            else:
                out[PARAM_INDEX[nm]] = v[k]
        return out

    def describe(self) -> str:
        """给日志用的说明（两组各含哪些参数 + 固定参数）。"""
        parts = []
        if self.real_names:
            parts.append("全体实数(raw): " + ",".join(self.real_names))
        if self.positive_names:
            parts.append("正数(exp(raw)): " + ",".join(self.positive_names))
        if self.p_along_d is not None:
            ux, uy = self.p_along_d
            parts.append(f"P=|P|·d̂(d̂=({ux:+.4f},{uy:+.4f})) 单标量 {P_ALONG_D_NAME}")
        if self.fixed_names:
            parts.append("固定(不优化): " + ",".join(self.fixed_names))
        return "; ".join(parts) if parts else "（全部固定，无可学习参数）"


def resolve_fixed_names(fit_axis: str = "both", freeze_params=(), freeze_backlash_through: bool = True,
                        p_constraint: str = "free", use_beta_column: bool = True) -> tuple:
    """把旧的"冻结语义"（fit_axis / 下标冻结 / γ / P 约束 / β 列）翻译成**固定参数名**。

    返回的名字交给 :class:`ParamGroups`，它们就完全不属于两组可学习参数了。

    ★ ``backlash_beta`` 现在**恒为固定参数**：β 是必需数据、永远取当前拟合帧的数据列
      （``--beta-mode=fit`` 已移除，β 不再是可学习参数）。
    """
    fixed = set()
    if fit_axis == "big":
        # 只拟合大 yaw：小 yaw 摩擦不可辨识 → 固定
        fixed |= {"fc_small", "fv_small"}
    elif fit_axis == "small":
        # 只拟合小 yaw：大 yaw 惯量/摩擦、以及电机侧+背隙那 8 个参数都不进可观测量
        fixed |= {"Jbig_eff", "fc_big", "fv_big"}
        fixed |= set(EXTRA_PARAM_NAMES)
    elif fit_axis != "both":
        raise ValueError("fit_axis 必须是 both / big / small")
    for j in freeze_params:
        fixed.add(PARAM_NAMES[int(j)])
    if freeze_backlash_through:
        # γ：默认钉在初值上。它的定位只是死区内给优化器一个非零梯度；一旦放开，优化器会
        # 拿它去"填"刚性接触的台阶（实测 δ 被撑到 0.25 rad，真值 0.087）。
        fixed.add("backlash_through")
    mode = str(p_constraint).replace("-", "_")
    if mode not in ("free", "along_d", "zero"):
        raise ValueError("p_constraint 必须是 free / along_d / zero")
    if mode == "zero":
        fixed |= {"Px", "Py"}
    # ★ β 恒固定（必需数据列给出，不参与优化；--beta-mode=fit 已移除）
    fixed.add("backlash_beta")
    return tuple(fixed)
