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

    0 X_b   1 Y_b   2 X_s   3 Y_s   4 I_b   5 I_s
    6 mu    7 f_bc  8 f_bv  9 f_sc  10 f_sv

★ **辨识模型 = 两刚体平面拉格朗日模型的缩合参数版**（`D=(dx,dy)` 已知、不辨识）:

    J_b = I_b + (X_b²+Y_b²)              （m_b 固定为 1）
    J_s = I_s + (X_s²+Y_s²)/μ
    A   = J_b + μ|D|²,   B = J_s
    μK  = D·R(θ_s)·(X_s,Y_s) = D_x·Q_x + D_y·Q_y
    Δ   = A·B − (μK)²                    ≥ J_s·J_b + μ|D|²·I_s > 0  ★构造保证

★ 用 `(I_b, I_s)` 而不是字面上的 `(J_b, J_s)` 当自由参数: 两者等价、个数相同，但
  `I_b, I_s, μ > 0` 让 `Δ > 0` 成为**构造性质**（否则 `I_s<0` 会让 Δ 在某些 θ_s 上变负、
  加速度爆炸——这正是旧 3-DOF 模型 `det2<0` 那条 NaN 通路）。
★ `μ` 是**规范自由度**（只以 `μ|D|²`、`X_b+μD_x`、`Y_b+μD_y`、`(X_s²+Y_s²)/μ` 组合进入），
  **默认参与辨识**（不固定任何参数）；若希望拟合里不带这条平方向，用 `--fix-mu` 把它钉住。

★ **`(I_b, I_s)` 而不是字面上的 `(J_b, J_s)`**：`J_b = I_b+(X_b²+Y_b²)`、
  `J_s = I_s+(X_s²+Y_s²)/μ`。两者等价、个数相同，但 `I_b, I_s, μ > 0` 让
  `Δ = A·B − (μK)² ≥ J_s·J_b + μ|D|²·I_s > 0` 成为**构造性质**。
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
AXIS_CHANNELS = {AXIS_BIG: (0,), AXIS_SMALL: (1,)}   # 2-DOF: 0=b(云台), 1=s(小yaw)


# ============================================================================
# ★★ 控制力矩符号（**只作用于辨识环境**；主工程/控制器完全不受影响）
#   两路**独立**控制:
#     · big   → `tau[:, 0]` = τ_cmd   （发给大 yaw 电机的力矩）
#     · small → `tau[:, 1]` = τ_small （发给小 yaw 的力矩）
#   ★ 默认: **大 yaw 不取反 (+1)、小 yaw 取反 (−1)**。想恢复"两路都原样"就显式给
#     `--tau-sign-small=1`。
#
#   施加点是**唯一**的: `data.segment_from_arrays` 构造 Segment 时统一施加 ⇒
#   Adam / CMA-ES / 手动标定 GUI / 画图脚本、以及三个前向后端（torch / numpy / C++）
#   看到的是同一组符号，不需要在每个模型实现里再加一份开关（否则很容易只改一半）。
#   ★ 只动控制力矩: 重力列、β 列、θ/ω、以及所有模型参数（含 `tau_offset_*`）都不参与。
#   两路都为 +1 时 `apply_tau_sign` 原样返回输入对象 ⇒ 该组合下数值**逐位不变**。
# ============================================================================
TAU_SIGN_BIG_DEFAULT = 1.0        # 大 yaw 电机通道 τ_cmd
TAU_SIGN_SMALL_DEFAULT = -1.0     # ★ 小 yaw 通道 τ_small（默认**取反**）

_TAU_SIGN = (TAU_SIGN_BIG_DEFAULT, TAU_SIGN_SMALL_DEFAULT)     # (big, small)


def _check_sign(v, what: str) -> float:
    s = float(v)
    if s not in (-1.0, 1.0):
        raise ValueError(f"{what} 的 τ 符号只接受 +1 或 -1，收到 {v!r}")
    return s


def set_tau_sign_big(v) -> float:
    """设置**大 yaw 电机**通道（τ_cmd）的符号；返回生效后的值。"""
    global _TAU_SIGN
    _TAU_SIGN = (_check_sign(v, "大 yaw"), _TAU_SIGN[1])
    return _TAU_SIGN[0]


def set_tau_sign_small(v) -> float:
    """设置**小 yaw** 通道（τ_small）的符号；返回生效后的值。"""
    global _TAU_SIGN
    _TAU_SIGN = (_TAU_SIGN[0], _check_sign(v, "小 yaw"))
    return _TAU_SIGN[1]


def set_tau_sign(big=None, small=None):
    """同时/分别设置两路符号（None = 保持不变）。返回 ``(big, small)``。"""
    if big is not None:
        set_tau_sign_big(big)
    if small is not None:
        set_tau_sign_small(small)
    return _TAU_SIGN


def reset_tau_sign():
    """恢复默认符号（大 yaw +1、小 yaw −1）。"""
    global _TAU_SIGN
    _TAU_SIGN = (TAU_SIGN_BIG_DEFAULT, TAU_SIGN_SMALL_DEFAULT)
    return _TAU_SIGN


def tau_sign() -> tuple:
    """当前两路符号 ``(big, small)``。"""
    return _TAU_SIGN


def tau_sign_big() -> float:
    return _TAU_SIGN[0]


def tau_sign_small() -> float:
    return _TAU_SIGN[1]


def tau_sign_desc() -> str:
    """一行描述当前 τ 符号（写进结果文件的配方行，保证参数文件自描述）。"""
    sb, ss = _TAU_SIGN
    tag = "默认" if _TAU_SIGN == (TAU_SIGN_BIG_DEFAULT, TAU_SIGN_SMALL_DEFAULT) else "改过"
    return f"τ符号[{tag}]: 大yaw={sb:+.0f} 小yaw={ss:+.0f}"


def apply_tau_sign(tau):
    """按两路符号给控制力矩 ``[..., 2]`` 加符号；两路都是 +1 时**原样返回**。"""
    sb, ss = _TAU_SIGN
    if sb == 1.0 and ss == 1.0:
        return tau
    out = np.array(tau, dtype=np.float64, copy=True)
    if sb != 1.0:
        out[..., 0] *= sb
    if ss != 1.0:
        out[..., 1] *= ss
    return out

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
    ParamSpec("X_b", "kg·m", False, -0.004230, "m_b·P_bx（b 侧一阶矩 x，可正可负）"),
    ParamSpec("Y_b", "kg·m", False, -0.022760, "m_b·P_by"),
    ParamSpec("X_s", "kg·m", False, 0.003943, "m_s·P_sx（s 侧一阶矩 x）"),
    ParamSpec("Y_s", "kg·m", False, 0.009707, "m_s·P_sy"),
    ParamSpec("I_b", "kg·m²", True, 0.002113, "b 绕**质心**转动惯量（J_b = |X_b|²+I_b）"),
    ParamSpec("I_s", "kg·m²", True, 0.000307, "s 绕**质心**转动惯量（J_s = |X_s|²/μ+I_s）"),
    ParamSpec("mu", "kg", True, 0.257475, "★ m_s（规范自由度；默认**可学习**，--fix-mu 可钉住）"),
    ParamSpec("f_bc", "N·m", True, 0.001163, "b 侧库仑摩擦"),
    ParamSpec("f_bv", "N·m·s/rad", True, 0.018365, "b 侧粘滞摩擦"),
    ParamSpec("f_sc", "N·m", True, 0.005668, "s 侧库仑摩擦"),
    ParamSpec("f_sv", "N·m·s/rad", True, 0.325935, "s 侧粘滞摩擦"),
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

# ── 便捷子集（打印/`fit_axis` 用；**参数分组以可取值范围为准**）──
CORE_PARAM_NAMES = PARAM_NAMES[:6]                       # X_b,Y_b,X_s,Y_s,I_b,I_s
NCORE = len(CORE_PARAM_NAMES)
# `fit_axis="small"` 时要固定的 b 侧参数（b 侧惯量/摩擦在小 yaw 数据里不可观测）
EXTRA_PARAM_NAMES = ("X_b", "Y_b", "I_b", "f_bc", "f_bv")
# ★ 默认**不固定任何参数**（μ 也是可学习的）。注意 μ 是规范自由度（只以 μ|D|²、
#   X_b+μdx、Y_b+μdy、(X_s²+Y_s²)/μ 组合进入动力学）⇒ 拟合里它有一条平方向，
#   想消掉就用 `--fix-mu` 把它钉在初值上。
DEFAULT_FIXED = ()

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
    # ── ★ 11 个辨识参数的字段默认值 = **参数表 PARAM_SPECS 的 `default`**（唯一来源）──
    X_b: float = _spec_default("X_b")
    Y_b: float = _spec_default("Y_b")
    X_s: float = _spec_default("X_s")
    Y_s: float = _spec_default("Y_s")
    I_b: float = _spec_default("I_b")
    I_s: float = _spec_default("I_s")
    mu: float = _spec_default("mu")
    f_bc: float = _spec_default("f_bc")
    f_bv: float = _spec_default("f_bv")
    f_sc: float = _spec_default("f_sc")
    f_sv: float = _spec_default("f_sv")

    # ── 派生量（不是自由参数）──
    @property
    def J_b(self):
        """J_b = I_b + m_b|P_b|²（m_b = 1）。"""
        return self.I_b + self.X_b * self.X_b + self.Y_b * self.Y_b

    @property
    def J_s(self):
        """J_s = I_s + m_s|P_s|² = I_s + (X_s²+Y_s²)/μ。"""
        return self.I_s + (self.X_s * self.X_s + self.Y_s * self.Y_s) / self.mu

    @property
    def D2(self):
        """|D|² = dx² + dy²。"""
        return self.dx * self.dx + self.dy * self.dy

    @property
    def A(self):
        return self.J_b + self.mu * self.D2

    @property
    def B(self):
        return self.J_s

    def Delta_min(self):
        """Δ 的下界 = J_s·J_b + μ|D|²·I_s（> 0 ⇒ 恒正定）。"""
        return self.J_s * self.J_b + self.mu * self.D2 * self.I_s

    # ── 回写主模型头文件的 6 个参数（主模型仍是 3-DOF 背隙版，用这个换算）──
    def to_header_params(self) -> dict:
        """缩合参数 → `include/tcbs/mpc/planar_yaw_model.h` 的 6 个惯量/重力参数。

            Jbig_eff = A,  Js = J_s,  (Px,Py) = (X_s,Y_s),  (Pbx,Pby) = (X_b+μdx, Y_b+μdy)
        （背隙/电机那 6 个参数本模型不辨识，需沿用主模型现有值。）
        """
        return {"Jbig_eff": self.A, "Js": self.J_s, "Px": self.X_s, "Py": self.Y_s,
                "Pbx": self.X_b + self.mu * self.dx, "Pby": self.Y_b + self.mu * self.dy}

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
    """外生量（2-DOF 模型的输入）。

    `gravity_a = (g_ax, g_ay)` 是**A 系（随 b 转的转子系）**重力平面分量 (m/s²)，
    可以逐样本（数组）—— 数据列 `gravity_ax/ay` 就是这个，直接喂。
    `base_omega` 是底盘绕关节轴的角速度 ω_c（`chassis_yaw_rate`）。

    ★ 模型里**没有** α_c 项（α_c = 0）；θ_c 也不需要（A 系重力已含 ψ_b 的转动）。
    """

    gravity_a: tuple = (0.0, 0.0)
    base_omega: float = 0.0
    gravity_on: bool | None = None

    def __post_init__(self):
        if self.gravity_on is None:
            gx, gy = self.gravity_a
            self.gravity_on = bool(_nonzero(gx) or _nonzero(gy))


EXO_ZERO = Exo()


def exo_from_gravity(gx, gy, base_omega: float = 0.0) -> Exo:
    """按 A 系重力平面分量构造 Exo（自动置 gravity_on）。"""
    return Exo(gravity_a=(gx, gy), base_omega=base_omega)


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
            fixed |= {"X_s", "Y_s"}
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
        return float(phi0[PARAM_INDEX["X_s"]]) * ux + float(phi0[PARAM_INDEX["Y_s"]]) * uy

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
                kw["X_s"] = v * ux
                kw["Y_s"] = v * uy
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
                out[PARAM_INDEX["X_s"]] = v[k] * ux
                out[PARAM_INDEX["Y_s"]] = v[k] * uy
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


def resolve_fixed_names(fit_axis: str = "both", freeze_params=(), p_constraint: str = "free",
                        fix_mu: bool = False) -> tuple:
    """把 ``fit_axis`` / 下标冻结 / P 约束 / 是否钉住 μ，翻译成**固定参数名**。

    返回的名字交给 :class:`ParamGroups`，它们就完全不属于两组可学习参数。
    ★ 默认**不固定任何参数**（μ 也可学习）；`fix_mu=True` 才把 μ 钉在初值上。
    """
    fixed = set(DEFAULT_FIXED) | ({"mu"} if fix_mu else set())
    if fit_axis == "big":
        # 只拟合 b 通道：s 侧摩擦在 b 的数据里不可观测 → 固定
        fixed |= {"f_sc", "f_sv"}
    elif fit_axis == "small":
        # 只拟合 s 通道：b 侧的惯量/一阶矩/摩擦都不进可观测量
        fixed |= {"X_b", "Y_b", "I_b", "f_bc", "f_bv"}
        fixed |= set(EXTRA_PARAM_NAMES)
    elif fit_axis != "both":
        raise ValueError("fit_axis 必须是 both / big / small")
    for j in freeze_params:
        fixed.add(PARAM_NAMES[int(j)])
    mode = str(p_constraint).replace("-", "_")
    if mode not in ("free", "along_d", "zero"):
        raise ValueError("p_constraint 必须是 free / along_d / zero")
    if mode == "zero":
        fixed |= {"X_s", "Y_s"}
    return tuple(fixed)
