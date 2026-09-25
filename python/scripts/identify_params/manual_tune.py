#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手动标定参数 —— PyQt5 GUI：左 3×3 曲线，右 文件翻页 + 参数滑块（正数对数调节）。

布局
----
    左侧:  3 行（当前组 3 个采样文件）× 3 列（电机 / 云台 / 小 yaw）；
           每格 **实线 = 实测 θ**、**虚线 = 当前参数下的仿真 θ**（从该段记录初值出发，
           用该段记录的 τ / 重力 / β 前向积分，与辨识/评测同一套模型），
           外加**原始控制力矩曲线**（点线，右轴，记录值、与参数无关）:
              · 电机列 = τ_cmd、小 yaw 列 = τ_small；**云台列没有控制力矩 ⇒ 不画**。
    右侧:  「上一组 / 下一组」按 3 个文件一组滚动；下面每个参数一行
           （名 → 滑块 → 数值框）。**正数参数在对数范围内调节**（默认 [1e-4, 500]，
           与 CMA-ES 的搜索盒一致），实数参数（Px/Py/β/Pbx/Pby）线性调节。
           滑块与数值框双向同步，拖动时**实时**重算重绘。

跑法::

    cd python/scripts && python3 -m identify_params.manual_tune
    # 或直接: python3 python/scripts/identify_params/manual_tune.py --data='data/cars/Sentry1/sysid'
    # 先给一组数据: --data=<目录 或 glob>（目录取里面所有 npz/csv，按文件名排序）

依赖: PyQt5 + matplotlib（Qt5Agg 后端）—— 都已在本环境可用，无需额外安装。
"""

from __future__ import annotations

import glob as globmod
import os
import re
import sys

import numpy as np

# ★ 允许直接按文件路径运行（与 cli.py 同一套引导）
if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "identify_params"

from .data import load_segments
from .model import rollout_batched_np
from .params import (
    PARAM_NAMES,
    PARAM_UNITS,
    POSITIVE_PARAM_NAMES,
    TAU_SIGN_BIG_DEFAULT,
    TAU_SIGN_SMALL_DEFAULT,
    PlanarParams,
    default_param_vector,
    set_tau_sign,
    tau_sign_big,
    tau_sign_desc,
    tau_sign_small,
)
from .plotting import _setup_font
from .train import write_params_file

try:
    from PyQt5 import QtCore, QtWidgets
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[error] 需要 PyQt5: {exc}\n安装: pip install PyQt5") from exc

import matplotlib
try:
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[error] 需要 matplotlib 的 Qt5Agg 后端: {exc}") from exc

# 正数参数的滑块范围（物理量；与 CMA-ES 的搜索盒默认一致）
LOG_LO, LOG_HI = 1e-4, 500.0
# 实数参数（Px/Py/β/Pbx/Pby）的滑块范围: ±此值
REAL_RANGE = 1.0
SLIDER_STEPS = 1000
COL_NAMES = ("云台 θ_b", "小 yaw θ_s")    # 2-DOF: 只有这两个通道被建模


def _collect_files(spec: str) -> list:
    """把「目录 / glob / 逗号分隔路径」统一成文件列表（按名字排序）。"""
    out = []
    for pat in str(spec).split(","):
        pat = pat.strip()
        if not pat:
            continue
        if os.path.isdir(pat):
            hit = sorted(globmod.glob(os.path.join(pat, "*.npz"))
                         + globmod.glob(os.path.join(pat, "*.csv")))
        else:
            hit = sorted(globmod.glob(pat)) or ([pat] if os.path.isfile(pat) else [])
        out.extend(hit)
    seen = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def _parse_params_file(path: str) -> np.ndarray:
    """读 `名字 = 值` 文本（`--out` / 断点的格式）。

    ★ 兼容**旧格式**: 老参数文件写的是模型参数 `Jbig_eff`；现在辨识槽 0 是正定余量
      `Jbig_slack`，两者关系 `Jbig_eff = Jbig_slack + |d|²|P|²/Js`。这里按旧文件里的
      `Js/Px/Py` 反解出 `Jbig_slack`，并在余量 ≤0（旧文件本身不满足正定的组合）
      时给个提示。
    """
    vals = {}
    legacy = {}
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)", ln)
            if not m:
                continue
            nm, v = m.group(1), float(m.group(2))
            if nm in PARAM_NAMES:
                vals[nm] = v
            else:
                legacy[nm] = v            # 旧 18 参名（Jbig_eff/Js/Px/...）先收着
    phi = default_param_vector()
    for k, v in vals.items():
        phi[PARAM_NAMES.index(k)] = v
    if legacy:
        # 旧 18 参（3-DOF 背隙）→ 新 11 参（2-DOF 缩合）。固定 μ = 默认值做规范固定。
        mu = float(default_param_vector()[PARAM_NAMES.index("mu")])
        dx, dy = 0.0, 0.07
        g = lambda n: float(legacy.get(n, 0.0))                     # noqa: E731
        Jb = g("Jbig_eff") - mu * (dx * dx + dy * dy)
        Xb, Yb = g("Pbx") - mu * dx, g("Pby") - mu * dy
        Xs, Ys = g("Px"), g("Py")
        Js = g("Js")
        phi[:] = np.array([Xb, Yb, Xs, Ys,
                           Jb - (Xb * Xb + Yb * Yb),                 # I_b
                           Js - (Xs * Xs + Ys * Ys) / mu,            # I_s
                           mu, g("fc_big"), g("fv_big"), g("fc_small"), g("fv_small")])
        print(f"[params] 旧 18 参 → 2-DOF 11 参（μ 固定 {mu:g}）: "
              f"J_b={Jb:.5g} J_s={Js:.5g} (X_s,Y_s)=({Xs:.4g},{Ys:.4g})")
    return phi


class ParamRow(QtWidgets.QWidget):
    """一行参数控件: 名字 + 滑块 + 数值框（正数用对数映射）。"""

    def __init__(self, index: int, value: float, changed, is_positive: bool):
        super().__init__()
        self.index = int(index)
        self.is_positive = bool(is_positive)
        self._changed = changed
        name = PARAM_NAMES[self.index]
        self.lo, self.hi = ((LOG_LO, LOG_HI) if self.is_positive
                            else (-REAL_RANGE, REAL_RANGE))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        self.lbl = QtWidgets.QLabel(f"{self.index:>2} {name}")
        self.lbl.setFixedWidth(150)
        self.lbl.setToolTip(f"{name} [{PARAM_UNITS[self.index]}]"
                            + ("（正数：滑块为对数）" if self.is_positive else "（实数：线性）"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(6)
        self.spin.setRange(self.lo, self.hi)
        self.spin.setSingleStep(max(1e-4, (self.hi - self.lo) / 2000.0))
        self.spin.setFixedWidth(110)
        lay.addWidget(self.lbl)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._sync_guard = False
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(value)

    # ── 物理值 ↔ 滑块 ──
    def _to_slider(self, v: float) -> int:
        if self.is_positive:
            v = min(max(float(v), self.lo), self.hi)
            t = (np.log(v) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo))
        else:
            t = (float(v) - self.lo) / (self.hi - self.lo)
        return int(round(min(max(t, 0.0), 1.0) * SLIDER_STEPS))

    def _to_value(self, pos: int) -> float:
        t = pos / float(SLIDER_STEPS)
        if self.is_positive:
            return float(np.exp(np.log(self.lo) + t * (np.log(self.hi) - np.log(self.lo))))
        return float(self.lo + t * (self.hi - self.lo))

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.slider.setValue(self._to_slider(v))
            self.spin.setValue(float(np.clip(v, self.lo, self.hi)))
        finally:
            self._sync_guard = False

    def _from_slider(self, pos: int) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.spin.setValue(self._to_value(pos))
        finally:
            self._sync_guard = False
        self._changed(self.index, self.value())

    def _from_spin(self, v: float) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.slider.setValue(self._to_slider(v))
        finally:
            self._sync_guard = False
        self._changed(self.index, float(v))


class ManualTuneWindow(QtWidgets.QMainWindow):
    def __init__(self, files: list, nrow: int = 3, ncol: int = 2):
        super().__init__()
        self.setWindowTitle("identify_params —— 手动标定（实测 vs 仿真）")
        self.files = list(files)
        self.group = 0                                   # 当前组起始文件下标
        self.phi = default_param_vector()
        self._segs = None                                # 当前组的数据段
        self._files_cur = []
        self._sim = None            # C++ 内核（首次 _simulate 时建）
        self._backend = "未初始化"
        self._dirty = False

        # ── 左: 3×3 棋盘图 ──
        _setup_font()
        self.fig = Figure(figsize=(11, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(nrow, ncol, squeeze=False)
        self.lines = {}
        self.axt = {}
        for r in range(nrow):
            for c in range(ncol):
                ax = self.axes[r][c]
                ln_m, = ax.plot([], [], color=f"C{c}", lw=1.3, label="实测 θ")
                ln_s, = ax.plot([], [], color=f"C{c}", lw=1.3, ls="--", label="仿真 θ")
                # 右轴: **喂给模型的控制力矩**（两列各一路）
                axt, ln_t = None, None
                if c in (0, 1):
                    axt = ax.twinx()
                    tau_lab = "T_b [N·m]" if c == 0 else "T_s [N·m]"
                    ln_t, = axt.plot([], [], color="0.30", lw=1.0, ls=":",
                                     label=f"{tau_lab}（记录，右轴）")
                self.lines[(r, c)] = (ln_m, ln_s, ln_t)
                self.axt[(r, c)] = axt
                ax.grid(True, alpha=0.3)
                if r == 0:
                    ax.set_title(COL_NAMES[c], fontsize=10)
                if c == 0:
                    ax.set_ylabel("θ [rad]", fontsize=9)
                if r == nrow - 1:
                    ax.set_xlabel("t [s]", fontsize=9)
                ax.tick_params(labelsize=8)
                if axt is not None:
                    axt.tick_params(labelsize=7, colors="0.35")
                    axt.set_ylabel(("T_b" if c == 0 else "T_s") + " [N·m]",
                                   fontsize=8, color="0.35")
        self.axes[0][0].legend(handles=[self.lines[(0, 0)][0], self.lines[(0, 0)][1],
                                        self.lines[(0, 0)][2]], fontsize=7, loc="best")

        # ── 右: 数据翻页 + 参数滑块 ──
        right = QtWidgets.QWidget()
        right.setFixedWidth(430)
        rlay = QtWidgets.QVBoxLayout(right)

        gb_data = QtWidgets.QGroupBox("数据")
        dl = QtWidgets.QVBoxLayout(gb_data)
        row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("选择数据…")
        self.btn_prev = QtWidgets.QPushButton("◀ 上一组")
        self.btn_next = QtWidgets.QPushButton("下一组 ▶")
        for b in (self.btn_open, self.btn_prev, self.btn_next):
            row.addWidget(b)
        dl.addLayout(row)
        # ★ 控制力矩符号: 两路独立（在**加载时**施加 ⇒ 切换后必须重新读数据）
        self.chk_tau_big = QtWidgets.QCheckBox("大 yaw 电机 τ_cmd 取反")
        self.chk_tau_small = QtWidgets.QCheckBox("小 yaw τ_small 取反")
        for chk, tip in (
            (self.chk_tau_big, "只作用于本辨识环境（数据加载时施加），不影响主工程/控制器。"),
            (self.chk_tau_small, "只作用于本辨识环境（数据加载时施加），不影响主工程/控制器。"),
        ):
            chk.setToolTip(tip + "\n勾选后会重新读取当前组数据并重算曲线。\n"
                                 "★ 只动控制力矩：重力列、β 列、θ/ω 与所有模型参数都不变。")
        tau_box = QtWidgets.QHBoxLayout()
        tau_box.addWidget(self.chk_tau_big)
        tau_box.addWidget(self.chk_tau_small)
        dl.addLayout(tau_box)
        self.chk_tau_big.setChecked(tau_sign_big() < 0)      # 初值取自当前符号（连接信号之前）
        self.chk_tau_small.setChecked(tau_sign_small() < 0)
        self.lbl_group = QtWidgets.QLabel("")
        self.lbl_group.setWordWrap(True)
        dl.addWidget(self.lbl_group)
        rlay.addWidget(gb_data)

        gb_p = QtWidgets.QGroupBox("参数（正数=对数滑块；实数=线性滑块）")
        pl = QtWidgets.QVBoxLayout(gb_p)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(0)
        self.rows = []
        pos_set = set(POSITIVE_PARAM_NAMES)
        for i in range(len(PARAM_NAMES)):
            pr = ParamRow(i, float(self.phi[i]), self._on_param, PARAM_NAMES[i] in pos_set)
            self.rows.append(pr)
            il.addWidget(pr)
        il.addStretch(1)
        scroll.setWidget(inner)
        pl.addWidget(scroll)
        rlay.addWidget(gb_p, 1)

        gb_out = QtWidgets.QGroupBox("参数文件")
        ol = QtWidgets.QHBoxLayout(gb_out)
        self.btn_load = QtWidgets.QPushButton("载入…")
        self.btn_save = QtWidgets.QPushButton("另存为…")
        self.btn_reset = QtWidgets.QPushButton("复位")
        for b in (self.btn_load, self.btn_save, self.btn_reset):
            ol.addWidget(b)
        rlay.addWidget(gb_out)

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.addWidget(self.canvas, 1)
        h.addWidget(right)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.btn_open.clicked.connect(self.on_open)
        self.btn_prev.clicked.connect(lambda: self.shift_group(-1))
        self.btn_next.clicked.connect(lambda: self.shift_group(+1))
        self.btn_load.clicked.connect(self.on_load_params)
        self.btn_save.clicked.connect(self.on_save_params)
        self.btn_reset.clicked.connect(self.on_reset)
        self.chk_tau_big.toggled.connect(self.on_toggle_tau_sign)
        self.chk_tau_small.toggled.connect(self.on_toggle_tau_sign)

        # 拖动滑块时合并重算（30 ms 一次），避免每个像素都重画
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.load_group()
        self.resize(1500, 900)

    # ── 参数变化（合并刷新）──
    def _on_param(self, idx: int, value: float) -> None:
        self.phi[int(idx)] = float(value)
        self._dirty = True

    def _tick(self) -> None:
        if self._dirty:
            self._dirty = False
            self.redraw()

    # ── 数据 ──
    def on_open(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择采样数据目录")
        if d:
            self.files = _collect_files(d)
        else:
            fs, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "选择采样数据文件（可多选，按 3 个一组翻页）", "",
                "采样数据 (*.npz *.csv);;所有文件 (*)")
            if fs:
                self.files = list(fs)
        self.group = 0
        self.load_group()

    def on_toggle_tau_sign(self, _checked: bool = False) -> None:
        """两路 τ 符号都在**加载时**施加 ⇒ 任一路切换后都要重新读数据再重算。"""
        set_tau_sign(big=-1.0 if self.chk_tau_big.isChecked() else 1.0,
                     small=-1.0 if self.chk_tau_small.isChecked() else 1.0)
        self.load_group()

    def shift_group(self, step: int) -> None:
        n = len(self.files)
        if n == 0:
            return
        last = max(0, ((n - 1) // 3) * 3)          # 最后一组的起点
        self.group = int(min(max(self.group + step * 3, 0), last))
        self.load_group()

    def load_group(self) -> None:
        grp = self.files[self.group:self.group + 3]
        self._files_cur = grp
        self._segs = []
        for f in grp:
            try:
                got = load_segments([f], verbose=False)
                if got:
                    self._segs.append(got[0])
            except Exception as exc:
                print(f"[warn] 读不了 {f}: {exc}")
        n_grp = ((len(self.files) - 1) // 3 + 1) if self.files else 0
        g_idx = (self.group // 3 + 1) if self.files else 0
        names = "\n".join("  " + os.path.basename(f) for f in grp) or "  （未选择数据）"
        self.lbl_group.setText(f"第 {g_idx}/{max(1, n_grp)} 组（共 {len(self.files)} 个文件）\n"
                               f"{names}")
        self.btn_prev.setEnabled(self.group > 0)
        self.btn_next.setEnabled(self.group + 3 < len(self.files))
        self.redraw()

    # ── 仿真 + 画图 ──
    def _simulate(self):
        """返回 ``[(seg, θ_sim[T,2]), ...]``：从各段记录初值出发、用记录 τ/重力前向积分。

        ★ 走 **C++ `planar2_sim` 后端**（与拟合同一个内核）；不可用时回退 numpy。
        """
        if not self._segs:
            return []
        base = PlanarParams(dx=0.0, dy=0.07).with_vector(self.phi)
        out = []
        for s in self._segs:
            T = int(s.T)
            sc = np.zeros((1, 5))
            sc[0, 0:2] = s.theta[0, 1:]                  # (θ_b, θ_s)
            sc[0, 2:4] = s.dtheta[0, 1:]
            sc[0, 4] = float(getattr(s, "base_omega", 0.0) or 0.0)
            sv = np.zeros((1, T, 4))
            sv[0, :, 0:2] = s.tau
            if s.gravity is not None:
                sv[0, :, 2:4] = s.gravity
            if self._sim is None:
                self._sim = self._make_sim(base, float(s.dt))
            if self._sim is not None:
                th, _ = self._sim.rollout(np.repeat(base.vector()[None, :], 1, axis=0), sc, sv)
            else:
                th, _ = rollout_batched_np(base, sc, sv, float(s.dt), 4, "rk4")
            out.append((s, th[0]))
        return out

    def _make_sim(self, base: PlanarParams, dt: float):
        """建一次 C++ 内核（失败则返回 None ⇒ 回退 numpy），并在状态栏说明用的是哪个后端。"""
        try:
            from .planar2_sim import FastSimulator
            self._backend = "C++ planar2_sim"
            return FastSimulator(base, dt=dt, substeps=4, integrator="rk4")
        except Exception as exc:                                    # pragma: no cover
            self._backend = f"numpy（C++ 不可用: {type(exc).__name__}）"
            return None

    def redraw(self) -> None:
        sims = self._simulate()
        for r in range(self.axes.shape[0]):
            for c in range(self.axes.shape[1]):
                ax = self.axes[r][c]
                ln_m, ln_s, ln_t = self.lines[(r, c)]
                axt = self.axt[(r, c)]
                if r >= len(sims) or c >= self.axes.shape[1]:
                    ln_m.set_data([], []); ln_s.set_data([], [])
                    if ln_t is not None:
                        ln_t.set_data([], [])
                    ax.set_title("（无数据）" if r == 0 else "", fontsize=9)
                    continue
                seg, th = sims[r]
                t = np.arange(seg.T) * seg.dt
                # ★ 记录列: 0=电机侧（不建模）, 1=云台 θ_b, 2=小 yaw θ_s
                #   仿真通道: 0=θ_b, 1=θ_s  ⇒ 实测取 1+c，两边必须同一个物理量
                meas = np.asarray(seg.theta, dtype=np.float64)[:, 1 + c]
                pred = np.asarray(th, dtype=np.float64)[:, c]
                err = meas - pred                                 # 不 wrap（与拟合口径一致）
                rmse = float(np.degrees(np.sqrt(np.mean(err ** 2))))
                ln_m.set_data(t, meas)
                ln_s.set_data(t, pred)
                if ln_t is not None:                       # 喂给模型的控制力矩（含 τ 符号）
                    tau_k = np.asarray(seg.tau, dtype=np.float64)[:, 0 if c == 0 else 1]
                    ln_t.set_data(t, tau_k)
                    tm, tn = float(np.max(tau_k)), float(np.min(tau_k))
                    tpad = max(1e-6, 0.10 * (tm - tn))
                    axt.set_ylim(tn - tpad, tm + tpad)
                    axt.set_xlim(float(t[0]), float(t[-1]) if seg.T > 1 else 1.0)
                ax.set_xlim(float(t[0]), float(t[-1]) if seg.T > 1 else 1.0)
                m = float(np.max(meas)); mn = float(np.min(meas))
                pad = max(1e-3, 0.10 * (m - mn))
                ax.set_ylim(mn - pad, m + pad)
                title = f"{os.path.basename(seg.source)[:22]}  RMSE={rmse:.2f}°"
                if float(np.max(pred)) > m + pad or float(np.min(pred)) < mn - pad:
                    title += "  ⚠仿真出界"
                ax.set_title(title, fontsize=8)
        self.canvas.draw_idle()
        _p = PlanarParams().with_vector(self.phi)
        self.status.showMessage(
            "参数: " + " ".join(f"{n}={self.phi[i]:.5g}"
                                for i, n in enumerate(PARAM_NAMES[:6]))
            + f"  |  派生 J_b={float(_p.J_b):.5g} J_s={float(_p.J_s):.5g}"
              f" A={float(_p.A):.5g}  Δ下界={float(_p.Delta_min()):.4g}(>0 恒成立)"
            + f"  |  后端: {self._backend}")

    # ── 参数文件 ──
    def on_load_params(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "载入参数文件", "", "参数文本 (*.txt *.params);;所有文件 (*)")
        if not f:
            return
        try:
            self.phi = _parse_params_file(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "读取失败", str(exc))
            return
        for i, pr in enumerate(self.rows):
            pr.set_value(float(self.phi[i]))
        self.redraw()

    def on_save_params(self) -> None:
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存参数", "manual_params.txt", "参数文本 (*.txt);;所有文件 (*)")
        if not f:
            return
        write_params_file(f, self.phi,
                          recipe=f"手动标定（manual_tune.py）；{tau_sign_desc()}")
        self.status.showMessage(f"已写入 {f}", 5000)

    def on_reset(self) -> None:
        self.phi = default_param_vector()
        for i, pr in enumerate(self.rows):
            pr.set_value(float(self.phi[i]))
        self.redraw()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="手动标定参数（PyQt5 GUI）")
    ap.add_argument("--data", default=None,
                    help="初始数据: 目录 或 glob（目录取里面所有 npz/csv，按名排序）")
    ap.add_argument("--tau-sign-big", type=float, choices=(1.0, -1.0),
                    default=TAU_SIGN_BIG_DEFAULT,
                    help="★ τ_cmd（大 yaw 电机通道）符号: +1=原样（默认），-1=取反")
    ap.add_argument("--tau-sign-small", type=float, choices=(1.0, -1.0),
                    default=TAU_SIGN_SMALL_DEFAULT,
                    help="★ τ_small（小 yaw 通道）符号: -1=取反（**默认**），+1=原样")
    a, _ = ap.parse_known_args(argv)
    # ★ 必须在窗口构造（= 首次读数据）之前生效；GUI 里两个勾选框可实时切换
    set_tau_sign(big=a.tau_sign_big, small=a.tau_sign_small)
    app = QtWidgets.QApplication(sys.argv[:1])
    win = ManualTuneWindow(_collect_files(a.data) if a.data else [])
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
