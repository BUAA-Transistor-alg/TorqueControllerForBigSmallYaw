#!/usr/bin/env python3
"""mpc_demo.py — 双级 yaw 云台控制台示例（无需 pygame / 无图形依赖）

功能:
  用高层 ``TcbsRobotController``（= C++ 状态估计 + 耦合 MPC + 后台发送线程）让
  **小 yaw 世界方位角 ψ_small 做 3 s 周期的正弦**，而**大 yaw 世界方位角 ψ_big
  保持第一拍估计到的当前值**（即小 yaw 单独承担全部摆动，大 yaw 不跟着转）。
  每秒打印一次状态: 小 yaw 关节角、瞄准误差（目标方位角 − est.small_output_azimuth）、
  力矩、solve_ms、loop_fps，大 yaw 估计的**来源信息**（Provenance）与重力 A 系分量。

动力学模型（本版）:
  **平面 2 自由度 8 参模型**（include/tcbs/mpc/planar_yaw_model.h），8 个待辨识参数:
      Jbig_eff, Js, Px, Py, fc_big, fv_big, fc_small, fv_small
  加上实测几何 dx/dy/gravity/m_u_known 与固定项 frictionLambda/tau_offset_*。
  本脚本启动时打印当前参数，并支持命令行临时覆盖（``--model KEY=VALUE``，可重复）::

      --model Jbig_eff=0.024 --model Js=0.013 --model Px=0.001 --model fc_big=0.09
      # 摩擦参数两种写法都行: fc_big == fcBig

  pitch **不进入动力学**（平面模型不含 pitch 自由度），只作为下发给电控的目标角。

IMU 安装位置（可选开关）:
  ``--imu-location big``（默认）: IMU 固定在大 yaw 转子 A 上（现状，用 mount_* 标定）
  ``--imu-location head``        : IMU 装在头上（pitch 之后 H 系），配 ``--head-mount``
  切换只改变估计器内部的反解/重力/关节轴分支，所有对外字段语义不变（两种构型的
  对比与 ON_HEAD 的固有代价见 include/tcbs/communication/YawStateEstimator.h 顶部注释）。

语义提醒:
  * 目标接口用的是**世界方位角**（rad，多圈连续）:
        ψ_big   = 大 yaw 平台 x 轴的世界方位角（IMU 直测）
        ψ_small = 小 yaw 输出 x 轴的世界方位角 = ψ_big + θ_small
    ψ_small 与 ψ_big 之差就是小 yaw 关节角指令 θ_small（本脚本幅值默认 0.12 rad ≈ 6.9°，
    C++ 侧小 yaw 机械行程为 −25° ~ +20°（非对称，中心 −2.5°），且从两侧各向内 11.25° 起
    就加软限位代价（默认 ratio=0.75 ⇒ 软限位区 [−13.75°, +8.75°]）——所以默认幅值取在
    软限位区之内；要更大摆幅用 --amplitude 时请注意正侧只剩约 8.75° 的自由行程）。
  * 小 yaw 编码器（θ_small）是**可信实时量**；大 yaw 编码器有链路延迟，由估计器用
    IMU 角速度做延迟补偿 —— 打印里的 ``enc.age / prov.big_enc.delay_used``
    就是这条链路是否正常工作的判据。
  * **协议修订（MCU 端不提供时钟 + MCU2 同源）**: 大 yaw 与底盘 IMU 由 MCU2 一次送来，
    MCU 只给**一个**"新样本序号" ``mcu.mcu2_seq``（每次取到新数据 +1，值被保持时不变；
    判定新样本 = 首帧到达 || 序号变化 || 值变化，无 0 哨兵约定）；
    值的**年龄 age**、**新样本间隔 interval**、**采样过旧 stale** 全部由上位机计时。
  * ``est.gravity_a[3]`` 是**重力在 A 系（大 yaw 转子系）** 的投影（v4 起重命名；
    旧的 C 系字段 ``gravity_c`` 已删除）——底盘水平时其平面分量应 ≈ 0。

实车运行步骤:
  1) 构建动态库（含 C API）:
         cd /home/huhu233/rm2027/TorqueControllerForBigSmallYaw
         cmake -S . -B build && cmake --build build -j8 --target tcbs_robot_comm_c
  2) 串口权限（首次，插拔/重新登录后生效）:
         sudo usermod -aG dialout "$USER"
     MCU 与 IMU 由 udev 产品名自动识别（IMU 的产品名为 "AutoAim_IMU_Com"）。
  3) 运行（两条等价写法，任选）:
         # 让 Python 找到 torque_controller 包
         PYTHONPATH=python python3 python/scripts/mpc_demo.py --duration 30
         # 或直接进到 python/ 目录（脚本会自动把 ../ 加入 sys.path）
         cd python && python3 scripts/mpc_demo.py --duration 30
     库不在默认位置时显式指定:
         export TORQUE_BS_LIB=/abs/path/to/libtcbs_robot_comm_c.so
  4) 无硬件也能运行（本脚本会明确提示 valid=0）: 后台 loop 照常跑，只是不发数据、
     est.valid=0 → 力矩为 0、solve_ms=0、loop_fps 是 loop 实测频率。可用来验证环境与接口。
     接口级自检见 ``python/scripts/c_api_selftest.py``。

安全提示:
  * 首次在实车上跑请先架高云台/拆枪管，必要时用 ``--amplitude 0.05`` 小幅度试。
  * 模型参数默认值是**占位值**，实车前必须按 docs/calibration.md 标定；
    ``--model`` 只是临时覆盖，正式标定结果请写回 ``planar_yaw_params.h`` 或用标定脚本。
  * Ctrl-C 退出时脚本会关闭控制器（析构 → 停后台线程与串口），此后不再发送力矩。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# ── 导入 torque_controller（优先用已安装/PYTHONPATH 的包，否则用仓库内的 python/ 目录）──
try:
    from torque_controller import (
        IMU_ON_BIG_YAW,
        IMU_ON_HEAD,
        MODEL_PARAM_NAMES,
        TcbsDualYawModelParams,
        TcbsRobotController,
        TorqueControllerError,
        library_path,
        model_params_to_vector,
    )
except ImportError:  # 直接 `python3 python/scripts/mpc_demo.py` 时也能跑
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(_here, os.pardir)))
    from torque_controller import (
        IMU_ON_BIG_YAW,
        IMU_ON_HEAD,
        MODEL_PARAM_NAMES,
        TcbsDualYawModelParams,
        TcbsRobotController,
        TorqueControllerError,
        library_path,
        model_params_to_vector,
    )

# 摩擦参数的两种写法（辨识脚本里叫 fc_big，C 结构体里叫 fcBig）
_MODEL_ALIASES = {"fc_big": "fcBig", "fv_big": "fvBig",
                  "fc_small": "fcSmall", "fv_small": "fvSmall"}


def _bool(v) -> str:
    return "1" if v else "0"


def _age(age_s: float, zero_is_na: bool = False) -> str:
    """时间量格式化: <0 表示"从未收到"；zero_is_na 时 0 也表示"还没有两次样本"。"""
    if age_s < 0.0 or (zero_is_na and age_s <= 0.0):
        return "   n/a"
    return f"{age_s * 1e3:6.1f}ms"


def _src(name: str, s, tail: str = "") -> str:
    """一个数据源的紧凑描述（上位机计时的年龄 + 新样本/值保持计数 + 过旧标志）。"""
    return (f"{name}(valid={_bool(s.valid)} age={_age(s.age_s)} "
            f"n={s.count} new={s.new_samples} rej={s.rejected} stale={_bool(s.stale)}{tail})")


# ============================================================================
# 命令行 → 配置
# ============================================================================
def _parse_model_overrides(items) -> dict:
    """把 ``--model k=v`` 解析成 {ctypes 字段名: float}（未知字段立即报错）。"""
    valid = {name for name, *_ in TcbsDualYawModelParams._fields_}
    resolved = {}
    for item in items or ():
        if "=" not in item:
            raise TorqueControllerError(f"--model 需要 k=v 形式，收到: {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        field = _MODEL_ALIASES.get(key, key)
        if field not in valid:
            raise TorqueControllerError(
                f"--model 未知参数 {key!r}；可选: "
                + ", ".join(sorted(set(MODEL_PARAM_NAMES) | valid)))
        try:
            resolved[field] = float(raw)
        except ValueError:
            raise TorqueControllerError(f"--model {key}={raw} 不是合法数字")
    return resolved


def _parse_head_mount(text: str):
    """``--head-mount "yaw,pitch,roll"``（弧度；给 1~3 个分量，缺的补 0）。"""
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if not 1 <= len(parts) <= 3:
        raise TorqueControllerError("--head-mount 需要 1~3 个逗号分隔的弧度值")
    vals = [float(p) for p in parts]
    return (vals + [0.0, 0.0, 0.0])[:3]


def _apply_config(ctrl: TcbsRobotController, args) -> TcbsDualYawModelParams:
    """把命令行覆盖应用到控制器，返回生效后的模型参数。"""
    # 1) 估计器: IMU 安装位置（运行时开关: 大 yaw 转子 / 头）
    if args.imu_location == "head":
        mount = _parse_head_mount(args.head_mount)
        ctrl.set_estimator_config(imu_location=IMU_ON_HEAD,
                                  head_mount_yaw=mount[0],
                                  head_mount_pitch=mount[1],
                                  head_mount_roll=mount[2])
    else:
        ctrl.set_estimator_config(imu_location=IMU_ON_BIG_YAW)

    # 2) 模型: 8 参 + 几何的临时覆盖（一次下发；未给的字段用 C++ 默认值）
    overrides = _parse_model_overrides(args.model)
    if overrides:
        ctrl.set_model_params(**overrides)

    # 3) MPC: 可选覆盖（N / 迭代上限 / 积分器）
    mpc_kwargs = {}
    if args.mpc_n is not None:
        mpc_kwargs["N"] = int(args.mpc_n)
    if args.mpc_max_iter is not None:
        mpc_kwargs["max_iter"] = int(args.mpc_max_iter)
    if args.euler:
        mpc_kwargs["use_rk4"] = 0
    if mpc_kwargs:
        ctrl.set_mpc_config(**mpc_kwargs)
    return ctrl.get_model_params()


def _print_header(ctrl: TcbsRobotController, params: TcbsDualYawModelParams, args) -> None:
    est_cfg = ctrl.get_estimator_config()
    mpc_cfg = ctrl.get_mpc_config()
    vec = model_params_to_vector(params)
    loc = ("ON_HEAD（IMU 在头上）" if int(est_cfg.imu_location) == IMU_ON_HEAD
           else "ON_BIG_YAW（IMU 在大 yaw 转子上）")
    print("=" * 108)
    print("双级 yaw 云台 MPC 控制台示例 — 小 yaw 方位角正弦跟踪（大 yaw 方位角保持当前值）")
    print(f"  库: {library_path()}")
    print(f"  控制器模式: {ctrl.mode}（单点 set）    IMU 安装: {loc}")
    print(f"  正弦周期: {args.period:g} s   幅值: {args.amplitude:g} rad "
          f"({math.degrees(args.amplitude):.1f}°)   控制频率: {args.rate:g} Hz")
    print(f"  模型: 平面 2 自由度 8 参（几何 dx={params.dx:g}, dy={params.dy:g} m, "
          f"gravity={params.gravity:g} m/s², m_u_known={params.m_u_known:g} kg）")
    for i in range(0, len(MODEL_PARAM_NAMES), 4):
        chunk = "  ".join(f"{MODEL_PARAM_NAMES[j]}={vec[j]:.6g}"
                          for j in range(i, min(i + 4, len(vec))))
        print(f"       ★8 参[{i}..{min(i + 3, len(vec) - 1)}]: {chunk}")
    print(f"       frictionLambda={params.frictionLambda:g}  "
          f"tau_offset=({params.tau_offset_big:g}, {params.tau_offset_small:g}) N·m")
    print(f"  MPC: N={mpc_cfg.N} substeps={mpc_cfg.substeps} "
          f"{'RK4' if mpc_cfg.use_rk4 else '半隐式欧拉'} max_iter={mpc_cfg.max_iter} "
          f"dt={mpc_cfg.dt_control:g}s | max_torque big/small="
          f"{mpc_cfg.big.max_torque:g}/{mpc_cfg.small.max_torque:g} N·m | 小 yaw 限位=["
          f"{math.degrees(mpc_cfg.small.min_angle):+.1f}°, "
          f"{math.degrees(mpc_cfg.small.max_angle):+.1f}°]")
    print("  说明: MCU 端不提供时钟 + 大 yaw/底盘 IMU 同源（MCU2 一次送来）→ 只给一个"
          "新样本序号 mcu2_seq（值保持时不变），")
    print("        年龄(age)/新样本间隔(interval)/过旧(stale) 全部由上位机计时；"
          "est.gravity_a[3] 是重力在 **A 系（大 yaw 转子系）** 的投影。")
    print("=" * 108)


def _print_state(t: float, target_small: float, st, big_hold: float) -> None:
    est, mpc, mcu = st.est, st.mpc, st.mcu
    aim_err = target_small - est.small_output_azimuth     # 瞄准误差（方位角）
    prov = est.prov
    print(
        f"[{t:6.2f}s] ψ*_small={target_small:+.3f} θ_small={est.small_joint_angle:+.3f} "
        f"瞄准误差={aim_err:+.3f} rad({math.degrees(aim_err):+6.2f}°) "
        f"τ=({mpc.torque[0]:+.3f},{mpc.torque[1]:+.3f}) N·m "
        f"solve={mpc.solve_ms:5.2f}ms loop={mpc.loop_fps:6.1f}Hz "
        f"est.valid={_bool(est.valid)} sent={_bool(mpc.sent_ok)} "
        f"ticks_since_set={mpc.ticks_since_set}"
    )
    print(
        f"          大 yaw: ψ_big={est.platform_azimuth:+.3f}(hold {big_hold:+.3f}) "
        f"θ_big={est.big_joint_angle:+.3f} meas={est.big_joint_angle_meas:+.3f} "
        f"enc(valid={_bool(est.big_has_encoder)} age={_age(est.big_enc_age)} "
        f"interval={_age(est.big_sample_interval, zero_is_na=True)}) "
        f"innovation={est.big_enc_innovation:+.4f} rad"
    )
    print(
        f"          MCU2 序号: mcu2_seq={mcu.mcu2_seq}（大 yaw 与底盘 IMU 同源，值被保持时不变）  "
        f"chassis_imu(age={_age(est.chassis_imu_age)} n={prov.chassis_imu.count} "
        f"new={prov.chassis_imu.new_samples} stale={_bool(prov.chassis_imu.stale)})"
    )
    print(
        f"          来源: {_src('imu', prov.imu)} {_src('big_enc', prov.big_enc)} "
        f"{_src('small_enc', prov.small_enc)} | "
        f"{_src('pitch_enc', prov.pitch_enc)} {_src('chassis_imu', prov.chassis_imu)}"
    )
    print(
        f"          链路: rate_from_imu={_bool(prov.big_rate_from_imu)} "
        f"reverse_trusted={_bool(prov.reverse_from_trusted)} "
        f"delay(值的实测年龄)={prov.big_enc_delay_used * 1e3:6.1f}ms "
        f"interval={_age(prov.big_enc_interval_s, zero_is_na=True)} "
        f"sample_age={_age(prov.big_enc_sample_age_s, zero_is_na=True)} "
        f"used_mask=0x{prov.used_mask:02x}"
    )
    print(
        f"          外生量: g_A=({est.gravity_a[0]:+.3f},{est.gravity_a[1]:+.3f},"
        f"{est.gravity_a[2]:+.3f}) m/s²（A 系；平面分量≈0 表示底盘水平）  "
        f"ω_c=({est.base_omega[0]:+.3f},{est.base_omega[1]:+.3f},{est.base_omega[2]:+.3f}) "
        f"pitch_acc={est.pitch_acc:+.3f}"
    )


def run(args) -> int:
    with TcbsRobotController() as ctrl:
        params = _apply_config(ctrl, args)
        _print_header(ctrl, params, args)

        t0 = time.monotonic()
        period_inv = 1.0 / max(1e-6, args.rate)
        next_tick = t0
        next_print = 0.0
        big_hold = None          # 大 yaw 方位角保持值（第一拍取当前估计）
        warned = False
        last = None

        while True:
            now = time.monotonic()
            t = now - t0
            if args.duration > 0.0 and t >= args.duration:
                break

            # ── 反馈（世界方位角语义）──
            st = ctrl.get_state()
            last = st

            if big_hold is None:
                # 大 yaw 保持当前方位角：之后每拍都用这个常数作 ψ_big 目标
                big_hold = st.est.platform_azimuth

            if not warned and not st.est.valid:
                print("提示: 状态估计 valid=0（未收到 MCU/IMU 数据或大 yaw 无绝对基准）:"
                      " 力矩将为 0、solve_ms=0、MPC 不求解。若在实车上，请检查串口/权限/协议版本；"
                      " 纯离线调试时属正常（接口链路仍可用）。")
                warned = True

            # ── 目标: ψ_small = ψ_big(保持) + 幅值·sin(2πt/T) ──
            target_small = big_hold + args.amplitude * math.sin(2.0 * math.pi * t / args.period)
            # pitch: 平面模型不含 pitch 动力学，这里只把它作为下发给电控的目标角
            pitch_target = (args.pitch if args.pitch is not None else st.est.pitch_joint_angle)

            try:
                ctrl.set(
                    big_yaw_azimuth=big_hold,
                    small_yaw_azimuth=target_small,
                    pitch_target_angle=pitch_target,
                    auto_aim_enable=True,
                    big_torque_only=False,     # False → 力矩 + 电控位置/速度内环
                    small_torque_only=False,
                    fire=False,
                    integral_enable=False,
                )
            except TorqueControllerError as exc:
                print(f"set 失败: {exc}")
                return 2

            # ── 每秒打印一次状态 ──
            if t >= next_print:
                next_print += args.print_period
                _print_state(t, target_small, ctrl.get_state(), big_hold)

            next_tick += period_inv
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()   # 落后了就重新对齐（避免追帧雪崩）

        if last is not None and big_hold is not None:
            print("-" * 108)
            print("结束: 最后一次状态")
            _print_state(time.monotonic() - t0, big_hold, last, big_hold)
        print("控制器已关闭（后台 MPC 线程与串口线程停止）。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="双级 yaw 云台 MPC 控制台示例: 小 yaw 方位角正弦跟踪（大 yaw 保持）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--duration", type=float, default=15.0,
                   help="运行时长 s（<=0 表示一直跑到 Ctrl-C）")
    p.add_argument("--period", type=float, default=3.0, help="小 yaw 方位角正弦周期 s")
    p.add_argument("--amplitude", type=float, default=0.12,
                   help="正弦幅值 rad（默认 0.12≈6.9°: 小 yaw 行程 −25°~+20°、正侧软限位 +8.75°; 实车先小幅度试）")
    p.add_argument("--rate", type=float, default=100.0, help="set() 调用频率 Hz")
    p.add_argument("--pitch", type=float, default=None,
                   help="pitch 目标角 rad（默认: 保持当前估计值）")
    p.add_argument("--print-period", type=float, default=1.0, help="状态打印周期 s")
    # ── 平面 8 参模型（临时覆盖；正式标定后写回 planar_yaw_params.h）──
    p.add_argument("--model", action="append", metavar="KEY=VALUE", default=None,
                   help="临时覆盖模型参数，可重复。键: " + ", ".join(MODEL_PARAM_NAMES)
                        + " 以及 dx/dy/gravity/m_u_known/frictionLambda/tau_offset_big/"
                          "tau_offset_small（fc_big 与 fcBig 等价）")
    # ── MPC ──
    p.add_argument("--mpc-n", type=int, default=None, help="覆盖 MPC 预测步数 N")
    p.add_argument("--mpc-max-iter", type=int, default=None, help="覆盖 Ceres 迭代上限")
    p.add_argument("--euler", action="store_true",
                   help="积分器改用半隐式欧拉（默认 RK4；更快、精度略低）")
    # ── IMU 安装位置（运行时开关）──
    p.add_argument("--imu-location", choices=("big", "head"), default="big",
                   help="IMU 安装位置: big = 大 yaw 转子上（现状），head = 头上（pitch 之后）")
    p.add_argument("--head-mount", default="0,0,0", metavar="YAW,PITCH,ROLL",
                   help="ON_HEAD 时的 R_H_IMU 安装角（弧度，ZXY）；仅 --imu-location head 生效")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，退出（控制器已析构，不再发送指令）。")
        return 0
    except TorqueControllerError as exc:
        print(f"错误: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
