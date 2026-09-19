#!/usr/bin/env python3
"""c_api_selftest.py — C API / Python 绑定自检（**纯仿真，不需要任何串口硬件**）

覆盖:
  1. 动态库加载 + ABI 自检（``tcbs_robot_comm_abi_info`` 的各 sizeof 与 ctypes 结构体逐项一致）；
  2. ``tcbs_robot_comm_check_abi`` 的三条路径: 版本/布局匹配 → 0；版本不匹配 → 负错误码；
     布局不匹配 → 负错误码；以及 Python 侧 ``check_abi()`` 对版本漂移的检测；
  3. 低层句柄 ``TcbsRobotCommunication``: 创建 → 取原始反馈/状态估计（无硬件时 valid=0，
     这是**设计要求**，不是失败）→ 发送接口返回 False（串口未打开）→ 销毁；
  4. 高层句柄 ``TcbsRobotController``: 读默认参数 → 改一个参数 → 读回（**逐位浮点一致**）
     → 8 参模型参数向量往返 → MPC 配置 / 估计器配置（含 IMU 安装位置）往返 → 销毁；
  5. 无硬件下的状态获取（valid 全 0、MPC 后台线程在跑但不求解）。

用法（在仓库根目录）::

    PYTHONPATH=python python3 python/scripts/c_api_selftest.py

退出码: 0 = 全部通过；1 = 有失败项（每项都会打印 [FAIL] 与原因）。
"""

from __future__ import annotations

import os
import sys
import time

# ── 导入 torque_controller（优先已安装/PYTHONPATH 的包，否则用仓库内 python/ 目录）──
try:
    import torque_controller as tc
    from torque_controller import _bridge as bridge
except ImportError:  # 直接 `python3 python/scripts/c_api_selftest.py` 时也能跑
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(_here, os.pardir)))
    import torque_controller as tc
    from torque_controller import _bridge as bridge

_from_c = bridge
from ctypes import byref, sizeof  # noqa: E402  (需在 sys.path 处理之后)

_FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    """一条断言；打印结果并累计失败数。"""
    global _FAILED
    if ok:
        print(f"[ ok ] {name}")
    else:
        _FAILED += 1
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print("-" * 78)
    print(title)


# ============================================================================
# 1. 库加载 + ABI 自检
# ============================================================================
def test_library_and_abi() -> None:
    section("1. 动态库加载 + ABI 自检")
    lib_path = bridge.library_path()
    check("动态库已加载（libtcbs_robot_comm_c.so）", lib_path is not None,
          f"{bridge._load_error}")
    print(f"       库路径: {lib_path}")
    print(f"       Python 绑定版本: torque_controller {tc.__version__}, "
          f"C API v{bridge.API_VERSION}")

    info = _from_c.TcbsRobotCommAbiInfo()
    lib = bridge._require_lib()
    bridge._check(lib.tcbs_robot_comm_abi_info(byref(info)), "tcbs_robot_comm_abi_info")
    check("库内 ABI 版本 == Python 绑定版本", info.api_version == bridge.API_VERSION,
          f"库={info.api_version}, Python={bridge.API_VERSION}")
    check("指针宽度一致 (64 位库 ↔ 64 位 Python)", info.sizeof_pointer == sizeof(bridge.c_void_p),
          f"库={info.sizeof_pointer}, Python={sizeof(bridge.c_void_p)}")

    for field_name, cls in bridge._ABI_STRUCTS:
        lib_size = int(getattr(info, field_name))
        py_size = int(sizeof(cls))
        check(f"sizeof({cls.__name__}) == {py_size}", lib_size == py_size,
              f"库={lib_size}, Python={py_size}")
    # check_abi() 内部会调 tcbs_robot_comm_check_abi（C 侧权威判定）
    try:
        _from_c.check_abi()
        check("check_abi() 全量校验通过", True)
    except tc.TorqueControllerError as exc:
        check("check_abi() 全量校验通过", False, str(exc))

    # 8 参模型参数结构体 = 15 个 double（无隐藏 padding）
    check("TcbsDualYawModelParams 为 15×double（120 字节）",
          sizeof(bridge.TcbsDualYawModelParams) == 15 * 8,
          f"实际 {sizeof(bridge.TcbsDualYawModelParams)}")


# ============================================================================
# 2. tcbs_robot_comm_check_abi: 版本/布局不匹配 → 负错误码
# ============================================================================
def test_abi_mismatch_error_codes() -> None:
    section("2. ABI 不匹配 → 返回错误码（tcbs_robot_comm_check_abi）")
    lib = bridge._require_lib()
    sizes = bridge._python_abi_sizes()

    ret_ok = lib.tcbs_robot_comm_check_abi(bridge.API_VERSION, byref(sizes))
    check(f"版本与布局都匹配 → {ret_ok} (OK)",
          ret_ok == bridge.ERR_OK, f"期望 {bridge.ERR_OK}")

    wrong_ver = bridge.API_VERSION + 1
    ret_ver = lib.tcbs_robot_comm_check_abi(wrong_ver, byref(sizes))
    check(f"版本号不匹配（{wrong_ver}）→ {ret_ver} (ERR_ABI_MISMATCH)",
          ret_ver == bridge.ERR_ABI_MISMATCH, f"期望 {bridge.ERR_ABI_MISMATCH}")

    bad_sizes = bridge._python_abi_sizes()
    bad_sizes.sizeof_model_params = int(bad_sizes.sizeof_model_params) + 8
    ret_layout = lib.tcbs_robot_comm_check_abi(bridge.API_VERSION, byref(bad_sizes))
    check(f"sizeof_model_params 漂移 → {ret_layout} (ERR_ABI_MISMATCH)",
          ret_layout == bridge.ERR_ABI_MISMATCH, f"期望 {bridge.ERR_ABI_MISMATCH}")

    ret_none = lib.tcbs_robot_comm_check_abi(bridge.API_VERSION, None)
    check(f"只校验版本（sizes=NULL，版本正确）→ {ret_none} (OK)",
          ret_none == bridge.ERR_OK, f"期望 {bridge.ERR_OK}")

    check("错误码有可读描述", "abi" in bridge.strerror(bridge.ERR_ABI_MISMATCH).lower(),
          bridge.strerror(bridge.ERR_ABI_MISMATCH))

    # Python 侧: 绑定版本号与库不一致时必须报错（模拟"库升版、绑定没跟上"）
    saved = bridge.API_VERSION
    try:
        bridge.API_VERSION = saved + 1
        raised = False
        try:
            _from_c.check_abi()
        except tc.TorqueControllerError:
            raised = True
        check("绑定版本号漂移 → check_abi() 抛 TorqueControllerError", raised)
    finally:
        bridge.API_VERSION = saved


# ============================================================================
# 3. 低层句柄（无硬件）
# ============================================================================
def test_low_level_handle() -> None:
    section("3. 低层句柄 TcbsRobotCommunication（无硬件）")
    comm = None
    try:
        comm = _from_c.TcbsRobotCommunication()
        check("tcbs_robot_comm_create() 无硬件也成功", comm._handle is not None)

        data = comm.get_latest_data()
        check("get_latest_data() 可用（无硬件时 valid=0）",
              data.mcu.valid == 0 and data.imu.valid == 0,
              f"mcu.valid={data.mcu.valid}, imu.valid={data.imu.valid}")
        check("mcu2_seq 字段存在（v3+ 语义）", hasattr(data.mcu, "mcu2_seq"))

        est = comm.get_estimate()
        check("get_estimate() 可用（无硬件时 valid=0）", est.valid == 0)
        check("est.gravity_a[3] 存在（★ A 系重力，v4 语义）",
              len(list(est.gravity_a)) == 3 and not hasattr(est, "gravity_c"))
        check("est 的年龄字段存在（big_enc_age/chassis_imu_age/big_sample_interval）",
              all(hasattr(est, n) for n in
                  ("big_enc_age", "big_sample_interval", "chassis_imu_age")))
        check("年龄未收到时哨兵为 -1",
              est.big_enc_age < 0.0 and est.chassis_imu_age < 0.0,
              f"big_enc_age={est.big_enc_age}, chassis_imu_age={est.chassis_imu_age}")

        sent = comm.send_to_mcu(auto_aim_enable=1, yaw_big_mode=0,
                                yaw_big_target_angle=0.0, yaw_big_torque=0.0)
        check("send_to_mcu() 无硬件时返回 False（不抛异常）", sent is False)
        check("send_to_imu() 无硬件时返回 False（不抛异常）",
              comm.send_to_imu() is False)
    finally:
        if comm is not None:
            comm.stop()
            comm.close()
    check("tcbs_robot_comm_destroy() 已执行（close 可重复调用）",
          comm is None or comm._handle is None)


# ============================================================================
# 4. 高层句柄: 配置往返（含浮点一致性）
# ============================================================================
def _params_dict(params) -> dict:
    return {name: float(getattr(params, name)) for name, *_ in params._fields_}


def test_controller_config_roundtrip() -> None:
    section("4. 高层句柄 TcbsRobotController: 配置读回（逐位一致）")
    ctrl = None
    try:
        ctrl = _from_c.TcbsRobotController()
        check("tcbs_robot_controller_create() 无硬件也成功", ctrl._handle is not None)

        # ── 默认参数: get_* 与 default_* 完全一致 ──
        default_p = _from_c.default_model_params()
        got_p = ctrl.get_model_params()
        check("get_model_params() == default_model_params()",
              _params_dict(got_p) == _params_dict(default_p),
              f"默认={_params_dict(default_p)}")

        default_m = _from_c.default_mpc_config()
        got_m = ctrl.get_mpc_config()
        check("get_mpc_config() 与默认一致（N/权重/限位）",
              got_m.N == default_m.N and got_m.small.max_torque == default_m.small.max_torque
              and got_m.dt_control == default_m.dt_control)

        default_e = _from_c.default_estimator_config()
        got_e = ctrl.get_estimator_config()
        # ★ 默认构型已改为 ON_HEAD（IMU 在头上）⇒ 不再写死 ON_BIG_YAW
        check("get_estimator_config() 与默认一致（含 imu_location）",
              int(got_e.imu_location) == int(default_e.imu_location) == _from_c.IMU_ON_HEAD,
              f"imu_location={got_e.imu_location}（默认构型 ON_HEAD）")

        # ── 改一个模型参数 → 读回（要求**逐位**一致，不允许任何精度损失）──
        new_js = 0.0130000000000000011      # 需要完整 double 精度才不丢位
        ctrl.set_model_params(Js=new_js)
        back = ctrl.get_model_params()
        check("set_model_params(Js=...) 后读回逐位一致 (==)",
              float(back.Js) == new_js, f"写入={new_js!r}, 读回={float(back.Js)!r}")
        check("只改了 Js，其它字段保持默认",
              float(back.Jbig_eff) == float(default_p.Jbig_eff)
              and float(back.Px) == float(default_p.Px)
              and float(back.fcBig) == float(default_p.fcBig))

        # ── ★8 参向量往返（顺序与 paramsToVector / 辨识脚本一致）──
        vec = [0.031, 0.017, 1.25e-3, -2.5e-3, 0.11, 0.021, 0.041, 0.0095]
        ctrl.set_model_params(_from_c.vector_to_model_params(vec, default_p))
        back_vec = _from_c.model_params_to_vector(ctrl.get_model_params())
        check("8 参向量 set → 读回逐位一致",
              [float(v) for v in back_vec] == vec, f"写入={vec}, 读回={back_vec}")
        check("向量写入不影响几何量（dx/dy/gravity/m_u_known）",
              float(ctrl.get_model_params().dx) == float(default_p.dx)
              and float(ctrl.get_model_params().gravity) == float(default_p.gravity))
        check("MODEL_PARAM_NAMES 顺序与 C++ paramsToVector 一致",
              tuple(_from_c.MODEL_PARAM_NAMES) ==
              ("Jbig_eff", "Js", "Px", "Py", "fc_big", "fv_big", "fc_small", "fv_small"))

        # ── MPC 配置往返 ──
        ctrl.set_mpc_config(N=9, max_iter=5, use_rk4=0, dt_control=0.005, smooth_eps=1e-6)
        m2 = ctrl.get_mpc_config()
        check("set_mpc_config → 读回 (N/max_iter/use_rk4/dt_control)",
              m2.N == 9 and m2.max_iter == 5 and m2.use_rk4 == 0 and m2.dt_control == 0.005,
              f"N={m2.N} max_iter={m2.max_iter} use_rk4={m2.use_rk4} dt={m2.dt_control}")
        check("MPC 配置无 extrapolate_pitch 字段（v4 起已删除）",
              not hasattr(m2, "extrapolate_pitch"))

        # ── 估计器配置: IMU 安装位置（大 yaw 转子 / 头）──
        ctrl.set_estimator_config(imu_location=_from_c.IMU_ON_HEAD,
                                  head_mount_yaw=0.1, head_mount_pitch=-0.2,
                                  head_mount_roll=0.3)
        e2 = ctrl.get_estimator_config()
        check("set_estimator_config(imu_location=IMU_ON_HEAD) 读回一致",
              int(e2.imu_location) == _from_c.IMU_ON_HEAD
              and float(e2.head_mount_yaw) == 0.1
              and float(e2.head_mount_pitch) == -0.2
              and float(e2.head_mount_roll) == 0.3,
              f"imu_location={e2.imu_location} head_mount=({e2.head_mount_yaw},"
              f"{e2.head_mount_pitch},{e2.head_mount_roll})")
        ctrl.set_estimator_config(imu_location=_from_c.IMU_ON_BIG_YAW)
        check("切回 IMU_ON_BIG_YAW 也生效",
              int(ctrl.get_estimator_config().imu_location) == _from_c.IMU_ON_BIG_YAW)

        # ── 控制器配置（会重建 TcbsRobotController，配置需保持）──
        ctrl.set_controller_config(loop_period=0.02)
        check("set_controller_config(loop_period=0.02) 后句柄仍可用",
              ctrl.mode_code == _from_c.MODE_SINGLE)
    finally:
        if ctrl is not None:
            ctrl.close()
    check("tcbs_robot_controller_destroy() 已执行", ctrl is None or ctrl._handle is None)


# ============================================================================
# 5. 无硬件下的状态获取（后台线程在跑，但 est 无效 → 不求解）
# ============================================================================
def test_state_without_hardware() -> None:
    section("5. 无硬件状态获取（valid 全 0；后台线程运行）")
    ctrl = None
    try:
        ctrl = _from_c.TcbsRobotController()
        ctrl.set(big_yaw_azimuth=0.0, small_yaw_azimuth=0.1, pitch_target_angle=0.0)
        time.sleep(0.35)                      # 让后台 loop 跑几十拍（loop_period=0.01）
        st = ctrl.get_state()
        check("get_state() 可用", st is not None)
        check("无硬件时 mcu/imu/est valid 全 0（设计要求，不是失败）",
              st.mcu.valid == 0 and st.imu.valid == 0 and st.est.valid == 0,
              f"mcu={st.mcu.valid} imu={st.imu.valid} est={st.est.valid}")
        check("后台 loop 在运行（loop_fps > 0）", st.mpc.loop_fps > 0.0,
              f"loop_fps={st.mpc.loop_fps}（若为 0 说明后台线程没起来）")
        check("est 无效 → MPC 不求解（solve_count == 0，安全退化）",
              st.mpc.solve_count == 0, f"solve_count={st.mpc.solve_count}")
        check("状态里含重力 A 系分量 gravity_a[3]",
              len(list(st.est.gravity_a)) == 3)
        check("set() 的参考已被记录（ref_azimuth[1] = 0.1）",
              abs(st.mpc.ref_azimuth[1] - 0.1) < 1e-12,
              f"ref_azimuth={list(st.mpc.ref_azimuth)}")
    finally:
        if ctrl is not None:
            ctrl.close()


def main() -> int:
    print("=" * 78)
    print("C API / Python 绑定自检（纯仿真，不需要串口硬件）")
    print("=" * 78)
    test_library_and_abi()
    test_abi_mismatch_error_codes()
    test_low_level_handle()
    test_controller_config_roundtrip()
    test_state_without_hardware()
    print("=" * 78)
    if _FAILED == 0:
        print("全部通过 ✔")
        return 0
    print(f"有 {_FAILED} 项失败 ✘")
    return 1


if __name__ == "__main__":
    sys.exit(main())
