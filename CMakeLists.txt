cmake_minimum_required(VERSION 3.10)

# ============================================================================
# TorqueControllerForBigSmallYaw —— 双级 yaw（大 yaw + 小 yaw）力矩控制子模组
#
# 模块标识（pfx）= **tcbs**（TorqueController Big/Small），统一封装约定见
# docs/embedding.md。要点:
#   * target 名 / 产物文件名 / ctest 名一律带 tcbs_ 前缀，库另提供
#     tcbs::<原名> 形式的 ALIAS，供父工程 target_link_libraries 使用；
#   * 头文件物理位于 include/tcbs/...，对内外一律通过 "tcbs/..." 访问，
#     因此 target_include_directories 只暴露 include 这一层（不再暴露
#     include/communication、include/common 等二级目录）；
#   * 所有 C++ 代码在 namespace tcbs 内，C ABI 的符号/类型/宏带 tcbs_ / Tcbs 前缀；
#   * 本文件既可**独立**构建（bash build.sh），也可被父工程以
#         add_subdirectory(<本目录> <binary_dir> EXCLUDE_FROM_ALL)
#     引入。作为子目录时**不**调用 project() / enable_testing()，也**不**使用
#     任何会污染父作用域的全局命令（CMAKE_* 全局设置 / add_compile_options /
#     include_directories），全部改用 target_* 形式。
# ============================================================================

set(TCBS_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TCBS_IS_TOP_LEVEL ON)
endif()

if(TCBS_IS_TOP_LEVEL)
    project(TorqueControllerForBigSmallYaw LANGUAGES CXX)
    # 独立构建时保持原有行为（默认 Release + -O2）；作为子目录时完全沿用父工程配置
    # （以下 set 均为**目录作用域**变量，不会影响父工程与本目录之外的 target）
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    enable_testing()
endif()

# ============================================================================
# 依赖
# ============================================================================
find_library(TCBS_UDEV_LIBRARY udev)
if(NOT TCBS_UDEV_LIBRARY)
    message(FATAL_ERROR "libudev not found")
endif()
find_package(Ceres REQUIRED)

# 本仓库对外暴露的唯一 include 根目录（第二级目录名不再出现在搜索路径里）
set(TCBS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# 通信库（串口协议 + 编码器映射 + 状态估计）
# ============================================================================
set(TCBS_COMMUNICATION_SOURCES
    src/communication/CRC.cpp
    src/communication/McuDataPreprocessor.cpp
    src/communication/YawStateEstimator.cpp
)

add_library(tcbs_communication STATIC ${TCBS_COMMUNICATION_SOURCES})
add_library(tcbs::communication ALIAS tcbs_communication)
set_target_properties(tcbs_communication PROPERTIES OUTPUT_NAME tcbs_communication)
target_include_directories(tcbs_communication PUBLIC ${TCBS_INCLUDE_DIR})
target_compile_features(tcbs_communication PUBLIC cxx_std_17)
target_link_libraries(tcbs_communication PUBLIC ${TCBS_UDEV_LIBRARY})

# ============================================================================
# MPC + 一体化封装（动态库，供外部 cpp 与 Python ctypes 使用）
# ============================================================================
set(TCBS_MPC_SOURCES
    src/mpc/dual_yaw_mpc.cpp
    src/mpc/mcu_mpc_controller.cpp
    src/common/FrameRateCounter.cpp
)

add_library(tcbs_robot_comm_c SHARED
    ${TCBS_COMMUNICATION_SOURCES}
    ${TCBS_MPC_SOURCES}
    src/RobotController.cpp
    src/c_api/RobotCommunicationC.cpp   # C ABI（v4: 平面 8 参模型）
)
add_library(tcbs::robot_comm_c ALIAS tcbs_robot_comm_c)
set_target_properties(tcbs_robot_comm_c PROPERTIES OUTPUT_NAME tcbs_robot_comm_c)
target_include_directories(tcbs_robot_comm_c PUBLIC
    ${TCBS_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)
target_compile_features(tcbs_robot_comm_c PUBLIC cxx_std_17)
target_link_libraries(tcbs_robot_comm_c PUBLIC ${TCBS_UDEV_LIBRARY} ${CERES_LIBRARIES})

# ============================================================================
# 测试（ctest 名带 tcbs_ 前缀；父工程 enable_testing() 后不会与其它子模组撞名）
# ============================================================================
add_executable(tcbs_test_planar_yaw_model tests/test_planar_yaw_model.cpp)
set_target_properties(tcbs_test_planar_yaw_model PROPERTIES
    OUTPUT_NAME tcbs_test_planar_yaw_model)
target_include_directories(tcbs_test_planar_yaw_model PRIVATE ${TCBS_INCLUDE_DIR})
target_compile_features(tcbs_test_planar_yaw_model PRIVATE cxx_std_17)
add_test(NAME tcbs_model_verification COMMAND tcbs_test_planar_yaw_model)

add_executable(tcbs_test_yaw_state_estimator tests/test_yaw_state_estimator.cpp)
set_target_properties(tcbs_test_yaw_state_estimator PROPERTIES
    OUTPUT_NAME tcbs_test_yaw_state_estimator)
target_include_directories(tcbs_test_yaw_state_estimator PRIVATE ${TCBS_INCLUDE_DIR})
target_compile_features(tcbs_test_yaw_state_estimator PRIVATE cxx_std_17)
target_link_libraries(tcbs_test_yaw_state_estimator PRIVATE tcbs::communication)
add_test(NAME tcbs_state_estimator COMMAND tcbs_test_yaw_state_estimator)

add_executable(tcbs_test_dual_yaw_mpc tests/test_dual_yaw_mpc.cpp ${TCBS_MPC_SOURCES})
set_target_properties(tcbs_test_dual_yaw_mpc PROPERTIES OUTPUT_NAME tcbs_test_dual_yaw_mpc)
target_include_directories(tcbs_test_dual_yaw_mpc PRIVATE
    ${TCBS_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)
target_compile_features(tcbs_test_dual_yaw_mpc PRIVATE cxx_std_17)
target_link_libraries(tcbs_test_dual_yaw_mpc PRIVATE tcbs::communication ${CERES_LIBRARIES})
add_test(NAME tcbs_mpc_closed_loop COMMAND tcbs_test_dual_yaw_mpc)

# 参数传递检查: 用逐项非默认的 RobotController::Config 构造，再从各子系统 getter 读回比对。
# 目的: ① 证明"全部标定参数都能在构造时传入"；② 防止将来往 Config 加字段却忘了传下去。
add_executable(tcbs_test_config_propagation tests/test_config_propagation.cpp
    src/RobotController.cpp
    ${TCBS_MPC_SOURCES}
    src/common/FrameRateCounter.cpp
)
set_target_properties(tcbs_test_config_propagation PROPERTIES
    OUTPUT_NAME tcbs_test_config_propagation)
target_include_directories(tcbs_test_config_propagation PRIVATE
    ${TCBS_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)
target_compile_features(tcbs_test_config_propagation PRIVATE cxx_std_17)
target_link_libraries(tcbs_test_config_propagation PRIVATE
    tcbs::communication ${CERES_LIBRARIES})
add_test(NAME tcbs_config_propagation COMMAND tcbs_test_config_propagation)

# ============================================================================
# 示例 / 工具
# ============================================================================
add_executable(tcbs_control_demo src/control_demo.cpp)
set_target_properties(tcbs_control_demo PROPERTIES OUTPUT_NAME tcbs_control_demo)
target_compile_features(tcbs_control_demo PRIVATE cxx_std_17)
target_link_libraries(tcbs_control_demo PRIVATE tcbs::robot_comm_c)

# 采集工具改为 Python（复用录制目标序列 + 增强 + PID，与原仓库一致）:
#   python/scripts/collect_sysid.py

add_executable(tcbs_identify_params tools/identify_params.cpp)
set_target_properties(tcbs_identify_params PROPERTIES OUTPUT_NAME tcbs_identify_params)
target_compile_features(tcbs_identify_params PRIVATE cxx_std_17)
target_link_libraries(tcbs_identify_params PRIVATE tcbs::robot_comm_c)

# pitch 轴映射标定（两段线性拟合，需临时把 IMU 装到头上 = ON_HEAD 构型）:
#   ./build/tcbs_pitch_calibration --sim        # 无硬件自检（虚拟台架 + 断言）
#   ./build/tcbs_pitch_calibration --selftest   # 纯数学自检
#   实车用法与安全约定见 tools/pitch_calibration.cpp 文件头与 docs/calibration.md §3.3
add_executable(tcbs_pitch_calibration tools/pitch_calibration.cpp)
set_target_properties(tcbs_pitch_calibration PROPERTIES OUTPUT_NAME tcbs_pitch_calibration)
target_compile_features(tcbs_pitch_calibration PRIVATE cxx_std_17)
target_link_libraries(tcbs_pitch_calibration PRIVATE tcbs::communication)
# ⚠ 关于 `add_library(tcbs::pitch_calibration ALIAS ...)`: CMake **不支持**给 add_executable
#   目标建 ALIAS —— `add_library(<name> ALIAS <target>)` 要求 <target> 必须是库，
#   对可执行文件实测报错 "add_library cannot create ALIAS target ... because target
#   ... is not a library"（CMake 3.22，且没有版本提供该能力）。
#   本工具是**可执行文件**，父工程不需要"链接"它、只需"确保它被构建"，故约定用
#       add_dependencies(<父工程 target> tcbs_pitch_calibration)
#   （对外名字仍带 tcbs_ 前缀；可执行文件从不进 target_link_libraries。）

# 串口链路自检（移植自原仓库 src/test_serial.cpp → tcs_test_serial）:
#   ./build/tcbs_test_serial --list       # 枚举串口 + 打印 MCU/IMU 选择结果（无需硬件）
#   ./build/tcbs_test_serial --selftest   # 纯软件自检（包布局/CRC/安全不变量）
#   ./build/tcbs_test_serial              # 实车: 发零力矩帧 + 打印收到的每帧
# ⚠ 名字里带 test 但**不是 ctest 用例**（不 add_test）: 它是实车链路工具，
#   不依赖 gtest / 断言框架，靠自身退出码表达成败（0/1/2/130）。
#   与 tcbs_test_* 三个 ctest 程序的命名区分见 docs/calibration.md §3.0。
add_executable(tcbs_test_serial tools/test_serial.cpp)
set_target_properties(tcbs_test_serial PROPERTIES OUTPUT_NAME tcbs_test_serial)
target_compile_features(tcbs_test_serial PRIVATE cxx_std_17)
target_link_libraries(tcbs_test_serial PRIVATE tcbs::communication)

# MPC 参数评估工具: 用辨识出来的参数跑闭环，比较不同 (模型 λ, 积分子步) 的表现
add_executable(tcbs_mpc_param_eval tools/mpc_param_eval.cpp ${TCBS_MPC_SOURCES})
set_target_properties(tcbs_mpc_param_eval PROPERTIES OUTPUT_NAME tcbs_mpc_param_eval)
target_compile_features(tcbs_mpc_param_eval PRIVATE cxx_std_17)
target_link_libraries(tcbs_mpc_param_eval PRIVATE tcbs::robot_comm_c)
