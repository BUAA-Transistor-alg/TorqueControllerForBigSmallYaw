# 把子模组封装成"可与其它子模组共存"的规范（嵌入 `UnifiedAutoAimPipeline` 用）

**问题**: 两个同源仓库（`TorqueController` 单 yaw 版 / `TorqueControllerForBigSmallYaw` 双级版）
放进同一个 CMake 工程时会直接冲突 —— 实测冲突项:

| 冲突类别 | 实测证据 | 后果 |
|---|---|---|
| CMake target 重名 | 两者都有 `communication`、`robot_comm_c`、`control_demo` | **configure 阶段硬报错** |
| 产物文件名重名 | 两者都产出 `librobot_comm_c.so` | 互相覆盖 / 链接到错的那个 |
| C 符号重名 | `robot_comm_create/destroy/get_latest_data/send_to_imu/send_to_mcu/stop`（6 个） | 同一进程加载两个 .so ⇒ **符号被抢占**，静默用到错实现 |
| C++ 符号重名且布局不同 | 两者都导出 `_ZN15RobotController8getStateEv`，但 `RobotController::State` 字段不同 | **静默内存错乱（UB）**，比编译错误危险 |
| 头文件路径重名 | `RobotController.h`、`communication/{Protocol,Communications,McuDataPreprocessor,CRC,SerialProtocol}.h(pp)`、`mpc/mcu_mpc_controller.h`、`c_api/RobotCommunicationC.h`、`common/FrameRateCounter.h`（9 个，内容不同） | `#include "RobotController.h"` 取到哪个取决于 include 顺序；甚至与父工程自己的 `include/common/FrameRateCounter.h` 撞名 |
| 全局命名空间重名 | 两者都有 `namespace imu`、`namespace mcu` | 同一进程内的 ODR 违规 |

**解决办法**: 每个子模组取一个**唯一的模块短标识**（pfx），把自己**所有对外名字**都带上它，
并把 C++ 代码整体收进 `namespace <pfx>`。这样两个子模组可以：
同一个 CMake 工程里 configure / 编译 / 链接、同一个进程里同时加载。

> 本仓库 pfx = **`tcbs`**（TorqueController Big/Small）。
> 单 yaw 原仓库建议 pfx = **`tcs`**；以后新加子模组照同样的规矩各取一个（`tcgimbal`、`tcpitch`…）。

---

## 1. 约定（照着抄就行）

| 类别 | 规则 | 本仓库（`tcbs`）示例 |
|---|---|---|
| CMake target | `<pfx>_<原名>` + `add_library(<pfx>::<原名> ALIAS ...)` | `tcbs_robot_comm_c` / alias `tcbs::robot_comm_c` |
| 产物文件名 | `set_target_properties(<tgt> PROPERTIES OUTPUT_NAME <pfx>_<原名>)` | `libtcbs_robot_comm_c.so` |
| ctest 名 | `add_test(NAME <pfx>_<原名> ...)` | `tcbs_mpc_closed_loop` |
| C 函数名 | `<pfx>_` 前缀 | `tcbs_robot_comm_create` |
| C 类型名 | `Tcbs` 驼峰前缀 | `TcbsRobotEstimate_C` |
| C 宏名 | `<PFX>_` 大写前缀 | `TCBS_ROBOT_COMM_C_API_VERSION` |
| 头文件路径 | 物理放到 `include/<pfx>/...`，仓库内外 include 都带前缀 | `#include "tcbs/RobotController.h"` |
| C++ 命名空间 | **全部** C++ 代码包进 `namespace <pfx> { ... }`（原有子命名空间**嵌进去**） | `tcbs::RobotController`、`tcbs::dual_yaw::ModelParams`、`tcbs::rot::Mat3`、`tcbs::mcu::SendPacket` |
| C API 头 | **例外**: 保持合法 C11 / `extern "C"`，**不要**放进 C++ 命名空间，只做前缀化 | `include/tcbs/c_api/RobotCommunicationC.h` |
| **include guard** | **也必须加前缀**（容易漏！） | `ROBOT_CONTROLLER_H` → `TCBS_ROBOT_CONTROLLER_H` |
| CMake cache 变量 | `find_library`/`find_package` 结果等变量加 pfx（父工程里是全局 cache） | `UDEV_LIBRARY` → `TCBS_UDEV_LIBRARY` |

---

## 2. 落地步骤（机械操作，约 1~2 小时/仓库）

1. **搬迁头文件**：`include/*` → `include/<pfx>/*`（内部结构不动）。
2. **改 include 前缀**：仓库内所有 `#include "RobotController.h"` / `"mpc/xxx.h"` / `"communication/xxx.hpp"`
   → `#include "<pfx>/..."`。用 grep 收尾核对：
   ```bash
   grep -rn '#include "' src tests tools include | grep -v "<pfx>/" | grep -v '^.*<std\|^.*<Eigen\|^.*<ceres'
   ```
   应该是空（第三方头除外）。
3. **包命名空间 + 改 include guard**：每个 `.h/.hpp/.cpp` 在 include 之后插 `namespace <pfx> {`、文件末尾插 `}`；
   **同一步里把 include guard 也加前缀**（`ROBOT_CONTROLLER_H` → `TCBS_ROBOT_CONTROLLER_H`）——
   同源仓库改造前 guard 往往**完全相同**，只改路径不改 guard，两个同名头进同一个 TU 时第二个会被
   **静默跳过**（不报错，直接用错实现，比链接错误隐蔽得多）。
   include guard / `extern "C"` 块留在命名空间外；`main()` 留在命名空间外。
   `.cpp` 里的 `Foo::bar(...)` 定义放进 `namespace <pfx> {}` 后自动变成 `<pfx>::Foo::bar`，不用手改限定名。
4. **C ABI 前缀化**：`include/<pfx>/c_api/*.h` + 其 `.cpp` 里所有函数名、类型名、宏名加前缀；
   函数与类型名建议一次性 `sed` 全仓库替换（含 Python 绑定与文档）。
5. **CMake**：
   - `target_include_directories(<tgt> PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)` —— **只暴露这一层**
     （不要再暴露 `include/communication`、`include/common` 等二级目录，否则前缀化白做）；
   - target 名 + ALIAS + `OUTPUT_NAME` + `add_test` 名按 §1 改；
   - **`add_subdirectory` 友好**：子目录里**不要**调用 `project()` / `enable_testing()`；
     **不要**用会污染父作用域的 `set(CMAKE_*)` / `add_compile_options` / `include_directories`，
     改用 `target_*` 形式（父工程会以 `add_subdirectory(<repo> <bin> EXCLUDE_FROM_ALL)` 引入）。
6. **Python / 脚本 / 文档**：绑定文件里的符号名、`.so` 名、示例脚本、README/docs 全量同步；
   库名搜索最好支持环境变量覆盖（如 `TORQUE_BS_LIB`）。
7. **保持算法/协议/数值逻辑完全不变** —— 只改名字与命名空间。

---

## 3. 验收清单

```bash
# ① 独立编译
cmake -S . -B build && cmake --build build -j          # 0 error / 0 new warning
cd build && ctest --output-on-failure                  # 名字已带 pfx 前缀，全过

# ② 符号检查: C 符号带前缀、C++ 符号 mangled 名里出现 pfx
nm -D --defined-only build/lib<pfx>_robot_comm_c.so | grep " T " | head
#   期望: tcbs_robot_comm_*、_ZN4tcbs...（不再有裸 _ZN15RobotController...）

# ③ Python 绑定
PYTHONPATH=python python3 python/scripts/c_api_selftest.py

# ④ ★ 共存验证: 一个临时 CMake 工程同时 add_subdirectory 两个子模组
#    （两仓库都做了 pfx 化之后，configure/编译/链接/运行都要成功）
cmake -S /tmp/coexist -B /tmp/coexist/build && cmake --build /tmp/coexist/build -j
/tmp/coexist/build/coexist_demo
```

最小共存工程（`/tmp/coexist/CMakeLists.txt`）:

```cmake
cmake_minimum_required(VERSION 3.10)
project(coexist LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(/path/to/TorqueControllerForBigSmallYaw ${CMAKE_BINARY_DIR}/tcbs EXCLUDE_FROM_ALL)
add_subdirectory(/path/to/TorqueController                 ${CMAKE_BINARY_DIR}/tcs  EXCLUDE_FROM_ALL)
add_executable(coexist_demo main.cpp)
target_link_libraries(coexist_demo PRIVATE tcbs::robot_comm_c robot_comm_c)   # 旧仓库尚未前缀化
```

```cpp
// main.cpp —— 关键: 两个模块的头都能各自解析、两个 .so 能同时加载
#include "tcbs/RobotController.h"      // 双级版
#include "RobotController.h"           // 原单 yaw 版（尚未前缀化的裸路径）
#include <cstdio>
int main() {
    tcbs::RobotController::Config a{};
    RobotController::Config b{};
    std::printf("both modules linked OK\n");
    return 0;
}
```

---

## 4. 常见坑

0. **include guard 必须前缀化**（实测坑）：本仓库与 `TorqueController` 改造前的 guard 完全相同
   （`ROBOT_CONTROLLER_H`、`ROBOT_COMMUNICATION_C_H`），只做路径前缀化会在"两个同名头同时进一个 TU"时
   静默 skip 掉第二个 —— 共存验证里我们特意让两个 `RobotController.h` 与两个 C API 头进同一个 TU 来卡这条。
1. **C API 头千万别包进命名空间** —— 它要能被 C 代码直接 include（`extern "C"` 只解决链接名，不解决
   C++ 命名空间/C 结构体同名）。
2. **`OUTPUT_NAME` 必须改** —— 只改 target 名不够，两个子模组仍会产出同名 `.so`。
3. **只暴露一层 include 目录** —— 否则 `include/communication/Protocol.hpp` 这类裸路径仍会撞。
4. **`add_test` 名要带前缀** —— 父工程 `enable_testing()` 后所有子目录的测试名在同一命名空间里。
5. **父工程侧也要改**：链接名（`robot_comm_c` → `tcbs::robot_comm_c`）、include 路径
   （`#include "RobotController.h"` → `#include "tcbs/RobotController.h"`）、以及类型名
   （`RobotController::State` → `tcbs::RobotController::State`）。父工程自己的 `include/common/...`
   也要注意别和子模组的 `common/` 撞名（本规范把子模组都收进 `<pfx>/` 后即可避免）。
6. **`EXCLUDE_FROM_ALL`**：父工程只想链接库、不想让子模组的 demo/测试参与默认构建时用它；
   但 `ctest` 需要显式 `cmake --build build --target tcbs_test_...` 或去掉 `EXCLUDE_FROM_ALL`。
7. **`find_library`/cache 变量加 pfx**：`UDEV_LIBRARY`、`CERES_LIBRARIES` 这类在父工程里是全局 cache，
   两个子模组都写同一个名字会互相覆盖（本仓库已改成 `TCBS_UDEV_LIBRARY`）。
8. **同时加载两个 .so 的运行时细节**：即使符号已区分，若两者都链接了同一份第三方静态库
   （Ceres/Eigen 头无碍，静态库有碍），注意 `-Wl,-Bsymbolic` 或动态库形式；本仓库用 `SHARED` +
   系统 Ceres 动态库，实测无问题。
