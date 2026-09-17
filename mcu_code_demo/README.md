# mcu_code_demo —— 电控(MCU)侧接入说明

本目录是**给电控同学参考/直接抄进嵌入式工程**的示例代码，不属于上位机的构建系统
（`CMakeLists.txt` 不编译这里的文件）。

| 文件 | 说明 |
| --- | --- |
| `yaw_control_single_reference.c` | 旧版**单轴** yaw 示例（仅作写法参考：多圈累计、限幅、力矩下发） |
| `dual_yaw_control.c` | **本版双级 yaw**（大 yaw + 小 yaw）完整示例：协议解析/组包、两关节控制、小 yaw 安全层（行程 ±30°）、看门狗、**MCU2 新样本序号** |
| `dual_yaw_control_selftest.c` | 小 yaw 安全层的**PC 自检 runner**（自带硬件桩 + `main`，`gcc ... -lm` 直接跑） |

协议权威定义在 `include/tcbs/communication/Protocol.hpp`（v0x03），CRC 实现在
`include/tcbs/communication/CRC.h` + `src/communication/CRC.cpp`。`dual_yaw_control.c`
把这两者的字节布局用 `#pragma pack(1)` 复刻了一遍，并在文件里用 `_Static_assert`
逐字段核对偏移与帧长 —— 若上位机侧改了协议，这里会**编译期报错**而不是静默错位。

## 0. 系统结构与「新样本序号」（本文件的核心语义）

```
上位机(PC) ⟷ 串口 ⟷ MCU1 ⟷ 内部链路(约10Hz,不稳定) ⟷ MCU2
                    │                                    │
                    ├─ pitch   (每帧新值)                 ├─ 大 yaw 编码器/电机
                    ├─ 小 yaw  (每帧新值)                 └─ 底盘 IMU
                    └─ 大 yaw 指令转发
```

- MCU1 ↔ MCU2 的链路**约 10Hz 量级、间隔不规则**（基本不低于 3Hz），且 MCU1 在没收到新数据时会
  **一直沿用旧值**；于是上位机收到的每一帧里，大 yaw / 底盘 IMU 可能是几十~几百毫秒前的旧值。
- **电控侧不提供任何时间信息**（MCU 端没有可靠毫秒计数，只有上位机能计时），
  因此改用**一个「MCU2 新样本序号」**来表达"这是不是一个新样本"：
  1. `mcu2_seq` 是**新样本计数器**：MCU1 每**真正从 MCU2 取到一次新数据**才 `+1`；
  2. 大 yaw 与底盘 IMU **同源同批**（MCU2 一次把两路一起送来），所以**共用一个序号** ——
     同一次取数里两路一起刷新，只 `+1` 一次（不是两路各加一次）；
  3. 值被保持的那些帧里，**序号与值都原样重复**，绝不递增
     —— 上位机判定规则是「**首帧到达 或 序号变化 或 值变化**」三者取或 ⇒ 新样本
     （即使新样本的值与上一帧恰好相同，序号变化也能识别出来）；
  4. 序号用 `uint8`，自然回绕即可（上位机只比较"是否与上次不同"）；
  5. **不需要"0 = 从未收到"的约定**（不做可用性检测）：上电初值 0，收到的第一帧就当一次更新；
  6. **反馈采集发生在看门狗/使能判断之前**：看门狗超时、自瞄未使能等状态下
     **照常跟踪 MCU2 并照常递增序号**；被冻结的只有**力矩输出**。
     理由：「不要输出力矩」与「是否读取反馈」是两件事，回传帧在故障期间本来就照常发送；
     若连反馈一起冻结，恢复控制的那一刻只能拿一个陈旧值当锚点（大 yaw 若在故障期间被外力
     转动过，上位机的修正还要被 `big_enc_max_jump` 限幅，需 2~3 个新样本 ≈0.3s@10Hz 才收敛）；
     保持跟踪则恢复瞬间锚点就是新鲜的，控制恢复无瞬态。
     告警能力并不因此丢失：故障期间若 MCU2 也断了，序号同样不再递增、上位机看到的"年龄"照样增长。
  7. 在这套语义下，**序号不变且值也不变**只有一个含义：**MCU1 没有从 MCU2 收到更新的数据**。
- 实现见 `dual_yaw_control.c` 的 `yaw_feedback_update()`：取完两路后，只要有任意一路报告
  "有新数据"就 `mcu2_seq++` **一次**；两路都没新数据时**值与序号一起保持**。
- 小 yaw / pitch 每帧都是新样本，**不带**序号。
- 值的年龄由**上位机**用自己的时钟测量（从"首次看到该序号"起算）+ 链路传输时延；
  上位机据此对大 yaw / 底盘姿态角做**角速度一阶外推**，对底盘角速度做**零阶保持**。

`YAW_BIG_SOURCE_FROM_MCU2`（默认 1）描述的就是这个结构：大 yaw 的**反馈值来自 MCU2**（带序号），
大 yaw 的**控制指令转发给 MCU2**（由 MCU2 用自己实时的编码器跑内环，避免用旧角度做闭环）。
若你的机器是「单 MCU 直连大 yaw」（上一版示例的做法），把它设为 0 即可，两种接法都能编译
（此时大 yaw 每个反馈周期都是新样本，序号随之 +1）。

---

## 1. 编译

只需要一个 C11 编译器，无外部依赖（不用 `math.h` 之外的库；`-lm`）：

```bash
gcc -std=c11 -Wall -Wextra -c dual_yaw_control.c -o dual_yaw_control.o
```

本文件已用 `gcc -std=c11 -Wall -Wextra -c` 验证：**0 error / 0 warning**
（两种结构 `YAW_BIG_SOURCE_FROM_MCU2=1` 与 `=0` 都验证过）。

安全层自检（可选，PC 上跑，见 §6.1）：

```bash
gcc -std=c11 -Wall -Wextra -pedantic -o /tmp/yaw_selftest \
    dual_yaw_control_selftest.c -lm && /tmp/yaw_selftest
```

## 2. 三个调用点

```c
#include "dual_yaw_control.c"   /* 或把它加入你的工程源文件列表 */

/* ① 上电初始化一次 */
dual_yaw_control_init();

/* ② 串口每收到 1 字节就喂进来（可以放在 USART 接收中断里） */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    dual_yaw_rx_byte(g_rx_byte);
    HAL_UART_Receive_IT(huart, &g_rx_byte, 1);
}
/* 若一次收到一整块: dual_yaw_rx_bytes(buf, len); */

/* ③ 定时器周期任务, 建议 1kHz */
void YawTask_1kHz(void) {
    dual_yaw_control_step(HAL_GetTick());   /* 参数 = 本地毫秒计数（仅用于周期调度与看门狗） */
}
```

`dual_yaw_control_step()` 内部依次完成：
取走最新合法帧 → 大 yaw 多圈累计（仅单 MCU 方案）
→ **反馈采集：取 MCU2 新样本 → 刷新值 + 序号 +1（在看门狗/使能判断之前，故障期间照常进行）**
→ 看门狗判定（超时 → **只把力矩清零**）→ 使能时执行大/小 yaw 控制（含小 yaw 安全层）→ 下发
→ 按 `YAW_FEEDBACK_TX_PERIOD_MS` (默认 5ms) 回传反馈帧（47B）。

## 3. 需要你填写的桩函数清单（全部标了 `// TODO: 用户填写`）

| 桩函数 | 作用 | 要点 |
| --- | --- | --- |
| `get_encoder_big_raw()` | 大 yaw 单圈原始计数 | 返回 0~8191；**仅单 MCU 方案**（`YAW_BIG_SOURCE_FROM_MCU2=0`）用 |
| `get_encoder_big_velocity_rpm()` | 大 yaw 速度 | 电机速度报文 rpm；同上 |
| `get_encoder_small_counts()` | 小 yaw 该关节绝对/相对计数 | 换算宏 `YAW_SMALL_RAD_PER_COUNT`、零点 `YAW_SMALL_ENCODER_ZERO_COUNTS` |
| `get_encoder_small_velocity_rpm()` | 小 yaw 速度 | rpm |
| `can_send_torque_big(cmd)` | 大 yaw 力矩/电流指令下发 | `cmd` ∈ ±16384；**仅单 MCU 方案**用 |
| `can_send_torque_small(cmd)` | 小 yaw 力矩/电流指令下发 | `cmd` ∈ ±16384 |
| `uart_send_bytes(buf, len)` | 回传反馈帧 | 长度恒为 **48** |
| `mcu2_get_yaw_big_sample(&angle, &omega)` | **MCU2 的大 yaw 是否有新样本** | 返回 1=有新样本（调用方会让 `mcu2_seq++` **一次**）；返回 0 时**不要动出参**（旧值与旧序号要一起保持） |
| `mcu2_get_chassis_imu_sample(&yaw, &omega)` | **MCU2 的底盘 IMU 是否有新样本** | 同上 |
| `mcu2_send_yaw_big_command(mode, θ*, ω*, τ)` | 把大 yaw 指令转发给 MCU2 | MCU2 用自己实时的编码器跑内环；传 0 力矩 = 大 yaw 归零 |
| `get_pitch_angle_raw()` | pitch 关节**原始**角 | 电控**不做**线性映射（由上位机做） |
| `get_bullet_velocity()` | 弹速 m/s | |
| `get_chassis_imu_yaw()` / `get_chassis_imu_omega()` | 底盘 IMU 姿态/角速度 | 0~2π / rad·s⁻¹；**仅单 MCU 方案**用（真实结构由 MCU2 提供） |
| `get_mark_color()` | 颜色 | |
| `get_auto_aim_switch()` | 电控自瞄开关 | 与上位机 `auto_aim_enable` 相与 |
| `get_motor_temperature_big()` / `_small()` | 两电机温度 | uint8，℃ |
| `yaw_enter_critical()` / `yaw_exit_critical()` | 关/开中断 | 用于接收中断与主循环交换命令缓冲 |

> 链路桩的关键约定：**返回 0 时不要改出参**；**不要"每次都返回 1"**（那会让序号每帧 +1，
> 上位机会把被保持的旧值误判成每帧刷新的新样本）；同一帧被重复读到也不算新样本。

## 4. 调参入口

所有可调量都集中在 `dual_yaw_control.c` 第 1 节（文件头有完整清单）：

- 系统结构：`YAW_BIG_SOURCE_FROM_MCU2`（默认 1 = 大 yaw 反馈来自 MCU2、指令转发 MCU2；
  设 0 = 单 MCU 直连大 yaw 编码器 + 本地 CAN 下发，两种接法都能编译）
- 内环增益：`YAW_BIG_KP/KD`、`YAW_SMALL_KP/KD`
  （大 yaw 由 MCU2 驱动时，大 yaw 的 KP/KD 与力矩换算属于 **MCU2 的工程**）
- 力矩换算：`YAW_*_KT_NM_PER_A`、`YAW_*_GEAR_RATIO`、`YAW_*_CURRENT_FS_A`
  → `YAW_*_TORQUE_TO_CMD_SCALE`（两电机的力矩常数/减速比不同，**必须分别标定**）
- 小 yaw 限位（**机械行程 ±30°**，中心 0）：
  硬限位 `YAW_SMALL_MIN_RAD = −30°` / `YAW_SMALL_MAX_RAD = +30°`、
  目标角夹取余量 2°（→ `[−28°, +28°]`）、减速区 `YAW_SMALL_DECEL_ZONE_LEN_RAD = 10°`
  （距任一侧限位 10° 起降速，即 `[−15°, +10°]` 之外）、
  限速 `YAW_SMALL_RATE_MAX_RAD_S` / `YAW_SMALL_RATE_AT_HARD_LIMIT_RAD_S`、
  硬限位回中力矩 `YAW_SMALL_HARD_CENTER_TORQUE_NM`（默认 0 = 零力矩）
  ⚠ **两侧必须独立判断**：`YAW_SMALL_TARGET_MIN_RAD/MAX_RAD` 是两个宏，
  不要写 `±LIMIT` 或 `fabsf(angle) > LIMIT`（那会让负侧按正侧的余量算，提前 5° 就限速）
- 看门狗：`YAW_WATCHDOG_TIMEOUT_MS`（默认 50）
- 回传周期：`YAW_FEEDBACK_TX_PERIOD_MS`（默认 5）

## 5. 协议要点（易踩坑）

1. **整帧长度**：`3 前导 + 1 data_size + payload + 1 CRC8`
   - PC→MCU：`data_size = 36`，整帧 **41** 字节（未变）
   - MCU→PC：`data_size = 43`，整帧 **48** 字节（v0x03 演进：50 → 58 → **48**）
   - 两个常见错误：只发「前导 + data_size + payload」（41/47 字节）而**漏掉最后的 CRC 字节**；
     或 `data_size` 与实际 payload 不一致 —— 两者都会让上位机永远解析不出帧。
2. **CRC8**：查表法，初值 `0xFF`，覆盖范围为「整帧 − 1 字节」（即不含 CRC 自身），
   与 `CRC8_Check_Sum(ptr, sizeof(packet) - 1)` 一致。CRC 表已与
   `src/communication/CRC.cpp` 逐项核对一致。
3. **模式位语义**：`0 = 仅力矩`，`1 = 力矩 + 位置/速度内环`。
   注意这与旧版单轴示例的 `yaw_torque_only_mode`（1 = 仅力矩）**相反**。
4. **double 字段**（`yaw_big_target_angle`、`yaw_big_angle`）必须用 `memcpy`
   按 8 字节存取，不要直接对 packed 结构体成员取地址/赋值。
5. **前导两边相同**（都是 `0x42 0x52 0x03`），靠 `data_size` 区分收发方向；
   接收状态机对 `data_size != 36` 的帧会丢弃，因此串口回环自己发出去的帧也不会被误收。
6. 前导之外的**每帧数据都用 CRC 兜底**，CRC 不过 → 丢弃并重新同步（只丢 1 字节）。
7. **帧内没有任何时间字段**（MCU 端计时不可用）：`mcu2_seq`（偏移 45）是 **MCU2 新样本序号**，
   `crc8` 在偏移 46。

## 6. 安全逻辑（小 yaw，最后一道防线）

机械行程是 **±30°**（中心 0），撞死会打坏电机/线束，
因此 `small_yaw_guard()` 做四件事：

1. **目标角限位**：θ\* 夹到 `[−28°, +28°]`（两侧各留 2° 余量）→ 越界后位置误差天然指向内侧；
2. **接近限位限速**：距**任一侧**硬限位 10° 以内（即 θ < −15° 或 θ > +10°）起，
   允许 |ω\*| 随剩余角度线性下降，贴到限位时只剩 0.2 rad/s；
3. **越软限位禁止向外施力**：正侧禁正力矩、负侧禁负力矩，**只允许回中方向力矩**；
4. **越硬限位**：整帧力矩作废，只允许零力矩/回中力矩。

⚠ 第 2/3/4 条**两侧独立判断**：全部走 `hard_min/hard_max`、`soft_min/soft_max` 分侧标志
（`hard_limit/soft_limit` 只是两者的"或"，供日志用），**没有任何 `fabsf(angle) > LIMIT`
的对称写法** —— 非对称行程下对称写法会让负侧提前 5° 就限速、并错误地清掉回中方向力矩。

注意第 3、4 条是**双重保险**：`small_yaw_control()` 内环算完之后，还会再调用
`small_yaw_apply_limit_torque()` 对总力矩执行一次同样的规则 —— 内环输出也休想
把关节继续往限位外推。

### 6.1 安全层自检（PC 上跑，防"某一处又写回对称形式"）

`dual_yaw_control.c` 末尾有一节**可选编译**的自检代码（`YAW_SMALL_LIMIT_SELFTEST=1` 打开），
配合 `dual_yaw_control_selftest.c`（提供硬件桩 + `main`）可以在 PC 上直接跑，把安全层
**两侧**逐条钉死（宏关系、两侧夹取角、两侧减速区映射、两侧软/硬限位标志、力矩方向、末级约束、NaN/NULL）：

```bash
gcc -std=c11 -Wall -Wextra -pedantic -o /tmp/yaw_selftest \
    mcu_code_demo/dual_yaw_control_selftest.c -lm && /tmp/yaw_selftest
# → 共 47 项, 失败 0 项 → 全部通过（退出码 0）
```

这项自检是纯逻辑（不碰硬件），**改了行程/余量/减速区长度后必须重跑**。

**看门狗（`YAW_WATCHDOG_TIMEOUT_MS`，默认 50ms）是必需的**：上位机（MPC/自瞄）
死机、串口掉线或线缆松脱时，如果继续执行最后一帧的力矩指令，云台会持续朝一个
方向加速（典型"满舵打转"），极易损坏机械并伤及人员。因此超时后两关节力矩**立即
清零**，上电到收到第一帧合法指令之前也一律零力矩。大 yaw 的归零同样不能漏：
在真实结构下由 `yaw_big_zero_torque()` **转发一条零力矩指令给 MCU2**（本机不再直接发 CAN）。

⚠ 注意看门狗/未使能**只冻结力矩输出**，**不冻结反馈采集**：`yaw_feedback_update()` 在
`dual_yaw_control_step()` 里位于看门狗/使能判断**之前**，因此故障期间大 yaw / 底盘 IMU 的
值与新样本序号照常更新（回传帧本来也照常发送），只是两关节力矩为零。这样上位机在控制
恢复的瞬间就能拿到新鲜锚点，不必用陈旧值重锚后再花几个样本收敛。

## 7. 上位机侧怎么对上

- 上位机 `mcu::SendPacket`（41B / `data_size=36`）**未变**；
- 上位机 `mcu::ReceivePacket` 现在是 **47B / `data_size=42`**：
  - 末尾为 `mcu2_seq`（偏移 **45**）、`crc8`（偏移 **46**）；
  - **没有** `mcu_tick_ms` / `yaw_big_tick_ms` / `chassis_imu_tick_ms`（MCU 端不计时）。
- `yaw_big_angle` 是**多圈连续** double（由 MCU2 计算并累计多圈），"值可能被保持"；
- `yaw_small_angle` 是**相对大 yaw 的关节角**，每帧都是新样本；
- **上位机判新的正确姿势**：
  - 判定规则：**首帧到达 或 `mcu2_seq` 变化 或 值变化** ⇒ 新样本（三者取或）；只有全都不变才是被保持的旧值，
    才把这组值喂给状态估计（**即使值与上一帧恰好相同，序号变化也说明是新样本**）；
  - 序号为 **0** 表示该通道**从未收到过数据** → 视为无效，退化为"按值变化"判定；
  - 值的年龄由**上位机自己的时钟**测量（从首次看到该序号起算）+ 链路传输时延，
    据此对大 yaw / 底盘姿态角做**角速度一阶外推**、对底盘角速度做**零阶保持**。
- pitch 的「编码器/指令 ↔ 角度」线性映射由上位机 `McuDataPreprocessor` 完成，
  电控只回传**原始角**。
