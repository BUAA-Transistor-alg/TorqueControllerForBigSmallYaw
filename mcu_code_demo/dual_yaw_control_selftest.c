/* ============================================================================
 * dual_yaw_control_selftest.c —— 小 yaw 安全层自检 runner（**PC 上跑, 不烧板**）
 *
 * 为什么需要它: 小 yaw 的机械行程与电控限位一旦改动, 最容易犯的错误就是
 * `±LIMIT` / `fabsf(angle) > LIMIT` 的对称形式, 负侧就会按正侧的余量限速/夹角
 * —— 这种错误编译期看不出来, 只有把两侧逐条钉死才能防回归。
 *
 * 用法（无外部依赖, 只需 C11 编译器）:
 *     gcc -std=c11 -Wall -Wextra -pedantic -o /tmp/yaw_selftest \
 *         mcu_code_demo/dual_yaw_control_selftest.c -lm && /tmp/yaw_selftest
 * 退出码: 0 = 全部通过; 1 = 有失败项。
 *
 * 实现: 直接用 `#include "dual_yaw_control.c"` 把它编进同一个翻译单元
 *      （因此自检可以调用文件内的 static 函数, 例如末级力矩约束）, 并在这里提供
 *      控制代码需要的**硬件桩函数**（全部空实现: 自检只测安全层的纯逻辑）。
 *      同时定义 YAW_SMALL_LIMIT_SELFTEST=1 打开被检文件末尾的自检代码块。
 * ==========================================================================*/
#define YAW_SMALL_LIMIT_SELFTEST 1

#include "dual_yaw_control.c"

#include <stdio.h>

/* ============================================================================
 * 硬件桩函数（自检不需要真实硬件: 全部返回安全默认值/空实现）
 * 说明: 这些函数在 dual_yaw_control.c 里都标了 "TODO: 用户填写", 真实工程里由用户实现。
 * ==========================================================================*/
uint16_t get_encoder_big_raw(void)            { return 0u; }
int16_t  get_encoder_big_velocity_rpm(void)   { return 0; }
int32_t  get_encoder_small_counts(void)       { return 0; }
int16_t  get_encoder_small_velocity_rpm(void) { return 0; }

void can_send_torque_big(int16_t cmd)         { (void)cmd; }
void can_send_torque_small(int16_t cmd)       { (void)cmd; }

void uart_send_bytes(const uint8_t *data, uint32_t len) { (void)data; (void)len; }

float   get_pitch_angle_raw(void)        { return 0.0f; }
float   get_bullet_velocity(void)        { return 0.0f; }
float   get_chassis_imu_yaw(void)        { return 0.0f; }
float   get_chassis_imu_omega(void)      { return 0.0f; }
uint8_t get_mark_color(void)             { return 0u; }
uint8_t get_auto_aim_switch(void)        { return 0u; }
uint8_t get_motor_temperature_big(void)  { return 0u; }
uint8_t get_motor_temperature_small(void){ return 0u; }

void yaw_enter_critical(void)            { }
void yaw_exit_critical(void)             { }

uint8_t mcu2_get_yaw_big_sample(double *angle_rad, float *omega_rad_s)
{
    (void)angle_rad; (void)omega_rad_s;
    return 0u;      /* 没有新样本: 出参不改（值被保持） */
}
uint8_t mcu2_get_chassis_imu_sample(float *yaw_rad, float *omega_rad_s)
{
    (void)yaw_rad; (void)omega_rad_s;
    return 0u;
}
void mcu2_send_yaw_big_command(uint8_t mode, double target_angle,
                               float target_velocity, float torque_nm)
{
    (void)mode; (void)target_angle; (void)target_velocity; (void)torque_nm;
}

/* ============================================================================
 * 自检打印与 main
 * ==========================================================================*/
static int g_total = 0;
static int g_failed = 0;

static void selftest_report(const char *name, int ok)
{
    ++g_total;
    if (ok) {
        printf("  [PASS] %s\n", name);
    } else {
        ++g_failed;
        printf("  [FAIL] %s\n", name);
    }
}

int main(void)
{
    int fail;

    printf("=== 小 yaw 安全层自检（行程 ±30°，电控夹取 [−28°,+28°]）===\n");
    printf("硬限位   : [%.2f°, %.2f°]\n",
           (double)YAW_SMALL_MIN_RAD / YAW_DEG2RAD, (double)YAW_SMALL_MAX_RAD / YAW_DEG2RAD);
    printf("目标夹取 : [%.2f°, %.2f°]（两侧各留 %.2f°）\n",
           (double)YAW_SMALL_TARGET_MIN_RAD / YAW_DEG2RAD,
           (double)YAW_SMALL_TARGET_MAX_RAD / YAW_DEG2RAD,
           (double)YAW_SMALL_SOFT_MARGIN_RAD / YAW_DEG2RAD);
    printf("减速区   : 距任一侧限位 %.2f° 起（即 [%.2f°, %.2f°] 之外）\n\n",
           (double)YAW_SMALL_DECEL_ZONE_LEN_RAD / YAW_DEG2RAD,
           (double)YAW_SMALL_DECEL_START_MIN_RAD / YAW_DEG2RAD,
           (double)YAW_SMALL_DECEL_START_MAX_RAD / YAW_DEG2RAD);

    fail = yaw_small_limit_selftest(selftest_report);

    printf("\n共 %d 项, 失败 %d 项 → %s\n", g_total, fail, (fail == 0) ? "全部通过" : "存在失败");
    (void)g_failed;
    return (fail == 0) ? 0 : 1;
}
