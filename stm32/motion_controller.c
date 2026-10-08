/*
 * 运动控制器实现 (简化版本)
 *
 * 设计理念：
 * 1. **简化X/Y轴运动控制** - 不需要精确速度闭环
 *    - 目标：产生加速度即可，不追求精确速度控制
 *    - 方法：直接设置电机基准PWM偏移量
 *
 * 2. **控制逻辑**：
 *    - X方向运动(前后): 设置Pitch轴(M4,M5)的PWM偏移量
 *    - Y方向运动(左右): 设置Roll轴(M2,M3)的PWM偏移量
 *    - Yaw轴(M0,M1): 仅做角度稳定,不接收运动偏移
 *
 * 3. **最终PWM计算** (在axis_stabilizer中):
 *    PWM = base_pwm + delta_sign * delta + motion_offset
 *
 *    其中:
 *    - base_pwm: 从motor_calibration.h读取的基准值
 *    - delta: 稳定器计算的差速控制量 (PID输出)
 *    - motion_offset: 本模块提供的运动偏移量
 *
 * 4. **三轴工作模式**:
 *    - Roll轴: PID角度稳定 + 接收左右运动偏移
 *    - Pitch轴: 无PID控制, 仅接收前后运动偏移 (震荡小,不需要稳定)
 *    - Yaw轴: PID角度稳定(带滤波), 无运动偏移
 */

#include "motion_controller.h"
#include "motor_control.h"
#include <math.h>

// =====================================================
// 配置参数
// =====================================================

// 速度到PWM偏移的映射增益
// velocity范围: -1.0 到 +1.0 (归一化)
// PWM偏移范围: -100 到 +100 (对应满速运动)
#define VELOCITY_TO_PWM_OFFSET  100.0f

// PWM偏移限制 (防止过大影响稳定性)
#define PWM_OFFSET_MAX_PITCH  150  // Pitch轴最大偏移
#define PWM_OFFSET_MAX_ROLL   150  // Roll轴最大偏移

// =====================================================
// 内部状态
// =====================================================

static bool g_motion_enabled = false;

// 运动控制产生的PWM偏移量
static int16_t g_pwm_offset_pitch = 0;  // Pitch轴(M4,M5)偏移量 - 前后运动
static int16_t g_pwm_offset_roll = 0;   // Roll轴(M2,M3)偏移量 - 左右运动

// 当前速度指令
static float g_velocity_x = 0.0f;  // X方向速度 (-1.0 ~ +1.0)
static float g_velocity_y = 0.0f;  // Y方向速度 (-1.0 ~ +1.0)

// =====================================================
// 内部函数
// =====================================================

static inline float clampf(float x, float lo, float hi){
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline int16_t clampi(int16_t x, int16_t lo, int16_t hi){
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

// =====================================================
// 公共API实现
// =====================================================

void motion_controller_init(void){
    g_motion_enabled = false;
    g_pwm_offset_pitch = 0;
    g_pwm_offset_roll = 0;
    g_velocity_x = 0.0f;
    g_velocity_y = 0.0f;
}

void motion_set_roll_target(float target_angle_deg){
    // 保留接口，暂不实现
    // Roll控制通过velocity_y实现
    (void)target_angle_deg;
}

void motion_set_pitch_target(float target_angle_deg){
    // 保留接口，暂不实现
    // Pitch控制通过velocity_x实现
    (void)target_angle_deg;
}

void motion_set_velocity(float velocity_x, float velocity_y){
    if (!g_motion_enabled) {
        return;
    }

    // 限制输入范围
    velocity_x = clampf(velocity_x, -1.0f, 1.0f);
    velocity_y = clampf(velocity_y, -1.0f, 1.0f);

    g_velocity_x = velocity_x;
    g_velocity_y = velocity_y;

    // ===== X方向运动 (前后) =====
    // velocity_x控制Pitch轴电机(M4,M5)的整体PWM偏移
    // velocity_x > 0: 向前，M4和M5都增加PWM
    // velocity_x < 0: 向后，M4和M5都减少PWM
    g_pwm_offset_pitch = (int16_t)(velocity_x * VELOCITY_TO_PWM_OFFSET);
    g_pwm_offset_pitch = clampi(g_pwm_offset_pitch, -PWM_OFFSET_MAX_PITCH, PWM_OFFSET_MAX_PITCH);

    // ===== Y方向运动 (左右) =====
    // velocity_y控制Roll轴电机(M2,M3)的整体PWM偏移
    // velocity_y > 0: 向右，M2和M3都增加PWM
    // velocity_y < 0: 向左，M2和M3都减少PWM
    g_pwm_offset_roll = (int16_t)(velocity_y * VELOCITY_TO_PWM_OFFSET);
    g_pwm_offset_roll = clampi(g_pwm_offset_roll, -PWM_OFFSET_MAX_ROLL, PWM_OFFSET_MAX_ROLL);

    // 注意：这些偏移量会在axis_stabilizer中被读取并叠加到最终PWM
    // 公式: final_pwm = base_pwm + delta_sign * delta + motion_offset
}

void motion_set_yaw_rate(float omega_z){
    // Yaw旋转现在由独立的Yaw稳定器控制
    // 此函数保留用于未来扩展(如设置Yaw目标角度)
    (void)omega_z;

    // TODO: 如果需要控制Yaw旋转速度而非角度稳定,
    // 可以在这里添加接口调用Yaw稳定器设置目标角速度
}

void motion_stop(void){
    // 停止所有运动：清零所有偏移量
    g_velocity_x = 0.0f;
    g_velocity_y = 0.0f;

    g_pwm_offset_pitch = 0;
    g_pwm_offset_roll = 0;
}

float motion_get_roll_target(void){
    return g_velocity_y;
}

float motion_get_pitch_target(void){
    return g_velocity_x;
}

void motion_set_enabled(bool enabled){
    g_motion_enabled = enabled;

    if (!enabled) {
        motion_stop();
    }
}

bool motion_get_enabled(void){
    return g_motion_enabled;
}

// =====================================================
// 获取运动控制偏移量（供稳定器调用）
// =====================================================

/**
 * @brief 获取Pitch轴运动控制PWM偏移量
 * @return PWM偏移量（正=向前，负=向后）
 */
int16_t motion_get_pitch_pwm_offset(void){
    return g_pwm_offset_pitch;
}

/**
 * @brief 获取Roll轴运动控制PWM偏移量
 * @return PWM偏移量（正=向右，负=向左）
 */
int16_t motion_get_roll_pwm_offset(void){
    return g_pwm_offset_roll;  // Roll轴接收左右运动偏移
}

/**
 * @brief 获取Yaw轴运动控制PWM偏移量
 * @return PWM偏移量（Yaw轴不接收运动偏移,返回0）
 */
int16_t motion_get_yaw_pwm_offset(void){
    return 0;  // Yaw轴仅做角度稳定,不接收运动偏移
}
