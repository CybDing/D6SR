/*
 * 通用轴稳定器实现
 *
 * 核心算法:
 * 1. PID控制: u_pid = Kp*e + Ki*∫e + Kd*de/dt
 * 2. 重力补偿: grav = Kgrav * sin((angle - stable_angle) * π/180)
 * 3. 前馈控制: ff = Kff * gyro_rate (暂时忽略)
 * 4. 总控制量: u = u_pid + grav_sign * grav + ff
 * 5. PWM计算: M_neg = base_neg + delta_sign_neg * delta + offset
 *            M_pos = base_pos + delta_sign_pos * delta + offset
 */

#include "axis_stabilizer.h"
#include "motor_control.h"
#include "motor_calibration.h"
#include "FreeRTOS.h"
#include "queue.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// =====================================================
// 外部全局 IMU 数据 (由 main.c 中 imu_reader_task 更新)
// =====================================================
// sys_state.imu_data[] 索引:
// [0]=RollSpeed, [1]=PitchSpeed, [2]=YawSpeed
// [3]=Roll, [4]=Pitch, [5]=Yaw
// [6]=qw, [7]=qx, [8]=qy, [9]=qz, [10]=timestamp
typedef struct {
    uint16_t motor_pwm[6];
    float imu_data[11];
    void* pid_params;       // 不使用，仅占位
    bool pid_enabled;
    bool system_ready;
} extern_sys_state_t;

extern extern_sys_state_t sys_state;

// =====================================================
// 控制参数常量
// =====================================================

#define DEG2RAD (3.1415926f / 180.0f)

// PWM映射增益 (控制量 → PWM增量)
#define PWM_PER_DEG  2.0f

// 差速变化速率限制 (防止电机突变)
// 值越大响应越快，但可能导致电机抖动
#define DELTA_SLEW_RATE  3

// 微分滤波系数 (低通滤波,减少噪声)
// 值越大滤波越弱，响应越快
#define D_FILTER_ALPHA  0.15f

// 前馈滤波系数
#define FF_FILTER_ALPHA  0.3f

// 积分限幅
#define INTEGRAL_LIMIT  100.0f

// Yaw角度滤波系数 (低通滤波,减少漂移)
// alpha越小滤波越强,但响应越慢
// 建议范围: 0.1~0.3
#define YAW_FILTER_ALPHA  0.15f

// =====================================================
// 内部辅助函数
// =====================================================

static inline float clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

// 获取外部UART发送队列 (用于调试输出)
extern QueueHandle_t uart_tx_queue;

// =====================================================
// 公共API实现
// =====================================================

void axis_stabilizer_init(AxisStabilizer* stab, const AxisControlConfig* config)
{
    if (!stab || !config) return;

    stab->config = config;
    stab->enabled = false;
    stab->initialized = true;

    // 重置状态
    stab->integral = 0.0f;
    stab->prev_error = 0.0f;
    stab->d_filtered = 0.0f;
    stab->prev_delta = 0;
    stab->ff_filtered = 0.0f;

    // 默认使用配置文件参数
    stab->use_runtime_pid = false;
    stab->runtime_kp = config->kp;
    stab->runtime_ki = config->ki;
    stab->runtime_kd = config->kd;
    stab->runtime_kff = config->kff;
    stab->runtime_kgrav = config->gravity_comp_gain;

    // 统计信息
    stab->step_count = 0;
    stab->fail_count = 0;

    // 外部偏移量
    stab->external_pwm_offset = 0;

    // Yaw滤波器初始化
    stab->yaw_filtered = 0.0f;
    stab->yaw_filter_initialized = false;

}

void axis_stabilizer_set_enabled(AxisStabilizer* stab, bool enabled)
{
    if (!stab || !stab->initialized) return;

    stab->enabled = enabled;

    if (!enabled) {
        // 禁用时重置所有状态
        stab->integral = 0.0f;
        stab->prev_error = 0.0f;
        stab->d_filtered = 0.0f;
        stab->prev_delta = 0;
        stab->ff_filtered = 0.0f;
        stab->yaw_filter_initialized = false;  // 重置Yaw滤波器
    }
}

bool axis_stabilizer_get_enabled(const AxisStabilizer* stab)
{
    if (!stab || !stab->initialized) return false;
    return stab->enabled;
}

void axis_stabilizer_set_pwm_offset(AxisStabilizer* stab, int16_t offset)
{
    if (!stab || !stab->initialized) return;
    stab->external_pwm_offset = offset;
}

void axis_stabilizer_step(AxisStabilizer* stab, float dt_s)
{
    if (!stab || !stab->initialized) return;
    if (!stab->enabled) return;

    const AxisControlConfig* cfg = stab->config;
    stab->step_count++;

    // ===== 1. 从全局 sys_state 读取 IMU 数据 (无阻塞) =====
    // sys_state.imu_data[] 由 imu_reader_task 以固定频率更新
    // 索引: [0]=RollSpeed, [1]=PitchSpeed, [2]=YawSpeed
    //       [3]=Roll, [4]=Pitch, [5]=Yaw

    // 根据轴类型选择对应的角度和角速度
    float current_angle = 0.0f;
    float gyro_rate = 0.0f;
    const char* axis_label = cfg->axis_name;

    if (strcmp(cfg->axis_name, "Roll") == 0) {
        current_angle = sys_state.imu_data[3];  // Roll
        gyro_rate = sys_state.imu_data[0];      // RollSpeed
    } else if (strcmp(cfg->axis_name, "Pitch") == 0) {
        current_angle = sys_state.imu_data[4];  // Pitch
        gyro_rate = sys_state.imu_data[1];      // PitchSpeed
    } else if (strcmp(cfg->axis_name, "Yaw") == 0) {
        // Yaw角度需要滤波处理,减少传感器漂移
        float yaw_raw = sys_state.imu_data[5];  // Yaw

        // 初次运行时,直接使用原始值
        if (!stab->yaw_filter_initialized) {
            stab->yaw_filtered = yaw_raw;
            stab->yaw_filter_initialized = true;
        } else {
            // 一阶低通滤波: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
            // alpha小 = 滤波强,响应慢,抗漂移好
            stab->yaw_filtered = YAW_FILTER_ALPHA * yaw_raw +
                                (1.0f - YAW_FILTER_ALPHA) * stab->yaw_filtered;
        }

        current_angle = stab->yaw_filtered;  // 使用滤波后的值
        gyro_rate = sys_state.imu_data[2];   // YawSpeed
    }

    // ===== 2. 保护机制 =====
    if (fabsf(current_angle) > cfg->protect_limit_deg) {
        stab->enabled = false;
        stab->integral = 0.0f;
        stab->prev_error = 0.0f;
        stab->d_filtered = 0.0f;
        stab->prev_delta = 0;
        stab->ff_filtered = 0.0f;

        // 电机停止到中位
        motor_set_pwm(cfg->motor_neg_index, 1500);
        motor_set_pwm(cfg->motor_pos_index, 1500);

        // 发送保护警告
        if (uart_tx_queue != NULL) {
            char msg[80];
            snprintf(msg, sizeof(msg),
                    "DEBUG:%s-stab PROTECT! Angle=%.1f exceeds limit, disabled\n",
                    axis_label, current_angle);
            xQueueSend(uart_tx_queue, msg, 0);
        }
        return;
    }

    // ===== 3. 计算误差 =====
    float error = 0.0f;
    /* 使用运行时可配置的 target angle（STABLE 指令修改的是 target） */
    float target_angle = cfg->target_angle_deg;
    float runtime_target = 0.0f;
    if (motor_get_target_angle(cfg->axis_name, &runtime_target)) {
        target_angle = runtime_target;
    }

    float angle_error = target_angle - current_angle; /* 目标 - 真实 */
    if (fabsf(angle_error) > cfg->deadband_deg) {
        error = angle_error;
    } else {
        /* 在死区内，衰减积分项 */
        stab->integral *= 0.95f;
    }

    // ===== 4. PID计算 =====
    // 获取当前参数 (运行时 or 配置)
    float kp = stab->use_runtime_pid ? stab->runtime_kp : cfg->kp;
    float ki = stab->use_runtime_pid ? stab->runtime_ki : cfg->ki;
    float kd = stab->use_runtime_pid ? stab->runtime_kd : cfg->kd;
    float kff = stab->use_runtime_pid ? stab->runtime_kff : cfg->kff;
    float kgrav = stab->use_runtime_pid ? stab->runtime_kgrav : cfg->gravity_comp_gain;

    // 微分项 (带滤波)
    float d_raw = (error - stab->prev_error) / fmaxf(dt_s, 1e-3f);
    stab->d_filtered = D_FILTER_ALPHA * d_raw + (1.0f - D_FILTER_ALPHA) * stab->d_filtered;

    // 前馈项 (带滤波)
    float ff_raw = -gyro_rate;
    stab->ff_filtered = FF_FILTER_ALPHA * ff_raw + (1.0f - FF_FILTER_ALPHA) * stab->ff_filtered;
    float ff_term = kff * stab->ff_filtered;

    // PID控制量
    float u_pid = kp * error + ki * stab->integral + kd * stab->d_filtered + ff_term;

    // ===== 5. 重力补偿 =====
    float grav_term = kgrav * sinf((current_angle - cfg->stable_angle_deg) * DEG2RAD);

    // ===== 6. 总控制量 =====
    float u_total = u_pid + cfg->grav_sign * grav_term;

    // ===== 7. 积分更新 =====
    stab->integral += error * dt_s;
    stab->integral = clampf(stab->integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);
    stab->prev_error = error;

    // ===== 8. 控制量 → PWM增量 =====
    int16_t delta = (int16_t)lroundf(u_total * PWM_PER_DEG);

    // 速率限制 (防止突变)
    int16_t d_diff = delta - stab->prev_delta;
    if (d_diff >  DELTA_SLEW_RATE) delta = stab->prev_delta + DELTA_SLEW_RATE;
    if (d_diff < -DELTA_SLEW_RATE) delta = stab->prev_delta - DELTA_SLEW_RATE;
    stab->prev_delta = delta;

    // ===== 9. 计算最终PWM =====
    // 公式: PWM = base + delta_sign * delta + external_offset
    int16_t new_neg = cfg->base_pwm_neg + cfg->delta_sign_neg * delta + stab->external_pwm_offset;
    int16_t new_pos = cfg->base_pwm_pos + cfg->delta_sign_pos * delta + stab->external_pwm_offset;

    // PWM限幅
    int16_t min_neg = cfg->base_pwm_neg - cfg->pwm_adjust_range;
    int16_t max_neg = cfg->base_pwm_neg + cfg->pwm_adjust_range;
    int16_t min_pos = cfg->base_pwm_pos - cfg->pwm_adjust_range;
    int16_t max_pos = cfg->base_pwm_pos + cfg->pwm_adjust_range;

    bool saturated = false;
    if (new_neg < min_neg) { new_neg = min_neg; saturated = true; }
    if (new_neg > max_neg) { new_neg = max_neg; saturated = true; }
    if (new_pos < min_pos) { new_pos = min_pos; saturated = true; }
    if (new_pos > max_pos) { new_pos = max_pos; saturated = true; }

    // ===== 10. 抗饱和 =====
    if (saturated && fabsf(error) > 1.0f) {
        // 如果饱和且误差较大,防止积分继续累积
        if ((error > 0 && stab->integral > 0) || (error < 0 && stab->integral < 0)) {
            stab->integral *= 0.9f;
        }
    }

    // ===== 11. 设置电机PWM =====
    motor_set_pwm(cfg->motor_neg_index, (uint16_t)new_neg);
    motor_set_pwm(cfg->motor_pos_index, (uint16_t)new_pos);

    // ===== 12. 调试输出 =====
    static uint32_t debug_counter = 0;
    debug_counter++;
    if (debug_counter >= 25) {  // 每次都输出,可改为>=50降低频率
        debug_counter = 0;
        if (uart_tx_queue != NULL) {
            char msg[140];
            snprintf(msg, sizeof(msg),
                    "%s:%.1f,e%.1f,pid%.1f,g%.1f,u%.1f,ff%.2f,d%d,M%d=%d,M%d=%d%s\n",
                    axis_label, current_angle, error, u_pid, grav_term, u_total, ff_term,
                    delta, cfg->motor_neg_index, new_neg, cfg->motor_pos_index, new_pos,
                    saturated ? ",SAT!" : "");
            xQueueSend(uart_tx_queue, msg, 0);
        }
    }
}

// =====================================================
// 运行时参数调整
// =====================================================

void axis_stabilizer_set_pid(AxisStabilizer* stab, float kp, float ki, float kd)
{
    if (!stab || !stab->initialized) return;

    stab->use_runtime_pid = true;
    stab->runtime_kp = kp;
    stab->runtime_ki = ki;
    stab->runtime_kd = kd;
}

void axis_stabilizer_set_feedforward(AxisStabilizer* stab, float kff)
{
    if (!stab || !stab->initialized) return;

    stab->use_runtime_pid = true;
    stab->runtime_kff = kff;
}

void axis_stabilizer_set_gravity_comp(AxisStabilizer* stab, float kgrav)
{
    if (!stab || !stab->initialized) return;

    stab->use_runtime_pid = true;
    stab->runtime_kgrav = kgrav;
}

void axis_stabilizer_get_pid(const AxisStabilizer* stab, float* kp, float* ki, float* kd)
{
    if (!stab || !stab->initialized || !kp || !ki || !kd) return;

    if (stab->use_runtime_pid) {
        *kp = stab->runtime_kp;
        *ki = stab->runtime_ki;
        *kd = stab->runtime_kd;
    } else {
        *kp = stab->config->kp;
        *ki = stab->config->ki;
        *kd = stab->config->kd;
    }
}

void axis_stabilizer_reset_to_config(AxisStabilizer* stab)
{
    if (!stab || !stab->initialized) return;

    stab->use_runtime_pid = false;
    stab->runtime_kp = stab->config->kp;
    stab->runtime_ki = stab->config->ki;
    stab->runtime_kd = stab->config->kd;
    stab->runtime_kff = stab->config->kff;
    stab->runtime_kgrav = stab->config->gravity_comp_gain;
}

// =====================================================
// 调试API
// =====================================================

void axis_stabilizer_get_stats(const AxisStabilizer* stab,
                               uint32_t* step_count,
                               uint32_t* fail_count)
{
    if (!stab || !stab->initialized) return;

    if (step_count) *step_count = stab->step_count;
    if (fail_count) *fail_count = stab->fail_count;
}

void axis_stabilizer_reset_integral(AxisStabilizer* stab)
{
    if (!stab || !stab->initialized) return;
    stab->integral = 0.0f;
}
