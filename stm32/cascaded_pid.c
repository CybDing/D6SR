/*
 * 双环级联PID控制器实现
 *
 * 控制流程:
 * 1. 外环 (位置环): 角度误差 → 速度目标
 * 2. 内环 (速度环): 速度误差 → PWM增量
 * 3. PWM输出: 基准PWM + 增量*符号 + 外部偏移
 *
 * ============================================================================
 * FIXME (kinematic coupling — to be fixed, see branch `test_jacobian`)
 * ============================================================================
 * 当前在 motion_controller.c 中对 roll / pitch / yaw 三个轴各自独立调用
 * cascaded_pid_step()，外环输出的速度目标被直接当作内环要跟踪的目标，并与
 * IMU 测得的体坐标系角速度做差。这一做法忽略了欧拉角速率与体坐标系角速度
 * 之间的运动学耦合：
 *
 *     omega_body = E(phi, theta) * [r_dot, p_dot, y_dot]^T
 *
 *     E(phi, theta) =
 *         [[1,    0,         -sin(theta)              ],
 *          [0,    cos(phi),   sin(phi)*cos(theta)     ],
 *          [0,   -sin(phi),   cos(phi)*cos(theta)     ]]
 *
 *     phi = roll, theta = pitch
 *
 * 仅当 phi ~= theta ~= 0 时 E ~= I，独立轴 PID 才近似成立。当机器人显著倾斜
 * 时，三轴控制目标会出现串扰，姿态稳定与跟踪精度下降。
 *
 * 仿真侧 (simulation/mujoco_sim/control/cascaded_pid.py) 已存在等价的 FIXME
 * 与对应修复方案，详见 `test_jacobian` 分支。固件侧需要同步重构：
 *   1. 拆分 cascaded_pid_step() 为 cascaded_pid_outer_step() 与
 *      cascaded_pid_inner_step()。
 *   2. 在 motion_controller.c 中插入欧拉速率到体角速率的 E 变换。
 *   3. PID 增益需要重新整定。
 * ============================================================================
 */

#include "cascaded_pid.h"
#include "motor_control.h"
#include "FreeRTOS.h"
#include "queue.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// ===== 常量定义 =====

// 微分滤波系数 (低通滤波)
#define D_FILTER_ALPHA  0.2f

// PWM速率限制 (防止突变)
#define PWM_SLEW_RATE   10

// 外环输出(速度目标)速率限制 (°/s per step)
#define VELOCITY_TARGET_SLEW_RATE  5.0f

// 外环积分限幅
#define OUTER_INTEGRAL_LIMIT  100.0f

// 内环积分限幅
#define INNER_INTEGRAL_LIMIT  100.0f

// 调试输出频率分频 (每N次输出一次)
#define DEBUG_OUTPUT_DIVIDER  25

// 角度转弧度
#define DEG2RAD  0.017453292f

// 角度归一化到0-360范围
static inline float normalize_angle_360(float angle) {
    while (angle < 0.0f) angle += 360.0f;
    while (angle >= 360.0f) angle -= 360.0f;
    return angle;
}

/**
 * 计算Yaw角度误差 (考虑360度环绕)
 *
 * @param target 目标角度 (0-360度)
 * @param current 当前角度 (0-360度)
 * @return 最短路径的角度误差 (-180 ~ +180度)
 *
 * 原理:
 * - 正误差: 需要正向旋转 (逆时针)
 * - 负误差: 需要负向旋转 (顺时针)
 * - 保证误差绝对值 ≤ 180度 (最短路径)
 *
 * 示例:
 * - target=10, current=350 → error=+20 (不是-340, 正向转20度更近)
 * - target=350, current=10 → error=-20 (不是+340, 负向转20度更近)
 * - target=180, current=0  → error=+180 或 -180 (两个方向等价)
 */
static inline float calculate_angle_error(float target, float current) {
    // 先归一化到0-360范围
    target = normalize_angle_360(target);
    current = normalize_angle_360(current);

    // 计算初始误差
    float error = target - current;

    // 处理环绕: 如果误差 > 180度, 说明反向更近
    if (error > 180.0f) {
        error -= 360.0f;  // 例: error=340 → -20
    } else if (error < -180.0f) {
        error += 360.0f;  // 例: error=-340 → +20
    }

    return error;  // 范围: -180 ~ +180
}

// ===== 外部声明 =====

extern QueueHandle_t uart_tx_queue;  // UART发送队列

// ===== 内部辅助函数 =====

/**
 * 浮点数限幅
 */
static inline float clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

/**
 * 整数限幅
 */
static inline int16_t clampi(int16_t x, int16_t lo, int16_t hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

// ===== API实现 =====

void cascaded_pid_init(CascadedPIDController* ctrl,
                       const CascadedPIDConfig* cfg,
                       int motor_neg_idx,
                       int motor_pos_idx)
{
    if (!ctrl || !cfg) return;

    // 绑定配置
    ctrl->config = cfg;

    // 电机索引
    ctrl->motor_neg_index = motor_neg_idx;
    ctrl->motor_pos_index = motor_pos_idx;

    // 重置所有状态
    ctrl->outer_integral = 0.0f;
    ctrl->outer_prev_angle = 0.0f;
    ctrl->outer_d_filtered = 0.0f;

    ctrl->inner_integral = 0.0f;
    ctrl->inner_prev_velocity = 0.0f;
    ctrl->inner_d_filtered = 0.0f;

    ctrl->target_angle = 0.0f;
    ctrl->velocity_target = 0.0f;
    ctrl->prev_velocity_target = 0.0f;

    ctrl->external_pwm_offset = 0;
    ctrl->prev_pwm_delta = 0;

    // 控制标志
    ctrl->enabled = false;
    ctrl->initialized = true;
    ctrl->debug_enabled = true;  // 默认启用调试输出

    // 统计
    ctrl->step_count = 0;

    // 重力补偿输出
    ctrl->gravity_comp_output = 0.0f;

    // 轴名称默认值
    ctrl->axis_name = "Unknown";
}

void cascaded_pid_set_enabled(CascadedPIDController* ctrl, bool enabled)
{
    if (!ctrl || !ctrl->initialized) return;

    ctrl->enabled = enabled;

    if (!enabled) {
        // 禁用时重置所有状态
        ctrl->outer_integral = 0.0f;
        ctrl->outer_prev_angle = 0.0f;
        ctrl->outer_d_filtered = 0.0f;

        ctrl->inner_integral = 0.0f;
        ctrl->inner_prev_velocity = 0.0f;
        ctrl->inner_d_filtered = 0.0f;

        ctrl->velocity_target = 0.0f;
        ctrl->prev_velocity_target = 0.0f;
        ctrl->prev_pwm_delta = 0;

        // 电机回中位
        motor_set_pwm(ctrl->motor_neg_index, 1500);
        motor_set_pwm(ctrl->motor_pos_index, 1500);
    }
}

void cascaded_pid_set_target_angle(CascadedPIDController* ctrl, float angle_deg)
{
    if (!ctrl || !ctrl->initialized) return;
    ctrl->target_angle = angle_deg;
}

void cascaded_pid_set_pwm_offset(CascadedPIDController* ctrl, int16_t offset)
{
    if (!ctrl || !ctrl->initialized) return;
    ctrl->external_pwm_offset = offset;
}

void cascaded_pid_step(CascadedPIDController* ctrl,
                       float current_angle,
                       float current_velocity)
{
    if (!ctrl || !ctrl->initialized || !ctrl->enabled) return;

    const CascadedPIDConfig* cfg = ctrl->config;
    ctrl->step_count++;

    // ===== 外环 (位置环) 计算 =====

    // 1. 计算角度误差
    float angle_error;
    if (cfg->use_angle_wrap) {
        // Yaw: 360度环绕, 取最短路径 (-180 ~ +180)
        angle_error = calculate_angle_error(ctrl->target_angle, current_angle);
    } else {
        // Pitch: 无环绕, 直接相减
        angle_error = ctrl->target_angle - current_angle;
    }

    // 2. 死区处理
    if (fabsf(angle_error) < cfg->outer_deadband) {
        angle_error = 0.0f;
        ctrl->outer_integral *= 0.95f;  // 死区内衰减积分
    }

    // 3. 外环PID计算
    float outer_p = cfg->outer_kp * angle_error;

    // 积分项 (带抗饱和)
    ctrl->outer_integral += angle_error * cfg->dt;
    ctrl->outer_integral = clampf(ctrl->outer_integral,
                                  -OUTER_INTEGRAL_LIMIT,
                                  OUTER_INTEGRAL_LIMIT);
    float outer_i = cfg->outer_ki * ctrl->outer_integral;

    // 微分项 (Derivative on Measurement, 带低通滤波)
    // 对测量值微分而不是误差微分，避免target阶跃时D项spike
    // d(error)/dt = d(target - angle)/dt = -d(angle)/dt (当target不变时)
    // 所以用 -(current_angle - prev_angle) 来得到等效的误差微分方向
    float outer_d_raw = -(current_angle - ctrl->outer_prev_angle) / fmaxf(cfg->dt, 1e-3f);
    ctrl->outer_d_filtered = D_FILTER_ALPHA * outer_d_raw +
                            (1.0f - D_FILTER_ALPHA) * ctrl->outer_d_filtered;
    float outer_d = cfg->outer_kd * ctrl->outer_d_filtered;

    // 4. 外环输出: 速度目标 (限幅)
    float velocity_target_raw = clampf(outer_p + outer_i + outer_d,
                                       -cfg->outer_max_output,
                                       cfg->outer_max_output);

    // 5. 外环输出slew rate限制 (防止速度目标突变导致内环震荡)
    float v_diff = velocity_target_raw - ctrl->prev_velocity_target;
    if (v_diff > VELOCITY_TARGET_SLEW_RATE) {
        ctrl->velocity_target = ctrl->prev_velocity_target + VELOCITY_TARGET_SLEW_RATE;
    } else if (v_diff < -VELOCITY_TARGET_SLEW_RATE) {
        ctrl->velocity_target = ctrl->prev_velocity_target - VELOCITY_TARGET_SLEW_RATE;
    } else {
        ctrl->velocity_target = velocity_target_raw;
    }
    ctrl->prev_velocity_target = ctrl->velocity_target;

    // 6. 更新外环历史
    ctrl->outer_prev_angle = current_angle;

    // ===== 内环 (速度环) 计算 =====
    // ctrl->velocity_target = -40.0f;
    // 1. 计算速度误差
    float velocity_error = ctrl->velocity_target - current_velocity;


    // 2. 内环PID计算
    float inner_p = cfg->inner_kp * velocity_error;

    // 积分项 (带抗饱和)
    ctrl->inner_integral += velocity_error * cfg->dt;
    ctrl->inner_integral = clampf(ctrl->inner_integral,
                                  -INNER_INTEGRAL_LIMIT,
                                  INNER_INTEGRAL_LIMIT);
    float inner_i = cfg->inner_ki * ctrl->inner_integral;

    // 微分项 (Derivative on Measurement, 带低通滤波)
    // 对速度测量值微分而不是误差微分，避免velocity_target变化时D项spike
    // d(error)/dt = d(target - velocity)/dt = -d(velocity)/dt (当target不变时)
    float inner_d_raw = -(current_velocity - ctrl->inner_prev_velocity) / fmaxf(cfg->dt, 1e-3f);
    ctrl->inner_d_filtered = D_FILTER_ALPHA * inner_d_raw +
                            (1.0f - D_FILTER_ALPHA) * ctrl->inner_d_filtered;
    float inner_d = cfg->inner_kd * ctrl->inner_d_filtered;

    // 3. 内环输出: PWM增量 (限幅)
    float pwm_delta_raw = clampf((inner_p + inner_i + inner_d),
                                 -cfg->inner_max_output,
                                 cfg->inner_max_output);

    // 4. 重力补偿计算 (仅当 gravity_comp_gain != 0 时启用)
    float grav_comp = 0.0f;
    if (fabsf(cfg->gravity_comp_gain) > 0.001f) {
        // 重力补偿 = Kgrav * sin(angle - stable_angle)
        // 当 angle > stable_angle 时, 需要正向力矩抵抗重力
        grav_comp = cfg->gravity_comp_gain *
                    sinf((current_angle - cfg->stable_angle_deg) * DEG2RAD);
        ctrl->gravity_comp_output = grav_comp;
    } else {
        ctrl->gravity_comp_output = 0.0f;
    }

    // 5. 合并PID输出和重力补偿
    float pwm_delta_with_grav = pwm_delta_raw + cfg->grav_sign * grav_comp;
    pwm_delta_with_grav = clampf(pwm_delta_with_grav,
                                 -cfg->inner_max_output,
                                 cfg->inner_max_output);

    // 6. 速率限制 (防止突变)
    int16_t pwm_delta = (int16_t)lroundf(pwm_delta_with_grav);
    int16_t d_diff = pwm_delta - ctrl->prev_pwm_delta;
    if (d_diff >  PWM_SLEW_RATE) pwm_delta = ctrl->prev_pwm_delta + PWM_SLEW_RATE;
    if (d_diff < -PWM_SLEW_RATE) pwm_delta = ctrl->prev_pwm_delta - PWM_SLEW_RATE;
    ctrl->prev_pwm_delta = pwm_delta;

    // 5. 更新内环历史
    ctrl->inner_prev_velocity = current_velocity;

    // ===== PWM输出计算 =====

    /*
     * "单电机增速"策略:
     * - 基准PWM=1550 (保证电机始终能转动)
     * - 根据delta方向,只有一个电机增速,另一个保持基准或减速到基准
     * - delta > 0: M_pos增速, M_neg保持基准
     * - delta < 0: M_neg增速, M_pos保持基准
     * - 避免死区问题,确保方向控制可靠
     */

    int16_t pwm_neg, pwm_pos;

    if (pwm_delta > 0) {
        // 正向控制: M_pos增速, M_neg保持基准
        pwm_pos = cfg->base_pwm_pos + pwm_delta + ctrl->external_pwm_offset;
        pwm_neg = cfg->base_pwm_neg + ctrl->external_pwm_offset;
    } else if (pwm_delta < 0) {
        // 负向控制: M_neg增速(delta为负), M_pos保持基准
        pwm_neg = cfg->base_pwm_neg - pwm_delta + ctrl->external_pwm_offset;  
        pwm_pos = cfg->base_pwm_pos + ctrl->external_pwm_offset;
    } else {
        // delta=0: 两个电机都保持基准
        pwm_neg = cfg->base_pwm_neg + ctrl->external_pwm_offset;
        pwm_pos = cfg->base_pwm_pos + ctrl->external_pwm_offset;
    }

    // PWM限幅 (1400~1700范围)
    int16_t min_pwm = cfg->base_pwm_neg - cfg->pwm_range;
    int16_t max_pwm = cfg->base_pwm_pos + cfg->pwm_range;

    pwm_neg = clampi(pwm_neg, min_pwm, max_pwm);
    pwm_pos = clampi(pwm_pos, min_pwm, max_pwm);

    // 3. 设置电机PWM
    motor_set_pwm(ctrl->motor_neg_index, (uint16_t)pwm_neg);
    motor_set_pwm(ctrl->motor_pos_index, (uint16_t)pwm_pos);

    // ===== 调试输出 =====

    // 只在调试使能时输出
    if (ctrl->debug_enabled) {
        static uint32_t debug_counter = 0;
        debug_counter++;

        if (debug_counter >= DEBUG_OUTPUT_DIVIDER) {
            debug_counter = 0;

            if (uart_tx_queue != NULL) {
                char msg[180];
                // 根据是否启用angle_wrap选择显示格式
                float display_angle = cfg->use_angle_wrap ?
                                      normalize_angle_360(current_angle) : current_angle;

                snprintf(msg, sizeof(msg),
                        "%s-Cascade: ang=%.1f, tgt=%.1f, aerr=%.1f, "
                        "v_fb=%.1f, vtgt=%.1f, verr=%.1f, grav=%.1f, "
                        "delta=%d, M%d=%d, M%d=%d\n",
                        ctrl->axis_name,
                        display_angle,
                        ctrl->target_angle,
                        angle_error,
                        current_velocity,
                        ctrl->velocity_target,
                        velocity_error,
                        ctrl->gravity_comp_output,
                        pwm_delta,
                        ctrl->motor_neg_index, pwm_neg,
                        ctrl->motor_pos_index, pwm_pos);
                xQueueSend(uart_tx_queue, msg, 0);
            }
        }
    }
}

void cascaded_pid_reset(CascadedPIDController* ctrl)
{
    if (!ctrl || !ctrl->initialized) return;

    // 重置所有状态变量
    ctrl->outer_integral = 0.0f;
    ctrl->outer_prev_angle = 0.0f;
    ctrl->outer_d_filtered = 0.0f;

    ctrl->inner_integral = 0.0f;
    ctrl->inner_prev_velocity = 0.0f;
    ctrl->inner_d_filtered = 0.0f;

    ctrl->velocity_target = 0.0f;
    ctrl->prev_velocity_target = 0.0f;
    ctrl->prev_pwm_delta = 0;
}

void cascaded_pid_set_outer_pid(CascadedPIDController* ctrl, float kp, float ki, float kd)
{
    if (!ctrl || !ctrl->initialized) return;

    // 注意: 这里需要修改const配置
    // 实际上应该使用运行时参数覆盖机制
    // 简化实现: 直接修改配置 (需要去除const)
    // 更好的做法: 在控制器中添加运行时参数字段

    // 由于配置是const，这里仅作为占位符
    // 实际使用时需要添加runtime_outer_kp等字段
    (void)kp;
    (void)ki;
    (void)kd;

    // TODO: 实现运行时参数覆盖机制
}

void cascaded_pid_set_inner_pid(CascadedPIDController* ctrl, float kp, float ki, float kd)
{
    if (!ctrl || !ctrl->initialized) return;

    // 同上，需要添加运行时参数覆盖机制
    (void)kp;
    (void)ki;
    (void)kd;

    // TODO: 实现运行时参数覆盖机制
}

void cascaded_pid_set_debug_enabled(CascadedPIDController* ctrl, bool enabled)
{
    if (!ctrl || !ctrl->initialized) return;
    ctrl->debug_enabled = enabled;
}
