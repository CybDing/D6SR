/*
 * 双环级联PID控制器 (Cascaded PID Controller)
 *
 * 用于Yaw轴的位置环→速度环双层PID控制
 * 外环: 角度误差 → 速度目标
 * 内环: 速度误差 → PWM增量
 */

#ifndef CASCADED_PID_H
#define CASCADED_PID_H

#include <stdint.h>
#include <stdbool.h>

/* ===== 配置结构体 ===== */

typedef struct {
    // 外环 (位置环) PID参数
    float outer_kp;            // 外环比例增益
    float outer_ki;            // 外环积分增益
    float outer_kd;            // 外环微分增益
    float outer_max_output;    // 速度目标上限 (°/s)
    float outer_deadband;      // 位置死区 (degrees)

    // 内环 (速度环) PID参数
    float inner_kp;            // 内环比例增益
    float inner_ki;            // 内环积分增益
    float inner_kd;            // 内环微分增益
    float inner_max_output;    // PWM增量上限

    // 通用参数
    float dt;                  // 控制周期 (s)
    uint16_t base_pwm_neg;     // M0基准PWM
    uint16_t base_pwm_pos;     // M1基准PWM
    uint16_t pwm_range;        // PWM调整范围 (±range)

    // 重力补偿参数 (可选, gain=0则禁用)
    float gravity_comp_gain;   // 重力补偿增益 Kgrav
    float stable_angle_deg;    // 物理平衡角度 (重力补偿=0的点)
    int8_t grav_sign;          // 重力补偿符号 (+1 或 -1)

    // 控制模式
    bool use_angle_wrap;       // true=Yaw(360度环绕), false=Pitch(无环绕)
} CascadedPIDConfig;

/* ===== 控制器状态结构体 ===== */

typedef struct {
    const CascadedPIDConfig* config;  // 指向配置常量

    // 外环 (位置环) 状态
    float outer_integral;      // 外环积分累积
    float outer_prev_angle;    // 外环上一次角度测量值 (用于derivative on measurement)
    float outer_d_filtered;    // 外环微分滤波值

    // 内环 (速度环) 状态
    float inner_integral;      // 内环积分累积
    float inner_prev_velocity; // 内环上一次速度测量值 (用于derivative on measurement)
    float inner_d_filtered;    // 内环微分滤波值

    // 外环输出速率限制状态
    float prev_velocity_target; // 上一次速度目标 (用于slew rate)

    // 电机索引
    int motor_neg_index;       // 负方向电机索引
    int motor_pos_index;       // 正方向电机索引

    // 目标值和中间变量
    float target_angle;        // 目标角度 (°)
    float velocity_target;     // 速度目标 (°/s, 外环输出)

    // 外部PWM偏移 (motion_controller集成)
    int16_t external_pwm_offset;

    // 速率限制状态
    int16_t prev_pwm_delta;    // 上一次PWM增量

    // 控制标志
    bool enabled;              // 控制器使能标志
    bool initialized;          // 初始化标志
    bool debug_enabled;        // 调试输出使能标志

    // 调试统计
    uint32_t step_count;       // 控制步数计数器

    // 重力补偿输出 (用于调试显示)
    float gravity_comp_output; // 最近一次重力补偿值

    // 轴名称 (用于调试输出区分)
    const char* axis_name;     // "Yaw" 或 "Pitch"
} CascadedPIDController;

/* ===== 函数声明 ===== */

/**
 * 初始化双环PID控制器
 * @param ctrl 控制器实例指针
 * @param cfg 配置常量指针
 * @param motor_neg_idx 负方向电机索引
 * @param motor_pos_idx 正方向电机索引
 */
void cascaded_pid_init(CascadedPIDController* ctrl,
                       const CascadedPIDConfig* cfg,
                       int motor_neg_idx,
                       int motor_pos_idx);

/**
 * 设置控制器使能状态
 * @param ctrl 控制器实例指针
 * @param enabled true=启用, false=禁用
 */
void cascaded_pid_set_enabled(CascadedPIDController* ctrl, bool enabled);

/**
 * 设置目标角度
 * @param ctrl 控制器实例指针
 * @param angle_deg 目标角度 (degrees)
 */
void cascaded_pid_set_target_angle(CascadedPIDController* ctrl, float angle_deg);

/**
 * 设置外部PWM偏移 (用于motion_controller集成)
 * @param ctrl 控制器实例指针
 * @param offset PWM偏移量 (±150)
 */
void cascaded_pid_set_pwm_offset(CascadedPIDController* ctrl, int16_t offset);

/**
 * 执行一次双环PID控制步
 * @param ctrl 控制器实例指针
 * @param current_angle 当前角度 (degrees)
 * @param current_velocity 当前角速度 (degrees/s)
 */
void cascaded_pid_step(CascadedPIDController* ctrl,
                       float current_angle,
                       float current_velocity);

/**
 * 重置控制器状态 (清空积分、微分等)
 * @param ctrl 控制器实例指针
 */
void cascaded_pid_reset(CascadedPIDController* ctrl);

/**
 * 运行时设置外环PID参数
 * @param ctrl 控制器实例指针
 * @param kp 外环比例增益
 * @param ki 外环积分增益
 * @param kd 外环微分增益
 */
void cascaded_pid_set_outer_pid(CascadedPIDController* ctrl, float kp, float ki, float kd);

/**
 * 运行时设置内环PID参数
 * @param ctrl 控制器实例指针
 * @param kp 内环比例增益
 * @param ki 内环积分增益
 * @param kd 内环微分增益
 */
void cascaded_pid_set_inner_pid(CascadedPIDController* ctrl, float kp, float ki, float kd);

/**
 * 设置调试输出使能状态
 * @param ctrl 控制器实例指针
 * @param enabled true=启用调试输出, false=禁用
 */
void cascaded_pid_set_debug_enabled(CascadedPIDController* ctrl, bool enabled);

#endif /* CASCADED_PID_H */
