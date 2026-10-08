/*
 * PID控制模块头文件
 */

#ifndef PID_CONTROL_H
#define PID_CONTROL_H

#include <stdint.h>
#include <stdbool.h>
#include "imu_reader.h"

/* PID参数结构体 */
typedef struct {
    float kp;           /* 比例系数 */
    float ki;           /* 积分系数 */
    float kd;           /* 微分系数 */
} pid_params_t;

/* PID控制器状态 */
typedef struct {
    float error_sum;    /* 误差积分 */
    float last_error;   /* 上次误差 */
    uint32_t last_time; /* 上次计算时间 */
    bool initialized;   /* 初始化标志 */
} pid_state_t;

/* 目标姿态结构体 */
typedef struct {
    float target_roll;   /* 目标横滚角 (°) */
    float target_pitch;  /* 目标俯仰角 (°) */
    float target_yaw;    /* 目标偏航角 (°) */
} target_attitude_t;

/* 函数声明 */
void pid_init(const pid_params_t *params);
void pid_set_params(const pid_params_t *params);
void pid_set_target(const target_attitude_t *target);
void pid_calculate(const imu_data_t *imu_data, const uint16_t *base_pwm, uint16_t *output_pwm);
void pid_reset(void);
void pid_enable(bool enable);
bool pid_is_enabled(void);

/* 辅助函数 */
float calculate_roll(float ax, float ay, float az);
float calculate_pitch(float ax, float ay, float az);

#endif /* PID_CONTROL_H */