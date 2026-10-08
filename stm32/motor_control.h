/*
 * 电机控制模块头文件
 */

#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

/* 电机数量 */
#define NUM_MOTORS      6

/* PWM平滑控制参数 */
#define PWM_RAMP_STEP   1       /* 每次最大变化步长 (可调整) */

/* 函数声明 */
void motor_control_init(void);
void motor_set_pwm(int motor_id, uint16_t pwm_value);
void motor_set_all_pwm(uint16_t *pwm_values);
void motor_emergency_stop(void);
uint16_t motor_get_pwm(int motor_id);
void motor_debug_status(void);  /* PWM调试函数 */
void motor_gpio_test(void);     /* GPIO引脚测试 */
void motor_pwm_restore(void);   /* PWM模式恢复 */

/* PWM平滑控制 */
void motor_set_target_pwm(int motor_id, uint16_t target_pwm);
void motor_set_all_target_pwm(uint16_t *target_pwm);
void motor_update_smooth(void); /* 平滑更新函数，需在控制任务中周期调用 */
void motor_set_smooth_enabled(bool enabled);
bool motor_get_smooth_enabled(void);
uint16_t motor_get_target_pwm(int motor_id);

#endif /* MOTOR_CONTROL_H */