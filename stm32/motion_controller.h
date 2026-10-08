/*
 * 运动控制器头文件 (简化版)
 *
 * 功能: 提供X/Y方向运动的PWM偏移量
 *
 * 设计理念:
 * - 不需要精确速度控制,只需产生加速度
 * - 通过设置基准PWM偏移量实现运动
 * - 稳定器在偏移量基础上进行差速平衡控制
 *
 * 使用方法:
 * 1. 初始化: motion_controller_init()
 * 2. 启用: motion_set_enabled(true)
 * 3. 设置速度: motion_set_velocity(vx, vy)  // -1.0 ~ +1.0
 * 4. 稳定器通过motion_get_xxx_pwm_offset()获取偏移量
 * 5. 最终PWM = base_pwm + delta_sign * delta + motion_offset
 */

#ifndef MOTION_CONTROLLER_H
#define MOTION_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief 初始化运动控制器
 */
void motion_controller_init(void);

/**
 * @brief 设置Roll轴目标角度 (动态调整)
 * @param target_angle_deg 目标角度(度)
 *
 * 用途: 控制机器人在Roll方向的运动
 * 例如: 设置为20度会使机器人产生持续的前进/后退驱动力
 */
void motion_set_roll_target(float target_angle_deg);

/**
 * @brief 设置Pitch轴目标角度 (动态调整)
 * @param target_angle_deg 目标角度(度)
 *
 * 用途: 控制机器人在Pitch方向的运动
 */
void motion_set_pitch_target(float target_angle_deg);

/**
 * @brief 设置速度控制模式
 * @param velocity_x X方向速度 (-1.0 到 +1.0, 归一化)
 * @param velocity_y Y方向速度 (-1.0 到 +1.0, 归一化)
 *
 * 功能: 根据期望速度自动计算并设置目标角度
 *
 * 示例:
 *   motion_set_velocity(0.5, 0.0);  // 50%速度向前运动
 *   motion_set_velocity(-0.3, 0.0); // 30%速度向后运动
 *   motion_set_velocity(0.0, 0.5);  // 50%速度向左运动
 */
void motion_set_velocity(float velocity_x, float velocity_y);

/**
 * @brief 设置Yaw旋转速度
 * @param omega_z 旋转角速度 (-1.0 到 +1.0, 归一化)
 *
 * 功能: 控制机器人原地旋转
 * 方法: 通过M5/M6差速驱动
 *
 * 示例:
 *   motion_set_yaw_rate(0.5);  // 50%速度顺时针旋转
 *   motion_set_yaw_rate(-0.5); // 50%速度逆时针旋转
 */
void motion_set_yaw_rate(float omega_z);

/**
 * @brief 停止所有运动
 *
 * 功能: 将所有轴恢复到自然平衡角度
 */
void motion_stop(void);

/**
 * @brief 获取当前Roll目标角度
 * @return 当前Roll轴目标角度
 */
float motion_get_roll_target(void);

/**
 * @brief 获取当前Pitch目标角度
 * @return 当前Pitch轴目标角度
 */
float motion_get_pitch_target(void);

/**
 * @brief 启用/禁用运动控制器
 * @param enabled true启用，false禁用
 */
void motion_set_enabled(bool enabled);

/**
 * @brief 获取运动控制器启用状态
 * @return true已启用，false已禁用
 */
bool motion_get_enabled(void);

// =====================================================
// PWM偏移量获取接口 (供axis_stabilizer调用)
// =====================================================

/**
 * @brief 获取Pitch轴PWM偏移量 (用于X方向前后运动)
 * @return PWM偏移量 (-150 ~ +150)
 */
int16_t motion_get_pitch_pwm_offset(void);

/**
 * @brief 获取Roll轴PWM偏移量 (Roll轴不接收运动偏移)
 * @return 0 (Roll轴仅做角度稳定)
 */
int16_t motion_get_roll_pwm_offset(void);

/**
 * @brief 获取Yaw轴PWM偏移量 (用于Y方向左右运动)
 * @return PWM偏移量 (-150 ~ +150)
 */
int16_t motion_get_yaw_pwm_offset(void);

#endif /* MOTION_CONTROLLER_H */
