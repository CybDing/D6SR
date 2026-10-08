/*
 * 通用轴稳定器框架
 *
 * 功能: 为Roll/Pitch/Yaw三轴提供统一的PID稳定控制
 *
 * 设计原理:
 * - 从 motor_calibration.h 读取轴配置参数
 * - PID控制 + 重力补偿(可选) + 前馈控制(可选)
 * - 支持运动控制PWM偏移量叠加
 * - 统一的保护机制和调试输出
 *
 * 使用方法:
 * 1. 定义轴稳定器实例: AxisStabilizer roll_stab;
 * 2. 初始化: axis_stabilizer_init(&roll_stab, &ROLL_AXIS_CONFIG);
 * 3. 启用: axis_stabilizer_set_enabled(&roll_stab, true);
 * 4. 定期调用: axis_stabilizer_step(&roll_stab, dt_s);
 */

#ifndef AXIS_STABILIZER_H
#define AXIS_STABILIZER_H

#include <stdbool.h>
#include <stdint.h>
#include "motor_calibration.h"

// =====================================================
// 数据结构定义
// =====================================================

/**
 * @brief 轴稳定器内部状态
 */
typedef struct {
    // 配置指针
    const AxisControlConfig* config;

    // 控制状态
    bool enabled;
    bool initialized;

    // PID状态
    float integral;           // 积分项
    float prev_error;         // 上次误差
    float d_filtered;         // 滤波后的微分项
    int16_t prev_delta;       // 上次PWM增量

    // 前馈状态
    float ff_filtered;        // 滤波后的前馈项

    // 运行时可调参数 (覆盖配置值)
    bool use_runtime_pid;     // 是否使用运行时PID参数
    float runtime_kp;
    float runtime_ki;
    float runtime_kd;
    float runtime_kff;
    float runtime_kgrav;

    // 统计信息
    uint32_t step_count;      // 步数计数
    uint32_t fail_count;      // IMU读取失败计数

    // 外部PWM偏移量(用于运动控制)
    int16_t external_pwm_offset;

    // Yaw角度滤波器状态 (仅Yaw轴使用)
    float yaw_filtered;           // 滤波后的Yaw角度
    bool yaw_filter_initialized;  // Yaw滤波器是否已初始化
} AxisStabilizer;

// =====================================================
// 公共API
// =====================================================

/**
 * @brief 初始化轴稳定器
 * @param stab 稳定器实例指针
 * @param config 轴配置参数(来自motor_calibration.h)
 */
void axis_stabilizer_init(AxisStabilizer* stab, const AxisControlConfig* config);

/**
 * @brief 执行一次控制步进
 * @param stab 稳定器实例指针
 * @param dt_s 时间步长(秒)
 */
void axis_stabilizer_step(AxisStabilizer* stab, float dt_s);

/**
 * @brief 启用/禁用稳定器
 * @param stab 稳定器实例指针
 * @param enabled true=启用, false=禁用
 */
void axis_stabilizer_set_enabled(AxisStabilizer* stab, bool enabled);

/**
 * @brief 获取稳定器启用状态
 * @param stab 稳定器实例指针
 * @return true=已启用, false=已禁用
 */
bool axis_stabilizer_get_enabled(const AxisStabilizer* stab);

/**
 * @brief 设置外部PWM偏移量(用于运动控制)
 * @param stab 稳定器实例指针
 * @param offset PWM偏移量
 *
 * 说明: 最终PWM = 基准PWM + 差速delta + 外部偏移量
 *      外部偏移量用于叠加运动控制的整体推力
 */
void axis_stabilizer_set_pwm_offset(AxisStabilizer* stab, int16_t offset);

// =====================================================
// 运行时参数调整API (用于在线调试)
// =====================================================

/**
 * @brief 设置运行时PID参数(覆盖配置文件)
 * @param stab 稳定器实例指针
 * @param kp, ki, kd PID参数
 */
void axis_stabilizer_set_pid(AxisStabilizer* stab, float kp, float ki, float kd);

/**
 * @brief 设置前馈增益
 */
void axis_stabilizer_set_feedforward(AxisStabilizer* stab, float kff);

/**
 * @brief 设置重力补偿增益
 */
void axis_stabilizer_set_gravity_comp(AxisStabilizer* stab, float kgrav);

/**
 * @brief 获取当前PID参数
 */
void axis_stabilizer_get_pid(const AxisStabilizer* stab, float* kp, float* ki, float* kd);

/**
 * @brief 重置为配置文件参数
 */
void axis_stabilizer_reset_to_config(AxisStabilizer* stab);

// =====================================================
// 调试API
// =====================================================

/**
 * @brief 获取稳定器统计信息
 */
void axis_stabilizer_get_stats(const AxisStabilizer* stab,
                               uint32_t* step_count,
                               uint32_t* fail_count);

/**
 * @brief 重置积分项(用于故障恢复)
 */
void axis_stabilizer_reset_integral(AxisStabilizer* stab);

#endif /* AXIS_STABILIZER_H */
