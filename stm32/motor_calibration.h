/*
 * 电机控制PWM符号配置系统
 *
 * 两个符号：
 *
 * 1. 差速控制符号 (delta_sign)
 *    - 哪个电机 += delta，哪个电机 -= delta
 *    - 可以整体反号来调整控制方向
 *    - 设为0则禁用该电机的差速调整
 *
 * 2. 重力补偿符号 (grav_sign)
 *    - 控制量计算：u = u_pid + grav_sign * grav_term
 *    - 需要根据物理分析确定正负
 */

#ifndef MOTOR_CALIBRATION_H
#define MOTOR_CALIBRATION_H

#include <stdint.h>
#include <stdbool.h>
#include "cascaded_pid.h"

typedef struct {
    const char* axis_name;       // 轴名称

    // 电机索引
    uint8_t motor_neg_index;     // 负向电机索引
    uint8_t motor_pos_index;     // 正向电机索引

    // 基准PWM (实测得到的最佳工作点)
    int16_t base_pwm_neg;
    int16_t base_pwm_pos;

    // ===== 符号配置 =====

    // 差速控制符号
    // PWM计算: new_neg = base_neg + delta_sign_neg * delta
    //         new_pos = base_pos + delta_sign_pos * delta
    int8_t delta_sign_neg;       // M_neg的delta符号 (+1 or -1 or 0)
    int8_t delta_sign_pos;       // M_pos的delta符号 (+1 or -1 or 0)

    // 重力补偿符号
    // 控制量: u = u_pid + grav_sign * grav_term
    int8_t grav_sign;            // 重力补偿符号 (+1 or -1)

    // 控制参数
    float stable_angle_deg;      // 自然平衡角度
    float target_angle_deg;      // 运行时目标角度 (可通过 STABLE 指令修改)
    float gravity_comp_gain;     // 重力补偿增益 (g_kgrav_deg)
    float kp, ki, kd, kff;       // PID参数

    // 限制参数
    int16_t pwm_adjust_range;
    float deadband_deg;
    float protect_limit_deg;

} AxisControlConfig;

// =====================================================
// Roll轴配置 (从y_stabilizer.c)
// =====================================================
/*
 * Roll轴控制说明:
 * - 电机: M2(索引2), M3(索引3)
 * - 基准PWM: M2=1560, M3=1440
 * - 控制方程:
 *     new_neg = BASE_PWM_NEG - delta  → M2 = 1560 - delta
 *     new_pos = BASE_PWM_POS + delta  → M3 = 1440 + delta
 * - 重力补偿: u = u_pid + grav_term
 *
 * 符号配置:
 * - delta_sign_neg = -1 (M2 = base - delta)
 * - delta_sign_pos = +1 (M3 = base + delta)
 * - grav_sign = +1 (u = u_pid + grav_term)
 */
extern AxisControlConfig ROLL_AXIS_CONFIG;

// =====================================================
// Pitch轴配置 (仅用于前后运动,不需要PID稳定)
// =====================================================
/*
 * Pitch轴说明:
 * - 功能: 控制前后运动
 * - 电机: M4, M5 (索引4, 5)
 * - 控制方式: 直接PWM偏移,不需要PID稳定(震荡不大)
 * - 基准PWM: M4=M5=1650 (对称)
 * - 运动控制: 前后运动时统一增减PWM偏移量
 */
extern AxisControlConfig PITCH_AXIS_CONFIG;

// =====================================================
// Yaw轴配置
// =====================================================
/*
 * Yaw轴控制说明:
 * - 目标: 保持Yaw角度稳定(默认0度),防止机器人自旋
 * - 电机: M5(索引4), M6(索引5)
 * - 基准PWM: M5=1500, M6=1520 (轻微不对称,改善死区响应)
 * - 差速控制: M5减少PWM, M6增加PWM (根据实际测试调整)
 * - 无重力补偿: Yaw是水平旋转,无重力影响
 * - PID参数: 需要实际调试,建议从小开始(Kp=0.3, Ki=0.05, Kd=0.08)
 *
 * 注意: 基准PWM轻微不对称的原因:
 *   1. 克服摩擦力和惯性,提供最小预紧力
 *   2. 改善死区内的控制响应性
 *   3. 偏移量小(20),不会显著影响Roll/Pitch运动
 *   4. 可根据实际测试调整M6的基准值(1510-1530)
 */
extern AxisControlConfig YAW_AXIS_CONFIG;

// =====================================================
// Yaw轴双环PID配置 (Cascaded PID for Yaw axis)
// =====================================================
/*
 * Yaw轴双环级联PID控制配置
 * - 外环: 角度误差 → 速度目标 (±60°/s)
 * - 内环: 速度误差 → PWM增量
 * - 适用于大角度调整时的平稳控制
 * - 基准PWM修复为1500 (对称控制)
 * - delta符号修复: neg=-1, pos=+1 (对称差速)
 * - 无重力补偿 (Yaw是水平旋转,无重力影响)
 * - use_angle_wrap=true (360度环绕处理)
 */
extern const CascadedPIDConfig YAW_CASCADED_PID_CONFIG;

// =====================================================
// Pitch轴双环PID配置 (Cascaded PID for Pitch axis)
// =====================================================
/*
 * Pitch轴双环级联PID控制配置
 * - 外环: 角度误差 → 速度目标
 * - 内环: 速度误差 → PWM增量
 * - 电机: M4(索引4), M5(索引5)
 * - 基准PWM: M4=1650, M5=1650
 * - 重力补偿: 启用 (gravity_comp_gain > 0)
 * - use_angle_wrap=false (无360度环绕)
 */
extern const CascadedPIDConfig PITCH_CASCADED_PID_CONFIG;


/**
 * @brief 计算电机PWM值 (统一接口)
 *
 * @param config 轴配置
 * @param delta 控制增量
 * @param is_neg_motor true=负向电机, false=正向电机
 * @return 电机PWM值
 *
 * 公式:
 *   new_neg = base_neg + delta_sign_neg * delta
 *   new_pos = base_pos + delta_sign_pos * delta
 */
static inline int16_t motor_calc_pwm_from_delta(
    const AxisControlConfig* config,
    int16_t delta,
    bool is_neg_motor)
{
    if (is_neg_motor) {
        return config->base_pwm_neg + config->delta_sign_neg * delta;
    } else {
        return config->base_pwm_pos + config->delta_sign_pos * delta;
    }
}

/**
 * @brief 验证配置与ystab一致性
 *
 * @param config Roll轴配置
 * @param delta 测试用的delta值
 *
 * 测试: 给定delta=100，计算PWM应该得到:
 *   M3 = 1570 - 100 = 1470
 *   M4 = 1430 + 100 = 1530
 */
static inline bool verify_roll_config(const AxisControlConfig* config, int16_t delta) {
    int16_t m3_pwm = motor_calc_pwm_from_delta(config, delta, true);
    int16_t m4_pwm = motor_calc_pwm_from_delta(config, delta, false);

    // 期望值 (根据ystab代码)
    int16_t expected_m3 = 1570 - delta;
    int16_t expected_m4 = 1430 + delta;

    return (m3_pwm == expected_m3) && (m4_pwm == expected_m4);
}

/* Runtime API */
void motor_calibration_init(void);

/**
 * 设置指定轴的stable angle（度，配置文件含义，用于重力补偿）
 */
bool motor_set_stable_angle(const char *axis_name, float angle_deg);

/**
 * 获取指定轴的stable angle
 */
bool motor_get_stable_angle(const char *axis_name, float *out_angle);

/**
 * 设置/查询运行时的 target angle（STABLE指令修改的是target angle）
 */
bool motor_set_target_angle(const char *axis_name, float angle_deg);
bool motor_get_target_angle(const char *axis_name, float *out_angle);

#endif /* MOTOR_CALIBRATION_H */
