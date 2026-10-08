/*
 * PID控制模块实现
 */

#include "pid_control.h"
#include "pin_config.h"
#include <math.h>
#include <string.h>

/* PID控制器状态 */
static pid_params_t pid_params = {1.0f, 0.1f, 0.05f};
static pid_state_t roll_pid = {0};
static pid_state_t pitch_pid = {0};
static pid_state_t yaw_pid = {0};

/* 目标姿态 */
static target_attitude_t target_attitude = {0.0f, 0.0f, 0.0f};

/* 控制使能 */
static bool pid_enabled = false;

/* 常量定义 */
#define MAX_ERROR_SUM       1000.0f     /* 积分限幅 */
#define MAX_PWM_ADJUSTMENT  200         /* 最大PWM调整量 */
#define RAD_TO_DEG          57.2958f    /* 弧度转角度 */

/*********************************************************************
 * PID控制器初始化
 *********************************************************************/
void pid_init(const pid_params_t *params) {
    if (params) {
        pid_params = *params;
    }
    
    /* 重置PID状态 */
    memset(&roll_pid, 0, sizeof(pid_state_t));
    memset(&pitch_pid, 0, sizeof(pid_state_t));
    memset(&yaw_pid, 0, sizeof(pid_state_t));
    
    pid_enabled = false;
}

/*********************************************************************
 * 设置PID参数
 *********************************************************************/
void pid_set_params(const pid_params_t *params) {
    if (params) {
        pid_params = *params;
    }
}

/*********************************************************************
 * 设置目标姿态
 *********************************************************************/
void pid_set_target(const target_attitude_t *target) {
    if (target) {
        target_attitude = *target;
    }
}

/*********************************************************************
 * 单轴PID计算
 *********************************************************************/
static float pid_compute_single(pid_state_t *pid, float error, float dt) {
    if (!pid->initialized) {
        pid->last_error = error;
        pid->error_sum = 0.0f;
        pid->initialized = true;
        return 0.0f;
    }
    
    /* 积分项 */
    pid->error_sum += error * dt;
    
    /* 积分限幅 */
    if (pid->error_sum > MAX_ERROR_SUM) {
        pid->error_sum = MAX_ERROR_SUM;
    } else if (pid->error_sum < -MAX_ERROR_SUM) {
        pid->error_sum = -MAX_ERROR_SUM;
    }
    
    /* 微分项 */
    float error_diff = (error - pid->last_error) / dt;
    
    /* PID输出 */
    float output = pid_params.kp * error + 
                   pid_params.ki * pid->error_sum + 
                   pid_params.kd * error_diff;
    
    pid->last_error = error;
    
    return output;
}

/*********************************************************************
 * 计算横滚角 (Roll)
 *********************************************************************/
float calculate_roll(float ax, float ay, float az) {
    return atan2(ay, sqrt(ax * ax + az * az)) * RAD_TO_DEG;
}

/*********************************************************************
 * 计算俯仰角 (Pitch)
 *********************************************************************/
float calculate_pitch(float ax, float ay, float az) {
    return atan2(-ax, sqrt(ay * ay + az * az)) * RAD_TO_DEG;
}

/*********************************************************************
 * 电机混控矩阵
 * 根据球形机器人的6轴电机布局进行姿态控制
 *********************************************************************/
static void motor_mixing(float roll_output, float pitch_output, float yaw_output,
                        const uint16_t *base_pwm, uint16_t *output_pwm) {
    
    /* 球形机器人电机布局 (假设对称布局) */
    /* 电机1,2: X轴正反 */
    /* 电机3,4: Y轴正反 */  
    /* 电机5,6: Z轴正反 */
    
    int16_t adjustments[6] = {0};
    
    /* X轴控制 (Roll) */
    adjustments[0] = (int16_t)(roll_output);     /* 电机1: +Roll */
    adjustments[1] = (int16_t)(-roll_output);    /* 电机2: -Roll */
    
    /* Y轴控制 (Pitch) */
    adjustments[2] = (int16_t)(pitch_output);    /* 电机3: +Pitch */
    adjustments[3] = (int16_t)(-pitch_output);   /* 电机4: -Pitch */
    
    /* Z轴控制 (Yaw) */
    adjustments[4] = (int16_t)(yaw_output);      /* 电机5: +Yaw */
    adjustments[5] = (int16_t)(-yaw_output);     /* 电机6: -Yaw */
    
    /* 应用调整并限制范围 */
    for (int i = 0; i < 6; i++) {
        int32_t new_pwm = (int32_t)base_pwm[i] + adjustments[i];
        
        /* 限制PWM范围 */
        if (new_pwm > PWM_MAX_PULSE) {
            new_pwm = PWM_MAX_PULSE;
        } else if (new_pwm < PWM_MIN_PULSE) {
            new_pwm = PWM_MIN_PULSE;
        }
        
        output_pwm[i] = (uint16_t)new_pwm;
    }
}

/*********************************************************************
 * PID控制主计算函数
 *********************************************************************/
void pid_calculate(const imu_data_t *imu_data, const uint16_t *base_pwm, uint16_t *output_pwm) {
    if (!pid_enabled || !imu_data || !base_pwm || !output_pwm) {
        /* 如果未使能或参数无效，直接复制基础PWM */
        if (base_pwm && output_pwm) {
            memcpy(output_pwm, base_pwm, 6 * sizeof(uint16_t));
        }
        return;
    }
    
    /* 直接使用IMU提供的姿态角，或者从加速度计数据计算 */
    float current_roll = imu_data->Roll;   // 直接使用已计算的姿态角
    float current_pitch = imu_data->Pitch; // 直接使用已计算的姿态角
    float current_yaw = 0.0f;  /* 偏航角需要陀螺仪积分计算，这里简化 */
    
    /* 计算误差 */
    float roll_error = target_attitude.target_roll - current_roll;
    float pitch_error = target_attitude.target_pitch - current_pitch;
    float yaw_error = target_attitude.target_yaw - current_yaw;
    
    /* 计算时间间隔 */
    uint32_t current_time = imu_data->timestamp;
    float dt = 0.01f;  /* 假设10ms控制周期 */
    
    if (roll_pid.last_time > 0) {
        dt = (float)(current_time - roll_pid.last_time) / 1000.0f;
    }
    
    roll_pid.last_time = current_time;
    pitch_pid.last_time = current_time;
    yaw_pid.last_time = current_time;
    
    /* PID计算 */
    float roll_output = pid_compute_single(&roll_pid, roll_error, dt);
    float pitch_output = pid_compute_single(&pitch_pid, pitch_error, dt);
    float yaw_output = pid_compute_single(&yaw_pid, yaw_error, dt);
    
    /* 输出限制 */
    if (roll_output > MAX_PWM_ADJUSTMENT) roll_output = MAX_PWM_ADJUSTMENT;
    if (roll_output < -MAX_PWM_ADJUSTMENT) roll_output = -MAX_PWM_ADJUSTMENT;
    
    if (pitch_output > MAX_PWM_ADJUSTMENT) pitch_output = MAX_PWM_ADJUSTMENT;
    if (pitch_output < -MAX_PWM_ADJUSTMENT) pitch_output = -MAX_PWM_ADJUSTMENT;
    
    if (yaw_output > MAX_PWM_ADJUSTMENT) yaw_output = MAX_PWM_ADJUSTMENT;
    if (yaw_output < -MAX_PWM_ADJUSTMENT) yaw_output = -MAX_PWM_ADJUSTMENT;
    
    /* 电机混控 */
    motor_mixing(roll_output, pitch_output, yaw_output, base_pwm, output_pwm);
}

/*********************************************************************
 * 重置PID控制器
 *********************************************************************/
void pid_reset(void) {
    memset(&roll_pid, 0, sizeof(pid_state_t));
    memset(&pitch_pid, 0, sizeof(pid_state_t));
    memset(&yaw_pid, 0, sizeof(pid_state_t));
}

/*********************************************************************
 * 使能/禁用PID控制
 *********************************************************************/
void pid_enable(bool enable) {
    if (enable && !pid_enabled) {
        /* 重新使能时重置状态 */
        pid_reset();
    }
    pid_enabled = enable;
}

/*********************************************************************
 * 获取PID使能状态
 *********************************************************************/
bool pid_is_enabled(void) {
    return pid_enabled;
}