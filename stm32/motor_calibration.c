#include "motor_calibration.h"
#include "cascaded_pid.h"
#include "FreeRTOS.h"
#include "semphr.h"
#include <string.h>
#include <ctype.h>

static SemaphoreHandle_t cal_mutex = NULL;

/* Mutable configs (initial values copied from previous header definitions) */
AxisControlConfig ROLL_AXIS_CONFIG = {
    .axis_name = "Roll",

    .motor_neg_index = 2,
    .motor_pos_index = 3,

    .base_pwm_neg = 1560,
    .base_pwm_pos = 1440,

    .delta_sign_neg = -1,
    .delta_sign_pos = +1,
    .grav_sign = +1,

    .stable_angle_deg = 22.70f,
    .target_angle_deg = 0.0f,
    .gravity_comp_gain = 37.5f,
    .kp = 0.3f,
    .ki = 0.1f,
    .kd = 0.03f,
    .kff = 0.00f,
    .pwm_adjust_range = 100,
    .deadband_deg = 1.5f,
    .protect_limit_deg = 50.0f
};

AxisControlConfig PITCH_AXIS_CONFIG = {
    .axis_name = "Pitch",

    .motor_neg_index = 4,
    .motor_pos_index = 5,

    .base_pwm_neg = 1650,
    .base_pwm_pos = 1650,

    .delta_sign_neg = 0,
    .delta_sign_pos = 0,
    .grav_sign = 0,

    .stable_angle_deg = 8.7f,
    .target_angle_deg = 0.0f,
    .gravity_comp_gain = 0.0f,
    .kp = 0.0f,
    .ki = 0.0f,
    .kd = 0.0f,
    .kff = 0.0f,
    .pwm_adjust_range = 200,
    .deadband_deg = 2.0f,
    .protect_limit_deg = 50.0f
};

AxisControlConfig YAW_AXIS_CONFIG = {
    .axis_name = "Yaw",

    .motor_neg_index = 0,
    .motor_pos_index = 1,

    .base_pwm_neg = 1600,
    .base_pwm_pos = 1600,

    .delta_sign_neg = 0,
    .delta_sign_pos = +1,
    .grav_sign = +1,

    .stable_angle_deg = 0.0f,
    .target_angle_deg = 0.0f,
    .gravity_comp_gain = 0.0f,
    .kp = 0.1f,
    .ki = 0.02f,
    .kd = 0.03f,
    .kff = 0.0f,
    .pwm_adjust_range = 50,
    .deadband_deg = 3.0f,
    .protect_limit_deg = 360.0f
};

/*
 * IMPORTANT: 双环PID配置中的base_pwm值必须与YAW_AXIS_CONFIG保持同步
 * 如果修改了电机基准PWM，需要同时更新两处配置
 *
 * 参数调优指南:
 * 1. 外环调优 (先调):
 *    - 从小Kp开始 (0.5), 逐步增大直到出现小幅震荡
 *    - 添加Kd (Kd ≈ Kp/10) 抑制震荡
 *    - 最后添加Ki (Ki ≈ Kp/20) 消除稳态误差
 *
 * 2. 内环调优 (后调):
 *    - Kp应较大 (0.5~1.5), 保证快速跟踪速度目标
 *    - Ki用于消除速度偏差 (0.05~0.2)
 *    - Kd通常较小 (0.02~0.08), 防止速度噪声放大
 *
 * 3. 极限调整:
 *    - outer_max_output: 过大会冲击内环, 过小响应慢
 *    - inner_max_output: 过大导致PWM震荡, 过小动力不足
 */

const CascadedPIDConfig YAW_CASCADED_PID_CONFIG = {
    // 外环 (位置环) - 输出速度目标
    .outer_kp = 0.5f,           // 外环比例增益 
    .outer_ki = 0.1f,           // 外环积分增益 ( 未测试，原来之前设置的系数是 0.0 ) 
    .outer_kd = 0.007f,         // 外环微分增益 
    .outer_max_output = 40.0f,  // 速度目标上限 (°/s)
    .outer_deadband = 8.0f,     // 位置死区 

    .inner_kp = 1.0f,
    .inner_ki = 0.15f,
    .inner_kd = 0.05f,
    .inner_max_output = 60.0f,

    // 通用参数
    .dt = 0.02f,                // 50Hz控制频率
    .base_pwm_neg = 1540,       // 电机1基准PWM (保证始终能转动)
    .base_pwm_pos = 1560,       // 电机2基准PWM (保证始终能转动)
    .pwm_range = 80,            // PWM调整范围

    // 重力补偿参数 (Yaw无重力影响)
    .gravity_comp_gain = 0.0f,
    .stable_angle_deg = 0.0f,
    .grav_sign = 0,

    // 控制模式
    .use_angle_wrap = true,     // Yaw: 360度环绕处理
};

/*
 * Pitch轴双环PID配置
 *
 * 参数调优指南:
 * 1. 先关闭重力补偿 (gravity_comp_gain=0), 调试双环PID
 * 2. 外环从小Kp开始 (0.3~0.5), 逐步增大
 * 3. 内环Kp较大 (0.5~1.5), 保证快速跟踪
 * 4. 稳定后开启重力补偿, 调整gain直到稳态误差最小
 */
const CascadedPIDConfig PITCH_CASCADED_PID_CONFIG = {
    // 外环 (位置环) - 输出速度目标
    .outer_kp = 0.5f,           // 外环比例增益
    .outer_ki = 0.05f,           // 外环积分增益 (初始关闭)
    .outer_kd = 0.005f,         // 外环微分增益
    .outer_max_output = 40.0f,  // 速度目标上限 (°/s)
    .outer_deadband = 3.0f,     // 位置死区 (degrees)

    // 内环 (速度环) - 输出PWM增量
    .inner_kp = 0.5f,
    .inner_ki = 0.05f,
    .inner_kd = 0.005f,
    .inner_max_output = 60.0f,

    // 通用参数
    .dt = 0.02f,                // 50Hz控制频率
    .base_pwm_neg = 1550,       // M5基准PWM
    .base_pwm_pos = 1550,       // M4基准PWM
    .pwm_range = 70,            // PWM调整范围

    // 重力补偿参数 (初始值,需要调试)
    .gravity_comp_gain = 130.0f, // 160
    .stable_angle_deg = 20.0f,   // 
    .grav_sign = +1,            

    // 控制模式
    .use_angle_wrap = true,    // Pitch: 无360度环绕
};

void motor_calibration_init(void) {
    /* 互斥锁在任务中首次使用时惰性创建，避免在调度器启动前创建 */
    /* 初始化函数现在是空的，互斥锁将在第一次调用时创建 */
}

static int equals_ignore_case(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return (*a == '\0' && *b == '\0');
}

static AxisControlConfig *find_config_by_name(const char *name) {
    if (name == NULL) return NULL;
    if (equals_ignore_case(name, "roll")) return &ROLL_AXIS_CONFIG;
    if (equals_ignore_case(name, "pitch")) return &PITCH_AXIS_CONFIG;
    if (equals_ignore_case(name, "yaw")) return &YAW_AXIS_CONFIG;
    return NULL;
}

bool motor_set_stable_angle(const char *axis_name, float angle_deg) {
    AxisControlConfig *cfg = find_config_by_name(axis_name);
    if (!cfg) return false;
    /* 惰性创建互斥锁（调度器启动后首次调用时创建） */
    if (cal_mutex == NULL) {
        cal_mutex = xSemaphoreCreateMutex();
    }
    if (cal_mutex) xSemaphoreTake(cal_mutex, portMAX_DELAY);
    cfg->stable_angle_deg = angle_deg;
    if (cal_mutex) xSemaphoreGive(cal_mutex);
    return true;
}

bool motor_get_stable_angle(const char *axis_name, float *out_angle) {
    AxisControlConfig *cfg = find_config_by_name(axis_name);
    if (!cfg || !out_angle) return false;
    /* 惰性创建互斥锁（调度器启动后首次调用时创建） */
    if (cal_mutex == NULL) {
        cal_mutex = xSemaphoreCreateMutex();
    }
    if (cal_mutex) xSemaphoreTake(cal_mutex, portMAX_DELAY);
    *out_angle = cfg->stable_angle_deg;
    if (cal_mutex) xSemaphoreGive(cal_mutex);
    return true;
}

/* target angle setters/getters (runtime target used by STABLE command) */
bool motor_set_target_angle(const char *axis_name, float angle_deg) {
    AxisControlConfig *cfg = find_config_by_name(axis_name);
    if (!cfg) return false;
    /* 惰性创建互斥锁（调度器启动后首次调用时创建） */
    if (cal_mutex == NULL) {
        cal_mutex = xSemaphoreCreateMutex();
    }
    if (cal_mutex) xSemaphoreTake(cal_mutex, portMAX_DELAY);
    cfg->target_angle_deg = angle_deg;
    if (cal_mutex) xSemaphoreGive(cal_mutex);
    return true;
}

bool motor_get_target_angle(const char *axis_name, float *out_angle) {
    AxisControlConfig *cfg = find_config_by_name(axis_name);
    if (!cfg || !out_angle) return false;
    /* 惰性创建互斥锁（调度器启动后首次调用时创建） */
    if (cal_mutex == NULL) {
        cal_mutex = xSemaphoreCreateMutex();
    }
    if (cal_mutex) xSemaphoreTake(cal_mutex, portMAX_DELAY);
    *out_angle = cfg->target_angle_deg;
    if (cal_mutex) xSemaphoreGive(cal_mutex);
    return true;
}