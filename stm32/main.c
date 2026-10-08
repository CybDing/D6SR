#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "math.h"

#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/stm32/timer.h>
#include <libopencm3/cm3/nvic.h>

#include "pin_config.h"
#include "uart_comm.h"
#include "motor_control.h"
#include "imu_reader.h"
#include "pid_control.h"
#include "motor_calibration.h"
#include "axis_stabilizer.h"      // 新的统一稳定器框架
#include "motion_controller.h"    // 运动控制器
#include "motor_calibration.h"    // 轴配置参数
#include "cascaded_pid.h"         // Yaw轴双环PID控制器

// 旧接口 (可选)
// #include "roll_stabilizer.h"
// #include "pitch_stabilizer.h"

#define TASK_PRIORITY_HIGH      (configMAX_PRIORITIES - 1)
#define TASK_PRIORITY_NORMAL    (configMAX_PRIORITIES - 2)
#define TASK_PRIORITY_LOW       (configMAX_PRIORITIES - 3)

static QueueHandle_t motor_cmd_queue;
static QueueHandle_t imu_data_queue;
QueueHandle_t uart_tx_queue;

// =====================================================
// 全局稳定器实例 (新框架)
// =====================================================
static AxisStabilizer g_roll_stabilizer;   // Roll轴稳定器 (Y方向运动)
static AxisStabilizer g_pitch_stabilizer;  // Pitch轴稳定器 (X方向运动)
static AxisStabilizer g_yaw_stabilizer;    // Yaw轴稳定器 (角度稳定, 单环PID - 已废弃)

// Yaw轴双环PID控制器 (新实现,替代单环g_yaw_stabilizer)
static CascadedPIDController g_yaw_cascaded_pid;

// Pitch轴双环PID控制器 (新实现,替代被动偏移控制)
static CascadedPIDController g_pitch_cascaded_pid;
typedef struct {
    uint16_t motor_pwm[6];      /* 6个电机PWM值 */
    float imu_data[11];          /* IMU数据: Rollv, Pitchv, Yawv, Roll, Pitch, Yaw, qw, qx, qy, qz */
    pid_params_t pid_params;    /* PID参数 */
    bool pid_enabled;           /* PID使能状态 */
    bool system_ready;          /* 系统就绪状态 */
} system_state_t;

// 全局系统状态 (供其他模块访问 IMU 数据)
system_state_t sys_state = {
    .motor_pwm = {PWM_CENTER_PULSE, PWM_CENTER_PULSE, PWM_CENTER_PULSE, 
                  PWM_CENTER_PULSE, PWM_CENTER_PULSE, PWM_CENTER_PULSE},
    .pid_params = {1.0f, 0.1f, 0.05f},
    .pid_enabled = false,
    .system_ready = false
};

// 硬件初始化
static void hardware_init(void) {
    /* 系统时钟配置 */
    rcc_clock_setup_pll(&rcc_hse_8mhz_3v3[RCC_CLOCK_3V3_168MHZ]);
    
    /* LED初始化 - F407 */
    rcc_periph_clock_enable(LED_RCC);
    gpio_mode_setup(LED_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, LED_PIN);
    gpio_set_output_options(LED_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_2MHZ, LED_PIN);
    gpio_clear(LED_PORT, LED_PIN);  /* LED关闭 */
    // gpio_set(LED_PORT, LED_PIN);
}

static void uart_comm_task(void *args) {
    (void)args;
    
    uart_comm_init();
    
    char rx_buffer[128];
    char tx_buffer[128];
    int len;
    
    for (;;) {
        if ((len = uart_receive_line(RPI_UART, rx_buffer, sizeof(rx_buffer))) > 0) {
            if (strncmp(rx_buffer, "PWM:", 4) == 0) {                
                uint16_t pwm_values[6];
                if (parse_pwm_command(rx_buffer + 4, pwm_values) == 6) {
                    /* 发送到电机控制队列 */
                    xQueueSend(motor_cmd_queue, pwm_values, 0);
                    
                    snprintf(tx_buffer, sizeof(tx_buffer), 
                            "DEBUG:PWM command received, the second value is %d\n",
                            pwm_values[1]);
                    xQueueSend(uart_tx_queue, tx_buffer, 0);
                }
            }
            else if (strncmp(rx_buffer, "GPIOTEST", 8) == 0) {
                /* GPIO引脚测试命令 */
                motor_gpio_test();
                snprintf(tx_buffer, sizeof(tx_buffer), "GPIO Test executed\n");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "PWMRESTORE", 10) == 0) {
                /* PWM模式恢复命令 */
                motor_pwm_restore();
                snprintf(tx_buffer, sizeof(tx_buffer), "PWM mode restored\n");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "SMOOTH:", 7) == 0) {
                /* 平滑控制开关命令 SMOOTH:1 或 SMOOTH:0 */
                // int enable = atoi(rx_buffer + 7);
                // motor_set_smooth_enabled(enable != 0);
                // snprintf(tx_buffer, sizeof(tx_buffer),
                //         "DEBUG:Smooth mode %s\n",
                //         enable ? "enabled" : "disabled");
                // xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "YSTAB:", 6) == 0) {
                /* Y轴(Roll)稳定器开关命令 YSTAB:1 或 YSTAB:0 */
                int enable = atoi(rx_buffer + 6);
                // y_stab_set_enabled(enable != 0);  // 旧接口,已弃用
                axis_stabilizer_set_enabled(&g_roll_stabilizer, enable != 0);
                snprintf(tx_buffer, sizeof(tx_buffer),
                        "DEBUG:Roll stabilizer %s\n",
                        enable ? "enabled" : "disabled");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "XSTAB:", 6) == 0) {
                /* X轴(Pitch)稳定器开关命令 XSTAB:1 或 XSTAB:0 */
                int enable = atoi(rx_buffer + 6);
                // 使用新的双环级联PID控制器
                cascaded_pid_set_enabled(&g_pitch_cascaded_pid, enable != 0);
                snprintf(tx_buffer, sizeof(tx_buffer),
                        "DEBUG:Pitch cascaded PID %s\n",
                        enable ? "enabled" : "disabled");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "XDEBUG:", 7) == 0) {
                /* Pitch调试输出开关命令 XDEBUG:1 或 XDEBUG:0 */
                int enable = atoi(rx_buffer + 7);
                cascaded_pid_set_debug_enabled(&g_pitch_cascaded_pid, enable != 0);
                snprintf(tx_buffer, sizeof(tx_buffer),
                        "DEBUG:Pitch debug output %s\n",
                        enable ? "enabled" : "disabled");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "ZSTAB:", 6) == 0) {
                /* Yaw轴稳定器开关命令 ZSTAB:1 或 ZSTAB:0 */
                int enable = atoi(rx_buffer + 6);
                cascaded_pid_set_enabled(&g_yaw_cascaded_pid, enable != 0);
                snprintf(tx_buffer, sizeof(tx_buffer),
                        "DEBUG:Yaw cascaded PID %s\n",
                        enable ? "enabled" : "disabled");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "ZDEBUG:", 7) == 0) {
                /* Yaw调试输出开关命令 ZDEBUG:1 或 ZDEBUG:0 */
                int enable = atoi(rx_buffer + 7);
                cascaded_pid_set_debug_enabled(&g_yaw_cascaded_pid, enable != 0);
                snprintf(tx_buffer, sizeof(tx_buffer),
                        "DEBUG:Yaw debug output %s\n",
                        enable ? "enabled" : "disabled");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "MOTION:", 7) == 0) {
                /* 运动控制命令 MOTION:vx,vy  范围:-1.0~+1.0 */
                float vx, vy;
                if (sscanf(rx_buffer + 7, "%f,%f", &vx, &vy) == 2) {
                    motion_set_velocity(vx, vy);
                    snprintf(tx_buffer, sizeof(tx_buffer),
                            "DEBUG:Motion velocity set: vx=%.2f, vy=%.2f\n",
                            vx, vy);
                    xQueueSend(uart_tx_queue, tx_buffer, 0);
                }
            }
            else if (strncmp(rx_buffer, "MOTIONSTOP", 10) == 0) {
                /* 停止运动命令 */
                motion_stop();
                snprintf(tx_buffer, sizeof(tx_buffer),
                        "DEBUG:Motion stopped\n");
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "PID:", 4) == 0) {
                /* PID参数设置 */
                float kp, ki, kd;
                if (sscanf(rx_buffer + 4, "%f,%f,%f", &kp, &ki, &kd) == 3) {
                    sys_state.pid_params.kp = kp;
                    sys_state.pid_params.ki = ki;
                    sys_state.pid_params.kd = kd;
                    
                    snprintf(tx_buffer, sizeof(tx_buffer), 
                            "DEBUG:PID params updated: %.2f,%.2f,%.2f\n", 
                            kp, ki, kd);
                    xQueueSend(uart_tx_queue, tx_buffer, 0);
                }
            }
            else if (strncmp(rx_buffer, "IMU", 3) == 0) {
                /* IMU数据请求 */
                snprintf(tx_buffer, sizeof(tx_buffer), 
                        "IMU:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        sys_state.imu_data[0], sys_state.imu_data[1], sys_state.imu_data[2], 
                        sys_state.imu_data[3], sys_state.imu_data[4], sys_state.imu_data[5],
                        sys_state.imu_data[6], sys_state.imu_data[7], sys_state.imu_data[8], 
                        sys_state.imu_data[9]);

                gpio_toggle(LED_PORT, LED_PIN);
                uart_send_string(RPI_UART, tx_buffer);
            }
            else if (strncmp(rx_buffer, "STABLE:", 7) == 0) {
                /* 运行时设置stable角度: STABLE:<axis>,<angle> 例如 STABLE:Roll,30 */
                char axis_name[16];
                float angle_deg;
                if (sscanf(rx_buffer + 7, "%15[^,],%f", axis_name, &angle_deg) == 2) {
                    if (motor_set_target_angle(axis_name, angle_deg)) {
                        snprintf(tx_buffer, sizeof(tx_buffer), "DEBUG:TARGET %s set to %.2f\n", axis_name, angle_deg);
                    } else {
                        snprintf(tx_buffer, sizeof(tx_buffer), "ERROR:Unknown axis %s\n", axis_name);
                    }
                } else {
                    snprintf(tx_buffer, sizeof(tx_buffer), "ERROR:Invalid STABLE format, use STABLE:<axis>,<angle>\n");
                }
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "GET_STABLE:", 11) == 0) {
                /* 查询target角度: GET_STABLE:<axis> 例如 GET_STABLE:Roll */
                char axis_name[16];
                if (sscanf(rx_buffer + 11, "%15s", axis_name) == 1) {
                    float angle;
                    if (motor_get_target_angle(axis_name, &angle)) {
                        snprintf(tx_buffer, sizeof(tx_buffer), "STABLE:%s,%.2f\n", axis_name, angle);
                    } else {
                        snprintf(tx_buffer, sizeof(tx_buffer), "ERROR:Unknown axis %s\n", axis_name);
                    }
                } else {
                    snprintf(tx_buffer, sizeof(tx_buffer), "ERROR:Invalid GET_STABLE format, use GET_STABLE:<axis>\n");
                }
                xQueueSend(uart_tx_queue, tx_buffer, 0);
            }
            else if (strncmp(rx_buffer, "STATUS", 6) == 0) {
                /* 系统状态查询 */
                snprintf(tx_buffer, sizeof(tx_buffer), 
                        "STATUS:%d,%d,%d\n",
                        sys_state.system_ready ? 1 : 0,
                        1,  /* IMU状态 */
                        sys_state.pid_enabled ? 1 : 0);
            
                uart_send_string(RPI_UART, tx_buffer);
                gpio_toggle(LED_PORT, LED_PIN);
                // uart_send_string(RPI_UART, "A");
            }
        }
        
        if (xQueueReceive(uart_tx_queue, tx_buffer, 10) == pdPASS) {
            uart_send_string(RPI_UART, tx_buffer);
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
        gpio_toggle(LED_PORT, LED_PIN);
    }
}

/*********************************************************************
 * 电机控制任务
 *********************************************************************/
static void motor_control_task(void *args) {
    (void)args;

    motor_control_init();

    uint16_t motor_pwm[6];

    for (;;) {
        /* 接收PWM命令（非阻塞，使用超时） */
        if (xQueueReceive(motor_cmd_queue, motor_pwm, pdMS_TO_TICKS(10)) == pdPASS) {
            /* 更新目标PWM值 */
            for (int i = 0; i < 6; i++) {
                sys_state.motor_pwm[i] = motor_pwm[i];
            }

            
            motor_set_all_pwm(motor_pwm);
        

            /* LED指示 */
            gpio_toggle(LED_PORT, LED_PIN);
        }

        /* 平滑更新已禁用，由稳定器直接控制电机 */
        // motor_update_smooth();

        /* 控制周期延时 */
        vTaskDelay(pdMS_TO_TICKS(10));  /* 100Hz控制频率 */
    }
}

/*********************************************************************
 * IMU数据读取任务
 *********************************************************************/
static void imu_reader_task(void *args) {
    (void)args;
    
    imu_reader_init();
    
    imu_data_t imu_data;
    
    for (;;) {
        /* 读取IMU数据 */

        if (imu_read_data(&imu_data) == 0) {
            /* 更新全局状态 */
            sys_state.imu_data[0] = imu_data.RollSpeed;
            sys_state.imu_data[1] = imu_data.PitchSpeed;
            sys_state.imu_data[2] = imu_data.YawSpeed;
            sys_state.imu_data[3] = imu_data.Roll;
            sys_state.imu_data[4] = imu_data.Pitch;
            // sys_state.imu_data[5] = imu_data.Yaw;
            sys_state.imu_data[6] = imu_data.qw;
            sys_state.imu_data[7] = imu_data.qx;
            sys_state.imu_data[8] = imu_data.qy;
            sys_state.imu_data[9] = imu_data.qz;
            sys_state.imu_data[10] = imu_data.timestamp; 
            float qw = imu_data.qw;
            float qx = imu_data.qx;
            float qy = imu_data.qy;
            float qz = imu_data.qz;
            sys_state.imu_data[5] = atan2f(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz)) * 57.29578f;
            
            char msg[50];
            // snprintf(msg, sizeof(msg), "Roll:%.2f Pitch:%.2f Yaw:%.2f", 
            //          imu_data.Roll, imu_data.Pitch, imu_data.Yaw);
            // xQueueSend(uart_tx_queue, msg, 0);
 
        }else{
            char* msgg= get_imu_status();
            xQueueSend(uart_tx_queue, msgg, 0);
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));  /* 10Hz采样 */
    }
}

/*********************************************************************
 * PID控制任务
 *********************************************************************/
static void pid_control_task(void *args) {
    (void)args;
    
    pid_init(&sys_state.pid_params);
    
    imu_data_t imu_data;
    uint16_t motor_corrections[6];
    
    for (;;) {
        if (sys_state.pid_enabled && 
            xQueueReceive(imu_data_queue, &imu_data, 100) == pdPASS) {
            
            pid_calculate(&imu_data, sys_state.motor_pwm, motor_corrections);
            
            xQueueSend(motor_cmd_queue, motor_corrections, 0);
        }
        
        vTaskDelay(pdMS_TO_TICKS(20));  /* 100Hz控制频率 */
    }
}

/*********************************************************************
 * 系统监控任务
 *********************************************************************/
static void system_monitor_task(void *args) {
    (void)args;
    char debug_msg[64];
    
    for (;;) {
        /* 系统状态检查 */
        sys_state.system_ready = true;
        
        /* 定期发送系统信息 */
        snprintf(debug_msg, sizeof(debug_msg), 
                "DEBUG:System running, free heap: %u\n", 
                xPortGetFreeHeapSize());
        xQueueSend(uart_tx_queue, debug_msg, 0);
        
        vTaskDelay(pdMS_TO_TICKS(1000)); 
    }
}
static int counter_for_debug = 0;
static void SendDebugProcess(void *args){
    (void)args;
    
    for (;;) {
        counter_for_debug ++;
        char send_buffer[64];  // 减小缓冲区
        snprintf(send_buffer, sizeof(send_buffer), 
                "Debug_msg sent from stm32! Counter: %\n", counter_for_debug);
        // 使用队列发送，避免直接UART冲突
        xQueueSend(uart_tx_queue, send_buffer, 0);
        vTaskDelay(pdMS_TO_TICKS(10000));  // 降低频率到10秒
    }
}

// static void y_axis_stab_task(void *args){
//     (void)args;
//     const float dt = 0.02f; // 20ms
//     for(;;){
//         stabilize_y_axis_step(dt);
//         vTaskDelay(pdMS_TO_TICKS(20));
//     }
// }

// static void x_axis_stab_task(void *args){
//     (void)args;
//     const float dt = 0.02f; // 20ms
//     for(;;){
//         stabilize_x_axis_step(dt);
//         vTaskDelay(pdMS_TO_TICKS(20));
//     }
// }

/*********************************************************************
 * 新框架稳定器任务
 *********************************************************************/
static void roll_stab_task_new(void *args) {
    (void)args;
    const float dt = 0.02f; // 20ms = 50Hz控制频率

    for (;;) {
        // 从motion_controller获取PWM偏移量并设置
        int16_t offset = motion_get_roll_pwm_offset();
        axis_stabilizer_set_pwm_offset(&g_roll_stabilizer, offset);

        // 执行控制步进
        axis_stabilizer_step(&g_roll_stabilizer, dt);

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

static void pitch_stab_task_new(void *args) {
    (void)args;

    for (;;) {
        if (g_pitch_cascaded_pid.enabled) {
            // 从motion_controller获取PWM偏移量并设置
            int16_t offset = motion_get_pitch_pwm_offset();
            cascaded_pid_set_pwm_offset(&g_pitch_cascaded_pid, offset);

            // 从PITCH_AXIS_CONFIG同步目标角度到cascaded PID控制器
            cascaded_pid_set_target_angle(&g_pitch_cascaded_pid,
                                          PITCH_AXIS_CONFIG.target_angle_deg);

            // 直接从全局 sys_state 读取 IMU 数据 (无阻塞)
            // sys_state.imu_data[4] = Pitch, sys_state.imu_data[1] = PitchSpeed
            float pitch_angle = sys_state.imu_data[4];
            float pitch_speed = sys_state.imu_data[1];

            // 执行双环PID控制
            cascaded_pid_step(&g_pitch_cascaded_pid, pitch_angle, pitch_speed);
        }
        // 禁用时不覆盖PWM,允许手动控制或其他任务设置

        vTaskDelay(pdMS_TO_TICKS(20));  // 50Hz
    }
}

static void yaw_stab_task_new(void *args) {
    (void)args;

    for (;;) {
        if (g_yaw_cascaded_pid.enabled) {
            // 获取motion_controller的偏移量 (保持兼容性)
            int16_t offset = motion_get_yaw_pwm_offset();
            cascaded_pid_set_pwm_offset(&g_yaw_cascaded_pid, offset);

            // 从YAW_AXIS_CONFIG同步目标角度到cascaded PID控制器
            cascaded_pid_set_target_angle(&g_yaw_cascaded_pid, YAW_AXIS_CONFIG.target_angle_deg);

            // 直接从全局 sys_state 读取 IMU 数据 (无阻塞)
            // sys_state.imu_data[5] = Yaw, sys_state.imu_data[2] = YawSpeed
            float yaw_angle = sys_state.imu_data[5];
            float yaw_speed = sys_state.imu_data[2];

            // 执行双环PID控制
            cascaded_pid_step(&g_yaw_cascaded_pid, yaw_angle, yaw_speed);
        }
        // 禁用时不覆盖PWM,允许手动控制或其他任务设置

        vTaskDelay(pdMS_TO_TICKS(20));  // 50Hz
    }
}

/*********************************************************************
 * FreeRTOS钩子函数
 *********************************************************************/
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    (void)xTask;
    (void)pcTaskName;
    
    /* 栈溢出处理 - 停止系统 */
    for (;;) {
        gpio_set(LED_PORT, LED_PIN);    /* LED常亮 */
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_clear(LED_PORT, LED_PIN);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void vApplicationMallocFailedHook(void) {
    /* 内存分配失败处理 */
    for (;;) {
        gpio_set(LED_PORT, LED_PIN);    /* LED快闪 */
        vTaskDelay(pdMS_TO_TICKS(50));
        gpio_clear(LED_PORT, LED_PIN);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

int main(void) {
    hardware_init();

    motor_cmd_queue = xQueueCreate(4, sizeof(uint16_t) * 6);
    imu_data_queue = xQueueCreate(4, sizeof(imu_data_t));
    uart_tx_queue = xQueueCreate(8, 128);

    uart_comm_init();

    /* 初始化运行时可调整的校准参数 */
    motor_calibration_init();

    // ===== 初始化新框架 =====
    // 初始化三个轴的稳定器
    axis_stabilizer_init(&g_roll_stabilizer, &ROLL_AXIS_CONFIG);
    axis_stabilizer_init(&g_pitch_stabilizer, &PITCH_AXIS_CONFIG);
    // axis_stabilizer_init(&g_yaw_stabilizer, &YAW_AXIS_CONFIG);  // 已废弃,使用双环PID

    // 初始化Yaw轴双环PID控制器 (替代单环g_yaw_stabilizer)
    // 电机索引: 0=neg, 1=pos
    cascaded_pid_init(&g_yaw_cascaded_pid, &YAW_CASCADED_PID_CONFIG, 0, 1);
    g_yaw_cascaded_pid.axis_name = "Yaw";

    // 初始化Pitch轴双环PID控制器 (替代被动偏移控制)
    // 电机索引: 4=neg, 5=pos
    cascaded_pid_init(&g_pitch_cascaded_pid, &PITCH_CASCADED_PID_CONFIG, 4, 5);
    g_pitch_cascaded_pid.axis_name = "Pitch";

    // 初始化运动控制器
    motion_controller_init();

    // 默认关闭稳定控制器
    axis_stabilizer_set_enabled(&g_roll_stabilizer, false);
    axis_stabilizer_set_enabled(&g_pitch_stabilizer, false);
    // axis_stabilizer_set_enabled(&g_yaw_stabilizer, false);  // 已废弃
    cascaded_pid_set_enabled(&g_yaw_cascaded_pid, false);    // Yaw双环PID默认关闭
    cascaded_pid_set_enabled(&g_pitch_cascaded_pid, false);  // Pitch双环PID默认关闭


    motion_set_enabled(true);

    /* 创建任务 */
    xTaskCreate(uart_comm_task, "UART", 500, NULL, TASK_PRIORITY_HIGH, NULL);
    xTaskCreate(motor_control_task, "MOTOR", 500, NULL, TASK_PRIORITY_NORMAL, NULL);
    xTaskCreate(imu_reader_task, "IMU", 500, NULL, TASK_PRIORITY_NORMAL, NULL);

    // ===== 选择使用新框架或旧框架 =====
    // 旧框架 (已弃用,可用于回退测试)
    // xTaskCreate(y_axis_stab_task, "YSTAB", 400, NULL, TASK_PRIORITY_NORMAL, NULL);
    // xTaskCreate(x_axis_stab_task, "XSTAB", 400, NULL, TASK_PRIORITY_NORMAL, NULL);

    xTaskCreate(roll_stab_task_new, "ROLL", 400, NULL, TASK_PRIORITY_NORMAL, NULL);
    xTaskCreate(pitch_stab_task_new, "PITCH", 400, NULL, TASK_PRIORITY_NORMAL, NULL);
    xTaskCreate(yaw_stab_task_new, "YAW", 400, NULL, TASK_PRIORITY_NORMAL, NULL);
    
    // xTaskCreate(pid_control_task, "PID", 100, NULL, TASK_PRIORITY_NORMAL, NULL);
    // xTaskCreate(system_monitor_task, "MONITOR", 100, NULL, TASK_PRIORITY_LOW, NULL);

    vTaskStartScheduler();
    for (;;);
    return 0;
}