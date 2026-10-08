/*
 * 电机控制模块实现
 */

#include "motor_control.h"
#include "pin_config.h"
#include "uart_comm.h"
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/timer.h>

/* 电机配置结构体 */
typedef struct {
    uint32_t timer;
    enum tim_oc_id channel;
    uint32_t gpio_port;
    uint32_t gpio_rcc;
    uint16_t gpio_pin;
    uint32_t timer_rcc;
    uint8_t gpio_af;
} motor_config_t;

/* 6个电机配置 - 使用清晰的宏定义 */
static const motor_config_t motor_configs[NUM_MOTORS] = {
    /* 电机1 */
    {
        .timer = MOTOR1_TIM,
        .channel = MOTOR1_CH,
        .gpio_port = MOTOR1_PORT,
        .gpio_rcc = MOTOR1_GPIO_RCC,
        .gpio_pin = MOTOR1_PIN,
        .timer_rcc = MOTOR1_TIM_RCC,
        .gpio_af = MOTOR1_AF
    },
    /* 电机2 */
    {
        .timer = MOTOR2_TIM,
        .channel = MOTOR2_CH,
        .gpio_port = MOTOR2_PORT,
        .gpio_rcc = MOTOR2_GPIO_RCC,
        .gpio_pin = MOTOR2_PIN,
        .timer_rcc = MOTOR2_TIM_RCC,
        .gpio_af = MOTOR2_AF
    },
    /* 电机3 */
    {
        .timer = MOTOR3_TIM,
        .channel = MOTOR3_CH,
        .gpio_port = MOTOR3_PORT,
        .gpio_rcc = MOTOR3_GPIO_RCC,
        .gpio_pin = MOTOR3_PIN,
        .timer_rcc = MOTOR3_TIM_RCC,
        .gpio_af = MOTOR3_AF
    },
    /* 电机4 */
    {
        .timer = MOTOR4_TIM,
        .channel = MOTOR4_CH,
        .gpio_port = MOTOR4_PORT,
        .gpio_rcc = MOTOR4_GPIO_RCC,
        .gpio_pin = MOTOR4_PIN,
        .timer_rcc = MOTOR4_TIM_RCC,
        .gpio_af = MOTOR4_AF
    },
    /* 电机5 */
    {
        .timer = MOTOR5_TIM,
        .channel = MOTOR5_CH,
        .gpio_port = MOTOR5_PORT,
        .gpio_rcc = MOTOR5_GPIO_RCC,
        .gpio_pin = MOTOR5_PIN,
        .timer_rcc = MOTOR5_TIM_RCC,
        .gpio_af = MOTOR5_AF
    },
    /* 电机6 */
    {
        .timer = MOTOR6_TIM,
        .channel = MOTOR6_CH,
        .gpio_port = MOTOR6_PORT,
        .gpio_rcc = MOTOR6_GPIO_RCC,
        .gpio_pin = MOTOR6_PIN,
        .timer_rcc = MOTOR6_TIM_RCC,
        .gpio_af = MOTOR6_AF
    }
};

/* 当前PWM值存储 */
static uint16_t current_pwm[NUM_MOTORS];

/* PWM平滑控制变量 */
static uint16_t target_pwm[NUM_MOTORS];    /* 目标PWM值 */
static bool smooth_enabled = false;         /* 平滑模式开关 - 禁用，由稳定器直接控制 */

/*********************************************************************
 * 定时器初始化
 *********************************************************************/
static void timer_init(uint32_t timer, uint32_t timer_rcc) {
    static bool tim3_initialized = false;
    static bool tim4_initialized = false;
    
    /* 避免重复初始化 */
    if (timer == TIM3 && tim3_initialized) return;
    if (timer == TIM4 && tim4_initialized) return;
    
    /* 使能定时器时钟 */
    rcc_periph_clock_enable(timer_rcc);
    
    /* 调试信息：确认时钟使能 */
    char debug_msg[64];
    snprintf(debug_msg, sizeof(debug_msg), "Timer Init: TIM%d clock enabled\n", 
             (timer == TIM3) ? 3 : (timer == TIM4) ? 4 : 0);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    /* 定时器配置 */
    timer_disable_counter(timer);
    
    /* 先配置基础参数，再复位，避免参数被清零 */
    timer_set_mode(timer, TIM_CR1_CKD_CK_INT, TIM_CR1_CMS_EDGE, TIM_CR1_DIR_UP);
    timer_set_prescaler(timer, PWM_PRESCALER - 1);
    timer_set_period(timer, PWM_PERIOD - 1);
    timer_enable_preload(timer);
    timer_continuous_mode(timer);
    
    /* 调试信息：验证配置值 */
    snprintf(debug_msg, sizeof(debug_msg), "Timer Config: PSC=%d ARR=%d\n", 
             PWM_PRESCALER - 1, PWM_PERIOD - 1);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    /* 触发一次更新事件，将PSC/ARR预装到影子寄存器 */
    timer_generate_event(timer, TIM_EGR_UG);
    
    /* 注意：定时器暂不启动，等所有通道配置完成后再启动 */
    
    /* 标记已初始化 */
    if (timer == TIM3) tim3_initialized = true;
    if (timer == TIM4) tim4_initialized = true;
}

/*********************************************************************
 * GPIO初始化
 *********************************************************************/
static void gpio_init_motor(int motor_id) {
    const motor_config_t *config = &motor_configs[motor_id];
    
    /* 使能GPIO时钟 */
    rcc_periph_clock_enable(config->gpio_rcc);
    
    /* F407 GPIO配置 */
    gpio_mode_setup(config->gpio_port, GPIO_MODE_AF, GPIO_PUPD_NONE, config->gpio_pin);
    gpio_set_output_options(config->gpio_port, GPIO_OTYPE_PP, GPIO_OSPEED_50MHZ, config->gpio_pin);
    gpio_set_af(config->gpio_port, config->gpio_af, config->gpio_pin);
}

/*********************************************************************
 * PWM通道初始化
 *********************************************************************/
static void pwm_channel_init(uint32_t timer, enum tim_oc_id channel) {
    /* 先禁用输出，避免配置过程中的不确定信号 */
    timer_disable_oc_output(timer, channel);
    
    /* 配置PWM模式1 */
    timer_set_oc_mode(timer, channel, TIM_OCM_PWM1);
    timer_enable_oc_preload(timer, channel);  /* 使能预装载，避免更新时的毛刺 */
    
    /* 设置初始PWM值为中间值 */
    timer_set_oc_value(timer, channel, PWM_CENTER_PULSE);
    
    /* 配置完成后使能输出 */
    timer_enable_oc_output(timer, channel);
}

/*********************************************************************
 * 电机控制初始化
 *********************************************************************/
void motor_control_init(void) {
    /* 初始化所有电机 */
    for (int i = 0; i < NUM_MOTORS; i++) {
        const motor_config_t *config = &motor_configs[i];

        /* 初始化GPIO */
        gpio_init_motor(i);

        /* 初始化定时器（不启动） */
        timer_init(config->timer, config->timer_rcc);

        /* 初始化PWM通道 */
        pwm_channel_init(config->timer, config->channel);

        /* 设置初始PWM值 */
        current_pwm[i] = PWM_CENTER_PULSE;
        target_pwm[i] = PWM_CENTER_PULSE;  /* 初始化目标值 */
    }

    /* 所有通道配置完成后，启动定时器 */
    timer_enable_counter(TIM3);
    timer_enable_counter(TIM4);
}

/*********************************************************************
 * 设置单个电机PWM
 *********************************************************************/
void motor_set_pwm(int motor_id, uint16_t pwm_value) {
    if (motor_id < 0 || motor_id >= NUM_MOTORS) {
        return;  /* 无效电机ID */
    }
    
    /* 限制PWM值范围 */
    if (pwm_value < PWM_MIN_PULSE) pwm_value = PWM_MIN_PULSE;
    if (pwm_value > PWM_MAX_PULSE) pwm_value = PWM_MAX_PULSE;
    
    const motor_config_t *config = &motor_configs[motor_id];

    // char debug_msg[64];
    // snprintf(debug_msg, sizeof(debug_msg), "Set motor %d PWM: %d\n", motor_id, pwm_value);
    // xQueueSend(uart_tx_queue, debug_msg, 0);   
    
    timer_set_oc_value(config->timer, config->channel, pwm_value);
    current_pwm[motor_id] = pwm_value;

    /* 注意: 不要调用 timer_generate_event(TIM_EGR_UG)！
     * 该调用会重置定时器计数器，当多个稳定器同时控制同一个定时器上的
     * 不同电机时，会导致PWM波形被反复打断，造成电机卡顿。
     * 使用预装载机制，CCR值会在下一个更新事件自动生效。
     */
}

/*********************************************************************
 * 设置所有电机PWM
 *********************************************************************/
void motor_set_all_pwm(uint16_t *pwm_values) {
    for (int i = 0; i < NUM_MOTORS; i++) {
        /* 0xFFFF 表示保持当前值不变，跳过该电机 */
        if (pwm_values[i] != 0xFFFF) {
            motor_set_pwm(i, pwm_values[i]);
        }
    }
}

/*********************************************************************
 * 紧急停止所有电机
 *********************************************************************/
void motor_emergency_stop(void) {
    for (int i = 0; i < NUM_MOTORS; i++) {
        motor_set_pwm(i, PWM_CENTER_PULSE);
    }
}

/*********************************************************************
 * 获取电机PWM值
 *********************************************************************/
uint16_t motor_get_pwm(int motor_id) {
    if (motor_id < 0 || motor_id >= NUM_MOTORS) {
        return 0;
    }
    return current_pwm[motor_id];
}

/*********************************************************************
 * PWM平滑控制函数
 *********************************************************************/

/* 设置单个电机目标PWM值 */
void motor_set_target_pwm(int motor_id, uint16_t target) {
    if (motor_id < 0 || motor_id >= NUM_MOTORS) {
        return;
    }

    /* 限制范围 */
    if (target < PWM_MIN_PULSE) target = PWM_MIN_PULSE;
    if (target > PWM_MAX_PULSE) target = PWM_MAX_PULSE;

    target_pwm[motor_id] = target;
}

/* 设置所有电机目标PWM值 */
void motor_set_all_target_pwm(uint16_t *targets) {
    for (int i = 0; i < NUM_MOTORS; i++) {
        /* 0xFFFF 表示保持当前值不变，跳过该电机 */
        if (targets[i] != 0xFFFF) {
            motor_set_target_pwm(i, targets[i]);
        }
    }
}

/* 平滑更新PWM值 - 需在控制任务中周期调用 */
void motor_update_smooth(void) {
    if (!smooth_enabled) {
        return;  /* 平滑模式未启用，不做处理 */
    }

    for (int i = 0; i < NUM_MOTORS; i++) {
        int16_t error = target_pwm[i] - current_pwm[i];

        if (error == 0) {
            continue;  /* 已到达目标值 */
        }

        /* 计算本次步进值 */
        int16_t step;
        if (error > 0) {
            step = (error > PWM_RAMP_STEP) ? PWM_RAMP_STEP : error;
        } else {
            step = (error < -PWM_RAMP_STEP) ? -PWM_RAMP_STEP : error;
        }

        /* 更新PWM值 */
        uint16_t new_pwm = current_pwm[i] + step;
        motor_set_pwm(i, new_pwm);
    }
}

/* 启用/禁用平滑模式 */
void motor_set_smooth_enabled(bool enabled) {
    smooth_enabled = enabled;

    /* 无论启用还是禁用，都将目标值同步到当前值 */
    /* 这样可以避免切换模式时电机意外转动 */
    for (int i = 0; i < NUM_MOTORS; i++) {
        target_pwm[i] = current_pwm[i];
    }
}

/* 获取平滑模式状态 */
bool motor_get_smooth_enabled(void) {
    return smooth_enabled;
}

/* 获取目标PWM值 */
uint16_t motor_get_target_pwm(int motor_id) {
    if (motor_id < 0 || motor_id >= NUM_MOTORS) {
        return 0;
    }
    return target_pwm[motor_id];
}

/*********************************************************************
 * GPIO引脚测试 - 直接控制引脚输出高低电平
 *********************************************************************/
void motor_gpio_test(void) {
    /* 临时设置为GPIO输出模式测试引脚 */
    rcc_periph_clock_enable(RCC_GPIOE);
    rcc_periph_clock_enable(RCC_GPIOC);
    
    /* PE9, PE11, PE13, PE14设为GPIO输出 */
    gpio_mode_setup(GPIOE, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO9 | GPIO11 | GPIO13 | GPIO14);
    gpio_set_output_options(GPIOE, GPIO_OTYPE_PP, GPIO_OSPEED_50MHZ, GPIO9 | GPIO11 | GPIO13 | GPIO14);
    
    /* PC6, PC7设为GPIO输出 */
    gpio_mode_setup(GPIOC, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO6 | GPIO7);
    gpio_set_output_options(GPIOC, GPIO_OTYPE_PP, GPIO_OSPEED_50MHZ, GPIO6 | GPIO7);
    
    /* 输出高电平 */
    gpio_set(GPIOE, GPIO9 | GPIO11 | GPIO13 | GPIO14);
    gpio_set(GPIOC, GPIO6 | GPIO7);
    
    char debug_msg[64];
    snprintf(debug_msg, sizeof(debug_msg), "GPIO Test: All pins set HIGH\n");
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
}

/*********************************************************************
 * PWM模式恢复 - 重新设置GPIO为PWM复用功能
 *********************************************************************/
void motor_pwm_restore(void) {
    /* 重新初始化所有电机的GPIO为PWM模式 */
    for (int i = 0; i < NUM_MOTORS; i++) {
        gpio_init_motor(i);
    }
    
    char debug_msg[64];
    snprintf(debug_msg, sizeof(debug_msg), "PWM mode restored for all motors\n");
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
}

/*********************************************************************
 * PWM调试测试函数 - 输出详细状态信息
 *********************************************************************/
void motor_debug_status(void) {
    char debug_msg[128];
    
    /* 强制重新使能时钟 - 调试用 */
    rcc_periph_clock_enable(RCC_TIM1);
    rcc_periph_clock_enable(RCC_TIM8);
    
    /* 强制重新配置定时器基础参数 */
    timer_set_prescaler(TIM1, PWM_PRESCALER - 1);
    timer_set_period(TIM1, PWM_PERIOD - 1);
    timer_set_prescaler(TIM8, PWM_PRESCALER - 1);
    timer_set_period(TIM8, PWM_PERIOD - 1);
    
    /* 强制重新配置PWM通道 */
    for (int ch = 1; ch <= 4; ch++) {
        enum tim_oc_id channel = (ch == 1) ? TIM_OC1 : (ch == 2) ? TIM_OC2 : 
                                 (ch == 3) ? TIM_OC3 : TIM_OC4;
        
        /* 配置PWM模式1 */
        timer_set_oc_mode(TIM1, channel, TIM_OCM_PWM1);
        timer_enable_oc_preload(TIM1, channel);
        timer_set_oc_polarity_high(TIM1, channel);
        timer_set_oc_value(TIM1, channel, PWM_CENTER_PULSE);
        timer_enable_oc_output(TIM1, channel);
        
        if (ch <= 2) { /* TIM8只有2个通道 */
            timer_set_oc_mode(TIM8, channel, TIM_OCM_PWM1);
            timer_enable_oc_preload(TIM8, channel);
            timer_set_oc_polarity_high(TIM8, channel);
            timer_set_oc_value(TIM8, channel, PWM_CENTER_PULSE);
            timer_enable_oc_output(TIM8, channel);
        }
    }
    
    /* 触发更新事件 */
    timer_generate_event(TIM1, TIM_EGR_UG);
    timer_generate_event(TIM8, TIM_EGR_UG);
    
    /* 强制重启定时器 */
    timer_enable_counter(TIM1);
    timer_enable_counter(TIM8);
    timer_enable_break_main_output(TIM1);
    timer_enable_break_main_output(TIM8);
    
    /* 检查TIM1状态 */
    uint32_t tim1_cr1 = TIM_CR1(TIM1);
    uint32_t tim1_bdtr = TIM_BDTR(TIM1);
    bool tim1_enabled = (tim1_cr1 & TIM_CR1_CEN) != 0;
    bool tim1_moe = (tim1_bdtr & TIM_BDTR_MOE) != 0;
    
    /* 详细检查TIM1寄存器 */
    uint32_t tim1_ccr1 = TIM_CCR1(TIM1);
    uint32_t tim1_ccr2 = TIM_CCR2(TIM1);
    uint32_t tim1_ccmr1 = TIM_CCMR1(TIM1);
    uint32_t tim1_ccer = TIM_CCER(TIM1);
    uint32_t tim1_arr = TIM_ARR(TIM1);
    uint32_t tim1_psc = TIM_PSC(TIM1);
    
    snprintf(debug_msg, sizeof(debug_msg), "TIM1: CEN=%d MOE=%d ARR=%lu PSC=%lu\n", 
             tim1_enabled ? 1 : 0, tim1_moe ? 1 : 0, tim1_arr, tim1_psc);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    uint32_t tim1_ccr3 = TIM_CCR3(TIM1);
    uint32_t tim1_ccr4 = TIM_CCR4(TIM1);
    uint32_t tim1_ccmr2 = TIM_CCMR2(TIM1);
    
    snprintf(debug_msg, sizeof(debug_msg), "TIM1: CCR1=%lu CCR2=%lu CCR3=%lu CCR4=%lu\n", 
             tim1_ccr1, tim1_ccr2, tim1_ccr3, tim1_ccr4);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    snprintf(debug_msg, sizeof(debug_msg), "TIM1: CCER=0x%08lx CCMR1=0x%08lx CCMR2=0x%08lx\n", 
             tim1_ccer, tim1_ccmr1, tim1_ccmr2);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    /* 分析CCER寄存器各位 */
    bool cc1e = (tim1_ccer & 0x01) != 0;  /* CH1输出使能 */
    bool cc2e = (tim1_ccer & 0x10) != 0;  /* CH2输出使能 */
    bool cc3e = (tim1_ccer & 0x100) != 0; /* CH3输出使能 */
    bool cc4e = (tim1_ccer & 0x1000) != 0;/* CH4输出使能 */
    
    snprintf(debug_msg, sizeof(debug_msg), "CCER bits: CH1E=%d CH2E=%d CH3E=%d CH4E=%d\n", 
             cc1e?1:0, cc2e?1:0, cc3e?1:0, cc4e?1:0);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    /* 检查GPIO寄存器 */
    uint32_t gpioe_moder = GPIO_MODER(GPIOE);
    uint32_t gpioe_afrl = GPIO_AFRL(GPIOE);
    uint32_t gpioe_afrh = GPIO_AFRH(GPIOE);
    
    snprintf(debug_msg, sizeof(debug_msg), "GPIOE: MODER=0x%08lx AFRL=0x%08lx AFRH=0x%08lx\n", 
             gpioe_moder, gpioe_afrl, gpioe_afrh);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    /* 检查TIM8状态 */
    uint32_t tim8_cr1 = TIM_CR1(TIM8);
    uint32_t tim8_bdtr = TIM_BDTR(TIM8);
    bool tim8_enabled = (tim8_cr1 & TIM_CR1_CEN) != 0;
    bool tim8_moe = (tim8_bdtr & TIM_BDTR_MOE) != 0;
    
    snprintf(debug_msg, sizeof(debug_msg), "TIM8: CR1=0x%08lx CEN=%d MOE=%d\n", 
             tim8_cr1, tim8_enabled ? 1 : 0, tim8_moe ? 1 : 0);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
    
    /* 检查RCC时钟状态 */
    uint32_t ahb1enr = RCC_AHB1ENR;
    uint32_t apb2enr = RCC_APB2ENR;
    bool gpioe_en = (ahb1enr & RCC_AHB1ENR_GPIOEEN) != 0;
    bool gpioc_en = (ahb1enr & RCC_AHB1ENR_GPIOCEN) != 0;
    bool tim1_en = (apb2enr & RCC_APB2ENR_TIM1EN) != 0;
    bool tim8_en = (apb2enr & RCC_APB2ENR_TIM8EN) != 0;
    
    snprintf(debug_msg, sizeof(debug_msg), "RCC: GPIOE=%d GPIOC=%d TIM1=%d TIM8=%d\n", 
             gpioe_en ? 1 : 0, gpioc_en ? 1 : 0, tim1_en ? 1 : 0, tim8_en ? 1 : 0);
    if (uart_tx_queue != NULL) {
        xQueueSend(uart_tx_queue, debug_msg, 0);
    }
}