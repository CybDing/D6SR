#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/timer.h>
#include <libopencm3/stm32/usart.h>

/* 目标硬件: STM32F407VET6 (唯一支持的平台) */
    #define SYSTEM_CLOCK_HZ     168000000
    
    /* LED - PD12 (板载LED) */
    #define LED_PORT            GPIOD
    #define LED_PIN             GPIO12
    #define LED_RCC             RCC_GPIOD
    
    /* 树莓派通信 - USART1 (PA9/PA10) */
    #define RPI_UART            USART1
    #define RPI_UART_RCC        RCC_USART1
    #define RPI_UART_GPIO_RCC   RCC_GPIOA
    #define RPI_UART_PORT       GPIOA
    #define RPI_UART_TX_PIN     GPIO9
    #define RPI_UART_RX_PIN     GPIO10
    
    /* IMU通信 - USART2 (PA2/PA3) */
    #define IMU_UART            USART2
    #define IMU_UART_RCC        RCC_USART2
    #define IMU_UART_GPIO_RCC   RCC_GPIOA
    #define IMU_UART_PORT       GPIOA
    #define IMU_UART_TX_PIN     GPIO2
    #define IMU_UART_RX_PIN     GPIO3
    
    /* 6个电机PWM配置 - 使用TIM3/TIM4避开FSMC冲突 */
    
    /* 电机1: TIM3_CH1, PC6 */
    #define MOTOR1_TIM          TIM3
    #define MOTOR1_TIM_RCC      RCC_TIM3
    #define MOTOR1_CH           TIM_OC1
    #define MOTOR1_PORT         GPIOC
    #define MOTOR1_GPIO_RCC     RCC_GPIOC
    #define MOTOR1_PIN          GPIO6
    #define MOTOR1_AF           GPIO_AF2
    
    /* 电机2: TIM3_CH2, PC7 */
    #define MOTOR2_TIM          TIM3
    #define MOTOR2_TIM_RCC      RCC_TIM3
    #define MOTOR2_CH           TIM_OC2
    #define MOTOR2_PORT         GPIOC
    #define MOTOR2_GPIO_RCC     RCC_GPIOC
    #define MOTOR2_PIN          GPIO7
    #define MOTOR2_AF           GPIO_AF2
    
    /* 电机3: TIM3_CH3, PC8 */
    #define MOTOR3_TIM          TIM3
    #define MOTOR3_TIM_RCC      RCC_TIM3
    #define MOTOR3_CH           TIM_OC3
    #define MOTOR3_PORT         GPIOC
    #define MOTOR3_GPIO_RCC     RCC_GPIOC
    #define MOTOR3_PIN          GPIO8
    #define MOTOR3_AF           GPIO_AF2
    
    /* 电机4: TIM3_CH4, PC9 */
    #define MOTOR4_TIM          TIM3
    #define MOTOR4_TIM_RCC      RCC_TIM3
    #define MOTOR4_CH           TIM_OC4
    #define MOTOR4_PORT         GPIOC
    #define MOTOR4_GPIO_RCC     RCC_GPIOC
    #define MOTOR4_PIN          GPIO9
    #define MOTOR4_AF           GPIO_AF2
    
    /* 电机5: TIM4_CH1, PB6 */
    #define MOTOR5_TIM          TIM4
    #define MOTOR5_TIM_RCC      RCC_TIM4
    #define MOTOR5_CH           TIM_OC1
    #define MOTOR5_PORT         GPIOB
    #define MOTOR5_GPIO_RCC     RCC_GPIOB
    #define MOTOR5_PIN          GPIO6
    #define MOTOR5_AF           GPIO_AF2
    
    /* 电机6: TIM4_CH2, PB7 */
    #define MOTOR6_TIM          TIM4
    #define MOTOR6_TIM_RCC      RCC_TIM4
    #define MOTOR6_CH           TIM_OC2
    #define MOTOR6_PORT         GPIOB
    #define MOTOR6_GPIO_RCC     RCC_GPIOB
    #define MOTOR6_PIN          GPIO7
    #define MOTOR6_AF           GPIO_AF2


#define PWM_FREQUENCY_HZ    50      /* 50Hz for ESCs - 无刷电调标准频率 */

  #define TIMER_CLOCK_HZ      84000000
  #define PWM_PRESCALER       (TIMER_CLOCK_HZ / 1000000)  /* 84: 1MHz时基 */

#define PWM_PERIOD          20000  
#define PWM_MIN_PULSE       1350    
#define PWM_MAX_PULSE       1650   
#define PWM_CENTER_PULSE    1500    

#define RPI_UART_BAUDRATE   115200
#define IMU_UART_BAUDRATE   115200

#endif
