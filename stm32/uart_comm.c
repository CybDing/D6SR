/*
 * UART通信模块实现
 */

#include "uart_comm.h"
#include "pin_config.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/cm3/nvic.h>

static char rx_buffer[256];
static volatile int rx_write_idx = 0;
static volatile int rx_read_idx = 0;

/*********************************************************************
 * UART初始化
 *********************************************************************/
void uart_comm_init(void) {
    /* 树莓派通信UART1初始化 */
    rcc_periph_clock_enable(RPI_UART_RCC);
    rcc_periph_clock_enable(RPI_UART_GPIO_RCC);
    
    /* F407 GPIO配置 */
    gpio_mode_setup(RPI_UART_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP, 
                    RPI_UART_TX_PIN | RPI_UART_RX_PIN);
    gpio_set_af(RPI_UART_PORT, GPIO_AF7, RPI_UART_TX_PIN | RPI_UART_RX_PIN);
    gpio_set_output_options(RPI_UART_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_50MHZ, RPI_UART_TX_PIN);
    
    /* UART配置 */
    usart_set_baudrate(RPI_UART, RPI_UART_BAUDRATE);
    usart_set_databits(RPI_UART, 8);
    usart_set_stopbits(RPI_UART, USART_STOPBITS_1);
    usart_set_parity(RPI_UART, USART_PARITY_NONE);
    usart_set_flow_control(RPI_UART, USART_FLOWCONTROL_NONE);
    usart_set_mode(RPI_UART, USART_MODE_TX_RX);
    
    /* 使能接收中断 */
    usart_enable_rx_interrupt(RPI_UART);
    nvic_enable_irq(NVIC_USART1_IRQ);
    
    usart_enable(RPI_UART);
    
    /* IMU通信UART2初始化 */
    rcc_periph_clock_enable(IMU_UART_RCC);
    rcc_periph_clock_enable(IMU_UART_GPIO_RCC);
    
    gpio_mode_setup(IMU_UART_PORT, GPIO_MODE_AF, GPIO_PUPD_NONE, 
                    IMU_UART_TX_PIN | IMU_UART_RX_PIN);
    gpio_set_af(IMU_UART_PORT, GPIO_AF7, IMU_UART_TX_PIN | IMU_UART_RX_PIN);
    
    usart_set_baudrate(IMU_UART, IMU_UART_BAUDRATE);
    usart_set_databits(IMU_UART, 8);
    usart_set_stopbits(IMU_UART, USART_STOPBITS_1);
    usart_set_parity(IMU_UART, USART_PARITY_NONE);
    usart_set_flow_control(IMU_UART, USART_FLOWCONTROL_NONE);
    usart_set_mode(IMU_UART, USART_MODE_TX_RX);
    
    usart_enable(IMU_UART);
}

/*********************************************************************
 * UART1中断服务函数
 *********************************************************************/
void usart1_isr(void) {
    /* 检查接收中断 */
    if (((USART_SR(RPI_UART) & USART_SR_RXNE) != 0) && 
        ((USART_CR1(RPI_UART) & USART_CR1_RXNEIE) != 0)) {
        
        char c = usart_recv(RPI_UART);
        
        /* 将字符存入环形缓冲区 */
        rx_buffer[rx_write_idx] = c;
        rx_write_idx = (rx_write_idx + 1) % sizeof(rx_buffer);
        
        /* 防止缓冲区溢出 理论上应该不会出现这种情况*/
        if (rx_write_idx == rx_read_idx) {
            rx_read_idx = (rx_read_idx + 1) % sizeof(rx_buffer);
        }
    }
}

/*********************************************************************
 * 接收一行数据
 *********************************************************************/
int uart_receive_line(uint32_t uart, char *buffer, int max_len) {
    (void)uart;  /* 暂时只支持RPI_UART */
    
    int len = 0;
    char c;
    
    while (rx_read_idx != rx_write_idx && len < (max_len - 1)) {
        c = rx_buffer[rx_read_idx];
        rx_read_idx = (rx_read_idx + 1) % sizeof(rx_buffer);
        
        if (c == '\n' || c == '\r') {
            if (len > 0) {
                buffer[len] = '\0';
                return len;
            }
        } else {
            buffer[len++] = c;
        }
    }
    
    return 0; 
}

/*********************************************************************
 * 发送字符串
 *********************************************************************/
void uart_send_string(uint32_t uart, const char *str) {
    while (*str) {
        usart_send_blocking(uart, *str++);
    }
}

/*********************************************************************
 * 解析PWM命令
 *********************************************************************/
int parse_pwm_command(const char *cmd_str, uint16_t *pwm_values) {
    char *str_copy = malloc(strlen(cmd_str) + 1);
    strcpy(str_copy, cmd_str);

    char *token = strtok(str_copy, ",");
    int count = 0;

    while (token != NULL && count < 6) {
        int value = atoi(token);

        /* 特殊值 -1 表示保持当前值不变 */
        if (value == -1) {
            pwm_values[count++] = 0xFFFF;  /* 使用0xFFFF作为"保持不变"的标记 */
        } else {
            /* 限制PWM值范围 */
            if (value < PWM_MIN_PULSE) value = PWM_MIN_PULSE;
            if (value > PWM_MAX_PULSE) value = PWM_MAX_PULSE;

            pwm_values[count++] = (uint16_t)value;
        }
        token = strtok(NULL, ",");
    }

    free(str_copy);
    return count;
}