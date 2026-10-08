/*
 * UART通信模块头文件
 */

#ifndef UART_COMM_H
#define UART_COMM_H

#include <stdint.h>
#include <libopencm3/stm32/usart.h>
#include "FreeRTOS.h"
#include "queue.h"

/* 函数声明 */
void uart_comm_init(void);
int uart_receive_line(uint32_t uart, char *buffer, int max_len);
void uart_send_string(uint32_t uart, const char *str);
int parse_pwm_command(const char *cmd_str, uint16_t *pwm_values);
extern QueueHandle_t uart_tx_queue;
#endif /* UART_COMM_H */