#include "imu_reader.h"
#include "pin_config.h"
#include "FreeRTOS.h"
#include "task.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <libopencm3/stm32/usart.h>
#include <libopencm3/cm3/nvic.h>
#include <libopencm3/cm3/cortex.h>

/* IMU数据缓冲区 */
static volatile uint8_t data_buffer[80]; // save data bits read by uart everytime the isr is raised
static volatile uint8_t counts = 0;

#define FRAME_START 0xFC
#define FRAME_END 0xFD
#define IMU_DATA_TYPE 0x41  // 修改为实际值
#define IMU_DATA_LEN 0x30   // 修改为实际值 (48字节)
#define IMU_DATA_RS 56

#define INVALID_LEN 0x03
#define OVERFLOW_ERROR 0x04
#define READY 0x01
#define UNINITIALIZED 0x00
#define DATA_TYPE_ERROR 0x02

static imu_data_t imu_data = {0};
static volatile int last_receive = FRAME_END;
static volatile int imu_status = UNINITIALIZED;

// 调试计数器
static volatile int debug_recv_count = 0;
static volatile int debug_frame_start_count = 0;
static volatile int debug_max_counts = 0;
static volatile uint8_t debug_header[3] = {0}; // 保存最新的头部数据

static float data_trans(uint8_t data1, uint8_t data2, uint8_t data3, uint8_t data4);
static int parse_imu_data(imu_data_t* imu_data_ptr, volatile uint8_t *buffer, char type);

// 使用中断读取不需要考虑数据重叠问题
// static volatile uint8_t imu_data_ready[IMU_DATA_LEN]; // 防止imu数据一部分是之前的，一部分是现在的导致错误 
// static volatile bool is_imu_data_ready = false;
/*********************************************************************
 * IMU读取器初始化
 *********************************************************************/
void imu_reader_init(void) {
    /* IMU UART已在uart_comm_init中初始化 */
    
    usart_enable_rx_interrupt(IMU_UART);
    
    // 设置中断优先级，确保不阻塞FreeRTOS调度
    nvic_set_priority(NVIC_USART2_IRQ, 0x40);  // 较低优先级
    nvic_enable_irq(NVIC_USART2_IRQ);
    if(imu_status == UNINITIALIZED){
        imu_status = READY;
    }
}

/*********************************************************************
 * UART2中断服务函数 接受数据、类别判断
 *********************************************************************/
static bool waiting_flag = true;
void usart2_isr(void) {
    /* 检查接收中断 */
    if (((USART_SR(IMU_UART) & USART_SR_RXNE) != 0) && 
        ((USART_CR1(IMU_UART) & USART_CR1_RXNEIE) != 0)) {
        
        uint8_t new_data = usart_recv(IMU_UART);
        debug_recv_count++;  // 统计接收字节数
        
        if(new_data == FRAME_START) {
            debug_frame_start_count++;  // 统计FRAME_START数量
        }
        
        // 检测帧开始，重置等待标志
        // 在等待状态下，任何FRAME_START都应该触发重置
        if((last_receive == FRAME_END && new_data == FRAME_START) || 
           (waiting_flag && new_data == FRAME_START)) {
            waiting_flag = false;
            counts = 0;  
            // imu_status = READY;
            data_buffer[counts] = new_data;
            counts++;
        }else if(!waiting_flag && (counts > 0)){
            // 边界检查防止溢出
            if(counts >= sizeof(data_buffer)) {
                counts = 0; 
                imu_status = OVERFLOW_ERROR;
                waiting_flag = true;  // 等待下一个FRAME_START
                last_receive = FRAME_END;  // 重置以便检测下个FRAME_START
            } else {
                data_buffer[counts] = new_data;
                counts++;
                
                // 记录最大接收长度用于调试
                if(counts > debug_max_counts) {
                    debug_max_counts = counts;
                }
                
                // 检查数据类型（在收到足够的头部数据后）
                if(counts == 3) {  // 有FRAME_START + DATA_TYPE + DATA_LEN
                    // 保存头部数据用于调试
                    debug_header[0] = data_buffer[0];
                    debug_header[1] = data_buffer[1];
                    debug_header[2] = data_buffer[2];
                    
                    if(data_buffer[1] == IMU_DATA_TYPE && data_buffer[2] == IMU_DATA_LEN){
                        imu_status = IMU_DATA_TYPE;  // 找到类型2数据 (0x41, 0x30)
                    } else {
                        // 忽略类型1数据或其他格式，直接等待下一个FRAME_START
                        waiting_flag = true;  
                        counts = 0;
                        last_receive = FRAME_END;
                        // 不设置错误状态，只是跳过这个数据包
                    }
                }
            }
        }
        if(imu_status == IMU_DATA_TYPE && counts == IMU_DATA_RS){
            if(data_buffer[IMU_DATA_RS - 1] == FRAME_END){
                
                if(parse_imu_data(&imu_data, data_buffer, imu_status) == -1){
                    imu_status = DATA_TYPE_ERROR;
                }
                // 重置接收状态
                counts = 0;
                last_receive = FRAME_END;
                waiting_flag = true;
            }
        }
        if (!waiting_flag){
            last_receive = new_data;
        }
    }
}
static float data_trans(uint8_t data1, uint8_t data2, uint8_t data3, uint8_t data4){
    // 使用union避免strict-aliasing警告
    union { int i; float f; } converter;
    converter.i = ((uint32_t)data4 << 24) |
                  ((uint32_t)data3 << 16) |
                  ((uint32_t)data2 << 8)  |
                  ((uint32_t)data1);
    return converter.f;
}

// 弧度转角度常数
#define RAD_TO_DEG  57.29577951f

static int parse_imu_data(imu_data_t* imu_data_ptr, volatile uint8_t *buffer, char type) {
    if (type == IMU_DATA_TYPE && type == buffer[1] && buffer[2] == IMU_DATA_LEN){
        // 将数据四个字节拼接形成标准浮点数格式进行读取，跳过前3个字节的头部
        // 角速度通常已经是°/s，不需要转换(如果是rad/s则需要乘以RAD_TO_DEG)
        imu_data_ptr->RollSpeed = data_trans(buffer[7], buffer[8], buffer[9], buffer[10]) * RAD_TO_DEG;
        imu_data_ptr->PitchSpeed = data_trans(buffer[11], buffer[12], buffer[13], buffer[14]) * RAD_TO_DEG;
        imu_data_ptr->YawSpeed = data_trans(buffer[15], buffer[16], buffer[17], buffer[18]) * RAD_TO_DEG;

        // 角度从弧度转换为度
        imu_data_ptr->Roll = data_trans(buffer[19], buffer[20], buffer[21], buffer[22]) * RAD_TO_DEG;
        imu_data_ptr->Pitch = data_trans(buffer[23], buffer[24], buffer[25], buffer[26]) * RAD_TO_DEG;
        imu_data_ptr->Yaw = data_trans(buffer[27], buffer[28], buffer[29], buffer[30]) * RAD_TO_DEG;
        imu_data_ptr->qw = data_trans(buffer[31], buffer[32], buffer[33], buffer[34]);
        imu_data_ptr->qx = data_trans(buffer[35], buffer[36], buffer[37], buffer[38]);
        imu_data_ptr->qy = data_trans(buffer[39], buffer[40], buffer[41], buffer[42]);
        imu_data_ptr->qz = data_trans(buffer[43], buffer[44], buffer[45], buffer[46]);
        
        // timestamp是uint32_t，直接按字节拼接，不需要浮点转换
        imu_data_ptr->timestamp = (uint32_t)buffer[47] | 
                                 ((uint32_t)buffer[48] << 8) |
                                 ((uint32_t)buffer[49] << 16) |
                                 ((uint32_t)buffer[50] << 24);
        
        return 0; // 解析成功

    }else{
        return -1;  
    }
}

/*********************************************************************
 * 读取IMU数据 
 *********************************************************************/
int imu_read_data(imu_data_t *data) {
    if (imu_status != IMU_DATA_TYPE || !data) {
        return -1;
    }
    
    // 临时禁用中断防止数据竞态
    cm_disable_interrupts();
    *data = imu_data;
    cm_enable_interrupts();
    
    return 0;
}

char* get_imu_status(void){
    static char debug_msg[128];  // 使用静态缓冲区
    
    snprintf(debug_msg, sizeof(debug_msg), 
             "Status:%d, Recv:%d, Start:%d, Header:[%02X,%02X,%02X], Wait:%d", 
             imu_status, debug_recv_count, debug_frame_start_count, 
             debug_header[0], debug_header[1], debug_header[2], waiting_flag ? 1 : 0);
    
    return debug_msg;
}

bool imu_is_ready(void) {
    return (imu_status == READY || imu_status == IMU_DATA_TYPE);
}

/*********************************************************************
 * IMU校准
 * 注意: 具体校准过程需要根据IMU类型实现
 *********************************************************************/
void imu_calibrate(void) {
    /* 发送校准命令到IMU (如果支持) */
    const char *cal_cmd = "CAL\n";
    for (const char *p = cal_cmd; *p; p++) {
        usart_send_blocking(IMU_UART, *p);
    }
    
}