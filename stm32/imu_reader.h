/*
 * IMU数据读取模块头文件
 */

#ifndef IMU_READER_H
#define IMU_READER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    float RollSpeed;    /* 角速度X轴 (°/s) */
    float PitchSpeed;   /* 角速度Y轴 (°/s) */
    float YawSpeed;     /* 角速度Z轴 (°/s) */
    float Roll;         /* 横滚角 (度°) */
    float Pitch;        /* 俯仰角 (度°) */
    float Yaw;          /* 航向角 (度°) */

    float qw;           /* 四元数w */
    float qx;           /* 四元数x */
    float qy;           /* 四元数y */
    float qz;           /* 四元数z */

    uint32_t timestamp; /* 时间戳 */

} imu_data_t;

/* 函数声明 */
void imu_reader_init(void);
int imu_read_data(imu_data_t *data);
bool imu_is_ready(void);
void imu_calibrate(void);
char* get_imu_status(void); 

#endif /* IMU_READER_H */