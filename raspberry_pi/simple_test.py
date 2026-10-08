#!/usr/bin/env python3
"""
简单的树莓派测试程序
用于快速测试与STM32的通信
"""

import serial
import time

def main():
    # 串口配置
    SERIAL_PORT = '/dev/ttyUSB0'  # 根据实际情况修改
    BAUDRATE = 115200
    
    try:
        # 打开串口
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"已连接到 {SERIAL_PORT}")
        
        # 等待系统稳定
        time.sleep(2)
        
        print("开始测试...")
        
        # 测试1: 发送PWM命令
        print("\n1. 测试PWM命令")
        test_pwm_commands = [
            "PWM:1500,1500,1500,1500,1500,1500",  # 中间位置
            "PWM:1600,1400,1500,1500,1500,1500",  # 电机1,2测试
            "PWM:1500,1500,1600,1400,1500,1500",  # 电机3,4测试
            "PWM:1500,1500,1500,1500,1600,1400",  # 电机5,6测试
            "PWM:1500,1500,1500,1500,1500,1500",  # 回到中间
        ]
        
        for cmd in test_pwm_commands:
            print(f"发送: {cmd}")
            ser.write((cmd + '\n').encode())
            time.sleep(1)
            
            # 读取回复
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"接收: {response}")
        
        # 测试2: 设置PID参数
        print("\n2. 测试PID参数设置")
        pid_cmd = "PID:1.2,0.15,0.08"
        print(f"发送: {pid_cmd}")
        ser.write((pid_cmd + '\n').encode())
        time.sleep(0.5)
        
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"接收: {response}")
        
        # 测试3: 请求IMU数据
        print("\n3. 测试IMU数据请求")
        for i in range(5):
            imu_cmd = "IMU"
            print(f"发送: {imu_cmd}")
            ser.write((imu_cmd + '\n').encode())
            time.sleep(0.2)
            
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"接收: {response}")
        
        # 测试4: 请求系统状态
        print("\n4. 测试系统状态请求")
        status_cmd = "STATUS"
        print(f"发送: {status_cmd}")
        ser.write((status_cmd + '\n').encode())
        time.sleep(0.5)
        
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"接收: {response}")
        
        # 持续接收数据（可选）
        print("\n5. 持续接收数据 (按Ctrl+C停止)")
        try:
            while True:
                if ser.in_waiting > 0:
                    data = ser.readline().decode().strip()
                    if data:
                        print(f"接收: {data}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n停止接收")
        
    except serial.SerialException as e:
        print(f"串口错误: {e}")
        print("请检查:")
        print("1. 串口设备路径是否正确")
        print("2. 串口是否被其他程序占用")
        print("3. 用户是否有串口访问权限")
    except Exception as e:
        print(f"其他错误: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")

if __name__ == "__main__":
    main()