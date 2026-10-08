#!/usr/bin/env python3
"""
简单的串口调试测试程序
用于排查STM32与树莓派之间的基本通信问题
"""

import serial
import time
import threading
import sys

class SerialDebugger:
    def __init__(self, port='/dev/ttyS0', baudrate=115200):
        self.port = port  
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        
    def connect(self):
        """连接串口"""
        try:
            print(f"正在连接串口: {self.port} @ {self.baudrate}bps")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            
            if self.ser.is_open:
                print(f"✓ 串口连接成功: {self.port}")
                print(f"✓ 串口参数: {self.baudrate}-8-N-1")
                return True
            else:
                print(f"✗ 无法打开串口: {self.port}")
                return False
                
        except Exception as e:
            print(f"✗ 串口连接失败: {e}")
            return False
    
    def receive_thread(self):
        """接收数据线程"""
        print("接收线程已启动")
        buffer = ""
        
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    
                    # 处理完整的行
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            print(f"STM32 -> 树莓派: {line}")
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"接收数据错误: {e}")
                break
        
        print("接收线程已退出")
    
    def send_command(self, command):
        """发送命令"""
        if not self.ser or not self.ser.is_open:
            print("✗ 串口未连接")
            return False
            
        try:
            cmd_bytes = (command + '\n').encode('utf-8')
            bytes_written = self.ser.write(cmd_bytes)
            self.ser.flush()
            print(f"树莓派 -> STM32: {command} ({bytes_written}字节)")
            return True
        except Exception as e:
            print(f"✗ 发送失败: {e}")
            return False
    
    def start_receive(self):
        """启动接收线程"""
        self.running = True
        self.rx_thread = threading.Thread(target=self.receive_thread, daemon=True)
        self.rx_thread.start()
    
    def stop(self):
        """停止并断开连接"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已断开")

def main():
    print("=== STM32 <-> 树莓派 串口通信调试工具 ===")
    
    # 创建调试器
    debugger = SerialDebugger()
    
    # 连接串口
    if not debugger.connect():
        print("无法连接串口，程序退出")
        return
    
    # 启动接收线程
    debugger.start_receive()
    
    print("\n可用命令:")
    print("  imu      - 请求IMU数据")
    print("  status   - 请求系统状态")
    print("  pwm      - 发送默认PWM命令")
    print("  ping     - 发送测试命令")
    print("  quit     - 退出程序")
    print()
    
    try:
        while True:
            cmd = input(">>> ").strip()
            
            if cmd == 'quit':
                break
            elif cmd == 'imu':
                debugger.send_command("IMU")
            elif cmd == 'status':
                debugger.send_command("STATUS") 
            elif cmd == 'pwm':
                debugger.send_command("PWM:1500,1500,1500,1500,1500,1500")
            elif cmd == 'ping':
                debugger.send_command("PING")
            elif cmd.startswith('raw '):
                # 发送原始命令
                raw_cmd = cmd[4:]
                debugger.send_command(raw_cmd)
            elif cmd:
                print("未知命令，请重新输入")
            
            # 给一些时间让响应返回
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n程序被中断")
    
    finally:
        debugger.stop()
        print("调试程序已退出")

if __name__ == "__main__":
    main()