#!/usr/bin/env python3
"""
Yaw轴双环PID交互式调节工具

功能:
1. 通过串口与STM32通信
2. 实时显示Yaw调试输出
3. 快速发送调试命令
4. 记录调节参数历史

使用方法:
    python3 yaw_pid_tuner.py /dev/ttyUSB0
"""

import serial
import sys
import threading
import time
from datetime import datetime

class YawPIDTuner:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.log_file = f"yaw_tuning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def connect(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"✓ 已连接到 {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

    def disconnect(self):
        """断开串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("已断开连接")

    def send_command(self, cmd):
        """发送命令到STM32"""
        if not self.ser or not self.ser.is_open:
            print("✗ 串口未连接")
            return False

        try:
            self.ser.write((cmd + '\n').encode('utf-8'))
            self.log_to_file(f">>> {cmd}")
            return True
        except Exception as e:
            print(f"✗ 发送失败: {e}")
            return False

    def read_thread(self):
        """读取线程,持续接收STM32数据"""
        while self.running:
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # 高亮Yaw调试输出
                        if "Yaw-Cascade" in line:
                            print(f"\033[92m{line}\033[0m")  # 绿色
                        elif "DEBUG" in line:
                            print(f"\033[93m{line}\033[0m")  # 黄色
                        else:
                            print(line)

                        self.log_to_file(f"<<< {line}")
            except Exception as e:
                if self.running:
                    print(f"读取错误: {e}")
            time.sleep(0.01)

    def log_to_file(self, msg):
        """记录到日志文件"""
        try:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                f.write(f"[{timestamp}] {msg}\n")
        except:
            pass

    def print_help(self):
        """打印帮助信息"""
        print("\n" + "="*70)
        print(" Yaw轴双环PID调节工具 - 快捷命令")
        print("="*70)
        print("\n【基本控制】")
        print("  on          - 启用Yaw双环PID (ZSTAB:1)")
        print("  off         - 禁用Yaw双环PID (ZSTAB:0)")
        print("  debug       - 开启调试输出 (ZDEBUG:1)")
        print("  nodebug     - 关闭调试输出 (ZDEBUG:0)")
        print("  reset       - 重置PID状态")
        print()
        print("【目标角度设置】")
        print("  t0          - 设置目标0° (STABLE:yaw:0)")
        print("  t90         - 设置目标90° (STABLE:yaw:90)")
        print("  t180        - 设置目标180° (STABLE:yaw:180)")
        print("  t270        - 设置目标270° (STABLE:yaw:270)")
        print("  t<angle>    - 设置任意角度 (如: t45)")
        print()
        print("【电机测试】")
        print("  m0          - 电机回中位 (PWM:1500,1500,...)")
        print("  m+          - 电机微增 (PWM:1520,1520,...)")
        print("  m-          - 电机微减 (PWM:1480,1480,...)")
        print()
        print("【状态查询】")
        print("  status      - 查询系统状态 (STATUS)")
        print("  imu         - 查询IMU数据 (IMU)")
        print()
        print("【其他】")
        print("  help/h/?    - 显示此帮助")
        print("  log         - 显示日志文件路径")
        print("  quit/exit   - 退出程序")
        print()
        print("【自定义命令】")
        print("  直接输入STM32命令,如: PWM:1500,1500,1500,1500,1500,1500")
        print("="*70 + "\n")

    def parse_shortcut(self, user_input):
        """解析快捷命令"""
        cmd = user_input.strip().lower()

        # 基本控制
        if cmd == 'on':
            return 'ZSTAB:1'
        elif cmd == 'off':
            return 'ZSTAB:0'
        elif cmd == 'debug':
            return 'ZDEBUG:1'
        elif cmd == 'nodebug':
            return 'ZDEBUG:0'
        elif cmd == 'reset':
            return 'ZRESET'

        # 目标角度
        elif cmd == 't0':
            return 'STABLE:yaw:0'
        elif cmd == 't90':
            return 'STABLE:yaw:90'
        elif cmd == 't180':
            return 'STABLE:yaw:180'
        elif cmd == 't270':
            return 'STABLE:yaw:270'
        elif cmd.startswith('t') and cmd[1:].replace('.','').replace('-','').isdigit():
            angle = cmd[1:]
            return f'STABLE:yaw:{angle}'

        # 电机测试
        elif cmd == 'm0':
            return 'PWM:1500,1500,1500,1500,1500,1500'
        elif cmd == 'm+':
            return 'PWM:1520,1520,1520,1520,1520,1520'
        elif cmd == 'm-':
            return 'PWM:1480,1480,1480,1480,1480,1480'

        # 状态查询
        elif cmd == 'status':
            return 'STATUS'
        elif cmd == 'imu':
            return 'IMU'

        # 帮助
        elif cmd in ['help', 'h', '?']:
            self.print_help()
            return None

        # 日志
        elif cmd == 'log':
            print(f"日志文件: {self.log_file}")
            return None

        # 退出
        elif cmd in ['quit', 'exit', 'q']:
            return 'QUIT'

        # 直接返回原始命令
        else:
            return user_input.strip()

    def run(self):
        """主循环"""
        if not self.connect():
            return

        # 启动读取线程
        self.running = True
        read_thread = threading.Thread(target=self.read_thread, daemon=True)
        read_thread.start()

        print("\n" + "="*70)
        print(" 🎛️  Yaw轴双环PID调节工具已启动")
        print("="*70)
        print(f" 串口: {self.port} @ {self.baudrate}")
        print(f" 日志: {self.log_file}")
        print(" 输入 'help' 查看命令列表")
        print("="*70 + "\n")

        # 初始化:开启调试输出
        print(">>> 自动开启调试输出...")
        self.send_command("ZDEBUG:1")
        time.sleep(0.1)

        try:
            while self.running:
                try:
                    user_input = input("\033[94m> \033[0m")  # 蓝色提示符

                    if not user_input.strip():
                        continue

                    # 解析快捷命令
                    cmd = self.parse_shortcut(user_input)

                    if cmd is None:
                        continue

                    if cmd == 'QUIT':
                        print("正在退出...")
                        break

                    # 发送命令
                    self.send_command(cmd)

                except KeyboardInterrupt:
                    print("\n检测到Ctrl+C,正在退出...")
                    break
                except EOFError:
                    break

        finally:
            self.running = False
            self.disconnect()
            print(f"\n日志已保存到: {self.log_file}")
            print("再见! 👋\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 yaw_pid_tuner.py <串口设备>")
        print("示例: python3 yaw_pid_tuner.py /dev/ttyUSB0")
        print("      python3 yaw_pid_tuner.py COM3  (Windows)")
        sys.exit(1)

    port = sys.argv[1]
    tuner = YawPIDTuner(port)
    tuner.run()


if __name__ == '__main__':
    main()
