import serial
import time
import threading
import multiprocessing
import json
import os
from typing import List, Optional, Dict, Any
import paho.mqtt.client as mqtt
import logging
from collections import deque
import numpy as np
from datetime import datetime


def _plot_in_process(plot_func_name: str, data_dict: dict, plot_save_dir: str, extra_args: dict = None):
    """
    在独立进程中执行绘图，避免matplotlib多线程竞争问题

    Args:
        plot_func_name: 绘图函数名称 ('roll', 'pitch', 'pitch_cascaded', 'yaw')
        data_dict: 数据字典（从deque转换而来）
        plot_save_dir: 图片保存目录
        extra_args: 额外参数
    """
    # 在子进程中导入matplotlib，确保每个进程有独立的实例
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        if plot_func_name == 'roll':
            _plot_roll_stabilizer(data_dict, plot_save_dir)
        elif plot_func_name == 'pitch_cascaded':
            _plot_pitch_cascaded(data_dict, plot_save_dir)
        elif plot_func_name == 'yaw':
            _plot_yaw_cascaded(data_dict, plot_save_dir)
    except Exception as e:
        print(f"[绘图错误] {plot_func_name}: {e}")


def _plot_roll_stabilizer(data_dict: dict, plot_save_dir: str):
    """Roll轴单环PID绘图（在独立进程中执行）"""
    import matplotlib.pyplot as plt

    if len(data_dict['timestamp']) < 10:
        print("[警告] Roll轴数据点太少，跳过绘图")
        return

    t = np.array(data_dict['timestamp'])
    angle = np.array(data_dict['roll'])
    error = np.array(data_dict['error'])
    pid_control = np.array(data_dict['pid_control'])
    gravity_comp = np.array(data_dict['gravity_comp'])
    control = np.array(data_dict['control'])
    feedforward = np.array(data_dict['feedforward'])
    delta = np.array(data_dict['delta'])
    motor2 = np.array(data_dict['motor2'])
    motor3 = np.array(data_dict['motor3'])

    fig, axes = plt.subplots(5, 1, figsize=(14, 15))
    fig.suptitle(f'Roll-Axis Stabilizer Performance Analysis\n'
                f'Data points: {len(t)}, Duration: {t[-1]:.1f}s',
                fontsize=14, fontweight='bold')

    # 子图1: 角度和误差
    ax1 = axes[0]
    ax1.plot(t, angle, 'b-', linewidth=1.5, label='Roll Angle', alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.plot(t, error, 'r-', linewidth=1, label='Error', alpha=0.7)
    ax1.set_ylabel('Angle (degrees)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Roll Angle & Error', fontsize=12, fontweight='bold')

    # 子图2: PID控制 vs 重力补偿
    ax2 = axes[1]
    ax2.plot(t, pid_control, 'g-', linewidth=1.5, label='PID Control (u_pid)', alpha=0.8)
    ax2.plot(t, gravity_comp, 'orange', linewidth=1.5, label='Gravity Compensation (grav)', alpha=0.8)
    ax2.plot(t, control, 'b--', linewidth=1, label='Total Control (u_total)', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Control Value', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Control Components: PID vs Gravity Compensation', fontsize=12, fontweight='bold')

    # 子图3: 总控制量和前馈
    ax3 = axes[2]
    ax3.plot(t, control, 'g-', linewidth=1.5, label='Total Control (u)', alpha=0.8)
    ax3.plot(t, feedforward, 'm-', linewidth=1.2, label='Feedforward (ff)', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Control Value', fontsize=11)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Total Control & Feedforward', fontsize=12, fontweight='bold')

    # 子图4: Delta (PWM调整量)
    ax4 = axes[3]
    ax4.plot(t, delta, 'c-', linewidth=1.5, label='Delta (PWM adjustment)', alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.set_ylabel('Delta (PWM units)', fontsize=11)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('PWM Adjustment Delta', fontsize=12, fontweight='bold')

    # 子图5: 电机PWM值
    ax5 = axes[4]
    ax5.plot(t, motor2, 'orange', linewidth=1.5, label='Motor M2 PWM', alpha=0.8)
    ax5.plot(t, motor3, 'purple', linewidth=1.5, label='Motor M3 PWM', alpha=0.8)
    ax5.axhline(y=1500, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label='Neutral (1500)')
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('PWM Value', fontsize=11)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Motor PWM Output (Roll: M2, M3)', fontsize=12, fontweight='bold')
    ax5.set_ylim([1300, 1700])

    plt.tight_layout()

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'roll_stab_analysis_{timestamp_str}.png'
    filepath = os.path.join(plot_save_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[绘图] Roll轴稳定器数据分析图已保存: {filepath}")
    print(f"       数据点: {len(t)}, 持续时间: {t[-1]:.1f}秒\n")

def _plot_pitch_cascaded(data_dict: dict, plot_save_dir: str):
    """Pitch轴双环PID绘图（在独立进程中执行）"""
    import matplotlib.pyplot as plt

    if len(data_dict['timestamp']) < 10:
        print("[警告] Pitch双环PID数据点太少，跳过绘图")
        return

    t = np.array(data_dict['timestamp'])
    angle = np.array(data_dict['angle'])
    target_angle = np.array(data_dict['target_angle'])
    angle_error = np.array(data_dict['angle_error'])
    velocity = np.array(data_dict['velocity'])
    velocity_target = np.array(data_dict['velocity_target'])
    velocity_error = np.array(data_dict['velocity_error'])
    gravity_comp = np.array(data_dict['gravity_comp'])
    delta = np.array(data_dict['delta'])
    motor4 = np.array(data_dict['motor4'])
    motor5 = np.array(data_dict['motor5'])

    fig, axes = plt.subplots(5, 1, figsize=(14, 15))
    fig.suptitle('Pitch-Axis Cascaded PID Performance Analysis (with Gravity Compensation)\n'
                f'Data points: {len(t)}, Duration: {t[-1]:.1f}s',
                fontsize=14, fontweight='bold')

    # 子图1: 角度跟踪 (外环)
    ax1 = axes[0]
    ax1.plot(t, angle, 'b-', linewidth=1.5, label='Current Angle (Pitch)', alpha=0.8)
    ax1.plot(t, target_angle, 'g--', linewidth=1.5, label='Target Angle', alpha=0.7)
    ax1.plot(t, angle_error, 'r-', linewidth=1, label='Angle Error', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Angle (degrees)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Outer Loop: Angle Tracking', fontsize=12, fontweight='bold')

    # 子图2: 速度跟踪 (内环)
    ax2 = axes[1]
    ax2.plot(t, velocity, 'b-', linewidth=1.5, label='Current Velocity (PitchSpeed)', alpha=0.8)
    ax2.plot(t, velocity_target, 'orange', linewidth=1.5, label='Velocity Target (Outer Output)', alpha=0.8)
    ax2.plot(t, velocity_error, 'r-', linewidth=1, label='Velocity Error', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=40, color='r', linestyle=':', linewidth=0.8, alpha=0.5, label='Speed Limit (±40°/s)')
    ax2.axhline(y=-40, color='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Velocity (°/s)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Inner Loop: Velocity Tracking', fontsize=12, fontweight='bold')

    # 子图3: 重力补偿
    ax3 = axes[2]
    ax3.plot(t, gravity_comp, 'm-', linewidth=1.5, label='Gravity Compensation', alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Gravity Comp (PWM units)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Gravity Compensation Output', fontsize=12, fontweight='bold')

    # 子图4: PWM增量 (内环输出 + 重力补偿)
    ax4 = axes[3]
    ax4.plot(t, delta, 'c-', linewidth=1.5, label='PWM Delta (Inner Output + Grav)', alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.axhline(y=60, color='r', linestyle=':', linewidth=0.8, alpha=0.5, label='Delta Limit (±60)')
    ax4.axhline(y=-60, color='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax4.set_ylabel('Delta (PWM units)', fontsize=11)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Control Output: Total PWM Adjustment', fontsize=12, fontweight='bold')

    # 子图5: 电机PWM值
    ax5 = axes[4]
    ax5.plot(t, motor4, 'orange', linewidth=1.5, label='Motor M4 PWM', alpha=0.8)
    ax5.plot(t, motor5, 'purple', linewidth=1.5, label='Motor M5 PWM', alpha=0.8)
    ax5.axhline(y=1650, color='g', linestyle='--', linewidth=0.8, alpha=0.5, label='Base PWM (1650)')
    ax5.axhline(y=1570, color='r', linestyle=':', linewidth=0.8, alpha=0.5, label='PWM Limits')
    ax5.axhline(y=1730, color='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('PWM Value', fontsize=11)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Motor PWM Output (Pitch: M4, M5)', fontsize=12, fontweight='bold')
    ax5.set_ylim([1500, 1800])

    plt.tight_layout()

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'pitch_cascaded_pid_analysis_{timestamp_str}.png'
    filepath = os.path.join(plot_save_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[绘图] Pitch轴双环级联PID数据分析图已保存: {filepath}")
    print(f"       数据点: {len(t)}, 持续时间: {t[-1]:.1f}秒")
    print(f"       外环(角度): 误差范围 [{angle_error.min():.1f}, {angle_error.max():.1f}]°")
    print(f"       内环(速度): 目标范围 [{velocity_target.min():.1f}, {velocity_target.max():.1f}]°/s")
    print(f"       重力补偿: 范围 [{gravity_comp.min():.1f}, {gravity_comp.max():.1f}]\n")


def _plot_yaw_cascaded(data_dict: dict, plot_save_dir: str):
    """Yaw轴双环PID绘图（在独立进程中执行）"""
    import matplotlib.pyplot as plt

    if len(data_dict['timestamp']) < 10:
        print("[警告] Yaw双环PID数据点太少，跳过绘图")
        return

    t = np.array(data_dict['timestamp'])
    angle = np.array(data_dict['angle'])
    target_angle = np.array(data_dict['target_angle'])
    angle_error = np.array(data_dict['angle_error'])
    velocity = np.array(data_dict['velocity'])
    velocity_target = np.array(data_dict['velocity_target'])
    velocity_error = np.array(data_dict['velocity_error'])
    delta = np.array(data_dict['delta'])
    motor0 = np.array(data_dict['motor0'])
    motor1 = np.array(data_dict['motor1'])

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Yaw-Axis Cascaded PID Performance Analysis (Dual-Loop Control)\n'
                f'Data points: {len(t)}, Duration: {t[-1]:.1f}s',
                fontsize=14, fontweight='bold')

    # 子图1: 角度跟踪 (外环)
    ax1 = axes[0]
    ax1.plot(t, angle, 'b-', linewidth=1.5, label='Current Angle', alpha=0.8)
    ax1.plot(t, target_angle, 'g--', linewidth=1.5, label='Target Angle', alpha=0.7)
    ax1.plot(t, angle_error, 'r-', linewidth=1, label='Angle Error', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Angle (degrees)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Outer Loop: Angle Tracking', fontsize=12, fontweight='bold')

    # 子图2: 速度跟踪 (内环)
    ax2 = axes[1]
    ax2.plot(t, velocity, 'b-', linewidth=1.5, label='Current Velocity (YawSpeed)', alpha=0.8)
    ax2.plot(t, velocity_target, 'orange', linewidth=1.5, label='Velocity Target (Outer Output)', alpha=0.8)
    ax2.plot(t, velocity_error, 'r-', linewidth=1, label='Velocity Error', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=60, color='r', linestyle=':', linewidth=0.8, alpha=0.5, label='Speed Limit (±60°/s)')
    ax2.axhline(y=-60, color='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Velocity (°/s)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Inner Loop: Velocity Tracking', fontsize=12, fontweight='bold')

    # 子图3: PWM增量 (内环输出)
    ax3 = axes[2]
    ax3.plot(t, delta, 'c-', linewidth=1.5, label='PWM Delta (Inner Output)', alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.axhline(y=100, color='r', linestyle=':', linewidth=0.8, alpha=0.5, label='Delta Limit (±100)')
    ax3.axhline(y=-100, color='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Delta (PWM units)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Control Output: PWM Adjustment', fontsize=12, fontweight='bold')

    # 子图4: 电机PWM值
    ax4 = axes[3]
    ax4.plot(t, motor0, 'orange', linewidth=1.5, label='Motor M0 PWM', alpha=0.8)
    ax4.plot(t, motor1, 'purple', linewidth=1.5, label='Motor M1 PWM', alpha=0.8)
    ax4.axhline(y=1550, color='g', linestyle='--', linewidth=0.8, alpha=0.5, label='Base PWM (1550)')
    ax4.axhline(y=1400, color='r', linestyle=':', linewidth=0.8, alpha=0.5, label='PWM Limits')
    ax4.axhline(y=1700, color='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('PWM Value', fontsize=11)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Motor PWM Output (Yaw: M0, M1)', fontsize=12, fontweight='bold')
    ax4.set_ylim([1300, 1800])

    plt.tight_layout()

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'yaw_cascaded_pid_analysis_{timestamp_str}.png'
    filepath = os.path.join(plot_save_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[绘图] Yaw轴双环级联PID数据分析图已保存: {filepath}")
    print(f"       数据点: {len(t)}, 持续时间: {t[-1]:.1f}秒")
    print(f"       外环(角度): 误差范围 [{angle_error.min():.1f}, {angle_error.max():.1f}]°")
    print(f"       内环(速度): 目标范围 [{velocity_target.min():.1f}, {velocity_target.max():.1f}]°/s\n")

class SphericalRobotController:
    def __init__(self, serial_port: str = '/dev/ttyAMA0', baudrate: int = 115200, 
                 mqtt_broker: str = "192.168.43.154", mqtt_port: int = 1883,
                 mqtt_username: str = None, mqtt_password: str = None):
         
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.running = False
        self.rx_thread = None
        
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
        self.mqtt_client = None
        self.mqtt_connected = False
        
        self.mqtt_topics = {
            'pwm_cmd': 'spherical_robot/cmd/pwm',      # PWM控制命令
            'pid_cmd': 'spherical_robot/cmd/pid',      # PID参数设置命令
            'imu_cmd': 'spherical_robot/cmd/imu',      # IMU数据请求命令
            'status_cmd': 'spherical_robot/cmd/status', # 状态请求命令
            'smooth_cmd': 'spherical_robot/cmd/smooth', # PWM平滑控制开关
            'ystab_cmd': 'spherical_robot/cmd/ystab',  # Y轴稳定器开关
            'imu_data': 'spherical_robot/data/imu',    # IMU数据输出
            'status_data': 'spherical_robot/data/status', # 系统状态输出
            'debug': 'spherical_robot/data/debug'      # 调试信息输出
        }

        self.motor_pwm = [1500] * 6  # 6个电机PWM值
        self.imu_data = {
                    'rollspeed': 0,     # 角速度X轴
                    'pitchspeed': 0,    # 角速度Y轴  
                    'yawspeed': 0,      # 角速度Z轴
                    'roll': 0,          # 陀螺仪X轴角度
                    'pitch': 0,         # 陀螺仪Y轴角度
                    'yaw': 0,           # 陀螺仪Z轴角度
                    'qw': 0,            # 四元数w
                    'qx': 0,            # 四元数x
                    'qy': 0,            # 四元数y
                    'qz': 0,            # 四元数z
                    'timestamp': time.time()
                }
        self.system_status = {
            'motor_ready': False,
            'imu_ready': False, 
            'pid_enabled': False
        }
        
        self.pid_params = {
            'kp': 1.0,
            'ki': 0.1,
            'kd': 0.05
        }

        # Roll轴稳定器数据采集和绘图 (对应原Y轴,电机M2,M3)
        self.ystab_enabled = False  # 稳定器运行状态
        self.ystab_data_buffer = {
            'timestamp': deque(maxlen=10000),  # 最多保存10000个数据点
            'roll': deque(maxlen=10000),
            'error': deque(maxlen=10000),
            'pid_control': deque(maxlen=10000),      # PID控制量（不含重力补偿）
            'gravity_comp': deque(maxlen=10000),     # 重力补偿项
            'control': deque(maxlen=10000),          # 总控制量
            'feedforward': deque(maxlen=10000),
            'delta': deque(maxlen=10000),
            'motor2': deque(maxlen=10000),
            'motor3': deque(maxlen=10000)
        }
        self.ystab_start_time = None

        # Yaw轴双环PID数据采集和绘图 (电机M0,M1)
        self.zstab_enabled = False  # 稳定器运行状态
        self.zstab_data_buffer = {
            'timestamp': deque(maxlen=10000),
            'angle': deque(maxlen=10000),          # 当前角度
            'target_angle': deque(maxlen=10000),   # 目标角度
            'angle_error': deque(maxlen=10000),    # 角度误差
            'velocity': deque(maxlen=10000),       # 当前角速度
            'velocity_target': deque(maxlen=10000),# 速度目标（外环输出）
            'velocity_error': deque(maxlen=10000), # 速度误差（计算得出）
            'gravity_comp': deque(maxlen=10000),   # 重力补偿
            'delta': deque(maxlen=10000),          # PWM增量
            'motor0': deque(maxlen=10000),         # M0 PWM
            'motor1': deque(maxlen=10000)          # M1 PWM
        }
        self.zstab_start_time = None

        # Pitch轴双环PID数据采集和绘图 (电机M4,M5)
        self.pitch_cascaded_enabled = False  # 稳定器运行状态
        self.pitch_cascaded_buffer = {
            'timestamp': deque(maxlen=10000),
            'angle': deque(maxlen=10000),          # 当前角度
            'target_angle': deque(maxlen=10000),   # 目标角度
            'angle_error': deque(maxlen=10000),    # 角度误差
            'velocity': deque(maxlen=10000),       # 当前角速度
            'velocity_target': deque(maxlen=10000),# 速度目标（外环输出）
            'velocity_error': deque(maxlen=10000), # 速度误差（计算得出）
            'gravity_comp': deque(maxlen=10000),   # 重力补偿
            'delta': deque(maxlen=10000),          # PWM增量
            'motor4': deque(maxlen=10000),         # M4 PWM
            'motor5': deque(maxlen=10000)          # M5 PWM
        }
        self.pitch_cascaded_start_time = None

        self.plot_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(self.plot_save_dir, exist_ok=True)

        # 配置日志同时输出到文件和控制台
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 创建logger - 使用唯一名称避免冲突
        self.logger = logging.getLogger('spherical_robot')
        self.logger.setLevel(logging.DEBUG)

        # 清除已有的handlers（避免重复输出）
        self.logger.handlers.clear()

        # 文件handler - 保存所有调试信息
        # 使用当前工作目录或脚本所在目录
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            script_dir = os.getcwd()
        log_file = os.path.join(script_dir, 'spherical_robot_debug.log')

        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_formatter)
            self.logger.addHandler(file_handler)
            print(f"日志文件创建成功: {log_file}")
        except Exception as e:
            print(f"警告: 无法创建日志文件 {log_file}: {e}")

        # 控制台handler - 只显示INFO及以上级别
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        # 防止日志向上传播到root logger
        self.logger.propagate = False
        
        # 记录启动信息
        self.logger.info("="*50)
        self.logger.info(f"球形机器人控制器启动 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info("="*50)
    
    def setup_mqtt(self):
        self.mqtt_client = mqtt.Client()
        
        if self.mqtt_username and self.mqtt_password:
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)
            self.logger.info("MQTT认证已配置")
        
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message

        # 启动时刻发送一次status信息，进行测试
        self.mqtt_client.will_set(self.mqtt_topics['status_data'], "offline", retain=True)
    
    def connect_mqtt(self) -> bool:
        try:
            self.logger.info(f"正在连接MQTT服务器: {self.mqtt_broker}:{self.mqtt_port}")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
            for _ in range(50):  # 5秒超时
                if self.mqtt_connected:
                    break
                time.sleep(0.1)
            
            return self.mqtt_connected
        except Exception as e:
            self.logger.error(f"MQTT连接失败: {e}")
            print("[!] mqtt连接失败!") 
            return False
        
    def connect(self) -> bool:
        success = True
        
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            
            if self.ser.is_open:
                self.logger.info(f"成功连接到串口 {self.serial_port}")
                print(f"成功连接到串口{self.serial_port}")
                self.running = True
                
                # 启动接收线程
                self.rx_thread = threading.Thread(target=self._receive_thread, daemon=True)
                self.rx_thread.start()
                
                time.sleep(1)
                
                self.request_status()
            else:
                self.logger.error(f"无法打开串口 {self.serial_port}")
                print("[!] 无法打开串口!")
                success = False
                
        except Exception as e:
            self.logger.error(f"串口连接失败: {e}")
            print(f"串口连接失败: {e}")
            success = False
        
        if success:
            self.setup_mqtt()
            if not self.connect_mqtt():
                self.logger.warning("MQTT连接失败, 但串口连接成功, 程序继续运行")
        
        return success
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            self.logger.info("MQTT连接成功")
            print("MQTT连接成功!")
            
            # 订阅所有命令话题
            topics = [
                self.mqtt_topics['pwm_cmd'],
                self.mqtt_topics['pid_cmd'],
                self.mqtt_topics['imu_cmd'],
                self.mqtt_topics['status_cmd'],
                self.mqtt_topics['smooth_cmd'],
                self.mqtt_topics['ystab_cmd']
            ]
            
            for topic in topics:
                client.subscribe(topic)
                self.logger.info(f"已订阅话题: {topic}")
                print(f"已订阅话题: {topic}")
            
            # 发布上线状态
            client.publish(self.mqtt_topics['status_data'], "online", retain=True)
            
        else:
            self.logger.error(f"MQTT连接失败, 错误码: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        self.logger.warning("MQTT连接断开")
        print("mqtt连接已断开!")
    
    def _on_mqtt_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            self.logger.info(f"收到MQTT消息: {topic} -> {payload}")
            print(f"收到MQTT消息: {topic} -> {payload}")
            
            # 根据话题类型处理消息
            if topic == self.mqtt_topics['pwm_cmd']:
                self._process_pwm_command(payload)
            elif topic == self.mqtt_topics['pid_cmd']:
                self._process_pid_command(payload)
            elif topic == self.mqtt_topics['imu_cmd']:
                self._process_imu_command(payload)
            elif topic == self.mqtt_topics['status_cmd']:
                self._process_status_command(payload)
            elif topic == self.mqtt_topics['smooth_cmd']:
                self._process_smooth_command(payload)
            elif topic == self.mqtt_topics['ystab_cmd']:
                self._process_ystab_command(payload)
                
        except Exception as e:
            self.logger.error(f"处理MQTT消息错误: {e}")
    
    def _process_pwm_command(self, payload: str):
        """处理PWM控制命令"""
        try:
            # 直接解析PWM数组: [1500,1500,1500,1500,1500,1500] 
            pwm_values = json.loads(payload)
            if isinstance(pwm_values, list) and len(pwm_values) == 6:
                self.logger.debug(f"PWM命令: {pwm_values}")
                self.set_motor_pwm(pwm_values)
                # 简单确认回复
                if self.mqtt_client and self.mqtt_connected:
                    self.mqtt_client.publish(self.mqtt_topics['status_data'], "pwm_ok")
            else:
                self.logger.error("PWM格式错误: 需要6个数值的数组")
        except Exception as e:
            self.logger.error(f"PWM命令处理错误: {e}")
    
    def _process_pid_command(self, payload: str):
        """处理PID参数设置命令"""
        try:
            # 直接解析PID数组: [1.0,0.1,0.05]
            pid_values = json.loads(payload)
            if isinstance(pid_values, list) and len(pid_values) == 3:
                kp, ki, kd = pid_values
                self.logger.debug(f"PID命令: kp={kp}, ki={ki}, kd={kd}")
                self.set_pid_params(kp, ki, kd)
                # 简单确认回复
                if self.mqtt_client and self.mqtt_connected:
                    self.mqtt_client.publish(self.mqtt_topics['status_data'], "pid_ok")
            else:
                self.logger.error("PID格式错误: 需要3个数值的数组")
        except Exception as e:
            self.logger.error(f"PID命令处理错误: {e}")
    
    def _process_imu_command(self, payload: str):
        """处理IMU数据请求命令"""
        try:
            # 任何消息都触发IMU请求，比如发送 "get"
            self.logger.debug(f"IMU数据请求: {payload}")
            self.request_imu_data()
        except Exception as e:
            self.logger.error(f"IMU命令处理错误: {e}")
    
    def _process_status_command(self, payload: str):
        """处理状态请求命令"""
        try:
            # 任何消息都触发状态请求，比如发送 "get"
            self.logger.debug(f"状态数据请求: {payload}")
            self.request_status()
        except Exception as e:
            self.logger.error(f"状态命令处理错误: {e}")

    def _process_smooth_command(self, payload: str):
        """处理PWM平滑控制开关命令"""
        try:
            # 期望 "1" 或 "0"
            enable = int(payload.strip())
            self.logger.debug(f"平滑控制命令: {enable}")
            self.set_smooth_mode(enable)
            if self.mqtt_client and self.mqtt_connected:
                self.mqtt_client.publish(self.mqtt_topics['status_data'], "smooth_ok")
        except Exception as e:
            self.logger.error(f"平滑控制命令处理错误: {e}")

    def _process_ystab_command(self, payload: str):
        """处理Y轴稳定器开关命令"""
        try:
            # 期望 "1" 或 "0"
            enable = int(payload.strip())
            self.logger.debug(f"Y轴稳定器命令: {enable}")
            self.set_ystab_mode(enable)
            if self.mqtt_client and self.mqtt_connected:
                self.mqtt_client.publish(self.mqtt_topics['status_data'], "ystab_ok")
        except Exception as e:
            self.logger.error(f"Y轴稳定器命令处理错误: {e}")
    
    def _publish_simple_status(self, message: str):
        """发布简单状态消息"""
        if self.mqtt_client and self.mqtt_connected:
            self.mqtt_client.publish(self.mqtt_topics['status_data'], message)
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        
        # 发送离线状态并断开MQTT连接
        if self.mqtt_client and self.mqtt_connected:
            try:
                self.mqtt_client.publish(self.mqtt_topics['status_data'], "offline", retain=True)
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception as e:
                self.logger.error(f"MQTT断开连接错误: {e}")
        
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=2)
        
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("已断开所有连接")
    
    def _send_command(self, command: str):
        if not self.ser or not self.ser.is_open:
            self.logger.error("串口未连接，无法发送命令")
            print("串口未连接，无法发送命令")
            return
        
        try:
            cmd_bytes = (command + '\n').encode('utf-8')
            bytes_written = self.ser.write(cmd_bytes)
            self.ser.flush()  # 确保数据立即发送
            self.logger.info(f"串口发送: '{command}' ({bytes_written}字节)")
            print(f"串口发送: '{command}' ({bytes_written}字节)")
        except Exception as e:
            self.logger.error(f"串口发送失败: {e}")
            print("command在串口发送失败!")
    
    def _receive_thread(self):
        buffer = ""
        self.logger.info("数据接收线程已启动")
        print("串口正在接收数据!")
        
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    bytes_waiting = self.ser.in_waiting
                    if bytes_waiting > 0:
                        self.logger.debug(f"串口有{bytes_waiting}字节待读取")
                        data = self.ser.read(bytes_waiting).decode('utf-8', errors='ignore')
                        self.logger.debug(f"原始接收数据: {repr(data)}")
                        buffer += data
                        
                        # 处理完整的行
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if line:
                                self.logger.debug(f"处理完整行: {line}")
                                self._process_received_data(line)
                else:
                    self.logger.warning("串口未打开或已断开")
                    break
                
                time.sleep(0.01)  # 避免占用过多CPU
                
            except Exception as e:
                self.logger.error(f"串口接收数据错误: {e}")
                time.sleep(0.1)
        
        self.logger.info("数据接收线程已退出")
    
    def _process_received_data(self, data: str):
        # 消息分类处理:
        # 1. Roll:/Pitch:/Yaw: 角度调试数据 - 仅写入日志(DEBUG级别), 不在控制台显示
        # 2. DEBUG: 状态消息 - 写入日志并在控制台显示(INFO级别)
        # 3. IMU:/STATUS: 手动请求响应 - 写入日志并在控制台显示(INFO级别)
        # 4. 其他消息 - 写入日志并在控制台显示(INFO级别)

        # 角度调试数据: 仅写入日志文件
        # 注意: Yaw-Cascade: 和 Pitch-Cascade: 是STM32双环PID发送的格式，需要单独匹配
        if data.startswith("Roll:") or data.startswith("Pitch:") or data.startswith("Yaw:") or data.startswith("Yaw-Cascade:") or data.startswith("Pitch-Cascade:"):
            self.logger.debug(f"STM32->树莓派: {data}")
        # DEBUG状态消息: 写入日志并在控制台显示
        elif data.startswith("DEBUG:"):
            self.logger.info(f"STM32->树莓派: {data}")
            print(f"STM32->树莓派: {data}")  # 在控制台显示重要状态
        # 其他消息(IMU/STATUS等): 写入日志并在控制台显示
        else:
            self.logger.info(f"STM32->树莓派: {data}")

        # 数据处理逻辑（与日志级别分离）
        if data.startswith("IMU:"):
            self._parse_imu_data(data[4:])
        elif data.startswith("STATUS:"):
            self._parse_status_data(data[7:])
        elif data.startswith("Roll:"):  # Roll轴稳定器数据 (M2,M3)
            self._handle_roll_debug(data.split(":", 1)[1])
        elif data.startswith("Pitch-Cascade:"):  # Pitch轴双环PID数据 (M4,M5)
            self._handle_pitch_cascaded_debug(data)  # 直接传递整个消息
        elif data.startswith("Yaw-Cascade:"):  # Yaw轴双环PID数据 (M0,M1) - 新格式
            self._handle_yaw_debug(data)  # 直接传递整个消息，_collect_yaw_data 会解析
        elif data.startswith("Yaw:"):  # Yaw轴稳定器数据 (M0,M1) - 旧格式兼容
            self._handle_yaw_debug(data.split(":", 1)[1])
        elif data.startswith("DEBUG:"):
            self._handle_debug_message(data[6:])
    
    def _parse_imu_data(self, data_str: str):
        """解析IMU数据并转发到MQTT"""
        try:
            values = [float(x) for x in data_str.split(',')]
            if len(values) == 10:
                self.imu_data = {
                    'rollspeed': values[0],     # 角速度X轴
                    'pitchspeed': values[1],    # 角速度Y轴  
                    'yawspeed': values[2],      # 角速度Z轴
                    'roll': values[3],          # 陀螺仪X轴角度
                    'pitch': values[4],         # 陀螺仪Y轴角度
                    'yaw': values[5],           # 陀螺仪Z轴角度
                    'qw': values[6],            # 四元数w
                    'qx': values[7],            # 四元数x
                    'qy': values[8],            # 四元数y
                    'qz': values[9],            # 四元数z
                    'timestamp': time.time()
                }
            else:
                self.logger.error(f"IMU数据格式错误: 需要10个值,收到{len(values)}个")
                return
                
            if self.mqtt_client and self.mqtt_connected:
                # 直接发布IMU数据数组
                imu_array = [
                    self.imu_data['rollspeed'], self.imu_data['pitchspeed'], self.imu_data['yawspeed'],
                    self.imu_data['roll'], self.imu_data['pitch'], self.imu_data['yaw'],
                    self.imu_data['qw'], self.imu_data['qx'], self.imu_data['qy'], self.imu_data['qz']
                ]
                self.mqtt_client.publish(self.mqtt_topics['imu_data'], json.dumps(imu_array))

        except ValueError as e:
            self.logger.error(f"IMU数据解析错误: {e}")
    
    def _parse_status_data(self, data_str: str):
        """解析状态数据并转发到MQTT"""
        try:
            values = [int(x) for x in data_str.split(',')]
            if len(values) == 3:
                self.system_status = {
                    'motor_ready': bool(values[0]),
                    'imu_ready': bool(values[1]),
                    'pid_enabled': bool(values[2])
                }
                
                # 发送到MQTT - 发布简单的状态数组
                if self.mqtt_client and self.mqtt_connected:
                    status_array = [
                        int(self.system_status['motor_ready']),
                        int(self.system_status['imu_ready']),
                        int(self.system_status['pid_enabled'])
                    ]
                    self.mqtt_client.publish(self.mqtt_topics['status_data'], json.dumps(status_array))

        except ValueError as e:
            self.logger.error(f"状态数据解析错误: {e}")
    
    def _handle_roll_debug(self, message: str):
        """处理Roll轴稳定器调试信息并收集数据"""
        # 格式: <angle>,e<error>,pid<pid_ctrl>,g<grav_comp>,u<total_ctrl>,ff<feedforward>,d<delta>,M2=<pwm>,M3=<pwm>
        # 例如: -5.2,e5.2,pid0.5,g1.0,u1.5,ff0.00,d3,M2=1587,M3=1413

        # 只写入日志文件，不在控制台显示
        self.logger.debug(f"[Roll稳定] {message}")

        if self.mqtt_client and self.mqtt_connected:
            self.mqtt_client.publish(self.mqtt_topics['debug'], f"Roll:{message}")

        # 收集数据用于绘图
        self._collect_roll_data(message)

    def _collect_roll_data(self, message: str):
        """收集Roll轴稳定器数据用于绘图"""
        try:
            # 格式: -5.2,e5.2,pid0.5,g1.0,u1.5,ff0.00,d3,M2=1587,M3=1413
            parts = message.split(',')
            if len(parts) < 9:
                return

            # 初始化开始时间
            if self.ystab_start_time is None:
                self.ystab_start_time = time.time()

            # 解析各个字段 (Roll轴电机M2,M3)
            roll = float(parts[0])  # 直接就是角度值
            error = float(parts[1].replace('e', ''))
            pid_control = float(parts[2].replace('pid', ''))
            gravity_comp = float(parts[3].replace('g', ''))
            control = float(parts[4].replace('u', ''))
            feedforward = float(parts[5].replace('ff', ''))
            delta = int(parts[6].replace('d', ''))
            motor2 = int(parts[7].replace('M2=', ''))
            motor3 = int(parts[8].replace('M3=', '').replace('SAT!', '').replace(',SAT!', ''))

            # 存储数据
            current_time = time.time() - self.ystab_start_time
            self.ystab_data_buffer['timestamp'].append(current_time)
            self.ystab_data_buffer['roll'].append(roll)
            self.ystab_data_buffer['error'].append(error)
            self.ystab_data_buffer['pid_control'].append(pid_control)
            self.ystab_data_buffer['gravity_comp'].append(gravity_comp)
            self.ystab_data_buffer['control'].append(control)
            self.ystab_data_buffer['feedforward'].append(feedforward)
            self.ystab_data_buffer['delta'].append(delta)
            self.ystab_data_buffer['motor2'].append(motor2)
            self.ystab_data_buffer['motor3'].append(motor3)

        except (ValueError, IndexError) as e:
            self.logger.debug(f"Roll稳定器数据解析错误: {e}, 数据: {message}")

    def _handle_yaw_debug(self, message: str):
        """处理Yaw轴稳定器调试信息并收集数据"""
        # 格式: <angle>,e<error>,pid<pid_ctrl>,g<grav_comp>,u<total_ctrl>,ff<feedforward>,d<delta>,M0=<pwm>,M1=<pwm>

        # 只写入日志文件，不在控制台显示
        self.logger.debug(f"[Yaw稳定] {message}")

        if self.mqtt_client and self.mqtt_connected:
            self.mqtt_client.publish(self.mqtt_topics['debug'], f"Yaw:{message}")

        # 收集数据用于绘图
        self._collect_yaw_data(message)

    def _collect_yaw_data(self, message: str):
        """收集Yaw轴双环PID数据用于绘图

        调试格式: Yaw-Cascade: ang=5.2, tgt=0.0, aerr=-5.2, v_fb=4.5, vtgt=-7.8, verr=25.5, grav=0.0, delta=-15, M0=1485, M1=1515

        注意：
        - aerr是STM32端经过360度环绕处理的最短路径角度误差（-180~+180°）
        """
        try:
            # 检查是否是新的双环PID格式
            if 'Yaw-Cascade:' in message:
                # 移除前缀
                data_part = message.split('Yaw-Cascade:')[-1].strip()

                # 解析键值对
                pairs = data_part.split(',')
                data_dict = {}
                for pair in pairs:
                    pair = pair.strip()
                    if '=' in pair:
                        key, value = pair.split('=')
                        data_dict[key.strip()] = value.strip()

                # 初始化开始时间（首次记录数据时）
                if self.zstab_start_time is None:
                    self.zstab_start_time = time.time()

                # 提取数据
                angle = float(data_dict.get('ang', 0))
                target_angle = float(data_dict.get('tgt', 0))
                angle_error = float(data_dict.get('aerr', 0))  # 已经是处理后的最短路径误差
                velocity_target = float(data_dict.get('vtgt', 0))
                velocity = float(data_dict.get('v_fb', 0))
                gravity_comp = float(data_dict.get('grav', 0))
                delta = int(data_dict.get('delta', 0))
                motor0 = int(data_dict.get('M0', 1500))
                motor1 = int(data_dict.get('M1', 1500))

                # 计算速度误差
                velocity_error = velocity_target - velocity

                # 存储数据
                current_time = time.time() - self.zstab_start_time
                self.zstab_data_buffer['timestamp'].append(current_time)
                self.zstab_data_buffer['angle'].append(angle)
                self.zstab_data_buffer['target_angle'].append(target_angle)
                self.zstab_data_buffer['angle_error'].append(angle_error)
                self.zstab_data_buffer['velocity'].append(velocity)
                self.zstab_data_buffer['velocity_target'].append(velocity_target)
                self.zstab_data_buffer['velocity_error'].append(velocity_error)
                self.zstab_data_buffer['gravity_comp'].append(gravity_comp)
                self.zstab_data_buffer['delta'].append(delta)
                self.zstab_data_buffer['motor0'].append(motor0)
                self.zstab_data_buffer['motor1'].append(motor1)

            else:
                # 旧格式兼容（如果需要）
                self.logger.debug(f"未识别的Yaw调试格式: {message}")

        except (ValueError, IndexError, KeyError) as e:
            self.logger.debug(f"Yaw双环PID数据解析错误: {e}, 数据: {message}")

    def _handle_pitch_cascaded_debug(self, message: str):
        """处理Pitch轴双环PID调试信息并收集数据"""
        # 只写入日志文件，不在控制台显示
        self.logger.debug(f"[Pitch双环PID] {message}")

        if self.mqtt_client and self.mqtt_connected:
            self.mqtt_client.publish(self.mqtt_topics['debug'], f"Pitch-Cascade:{message}")

        # 收集数据用于绘图
        self._collect_pitch_cascaded_data(message)

    def _collect_pitch_cascaded_data(self, message: str):
        """收集Pitch轴双环PID数据用于绘图

        调试格式: Pitch-Cascade: ang=5.2, tgt=0.0, aerr=-5.2, v_fb=4.5, vtgt=-7.8, verr=25.5, grav=15.0, delta=-15, M4=1635, M5=1665
        """
        try:
            # 检查是否是双环PID格式
            if 'Pitch-Cascade:' in message:
                # 移除前缀
                data_part = message.split('Pitch-Cascade:')[-1].strip()

                # 解析键值对
                pairs = data_part.split(',')
                data_dict = {}
                for pair in pairs:
                    pair = pair.strip()
                    if '=' in pair:
                        key, value = pair.split('=')
                        data_dict[key.strip()] = value.strip()

                # 初始化开始时间（首次记录数据时）
                if self.pitch_cascaded_start_time is None:
                    self.pitch_cascaded_start_time = time.time()

                # 提取数据
                angle = float(data_dict.get('ang', 0))
                target_angle = float(data_dict.get('tgt', 0))
                angle_error = float(data_dict.get('aerr', 0))
                velocity_target = float(data_dict.get('vtgt', 0))
                velocity = float(data_dict.get('v_fb', 0))
                gravity_comp = float(data_dict.get('grav', 0))
                delta = int(data_dict.get('delta', 0))
                motor4 = int(data_dict.get('M4', 1650))
                motor5 = int(data_dict.get('M5', 1650))

                # 计算速度误差
                velocity_error = velocity_target - velocity

                # 存储数据
                current_time = time.time() - self.pitch_cascaded_start_time
                self.pitch_cascaded_buffer['timestamp'].append(current_time)
                self.pitch_cascaded_buffer['angle'].append(angle)
                self.pitch_cascaded_buffer['target_angle'].append(target_angle)
                self.pitch_cascaded_buffer['angle_error'].append(angle_error)
                self.pitch_cascaded_buffer['velocity'].append(velocity)
                self.pitch_cascaded_buffer['velocity_target'].append(velocity_target)
                self.pitch_cascaded_buffer['velocity_error'].append(velocity_error)
                self.pitch_cascaded_buffer['gravity_comp'].append(gravity_comp)
                self.pitch_cascaded_buffer['delta'].append(delta)
                self.pitch_cascaded_buffer['motor4'].append(motor4)
                self.pitch_cascaded_buffer['motor5'].append(motor5)

            else:
                self.logger.debug(f"未识别的Pitch双环PID调试格式: {message}")

        except (ValueError, IndexError, KeyError) as e:
            self.logger.debug(f"Pitch双环PID数据解析错误: {e}, 数据: {message}")

    def _deque_to_list(self, data_buffer: dict) -> dict:
        """将deque缓冲区转换为普通列表，以便传递给子进程"""
        return {key: list(value) for key, value in data_buffer.items()}

    def _plot_ystab_data(self):
        """绘制Y轴(Roll)稳定器数据并保存 - 使用独立进程"""
        if len(self.ystab_data_buffer['timestamp']) < 10:
            self.logger.warning("Roll轴数据点太少，跳过绘图")
            return

        # 复制数据到普通字典（deque不能直接pickle）
        data_dict = self._deque_to_list(self.ystab_data_buffer)

        # 在独立进程中执行绘图
        p = multiprocessing.Process(
            target=_plot_in_process,
            args=('roll', data_dict, self.plot_save_dir)
        )
        p.start()
        self.logger.info("Roll轴绘图进程已启动")

    def _clear_ystab_buffer(self):
        """清空Y轴稳定器数据缓存"""
        for key in self.ystab_data_buffer:
            self.ystab_data_buffer[key].clear()
        self.ystab_start_time = None

    def _plot_cascaded_pid_data(self):
        """绘制Yaw轴双环级联PID性能分析图 - 使用独立进程"""
        if len(self.zstab_data_buffer['timestamp']) < 10:
            self.logger.warning("Yaw双环PID数据点太少，跳过绘图")
            return

        # 复制数据到普通字典
        data_dict = self._deque_to_list(self.zstab_data_buffer)

        # 在独立进程中执行绘图
        p = multiprocessing.Process(
            target=_plot_in_process,
            args=('yaw', data_dict, self.plot_save_dir)
        )
        p.start()
        self.logger.info("Yaw轴绘图进程已启动")

    def _plot_zstab_data(self):
        """绘制Yaw轴双环PID数据并保存"""
        self._plot_cascaded_pid_data()

    def _plot_pitch_cascaded_data(self):
        """绘制Pitch轴双环级联PID性能分析图 - 使用独立进程"""
        if len(self.pitch_cascaded_buffer['timestamp']) < 10:
            self.logger.warning("Pitch双环PID数据点太少，跳过绘图")
            return

        # 复制数据到普通字典
        data_dict = self._deque_to_list(self.pitch_cascaded_buffer)

        # 在独立进程中执行绘图
        p = multiprocessing.Process(
            target=_plot_in_process,
            args=('pitch_cascaded', data_dict, self.plot_save_dir)
        )
        p.start()
        self.logger.info("Pitch双环PID绘图进程已启动")

    def _clear_pitch_cascaded_buffer(self):
        """清空Pitch轴双环PID数据缓存"""
        for key in self.pitch_cascaded_buffer:
            self.pitch_cascaded_buffer[key].clear()
        self.pitch_cascaded_start_time = None

    def _clear_zstab_buffer(self):
        """清空Yaw轴稳定器数据缓存"""
        for key in self.zstab_data_buffer:
            self.zstab_data_buffer[key].clear()
        self.zstab_start_time = None

    def _plot_and_clear_ystab_data(self):
        """手动绘制Y轴(Roll)数据并清空缓冲区"""
        self._plot_ystab_data()
        # 不需要等待，因为数据已经复制到子进程
        self._clear_ystab_buffer()
        self.logger.info("Y轴数据缓冲区已清空")
        print("[数据清空] Y轴稳定器数据缓冲区已清空，下次绘图将显示新数据")

    def _plot_and_clear_zstab_data(self):
        """手动绘制Yaw轴数据并清空缓冲区"""
        self._plot_zstab_data()
        self._clear_zstab_buffer()
        self.logger.info("Yaw轴数据缓冲区已清空")
        print("[数据清空] Yaw轴稳定器数据缓冲区已清空，下次绘图将显示新数据")

    def _plot_and_clear_pitch_cascaded_data(self):
        """手动绘制Pitch轴双环PID数据并清空缓冲区"""
        self._plot_pitch_cascaded_data()
        self._clear_pitch_cascaded_buffer()
        self.logger.info("Pitch轴双环PID数据缓冲区已清空")
        print("[数据清空] Pitch轴双环PID数据缓冲区已清空，下次绘图将显示新数据")

    def _handle_debug_message(self, message: str):
        """处理调试信息并转发到MQTT"""
        if self.mqtt_client and self.mqtt_connected:
            # 直接发布调试消息文本
            self.mqtt_client.publish(self.mqtt_topics['debug'], message)
    
    def set_motor_pwm(self, pwm_values: List[int]) -> bool:
        """
        设置电机PWM值
        
        Args:
            pwm_values: 6个电机的PWM值列表 (1000-2000)
            
        Returns:
            bool: 是否发送成功
        """
        if len(pwm_values) != 6:
            print("错误: 必须提供6个电机PWM值")
            return False
        
        # 限制PWM值范围 (-1表示保持当前值不变)
        limited_pwm = []
        for pwm in pwm_values:
            if pwm == -1:
                limited_pwm.append(-1)  # 保持-1不变，表示跳过该电机
            elif pwm < 1000:
                limited_pwm.append(1000)
            elif pwm > 2000:
                limited_pwm.append(2000)
            else:
                limited_pwm.append(pwm)
        
        # 更新本地状态
        self.motor_pwm = limited_pwm
        
        # 发送命令
        pwm_str = ','.join(map(str, limited_pwm))
        self._send_command(f"PWM:{pwm_str}")
        
        return True
    
    def set_pid_params(self, kp: float, ki: float, kd: float) -> bool:
        """
        设置PID参数
        
        Args:
            kp: 比例系数
            ki: 积分系数  
            kd: 微分系数
            
        Returns:
            bool: 是否发送成功
        """
        self.pid_params = {'kp': kp, 'ki': ki, 'kd': kd}
        self._send_command(f"PID:{kp},{ki},{kd}")
        return True
    
    def request_imu_data(self):
        """请求IMU数据"""
        self._send_command("IMU")
    
    def request_status(self):
        """请求系统状态"""
        self._send_command("STATUS")

    def set_smooth_mode(self, enable: bool) -> bool:
        """
        设置PWM平滑控制模式

        Args:
            enable: True启用，False禁用

        """
        cmd_value = 1 if enable else 0
        self._send_command(f"SMOOTH:{cmd_value}")
        return True

    def set_ystab_mode(self, enable: bool) -> bool:
        """
        设置Y轴稳定器模式

        Args:
            enable: True启用，False禁用

        Returns:
            bool: 是否发送成功
        """
        cmd_value = 1 if enable else 0

        # 如果从启用变为禁用，触发绘图
        if self.ystab_enabled and not enable:
            self.logger.info("Y轴稳定器停止，开始生成数据分析图...")
            print("\n[数据分析] Y轴稳定器已停止，正在生成性能分析图...")
            # 在后台线程中绘图，避免阻塞
            threading.Thread(target=self._plot_ystab_data, daemon=True).start()

        # 如果从禁用变为启用，清空之前的数据
        if not self.ystab_enabled and enable:
            self._clear_ystab_buffer()
            self.logger.info("Y轴稳定器启动，开始收集新数据...")
            print("\n[数据采集] Y轴稳定器已启动，开始收集数据用于分析...")

        self.ystab_enabled = enable
        self._send_command(f"YSTAB:{cmd_value}")
        return True

    def set_xstab_mode(self, enable: bool) -> bool:
        """
        设置X轴(Pitch)稳定器模式 - 使用双环级联PID

        Args:
            enable: True启用，False禁用

        Returns:
            bool: 是否发送成功
        """
        cmd_value = 1 if enable else 0

        # 如果正在禁用，检查是否有双环PID数据需要绘图
        if not enable and len(self.pitch_cascaded_buffer['timestamp']) >= 10:
            self.logger.info("Pitch轴双环PID停止，开始生成数据分析图...")
            print("\n[数据分析] Pitch轴双环PID已停止，正在生成性能分析图...")
            threading.Thread(target=self._plot_pitch_cascaded_data, daemon=True).start()

        # 如果从禁用变为启用，清空之前的数据
        if enable:
            self._clear_pitch_cascaded_buffer()
            self.logger.info("Pitch轴双环PID启动，开始收集新数据...")
            print("\n[数据采集] Pitch轴双环PID已启动，开始收集数据用于分析...")

        self._send_command(f"XSTAB:{cmd_value}")
        return True

    def set_zstab_mode(self, enable: bool) -> bool:
        """
        设置Yaw轴(Z轴)稳定器模式

        Args:
            enable: True启用，False禁用

        Returns:
            bool: 是否发送成功
        """
        cmd_value = 1 if enable else 0

        # 如果正在禁用，检查是否有数据需要绘图
        # 改进：不依赖zstab_enabled标志，而是检查实际数据缓存
        if not enable and len(self.zstab_data_buffer['timestamp']) >= 10:
            self.logger.info("Yaw轴稳定器停止，开始生成数据分析图...")
            print("\n[数据分析] Yaw轴稳定器已停止，正在生成性能分析图...")
            # 在后台线程中绘图，避免阻塞
            threading.Thread(target=self._plot_zstab_data, daemon=True).start()

        # 如果从禁用变为启用，清空之前的数据
        if not self.zstab_enabled and enable:
            self._clear_zstab_buffer()
            self.logger.info("Yaw轴稳定器启动，开始收集新数据...")
            print("\n[数据采集] Yaw轴稳定器已启动，开始收集数据用于分析...")

        self.zstab_enabled = enable
        self._send_command(f"ZSTAB:{cmd_value}")
        return True

    def set_motion_velocity(self, velocity_x: float, velocity_y: float) -> bool:
        """
        设置运动速度 (简化版本,直接设置PWM偏移量)

        Args:
            velocity_x: X方向速度 (-1.0 到 +1.0), 正值向前
            velocity_y: Y方向速度 (-1.0 到 +1.0), 正值向左

        示例:
            set_motion_velocity(0.5, 0.0)  # 50%速度向前
            set_motion_velocity(0.0, -0.3)  # 30%速度向右
            set_motion_velocity(0.3, 0.2)  # 向前偏左运动
        """
        # 限制范围
        velocity_x = max(-1.0, min(1.0, velocity_x))
        velocity_y = max(-1.0, min(1.0, velocity_y))

        self._send_command(f"MOTION:{velocity_x:.2f},{velocity_y:.2f}")
        self.logger.info(f"设置运动速度: vx={velocity_x:.2f}, vy={velocity_y:.2f}")
        return True

    def set_target_angle(self, axis_name: str, angle_deg: float) -> bool:
        """
        设置指定轴的目标稳定角度

        Args:
            axis_name: 轴名称 ("Roll", "Pitch", 或 "Yaw")
            angle_deg: 目标角度(度)

        Returns:
            bool: 是否发送成功

        示例:
            set_target_angle("Roll", 30.0)   # 设置Roll轴目标角度为30度
            set_target_angle("Yaw", 0.0)     # 设置Yaw轴目标角度为0度
        """
        self._send_command(f"STABLE:{axis_name},{angle_deg:.2f}")
        self.logger.info(f"设置{axis_name}轴目标角度: {angle_deg:.2f}度")
        return True

    def get_target_angle(self, axis_name: str) -> bool:
        """
        查询指定轴的目标稳定角度

        Args:
            axis_name: 轴名称 ("Roll", "Pitch", 或 "Yaw")

        注意: 查询结果将通过串口返回，格式为 "STABLE:<axis>,<angle>"
        """
        self._send_command(f"GET_STABLE:{axis_name}")
        self.logger.info(f"查询{axis_name}轴目标角度")
        return True

    def motion_stop(self) -> bool:
        """
        停止前进方向的运动 (清零PWM偏移量,但保持稳定器运行)
        """
        self._send_command("MOTIONSTOP")
        self.logger.info("运动控制已停止")
        return True

    def motor_start(self, ) -> bool:
        """
        motor_start: 用来初始化电机转速，保证每个电机都正常开启后开始控制
        """

        self.logger.info("电机预热开启")
        self.logger.info("pwm初始设置: 1550, 1550, 1560, 1440, 1500, 1500")
        self._send_command("PWM:1550,1550,1560,1440,1500,1500")
        self.motor_pwm = [1550, 1550, 1560, 1440, 1500, 1500]

        return 0
 
    def emergency_stop(self) -> bool:
        """
        紧急停止 - 关闭所有稳定器并将所有电机设为中间值
        """
        self.logger.warning("!!! 紧急停止 !!!")
        print("\n!!! 紧急停止 - 关闭稳定器并停止所有电机 !!!\n")

        # 如果Y轴稳定器正在运行，触发绘图
        if self.ystab_enabled:
            self.logger.info("紧急停止触发，保存Y轴数据分析图...")
            # print("[数据分析] 紧急停止，正在保存Y轴数据分析图...")
            threading.Thread(target=self._plot_ystab_data, daemon=True).start()
            self.ystab_enabled = False

        # 如果Pitch轴双环PID正在运行，触发绘图
        if len(self.pitch_cascaded_buffer['timestamp']) >= 10:
            self.logger.info("紧急停止触发，保存Pitch轴双环PID数据分析图...")
            threading.Thread(target=self._plot_pitch_cascaded_data, daemon=True).start()

        # 1. 停止运动控制
        self._send_command("MOTIONSTOP")
        time.sleep(0.1)

        # 2. 关闭Y轴稳定器
        self._send_command("YSTAB:0")
        time.sleep(0.1)

        # 3. 关闭X轴稳定器
        self._send_command("XSTAB:0")
        time.sleep(0.1)

        # 4. 关闭Yaw轴稳定器
        self._send_command("ZSTAB:0")
        time.sleep(0.1)

        # 5. 关闭平滑控制
        self._send_command("SMOOTH:0")

        time.sleep(0.1)
        # 6. 将所有电机设为中间值(停止)
        self._send_command("PWM:1500,1500,1500,1500,1500,1500")

        # 更新本地状态
        self.motor_pwm = [1500] * 6

        return True


def main():

    MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.43.154")
    MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_USERNAME = os.getenv("MQTT_USERNAME", "main")
    MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "sws3009-20")

    # 创建控制器（包含MQTT配置）
    robot = SphericalRobotController(
        mqtt_broker=MQTT_BROKER,
        mqtt_port=MQTT_PORT,
        mqtt_username=MQTT_USERNAME,
        mqtt_password=MQTT_PASSWORD
    )
    
    # 连接到STM32
    if not robot.connect():
        print("连接失败，程序退出")
        return
    
    try:
        print("\n球形机器人控制器已启动")
        print("串口和MQTT双通道通信已建立")
        print(f"MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
        print(f"MQTT话题:")
        print(f"  PWM命令: {robot.mqtt_topics['pwm_cmd']}")
        print(f"  PID命令: {robot.mqtt_topics['pid_cmd']}")
        print(f"  IMU命令: {robot.mqtt_topics['imu_cmd']}")
        print(f"  状态命令: {robot.mqtt_topics['status_cmd']}")
        print(f"  IMU数据: {robot.mqtt_topics['imu_data']}")
        print(f"  状态数据: {robot.mqtt_topics['status_data']}")
        print(f"  调试信息: {robot.mqtt_topics['debug']}")
        print("\n串口命令:")
        print("  pwm <m1> <m2> <m3> <m4> <m5> <m6> - 设置电机PWM值")
        print("  pid <kp> <ki> <kd>                 - 设置PID参数")
        print("  imu                                - 请求IMU数据")
        print("  status                             - 请求系统状态")
        print("  smooth <1/0>                       - 启用/禁用PWM平滑控制")
        print("  ystab <1/0>                        - 启用/禁用Y轴(Roll)稳定器")
        print("  xstab <1/0>                        - 启用/禁用X轴(Pitch)双环PID")
        print("  zstab <1/0>                        - 启用/禁用Yaw轴双环PID")
        print("  target <axis> <angle>              - 设置轴目标角度(如: target Roll 30)")
        print("  gettarget <axis>                   - 查询轴目标角度(如: gettarget Yaw)")
        print("  motion <vx> <vy>                   - 设置运动速度(vx,vy范围:-1.0~1.0)")
        print("  motionstop                         - 停止运动(速度归零)")
        print("  stop                               - 紧急停止(关闭稳定器+停止电机)")
        print("  plot                               - 手动生成Roll轴数据分析图并清空缓冲区")
        print("  xplot                              - 手动生成Pitch轴双环PID数据分析图")
        print("  zplot                              - 手动生成Yaw轴双环PID数据分析图")
        print("  test                               - 运行测试序列")
        print("  mqtt_test                          - 测试MQTT功能")
        print("  quit                               - 退出程序")
        print("\n数据采集:")
        print("  - 稳定器运行时自动收集数据(最多10000个数据点)")
        print("  - 稳定器停止时自动生成性能分析图并保存到 plots/ 目录")
        print("  - 使用 plot/xplot/zplot 命令手动绘图后会清空数据缓冲区")
        print("  - 每次手动绘图只显示上次清空后的新数据，方便观察单次控制效果")
        print("\nMQTT命令示例:")
        print("  PWM控制: mosquitto_pub -t 'spherical_robot/cmd/pwm' -m '[1500,1500,1500,1500,1500,1500]'")
        print("  PID设置: mosquitto_pub -t 'spherical_robot/cmd/pid' -m '[1.0,0.1,0.05]'")
        print("  IMU请求: mosquitto_pub -t 'spherical_robot/cmd/imu' -m 'get'")
        print("  状态请求: mosquitto_pub -t 'spherical_robot/cmd/status' -m 'get'")
        print("  平滑控制: mosquitto_pub -t 'spherical_robot/cmd/smooth' -m '1'")
        print("  Y轴稳定: mosquitto_pub -t 'spherical_robot/cmd/ystab' -m '1'")
        print("  X轴稳定: mosquitto_pub -t 'spherical_robot/cmd/xstab' -m '1'")
        print()
        
        while True:
            try:
                cmd = input(">>> ").strip().split()
                
                if not cmd:
                    continue
                
                if cmd[0] == 'quit':
                    break
                if cmd[0] == 'start':
                    robot.motor_start()
                elif cmd[0] == 'pwm' and len(cmd) == 7:
                    pwm_values = [int(x) for x in cmd[1:]]
                    robot.set_motor_pwm(pwm_values)
                elif cmd[0] == 'pid' and len(cmd) == 4:
                    kp, ki, kd = float(cmd[1]), float(cmd[2]), float(cmd[3])
                    robot.set_pid_params(kp, ki, kd)
                elif cmd[0] == 'imu':
                    robot.request_imu_data()
                elif cmd[0] == 'status':
                    robot.request_status()
                elif cmd[0] == 'smooth' and len(cmd) == 2:
                    enable = int(cmd[1])
                    robot.set_smooth_mode(enable)
                    print(f"PWM平滑控制已{'启用' if enable else '禁用'}")
                elif cmd[0] == 'ystab' and len(cmd) == 2:
                    enable = int(cmd[1])
                    robot.set_ystab_mode(enable)
                    print(f"Y轴稳定器已{'启用' if enable else '禁用'}")
                elif cmd[0] == 'xstab' and len(cmd) == 2:
                    enable = int(cmd[1])
                    robot.set_xstab_mode(enable)
                    print(f"X轴稳定器已{'启用' if enable else '禁用'}")
                elif cmd[0] == 'zstab' and len(cmd) == 2:
                    enable = int(cmd[1])
                    robot.set_zstab_mode(enable)
                    print(f"Yaw轴稳定器已{'启用' if enable else '禁用'}")
                elif cmd[0] == 'target' and len(cmd) == 3:
                    axis_name = cmd[1]
                    angle_deg = float(cmd[2])
                    robot.set_target_angle(axis_name, angle_deg)
                    print(f"设置{axis_name}轴目标角度为{angle_deg:.2f}度")
                elif cmd[0] == 'gettarget' and len(cmd) == 2:
                    axis_name = cmd[1]
                    robot.get_target_angle(axis_name)
                    print(f"查询{axis_name}轴目标角度...")
                elif cmd[0] == 'motion' and len(cmd) == 3:
                    vx, vy = float(cmd[1]), float(cmd[2])
                    robot.set_motion_velocity(vx, vy)
                    print(f"运动速度设置: vx={vx:.2f}, vy={vy:.2f}")
                elif cmd[0] == 'motionstop':
                    robot.motion_stop()
                    print("运动已停止")
                elif cmd[0] == 'stop':
                    robot.emergency_stop()
                elif cmd[0] == 'plot':
                    print("[绘图] 正在生成Y轴稳定器数据分析图并清空缓冲区...")
                    threading.Thread(target=robot._plot_and_clear_ystab_data, daemon=True).start()
                elif cmd[0] == 'xplot':
                    print("[绘图] 正在生成Pitch轴双环PID数据分析图并清空缓冲区...")
                    threading.Thread(target=robot._plot_and_clear_pitch_cascaded_data, daemon=True).start()
                elif cmd[0] == 'zplot':
                    print("[绘图] 正在生成Yaw轴稳定器数据分析图并清空缓冲区...")
                    threading.Thread(target=robot._plot_and_clear_zstab_data, daemon=True).start()
                elif cmd[0] == 'test':
                    # 运行测试序列
                    print("运行电机测试序列...")
                    test_sequences = [
                        [1500, 1500, 1500, 1500, 1500, 1500],  # 中间位置
                        [1600, 1400, 1500, 1500, 1500, 1500],  # 测试电机1,2
                        [1500, 1500, 1600, 1400, 1500, 1500],  # 测试电机3,4
                        [1500, 1500, 1500, 1500, 1600, 1400],  # 测试电机5,6
                        [1500, 1500, 1500, 1500, 1500, 1500],  # 回到中间位置
                    ]
                    
                    for i, pwm_vals in enumerate(test_sequences):
                        print(f"测试步骤 {i+1}: PWM = {pwm_vals}")
                        robot.set_motor_pwm(pwm_vals)
                        time.sleep(2)
                        
                elif cmd[0] == 'mqtt_test':
                    # 测试MQTT功能
                    if robot.mqtt_connected:
                        print("MQTT连接正常, 发送测试消息...")
                        
                        # 模拟MQTT命令
                        print("模拟PWM命令...")
                        robot._process_pwm_command('[1500,1500,1500,1500,1500,1500]')
                        time.sleep(1)
                        
                        print("模拟PID命令...")
                        robot._process_pid_command('[1.2,0.15,0.08]')
                        time.sleep(1)
                        
                        print("模拟IMU请求...")
                        robot._process_imu_command('get')
                        time.sleep(1)
                        
                        print("模拟状态请求...")
                        robot._process_status_command('get')
                    else:
                        print("MQTT未连接，无法测试")
                else:
                    print("无效命令，请重新输入")
                    
            except ValueError:
                print("参数错误，请检查输入格式")
            except KeyboardInterrupt:
                break
    
    finally:
        robot.disconnect()
        print("程序已退出")


if __name__ == "__main__":
    main()