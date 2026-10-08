<div align="center">

[English](README.md) | **简体中文**

# D6SR：机械解耦的六轴球形滚动机器人

**Design and Control of D6SR: A Mechanically Decoupled Six-Axis Spherical Rolling Robot for Stable Propeller-Driven Rolling**<br>
IROS 2026

Leqi Ding\*, Xijian Deng\*, Jiayi Chen, Jin Yang, Ping Wei<br>
西安交通大学 · \*共同第一作者

论文：即将公开 · [许可证：Apache-2.0](LICENSE)

<img src="docs/media/readme/real_robot.jpg" width="38%" alt="D6SR 样机">

</div>

D6SR 是一种螺旋桨驱动的球形机器人。六个可正反转的螺旋桨安装在内部机体上，内部机体通过三自由度万向节与被动滚动的外壳解耦。外壳滚动时内部机体保持水平，各推力方向在世界坐标系中保持不变，电机无需频繁换向。内部机体的姿态由串级 PID 控制；在此基础上，仅以 IMU 数据为输入、在仿真中训练的残差强化学习策略在粗糙地形上补充修正力矩。

<p align="center">
  <img src="docs/media/readme/real_rolling.jpg" width="95%" alt="实机滚动实验"><br>
  <em>实机滚动实验：机器人保持直线行进（蓝色为参考方向，黄色为实际轨迹）。</em>
</p>

## 目录

- [亮点](#亮点)
- [硬件](#硬件)
- [方法](#方法)
- [实验结果](#实验结果)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [复现论文结果](#复现论文结果)
- [串口协议](#串口协议)
- [引用](#引用)
- [许可证](#许可证)

## 亮点

- **机械解耦**：三自由度万向节将外壳转动与推进模块隔离，滚动过程中推力方向保持不变。
- **三正交六桨推进**：三对正交布置的可逆螺旋桨可产生任意机体力旋量，无需额外的地面驱动电机。
- **串级 PID + 残差强化学习**：Soft Actor-Critic 策略只在 PID 基线之上输出残差力矩。
- **标定后的仿真**：MuJoCo 模型按样机标定（单桨推力、轴承摩擦、传感器噪声、控制延迟、推力滞后）。仓库附带训练好的策略，以及重新生成论文中全部仿真图的脚本。

## 硬件

<p align="center">
  <img src="docs/media/readme/decoupling.jpg" width="95%" alt="解耦原理"><br>
  <em>(a) 无解耦时内部机体随外壳转动；(b) 有万向节时外壳滚动而内部机体静止；(c) 因此推力 F<sub>1</sub>、F<sub>2</sub> 在世界坐标系中保持不变，驱动机器人以速度 v 前进。</em>
</p>

| 项目 | 规格 |
|---|---|
| 外径 / 总质量 | 460 mm / 2.78 kg |
| 推进 | 6 个无刷电机 + 双向电调，单桨最大推力 14.7 N |
| 主控 | STM32F407VET6（libopencm3 + FreeRTOS），串级 PID 100 Hz |
| 上位机 | 树莓派 4B（串口桥接、日志、MQTT） |
| 姿态传感器 | IMU，UART，200 Hz |
| 通信 | UART 115200 bps，8N1 |

<details>
<summary>机械设计</summary>
<p align="center">
  <img src="docs/media/readme/mechanical_design.jpg" width="60%" alt="机械设计"><br>
  <em>(a) 外壳与万向节内部模块的爆炸图；(b) 集成六执行器推进与控制平台。</em>
</p>
</details>

## 方法

**姿态控制**：内部机体的 roll、pitch、yaw 各由一个串级 PID 控制，外环由角度误差给出角速度目标，内环由角速度误差给出力矩。固件（`stm32/cascaded_pid.c`）与仿真（`simulation/mujoco_sim/control/cascaded_pid.py`）实现的是同一个控制器。

**运动控制**：期望的平面加速度先按 IMU 测得的 yaw 角从世界坐标系转换到机体坐标系，再由速度 PID 跟踪。

**推力分配**：M0/M1、M2/M3、M4/M5 分别构成 yaw、roll、pitch 电机对，期望力旋量映射为带符号的归一化推力并限幅到 [-1, 1]。代码和实机中，每对电机产生姿态力矩时只增加其中一个电机的推力，而不是让两个电机一正一反。这是工程上的处理，用来避免为了微调姿态而让电机换向。

**残差强化学习**：策略在 PID 输出之上叠加一个有界的残差力矩。单帧观测为 11 维（roll、pitch、yaw 的正余弦，机体角速度，速度跟踪误差），策略输入为最近 4 帧的堆叠。训练使用 SAC，共 40 万步，并对万向节初始角、滚动摩擦、目标速度和地形粗糙度（≤ 0.2）做域随机化。完整超参数见 [docs/figure_reproduction.md](docs/figure_reproduction.md)。

## 实验结果

以下仿真图均由本仓库脚本和发布的策略生成。PID 增益的整定和策略的训练都只在粗糙度 ≤ 0.2 的地形上进行。

| 任务 | 无姿态控制 | 串级 PID | PID + 残差 RL |
|---|---|---|---|
| 圆轨迹，R = 10 m，0.5 m/s，粗糙度 0.2 | 翻滚，平均误差 24.35 m | — | **0.55 m，无翻滚** |
| 直线，1.0 m/s，粗糙度 0.3：5 次中翻滚次数 | 5 / 5 | 3 / 5 | **0 / 5** |

<table>
  <tr>
    <td align="center" width="50%"><img src="docs/media/readme/fig5_circle_calibrated.png" alt="圆轨迹跟踪"><br><em>粗糙度 0.2 地形上的圆轨迹滚动。</em></td>
    <td align="center" width="50%"><img src="docs/media/readme/fig6_terrain_ablation_calibrated.png" alt="直线地形测试"><br><em>粗糙度 0.3 地形上的直线滚动，每种控制器 5 次，× 表示翻滚。</em></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/media/readme/fig7_yaw_evolution_calibrated.png" alt="yaw 变化"><br><em>粗糙度 0.3 地形上的内部机体 yaw（4 次运行的均值 ± 1σ）。</em></td>
    <td align="center"><img src="docs/media/readme/benchmark_motor_allocation.png" alt="电机分配"><br><em>v<sub>x</sub> = 1.5 m/s 时的电机指令：(a) 电机固连外壳的方案（Rollocopter，Sabet 等）需要不断重新分配推力；(b) D6SR 的推力几乎恒定。</em></td>
  </tr>
</table>

<p align="center">
  <img src="docs/media/readme/fig_yaw.png" width="50%" alt="实机 yaw 阶跃响应"><br>
  <em>实机：滚动中 yaw 在阶跃参考指令下的稳定控制。</em>
</p>

## 仓库结构

```
D6SR/
├── stm32/                     # STM32F407 固件：FreeRTOS 任务、串级 PID、PWM、IMU、串口
├── raspberry_pi/              # 上位机：串口桥接、MQTT、日志与绘图
├── simulation/
│   ├── mujoco_sim/
│   │   ├── simulation.py      # RobotSimulation：建模、接触、滚动摩擦、IMU
│   │   ├── config/            # robot_params.yaml、标定后的模型参数、整定好的 PID 增益
│   │   ├── control/           # 串级 PID、推力分配、基线控制器、参考轨迹
│   │   ├── rl/                # 环境、SAC、残差训练；发布的策略在 rl/checkpoints/
│   │   ├── analysis/          # 论文图 5–7 的脚本与 PID 整定
│   │   ├── benchmark_*.py     # 基准脚本（benchmark_baseline.py 生成图 8）
│   │   └── video/             # 场景录制
│   ├── robot_v0/, robot_v1/   # URDF 与网格（v0：电机固连外壳的对照模型，v1：D6SR）
├── docs/                      # figure_reproduction.md 与插图
├── data/                      # 图 9 的实机 yaw 数据
├── tools/                     # yaw PID 调参与插图数据工具
├── libopencm3/                # 子模块
└── rtos/FreeRTOS-Kernel/      # 子模块
```

## 快速开始

### 仿真

Python ≥ 3.9：

```bash
git clone --recursive https://github.com/CybDing/D6SR.git
cd D6SR/simulation/mujoco_sim
pip install mujoco numpy scipy matplotlib pyyaml torch

# 基线姿态稳定 + 速度驱动（无界面运行并出图）
python control/run_baseline.py --plot --drive-x 0.5
```

所有脚本都从 `simulation/mujoco_sim/` 目录运行。macOS 上交互式查看需要用 `mjpython`（例如 `mjpython control/run_baseline.py --no-headless`）。物理参数集中在 `config/robot_params.yaml`，修改后用 `python robot_loader.py` 重新生成 MJCF。

### 固件

依赖：`arm-none-eabi-gcc` ≥ 10、`make`、`st-flash`（stlink-tools）。

```bash
git submodule update --init --recursive    # 克隆时未加 --recursive 才需要
make -C libopencm3 TARGETS=stm32/f4         # 只需一次

cd stm32
make                 # 编译
make f407-program    # 全量重新编译并烧录
make monitor         # 串口监视（minicom，/dev/ttyUSB0，115200）
```

引脚、定时器与串口分配见 `stm32/pin_config.h`。

### 上位机（树莓派）

```bash
cd raspberry_pi
pip3 install -r requirements.txt
python3 main.py          # 串口 + MQTT + 日志 + 绘图
python3 simple_test.py   # 连通性测试
```

## 复现论文结果

在 `simulation/mujoco_sim/` 下运行，`P=rl/checkpoints/calib_v7b_nojoint_fs4/best.pt`：

| 图 | 命令 |
|---|---|
| 图 5 圆轨迹 | `python analysis/fig_circle.py --policy $P` |
| 图 6 地形 | `python analysis/fig6_ablation.py --policy $P` |
| 图 7 yaw | `python analysis/fig_paper_yaw.py --policy $P` |
| 图 8 推力分配 | `python benchmark_baseline.py --no-rl` |

随机种子均已固定，重跑会得到与论文完全相同的数值。[docs/figure_reproduction.md](docs/figure_reproduction.md) 列出了每张图的校验数值、PID 整定与强化学习训练命令、论文超参数与代码参数的对照，以及实现与论文表述不同的地方。图 9 是实机实验，其数据从矢量图中还原，存放在 `data/fig9_yaw_hardware.csv`。

## 串口协议

帧格式为 `CMD:DATA1,DATA2,...\n`，UART 115200 bps，8N1。

| 指令 | 格式 | 方向 | 含义 |
|---|---|---|---|
| `PWM` | `PWM:m0,m1,m2,m3,m4,m5` | Pi → STM32 | 设置六路电机脉宽，1000–2000 µs，1500 为中位 |
| `PID` | `PID:kp,ki,kd` | Pi → STM32 | 更新 PID 参数 |
| `IMU` | `IMU` → `IMU:ax,ay,az,gx,gy,gz,roll,pitch,yaw` | 双向 | 请求 / 返回 IMU 数据 |
| `STATUS` | `STATUS` → `STATUS:motor,imu,pid` | 双向 | 子系统状态 |
| `DEBUG` | `DEBUG:message` | STM32 → Pi | 调试信息 |

## 引用

```bibtex
@inproceedings{ding2026d6sr,
  title     = {Design and Control of {D6SR}: A Mechanically Decoupled Six-Axis Spherical Rolling Robot for Stable Propeller-Driven Rolling},
  author    = {Ding, Leqi and Deng, Xijian and Chen, Jiayi and Yang, Jin and Wei, Ping},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## 许可证

本仓库代码以 [Apache License 2.0](LICENSE) 发布。第三方子模块沿用各自的许可证：`libopencm3/`（LGPL-3.0）、`rtos/FreeRTOS-Kernel/`（MIT）。
