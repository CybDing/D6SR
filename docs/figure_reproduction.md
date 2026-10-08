# 论文插图复现说明

记录论文 *Design and Control of D6SR* 中图 5–9 的生成方式：用哪个脚本、哪个 checkpoint、期望得到什么数值。所有仿真图的随机种子均已固定，因此重跑应当逐位复现，下表的「校验数值」就是用来确认这一点的——跑完对不上，说明参数或 checkpoint 用错了。

## 通用前提

- 全部命令从 `simulation/mujoco_sim/` 目录执行（相对路径依赖此工作目录）。
- 无需 `mjpython`：这些都是 headless 出图，用 `python` 即可（`mjpython` 只在需要交互式 viewer 时才必要）。
- 物理参数来自 `config/robot_params.yaml`，PID 增益来自 `config/tuned_gains_calibrated.json`（脚本默认值），均无需手动指定。
- 图 5/6/7 的排版（字体、子图标签）统一由 `simulation/mujoco_sim/paper_style.py` 提供，不要在各脚本里另写 rcParams。

## 图 5–8

| 图 | 命令（在 `simulation/mujoco_sim/` 下） | 输出 | 校验数值 |
|---|---|---|---|
| 5 | `python analysis/fig_circle.py --policy rl/checkpoints/calib_v7b_nojoint_fs4/best.pt` | `docs/media/fig5_circle_calibrated.pdf` | (a) mean err 24.35 m；(b) 0.55 m |
| 6 | `python analysis/fig6_ablation.py --policy rl/checkpoints/calib_v7b_nojoint_fs4/best.pt` | `docs/media/fig6_terrain_ablation_calibrated.pdf` | tier means 36.48 / 0.48 / 0.26 m |
| 7 | `python analysis/fig_paper_yaw.py --policy rl/checkpoints/calib_v7b_nojoint_fs4/best.pt` | `docs/media/fig7_yaw_evolution_calibrated.pdf` | (a) std 2103.3°；(b) std 50.0°；(c) std 3.6°, max 52.7° |
| 8 | `python benchmark_baseline.py --no-rl` | 见下方「图 8 的额外拷贝步骤」 | V0 travel +10.96 m；V1 travel +14.79 m |

### 图 7 的 policy 选择（重要）

图 5、6、7 **统一使用 `calib_v7b_nojoint_fs4`**。

早期提交的图 7 用的是 `calib_v6_fs4`，(c) 档得到 std = 4.3°、max = 44.4°；换成与图 5/6 相同的 `calib_v7b_nojoint_fs4` 后为 **std = 3.6°、max = 52.7°**。三张图现已统一到同一个 policy，论文正文中引用图 7 的数值需与此一致。

注意 (a)、(b) 两档不含 RL，与 policy 无关，无论用哪个 checkpoint 都复现为 std 2103.3° / 50.0°。**因此 policy 用错只会在 (c) 档暴露**，(a)(b) 对得上并不能说明命令正确。

### 图 8 的额外拷贝步骤

`benchmark_baseline.py` 把 PDF 写到脚本自身目录，而 `.gitignore` 忽略 `*.pdf`、只放行 `docs/media/**`，所以纳入版本管理的是 `docs/media/` 下那一份，需手动拷贝：

```bash
python benchmark_baseline.py --no-rl
cp benchmark_motor_allocation.pdf ../../docs/media/benchmark_motor_allocation.pdf
```

`--no-rl` 决定了图中只有 (a)、(b) 两个面板；去掉该参数会额外生成 PID-only 与 PID+RL 面板。

## 图 9（硬件）——不可重跑

图 9 是实机 yaw 阶跃响应，不是仿真结果，仓库中没有生成它的脚本。

**原始树莓派日志从未落盘**：`raspberry_pi/main.py` 的 `_plot_yaw_cascaded()` 只调用 `plt.savefig` 输出 PNG 到 `raspberry_pi/plots/`（该目录只存在于树莓派上），`data_dict` 中的数组从未序列化，git 也从未跟踪过。因此不存在可供找回的日志，重跑也需要实体机器人。

现存物：

- `docs/media/fig_yaw.pdf` —— 图 9 本体（矢量 PDF，投稿时所用）。
- `data/fig9_yaw_hardware.csv` —— 从该 PDF 中还原出的数据，156 行 `time_s, measured_yaw_deg, reference_yaw_deg`。
- `tools/extract_fig_yaw_data.py` —— 还原脚本，可重跑复核：

```bash
python tools/extract_fig_yaw_data.py
```

还原是**精确复原而非估读**：坐标轴刻度给出闭式线性映射（残差 ~1e-7 pt），并有两个独立校验位——参考阶跃还原为精确整数（0 → 70 → 130 → 120 度），156 个测量点全部落在 0.1° 的 IMU 网格上（偏差 ~1e-5）。数值已 snap 回上述网格以消除 PDF 六位小数的往返误差。

两点需知：

- 参考信号在 t ≈ 72–86 s 有一档 **130°**，在原图中被图例框完全遮挡，肉眼不可见；矢量还原把它取了出来。实际是四档而非三档。
- 时间戳为不规则墙钟时间（中位 0.51 s），且 matplotlib 的 path-simplification 在当年渲染时丢弃了约 2% 的近似共线采样点——这部分 PDF 里根本没有，无法恢复。故该 CSV 是**图上所绘的曲线**，而非完整原始日志。

## 控制器与训练配置（对照论文 Table I / II）

### 仿真平台（论文 Table I）

| 论文参数 | 值 | 代码位置 |
|---|---|---|
| MuJoCo timestep | 0.001 s | `config/robot_params.yaml` |
| 控制频率（PID / RL） | 50 Hz | `SphericalRobotEnv(pid_rate=50)`，每个控制周期 20 个物理步 |
| 滚动摩擦 μ_roll | 0.01 | `SphericalRobotEnv(mu_roll=0.01)` |
| 单桨最大推力 | 14.7 N | `config/realistic_config.json` → `force_max` |
| 总质量 | 2.78 kg | `realistic_config.json` 的 `mass_overrides`（仿真内合计 2.785 kg） |
| 轴承摩擦 | 0.01 N·m | `realistic_config.json` → `joint_frictionloss`（另有 `joint_damping` 0.002） |
| 姿态 / 陀螺噪声 | 0.5° / 0.02 rad/s | realism pack，`--realism` |
| 控制回路延迟 | 20 ms（一个周期） | realism pack，`imu_delay_steps=1` |
| 推力一阶滞后 | 80 ms | realism pack，`thrust_tau=0.08` |

### PID 整定

`config/tuned_gains_calibrated.json` 是在标定后的 plant 上、开启 realism、r = 0.2 地形下用坐标下降整定得到的增益，图 5–7 与 RL 训练都使用它：

```bash
python analysis/tune_pid.py --realism --roughness 0.2 --out config/tuned_gains_calibrated.json
```

### 残差 RL 训练（论文 Table II）

论文所用 policy 为 `rl/checkpoints/calib_v7b_nojoint_fs4/best.pt`（仓库中仅保留此文件）。训练命令：

```bash
python rl/train_residual.py --unidirectional --no-joint-obs \
  --sim-config-json config/realistic_config.json \
  --gains-json config/tuned_gains_calibrated.json \
  --ckpt-subdir calib_v7b_nojoint_fs4 \
  --yaw-reward 4 --act-penalty 1 --smooth-penalty 0.5 \
  --target-entropy -6 --init-alpha 0.05 --alpha-max 0.05 \
  --realism --frame-stack 4 --total-steps 400000
```

目标速度按论文在每个 episode 开始时从 U[-1, 1] m/s 随机采样（不加 `--fixed-drive`）。

| 论文参数 | 值 | 对应参数 / 默认值 |
|---|---|---|
| 总训练步数 | 400,000 | `--total-steps 400000` |
| Episode 长度 | 500 | `--episode-steps`（默认 500） |
| Replay buffer | 500,000 | `--buffer-size`（默认） |
| Batch size | 128 | `--batch-size`（默认） |
| "Target update interval" 2 | 2 | 实为 `--utd-ratio 2`：每个环境步做 2 次 critic 更新，每次 critic 更新后都对 target 做一次 τ = 0.005 的软更新；actor 与熵系数 α 每步只更新 1 次 |
| Actor / Critic / Entropy 学习率 | 3e-4 / 1e-4 / 3e-4 | `--lr-actor` / `--lr-critic`（默认），`lr_alpha` 写死为 3e-4 |
| γ / τ | 0.99 / 0.005 | `rl/sac.py` 默认值 |
| 隐层 | (128, 128) | `train_residual.py` 写死 |
| 观测维度 | 11 × 4 | `--no-joint-obs --frame-stack 4`（obs_dim = 44） |
| Target entropy | −6 | `--target-entropy -6` |
| 残差比例 β | 论文未给数值 | `--alpha-max 0.05` |
| 奖励系数 | 4 / 0.5 / 1.0 / 0.5 / +0.5,−10 | roll·pitch 的 4 写死在 `env.py`，yaw 的 4 来自 `--yaw-reward 4`；`--act-penalty 1`、`--smooth-penalty 0.5` |
| Warm-up | 前 5,000 步零残差 | `--warmup-random 5000`（默认） |
| 域随机化 | 关节初值 ±0.3 rad，μ_roll ×[0.5, 1.5]，p = 0.5 时 r ~ U[0.05, 0.2] | `env.py` 中 `randomize=True`；`--terrain-prob 0.5 --roughness-min 0.05 --roughness-max 0.2`（默认） |

`best.pt` 是训练过程中定期评估得分最高的快照（训练步 165,001），不是第 400k 步的最终权重。

### 代码实现与论文写法的差异

- **姿态环的 T_b(θ) 变换**：论文公式中外环输出的欧拉角速率经 T_b(θ) 变换为体坐标系角速度后再进入内环。图 5–7 与 RL 训练实际使用的是三轴独立的串级 PID（`BaselineController(euler_rate_transform=False)`，默认值），即小姿态角下 T_b ≈ I 的近似。带变换的实现已提供，可通过 `euler_rate_transform=True` / `control/run_baseline.py --euler-rate-transform` 开启，但开启后增益需要重新整定，论文图也不再逐位复现。
- **单向电机混控**：训练与图 5–7 使用 `unidirectional=True` 的混控（`control/run_baseline.py::mix_motors`）：每对电机产生姿态力矩时只增加一侧电机的推力，而不是像论文闭式分配 A(f̃, τ̃) 那样让两侧对称地一正一反。这是工程上的限制，用来避免为了姿态微调而频繁反转电机，真机也采用同样的做法。图 8（`benchmark_baseline.py`）使用的是论文中的双向对称分配。
