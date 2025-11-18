# FlexLoad项目介绍
## 项目结构

FlexLoad项目主要包含两个核心模块：资源利用率预测模块和任务调度模块。整体结构如下：

```
FlexLoad-Code/
├── src/
│   └── flexload/
│       ├── core/          # 仿真主循环与状态构造
│       ├── env/           # 平台与环境
│       ├── scheduler/     # 调度算法与选择器
│       └── models/        # 预测模型（Informer/GWNet/LSTM）
├── scripts/               # CLI 入口
├── configs/               # 配置文件
├── data/                  # 数据集
├── results/               # 仿真结果
├── outputs/               # 指标汇总
├── logs/                  # 日志
└── *.ipynb                # 旧 Notebook（保留作为参考，不参与运行）
```

## 核心模块详解

### 1. 资源利用率预测模块

主要包含三种不同的深度学习模型：

GWNet.py : 基于图卷积网络

Informer.py : 基于Transformer架构的高效时间序列预测模型，专为长序列预测设计，需要输入realtime的日期信息 

LSTM.py : 经典LSTM

### 2. 任务调度模块

包含四种不同的调度算法：

DQN.py 基于DQN的强化学习调度算法

KaiS.py 论文KaiS的复现算法

UCB.py 基于UCB的调度算法

generic.py 基于遗传算法的调度算法

### 3. 环境与平台模拟

platform.py 定义了边缘计算平台的核心实体类：
- `Cloud`：云服务器实体，拥有无限资源
- `Node`：边缘节点实体，具有有限的CPU和内存资源
- `Master`：主控节点，负责任务分发和管理
- `Docker`：容器实例，用于运行具体任务

env_run.py 实现了环境的运行逻辑，包括：
- 任务加载和预处理
- 任务队列管理
- 任务执行和状态更新
- 资源使用情况跟踪

## 4. 数据处理与可视化

- `draw_data.ipynb`：读取任务资源利用率随时间变化的数据集
- `process_ver1 copy.ipynb`：实验结果画图，主要是在调度里的变化
- `SingleFeatureTraining.ipynb`：训练预测模型用的脚本，里面不同模型的训练代码都写在相邻的cell，可以看每个cell的开头的模型初始化确定是哪个模型
- `TraceSink.ipynb`：老脚本，功能已经被切分了
- `SeeDataset.ipynb`：画预测训练结果的实验图
- `scheduling.ipynb`：完整的调度实验逻辑，`get_act`是选择调度算法的入口


## 纯 Python 工程与 CLI 使用指南（新增）

本项目已新增纯 Python 工程化实现，支持命令行运行完整仿真流程，无需依赖 Jupyter。

- 代码结构：
  - `src/flexload/core/simulator.py`：仿真主循环封装（替代 Notebook 的 execution）。
  - `src/flexload/core/state.py`：统一状态构造与转换（含 KaiS 所需 88 维向量）。
  - `src/flexload/scheduler/selector.py`：调度算法统一入口（KaiS/UCB/DQN/generic/Random/UCB_predict）。
  - `src/flexload/metrics/usage.py`：资源使用记录与持久化（替代 globals）。
  - `src/flexload/utils/config.py`：配置加载（YAML/JSON），含默认值。
  - `src/flexload/utils/logging.py`：日志初始化与目录创建（logs/results/outputs）。
  - `src/flexload/utils/seeding.py`：随机种子固定（random/numpy/torch）。
  - `scripts/run.py`：CLI 入口。

- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```

- 运行示例（冒烟）：
  ```bash
  python scripts/run.py --config configs/default.yaml --model KaiS --run_times 1 --break_point 200 --cho_cycle 50
  ```

- 产物说明：
  - `results/node_resource_usage_<MODEL>.json`：各 slot 的节点资源使用记录。
  - `outputs/summary.json`：一次或多次 run 的汇总指标（吞吐、响应率、奖励）。
  - `logs/run.log`：运行日志。

- 配置说明：
  - 默认配置位于 `configs/default.yaml`，可覆盖：
    - 设备选择（cpu/cuda）
    - 数据路径（`data_task_1`、`data_task_2`）。当 `num_masters>2` 时，多余的 master 将在这两份数据之间轮换（偶数索引用 Task_1，奇数索引用 Task_2）。
    - 可配置规模：`num_masters`（默认 2）、`nodes_per_master`（默认 3）。`action_dim` 将在代码中按 `num_masters*nodes_per_master + 1`（cloud）动态计算。
    - 是否启用 `informer`、`UCB_predict` 的预测类型等。
  - UCB_predict + Informer 使用说明：`informer.input_size` 为每步输入向量维度；每个时间步将所有 master 的所有节点 CPU 利用率（cpu/cpu_max，按全局节点顺序）拼接为向量，不足零填充、超出截断；推理采用 `torch.no_grad()` 包裹，预测到的下一步利用率会映射回各节点生成绝对 CPU 值 `[pred*cpu_max, cpu_max]` 并用于 UCB 决策。

- 已修复的已知问题：
  - `s_grid` 第二组状态使用 `cpu_list2/mem_list2`（修复 notebook 复用错误）。
  - `KaiS` 的状态维度统一转换为 88 维向量，避免维度错误。
  - 写文件前确保目录存在（`results/`、`outputs/`、`logs/`）。
  - 移除 `globals()` 的状态共享，改用对象记录与传参。

## 指标与输出（新增）

本工程在仿真管线中集成了两项关键指标，用于完整评估调度策略：

- **QoS 违约率（QVR）**
  - **时间视角 QVR_time_rate**：在每个仿真 slot 上，逐节点检测是否存在队列中的任务发生违约；违约条件参考 `env.update_docker` 的语义：
    1) 若现在开始执行会超时：`cur_time + (task_cpu / docker_cpu) > end_time`；
    2) 资源不足：`node.cpu < POD_CPU×service_coefficient[kind]` 或 `node.mem < POD_MEM×service_coefficient[kind]`。
    每个 slot 的违约时间占比 = 发生违约的节点数 / 总节点数；总体为所有 slot 的平均占比（等价于累计违约节点数 / (总节点数 × 总 slot 数)）。
  - **任务视角 QVR_task_rate**：在所有 run 结束后，以总未完成任务数 `undone` 除以（完成+未完成）任务数，作为粗粒度违约率。

- **调度决策开销（Scheduling Cost/Latency）**
  - 在每个 slot 的动作选择阶段，记录 `select_action` 的壁钟时间（毫秒），并汇总平均值（`avg_ms`）与 P95（`p95_ms`）。
  - 额外记录候选集大小 `|A|`（每个任务的 `ava_node` 长度之和）的平均值（`candidate_avg_size`），便于分析规模对延迟的影响。

### 网络维度与输入（新增）

为支持网络带宽与丢包率（PLR）维度，项目新增了网络 trace 的接入与 QoS 扩展：

- Trace 文件格式（CSV）支持两种形式：
  1) 窄表：每行 `time_slot,node_id,bw_mbps,plr`。
  2) 宽表：第一列为 `time_slot`，后续列按 `node_0_bw,node_0_plr,node_1_bw,node_1_plr,...` 组织（节点按全局索引 0..N-1）。
- 字段含义：
  - `bw_mbps`：节点该时间步的有效网络带宽（Mbps）。
  - `plr`：节点该时间步的丢包率（0~1）。
- 配置项（`utils/config.py` 中 DEFAULTS，或通过 YAML/JSON 覆盖）：
  - `network.enabled`：是否启用网络维度，默认 `True`。
  - `network.trace_path`：trace 文件路径，默认 `trace.csv`（可通过 CLI `--network_trace` 覆盖）。
  - `network.bw_demand_per_kind`：每种任务类型的带宽需求（Mbps），默认按 `MAX_TASK_TYPE` 填充为 `1.0`。
  - `network.plr_threshold_per_kind`：每种任务类型的最大可接受丢包率，默认按 `MAX_TASK_TYPE` 填充为 `0.05`。
- QoS 扩展逻辑：
  - 若 `node.net_bw_mbps < bw_need[kind]` 或 `node.plr > plr_threshold[kind]`，则该节点在当前 slot 视为违约；与原有资源/超时条件取“或”。
  - 额外提供 `qvr_network_only_rate`（仅网络维度违约率），便于隔离分析网络影响。

### 输出说明

- `outputs/summary.json` 增加字段：
  - `qvr_time_rate`, `qvr_task_rate`, `qvr_network_only_rate`,
  - `decision_latency_avg_ms`, `decision_latency_p95_ms`, `decision_latency_count`。
- `outputs/metrics.json`（新增）：集中存放上述指标的详细汇总结构，便于脚本或可视化工具直接消费。

### 示例使用（冒烟）

```bash
# 若工作区存在 trace.csv
python scripts/run.py --config configs/default.yaml --model UCB_predict --network_trace trace.csv --run_times 1 --break_point 60 --cho_cycle 20

# 若无 trace.csv（将自动禁用网络维度）
python scripts/run.py --config configs/default.yaml --model UCB_predict --run_times 1 --break_point 60 --cho_cycle 20
```

运行后，你可以在 `logs/run.log` 中看到吞吐、完成/失败、QVR（含网络维度）与决策开销的汇总日志；在 `outputs` 目录下看到 `summary.json` 与 `metrics.json`（包含 `qvr_network_only_rate`）。
