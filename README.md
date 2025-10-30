# FlexLoad项目介绍
## 项目结构

FlexLoad项目主要包含两个核心模块：资源利用率预测模块和任务调度模块。整体结构如下：

```
FlexLoad-Code/
├── models/                  # 资源利用率预测模型
│   ├── GWNet.py             # 图卷积网络预测模型
│   ├── Informer.py          # Transformer架构的预测模型
│   ├── LSTM.py              # LSTM预测模型
│   └── Informer_assets/     # Informer模型的一些需要的依赖函数
├── scheduling_util/         # 任务调度相关代码
│   ├── scheduler/           
│   │   ├── DQN.py           # 基于DQN的调度算法
│   │   ├── KaiS.py          # 基于KaiS的调度算法
│   │   ├── UCB.py           # 基于UCB调度算法
│   │   └── generic.py       # 基于遗传算法的调度算法
│   └── env/                 # 整个实验pipeline的环境和平台模拟
│       ├── env_run.py       # 环境运行逻辑
│       └── platform.py      # 平台实体类定义
├── data/                    # 数据集文件
├── *.ipynb                  # 数据处理和模型训练的Jupyter Notebook
└── best_*.pth               # 预训练模型文件
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
