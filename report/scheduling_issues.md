# 调度仿真实现问题审查报告（scheduling.ipynb）

## 问题摘要
对 FlexLoad 项目的核心仿真 Notebook（`scheduling.ipynb`）进行系统性审查，发现以下关键问题影响到仿真正确性、可复现性、可配置性与工程化质量：
- 关键状态数据结构与算法期望不匹配，导致 KaiS 调度器在运行时抛出维度错误（ValueError）。
- `s_grid` 的第二组状态错误地复用了 `cpu_list1/mem_list1`，应使用 `cpu_list2/mem_list2`。
- 资源使用记录通过 `globals()` 方式耦合，易导致不可控的隐式状态与副作用。
- 结果文件写入前未创建目录（`result/`），存在运行失败风险；目录命名还与其他路径不一致。
- 缺少随机种子固定与集中配置，硬编码参数较多，导致结果不可复现、不可配置。
- Notebook 中保留大量废弃/冗余代码（TF 残留、双重 `deploy_state` 赋值），影响可读性与维护性。
- 入口与流程组织依赖 Notebook Cell，难以脚本化与工程化运行；`UCB_predict` 的 Informer 输入形状不明、回退逻辑不健壮。

## 详细问题列表

### 1. 状态维度不匹配导致 KaiS 报错
- 症状：运行 KaiS 时抛出异常。
  - 报错信息见 `scheduling.ipynb` 最后一段输出：
    - `ValueError: expected sequence of length 6 at dim 2 (got 1)`（参见 Notebook `In[10]` 的回溯，调用链至 `KaiS.py:262`）。
- 根因：
  - `KaiS.Estimator.action()` 期望输入 `s` 为数值张量，形状约为 `[n_valid_node=2, state_dim=88]`。Notebook 中传入的是嵌套的 `s_grid` 结构：
    ```
    s_grid = [[deploy_state, [task_num1], cpu_list1, mem_list1],
              [deploy_state, [task_num2], cpu_list1, mem_list1]]
    ```
    直接 `torch.tensor(s_grid)` 会因嵌套结构与非统一长度导致张量化失败。
- 修复建议：
  - 在进入 KaiS 算法前将 `s_grid` 显式转换为 `[2, 88]` 数值数组，规则为：
    - 展平 `deploy_state`（6×12→72）+ 任务队列统计 4（总队列 + 3 个节点队列）+ CPU 6（3×[cpu,cpu_max]）+ MEM 6（3×[mem,mem_max]）= 88。
    - 如有偏差进行零填充或截断至 88 维。
  - 已在 `src/flexload/core/state.py::to_kais_state()` 中实现统一转换，并在选择器中对 KaiS 分支自动使用该转换。
- 涉及文件/行段：
  - `scheduling.ipynb`：`get_act('KaiS', s_grid, ...)`（错误调用）；异常回溯显示 `KaiS.py:262`；Notebook 报错发生在 `In[10]`。
  - `scheduling_util/scheduler/KaiS.py:224-315`（Estimator 定义与 `action` 实现）。

### 2. `s_grid` 第二组状态复用错误（CPU/MEM 列表）
- 症状：第二组（对应 master2）的状态仍使用 `cpu_list1/mem_list1`。
- 根因：构造 `s_grid` 时将两组都指向了 master1 的 CPU/MEM 列表：
  - `s_grid = [[..., cpu_list1, mem_list1], [..., cpu_list1, mem_list1]]`（参见 `scheduling.ipynb` 行【417】）。
- 修复建议：
  - 第二组应使用 `cpu_list2/mem_list2`。
  - 已在 `src/flexload/core/state.py::build_s_grid()` 中修复为：
    ```
    [deploy_state, [task_num1], cpu_list1, mem_list1],
    [deploy_state, [task_num2], cpu_list2, mem_list2]
    ```
- 涉及文件/行段：
  - `scheduling.ipynb` 行【417】。

### 3. 全局状态耦合（`node_resource_usage`）
- 症状：Notebook 中通过 `globals()['node_resource_usage'] = node_resource_usage` 暴露历史资源使用，供 `UCB_predict` 使用。
- 根因：使用 `globals()` 在 Notebook 环境中共享状态，破坏模块边界与可测试性，易出现难以追踪的副作用。
- 修复建议：
  - 引入指标记录对象 `ResourceUsageTracker`，通过依赖注入传入选择器（`selector.py`），避免全局耦合。
  - 已实现于 `src/flexload/metrics/usage.py`，并在 `Simulator` 内维护与传参。
- 涉及文件/行段：
  - `scheduling.ipynb` 行【345】-【347】。

### 4. 目录/文件写入健壮性不足
- 症状：结果 JSON 写入到 `./result/node_resource_usage_<MODEL>.json`，未确保目录存在；项目其他输出目录为 `outputs/` 与 `logs/`，存在命名不一致。
- 根因：Notebook 即写即跑，未进行目录存在性检查；且目录命名不统一。
- 修复建议：
  - 统一输出至 `results/`、`outputs/`、`logs/`，并在写文件前调用 `ensure_dir()` 创建目录。
  - 已在 `src/flexload/utils/logging.py::ensure_dir()` 与 `Simulator` 初始化中做统一处理。
- 涉及文件/行段：
  - `scheduling.ipynb` 行【586】-【588】（写 JSON）。

### 5. 可复现性/可配置性不足
- 症状：大量硬编码参数（如 `SLOT_TIME`、`POD_CPU`、`POD_MEM` 等）、缺少随机种子固定；设备选择、数据路径、模型切换等缺少集中配置入口。
- 根因：Notebook 原型代码，以演示为主，未构建配置系统。
- 修复建议：
  - 引入 YAML/JSON 配置加载：`src/flexload/utils/config.py`，提供默认参数与覆盖机制。
  - 引入随机种子固定：`src/flexload/utils/seeding.py::seed_all()` 固定 `random/numpy/torch`，可选启用 cudnn 确定性。
  - CLI 支持 `--config`、`--model`、`--run_times`、`--break_point`、`--train_times`、`--cho_cycle` 等参数。
- 涉及文件/行段：
  - `scheduling.ipynb` 中的硬编码参数定义（`execution()` 冒头处）。

### 6. 冗余/废弃代码
- 症状：大量 TF 相关注释与残留（如 `tf.Session()`、`ReplayMemory` 等）、双重 `deploy_state` 赋值（先给具体矩阵，后又全 1 覆盖）。
- 根因：历史遗留与算法探索残留。
- 修复建议：
  - 工程化时移除 TF 残留，保留必要的注释；统一 `deploy_state` 由配置驱动。
- 涉及文件/行段：
  - `scheduling.ipynb` 行【243】-【258】、【569】-【579】（TF 残留），行【276】-【281】（双重赋值）。

### 7. 入口与流程组织不利于脚本化
- 症状：入口为 Notebook Cell，难以通过命令行统一运行与集成；`tqdm` 等调度输出也难以控制。
- 根因：以 Notebook 为主要开发环境。
- 修复建议：
  - 重构为纯 Python 包与 CLI（见本次改造的 `src/flexload/**` 与 `scripts/run.py`）。
  - 使用标准 `logging` 控制台与文件输出，并保留 `tqdm` 的可选展示（当前以 logger 记录为主）。

### 8. `UCB_predict` 的输入构造与模型使用不清晰
- 症状：Notebook 在 `UCB_predict` 分支中尝试将历史 `node_resource_usage` 整形为 Informer 输入，但实际 `models/Informer.py` 的前向签名与训练时的输入维度未在 Notebook 中定义一致，容易产生形状错误；且回退逻辑仅在异常时退回 UCB。
- 根因：Informer 的简化版实现（`models/Informer.py` 中 `if True` 路径）期望输入形状为 `[batch, seq_len, input_size]`，而 Notebook 构造的是 `[1, seq_len, num_nodes, 1]`；两者不兼容。
- 修复建议：
  - 保留 Informer 的加载逻辑（可在配置中启用/关闭），但默认 `UCB_predict` 采用简单的历史均值预测 CPU 利用率；Informer 预测在未来补齐明确的输入构造后再启用。
  - 已在 `src/flexload/scheduler/selector.py` 的 `UCB_predict` 分支中实现均值预测与健壮回退。
- 涉及文件/行段：
  - `scheduling.ipynb` 行【124】-【188】（UCB_predict 构造与使用）。
  - `models/Informer.py`（简化版前向与输入签名）。

## 后续建议
1. 补齐 Informer 预测链路：明确输入特征构造与维度（例如构造长度为 `input_size=199` 的时间序列特征），并在 `UCB_predict` 中替换当前的均值预测逻辑。
2. 将节点资源参数与部署矩阵 `deploy_state` 完全外置到配置文件，支持不同集群规模与服务类型长度；在 `state.py` 中做一致性校验（例如 6×12 与资源维度的匹配）。
3. 为核心算法增加最小单元测试（尤其是 `selector` 的入参与返回值一致性），保证改动后的回归稳定。
4. 指标输出扩展：在 `outputs/summary.json` 中增加更细粒度的统计（吞吐随时间、各节点利用率分布、失败类型统计等），并考虑输出 CSV 以便绘图。
5. 将 `tqdm` 的进度条集成到 CLI 的可选参数中（例如 `--progress`），以便在批运行时关闭或在交互运行时开启。

---
审查完成。工程化重构与修复详见本次提交的 `src/flexload/**`、`scripts/run.py`、`configs/default.yaml` 与更新后的 `README.md`。
