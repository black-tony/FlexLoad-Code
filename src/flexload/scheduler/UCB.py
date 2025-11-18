import numpy as np

class UCB:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.counts = np.zeros(num_nodes)  # Number of times each node was selected
        self.values = np.zeros(num_nodes)  # Average reward of each node

    def select_node(self):
        total_counts = np.sum(self.counts)
        if total_counts < self.num_nodes:
            # Select each node at least once
            return total_counts
        ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, chosen_node, reward):
        self.counts[chosen_node] += 1
        n = self.counts[chosen_node]
        value = self.values[chosen_node]
        # Update the estimated value of the chosen node
        self.values[chosen_node] = ((n - 1) / n) * value + (1 / n) * reward


def _map_global_node_to_master_local(chosen_node: int, s_grid):
    """将全局节点索引映射到 (master_id, local_node)。cloud 返回 None。
    cloud 索引 = sum(len(cpu_list) for 每组)
    """
    total_nodes = 0
    lengths = []
    for g in s_grid:
        n = len(g[2])
        lengths.append(n)
        total_nodes += n
    cloud_index = total_nodes
    if chosen_node == cloud_index:
        return None
    acc = 0
    for midx, n in enumerate(lengths):
        if chosen_node < acc + n:
            return midx, chosen_node - acc
        acc += n
    return None


def _compute_node_score(s_grid, global_node: int, task, ctx) -> float:
    """
    基于当前任务与节点状态计算候选节点的打分：
    - 部署约束：deploy_state[global_node][kind] == 1（Cloud 始终可部署）
    - 执行时长估算：to_do_time = task_cpu / (POD_CPU * service_coefficient[kind])
    - 资源约束：node.cpu >= POD_CPU*coefficient[kind] 且 node.mem >= POD_MEM*coefficient[kind]
    - 截止时间约束：cur_time + to_do_time <= end_time
    - 评分：w_feasible*I(feasible) + w_headroom*((1-cpu_usage)+(1-mem_usage)) - w_queue*queue_len
      权重默认 {w_feasible=1.0, w_headroom=0.5, w_queue=0.1}，可从 ctx['ucb_weights'] 覆盖。
    """
    # 解析任务
    if not (isinstance(task, list) and len(task) >= 6) or int(task[0]) == -1:
        return -1e9  # 无效任务，给极低分
    kind = int(task[0])
    end_time = float(task[2])
    task_cpu = float(task[3])

    # 解析上下文参数
    POD_CPU = float(ctx.get("POD_CPU", 15.0))
    POD_MEM = float(ctx.get("POD_MEM", 1.0))
    service_coeff = list(ctx.get("service_coefficient", []))
    cur_time = float(ctx.get("cur_time", 0.0))
    weights = ctx.get("ucb_weights") or {"w_feasible": 1.0, "w_headroom": 0.5, "w_queue": 0.1}
    w_feasible = float(weights.get("w_feasible", 1.0))
    w_headroom = float(weights.get("w_headroom", 0.5))
    w_queue = float(weights.get("w_queue", 0.1))

    coeff_k = float(service_coeff[kind]) if (isinstance(service_coeff, list) and kind < len(service_coeff)) else 1.0
    docker_cpu = POD_CPU * coeff_k
    to_do_time = (task_cpu / docker_cpu) if docker_cpu > 0 else float('inf')

    # 计算总节点与 Cloud 索引
    total_nodes = 0
    group_lengths = []
    for g in s_grid:
        n = len(g[2])
        total_nodes += n
        group_lengths.append(n)
    cloud_index = total_nodes

    # 部署约束：Cloud 视为始终可部署（已在 Simulator 中初始化了 Cloud 的 docker）
    deploy_state = s_grid[0][0] if len(s_grid) > 0 else []
    if global_node != cloud_index:
        if not (isinstance(deploy_state, list) and global_node < len(deploy_state) and kind < len(deploy_state[global_node]) and int(deploy_state[global_node][kind]) == 1):
            return -1e6  # 不可部署

    # Cloud 直接评估可行性与基础分数
    if global_node == cloud_index:
        feasible = (cur_time + to_do_time <= end_time)
        headroom_sum = 2.0  # (1-0)+(1-0)
        queue_len = 0.0
        return (w_feasible * (1.0 if feasible else 0.0)) + (w_headroom * headroom_sum) - (w_queue * queue_len)

    # 非 Cloud：定位 master/local_node，提取资源与队列长度
    acc = 0
    master_id = 0
    local_node = 0
    for midx, n in enumerate(group_lengths):
        if global_node < acc + n:
            master_id = midx
            local_node = global_node - acc
            break
        acc += n
    g = s_grid[master_id]
    cpu_cur, cpu_max = float(g[2][local_node][0]), float(g[2][local_node][1])
    mem_cur, mem_max = float(g[3][local_node][0]), float(g[3][local_node][1])
    # 使用率（已用占比）
    cpu_usage = 1.0 - (cpu_cur / cpu_max if cpu_max > 0 else 0.0)
    mem_usage = 1.0 - (mem_cur / mem_max if mem_max > 0 else 0.0)
    cpu_usage = max(0.0, min(1.0, cpu_usage))
    mem_usage = max(0.0, min(1.0, mem_usage))
    headroom_sum = (1.0 - cpu_usage) + (1.0 - mem_usage)

    # 队列长度（task_nums 结构：[ [len(master_queue), len(node0), len(node1), ...] ]）
    node_queue_len = 0.0
    task_nums_nested = g[1]
    if isinstance(task_nums_nested, list) and len(task_nums_nested) > 0 and isinstance(task_nums_nested[0], list):
        stats = task_nums_nested[0]
        idx = local_node + 1
        if 0 <= idx < len(stats):
            try:
                node_queue_len = float(stats[idx])
            except Exception:
                try:
                    node_queue_len = float(int(stats[idx]))
                except Exception:
                    node_queue_len = 0.0

    # 资源与截止时间条件
    res_need_cpu = POD_CPU * coeff_k
    res_need_mem = POD_MEM * coeff_k
    resources_ok = (cpu_cur >= res_need_cpu) and (mem_cur >= res_need_mem)
    deadline_ok = (cur_time + to_do_time <= end_time)
    feasible = resources_ok and deadline_ok

    score = (w_feasible * (1.0 if feasible else 0.0)) + (w_headroom * headroom_sum) - (w_queue * node_queue_len)
    return float(score)


def get_act(s_grid, ava_node, context):
    num_tasks = len(ava_node)
    actions = []

    # 计算总节点与 Cloud 索引
    total_nodes = 0
    for g in s_grid:
        total_nodes += len(g[2])
    cloud_index = total_nodes

    for task_index in range(num_tasks):
        candidates = ava_node[task_index]
        # 获取该任务的上下文
        task = None
        if isinstance(context, dict):
            try:
                task = context.get("curr_tasks", [None] * num_tasks)[task_index]
            except Exception:
                task = None
        # 无候选（容错：回退 Cloud）
        if not candidates:
            actions.append(cloud_index)
            continue
        # 单候选（通常是仅 Cloud 可选）
        if len(candidates) == 1:
            actions.append(candidates[0])
            continue
        # 逐候选计算分数
        best_score = -1e18
        best_nodes = []
        for global_node in candidates:
            try:
                score = _compute_node_score(s_grid, int(global_node), task, context if isinstance(context, dict) else {})
            except Exception:
                score = -1e18
            if score > best_score + 1e-9:
                best_score = score
                best_nodes = [int(global_node)]
            elif abs(score - best_score) <= 1e-9:
                best_nodes.append(int(global_node))
        # 选择动作：若不可行或分数相同，优先 Cloud
        if len(best_nodes) == 0:
            actions.append(cloud_index if cloud_index in candidates else candidates[0])
        else:
            if cloud_index in best_nodes:
                actions.append(cloud_index)
            else:
                actions.append(best_nodes[0])

    # 保持返回结构兼容上层
    return actions, None, None, None, None, None, None
