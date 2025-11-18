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


def _compute_node_score(s_grid, global_node, w_cpu: float = 0.5, w_mem: float = 0.5, w_queue: float = 0.1) -> float:
    """
    计算节点的综合得分：容量余量 + 队列惩罚
    - cpu_headroom = 1 - cpu_usage
    - mem_headroom = 1 - mem_usage
    - queue_penalty = node_queue_len（简单使用队列长度作为惩罚）
    - score = w_cpu*cpu_headroom + w_mem*mem_headroom - w_queue*queue_penalty

    返回值：该节点的分数（float）。若为 Cloud（映射为 None），返回 0.0。
    """
    mapped = _map_global_node_to_master_local(global_node, s_grid)
    if mapped is None:
        # Cloud：无具体资源状态，保持中性分数 0.0，以避免偏向 Cloud
        return 0.0

    master, local_node = mapped
    # CPU/MEM 使用率
    cpu_used = float(s_grid[master][2][local_node][0])
    cpu_cap = float(s_grid[master][2][local_node][1])
    mem_used = float(s_grid[master][3][local_node][0])
    mem_cap = float(s_grid[master][3][local_node][1])

    cpu_usage = (cpu_used / cpu_cap) if cpu_cap > 0 else 1.0
    mem_usage = (mem_used / mem_cap) if mem_cap > 0 else 1.0

    # 限制到 [0,1]
    cpu_usage = max(0.0, min(1.0, cpu_usage))
    mem_usage = max(0.0, min(1.0, mem_usage))

    cpu_headroom = 1.0 - cpu_usage
    mem_headroom = 1.0 - mem_usage

    # 队列长度（task_nums 结构：[ [len(master_queue), len(node0), len(node1), ...] ]）
    node_queue_len = 0
    task_nums_nested = s_grid[master][1]
    if isinstance(task_nums_nested, list) and len(task_nums_nested) > 0 and isinstance(task_nums_nested[0], list):
        stats = task_nums_nested[0]
        idx = local_node + 1  # 第一个元素是 master 总队列，节点从 +1 开始
        if 0 <= idx < len(stats):
            try:
                node_queue_len = int(stats[idx])
            except Exception:
                try:
                    node_queue_len = int(float(stats[idx]))
                except Exception:
                    node_queue_len = 0

    queue_penalty = float(node_queue_len)
    score = w_cpu * cpu_headroom + w_mem * mem_headroom - w_queue * queue_penalty
    return float(score)


def get_act(s_grid, ava_node, context):
    num_tasks = len(ava_node)
    actions = []

    # 可选：从 context 注入权重配置
    w_cpu = 0.5
    w_mem = 0.5
    w_queue = 0.1
    try:
        if isinstance(context, dict) and 'ucb_weights' in context:
            ws = context['ucb_weights']
            w_cpu = float(ws.get('w_cpu', w_cpu))
            w_mem = float(ws.get('w_mem', w_mem))
            w_queue = float(ws.get('w_queue', w_queue))
    except Exception:
        # 忽略配置解析错误，保持默认权重
        pass

    for task_index in range(num_tasks):
        candidates = ava_node[task_index]
        # 无候选（理论不会发生，因为至少有 Cloud），容错处理
        if not candidates:
            actions.append(None)
            continue
        # 单候选（通常是仅 Cloud 可选）
        if len(candidates) == 1:
            actions.append(candidates[0])
            continue
        # 计算每个候选的分数，选择分数最高的“全局节点ID”
        best_node = None
        best_score = None
        for global_node in candidates:
            score = _compute_node_score(s_grid, global_node, w_cpu=w_cpu, w_mem=w_mem, w_queue=w_queue)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_node = global_node
        actions.append(best_node)

    # 保持返回结构兼容上层
    return actions, None, None, None, None, None, None
