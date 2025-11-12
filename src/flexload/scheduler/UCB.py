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


def ucb_rounds(ucb_instance:UCB, s_grid, ava_node, task_index, rounds=500):
    for _ in range(rounds):
        chosen_index = int(ucb_instance.select_node())
        chosen_node = ava_node[task_index][chosen_index]

        # Simulate a reward based on CPU and memory utilization
        mapped = _map_global_node_to_master_local(chosen_node, s_grid)
        if mapped is not None:
            master, local_node = mapped
            cpu_usage = s_grid[master][2][local_node][0] / s_grid[master][2][local_node][1]
            mem_usage = s_grid[master][3][local_node][0] / s_grid[master][3][local_node][1]
            reward = cpu_usage + mem_usage
        else:
            reward = 0
        # Update UCB with the observed reward
        ucb_instance.update(chosen_index, reward)
    return int(ucb_instance.select_node())


def get_act(s_grid, ava_node, context):
    num_tasks = len(ava_node)
    actions = []

    # Initialize UCB for each task
    ucb_instances = [UCB(len(ava_node[i])) for i in range(num_tasks)]

    for task_index in range(num_tasks):
        if len(ava_node[task_index]) == 1:
            # If only cloud computing is available
            actions.append(ava_node[task_index][0])
            continue
        actions.append((ucb_rounds(ucb_instances[task_index], s_grid, ava_node, task_index)))

    return actions, None, None, None, None, None, None
