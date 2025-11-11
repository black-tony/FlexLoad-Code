# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any
import numpy as np

from scheduling_util.env.platform import Master


def _flatten_deploy_state(deploy_state: List[List[int]]) -> List[float]:
    flat = []
    for row in deploy_state:
        for v in row:
            flat.append(float(v))
    return flat


def _task_num(master: Master) -> List[int]:
    # [all_queue_len, node0_queue, node1_queue, node2_queue]
    nums = [len(master.task_queue)]
    for i in range(3):
        nums.append(len(master.node_list[i].task_queue))
    return nums


def build_s_grid(master1: Master, master2: Master, deploy_state: List[List[int]]):
    """
    构造 UCB/generic/DQN 所需的 s_grid 结构：
    [[deploy_state, [task_num1], cpu_list1, mem_list1],
     [deploy_state, [task_num2], cpu_list2, mem_list2]]
    注意：第二组必须使用 cpu_list2/mem_list2，修复原 notebook 中的复用错误。
    """
    cpu_list1 = []
    mem_list1 = []
    cpu_list2 = []
    mem_list2 = []

    task_num1 = _task_num(master1)
    task_num2 = _task_num(master2)

    for i in range(3):
        cpu_list1.append([master1.node_list[i].cpu, master1.node_list[i].cpu_max])
        mem_list1.append([master1.node_list[i].mem, master1.node_list[i].mem_max])
    for i in range(3):
        cpu_list2.append([master2.node_list[i].cpu, master2.node_list[i].cpu_max])
        mem_list2.append([master2.node_list[i].mem, master2.node_list[i].mem_max])

    s_grid = [
        [deploy_state, [task_num1], cpu_list1, mem_list1],
        [deploy_state, [task_num2], cpu_list2, mem_list2],
    ]
    return s_grid


def build_ava_nodes(deploy_state: List[List[int]], curr_task: List[List[Any]]):
    """
    构造可用节点列表 ava_node：
    - 对每个任务（两个），列出允许部署的节点（0..5）以及 6 表示 Cloud。
    """
    ava_node = []
    for i in range(len(curr_task)):
        tmp_list = [6]  # Cloud computing
        if isinstance(curr_task[i], list) and len(curr_task[i]) > 0 and curr_task[i][0] != -1:
            task_kind = curr_task[i][0]
            for ii in range(len(deploy_state)):
                if deploy_state[ii][task_kind] == 1:
                    tmp_list.append(ii)
        ava_node.append(tmp_list)
    return ava_node


def to_kais_state(s_grid) -> np.ndarray:
    """
    将 s_grid 转换为 KaiS.Estimator 所需的数值张量形状 [n_valid_node=2, state_dim=88]。
    规则：
    - 展平 deploy_state: 6x12 -> 72
    - 展平任务队列统计: [len(all_queue), len(node0), len(node1), len(node2)] -> 4
    - 展平 CPU 列表: 3 x [cpu, cpu_max] -> 6
    - 展平 MEM 列表: 3 x [mem, mem_max] -> 6
    合计：72 + 4 + 6 + 6 = 88
    如有偏差将按 88 维进行截断或零填充，以满足 Estimator.state_dim。
    """
    groups = []
    for g in s_grid:
        deploy_state, task_num_nested, cpu_list, mem_list = g
        vec = []
        vec.extend(_flatten_deploy_state(deploy_state))
        # task_num_nested 形如 [ [a,b,c,d] ]
        task_stats = task_num_nested[0] if len(task_num_nested) > 0 else []
        vec.extend([float(x) for x in task_stats])
        for c in cpu_list:
            vec.extend([float(c[0]), float(c[1])])
        for m in mem_list:
            vec.extend([float(m[0]), float(m[1])])
        # 归一到 88 维
        if len(vec) < 88:
            vec += [0.0] * (88 - len(vec))
        elif len(vec) > 88:
            vec = vec[:88]
        groups.append(vec)
    return np.array(groups, dtype=np.float32)
