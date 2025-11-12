# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any
import numpy as np

from flexload.env.platform import Master


def _flatten_deploy_state(deploy_state: List[List[int]]) -> List[float]:
    flat = []
    for row in deploy_state:
        for v in row:
            flat.append(float(v))
    return flat


def _task_num(master: Master) -> List[int]:
    # [all_queue_len, node0_queue, node1_queue, ...]
    nums = [len(master.task_queue)]
    for i in range(len(master.node_list)):
        nums.append(len(master.node_list[i].task_queue))
    return nums


def build_s_grid_multi(masters: List[Master], deploy_state: List[List[int]]):
    """
    构造多 master 的 s_grid：
    [[deploy_state, [task_num_m], cpu_list_m, mem_list_m] for m in masters]
    """
    s_grid = []
    for master in masters:
        cpu_list = []
        mem_list = []
        task_nums = [len(master.task_queue)]
        for i in range(len(master.node_list)):
            cpu_list.append([master.node_list[i].cpu, master.node_list[i].cpu_max])
            mem_list.append([master.node_list[i].mem, master.node_list[i].mem_max])
            task_nums.append(len(master.node_list[i].task_queue))
        s_grid.append([deploy_state, [task_nums], cpu_list, mem_list])
    return s_grid


def build_ava_nodes(deploy_state: List[List[int]], curr_task: List[List[Any]]):
    """
    构造可用节点列表 ava_node：
    - 对每个任务，列出允许部署的节点（全局索引 0..total_nodes-1）以及 total_nodes 表示 Cloud。
    """
    ava_node = []
    total_nodes = len(deploy_state)
    for i in range(len(curr_task)):
        tmp_list = [total_nodes]  # Cloud computing index
        if isinstance(curr_task[i], list) and len(curr_task[i]) > 0 and curr_task[i][0] != -1:
            task_kind = curr_task[i][0]
            for ii in range(total_nodes):
                if deploy_state[ii][task_kind] == 1:
                    tmp_list.append(ii)
        ava_node.append(tmp_list)
    return ava_node


def to_kais_state(s_grid) -> np.ndarray:
    """
    将 s_grid 转换为 KaiS.Estimator 所需的数值张量形状 [n_valid_node=len(s_grid), state_dim=88]。
    规则：
    - 展平 deploy_state: total_nodes x MAX_TASK_TYPE -> 72（按 88 维做截断/填充）
    - 展平任务队列统计: [len(all_queue), len(node0), len(node1), ...] -> 动态长度
    - 展平 CPU 列表: nodes_per_master x [cpu, cpu_max] -> 动态长度
    - 展平 MEM 列表: nodes_per_master x [mem, mem_max] -> 动态长度
    合计按 88 维进行截断或零填充，以满足 Estimator.state_dim。
    """
    groups = []
    for g in s_grid:
        deploy_state, task_num_nested, cpu_list, mem_list = g
        vec = []
        vec.extend(_flatten_deploy_state(deploy_state))
        # task_num_nested 形如 [ [a,b,c,d,...] ]
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
