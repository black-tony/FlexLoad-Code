# -*- coding: utf-8 -*-
from typing import List, Any

# QoS 违约率（QVR）
# - 时间视角：每个 slot 上，发生违约的节点占比；总体为所有 slot 的平均占比
# - 违约条件（参考 env.update_docker 语义）：
#   对节点队列中的每个任务，若满足以下任一条件则视为该节点在当前 slot 发生违约：
#     1) 若现在开始执行会超时：cur_time + (task_cpu / docker_cpu) > task_end_time
#     2) 资源不足：node.cpu < POD_CPU*service_coeff[kind] 或 node.mem < POD_MEM*service_coeff[kind]
#   只要该节点存在任意一个任务满足上述条件，即记为该节点违约。

def compute_qvr_slot(masters: List[Any], service_coefficient: List[float], POD_CPU: float, POD_MEM: float, cur_time: float,
                       bw_demand_per_kind: List[float], plr_threshold_per_kind: List[float]) -> int:
    violation_nodes = 0
    # 遍历每个 master 的每个节点
    for m in masters:
        for node in m.node_list:
            violated = False
            # 逐任务判断是否违约（资源或网络任一维度）
            for task in node.task_queue:
                # task 格式：[kind, start_time, end_time, cpu_req, mem_req, master_id]
                if not (isinstance(task, list) and len(task) >= 6) or task[0] == -1:
                    continue
                kind = int(task[0])
                end_time = float(task[2])
                task_cpu = float(task[3])
                # 找到对应 kind 的 docker（用于计算执行时长）
                docker_cpu = None
                for d in node.service_list:
                    if int(d.kind) == kind:
                        docker_cpu = float(d.cpu)
                        break
                # 若没有对应 docker，视为无法满足部署约束，判定为违约
                if docker_cpu is None:
                    violated = True
                    break
                # 执行时长估计（与 env.update_docker 相同）
                to_do_cpu = task_cpu / docker_cpu if docker_cpu > 0 else float('inf')
                # 资源需求（与部署时消耗一致）
                cpu_need = POD_CPU * float(service_coefficient[kind])
                mem_need = POD_MEM * float(service_coefficient[kind])
                # 网络需求与阈值（按任务类型）
                bw_need = float(bw_demand_per_kind[kind]) if kind < len(bw_demand_per_kind) else 0.0
                plr_thr = float(plr_threshold_per_kind[kind]) if kind < len(plr_threshold_per_kind) else 1.0
                # 条件 1：若现在开始会超时
                if cur_time + to_do_cpu > end_time:
                    violated = True
                    break
                # 条件 2：资源不足
                if (float(node.cpu) < cpu_need) or (float(node.mem) < mem_need):
                    violated = True
                    break
                # 条件 3：网络不足（带宽低于需求或丢包率超过阈值）
                if (getattr(node, "net_bw_mbps", 0.0) < bw_need) or (getattr(node, "plr", 0.0) > plr_thr):
                    violated = True
                    break
            if violated:
                violation_nodes += 1
    return violation_nodes


def finalize_qvr(total_slots: int, total_nodes: int, violation_count: int) -> float:
    denom = int(total_slots) * int(total_nodes)
    if denom <= 0:
        return 0.0
    return float(violation_count) / float(denom)


def compute_qvr_network_only_slot(masters: List[Any], bw_demand_per_kind: List[float], plr_threshold_per_kind: List[float]) -> int:
    """仅按网络维度统计当前 slot 违约的节点数。
    条件：node.net_bw_mbps < bw_need[kind] 或 node.plr > plr_thr[kind]，只要队列中存在任意任务类型满足上述条件，则该节点记为违约。
    """
    violation_nodes = 0
    for m in masters:
        for node in m.node_list:
            violated = False
            for task in node.task_queue:
                if not (isinstance(task, list) and len(task) >= 1) or task[0] == -1:
                    continue
                kind = int(task[0])
                bw_need = float(bw_demand_per_kind[kind]) if kind < len(bw_demand_per_kind) else 0.0
                plr_thr = float(plr_threshold_per_kind[kind]) if kind < len(plr_threshold_per_kind) else 1.0
                if (getattr(node, "net_bw_mbps", 0.0) < bw_need) or (getattr(node, "plr", 0.0) > plr_thr):
                    violated = True
                    break
            if violated:
                violation_nodes += 1
    return violation_nodes
