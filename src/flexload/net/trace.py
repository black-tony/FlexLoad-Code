# -*- coding: utf-8 -*-
from typing import List, Dict, Any
import csv
import os

# 解析网络 trace
# 支持两种格式：
# 1) 窄表：time_slot,node_id,bw_mbps,plr
# 2) 宽表：第一列为 time_slot，后续按 node_0_bw,node_0_plr,node_1_bw,node_1_plr,...
# 返回：List[Dict[int, Dict[str, float]]]，每个时间步一个字典：{node_id: {"bw_mbps": x, "plr": y}}

def read_trace_csv(path: str) -> List[Dict[int, Dict[str, float]]]:
    if not os.path.exists(path):
        return []
    trace: List[Dict[int, Dict[str, float]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    # 判断是否有表头（含非纯数字/长度不等）
    has_header = any([not c.isdigit() for c in header])
    data_rows = rows[1:] if has_header else rows
    # 格式判断：窄表（4列）或宽表（>=3列且第一列为 time_slot/数字）
    if len(header) == 4 or (not has_header and len(header) == 4):
        # 窄表
        slot_map: Dict[float, Dict[int, Dict[str, float]]] = {}
        for r in data_rows:
            if len(r) < 4:
                continue
            try:
                t = float(r[0])
                node_id = int(r[1])
                bw = float(r[2])
                plr = float(r[3])
            except Exception:
                continue
            slot_map.setdefault(t, {})[node_id] = {"bw_mbps": bw, "plr": plr}
        # 按时间排序生成列表
        for t in sorted(slot_map.keys()):
            trace.append(slot_map[t])
    else:
        # 宽表：第一列为 time_slot，后续列按两个一组（bw, plr）
        # 尝试解析列名中的 node_x_bw/node_x_plr 或按列索引推断
        start_idx = 1
        # 统计最大节点数量（bw/plr 成对）
        # 每两个列对应一个节点
        for r in data_rows:
            step: Dict[int, Dict[str, float]] = {}
            try:
                _ = float(r[0])  # time_slot
            except Exception:
                continue
            col = start_idx
            node_id = 0
            while col + 1 < len(r):
                try:
                    bw = float(r[col]) if r[col] != "" else 0.0
                    plr = float(r[col + 1]) if r[col + 1] != "" else 0.0
                except Exception:
                    bw, plr = 0.0, 0.0
                step[node_id] = {"bw_mbps": bw, "plr": plr}
                node_id += 1
                col += 2
            trace.append(step)
    return trace


def get_step(trace: List[Dict[int, Dict[str, float]]], step_idx: int, total_nodes: int) -> Dict[int, Dict[str, float]]:
    if not trace or step_idx < 0 or step_idx >= len(trace):
        # 越界或空：返回全 0
        return {i: {"bw_mbps": 0.0, "plr": 0.0} for i in range(total_nodes)}
    step = trace[step_idx]
    # 补齐缺失节点
    for i in range(total_nodes):
        if i not in step:
            step[i] = {"bw_mbps": 0.0, "plr": 0.0}
        else:
            # 健壮性：缺失字段补 0
            step[i].setdefault("bw_mbps", 0.0)
            step[i].setdefault("plr", 0.0)
    return step


def apply_to_nodes(masters: List[Any], step_map: Dict[int, Dict[str, float]], nodes_per_master: int) -> None:
    # 将 bw/plr 写入每个 Node（新增字段：net_bw_mbps、plr）
    total_nodes = sum([len(m.node_list) for m in masters])
    for node_id in range(total_nodes):
        master_id = node_id // nodes_per_master
        local_idx = node_id % nodes_per_master
        if master_id >= len(masters) or local_idx >= len(masters[master_id].node_list):
            continue
        d = step_map.get(node_id, {"bw_mbps": 0.0, "plr": 0.0})
        masters[master_id].node_list[local_idx].net_bw_mbps = float(d.get("bw_mbps", 0.0))
        masters[master_id].node_list[local_idx].plr = float(d.get("plr", 0.0))
