# -*- coding: utf-8 -*-
import json
from typing import List, Dict, Any

class ResourceUsageTracker:
    """替代 globals 的资源使用记录器。"""
    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def append_slot(self, master1, master2, cur_time: float):
        """兼容旧接口：记录两个 master 的资源使用。"""
        self.append_slot_multi([master1, master2], cur_time)

    def append_slot_multi(self, masters: List[Any], cur_time: float):
        """记录任意数量 master 的节点资源使用。
        结构：{"time": t, "masters": [[node_dict...], [node_dict...], ...]}
        """
        slot_usage = {
            "time": cur_time,
            "masters": [],
        }
        for m in masters:
            group = []
            for i in range(len(m.node_list)):
                node = m.node_list[i]
                group.append({
                    "cpu": node.cpu,
                    "cpu_max": node.cpu_max,
                    "mem": node.mem,
                    "mem_max": node.mem_max,
                    # 网络维度（兼容旧结构，新增可选字段）
                    "net_bw_mbps": getattr(node, "net_bw_mbps", 0.0),
                    "plr": getattr(node, "plr", 0.0),
                })
            slot_usage["masters"].append(group)
        self.history.append(slot_usage)

    def save_json(self, path: str):
        # 写文件前确保目录存在由上层创建
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, separators=(",", ":"))
