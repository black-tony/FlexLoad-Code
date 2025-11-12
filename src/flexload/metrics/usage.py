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
                group.append({
                    "cpu": m.node_list[i].cpu,
                    "cpu_max": m.node_list[i].cpu_max,
                    "mem": m.node_list[i].mem,
                    "mem_max": m.node_list[i].mem_max,
                })
            slot_usage["masters"].append(group)
        self.history.append(slot_usage)

    def save_json(self, path: str):
        # 写文件前确保目录存在由上层创建
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, separators=(",", ":"))
