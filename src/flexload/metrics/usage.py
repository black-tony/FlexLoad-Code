# -*- coding: utf-8 -*-
import json
from typing import List, Dict, Any

class ResourceUsageTracker:
    """替代 globals 的资源使用记录器。"""
    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def append_slot(self, master1, master2, cur_time: float):
        slot_usage = {
            "time": cur_time,
            "master1": [],
            "master2": [],
        }
        for i in range(3):
            slot_usage["master1"].append({
                "cpu": master1.node_list[i].cpu,
                "cpu_max": master1.node_list[i].cpu_max,
                "mem": master1.node_list[i].mem,
                "mem_max": master1.node_list[i].mem_max,
            })
            slot_usage["master2"].append({
                "cpu": master2.node_list[i].cpu,
                "cpu_max": master2.node_list[i].cpu_max,
                "mem": master2.node_list[i].mem,
                "mem_max": master2.node_list[i].mem_max,
            })
        self.history.append(slot_usage)

    def save_json(self, path: str):
        # 写文件前确保目录存在由上层创建
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, separators=(",", ":"))
