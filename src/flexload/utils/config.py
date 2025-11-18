# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

DEFAULTS: Dict[str, Any] = {
    "SLOT_TIME": 0.5,
    "MAX_TASK_TYPE": 12,
    "POD_CPU": 15.0,
    "POD_MEM": 1.0,
    "service_coefficient": [
        0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4
    ],
    # 网络维度配置（默认启用），trace 默认路径为 trace.csv
    "network": {
        "enabled": True,
        "trace_path": "trace.csv",
        # 默认按照 MAX_TASK_TYPE=12 填充，仿真器中会根据实际 MAX_TASK_TYPE 截断或补齐
        "bw_demand_per_kind": [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ],
        "plr_threshold_per_kind": [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05
        ],
    },
    "device": "auto",  # auto/cpu/cuda
    # 可配置规模：master 数量与每个 master 的节点数
    "num_masters": 2,
    "nodes_per_master": 3,
    # 维度参数（若未显式给出，将在 init_algorithms 中依据规模动态计算）
    "action_dim": 7,
    "state_dim": 88,
    "epsilon": 0.1,
    "predict": {
        "enable": True,
        "type": "mean",  # mean/informer
        "lookback": 20,
    },
    "informer": {
        "enable": False,
        "model_path": "best_informer_model.pth",
        "input_size": 199,
        "embed_size": 256,
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 8,
        "ff_hid_dim": 2048,
        "dropout": 0.1,
    },
    "RESULTS_DIR": "results",
    "OUTPUTS_DIR": "outputs",
    "LOGS_DIR": "logs",
    "data_task_1": "data/Task_1.csv",
    "data_task_2": "data/Task_2.csv",
    "deploy_state": None,
    "model": "KaiS",
}


def load_config(path: str) -> Dict[str, Any]:
    """从 YAML 或 JSON 加载配置，并合并默认值。"""
    cfg = {}
    if path.endswith(".yaml") or path.endswith(".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml 未安装，无法读取 YAML 配置")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            cfg.update(raw)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            cfg.update(raw)
    else:
        # 尝试 YAML 优先
        if os.path.exists(path + ".yaml"):
            return load_config(path + ".yaml")
        if os.path.exists(path + ".json"):
            return load_config(path + ".json")
        raise FileNotFoundError(f"未找到配置文件: {path}")

    # 合并默认值（缺省项填入）
    merged = DEFAULTS.copy()
    for k, v in cfg.items():
        merged[k] = v
    return merged
