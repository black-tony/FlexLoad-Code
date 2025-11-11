# -*- coding: utf-8 -*-
from typing import Dict, Any, Tuple, Optional
import logging
import numpy as np
import torch

from flexload.scheduler.KaiS import Estimator
from flexload.scheduler.generic import get_generic_act
import flexload.scheduler.UCB as ucb_scheduler
import flexload.scheduler.DQN as DQN

from flexload.core.state import to_kais_state


def init_algorithms(cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """初始化可能用到的算法对象：KaiS、DQNAgent、Informer（可选）等。"""
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("device", "auto") == "cuda") else "cpu")
    # KaiS
    kais = Estimator(action_dim=int(cfg.get("action_dim", 7)), state_dim=int(cfg.get("state_dim", 88)), n_valid_node=2)
    # DQN Agent
    dqn_agent = DQN.DQNAgent(state_dim=int(cfg.get("state_dim", 88)), action_dim=int(cfg.get("action_dim", 7)), epsilon=float(cfg.get("epsilon", 0.1)), device=str(device))

    # Informer 模型（UCB_predict 可选使用）
    informer = None
    if bool(cfg.get("informer", {}).get("enable", False)):
        try:
            from flexload.models.Informer import Informer
            params = cfg.get("informer", {})
            informer = Informer(
                input_size=int(params.get("input_size", 199)),
                embed_size=int(params.get("embed_size", 256)),
                hidden_size=int(params.get("hidden_size", 512)),
                num_layers=int(params.get("num_layers", 4)),
                num_heads=int(params.get("num_heads", 8)),
                ff_hid_dim=int(params.get("ff_hid_dim", 2048)),
                dropout=float(params.get("dropout", 0.1)),
            ).to(device)
            model_path = params.get("model_path", "best_informer_model.pth")
            informer.load_state_dict(torch.load(model_path, map_location=device))
            logger and logger.info(f"Informer loaded from {model_path}")
        except Exception as e:
            informer = None
            logger and logger.warning(f"Informer load failed, fallback. err={e}")
    return {"kais": kais, "dqn": dqn_agent, "informer": informer, "device": device}


def select_action(model: str,
                  s_grid,
                  ava_node,
                  context,
                  epsilon: float,
                  cfg: Dict[str, Any],
                  algos: Dict[str, Any],
                  usage,
                  logger: Optional[logging.Logger] = None):
    """统一选择调度算法入口，保持返回格式一致性。"""
    model = str(model)
    if model == "KaiS":
        # 将 s_grid 转换为 [2,88] 的数值张量供 KaiS 使用
        s_kais = to_kais_state(s_grid)
        return algos["kais"].action(s_kais, ava_node, context, epsilon)
    if model == "UCB":
        return ucb_scheduler.get_act(s_grid, ava_node, context)
    if model == "DQN":
        return DQN.get_act(algos["dqn"], s_grid, ava_node, context)
    if model == "generic":
        return get_generic_act(s_grid, ava_node, context)
    if model == "Random":
        actions = [np.random.choice(ava_node[i]) for i in range(len(ava_node))]
        return actions, None, None, None, None, None, None
    if model == "UCB_predict":
        # 基于历史 usage 做简单预测（或可选使用 Informer）。
        LOOKBACK = int(cfg.get("predict", {}).get("lookback", 20))
        predictor = str(cfg.get("predict", {}).get("type", "mean"))  # mean/informer
        try:
            hist = usage.history
            if len(hist) == 0:
                return ucb_scheduler.get_act(s_grid, ava_node, context)
            seq = hist[-LOOKBACK:] if len(hist) >= LOOKBACK else hist[:]

            # 计算每个节点的 CPU 利用率预测（简单均值）
            # master1: idx 0..2; master2: idx 0..2
            m1_cpu_max = [s_grid[0][2][i][1] for i in range(3)]
            m2_cpu_max = [s_grid[1][2][i][1] for i in range(3)]
            m1_usage = [0.0]*3
            m2_usage = [0.0]*3
            for slot in seq:
                for i in range(3):
                    m1_cpu = float(slot['master1'][i]['cpu'])
                    m1_cpu_max_i = float(slot['master1'][i]['cpu_max'])
                    m1_usage[i] += (m1_cpu / m1_cpu_max_i) if m1_cpu_max_i != 0 else 0.0
                for i in range(3):
                    m2_cpu = float(slot['master2'][i]['cpu'])
                    m2_cpu_max_i = float(slot['master2'][i]['cpu_max'])
                    m2_usage[i] += (m2_cpu / m2_cpu_max_i) if m2_cpu_max_i != 0 else 0.0
            m1_usage = [u/len(seq) for u in m1_usage]
            m2_usage = [u/len(seq) for u in m2_usage]

            cpu_pred_master1 = [[float(m1_usage[i]) * float(m1_cpu_max[i]), float(m1_cpu_max[i])] for i in range(3)]
            cpu_pred_master2 = [[float(m2_usage[i]) * float(m2_cpu_max[i]), float(m2_cpu_max[i])] for i in range(3)]

            s_grid_pred = [
                [s_grid[0][0], s_grid[0][1], cpu_pred_master1, s_grid[0][3]],
                [s_grid[1][0], s_grid[1][1], cpu_pred_master2, s_grid[1][3]]
            ]

            # 如配置要求使用 Informer，则后续可替换上述均值预测逻辑（保留模型加载）
            if predictor == "informer" and algos.get("informer") is not None:
                logger and logger.info("predictor=informer is enabled, but input shaping is unspecified, fallback to mean predictor")

            return ucb_scheduler.get_act(s_grid_pred, ava_node, context)
        except Exception as e:
            logger and logger.warning(f"UCB_predict fallback due to error: {e}")
            return ucb_scheduler.get_act(s_grid, ava_node, context)
    # 默认回退到随机
    actions = [np.random.choice(ava_node[i]) for i in range(len(ava_node))]
    return actions, None, None, None, None, None, None
