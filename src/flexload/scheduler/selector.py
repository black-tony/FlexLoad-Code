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
    num_masters = int(cfg.get("num_masters", 2))
    nodes_per_master = int(cfg.get("nodes_per_master", 3))
    total_nodes = num_masters * nodes_per_master
    action_dim = total_nodes + 1
    state_dim = int(cfg.get("state_dim", 88))
    kais = Estimator(action_dim=action_dim, state_dim=state_dim, n_valid_node=num_masters)
    # DQN Agent
    dqn_agent = DQN.DQNAgent(state_dim=state_dim, action_dim=action_dim, epsilon=float(cfg.get("epsilon", 0.1)), device=str(device))

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
    return {"kais": kais, "dqn": dqn_agent, "informer": informer, "device": device, "total_nodes": total_nodes, "num_masters": num_masters, "nodes_per_master": nodes_per_master}


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
        # 将 s_grid 转换为 [len(s_grid),88] 的数值张量供 KaiS 使用
        s_kais = to_kais_state(s_grid)
        # 动态对齐有效节点数
        try:
            algos["kais"].n_valid_node = len(s_grid)
        except Exception:
            pass
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

            # 计算每个 master 的每个节点 CPU 利用率预测（简单均值）
            cpu_pred_groups = []
            for m_idx in range(len(s_grid)):
                # 当前 master 的节点数：取 s_grid 的 cpu_list 长度
                node_count = len(s_grid[m_idx][2])
                # 从历史记录中提取 masters[m_idx] 的节点使用
                usage_sum = [0.0] * node_count
                count = len(seq)
                # 兼容：历史中若不存在该 master（规模变化时），直接使用当前 s_grid 的 cpu_list
                safe = True
                for slot in seq:
                    masters_hist = slot.get("masters")
                    if not masters_hist or m_idx >= len(masters_hist):
                        safe = False
                        break
                    for i in range(node_count):
                        cpu = float(masters_hist[m_idx][i]["cpu"])
                        cpu_max_i = float(masters_hist[m_idx][i]["cpu_max"]) or 0.0
                        usage_sum[i] += (cpu / cpu_max_i) if cpu_max_i != 0 else 0.0
                if (not safe) or count == 0:
                    # 回退：使用当前 s_grid 的 cpu_list
                    cpu_pred_groups.append(s_grid[m_idx][2])
                else:
                    avg_usage = [u / count for u in usage_sum]
                    cpu_max_list = [float(s_grid[m_idx][2][i][1]) for i in range(node_count)]
                    cpu_pred = [[float(avg_usage[i]) * float(cpu_max_list[i]), float(cpu_max_list[i])] for i in range(node_count)]
                    cpu_pred_groups.append(cpu_pred)

            # 组装新的 s_grid，用预测的 cpu_list 替换原 cpu_list（其余信息保持不变）
            s_grid_pred = []
            for m_idx in range(len(s_grid)):
                s_grid_pred.append([
                    s_grid[m_idx][0], s_grid[m_idx][1], cpu_pred_groups[m_idx], s_grid[m_idx][3]
                ])

            # 如果配置要求使用 Informer，并且模型可用，则尝试用 Informer 进行下一时刻预测
            if predictor == "informer" and algos.get("informer") is not None:
                try:
                    informer = algos["informer"]
                    device = algos.get("device", torch.device("cpu"))
                    input_size = getattr(informer, "input_size", int(cfg.get("informer", {}).get("input_size", 199)))
                    
                    # 组装序列输入：[1, seq_len, input_size]；每个时间步拼接所有 master/节点的 CPU 利用率（归一化），不足零填充，超出截断
                    seq_len = len(seq)
                    arr = np.zeros((1, seq_len, input_size), dtype=np.float32)
                    
                    # 计算每组节点数量与总节点数（按当前 s_grid 的顺序）
                    group_counts = [len(s_grid[m][2]) for m in range(len(s_grid))]
                    total_nodes = int(np.sum(group_counts))
                    
                    for t_idx, slot in enumerate(seq):
                        vec = []
                        masters_hist = slot.get("masters")
                        if masters_hist and isinstance(masters_hist, list):
                            # 新结构：{"masters": [[{cpu,...},...], ...]}
                            for m_idx in range(len(s_grid)):
                                for i in range(group_counts[m_idx]):
                                    cpu = float(masters_hist[m_idx][i]["cpu"])
                                    cpu_max_i = float(masters_hist[m_idx][i]["cpu_max"]) or 0.0
                                    vec.append((cpu / cpu_max_i) if cpu_max_i != 0 else 0.0)
                        else:
                            # 兼容旧结构：{"master1": [...], "master2": [...]}，仅支持前两组
                            for m_idx in range(len(s_grid)):
                                if m_idx == 0:
                                    group = slot.get("master1", [])
                                elif m_idx == 1:
                                    group = slot.get("master2", [])
                                else:
                                    group = []
                                for i in range(group_counts[m_idx]):
                                    if i < len(group):
                                        cpu = float(group[i]["cpu"])
                                        cpu_max_i = float(group[i]["cpu_max"]) or 0.0
                                        vec.append((cpu / cpu_max_i) if cpu_max_i != 0 else 0.0)
                                    else:
                                        vec.append(0.0)
                    
                        # 零填充或截断到 input_size
                        if len(vec) < input_size:
                            vec = vec + [0.0] * (input_size - len(vec))
                        else:
                            vec = vec[:input_size]
                        arr[0, t_idx, :] = np.array(vec, dtype=np.float32)
                    
                    x = torch.tensor(arr, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        pred_next = informer(x)  # [1, input_size]
                    pred_vec = pred_next.detach().cpu().numpy()[0]
                    
                    # 将前 total_nodes 个预测值映射回各 master 的各节点，形成新的 s_grid_pred_informer
                    offset = 0
                    cpu_pred_groups_inf = []
                    for m_idx in range(len(s_grid)):
                        group_pred = []
                        for i in range(group_counts[m_idx]):
                            if offset + i < len(pred_vec):
                                pred_usage = float(np.clip(pred_vec[offset + i], 0.0, 1.0))
                            else:
                                pred_usage = 0.0
                            cpu_max = float(s_grid[m_idx][2][i][1])
                            group_pred.append([pred_usage * cpu_max, cpu_max])
                        cpu_pred_groups_inf.append(group_pred)
                        offset += group_counts[m_idx]
                    
                    s_grid_pred_inf = []
                    for m_idx in range(len(s_grid)):
                        s_grid_pred_inf.append([
                            s_grid[m_idx][0], s_grid[m_idx][1], cpu_pred_groups_inf[m_idx], s_grid[m_idx][3]
                        ])
                    
                    return ucb_scheduler.get_act(s_grid_pred_inf, ava_node, context)
                except Exception as ie:
                    logger and logger.warning(f"Informer predict failed, fallback to mean predictor. err={ie}")
            
            # 默认返回均值预测的结果
            return ucb_scheduler.get_act(s_grid_pred, ava_node, context)
        except Exception as e:
            logger and logger.warning(f"UCB_predict fallback due to error: {e}")
            return ucb_scheduler.get_act(s_grid, ava_node, context)
    # 默认回退到随机
    actions = [np.random.choice(ava_node[i]) for i in range(len(ava_node))]
    return actions, None, None, None, None, None, None
