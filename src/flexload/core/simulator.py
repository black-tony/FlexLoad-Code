# -*- coding: utf-8 -*-
import time
import json
import logging
import sys
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from flexload.env.platform import Node, Master, Cloud, Docker
from flexload.env.env_run import (
    get_all_task,
    update_task_queue,
    check_queue,
    update_docker,
)

from flexload.core.state import build_ava_nodes
from flexload.scheduler.selector import select_action, init_algorithms
from flexload.metrics.usage import ResourceUsageTracker
from flexload.utils.logging import ensure_dir


class Simulator:
    """
    统一的仿真执行器：封装原 scheduling.ipynb 的主循环。
    - 提供 run() 方法执行一次或多次仿真
    - 通过 selector 统一选择调度策略
    - 通过 metrics 记录资源使用与指标输出
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
        self.cfg = config
        self.logger = logger or logging.getLogger("flexload.simulator")
        # 环境参数
        self.SLOT_TIME = float(config.get("SLOT_TIME", 0.5))
        self.MAX_TASK_TYPE = int(config.get("MAX_TASK_TYPE", 12))
        self.POD_CPU = float(config.get("POD_CPU", 15.0))
        self.POD_MEM = float(config.get("POD_MEM", 1.0))
        self.service_coeff = list(config.get("service_coefficient", [
            0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4
        ]))
        # 如果给了超过 MAX_TASK_TYPE 的系数，截断；不足则补 1.0
        if len(self.service_coeff) < self.MAX_TASK_TYPE:
            self.service_coeff += [1.0] * (self.MAX_TASK_TYPE - len(self.service_coeff))
        self.service_coeff = self.service_coeff[:self.MAX_TASK_TYPE]

        # 可配置规模
        self.num_masters = int(config.get("num_masters", 2))
        self.nodes_per_master = int(config.get("nodes_per_master", 3))
        self.total_nodes = self.num_masters * self.nodes_per_master

        # 目录
        self.results_dir = config.get("RESULTS_DIR", "results")
        self.outputs_dir = config.get("OUTPUTS_DIR", "outputs")
        self.logs_dir = config.get("LOGS_DIR", "logs")
        for d in [self.results_dir, self.outputs_dir, self.logs_dir]:
            ensure_dir(d)

        # 选择器与算法实例（KaiS/DQN/Informer 等）
        self.algos = init_algorithms(config, self.logger)
        self.model_name = str(config.get("model", "KaiS"))

        # 资源使用记录器（替代 globals/node_resource_usage）
        self.usage_tracker = ResourceUsageTracker()

    @staticmethod
    def calculate_reward(master1: Master, master2: Master, cur_done: List[float], cur_undone: List[float]) -> List[float]:
        weight = 1.0
        all_task = [float(cur_done[0] + cur_undone[0]), float(cur_done[1] + cur_undone[1])]
        fail_task = [float(cur_undone[0]), float(cur_undone[1])]
        reward = []
        # The ratio of requests that violate delay requirements
        task_fail_rate = []
        task_fail_rate.append(fail_task[0] / all_task[0] if all_task[0] != 0 else 0.0)
        task_fail_rate.append(fail_task[1] / all_task[1] if all_task[1] != 0 else 0.0)

        # The standard deviation of the CPU and memory usage
        use_rate1 = []
        use_rate2 = []
        for i in range(3):
            use_rate1.append(master1.node_list[i].cpu / master1.node_list[i].cpu_max)
            use_rate1.append(master1.node_list[i].mem / master1.node_list[i].mem_max)
            use_rate2.append(master2.node_list[i].cpu / master2.node_list[i].cpu_max)
            use_rate2.append(master2.node_list[i].mem / master2.node_list[i].mem_max)
        standard_list = [np.std(use_rate1, ddof=1), np.std(use_rate2, ddof=1)]

        reward.append(np.exp(-task_fail_rate[0]) + weight * np.exp(-standard_list[0]))
        reward.append(np.exp(-task_fail_rate[1]) + weight * np.exp(-standard_list[1]))
        return reward

    def _init_cluster(self, current_time: float) -> Tuple[List[Master], Cloud]:
        """按配置初始化多 master 集群与 cloud。"""
        num_masters = int(self.cfg.get("num_masters", 2))
        nodes_per_master = int(self.cfg.get("nodes_per_master", 3))
        masters: List[Master] = []
        # 节点资源模式（循环使用）
        patterns = [(100.0, 4.0), (200.0, 6.0), (100.0, 8.0)]
        # 任务数据路径
        task_path1 = self.cfg.get("data_task_1", "data/Task_1.csv")
        task_path2 = self.cfg.get("data_task_2", "data/Task_2.csv")
        for m in range(num_masters):
            node_list = []
            for i in range(nodes_per_master):
                cpu, mem = patterns[i % len(patterns)]
                node_list.append(Node(cpu, mem, [], []))
            # 交替使用两份任务数据
            task_path = task_path1 if m % 2 == 0 else task_path2
            all_task = get_all_task(task_path)
            masters.append(Master(200.0, 8.0, node_list, [], all_task, 0, 0, 0, [0] * self.MAX_TASK_TYPE, [0] * self.MAX_TASK_TYPE))
        cloud = Cloud([], [], sys.maxsize, sys.maxsize)

        # 根据 deploy_state 初始化每个节点的 docker（按服务类型）
        deploy_state = self.cfg.get("deploy_state")
        total_nodes = num_masters * nodes_per_master
        if deploy_state is None:
            # 默认：全部部署为 1（大小为 total_nodes x MAX_TASK_TYPE）
            deploy_state = [[1] * self.MAX_TASK_TYPE for _ in range(total_nodes)]
        for i in range(total_nodes):
            for ii in range(self.MAX_TASK_TYPE):
                decision = deploy_state[i][ii]
                if decision != 1:
                    continue
                master_id = i // nodes_per_master
                local_node = i % nodes_per_master
                docker = Docker(self.POD_MEM * self.service_coeff[ii], self.POD_CPU * self.service_coeff[ii], current_time, ii, [-1])
                masters[master_id].node_list[local_node].service_list.append(docker)

        return masters, cloud

    def run(self,
            RUN_TIMES: int,
            BREAK_POINT: int,
            TRAIN_TIMES: int,
            CHO_CYCLE: int,
            epsilon: float = 0.5,
            gamma: float = 0.9) -> Dict[str, Any]:
        """执行仿真。
        返回：汇总指标字典（吞吐、响应率等），同时将资源使用与指标输出写入 results/ 与 outputs/。
        """
        throughput_list = []
        episode_rewards = []
        order_response_rate_episode = []

        for n_iter in range(int(RUN_TIMES)):
            self.logger.info(f"Run {n_iter+1}/{RUN_TIMES} starting...")
            batch_reward = []
            cur_time = 0.0
            order_response_rates = []
            pre_done = [0] * int(self.cfg.get("num_masters", 2))
            pre_undone = [0] * int(self.cfg.get("num_masters", 2))
            context = [1] * int(self.cfg.get("num_masters", 2))

            # 初始化集群
            masters, cloud = self._init_cluster(current_time=cur_time)
            num_masters = int(self.cfg.get("num_masters", 2))
            nodes_per_master = int(self.cfg.get("nodes_per_master", 3))
            total_nodes = num_masters * nodes_per_master
            deploy_state = self.cfg.get("deploy_state")
            if deploy_state is None:
                deploy_state = [[1]*self.MAX_TASK_TYPE for _ in range(total_nodes)]

            # 主循环
            for slot in range(int(BREAK_POINT)):
                cur_time += self.SLOT_TIME

                # 记录当前资源使用（多 master）
                self.usage_tracker.append_slot_multi(masters, cur_time)

                # 周期性进行一次决策（CHO_CYCLE）
                if slot % int(CHO_CYCLE) == 0 and slot != 0:
                    if len(batch_reward) > 0:
                        pass

                # 更新任务队列（按时间）
                for m_idx in range(num_masters):
                    masters[m_idx] = update_task_queue(masters[m_idx], cur_time, m_idx)

                # 从每个 master 取一个当前任务（若有）
                curr_tasks = []
                for m_idx in range(num_masters):
                    if len(masters[m_idx].task_queue) != 0:
                        task = masters[m_idx].task_queue[0]
                        del masters[m_idx].task_queue[0]
                    else:
                        task = [-1]
                    curr_tasks.append(task)

                # 构造 ava_node（可用节点列表，全局索引 + cloud=total_nodes）
                ava_node = build_ava_nodes(deploy_state, curr_tasks)

                # 构造 s_grid（每个 master 一组）
                from flexload.core.state import build_s_grid_multi
                s_grid = build_s_grid_multi(masters, deploy_state)

                # 选择动作
                act, valid_action_prob_mat, policy_state, action_choosen_mat, curr_state_value, curr_neighbor_mask, next_state_ids = \
                    select_action(self.model_name, s_grid, ava_node, context=context, epsilon=epsilon, cfg=self.cfg,
                                  algos=self.algos, usage=self.usage_tracker, logger=self.logger)

                # 将当前任务按动作入队
                for i in range(len(act)):
                    if curr_tasks[i][0] == -1:
                        continue
                    a = act[i]
                    if a == total_nodes:
                        cloud.task_queue.append(curr_tasks[i])
                    elif 0 <= a < total_nodes:
                        master_id = a // nodes_per_master
                        local_node = a % nodes_per_master
                        masters[master_id].node_list[local_node].task_queue.append(curr_tasks[i])
                    else:
                        self.logger.debug(f"Ignore invalid action: {a}")

                # 更新各 master 的 docker 执行状态
                for m_idx in range(num_masters):
                    for i in range(nodes_per_master):
                        masters[m_idx].node_list[i], undone, done, done_kind, undone_kind = update_docker(
                            masters[m_idx].node_list[i], cur_time, self.service_coeff, self.POD_CPU, self.POD_MEM)
                        for j in range(len(done_kind)):
                            masters[m_idx].done_kind[done_kind[j]] += 1
                        for j in range(len(undone_kind)):
                            masters[m_idx].undone_kind[undone_kind[j]] += 1
                        masters[m_idx].undone += undone[m_idx] if m_idx < len(undone) else 0
                        masters[m_idx].done += done[m_idx] if m_idx < len(done) else 0

                cloud, undone, done, done_kind, undone_kind = update_docker(
                    cloud, cur_time, self.service_coeff, self.POD_CPU, self.POD_MEM)
                # Cloud 的统计汇总到各 master（保持原逻辑简单化）
                for idx in range(num_masters):
                    masters[idx].undone += undone[idx] if idx < len(undone) else 0
                    masters[idx].done += done[idx] if idx < len(done) else 0

                # 计算即时奖励与指标
                cur_done = [masters[i].done - pre_done[i] for i in range(num_masters)]
                cur_undone = [masters[i].undone - pre_undone[i] for i in range(num_masters)]
                pre_done = [masters[i].done for i in range(num_masters)]
                pre_undone = [masters[i].undone for i in range(num_masters)]

                # reward：对每个 master 计算标准差与失败率
                immediate_reward = []
                weight = 1.0
                for m_idx in range(num_masters):
                    all_task = float(cur_done[m_idx] + cur_undone[m_idx])
                    fail_task = float(cur_undone[m_idx])
                    task_fail_rate = (fail_task / all_task) if all_task != 0 else 0.0
                    use_rates = []
                    for i in range(nodes_per_master):
                        use_rates.append(masters[m_idx].node_list[i].cpu / masters[m_idx].node_list[i].cpu_max)
                        use_rates.append(masters[m_idx].node_list[i].mem / masters[m_idx].node_list[i].mem_max)
                    # ddof=1 需至少 2 个样本
                    std_val = float(np.std(use_rates, ddof=1)) if len(use_rates) > 1 else float(np.std(use_rates))
                    immediate_reward.append(np.exp(-task_fail_rate) + weight * np.exp(-std_val))
                batch_reward.append(immediate_reward)

                # 汇总响应率（总完成/总完成+总失败）
                total_cur_done = float(sum(cur_done))
                total_cur_undone = float(sum(cur_undone))
                if (total_cur_done + total_cur_undone) != 0:
                    order_response_rates.append(float(total_cur_done / (total_cur_done + total_cur_undone)))
                else:
                    order_response_rates.append(0.0)

            # 汇总一次 run 的指标
            episode_reward = float(np.sum(batch_reward[1:])) if len(batch_reward) > 1 else float(np.sum(batch_reward))
            episode_rewards.append(episode_reward)
            n_iter_order_response_rate = float(np.mean(order_response_rates[1:])) if len(order_response_rates) > 1 else float(np.mean(order_response_rates))
            order_response_rate_episode.append(n_iter_order_response_rate)

            all_done = sum([m.done for m in masters])
            all_undone = sum([m.undone for m in masters])
            all_number = all_done + all_undone
            throughput = float(all_done / all_number) if all_number > 0 else 0.0
            throughput_list.append(throughput)
            self.logger.info(f"Run {n_iter+1} throughput={throughput:.4f}, achieve={all_done}, fail={all_undone}")

        # 写入结果文件
        model_tag = self.model_name
        usage_path = f"{self.results_dir}/node_resource_usage_{model_tag}.json"
        self.usage_tracker.save_json(usage_path)

        # 输出 summary
        summary = {
            "model": self.model_name,
            "run_times": int(RUN_TIMES),
            "break_point": int(BREAK_POINT),
            "cho_cycle": int(CHO_CYCLE),
            "throughput_list": throughput_list,
            "order_response_rate_episode": order_response_rate_episode,
            "episode_rewards": episode_rewards,
        }
        with open(f"{self.outputs_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, separators=(",", ":"))

        return summary
