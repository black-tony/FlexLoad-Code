# -*- coding: utf-8 -*-
import time
import json
import logging
import sys
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from scheduling_util.env.platform import Node, Master, Cloud, Docker
from scheduling_util.env.env_run import (
    get_all_task,
    update_task_queue,
    check_queue,
    update_docker,
)

from flexload.core.state import build_s_grid, build_ava_nodes
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

    def _init_cluster(self, current_time: float) -> Tuple[Master, Master, Cloud]:
        """初始化两个 master 集群与 cloud，按配置节点资源。"""
        # 节点资源（如需可从配置里逐个节点设置，这里给出默认值）
        node1_1 = Node(100.0, 4.0, [], [])
        node1_2 = Node(200.0, 6.0, [], [])
        node1_3 = Node(100.0, 8.0, [], [])
        node_list1 = [node1_1, node1_2, node1_3]

        node2_1 = Node(200.0, 8.0, [], [])
        node2_2 = Node(100.0, 2.0, [], [])
        node2_3 = Node(200.0, 6.0, [], [])
        node_list2 = [node2_1, node2_2, node2_3]

        # 任务数据路径
        task_path1 = self.cfg.get("data_task_1", "data/Task_1.csv")
        task_path2 = self.cfg.get("data_task_2", "data/Task_2.csv")
        all_task1 = get_all_task(task_path1)
        all_task2 = get_all_task(task_path2)

        master1 = Master(200.0, 8.0, node_list1, [], all_task1, 0, 0, 0, [0] * self.MAX_TASK_TYPE, [0] * self.MAX_TASK_TYPE)
        master2 = Master(200.0, 8.0, node_list2, [], all_task2, 0, 0, 0, [0] * self.MAX_TASK_TYPE, [0] * self.MAX_TASK_TYPE)
        cloud = Cloud([], [], sys.maxsize, sys.maxsize)

        # 根据 deploy_state 初始化每个节点的 docker（按服务类型）
        deploy_state = self.cfg.get("deploy_state")
        if deploy_state is None:
            # 默认：全部部署为 1（沿用 notebook 中的全 1 设置）
            deploy_state = [[1] * self.MAX_TASK_TYPE for _ in range(6)]

        for i in range(6):
            for ii in range(self.MAX_TASK_TYPE):
                decision = deploy_state[i][ii]
                if decision != 1:
                    continue
                if i < 3:
                    j = i
                    docker = Docker(self.POD_MEM * self.service_coeff[ii], self.POD_CPU * self.service_coeff[ii], current_time, ii, [-1])
                    master1.node_list[j].service_list.append(docker)
                else:
                    j = i - 3
                    docker = Docker(self.POD_MEM * self.service_coeff[ii], self.POD_CPU * self.service_coeff[ii], current_time, ii, [-1])
                    master2.node_list[j].service_list.append(docker)

        return master1, master2, cloud

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
            pre_done = [0, 0]
            pre_undone = [0, 0]
            context = [1, 1]

            # 初始化集群
            master1, master2, cloud = self._init_cluster(current_time=cur_time)
            deploy_state = self.cfg.get("deploy_state", [[1]*self.MAX_TASK_TYPE for _ in range(6)])
            if deploy_state is None:
                deploy_state = [[1]*self.MAX_TASK_TYPE for _ in range(6)]

            # 主循环
            for slot in range(int(BREAK_POINT)):
                cur_time += self.SLOT_TIME

                # 记录当前资源使用
                self.usage_tracker.append_slot(master1, master2, cur_time)

                # 周期性进行一次决策（CHO_CYCLE）
                if slot % int(CHO_CYCLE) == 0 and slot != 0:
                    if len(batch_reward) > 0:
                        # 这里采用均值作为这一周期的奖励记录
                        pass

                # 更新任务队列（按时间）
                master1 = update_task_queue(master1, cur_time, 0)
                master2 = update_task_queue(master2, cur_time, 1)

                # 取当前两个任务（若有）
                task1 = master1.task_queue[0] if len(master1.task_queue) != 0 else [-1]
                if len(master1.task_queue) != 0:
                    del master1.task_queue[0]
                task2 = master2.task_queue[0] if len(master2.task_queue) != 0 else [-1]
                if len(master2.task_queue) != 0:
                    del master2.task_queue[0]
                curr_task = [task1, task2]

                # 构造 ava_node（可用节点列表）
                ava_node = build_ava_nodes(deploy_state, curr_task)

                # 构造 s_grid（注意第二组 CPU/MEM 使用 master2 的）
                s_grid = build_s_grid(master1, master2, deploy_state)

                # 选择动作
                act, valid_action_prob_mat, policy_state, action_choosen_mat, curr_state_value, curr_neighbor_mask, next_state_ids = \
                    select_action(self.model_name, s_grid, ava_node, context=context, epsilon=epsilon, cfg=self.cfg,
                                  algos=self.algos, usage=self.usage_tracker, logger=self.logger)

                # 将当前任务按动作入队
                for i in range(len(act)):
                    if curr_task[i][0] == -1:
                        continue
                    a = act[i]
                    if a == 6:
                        cloud.task_queue.append(curr_task[i])
                    elif 0 <= a < 3:
                        master1.node_list[a].task_queue.append(curr_task[i])
                    elif 3 <= a < 6:
                        master2.node_list[a - 3].task_queue.append(curr_task[i])
                    else:
                        # 非法动作忽略
                        self.logger.debug(f"Ignore invalid action: {a}")

                # 更新各节点 docker 执行状态
                for i in range(3):
                    master1.node_list[i], undone, done, done_kind, undone_kind = update_docker(
                        master1.node_list[i], cur_time, self.service_coeff, self.POD_CPU, self.POD_MEM)
                    for j in range(len(done_kind)):
                        master1.done_kind[done_kind[j]] += 1
                    for j in range(len(undone_kind)):
                        master1.undone_kind[undone_kind[j]] += 1
                    master1.undone += undone[0]
                    master2.undone += undone[1]
                    master1.done += done[0]
                    master2.done += done[1]

                    master2.node_list[i], undone, done, done_kind, undone_kind = update_docker(
                        master2.node_list[i], cur_time, self.service_coeff, self.POD_CPU, self.POD_MEM)
                    for j in range(len(done_kind)):
                        master1.done_kind[done_kind[j]] += 1
                    for j in range(len(undone_kind)):
                        master1.undone_kind[undone_kind[j]] += 1
                    master1.undone += undone[0]
                    master2.undone += undone[1]
                    master1.done += done[0]
                    master2.done += done[1]

                cloud, undone, done, done_kind, undone_kind = update_docker(
                    cloud, cur_time, self.service_coeff, self.POD_CPU, self.POD_MEM)
                master1.undone += undone[0]
                master2.undone += undone[1]
                master1.done += done[0]
                master2.done += done[1]

                # 计算即时奖励与指标
                cur_done = [master1.done - pre_done[0], master2.done - pre_done[1]]
                cur_undone = [master1.undone - pre_undone[0], master2.undone - pre_undone[1]]
                pre_done = [master1.done, master2.done]
                pre_undone = [master1.undone, master2.undone]

                immediate_reward = Simulator.calculate_reward(master1, master2, cur_done, cur_undone)
                batch_reward.append(immediate_reward)

                if (sum(cur_done) + sum(cur_undone)) != 0:
                    order_response_rates.append(float(sum(cur_done) / (sum(cur_done) + sum(cur_undone))))
                else:
                    order_response_rates.append(0.0)

            # 汇总一次 run 的指标
            episode_reward = float(np.sum(batch_reward[1:])) if len(batch_reward) > 1 else float(np.sum(batch_reward))
            episode_rewards.append(episode_reward)
            n_iter_order_response_rate = float(np.mean(order_response_rates[1:])) if len(order_response_rates) > 1 else float(np.mean(order_response_rates))
            order_response_rate_episode.append(n_iter_order_response_rate)

            all_number = (master1.done + master2.done) + (master1.undone + master2.undone)
            throughput = float((master1.done + master2.done) / all_number) if all_number > 0 else 0.0
            throughput_list.append(throughput)
            self.logger.info(f"Run {n_iter+1} throughput={throughput:.4f}, achieve={master1.done + master2.done}, fail={master1.undone + master2.undone}")

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
