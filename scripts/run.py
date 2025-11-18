# -*- coding: utf-8 -*-
import argparse
import sys
import logging

# 允许通过脚本直接运行 src 包
import os
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(CURR_DIR, '..', 'src'))
BASE_DIR = os.path.normpath(os.path.join(CURR_DIR, '..'))
for p in (SRC_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from flexload.utils.config import load_config
from flexload.utils.logging import init_logging, ensure_dir
from flexload.utils.seeding import seed_all
from flexload.core.simulator import Simulator


def parse_args():
    parser = argparse.ArgumentParser(description="FlexLoad 仿真 CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径（YAML/JSON）")
    parser.add_argument("--model", type=str, default=None, help="调度算法：KaiS/UCB/DQN/generic/Random/UCB_predict")
    parser.add_argument("--run_times", type=int, default=1, help="运行次数")
    parser.add_argument("--break_point", type=int, default=1000, help="总 slot 数")
    parser.add_argument("--train_times", type=int, default=50, help="训练步数（保留占位）")
    parser.add_argument("--cho_cycle", type=int, default=100, help="决策周期（每多少个 slot 做一次策略学习占位）")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--network_trace", type=str, default=None, help="网络 trace CSV 路径（可选），提供则覆盖 cfg.network.trace_path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    # 网络 trace 覆盖（可选）
    if args.network_trace:
        cfg.setdefault("network", {})
        cfg["network"]["trace_path"] = args.network_trace
    # 固定随机种子
    seed_all(args.seed, deterministic=True)
    # 切换到仓库根目录，确保相对路径（data/、results/ 等）可用
    os.chdir(BASE_DIR)
    # 日志
    logger = init_logging(cfg.get("LOGS_DIR", "logs"), level=logging.INFO, name="flexload")
    logger.info(f"Config loaded from {args.config}")
    # 目录准备
    ensure_dir(cfg.get("RESULTS_DIR", "results"))
    ensure_dir(cfg.get("OUTPUTS_DIR", "outputs"))

    # 运行仿真
    sim = Simulator(cfg, logger=logger)
    summary = sim.run(RUN_TIMES=args.run_times,
                      BREAK_POINT=args.break_point,
                      TRAIN_TIMES=args.train_times,
                      CHO_CYCLE=args.cho_cycle,
                      epsilon=float(cfg.get("epsilon", 0.5)),
                      gamma=0.9)
    logger.info(f"Simulation finished. summary={summary}")


if __name__ == "__main__":
    sys.exit(main())
