# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def init_logging(log_dir: str, level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    ensure_dir(log_dir)
    logger_name = name or "flexload"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not logger.handlers:
        # 控制台
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(ch_fmt)
        logger.addHandler(ch)
        # 文件
        fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
        fh.setLevel(level)
        fh_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)
    return logger
