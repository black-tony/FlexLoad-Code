# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Any
import numpy as np

class DecisionCostTracker:
    """
    决策阶段的壁钟时延统计器。
    用法：
      - add(sample_ms, candidate_size=None)
      - to_summary() -> {avg_ms, p95_ms, count, candidate_avg_size}
    """
    def __init__(self) -> None:
        self.samples: List[float] = []
        self.candidate_sizes: List[int] = []

    def add(self, sample_ms: float, candidate_size: Optional[int] = None) -> None:
        try:
            self.samples.append(float(sample_ms))
        except Exception:
            # 忽略异常样本
            pass
        if candidate_size is not None:
            try:
                self.candidate_sizes.append(int(candidate_size))
            except Exception:
                pass

    def to_summary(self) -> Dict[str, Any]:
        if len(self.samples) == 0:
            return {"avg_ms": 0.0, "p95_ms": 0.0, "count": 0, "candidate_avg_size": 0.0}
        arr = np.array(self.samples, dtype=np.float64)
        avg = float(np.mean(arr))
        # 为稳健处理小样本，p95 使用最近邻分位数
        try:
            p95 = float(np.percentile(arr, 95))
        except Exception:
            arr_sorted = np.sort(arr)
            idx = int(round(0.95 * (len(arr_sorted) - 1)))
            p95 = float(arr_sorted[idx])
        cand_avg = float(np.mean(self.candidate_sizes)) if len(self.candidate_sizes) > 0 else 0.0
        return {
            "avg_ms": avg,
            "p95_ms": p95,
            "count": int(len(self.samples)),
            "candidate_avg_size": cand_avg,
        }
