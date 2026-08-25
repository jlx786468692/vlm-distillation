"""
维度3：部署效率评估
====================

测量学生模型的部署相关指标：
  - 参数量（总/可训练）
  - 模型目录磁盘大小（MB）
  - 峰值显存（GB，推理一次后 torch.cuda.max_memory_allocated）
  - 平均延迟（秒/样本，贪婪生成）
  - 吞吐（样本/秒）
"""

import os
import time
from pathlib import Path
from typing import Any, Dict

import torch


def _dir_size_mb(path: str) -> float:
    if not path or not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)


def evaluate(inferencer, records, images_root: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """运行部署效率评估。"""
    params = inferencer.count_parameters()
    disk_mb = _dir_size_mb(inferencer.model_path)

    # 计时样本数
    n_bench = min(int(cfg.get("benchmark_samples", 10)), len(records))

    latencies = []
    use_cuda = torch.cuda.is_available() and inferencer.device.startswith("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(inferencer.device)

    for i in range(n_bench):
        rec = records[i]
        img = _resolve_image(rec, images_root)
        if not img:
            continue
        q = rec.get("question", "")
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        inferencer.infer(img, q)
        if use_cuda:
            torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    avg_latency = sum(latencies) / max(len(latencies), 1)
    throughput = len(latencies) / max(sum(latencies), 1e-9)

    peak_vram_gb = None
    if use_cuda:
        peak_vram_gb = round(
            torch.cuda.max_memory_allocated(inferencer.device) / (1024 ** 3), 3
        )

    return {
        "dimension": "deployment_efficiency",
        "parameter_count": params,
        "total_parameters": params.get("total"),
        "trainable_parameters": params.get("trainable"),
        "model_dir_size_mb": disk_mb,
        "peak_vram_gb": peak_vram_gb,
        "avg_latency_seconds": round(avg_latency, 4),
        "throughput_samples_per_second": round(throughput, 4),
        "benchmark_samples": len(latencies),
    }


def _resolve_image(rec, images_root: str):
    p = rec.get("image_path", "")
    iid = rec.get("image_id")
    candidates = []
    if p:
        candidates.append(p)
        candidates.append(os.path.join(images_root, os.path.basename(p)))
    if iid is not None:
        try:
            candidates.append(os.path.join(images_root, f"COCO_val2014_{int(iid):012d}.jpg"))
        except (ValueError, TypeError):
            pass
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None
