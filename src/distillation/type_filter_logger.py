"""
类型过滤统计日志模块
=====================

功能：
1. JSON Lines 格式日志输出
2. 每 1000 样本汇总统计
3. Level 1 实时告警

作者: Claude
日期: 2026-08-13
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading


@dataclass
class SampleStats:
    """单个样本统计"""
    sample_id: str
    gt_answer: str
    gt_types: List[str]

    # 过滤统计
    total_tokens: int
    filtered_tokens: int
    filter_rate: float

    # Level 分布
    level_1_mismatches: int
    level_2_mismatches: int
    level_3_mismatches: int
    level_4_mismatches: int

    # GT 留存
    gt_retained: bool
    gt_fallback_applied: bool

    # KL 权重
    kl_weight: float

    # 时间戳
    timestamp: str


@dataclass
class BatchStats:
    """批次统计（1000 样本）"""
    batch_id: int
    batch_size: int
    start_time: str
    end_time: str

    # 过滤统计
    total_samples: int
    filtered_samples: int
    avg_filter_rate: float

    # Level 分布
    level_1_mismatch_samples: int
    level_2_mismatch_samples: int
    level_3_mismatch_samples: int
    level_4_mismatch_samples: int

    level_1_mismatch_rate: float
    level_2_mismatch_rate: float
    level_3_mismatch_rate: float
    level_4_mismatch_rate: float

    # GT 留存
    gt_retained_count: int
    gt_retention_rate: float

    gt_fallback_count: int
    gt_fallback_rate: float

    # KL 权重分布
    avg_kl_weight: float
    kl_weight_distribution: Dict[str, int]

    # 类型过滤率 TOP-10
    type_filter_rates: Dict[str, float]

    # 告警
    alerts: List[str]


class TypeFilterLogger:
    """
    类型过滤统计日志器

    核心功能：
    1. JSON Lines 格式输出
    2. 1000 样本批量汇总
    3. Level 1 实时告警
    """

    def __init__(
        self,
        output_dir: str = "./logs/type_filter",
        batch_size: int = 1000,
        enable_realtime_alert: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        初始化日志器

        Args:
            output_dir: 日志输出目录
            batch_size: 批次大小（默认 1000）
            enable_realtime_alert: 是否启用 Level 1 实时告警
            alert_thresholds: 告警阈值配置
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.enable_realtime_alert = enable_realtime_alert

        # 告警阈值
        self.alert_thresholds = alert_thresholds or {
            'type_filter_rate_warning': 0.8,
            'safety_critical_mismatch_abort': 0.3,
            'non_driving_data_max': 0.5,
            'gt_retention_min': 0.95
        }

        # 批次统计
        self.current_batch: List[SampleStats] = []
        self.batch_id = 0

        # 累计统计
        self.total_samples = 0
        self.total_level_1_mismatches = 0
        self.total_gt_missing = 0

        # 类型过滤统计
        self.type_filter_counts: Dict[str, int] = {}
        self.type_total_counts: Dict[str, int] = {}

        # 线程锁
        self.lock = threading.Lock()

        # 日志文件
        self.sample_log_file = None
        self.batch_log_file = None
        self.alert_log_file = None

        self._init_log_files()

    def _init_log_files(self):
        """初始化日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 样本日志（JSON Lines）
        self.sample_log_file = open(
            self.output_dir / f"samples_{timestamp}.jsonl",
            'a',
            encoding='utf-8'
        )

        # 批次日志（JSON Lines）
        self.batch_log_file = open(
            self.output_dir / f"batches_{timestamp}.jsonl",
            'a',
            encoding='utf-8'
        )

        # 告警日志（JSON Lines）
        self.alert_log_file = open(
            self.output_dir / f"alerts_{timestamp}.jsonl",
            'a',
            encoding='utf-8'
        )

    def record_sample(self, sample_stats: SampleStats):
        """
        记录单个样本统计

        Args:
            sample_stats: 样本统计
        """
        with self.lock:
            # 1. 写入样本日志（JSON Lines）
            self._write_json_line(self.sample_log_file, asdict(sample_stats))

            # 2. 添加到当前批次
            self.current_batch.append(sample_stats)
            self.total_samples += 1

            # 3. 更新累计统计
            if sample_stats.level_1_mismatches > 0:
                self.total_level_1_mismatches += 1
            if not sample_stats.gt_retained:
                self.total_gt_missing += 1

            # 4. Level 1 实时告警
            if self.enable_realtime_alert and sample_stats.level_1_mismatches > 0:
                self._trigger_realtime_alert(sample_stats)

            # 5. 检查批次是否满 1000
            if len(self.current_batch) >= self.batch_size:
                self._flush_batch()

    def _write_json_line(self, file, data: dict):
        """写入 JSON Lines 格式"""
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
        file.flush()

    def _trigger_realtime_alert(self, sample_stats: SampleStats):
        """
        Level 1 实时告警

        Args:
            sample_stats: 样本统计
        """
        alert_data = {
            'alert_type': 'LEVEL_1_MISMATCH',
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_stats.sample_id,
            'gt_answer': sample_stats.gt_answer,
            'level_1_mismatches': sample_stats.level_1_mismatches,
            'kl_weight': sample_stats.kl_weight,
            'message': f"Level 1 类型不匹配，样本将被丢弃: {sample_stats.sample_id}"
        }

        # 写入告警日志
        self._write_json_line(self.alert_log_file, alert_data)

        # 同时输出到控制台（ERROR 级别）
        logging.error(
            f"[LEVEL_1_ALERT] Sample {sample_stats.sample_id}: "
            f"GT='{sample_stats.gt_answer}', "
            f"Level 1 mismatches={sample_stats.level_1_mismatches}, "
            f"KL weight={sample_stats.kl_weight}"
        )

    def _flush_batch(self):
        """批次汇总并写入"""
        if not self.current_batch:
            return

        # 1. 计算批次统计
        batch_stats = self._compute_batch_stats()

        # 2. 写入批次日志（JSON Lines）
        self._write_json_line(self.batch_log_file, asdict(batch_stats))

        # 3. 输出控制台日志
        self._log_batch_summary(batch_stats)

        # 4. 检查告警
        alerts = self._check_alerts(batch_stats)
        if alerts:
            batch_stats.alerts = alerts
            # 重新写入（包含告警）
            self.batch_log_file.seek(self.batch_log_file.tell() - len(json.dumps(asdict(batch_stats))) - 1)
            self._write_json_line(self.batch_log_file, asdict(batch_stats))

        # 5. 清空当前批次
        self.current_batch = []
        self.batch_id += 1

    def _compute_batch_stats(self) -> BatchStats:
        """计算批次统计"""
        batch = self.current_batch
        batch_size = len(batch)

        # 过滤统计
        filtered_samples = sum(1 for s in batch if s.filtered_tokens > 0)
        avg_filter_rate = sum(s.filter_rate for s in batch) / batch_size

        # Level 分布
        level_1_samples = sum(1 for s in batch if s.level_1_mismatches > 0)
        level_2_samples = sum(1 for s in batch if s.level_2_mismatches > 0)
        level_3_samples = sum(1 for s in batch if s.level_3_mismatches > 0)
        level_4_samples = sum(1 for s in batch if s.level_4_mismatches > 0)

        # GT 留存
        gt_retained_count = sum(1 for s in batch if s.gt_retained)
        gt_fallback_count = sum(1 for s in batch if s.gt_fallback_applied)

        # KL 权重分布
        kl_weight_dist = {}
        for s in batch:
            kl_str = f"{s.kl_weight:.1f}"
            kl_weight_dist[kl_str] = kl_weight_dist.get(kl_str, 0) + 1

        avg_kl_weight = sum(s.kl_weight for s in batch) / batch_size

        # 时间范围
        timestamps = [s.timestamp for s in batch]
        start_time = min(timestamps)
        end_time = max(timestamps)

        return BatchStats(
            batch_id=self.batch_id,
            batch_size=batch_size,
            start_time=start_time,
            end_time=end_time,

            total_samples=self.total_samples,
            filtered_samples=filtered_samples,
            avg_filter_rate=avg_filter_rate,

            level_1_mismatch_samples=level_1_samples,
            level_2_mismatch_samples=level_2_samples,
            level_3_mismatch_samples=level_3_samples,
            level_4_mismatch_samples=level_4_samples,

            level_1_mismatch_rate=level_1_samples / batch_size,
            level_2_mismatch_rate=level_2_samples / batch_size,
            level_3_mismatch_rate=level_3_samples / batch_size,
            level_4_mismatch_rate=level_4_samples / batch_size,

            gt_retained_count=gt_retained_count,
            gt_retention_rate=gt_retained_count / batch_size,
            gt_fallback_count=gt_fallback_count,
            gt_fallback_rate=gt_fallback_count / batch_size,

            avg_kl_weight=avg_kl_weight,
            kl_weight_distribution=kl_weight_dist,

            type_filter_rates=self._compute_type_filter_rates(batch),
            alerts=[]
        )

    def _compute_type_filter_rates(self, batch: List[SampleStats]) -> Dict[str, float]:
        """计算类型过滤率"""
        # TODO: 实现类型过滤率统计
        return {}

    def _log_batch_summary(self, batch_stats: BatchStats):
        """输出批次摘要日志"""
        logging.info("=" * 80)
        logging.info(f"[Type Filter Stats] Batch #{batch_stats.batch_id} Summary")
        logging.info("=" * 80)
        logging.info(f"  总样本数: {batch_stats.total_samples}")
        logging.info(f"  过滤样本数: {batch_stats.filtered_samples} ({batch_stats.avg_filter_rate:.1%})")
        logging.info()
        logging.info(f"  Level 1 不匹配: {batch_stats.level_1_mismatch_samples} ({batch_stats.level_1_mismatch_rate:.1%})")
        logging.info(f"  Level 2 不匹配: {batch_stats.level_2_mismatch_samples} ({batch_stats.level_2_mismatch_rate:.1%})")
        logging.info(f"  Level 3 不匹配: {batch_stats.level_3_mismatch_samples} ({batch_stats.level_3_mismatch_rate:.1%})")
        logging.info(f"  Level 4 不匹配: {batch_stats.level_4_mismatch_samples} ({batch_stats.level_4_mismatch_rate:.1%})")
        logging.info()
        logging.info(f"  GT 留存率: {batch_stats.gt_retention_rate:.1%}")
        logging.info(f"  GT 兜底率: {batch_stats.gt_fallback_rate:.1%}")
        logging.info(f"  平均 KL 权重: {batch_stats.avg_kl_weight:.3f}")
        logging.info("=" * 80)

    def _check_alerts(self, batch_stats: BatchStats) -> List[str]:
        """检查告警条件"""
        alerts = []

        # Level 1 不匹配率 > 30%
        if batch_stats.level_1_mismatch_rate > self.alert_thresholds['safety_critical_mismatch_abort']:
            alert = f"⚠️ Level 1 不匹配率 {batch_stats.level_1_mismatch_rate:.1%} > 30%，建议终止"
            alerts.append(alert)
            logging.error(alert)

        # GT 留存率 < 95%
        if batch_stats.gt_retention_rate < self.alert_thresholds['gt_retention_min']:
            alert = f"⚠️ GT 留存率 {batch_stats.gt_retention_rate:.1%} < 95%，过滤可能过严"
            alerts.append(alert)
            logging.warning(alert)

        # 平均过滤率 > 80%
        if batch_stats.avg_filter_rate > self.alert_thresholds['type_filter_rate_warning']:
            alert = f"⚠️ 平均过滤率 {batch_stats.avg_filter_rate:.1%} > 80%，检查模型或配置"
            alerts.append(alert)
            logging.warning(alert)

        return alerts

    def finalize(self):
        """最终汇总（处理剩余样本）"""
        if self.current_batch:
            self._flush_batch()

        # 关闭文件
        if self.sample_log_file:
            self.sample_log_file.close()
        if self.batch_log_file:
            self.batch_log_file.close()
        if self.alert_log_file:
            self.alert_log_file.close()

        # 输出最终统计
        self._log_final_stats()

    def _log_final_stats(self):
        """输出最终统计"""
        logging.info("=" * 80)
        logging.info("[Type Filter] Final Statistics")
        logging.info("=" * 80)
        logging.info(f"  总样本数: {self.total_samples}")
        logging.info(f"  Level 1 不匹配样本数: {self.total_level_1_mismatches}")
        logging.info(f"  Level 1 不匹配率: {self.total_level_1_mismatches / max(self.total_samples, 1):.1%}")
        logging.info(f"  GT 缺失样本数: {self.total_gt_missing}")
        logging.info(f"  GT 缺失率: {self.total_gt_missing / max(self.total_samples, 1):.1%}")
        logging.info("=" * 80)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import time

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 1. 初始化日志器
    logger = TypeFilterLogger(
        output_dir="./logs/type_filter",
        batch_size=10,  # 测试用，设置为 10
        enable_realtime_alert=True
    )

    # 2. 模拟 25 个样本
    for i in range(25):
        sample_stats = SampleStats(
            sample_id=f"sample_{i:04d}",
            gt_answer="pedestrian",
            gt_types=["driving_pedestrian"],
            total_tokens=10,
            filtered_tokens=3,
            filter_rate=0.3,
            level_1_mismatches=1 if i % 10 == 0 else 0,  # 每 10 个样本一个 Level 1 不匹配
            level_2_mismatches=2 if i % 5 == 0 else 0,
            level_3_mismatches=1 if i % 3 == 0 else 0,
            level_4_mismatches=0,
            gt_retained=True,
            gt_fallback_applied=False,
            kl_weight=0.1 if i % 10 == 0 else 1.0,
            timestamp=datetime.now().isoformat()
        )

        logger.record_sample(sample_stats)
        time.sleep(0.01)  # 模拟处理间隔

    # 3. 最终汇总
    logger.finalize()

    print("\n日志文件已保存到: ./logs/type_filter/")