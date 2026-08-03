"""
数据分区存储器（基于分数阈值）
================================

实现清洗后数据的自动分区存储：
1. clean_valid/：final_score ≥ 70（合格样本）
2. need_fix/：40 ≤ final_score < 70（待修复样本）
3. discard/：final_score < 40（丢弃样本）

使用方式：
    from src.cleaning.data_partitioner import DataPartitioner

    partitioner = DataPartitioner(config)
    partitioner.partition(samples)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging


class DataPartitioner:
    """
    数据分区存储器（基于分数阈值）

    自动分类样本到三个分区：clean_valid、need_fix、discard
    - clean_valid: final_score ≥ 70
    - need_fix: 40 ≤ final_score < 70
    - discard: final_score < 40
    """

    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None, output_dir: Optional[str] = None):
        """
        初始化数据分区存储器

        Args:
            config: 配置管理器（从配置文件读取路径和阈值）
            logger: 日志记录器（可选，如果不提供则使用模块默认logger）
            output_dir: 输出目录路径（可选，覆盖配置文件中的路径）
        """
        # 🔧 修复：优先使用传入的 logger，确保日志统一
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config

        # ───────────────────────────────────────────────────────
        # 从配置文件读取分区路径
        # ───────────────────────────────────────────────────────
        if config:
            # 读取分区阈值
            self.clean_valid_threshold = config.get('cleaning.clean_valid_threshold', 70)
            self.need_fix_threshold = config.get('cleaning.need_fix_threshold', 40)

            # 读取分区路径（优先使用 output_dir 参数）
            if output_dir:
                base_dir = Path(output_dir)
                self.clean_valid_dir = base_dir / 'clean_valid'
                self.need_fix_dir = base_dir / 'need_fix'
                self.discard_dir = base_dir / 'discard'
            else:
                # 🔧 修复：从 cleaning.output 读取路径配置
                self.clean_valid_dir = Path(config.get('cleaning.output.clean_valid_dir', './outputs/cleaned/clean_valid'))
                self.need_fix_dir = Path(config.get('cleaning.output.need_fix_dir', './outputs/cleaned/need_fix'))
                self.discard_dir = Path(config.get('cleaning.output.discard_dir', './outputs/cleaned/discard'))
        else:
            # 默认路径（优先使用 output_dir 参数）
            if output_dir:
                base_dir = Path(output_dir)
                self.clean_valid_dir = base_dir / 'clean_valid'
                self.need_fix_dir = base_dir / 'need_fix'
                self.discard_dir = base_dir / 'discard'
            else:
                self.clean_valid_dir = Path("./outputs/cleaned/clean_valid")
                self.need_fix_dir = Path("./outputs/cleaned/need_fix")
                self.discard_dir = Path("./outputs/cleaned/discard")

            # 默认阈值
            self.clean_valid_threshold = 70
            self.need_fix_threshold = 40

        # 创建分区目录
        self._create_directories()

        # 统计信息
        self.stats = {
            'clean_valid_count': 0,
            'need_fix_count': 0,
            'discard_count': 0,
            'discard_reasons': {}
        }

        # 详细分类记录
        self.classification_details = []

        self.logger.info("✓ 数据分区存储器初始化完成")
        self.logger.info(f"  - 合格样本（≥{self.clean_valid_threshold}）: {self.clean_valid_dir}")
        self.logger.info(f"  - 待修复样本（{self.need_fix_threshold}~{self.clean_valid_threshold-1}）: {self.need_fix_dir}")
        self.logger.info(f"  - 丢弃样本（<{self.need_fix_threshold}）: {self.discard_dir}")

    def _create_directories(self):
        """创建分区目录"""
        self.clean_valid_dir.mkdir(parents=True, exist_ok=True)
        self.need_fix_dir.mkdir(parents=True, exist_ok=True)
        self.discard_dir.mkdir(parents=True, exist_ok=True)

    def partition(
        self,
        samples: List[Dict[str, Any]],
        cleaning_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        分区存储样本（基于 final_score）

        Args:
            samples: 样本列表（必须包含 final_score 字段）
            cleaning_metadata: 清洗元数据（可选）

        Returns:
            分区报告
        """
        self.logger.info(f"\n开始分区存储：{len(samples)} 个样本")

        # 重置统计
        self.stats = {
            'clean_valid_count': 0,
            'need_fix_count': 0,
            'discard_count': 0,
            'discard_reasons': {}
        }

        # 重置详细分类记录
        self.classification_details = []

        # 处理每个样本
        for sample in samples:
            partition, reason, details = self._classify_sample(sample)
            self._save_sample(sample, partition, reason)

            # 记录详细分类信息
            quality_score = sample.get('quality_score', {})
            self.classification_details.append({
                'image_id': sample.get('image_id', sample.get('id', 'unknown')),
                'partition': partition,
                'reason': reason,
                'details': details,
                'final_score': quality_score.get('final_score', 0),
                'rule_score': quality_score.get('rule_score', 0),
                'judge_score': quality_score.get('judge_score', 0)
            })

        # 生成分区报告
        partition_report = self._generate_partition_report(len(samples), cleaning_metadata)

        self.logger.info("\n" + "="*70)
        self.logger.info("分区存储完成")
        self.logger.info("="*70)

        # 统计
        total_count = len(samples)
        if total_count > 0:
            clean_valid_pct = self.stats['clean_valid_count'] / total_count * 100
            need_fix_pct = self.stats['need_fix_count'] / total_count * 100
            discard_pct = self.stats['discard_count'] / total_count * 100
        else:
            clean_valid_pct = 0.0
            need_fix_pct = 0.0
            discard_pct = 0.0

        self.logger.info(f"  - 合格样本（clean_valid）: {self.stats['clean_valid_count']} ({clean_valid_pct:.1f}%)")
        self.logger.info(f"  - 待修复样本（need_fix）: {self.stats['need_fix_count']} ({need_fix_pct:.1f}%)")
        self.logger.info(f"  - 丢弃样本（discard）: {self.stats['discard_count']} ({discard_pct:.1f}%)")

        return partition_report

    def _classify_sample(self, sample: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        分类样本到三个分区（基于 final_score）

        判定规则：
        1. 一票否决优先：如果 veto=true → 直接 discard
        2. 正常分区：
           - final_score ≥ 70 → clean_valid（合格样本）
           - 40 ≤ final_score < 70 → need_fix（待修复样本）
           - final_score < 40 → discard（丢弃样本）

        Args:
            sample: 样本数据（必须包含 final_score 字段）

        Returns:
            (partition, reason, details) 分区名称、原因、详细信息
        """
        details = {
            'issues': [],
            'suggestions': [],
            'quality_checks': {}
        }

        # ───────────────────────────────────────────────────────
        # 🔧 新增：一票否决优先判断
        # ───────────────────────────────────────────────────────
        quality_score = sample.get('quality_score', {})
        veto = quality_score.get('veto', False)

        if veto:
            # 一票否决：直接丢弃
            veto_reason = quality_score.get('deductions', {}).get('veto', {}).get('reason', '未知原因')
            details['issues'].append(f'⚠️ 一票否决: {veto_reason}')
            details['suggestions'].append('样本存在严重问题，直接丢弃')
            details['quality_checks']['veto'] = True
            details['quality_checks']['veto_reason'] = veto_reason

            return 'discard', f'⚠️ 一票否决: {veto_reason}', details

        # ───────────────────────────────────────────────────────
        # 直接从 quality_score 中读取 final_score（由 RewardModelScorer 计算）
        # ───────────────────────────────────────────────────────
        final_score = quality_score.get('final_score', None)

        # 如果没有 final_score，说明打分失败，丢弃样本
        if final_score is None:
            details['issues'].append('样本缺少 final_score，打分可能失败')
            details['suggestions'].append('检查 RewardModelScorer 是否正常运行')
            return 'discard', '样本缺少分数', details

        details['quality_checks']['final_score'] = final_score
        details['quality_checks']['rule_score'] = quality_score.get('rule_score', 0)
        details['quality_checks']['judge_score'] = quality_score.get('judge_score', 0)

        # ───────────────────────────────────────────────────────
        # 基于 final_score 分区
        # ───────────────────────────────────────────────────────

        if final_score >= self.clean_valid_threshold:
            # ≥70 → clean_valid
            details['issues'].append(f'质量合格（{final_score:.2f} ≥ {self.clean_valid_threshold}）')
            details['suggestions'].append('可直接用于训练')
            return 'clean_valid', f'质量合格（{final_score:.2f}分）', details

        elif final_score >= self.need_fix_threshold:
            # 40~69 → need_fix
            details['issues'].append(f'质量中等（{self.need_fix_threshold} ≤ {final_score:.2f} < {self.clean_valid_threshold}）')
            details['suggestions'].append('需要人工复核或自动修复')
            return 'need_fix', f'质量中等（{final_score:.2f}分）', details

        else:
            # <40 → discard
            details['issues'].append(f'质量过低（{final_score:.2f} < {self.need_fix_threshold}）')
            details['suggestions'].append('建议直接丢弃或重新生成')
            return 'discard', f'质量过低（{final_score:.2f}分）', details

    def _save_sample(self, sample: Dict[str, Any], partition: str, reason: str):
        """
        保存样本到对应分区

        Args:
            sample: 样本数据
            partition: 分区名称
            reason: 分类原因
        """
        # 获取样本ID
        image_id = sample.get('image_id', sample.get('id', 'unknown'))

        # 确定保存目录
        if partition == 'clean_valid':
            save_dir = self.clean_valid_dir
            self.stats['clean_valid_count'] += 1
            # clean_valid样本不添加额外字段
            sample_to_save = sample

        elif partition == 'need_fix':
            save_dir = self.need_fix_dir
            self.stats['need_fix_count'] += 1
            # need_fix样本添加过滤原因
            sample_to_save = {**sample, '_removal_reason': reason}

        else:  # discard
            save_dir = self.discard_dir
            self.stats['discard_count'] += 1

            # 记录丢弃原因
            if reason not in self.stats['discard_reasons']:
                self.stats['discard_reasons'][reason] = 0
            self.stats['discard_reasons'][reason] += 1

            # discard样本添加过滤原因
            sample_to_save = {**sample, '_removal_reason': reason}

        # 保存样本
        output_file = save_dir / f"{image_id}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_to_save, f, indent=2, ensure_ascii=False)

    def _generate_partition_report(
        self,
        total_count: int,
        cleaning_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成分区报告

        Args:
            total_count: 总样本数
            cleaning_metadata: 清洗元数据

        Returns:
            分区报告
        """
        # 获取输出根目录（三个分区目录的父目录）
        output_dir = self.clean_valid_dir.parent

        report = {
            'timestamp': datetime.now().isoformat(),
            'partition_criteria': {
                'clean_valid_threshold': self.clean_valid_threshold,
                'need_fix_threshold': self.need_fix_threshold,
                'description': f'final_score ≥ {self.clean_valid_threshold} → clean_valid, '
                              f'{self.need_fix_threshold} ≤ final_score < {self.clean_valid_threshold} → need_fix, '
                              f'final_score < {self.need_fix_threshold} → discard'
            },
            'summary': {
                'total_samples': total_count,
                'clean_valid_count': self.stats['clean_valid_count'],
                'need_fix_count': self.stats['need_fix_count'],
                'discard_count': self.stats['discard_count'],
                'clean_rate': self.stats['clean_valid_count'] / total_count if total_count > 0 else 0,
                'discard_rate': self.stats['discard_count'] / total_count if total_count > 0 else 0
            },
            'discard_reasons': self.stats['discard_reasons'],
            'output_directories': {
                'clean_valid': str(self.clean_valid_dir),
                'need_fix': str(self.need_fix_dir),
                'discard': str(self.discard_dir)
            }
        }

        if cleaning_metadata:
            report['cleaning_metadata'] = cleaning_metadata

        # 保存报告
        report_file = output_dir / "partition_report.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"分区报告已保存：{report_file}")

        return report

    def get_fixable_samples(self) -> List[Dict[str, Any]]:
        """
        获取可修复样本

        Returns:
            待修复样本列表
        """
        fixable_samples = []

        # 加载need_fix目录中的所有样本
        for json_file in self.need_fix_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
                fixable_samples.append(sample)

        self.logger.info(f"找到 {len(fixable_samples)} 个待修复样本")
        return fixable_samples

    def move_to_clean_valid(self, sample: Dict[str, Any]):
        """
        将修复后的样本移动到clean_valid

        Args:
            sample: 修复后的样本
        """
        image_id = sample.get('image_id', sample.get('id', 'unknown'))

        # 从need_fix删除
        old_file = self.need_fix_dir / f"{image_id}.json"
        if old_file.exists():
            old_file.unlink()

        # 添加到clean_valid
        sample['partition'] = 'clean_valid'
        sample['partition_reason'] = '修复后合格'
        sample['partition_timestamp'] = datetime.now().isoformat()

        new_file = self.clean_valid_dir / f"{image_id}.json"

        with open(new_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

        self.logger.info(f"样本 {image_id} 已移动到 clean_valid")


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    partitioner = DataPartitioner()

    print("\n" + "="*70)
    print("数据分区存储器测试")
    print("="*70)

    # 测试样本
    test_samples = [
        {
            "image_id": "test_001",
            "final_score": 85.5,
            "rule_score": 78,
            "judge_score": 90
        },
        {
            "image_id": "test_002",
            "final_score": 55.2,
            "rule_score": 50,
            "judge_score": 58
        },
        {
            "image_id": "test_003",
            "final_score": 25.8,
            "rule_score": 20,
            "judge_score": 30
        }
    ]

    print("\n测试分区：")
    print("-" * 70)

    report = partitioner.partition(test_samples)

    print(f"\n分区报告：")
    print(json.dumps(report, indent=2))

    print("\n" + "="*70)
    print("分区标准：")
    print(f"  clean_valid/ - final_score ≥ {partitioner.clean_valid_threshold}")
    print(f"  need_fix/    - {partitioner.need_fix_threshold} ≤ final_score < {partitioner.clean_valid_threshold}")
    print(f"  discard/     - final_score < {partitioner.need_fix_threshold}")
    print("="*70)