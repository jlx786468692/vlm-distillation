"""
置信度占比限流控制器
===================

**重要：此模块仅在数据验证阶段（quality_validation）使用，不在清洗阶段使用。**

清洗阶段（data_partitioner）职责：
- 置信度 < 0.4：强制 discard
- 置信度 0.4~0.95：正常保留
- 置信度 0.95~0.98：标记为 high，不丢弃
- 置信度 > 0.98：标记为 ultra_high，不丢弃

数据验证阶段（confidence_controller）职责：
- 计算ECE（Expected Calibration Error）
- 根据ECE动态调整置信度占比上限
- 剔除超限的高置信/超高置信样本

使用方式：
    # 在 quality_validation 阶段
    from src.cleaning.confidence_controller import ConfidenceController

    controller = ConfidenceController(config)
    result = controller.control_confidence_distribution(
        samples=cleaned_samples,
        output_dir='./outputs/cleaned',
        ece_score=ece_score
    )
"""

import json
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging


class ConfidenceController:
    """
    置信度占比限流控制器

    在 quality_validation 阶段运行，确保数据集置信度分布均衡
    """

    def __init__(self, config: Any = None):
        """
        初始化置信度控制器

        Args:
            config: 配置管理器（从配置文件读取阈值）
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # 🔧 从配置文件读取置信度阈值（与 data_partitioner 保持一致）
        if config:
            # 黄金区间
            medium_zone = config.get('cleaning.confidence.medium_zone', [0.4, 0.95])
            self.min_confidence = medium_zone[0]
            self.max_confidence = medium_zone[1]

            # 超高置信度阈值
            self.ultra_high_conf = config.get('cleaning.confidence.ultra_high_conf', 0.98)

            # 占比上限
            self.max_high_conf_ratio = config.get(
                'cleaning.confidence_control.max_high_conf_ratio',
                0.35
            )
            self.max_ultra_high_conf_ratio = config.get(
                'cleaning.confidence_control.max_ultra_high_conf_ratio',
                0.15
            )

            # ECE 阈值
            self.ECE_LOW_THRESHOLD = config.get(
                'cleaning.confidence_control.ece_low_threshold',
                0.1
            )
            self.ECE_HIGH_THRESHOLD = config.get(
                'cleaning.confidence_control.ece_high_threshold',
                0.2
            )
        else:
            # 默认值
            self.min_confidence = 0.4
            self.max_confidence = 0.95
            self.ultra_high_conf = 0.98
            self.max_high_conf_ratio = 0.35
            self.max_ultra_high_conf_ratio = 0.15
            self.ECE_LOW_THRESHOLD = 0.1
            self.ECE_HIGH_THRESHOLD = 0.2

        self.logger.info("✓ 置信度控制器初始化完成")
        self.logger.info(f"  - 置信度黄金区间: [{self.min_confidence}, {self.max_confidence}]")
        self.logger.info(f"  - 超高置信度阈值: {self.ultra_high_conf}")
        self.logger.info(f"  - 高置信度（>{self.max_confidence}）占比上限: {self.max_high_conf_ratio*100:.1f}%")
        self.logger.info(f"  - 超高置信度（>{self.ultra_high_conf}）占比上限: {self.max_ultra_high_conf_ratio*100:.1f}%")

    def control_confidence_distribution(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        ece_score: float = None
    ) -> Dict[str, Any]:
        """
        控制置信度分布占比

        Args:
            samples: 样本列表
            output_dir: 输出目录
            ece_score: ECE分数（可选，用于动态调整）

        Returns:
            {
                'total_samples': int,
                'removed_high_conf': int,
                'removed_ultra_high_conf': int,
                'final_high_conf_ratio': float,
                'final_ultra_high_conf_ratio': float,
                'ece_adjusted': bool
            }
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("置信度占比限流控制")
        self.logger.info("="*70)

        # ───────────────────────────────────────────────────────
        # Step 1: 统计置信度分布
        # ───────────────────────────────────────────────────────
        self.logger.info("\n[1/4] 统计置信度分布...")

        confidence_tiers = {
            'very_low': [],      # < 0.4 (已丢弃)
            'normal': [],        # 0.4 ~ 0.95
            'high': [],          # 0.95 ~ 0.98
            'ultra_high': []     # > 0.98
        }

        for idx, sample in enumerate(samples):
            vqa_data = sample.get('tasks', {}).get('vqa', {})
            inference_mode = vqa_data.get('inference_mode', 'closed')

            if inference_mode == 'closed':
                confidence = vqa_data.get('hard_label', {}).get('confidence', 0)

                # 🔧 使用配置文件中的阈值（与 data_partitioner 保持一致）
                if confidence < self.min_confidence:
                    confidence_tiers['very_low'].append(idx)
                elif confidence <= self.max_confidence:
                    confidence_tiers['normal'].append(idx)
                elif confidence <= self.ultra_high_conf:
                    confidence_tiers['high'].append(idx)
                else:
                    confidence_tiers['ultra_high'].append(idx)

        total_samples = len(samples)

        # 统计各层级占比
        very_low_count = len(confidence_tiers['very_low'])
        normal_count = len(confidence_tiers['normal'])
        high_count = len(confidence_tiers['high'])
        ultra_high_count = len(confidence_tiers['ultra_high'])

        self.logger.info(f"  - 极低置信度（< 0.4）: {very_low_count} 个（已在分区阶段丢弃）")
        self.logger.info(f"  - 正常置信度（0.4 ~ 0.95）: {normal_count} 个 ({normal_count/total_samples*100:.1f}%)")
        self.logger.info(f"  - 高置信度（0.95 ~ 0.98）: {high_count} 个 ({high_count/total_samples*100:.1f}%)")
        self.logger.info(f"  - 超高置信度（> 0.98）: {ultra_high_count} 个 ({ultra_high_count/total_samples*100:.1f}%)")

        # ───────────────────────────────────────────────────────
        # Step 2: ECE 动态调整占比上限
        # ───────────────────────────────────────────────────────
        self.logger.info("\n[2/4] ECE 动态调整占比上限...")

        max_high = self.max_high_conf_ratio
        max_ultra_high = self.max_ultra_high_conf_ratio
        ece_adjusted = False

        if ece_score is not None:
            if ece_score < self.ECE_LOW_THRESHOLD:
                # ECE 低（校准好）→ 放宽占比上限
                max_high = self.max_high_conf_ratio * 1.2  # 放宽 20%
                max_ultra_high = self.max_ultra_high_conf_ratio * 1.2
                ece_adjusted = True
                self.logger.info(f"  ✓ ECE={ece_score:.3f}（校准良好），放宽占比上限 20%")
                self.logger.info(f"    - 高置信度上限: {max_high*100:.1f}%")
                self.logger.info(f"    - 超高置信度上限: {max_ultra_high*100:.1f}%")

            elif ece_score > self.ECE_HIGH_THRESHOLD:
                # ECE 高（校准差）→ 收紧占比上限
                max_high = self.max_high_conf_ratio * 0.8  # 收紧 20%
                max_ultra_high = self.max_ultra_high_conf_ratio * 0.8
                ece_adjusted = True
                self.logger.warning(f"  ⚠️  ECE={ece_score:.3f}（校准失效），收紧占比上限 20%")
                self.logger.warning(f"    - 高置信度上限: {max_high*100:.1f}%")
                self.logger.warning(f"    - 超高置信度上限: {max_ultra_high*100:.1f}%")
                self.logger.warning(f"    - 建议：检查教师模型 Prompt/Logits 提取逻辑")
        else:
            self.logger.info(f"  - 未提供 ECE 分数，使用默认占比上限")

        # ───────────────────────────────────────────────────────
        # Step 3: 随机抽样剔除超限样本
        # ───────────────────────────────────────────────────────
        self.logger.info("\n[3/4] 随机抽样剔除超限样本...")

        removed_high = 0
        removed_ultra_high = 0
        samples_to_remove = set()

        # 检查高置信度样本是否超限
        current_high_ratio = high_count / total_samples if total_samples > 0 else 0

        if current_high_ratio > max_high:
            # 计算需要剔除的数量
            target_count = int(total_samples * max_high)
            remove_count = high_count - target_count

            # 随机抽样剔除
            random.seed(42)  # 固定种子，确保可复现
            to_remove = random.sample(confidence_tiers['high'], remove_count)
            samples_to_remove.update(to_remove)
            removed_high = remove_count

            self.logger.info(f"  - 高置信度样本超限：{current_high_ratio*100:.1f}% > {max_high*100:.1f}%")
            self.logger.info(f"  - 随机剔除 {remove_count} 个高置信度样本")

        # 检查超高置信度样本是否超限
        current_ultra_high_ratio = ultra_high_count / total_samples if total_samples > 0 else 0

        if current_ultra_high_ratio > max_ultra_high:
            # 计算需要剔除的数量
            target_count = int(total_samples * max_ultra_high)
            remove_count = ultra_high_count - target_count

            # 随机抽样剔除
            random.seed(42)
            to_remove = random.sample(confidence_tiers['ultra_high'], remove_count)
            samples_to_remove.update(to_remove)
            removed_ultra_high = remove_count

            self.logger.info(f"  - 超高置信度样本超限：{current_ultra_high_ratio*100:.1f}% > {max_ultra_high*100:.1f}%")
            self.logger.info(f"  - 随机剔除 {remove_count} 个超高置信度样本")

        # ───────────────────────────────────────────────────────
        # Step 4: 生成最终样本列表并保存报告
        # ───────────────────────────────────────────────────────
        self.logger.info("\n[4/4] 生成最终样本列表并保存报告...")

        # 过滤掉需要剔除的样本
        final_samples = [s for idx, s in enumerate(samples) if idx not in samples_to_remove]

        # 计算最终占比
        final_high_count = high_count - removed_high
        final_ultra_high_count = ultra_high_count - removed_ultra_high
        final_total = len(final_samples)

        final_high_ratio = final_high_count / final_total if final_total > 0 else 0
        final_ultra_high_ratio = final_ultra_high_count / final_total if final_total > 0 else 0

        self.logger.info(f"\n最终置信度分布：")
        self.logger.info(f"  - 总样本数：{final_total}（剔除 {len(samples_to_remove)} 个）")
        self.logger.info(f"  - 高置信度（0.95 ~ 0.98）：{final_high_count} 个 ({final_high_ratio*100:.1f}%)")
        self.logger.info(f"  - 超高置信度（> 0.98）：{final_ultra_high_count} 个 ({final_ultra_high_ratio*100:.1f}%)")

        # 保存限流报告
        report = {
            'total_samples_before': total_samples,
            'total_samples_after': final_total,
            'confidence_distribution': {
                'very_low': {'count': very_low_count, 'ratio': very_low_count/total_samples if total_samples > 0 else 0},
                'normal': {'count': normal_count, 'ratio': normal_count/total_samples if total_samples > 0 else 0},
                'high': {
                    'count': high_count,
                    'ratio_before': current_high_ratio,
                    'removed': removed_high,
                    'ratio_after': final_high_ratio,
                    'limit': max_high
                },
                'ultra_high': {
                    'count': ultra_high_count,
                    'ratio_before': current_ultra_high_ratio,
                    'removed': removed_ultra_high,
                    'ratio_after': final_ultra_high_ratio,
                    'limit': max_ultra_high
                }
            },
            'ece_score': ece_score,
            'ece_adjusted': ece_adjusted,
            'removed_indices': list(samples_to_remove)
        }

        report_file = Path(output_dir) / 'confidence_control_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n✓ 置信度限流报告已保存：{report_file}")

        return {
            'final_samples': final_samples,
            'report': report,
            'total_samples': final_total,
            'removed_high_conf': removed_high,
            'removed_ultra_high_conf': removed_ultra_high,
            'final_high_conf_ratio': final_high_ratio,
            'final_ultra_high_conf_ratio': final_ultra_high_ratio,
            'ece_adjusted': ece_adjusted
        }

    def calculate_ece(
        self,
        samples: List[Dict[str, Any]],
        ground_truth: Optional[Dict[str, str]] = None,
        num_bins: int = 10
    ) -> float:
        """
        计算 ECE（Expected Calibration Error）

        ECE 衡量模型置信度是否与正确率匹配：
        - ECE ≈ 0: 模型校准良好（高置信样本确实大概率正确）
        - ECE 较大: 模型校准失效（模型盲目自信）

        公式：
        ECE = Σ |accuracy(bin_i) - confidence(bin_i)| × |bin_i| / n

        Args:
            samples: 样本列表
            ground_truth: 真实标签字典 {image_id: correct_answer}
            num_bins: 分箱数量（默认10）

        Returns:
            ECE 分数（0-1，越小越好）
        """
        if not ground_truth:
            self.logger.warning("未提供真实标签（ground_truth），无法计算准确的ECE")
            self.logger.warning("返回估算值 0.15（中等校准）")
            self.logger.info("提示：如有真实标签，请传入 ground_truth 参数以获得准确的ECE")
            return 0.15

        # ───────────────────────────────────────────────────────
        # ECE 计算步骤
        # ───────────────────────────────────────────────────────

        # 1. 收集置信度和正确性
        confidence_correct_pairs = []

        for sample in samples:
            vqa_data = sample.get('tasks', {}).get('vqa', {})
            inference_mode = vqa_data.get('inference_mode', 'closed')

            if inference_mode == 'closed':
                # 获取模型预测
                predicted_answer = vqa_data.get('hard_label', {}).get('answer', '')
                confidence = vqa_data.get('hard_label', {}).get('confidence', 0)

                # 获取真实标签
                image_id = str(sample.get('image_id', ''))
                correct_answer = ground_truth.get(image_id, '')

                if predicted_answer and correct_answer:
                    # 判断是否正确
                    is_correct = (predicted_answer.lower().strip() ==
                                  correct_answer.lower().strip())

                    confidence_correct_pairs.append((confidence, is_correct))

        if not confidence_correct_pairs:
            self.logger.warning("没有足够的样本计算ECE")
            return 0.15

        # 2. 按置信度分箱
        bins = [[] for _ in range(num_bins)]

        for confidence, is_correct in confidence_correct_pairs:
            # 将置信度映射到bin索引
            bin_idx = min(int(confidence * num_bins), num_bins - 1)
            bins[bin_idx].append((confidence, is_correct))

        # 3. 计算每个bin的准确率和平均置信度
        ece = 0.0
        total_samples = len(confidence_correct_pairs)

        for bin_data in bins:
            if not bin_data:
                continue

            # 计算该bin的准确率
            correct_count = sum(1 for _, is_correct in bin_data if is_correct)
            accuracy = correct_count / len(bin_data)

            # 计算该bin的平均置信度
            avg_confidence = sum(conf for conf, _ in bin_data) / len(bin_data)

            # 累加ECE
            ece += abs(accuracy - avg_confidence) * len(bin_data) / total_samples

        self.logger.info(f"✓ ECE 计算完成: {ece:.4f}")
        self.logger.info(f"  - 样本数: {total_samples}")
        self.logger.info(f"  - 分箱数: {num_bins}")

        # 4. 解释ECE值
        if ece < 0.1:
            self.logger.info(f"  - 状态: 校准良好 ✓")
        elif ece < 0.2:
            self.logger.info(f"  - 状态: 校准一般")
        else:
            self.logger.warning(f"  - 状态: 校准失效 ⚠️")

        return ece


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("置信度占比限流控制器测试")
    print("="*70)

    print("\n置信度分层策略：")
    print("  - < 0.4: 极低置信度，直接丢弃")
    print("  - 0.4 ~ 0.95: 最优区间，正常保留")
    print("  - 0.95 ~ 0.98: 高置信度，占比 ≤ 35%")
    print("  - > 0.98: 超高置信度，占比 ≤ 15%")

    print("\nECE 动态调整：")
    print("  - ECE < 0.1: 校准良好，放宽占比上限 20%")
    print("  - ECE > 0.2: 校准失效，收紧占比上限 20%")

    print("\n" + "="*70)