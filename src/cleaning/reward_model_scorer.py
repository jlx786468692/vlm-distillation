"""
数据质量打分系统（官方标准）
=============================

总分设计：
- 基准分 = 60分（所有样本初始及格基线）
- 规则区间：0～60分
- Judge模型分：0～100分
- 融合公式：final_score = 0.35 * rule_score + 0.65 * judge_score

一票否决项：直接置 rule_score=0，跳过Judge推理
新增校验：
- 校验A：三元自洽校验（hard/soft/cot）
- 校验B：GT真值与Hard标签校验
分区阈值：
- final_score >= 70: clean_valid（进入训练集）
- final_score >= 40: need_fix（人工复核）
- final_score < 40: discard（直接丢弃）
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# 导入闭合样本校验器
try:
    from .closed_sample_validator import ClosedSampleValidator
    CLOSED_VALIDATOR_AVAILABLE = True
    print("✓ [导入成功] ClosedSampleValidator 模块已加载", flush=True)  # 直接输出到控制台
except ImportError as e:
    CLOSED_VALIDATOR_AVAILABLE = False
    ClosedSampleValidator = None
    # 🔧 改进：同时输出到 stderr 和日志，确保错误可见
    import sys
    error_msg = f"⚠️ [导入失败] 闭合样本校验器导入失败: {e}"
    print(error_msg, file=sys.stderr, flush=True)  # 输出到 stderr


class RewardModelScorer:
    """
    数据质量打分器（官方标准）

    严格按照扣分规则体系进行质量评分
    """

    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        """
        初始化打分器

        Args:
            config: 配置管理器（可选）
            logger: 日志记录器（可选，如果不提供则使用模块默认logger）
        """
        # 🔧 修复：优先使用传入的 logger，确保日志统一
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config

        # ───────────────────────────────────────────────────────
        # 打分参数配置
        # ───────────────────────────────────────────────────────
        self.BASE_SCORE = 60  # 基准分
        self.MIN_RULE_SCORE = 0  # 规则分下限
        self.MAX_RULE_SCORE = 60  # 规则分上限
        self.MIN_JUDGE_SCORE = 0  # Judge分下限
        self.MAX_JUDGE_SCORE = 100  # Judge分上限

        # 融合权重
        self.RULE_WEIGHT = 0.35
        self.JUDGE_WEIGHT = 0.65

        # 分区阈值
        self.CLEAN_VALID_THRESHOLD = 70
        self.NEED_FIX_THRESHOLD = 40

        # Tokenizer（用于token统计）
        self.tokenizer = None
        self._tokenizer_loaded = False

        # 模型打分器（延迟加载）
        self.model_judge = None
        self._judge_loaded = False

        # 闭合样本校验器（延迟加载）
        self.closed_validator = None
        self._closed_validator_loaded = False

        # strict_closed_mode 开关
        if self.config:
            self.strict_closed_mode = self.config.get('cleaning.strict_closed_mode', False)
        else:
            self.strict_closed_mode = False

        self.logger.info("✓ 数据质量打分器初始化完成（官方标准）")
        self.logger.info(f"  - 基准分: {self.BASE_SCORE}分")
        self.logger.info(f"  - 规则区间: {self.MIN_RULE_SCORE}~{self.MAX_RULE_SCORE}分")
        self.logger.info(f"  - Judge区间: {self.MIN_JUDGE_SCORE}~{self.MAX_JUDGE_SCORE}分")
        self.logger.info(f"  - 融合公式: {self.RULE_WEIGHT}×规则分 + {self.JUDGE_WEIGHT}×模型分")
        self.logger.info(f"  - strict_closed_mode: {self.strict_closed_mode}")

    def _get_field(self, sample: Dict, field: str, default: Any = None) -> Any:
        """
        通用字段获取（兼容两种数据结构）

        字段可能在：
        1. 顶层：sample[field]
        2. tasks.vqa 下：sample['tasks']['vqa'][field]

        Args:
            sample: 样本数据
            field: 字段名
            default: 默认值

        Returns:
            字段值
        """
        # 先检查顶层
        if field in sample:
            return sample[field]

        # 再检查 tasks.vqa 下
        vqa_data = sample.get('tasks', {}).get('vqa', {})
        if field in vqa_data:
            return vqa_data[field]

        return default

    def _load_tokenizer(self):
        """加载tokenizer（延迟加载）"""
        if self._tokenizer_loaded:
            return

        try:
            from transformers import AutoTokenizer

            model_path = "models/Qwen2.5-VL-32B-Instruct-AWQ"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self._tokenizer_loaded = True
            self.logger.info("✓ Tokenizer加载成功（用于token统计）")
        except Exception as e:
            self.logger.warning(f"Tokenizer加载失败: {e}，将使用字符数统计")
            self._tokenizer_loaded = True

    def count_tokens(self, text: str) -> int:
        """
        统计token数量

        Args:
            text: 文本内容

        Returns:
            token数量
        """
        if not self._tokenizer_loaded:
            self._load_tokenizer()

        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        else:
            # 备选方案：使用字符数
            return len(text.strip())

    def score(
        self,
        question: str,
        answer: str,
        question_type: Optional[str] = None,
        sample: Optional[Dict] = None,
        image_path: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        质量打分（官方标准）

        Args:
            question: 问题文本
            answer: 答案文本
            question_type: 问题类型（closed/open）
            sample: 完整样本数据
            image_path: 图像路径（用于模型打分）
            ground_truth: VQA标注答案

        Returns:
            {
                'final_score': float,      # 总分（0-100）
                'rule_score': float,       # 规则分（0-60）
                'judge_score': float,      # 模型分（0-100）
                'bucket': str,             # 分区（clean_valid/need_fix/discard）
                'is_valid': bool,          # 是否有效
                'issues': List[str],       # 问题列表
                'deductions': Dict,        # 扣分明细
                'veto': bool,              # 是否触发一票否决
                'cleaning_mode': str       # 清洗模式（closed/open）
            }
        """
        issues = []
        deductions = {}
        veto = False

        # 判断问题类型
        if sample:
            vqa_data = sample.get('tasks', {}).get('vqa', {})
            inference_mode = vqa_data.get('inference_mode', 'closed')
            # question_type 可能在顶层或 tasks.vqa 下
            question_type = sample.get('question_type') or vqa_data.get('question_type', question_type or 'unknown')
        else:
            inference_mode = 'closed' if question_type in ['count', 'color', 'binary', 'location'] else 'open'

        cleaning_mode = inference_mode

        # ───────────────────────────────────────────────────────
        # Step 1: 一票否决检查（直接0分，跳过Judge推理）
        # ───────────────────────────────────────────────────────
        veto, veto_reason = self._check_veto(sample or {'question': question, 'answer': answer}, inference_mode)

        if veto:
            self.logger.warning(f"样本触发一票否决: {veto_reason}")
            issues.append(f"一票否决: {veto_reason}")
            deductions['veto'] = {
                'reason': veto_reason,
                'forced_score': 0
            }

            # 直接返回0分，不调用Judge模型（节省算力）
            return {
                'final_score': 0.0,
                'rule_score': 0.0,
                'judge_score': 0.0,
                'bucket': 'discard',
                'is_valid': False,
                'issues': issues,
                'deductions': deductions,
                'veto': True,
                'cleaning_mode': cleaning_mode
            }

        # ───────────────────────────────────────────────────────
        # Step 2: 规则层打分（基准分60，扣分制 + 加分制）
        # ───────────────────────────────────────────────────────
        # 🔧 修复：先扣分后加分，严格全局钳位
        # 流程：
        # 1. 基准分 = 60
        # 2. 扣分累加（负数）
        # 3. 加分累加（正数）
        # 4. 最终钳位：max(0, min(score, 100))
        # ───────────────────────────────────────────────────────
        rule_score = self.BASE_SCORE  # 从60分开始
        total_bonus = 0  # 加分总和

        # 通用扣分规则（返回负数：累计扣分）
        general_deduction, general_issues, general_deductions = self._apply_general_rules(
            sample or {'question': question, 'answer': answer}, inference_mode
        )
        # 叠加扣分
        rule_score += general_deduction
        issues.extend(general_issues)
        deductions.update(general_deductions)

        # 差异化扣分规则（扣分 + 加分）
        if inference_mode == 'closed' or question_type in ['count', 'color', 'binary', 'location']:
            # 闭合问题专属规则
            closed_deduction, specific_issues, specific_deductions, closed_bonus = self._apply_closed_rules(
                sample or {}, ground_truth=ground_truth
            )
            # 叠加扣分
            rule_score += closed_deduction
            # 叠加加分
            total_bonus += closed_bonus
            issues.extend(specific_issues)
            deductions.update(specific_deductions)

            # ───────────────────────────────────────────────────────
            # 🔧 新增：检查校验A/B一票否决（strict_closed_mode=true）
            # ───────────────────────────────────────────────────────
            if deductions.get('validation_a_veto') or deductions.get('validation_b_veto'):
                # 校验失败导致的一票否决
                veto_reason = ""
                if deductions.get('validation_a_veto'):
                    veto_reason = deductions['validation_a_veto'].get('reason', '校验A失败')
                elif deductions.get('validation_b_veto'):
                    veto_reason = deductions['validation_b_veto'].get('reason', '校验B失败')

                self.logger.warning(f"校验失败触发一票否决: {veto_reason}")

                # 直接返回0分，标记为 veto
                return {
                    'final_score': 0.0,
                    'rule_score': 0.0,
                    'judge_score': 0.0,
                    'bucket': 'discard',
                    'is_valid': False,
                    'issues': issues,
                    'deductions': deductions,
                    'veto': True,
                    'cleaning_mode': cleaning_mode
                }
        else:
            # 开放问题专属规则
            open_deduction, specific_issues, specific_deductions, open_bonus = self._apply_open_rules(sample or {})
            # 叠加扣分
            rule_score += open_deduction
            # 叠加加分
            total_bonus += open_bonus
            issues.extend(specific_issues)
            deductions.update(specific_deductions)

        # 加分累加（先扣分后加分）
        rule_score += total_bonus

        # ───────────────────────────────────────────────────────
        # 最终全局钳位：max(0, min(score, 100))
        # ───────────────────────────────────────────────────────
        rule_score = max(0, min(rule_score, 100))

        # ───────────────────────────────────────────────────────
        # Step 3: Judge模型打分（0-100）
        # ───────────────────────────────────────────────────────
        judge_score = None
        judge_valid = True

        # 如果提供了图像路径，使用模型打分
        if image_path or (sample and sample.get('image_path')):
            img_path = image_path or sample.get('image_path')

            if img_path:
                # 延迟加载模型打分器
                if not self._judge_loaded:
                    self._load_model_judge()

                # 模型打分
                try:
                    model_result = self.model_judge.score(
                        image_path=img_path,
                        question=question,
                        answer=answer,
                        sample=sample
                    )

                    if model_result.get('is_valid', True):
                        judge_score = model_result.get('model_score')
                        # 钳位到 [0, 100]
                        judge_score = max(min(judge_score, self.MAX_JUDGE_SCORE), self.MIN_JUDGE_SCORE)
                    else:
                        judge_valid = False
                        judge_score = rule_score  # 策略A：降级为规则分
                        self.logger.warning(
                            f"Judge打分失败，降级为规则分: {judge_score}"
                        )

                except Exception as e:
                    self.logger.warning(f"模型打分异常: {e}")
                    judge_valid = False
                    judge_score = rule_score  # 策略A：降级为规则分

        # 如果没有调用Judge，使用默认值
        if judge_score is None:
            judge_score = 50  # 默认中等分数

        # ───────────────────────────────────────────────────────
        # Step 4: 总分融合计算
        # ───────────────────────────────────────────────────────
        final_score = self.RULE_WEIGHT * rule_score + self.JUDGE_WEIGHT * judge_score

        # 限制总分范围 [0, 100]
        final_score = max(min(final_score, 100.0), 0.0)

        # ───────────────────────────────────────────────────────
        # Step 5: 分区判断
        # ───────────────────────────────────────────────────────
        if final_score >= self.CLEAN_VALID_THRESHOLD:
            bucket = 'clean_valid'
        elif final_score >= self.NEED_FIX_THRESHOLD:
            bucket = 'need_fix'
        else:
            bucket = 'discard'

        is_valid = bucket != 'discard'

        return {
            'final_score': round(final_score, 2),
            'rule_score': round(rule_score, 2),
            'judge_score': round(judge_score, 2),
            'bucket': bucket,
            'is_valid': is_valid,
            'issues': issues,
            'deductions': deductions,
            'veto': False,
            'cleaning_mode': cleaning_mode,
            'judge_valid': judge_valid
        }

    # ============================================================
    # 一票否决检查
    # ============================================================

    def _check_veto(self, sample: Dict, inference_mode: str) -> Tuple[bool, str]:
        """
        一票否决检查（按 question_type 分流执行）

        满足任意一条 → rule_score = 0，跳过Judge推理

        Args:
            sample: 样本数据
            inference_mode: 推理模式（'open' 或 'closed'）

        Returns:
            (是否否决, 否决原因)
        """
        vqa_data = sample.get('tasks', {}).get('vqa', {})

        # ───────────────────────────────────────────────────────
        # 【通用一票否决】开放 + 闭合都执行
        # ───────────────────────────────────────────────────────

        # 1. image_path 缺失、图片无法读取
        image_path = sample.get('image_path', '')
        if not image_path:
            return True, "图像路径缺失"

        if not Path(image_path).exists():
            return True, f"图像文件不存在: {image_path}"

        # 2. question 字段为空
        # 🔧 修复：question 可能在顶层或 tasks.vqa 下
        question = sample.get('question', '') or vqa_data.get('question', '')
        if not question or not question.strip():
            return True, "question字段为空"

        # 3. 模型输出大量复读 system prompt / 复制 prompt 指令
        # 检查所有文本字段
        all_text = self._extract_all_text(sample)
        if self._is_repeating_prompt(all_text):
            return True, "模型输出复读System Prompt或大量复制输入指令"

        # 4. 文本包含大量乱码、不可见控制字符
        if self._has_severe_pollution(all_text):
            return True, "文本包含大量乱码或不可见控制字符"

        # ───────────────────────────────────────────────────────
        # 【开放问题专属一票否决】仅 open / open_descriptive 执行
        # ───────────────────────────────────────────────────────
        if inference_mode == 'open':
            # 开放问题：answer 为空 / 仅空格、无有效文本
            answer = vqa_data.get('answer', '')
            if not answer or not answer.strip():
                return True, "开放问题answer为空或仅包含空白字符"

        # ───────────────────────────────────────────────────────
        # 【闭合问题专属一票否决】仅 counting/binary/color 等闭合类型执行
        # ───────────────────────────────────────────────────────
        if inference_mode == 'closed':
            # 1. hard_label 缺失 || hard_label["answer"] 为空
            hard_label_data = vqa_data.get('hard_label', {})
            if not hard_label_data:
                return True, "闭合问题hard_label缺失"

            hard_label_answer = hard_label_data.get('answer', '')
            if not hard_label_answer or not hard_label_answer.strip():
                return True, "闭合问题hard_label['answer']为空"

            # 2. soft_label 缺失 || answer_distribution 为空
            soft_label_data = vqa_data.get('soft_label', {})
            if not soft_label_data:
                return True, "闭合问题soft_label缺失"

            answer_distribution = soft_label_data.get('answer_distribution', {})
            if not answer_distribution:
                return True, "闭合问题answer_distribution为空"

            # 3. cot_reasoning 缺失（如果流水线强制生成 CoT）
            # 检查是否需要CoT（通过配置或sample判断）
            requires_cot = vqa_data.get('requires_cot', True)  # 默认需要
            if requires_cot:
                cot_reasoning = vqa_data.get('cot_reasoning', {})
                if not cot_reasoning:
                    return True, "闭合问题cot_reasoning缺失（强制CoT模式）"

            # 4. hard_label["answer"] NOT in candidate_pool（重点！）
            candidate_pool = vqa_data.get('candidate_pool', [])
            if candidate_pool and hard_label_answer not in candidate_pool:
                return True, f"hard_label不在候选池: '{hard_label_answer}' not in {candidate_pool}"

            # ⚠️ 注意：闭合样本不判断 answer 字段是否为空
            # 闭合依靠软硬标签训练，开放依靠 answer 文本训练

        return False, ""

    def _extract_all_text(self, sample: Dict) -> str:
        """
        提取样本中所有文本字段（用于复读检测和污染检测）

        Args:
            sample: 样本数据

        Returns:
            所有文本合并
        """
        texts = []

        # vqa 相关文本
        vqa_data = sample.get('tasks', {}).get('vqa', {})

        # question（可能在顶层或 tasks.vqa 下）
        question = sample.get('question') or vqa_data.get('question')
        if question:
            texts.append(question)

        # answer（开放问题）
        if vqa_data.get('answer'):
            texts.append(vqa_data['answer'])

        # hard_label（闭合问题）
        hard_label = vqa_data.get('hard_label', {})
        if hard_label.get('answer'):
            texts.append(hard_label['answer'])

        # CoT reasoning
        cot_reasoning = vqa_data.get('cot_reasoning', {})
        if isinstance(cot_reasoning, dict):
            for key in ['observation', 'analysis', 'conclusion']:
                if cot_reasoning.get(key):
                    texts.append(cot_reasoning[key])

        return ' '.join(texts)

    def _is_repeating_prompt(self, text: str) -> bool:
        """
        检测是否复读Prompt

        Args:
            text: 文本内容

        Returns:
            是否复读
        """
        # 检测System Prompt关键词
        prompt_keywords = [
            'system', 'user', 'assistant', 'TASK:', 'CRITICAL RULES',
            'You are', 'Output only', 'FOUR NON-BREAKABLE', 'Rules:'
        ]

        keyword_count = sum(1 for kw in prompt_keywords if kw.lower() in text.lower())

        # 如果包含3个以上关键词，认为是复读
        if keyword_count >= 3:
            return True

        # 检测是否有大段重复的prompt内容
        # 如果文本中包含完整的指令段落（超过50个字符连续匹配）
        instruction_patterns = [
            'Your task is to',
            'Output only one',
            'DO NOT describe'
        ]

        for pattern in instruction_patterns:
            if pattern.lower() in text.lower():
                # 检查是否重复出现
                if text.lower().count(pattern.lower()) >= 2:
                    return True

        return False

    def _has_severe_pollution(self, text: str) -> bool:
        """
        检测严重越界污染

        Args:
            text: 文本内容

        Returns:
            是否污染
        """
        if not text:
            return False

        # 检测控制字符（ASCII < 32，不包括换行、回车、制表符）
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')

        # 检测乱码比例
        pollution_ratio = control_chars / len(text)
        if pollution_ratio > 0.1:  # 超过10%为严重污染
            return True

        return False

    # ============================================================
    # 通用扣分规则（开放 + 闭合）
    # ============================================================

    def _apply_general_rules(self, sample: Dict, inference_mode: str) -> Tuple[float, List[str], Dict]:
        """
        通用扣分规则（返回扣分总和）

        Args:
            sample: 样本数据
            inference_mode: 推理模式

        Returns:
            (扣分总和, 问题列表, 扣分明细)
            注意：返回的是负数（累计扣分），用于叠加到基准分上
        """
        score = 0  # 从0开始，累计扣分（负数）
        issues = []
        deductions = {}

        vqa_data = sample.get('tasks', {}).get('vqa', {})

        # 获取输出文本（用于检测）
        if inference_mode == 'open':
            output_text = vqa_data.get('answer', '')
        else:
            output_text = vqa_data.get('hard_label', {}).get('answer', '')

        # question（可能在顶层或 tasks.vqa 下）
        question = sample.get('question', '') or vqa_data.get('question', '')

        # ───────────────────────────────────────────────────────
        # 1. 文本存在 Markdown 符号（#、##、-、列表）
        # ───────────────────────────────────────────────────────
        markdown_patterns = [r'#{1,6}\s', r'\*\*.*?\*\*', r'^\s*-\s', r'^\s*\d+\.\s']
        has_markdown = any(re.search(p, output_text, re.MULTILINE) for p in markdown_patterns)

        if has_markdown:
            deduction = 10
            score -= deduction
            issues.append("文本包含Markdown符号")
            deductions['markdown'] = {'deduction': deduction}

        # ───────────────────────────────────────────────────────
        # 2. question 与输出文本相似度 > 0.75（复读）
        # ───────────────────────────────────────────────────────
        if question and output_text:
            similarity = self._calculate_similarity(question, output_text)
            if similarity > 0.75:
                deduction = 12
                score -= deduction
                issues.append(f"question与输出文本高度重复（相似度: {similarity:.2f}）")
                deductions['repetition'] = {
                    'similarity': similarity,
                    'threshold': 0.75,
                    'deduction': deduction
                }

        # ───────────────────────────────────────────────────────
        # 3. 无意义大量换行、连续空行
        # ───────────────────────────────────────────────────────
        newline_count = output_text.count('\n')
        consecutive_newlines = len(re.findall(r'\n{3,}', output_text))  # 3个或以上连续换行

        if consecutive_newlines > 0:
            deduction = 6
            score -= deduction
            issues.append(f"包含无意义连续空行（{consecutive_newlines}处）")
            deductions['excessive_newlines'] = {
                'count': consecutive_newlines,
                'deduction': deduction
            }

        return score, issues, deductions

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（简化版）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度（0-1）
        """
        # 简化版：基于词汇重叠
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    # ============================================================
    # 闭合VQA专属扣分规则
    # ============================================================

    def _apply_closed_rules(self, sample: Dict, ground_truth: Optional[str] = None) -> Tuple[float, List[str], Dict, float]:
        """
        闭合VQA专属规则（扣分 + 加分）

        Args:
            sample: 样本数据
            ground_truth: GT真值（从COCO标注获取，通过缓存机制加载）

        Returns:
            (扣分总和, 问题列表, 扣分明细, 加分总和)
            - 扣分总和：负数
            - 加分总和：正数
        """
        deduction_score = 0  # 扣分总和（负数）
        bonus_score = 0      # 加分总和（正数）
        issues = []
        deductions = {}
        bonuses = {}

        vqa_data = sample.get('tasks', {}).get('vqa', {})

        # ───────────────────────────────────────────────────────
        # 扣分规则
        # ───────────────────────────────────────────────────────

        # 1. 教师原始置信度 < 0.4
        confidence = vqa_data.get('hard_label', {}).get('confidence', 1.0)
        if confidence < 0.4:
            deduction = 14
            deduction_score -= deduction
            issues.append(f"教师置信度过低（{confidence:.2f} < 0.4）")
            deductions['low_confidence'] = {
                'confidence': confidence,
                'threshold': 0.4,
                'deduction': deduction
            }

        # 2. soft_label分布极度平坦（最大概率 < 0.3）
        soft_label = vqa_data.get('soft_label', {})
        answer_distribution = soft_label.get('answer_distribution', {})

        if answer_distribution:
            max_prob = max(answer_distribution.values()) if answer_distribution else 0
            if max_prob < 0.3:
                deduction = 10
                deduction_score -= deduction
                issues.append(f"soft_label分布平坦（最大概率: {max_prob:.2f} < 0.3）")
                deductions['flat_distribution'] = {
                    'max_prob': max_prob,
                    'threshold': 0.3,
                    'deduction': deduction
                }

        # 3. CoT结构化错乱（缺少关键段落）
        cot_reasoning = vqa_data.get('cot_reasoning', {})
        if cot_reasoning:
            # 🔧 修复：支持两种格式
            # 新格式（两段式）：cot_reasoning.reasoning_paragraph + cot_reasoning.answer
            # 旧格式（三段式）：cot_reasoning.structured_reasoning.{observation, analysis, conclusion}
            if 'reasoning_paragraph' in cot_reasoning or 'answer' in cot_reasoning:
                # 新格式：检查两段式字段
                missing_fields = []
                if 'reasoning_paragraph' not in cot_reasoning or not cot_reasoning['reasoning_paragraph']:
                    missing_fields.append('reasoning_paragraph')
                if 'answer' not in cot_reasoning or not cot_reasoning['answer']:
                    missing_fields.append('answer')

                if missing_fields:
                    deductions['incomplete_cot'] = {
                        'deduction': 10,
                        'reason': f"CoT缺少字段: {missing_fields}"
                    }
            elif 'structured_reasoning' in cot_reasoning:
                # 旧格式：嵌套结构
                structured = cot_reasoning.get('structured_reasoning', {})
                required_sections = ['observation', 'analysis', 'conclusion']
                missing_sections = [s for s in required_sections if s not in structured or not structured[s]]
                if missing_sections:
                    deductions['incomplete_cot'] = {
                        'deduction': 10,
                        'reason': f"CoT缺少段落: {missing_sections}"
                    }
            else:
                # 直接子字段结构
                required_sections = ['observation', 'analysis', 'conclusion']
                missing_sections = [s for s in required_sections if s not in cot_reasoning or not cot_reasoning[s]]

            if missing_sections:
                deduction = 9
                deduction_score -= deduction
                issues.append(f"CoT结构错乱，缺失: {', '.join(missing_sections)}")
                deductions['cot_structure'] = {
                    'missing_sections': missing_sections,
                    'deduction': deduction
                }

        # 4. 生成答案存在多token污染
        hard_label_answer = vqa_data.get('hard_label', {}).get('answer', '')
        candidate_pool = vqa_data.get('candidate_pool', [])

        if hard_label_answer and candidate_pool:
            # 检查候选是否都是单字单词
            all_single_word = all(len(c.split()) == 1 for c in candidate_pool)

            # 检查答案是否是多个词
            if all_single_word and len(hard_label_answer.split()) > 1:
                deduction = 11
                deduction_score -= deduction
                issues.append(f"多token污染：候选都是单字，但答案'{hard_label_answer}'包含多词")
                deductions['multi_token_pollution'] = {
                    'answer': hard_label_answer,
                    'candidate_pool': candidate_pool,
                    'deduction': deduction
                }

        # 5. 校验A：三元自洽校验（hard/soft/cot）
        # 延迟加载校验器
        if not self._closed_validator_loaded:
            self._load_closed_validator()

        self.logger.info(f"【校验A检查】closed_validator存在: {self.closed_validator is not None}")

        triple_consistent = True  # 标记三元是否一致
        if self.closed_validator:
            soft_label_primary_answer = vqa_data.get('soft_label', {}).get('primary_answer', '')
            # 🔧 修改：从新格式获取CoT答案
            cot_conclusion = vqa_data.get('cot_reasoning', {}).get('answer', '')

            self.logger.info(f"【校验A检查】准备执行校验A...")
            self.logger.info(f"  - hard_label_answer: {hard_label_answer}")
            self.logger.info(f"  - soft_label_primary_answer: {soft_label_primary_answer}")
            self.logger.info(f"  - cot_conclusion: {cot_conclusion}")

            is_consistent_a, reason_a = self.closed_validator.validate_triple_consistency(
                hard_label_answer,
                soft_label_primary_answer,
                cot_conclusion,
                candidate_pool
            )

            if not is_consistent_a:
                triple_consistent = False  # 标记不一致
                if self.strict_closed_mode:
                    # 一票否决
                    deductions['validation_a_veto'] = {
                        'veto': True,
                        'reason': reason_a
                    }
                    # 直接返回扣分-60
                    return -self.BASE_SCORE, issues + [f"校验A失败（一票否决）: {reason_a}"], deductions, 0
                else:
                    # 重度扣分
                    deduction = 22
                    deduction_score -= deduction
                    issues.append(f"三元不自洽: {reason_a}")
                    deductions['validation_a'] = {
                        'deduction': deduction,
                        'reason': reason_a
                    }

        # 6. 校验B：GT真值与Hard标签语义校验
        # ground_truth 已通过参数传入（从COCO标注或缓存获取）

        self.logger.info(f"【校验B检查】ground_truth存在: {ground_truth is not None}")
        self.logger.info(f"【校验B检查】closed_validator存在: {self.closed_validator is not None}")

        if ground_truth and self.closed_validator:
            self.logger.info("【校验B】开始执行GT与Hard一致性校验...")
            is_consistent_b, reason_b = self.closed_validator.validate_gt_hard_consistency(
                ground_truth,
                hard_label_answer
            )

            if not is_consistent_b:
                if self.strict_closed_mode:
                    # 一票否决
                    deductions['validation_b_veto'] = {
                        'veto': True,
                        'reason': reason_b
                    }
                    # 直接返回扣分-60
                    return -self.BASE_SCORE, issues + [f"校验B失败（一票否决）: {reason_b}"], deductions, 0
                else:
                    # 重度扣分
                    deduction = 20
                    deduction_score -= deduction
                    issues.append(f"GT与Hard语义不等价: {reason_b}")
                    deductions['validation_b'] = {
                        'deduction': deduction,
                        'reason': reason_b
                    }
        else:
            if not ground_truth:
                self.logger.info("【校验B跳过】样本缺少 ground_truth 字段")
            if not self.closed_validator:
                self.logger.info("【校验B跳过】closed_validator 未加载")

        # ───────────────────────────────────────────────────────
        # 加分规则（仅闭合样本拥有）
        # ───────────────────────────────────────────────────────

        # 1. 硬标签置信度 ≥0.75 → +6
        confidence = vqa_data.get('hard_label', {}).get('confidence', 1.0)
        if confidence >= 0.75:
            bonus = 6
            bonus_score += bonus
            issues.append(f"硬标签置信度高（{confidence:.2f} ≥ 0.75）")
            bonuses['high_confidence'] = {
                'confidence': confidence,
                'bonus': bonus
            }

        # 2. soft_label概率分布区分度高（top1/top2 差距 > 0.4）→ +5
        if answer_distribution and len(answer_distribution) >= 2:
            # 获取top1和top2的概率
            sorted_probs = sorted(answer_distribution.values(), reverse=True)
            top1_prob = sorted_probs[0]
            top2_prob = sorted_probs[1]
            gap = top1_prob - top2_prob

            if gap > 0.4:
                bonus = 5
                bonus_score += bonus
                issues.append(f"soft_label区分度高（top1/top2差距: {gap:.2f} > 0.4）")
                bonuses['high_distribution_gap'] = {
                    'gap': gap,
                    'top1': top1_prob,
                    'top2': top2_prob,
                    'bonus': bonus
                }

        # 3. 三元一致性校验完全通过（hard/soft/cot conclusion 一致）→ +7
        if triple_consistent and self.closed_validator:
            bonus = 7
            bonus_score += bonus
            issues.append("三元一致性校验完全通过")
            bonuses['triple_consistency'] = {
                'bonus': bonus
            }

        return deduction_score, issues, deductions, bonus_score

    # ============================================================
    # 开放VQA专属扣分规则
    # ============================================================

    def _apply_open_rules(self, sample: Dict) -> Tuple[float, List[str], Dict, float]:
        """
        开放VQA专属规则（扣分 + 加分）

        Args:
            sample: 样本数据

        Returns:
            (扣分总和, 问题列表, 扣分明细, 加分总和)
            - 扣分总和：负数
            - 加分总和：正数
        """
        deduction_score = 0  # 扣分总和（负数）
        bonus_score = 0      # 加分总和（正数）
        issues = []
        deductions = {}
        bonuses = {}

        vqa_data = sample.get('tasks', {}).get('vqa', {})
        answer = vqa_data.get('answer', '')

        # ───────────────────────────────────────────────────────
        # 扣分规则
        # ───────────────────────────────────────────────────────

        # 1. answer token 长度 < 60
        token_count = self.count_tokens(answer)
        if token_count < 60:
            deduction = 15
            deduction_score -= deduction
            issues.append(f"开放问题answer过短（{token_count} tokens < 60）")
            deductions['short_answer'] = {
                'tokens': token_count,
                'threshold': 60,
                'deduction': deduction
            }

        # 2. answer token 长度 > 1800
        if token_count > 1800:
            deduction = 12
            deduction_score -= deduction
            issues.append(f"开放问题answer超长（{token_count} tokens > 1800）")
            deductions['long_answer'] = {
                'tokens': token_count,
                'threshold': 1800,
                'deduction': deduction
            }

        # 3. 输出仅单个单词，无完整描述段落
        word_count = len(answer.split())
        if word_count <= 1:
            deduction = 16
            deduction_score -= deduction
            issues.append(f"开放问题输出仅单个单词（{word_count}个单词）")
            deductions['single_word'] = {
                'word_count': word_count,
                'deduction': deduction
            }

        # ───────────────────────────────────────────────────────
        # 加分规则（仅开放样本拥有）
        # ───────────────────────────────────────────────────────

        # 1. Answer token 长度区间在 120～1200（适中，不长不短）→ +6
        if 120 <= token_count <= 1200:
            bonus = 6
            bonus_score += bonus
            issues.append(f"答案长度适中（{token_count} tokens 在 120～1200）")
            bonuses['moderate_length'] = {
                'tokens': token_count,
                'bonus': bonus
            }

        # 2. 无幻觉、描述充分完整（初期仅长度判断）→ +5
        # 如果长度在 200～1000 tokens，认为描述充分
        if 200 <= token_count <= 1000:
            bonus = 5
            bonus_score += bonus
            issues.append(f"描述充分完整（{token_count} tokens）")
            bonuses['sufficient_description'] = {
                'tokens': token_count,
                'bonus': bonus
            }

        # 3. 段落流畅，无碎片短句、无 markdown → +7
        # 检查是否有 Markdown 符号（已在通用规则检查）
        # 如果没有 Markdown 且没有碎片短句（平均句子长度 > 10 个词）
        import re
        markdown_patterns = [r'#{1,6}\s', r'\*\*.*?\*\*', r'^\s*-\s', r'^\s*\d+\.\s']
        has_markdown = any(re.search(p, answer, re.MULTILINE) for p in markdown_patterns)

        if not has_markdown:
            # 检查平均句子长度（简单判断：按句号、问号、感叹号分割）
            sentences = re.split(r'[。！？.!?]', answer)
            sentences = [s.strip() for s in sentences if s.strip()]

            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if avg_sentence_length > 10:  # 平均句子长度 > 10 个词
                    bonus = 7
                    bonus_score += bonus
                    issues.append(f"段落流畅无碎片（平均句长: {avg_sentence_length:.1f} 词）")
                    bonuses['fluent_paragraph'] = {
                        'avg_sentence_length': avg_sentence_length,
                        'bonus': bonus
                    }

        return deduction_score, issues, deductions, bonus_score

    def _load_model_judge(self):
        """加载模型打分器（延迟加载）"""
        if self._judge_loaded:
            return

        try:
            from .reward_model_judge import RewardModelJudge

            # 🔧 修复：传递 logger 确保日志统一
            self.model_judge = RewardModelJudge(self.config, logger=self.logger)
            self._judge_loaded = True
            self.logger.info("✓ Judge模型打分器加载成功")

        except Exception as e:
            self.logger.warning(f"Judge模型打分器加载失败: {e}")
            self._judge_loaded = True

    def _load_closed_validator(self):
        """加载闭合样本校验器（延迟加载）"""
        self.logger.info("【调试】开始加载闭合样本校验器...")

        if self._closed_validator_loaded:
            self.logger.info("【调试】校验器已加载，跳过")
            return

        if not CLOSED_VALIDATOR_AVAILABLE:
            self.logger.warning("⚠️ 闭合样本校验器模块未找到，将跳过校验A和校验B")
            self.logger.warning("这可能是因为导入失败，请检查 closed_sample_validator.py 是否存在")
            self._closed_validator_loaded = True
            return

        try:
            # 🔧 修复：传递 logger 确保日志统一
            self.logger.info("【调试】正在创建 ClosedSampleValidator 实例...")
            self.closed_validator = ClosedSampleValidator(self.config, logger=self.logger)
            self._closed_validator_loaded = True
            self.logger.info("✓ 闭合样本校验器加载成功")

        except Exception as e:
            self.logger.error(f"闭合样本校验器初始化失败: {e}")
            self.closed_validator = None
            self._closed_validator_loaded = True

    def _load_ground_truths(self) -> Dict[Tuple[str, str], str]:
        """
        从COCO VQA标注加载GT真值（用于校验B）

        通过 COCODataLoader.build_gt_mapping() 加载，支持缓存机制：
        - "auto": 优先使用缓存，缓存不存在才构建并保存
        - "rebuild": 强制重新构建，更新缓存文件
        - "disabled": 禁用缓存，每次从COCO标注构建

        Returns:
            GT真值映射: {(image_id, question): answer}
        """
        # ───────────────────────────────────────────────────────
        # 从配置读取缓存参数
        # ───────────────────────────────────────────────────────
        gt_config = {}
        if self.config:
            gt_config = self.config.get('cleaning.gt_mapping', {})

        cache_mode = gt_config.get('cache_mode', 'auto')  # 默认：auto
        cache_file = gt_config.get('cache_file', './data/gt_mapping_cache.json')

        self.logger.info(f"【GT真值】准备加载GT映射...")
        self.logger.info(f"  - cache_mode: {cache_mode}")
        self.logger.info(f"  - cache_file: {cache_file}")

        # ───────────────────────────────────────────────────────
        # 通过 COCODataLoader 加载并构建GT映射
        # ───────────────────────────────────────────────────────
        try:
            # 导入COCO数据加载器
            from ..data.coco_loader import COCODataLoader

            # 创建加载器实例
            coco_loader = COCODataLoader(self.config)

            # 初始化（加载VQA问题和答案）
            coco_loader.initialize(split="val2014")

            # 构建GT真值映射（传递缓存配置）
            gt_mapping = coco_loader.build_gt_mapping(
                cache_mode=cache_mode,
                cache_file=cache_file
            )

            if not gt_mapping:
                self.logger.warning("GT真值为空，校验B将被跳过")

            return gt_mapping

        except Exception as e:
            self.logger.warning(f"加载GT真值失败: {e}")
            self.logger.warning("校验B（GT一致性校验）将被跳过")
            return {}

    def score_batch(
        self,
        samples: List[Dict[str, Any]],
        ground_truths: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量打分（简化接口，所有逻辑内部处理）

        Args:
            samples: 样本列表
            ground_truths: GT真值字典（可选，key=(image_id, question), value=GT答案）
                          如果不提供，会自动从COCO VQA标注加载

        Returns:
            打分后的样本列表（每个样本包含 quality_score 字段）
        """
        self.logger.info(f"\n开始批量打分：{len(samples)} 个样本")

        # ───────────────────────────────────────────────────────
        # 🔧 新增：自动加载GT真值（如果没有提供）
        # ───────────────────────────────────────────────────────
        if ground_truths is None:
            ground_truths = self._load_ground_truths()

        scored_samples = []
        stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'avg_score': 0.0
        }

        total_score = 0.0

        for sample in samples:
            try:
                # ───────────────────────────────────────────────────────
                # 内部提取所有必要信息
                # ───────────────────────────────────────────────────────
                vqa_data = sample.get('tasks', {}).get('vqa', {})
                question = vqa_data.get('question', '')
                inference_mode = vqa_data.get('inference_mode', 'closed')
                question_type = vqa_data.get('question_type', 'unknown')

                # 根据推理模式获取答案
                if inference_mode == 'open':
                    answer = vqa_data.get('answer', '')
                else:
                    answer = vqa_data.get('hard_label', {}).get('answer', '')

                # 获取图像路径
                image_path = sample.get('image_path', '')

                # 获取 GT 真值（如果有）
                image_id = sample.get('image_id', sample.get('id', ''))
                # GT真值的key是 (image_id, question) 元组
                ground_truth = ground_truths.get((str(image_id), question)) if ground_truths else None

                # ───────────────────────────────────────────────────────
                # 调用打分接口
                # ───────────────────────────────────────────────────────
                score_result = self.score(
                    question=question,
                    answer=answer,
                    question_type=question_type,
                    sample=sample,
                    image_path=image_path,
                    ground_truth=ground_truth
                )

                # 添加打分结果到样本
                sample['quality_score'] = score_result

                # 更新统计
                stats['total'] += 1
                if score_result.get('is_valid', False):
                    stats['valid'] += 1
                else:
                    stats['invalid'] += 1

                total_score += score_result.get('final_score', 0)

            except Exception as e:
                self.logger.warning(f"Failed to score sample {sample.get('image_id')}: {e}")
                sample['quality_score'] = {
                    'final_score': 0,
                    'rule_score': 0,
                    'judge_score': 0,
                    'bucket': 'discard',
                    'is_valid': False,
                    'error': str(e)
                }
                stats['invalid'] += 1

            scored_samples.append(sample)

        # 计算平均分数
        if stats['total'] > 0:
            stats['avg_score'] = total_score / stats['total']

        self.logger.info(f"✓ 打分完成：{stats['total']} 个样本")
        self.logger.info(f"  - 有效: {stats['valid']}, 无效: {stats['invalid']}")
        self.logger.info(f"  - 平均分: {stats['avg_score']:.2f}")

        return scored_samples


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("数据质量打分系统（官方标准）")
    print("="*70)

    print("\n打分设计：")
    print("  基准分: 60分")
    print("  规则区间: 0~60分")
    print("  Judge区间: 0~100分")
    print("  融合公式: 0.35×规则分 + 0.65×模型分")

    print("\n分区阈值：")
    print("  >= 70分: clean_valid（进入训练集）")
    print("  >= 40分: need_fix（人工复核）")
    print("  < 40分: discard（直接丢弃）")

    print("\n一票否决项：")
    print("  - answer为空")
    print("  - 复读System Prompt")
    print("  - hard_label不在候选池")
    print("  - 图像路径缺失")
    print("  - 严重越界污染")

    print("\n" + "="*70)