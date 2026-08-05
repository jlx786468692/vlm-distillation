"""
多模态打分模型基类
===================

实现 Qwen-VL Judge 模型打分功能：
1. 视觉匹配度（是否贴合图片真实内容）
2. 逻辑一致性（CoT推理和标签对齐）
3. 格式标准化（遵守开放/闭合样本格式）
4. 回答完整性（长度充足、无残缺）

使用方式：
    from src.cleaning.reward_model_judge import RewardModelJudge

    judge = RewardModelJudge()
    score = judge.score(image_path, question, answer, sample)
"""

import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import re
import yaml


class RewardModelJudge:
    """
    多模态打分模型（Qwen-VL Judge）

    使用教师模型进行质量评估，输出 1~5 分
    """

    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        """
        初始化打分模型

        Args:
            config: 配置管理器
            logger: 日志记录器（可选，如果不提供则使用模块默认logger）
        """
        # 🔧 修复：优先使用传入的 logger，确保日志统一
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config
        self.model = None
        self.processor = None
        self._model_loaded = False

        # 加载打分提示词
        self._load_prompts()

        self.logger.info("✓ 多模态打分模型初始化完成（待加载）")

    def _load_prompts(self):
        """
        从配置文件加载打分提示词
        """
        try:
            # 从配置文件读取
            prompts_file = Path(__file__).parent.parent.parent / 'configs' / 'prompts_en.yaml'

            if prompts_file.exists():
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    prompts_config = yaml.safe_load(f)

                self.judge_prompts = prompts_config.get('judge_prompts', {})

                # 🔧 关键修复：检查是否成功加载了有效内容
                if not self.judge_prompts:
                    self.logger.warning("配置文件中缺少 judge_prompts 字段或为空，使用默认提示词")
                    self._use_default_prompts()
                else:
                    self.logger.info("✓ 打分提示词已从配置文件加载")
            else:
                self.logger.warning(f"配置文件不存在: {prompts_file}")
                self._use_default_prompts()

        except Exception as e:
            self.logger.warning(f"加载打分提示词失败: {e}，使用默认提示词")
            self._use_default_prompts()

    def _use_default_prompts(self):
        """
        使用默认的打分提示词（分离 system 和 user）
        🔧 优化：明确打分任务，避免模型混淆为 VQA 任务
        """
        self.judge_prompts = {
            # 闭合问题打分 - system prompt（全局规范）
            'closed_system': """You are a QUALITY SCORING MODEL, NOT a VQA model.
⚠️ CRITICAL TASK DISTINCTION:
- Your task: SCORING sample quality (0 ~ 100 integer score)
- NOT your task: VQA, describing images or answering questions
- If you describe the image → WRONG TASK → Output invalid

Scoring criteria for closed-set VQA distilled sample:
90~100 Perfect: Hard label exists in candidate pool; hard label / soft primary answer / CoT conclusion consistent; zero visual hallucination; complete structured CoT (Observation/Analysis/Conclusion).
70~89 Good: Minor trivial format defects; core labels consistent; no severe hallucination.
50~69 Acceptable: Noticeable defects, core label valid; no serious hallucination.
30~49 Poor: Severe defects (label inconsistency / weak visual grounding).
0~29 Invalid: Heavy hallucination, label out of candidate pool or serious self-inconsistency.

OUTPUT RULE (STRICT, VIOLATION = INVALID SCORE):
- Output ONLY ONE integer between 0 and 100
- NO text, NO explanation, NO image description, NO reasoning
- Any extra words make your score invalid""",

            # 闭合问题打分 - user prompt（动态输入）
            'closed_user': """[SCORING TASK - DO NOT DESCRIBE IMAGE]
Grade this closed-set VQA sample quality (0 - 100):

Question: {question}
Candidates: {candidate_pool}
Predicted hard label: {hard_label}
CoT reasoning: {cot_reasoning}

⚠️ OUTPUT RULE: Return ONLY an integer number, nothing else.
Score:""",

            # 开放问题打分 - system prompt（全局规范）
            'open_system': """You are a QUALITY SCORING MODEL, NOT a VQA model.
⚠️ CRITICAL TASK DISTINCTION:
- Your task: SCORING sample quality (0 ~ 100 integer score)
- NOT your task: VQA, describing images or answering questions
- If you describe the image → WRONG TASK → Output invalid

Scoring criteria for open-ended VQA distilled sample:
90~100 Perfect: Answer strictly grounded on visible image content, no hallucination; complete continuous paragraph; fully respond to the question; no markdown symbols.
70~89 Good: Minor wording imperfections, reliable visual grounding, complete answer paragraph.
50~69 Acceptable: Visible flaws, no heavy hallucination; meets basic answer requirements.
30~49 Poor: Partial hallucination, overly short answer or incomplete reasoning.
0~29 Invalid: Severe hallucination, irrelevant content, single-word fragmented answer.

OUTPUT RULE (STRICT, VIOLATION = INVALID SCORE):
- Output ONLY ONE integer between 0 and 100
- NO text, NO explanation, NO image description, NO reasoning
- Any extra words make your score invalid""",

            # 开放问题打分 - user prompt（动态输入）
            'open_user': """[SCORING TASK - DO NOT DESCRIBE IMAGE]
Grade this open-ended VQA sample quality (0 - 100):

Question: {question}
Distilled Answer: {answer}

⚠️ OUTPUT RULE: Return ONLY an integer number, nothing else.
Score:"""
        }

    def _load_model(self):
        """
        加载打分模型（延迟加载）

        使用教师模型兼任打分器
        """
        if self._model_loaded:
            return

        self.logger.info("正在加载打分模型...")

        try:
            # 使用教师模型
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch

            # 从配置读取模型路径
            model_path = "models/Qwen2.5-VL-32B-Instruct-AWQ"

            self.logger.info(f"加载模型: {model_path}")

            # ───────────────────────────────────────────────────────
            # 🔧 关键修复：关闭视觉特征缓存，防止上下文泄漏
            # ───────────────────────────────────────────────────────
            # Judge 模型必须关闭缓存：
            # 1. 避免上一轮图像特征残留
            # 2. 防止输出格式崩坏
            # 3. 确保每次打分独立
            # ───────────────────────────────────────────────────────

            # 加载模型配置
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # 关闭缓存（关键！）
            config.use_cache = False  # 🔧 关闭 KV 缓存
            if hasattr(config, 'vision_config'):
                config.vision_config.use_cache = False  # 🔧 关闭视觉特征缓存

            # 加载模型（应用配置）
            # 🔧 AWQ 模型修复：明确指定只使用 GPU，避免 CPU 或磁盘分配
            # AWQ 模型不支持 device_map="auto"，必须完全加载到 GPU
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                config=config,  # 🔧 传入配置
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to("cuda")  # 🔧 明确加载到 GPU，不使用 device_map="auto"

            # 加载 processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # 🔧 额外保险：在模型层面再次确认关闭缓存
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False

            self._model_loaded = True
            self.logger.info("✓ 打分模型加载完成（视觉特征缓存已关闭）")

        except Exception as e:
            self.logger.error(f"加载打分模型失败: {e}")
            raise

    def score(
        self,
        image_path: str,
        question: str,
        answer: str,
        sample: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        多模态打分（输出 1~5 分）

        Args:
            image_path: 图像路径
            question: 问题文本
            answer: 答案文本
            sample: 完整样本数据（可选）

        Returns:
            {
                'model_score': float,     # 模型打分（1-5）
                'dimensions': Dict,       # 四维度得分（可选）
                'reasoning': str          # 打分理由（可选）
            }
        """
        # 延迟加载模型
        if not self._model_loaded:
            self._load_model()

        # 确定问题类型
        inference_mode = 'closed'
        if sample:
            inference_mode = sample.get('tasks', {}).get('vqa', {}).get('inference_mode', 'closed')

        # 🔧 修复：正确分离 system 和 user prompt
        # 获取 system prompt（全局规范，固定不变）
        system_key = f'{inference_mode}_system'
        system_prompt = self.judge_prompts.get(system_key, self.judge_prompts.get('closed_system', ''))

        # 获取 user prompt 模板（动态内容）
        user_key = f'{inference_mode}_user'
        user_template = self.judge_prompts.get(user_key, self.judge_prompts.get('closed_user', ''))

        # 🔧 改进：根据不同类型构建 user prompt 内容
        if inference_mode == 'open':
            # 开放问题：image + question + answer
            user_prompt = user_template.format(
                image_path=image_path,
                question=question,
                answer=answer
            )
        else:
            # 闭合问题：image + question + candidate_pool + hard_label + soft_label + cot_reasoning
            # 从 sample 中提取所需信息
            candidate_pool = ''
            hard_label = ''
            soft_label = ''
            cot_reasoning = ''

            if sample:
                vqa_data = sample.get('tasks', {}).get('vqa', {})
                # 提取候选池
                soft_label_data = vqa_data.get('soft_label', {})
                if isinstance(soft_label_data, dict):
                    allowed_answers = soft_label_data.get('allowed_answers', [])
                    candidate_pool = ', '.join(allowed_answers) if allowed_answers else ''

                # 提取 hard label
                hard_label_data = vqa_data.get('hard_label', {})
                if isinstance(hard_label_data, dict):
                    hard_label = hard_label_data.get('answer', '')

                # 提取 soft label distribution
                if isinstance(soft_label_data, dict):
                    answer_dist = soft_label_data.get('answer_distribution', {})
                    if answer_dist:
                        soft_label = ', '.join([f"{k}:{v}" for k, v in answer_dist.items()])

                # 提取 CoT reasoning
                cot_data = vqa_data.get('cot_reasoning', {})
                if isinstance(cot_data, dict):
                    # 🔧 修改：支持新格式（两段式）
                    if 'reasoning_paragraph' in cot_data:
                        # 新格式：reasoning_paragraph + answer
                        cot_reasoning = f"Reasoning: {cot_data.get('reasoning_paragraph', '')}\nAnswer: {cot_data.get('answer', '')}"
                    else:
                        # 旧格式：structured_reasoning
                        structured = cot_data.get('structured_reasoning', {})
                        if structured:
                            cot_reasoning = '\n'.join([
                                f"{k}: {v}" for k, v in structured.items()
                        ])

            # ───────────────────────────────────────────────────────
            # 🔧 关键修复：截断过长的 CoT 内容（避免挤压 assistant 空间）
            # ───────────────────────────────────────────────────────
            # 只保留前 300 字符，减少上下文冗余
            if len(cot_reasoning) > 300:
                cot_reasoning = cot_reasoning[:300] + "..."

            user_prompt = user_template.format(
                image_path=image_path,
                question=question,
                candidate_pool=candidate_pool,
                hard_label=hard_label,
                soft_label=soft_label,
                cot_reasoning=cot_reasoning
            )

        try:
            # 加载图像
            from PIL import Image
            image = Image.open(image_path).convert('RGB')

            # 🔧 修复：正确构建消息结构（分离 system 和 user）
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]

            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # ───────────────────────────────────────────────────────
            # 🔧 调试日志：查看聊天模板输出
            # ───────────────────────────────────────────────────────
            # self.logger.info(f"【聊天模板输出】长度: {len(text)} 字符")
            # self.logger.debug(f"【聊天模板内容】\n{text}")

            # 处理输入
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            # 移动到设备
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # ───────────────────────────────────────────────────────
            # 🔧 关键修复：彻底清空所有缓存和状态
            # ───────────────────────────────────────────────────────
            # 问题：历史上下文、图像特征、KV Cache可能残留
            # 解决：多重清理机制，确保每个样本独立处理
            # ───────────────────────────────────────────────────────
            try:
                # 方法1：重置推理状态（推荐，最彻底）
                if hasattr(self.model, 'reset_inference_state'):
                    self.model.reset_inference_state()
                    self.logger.debug("✓ 重置推理状态")

                # 方法2：清空 past_key_values（KV Cache）
                if hasattr(self.model, 'past_key_values'):
                    self.model.past_key_values = None
                    self.logger.debug("✓ 清空 past_key_values")

                # 方法3：重置注意力缓存
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
                    self.logger.debug("✓ 调用 clear_cache()")

                # 方法4：清空模型所有缓存属性
                for module in self.model.modules():
                    if hasattr(module, 'reset_cache'):
                        module.reset_cache()
                    if hasattr(module, 'clear_cache'):
                        module.clear_cache()

                # 方法5：强制垃圾回收（释放GPU内存碎片）
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 等待所有CUDA操作完成
                    self.logger.debug("✓ CUDA缓存已清空")

                self.logger.debug("✓ 所有缓存清理完成")

            except Exception as e:
                self.logger.warning(f"清空缓存时出现警告: {e}")

            # ───────────────────────────────────────────────────────
            # 生成配置
            # ───────────────────────────────────────────────────────
            # max_new_tokens=8: 严格控制输出长度
            # 打分输出格式："Score: XX" 或 "XX"（仅需3-5个token）
            # ───────────────────────────────────────────────────────
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=8,  # 🔧 严格控制为8，打分输出不需要过长
                    do_sample=False,   # 贪婪解码（确定性输出）
                    top_p=1.0,         # 保持默认（贪婪解码时此参数无效）
                    repetition_penalty=1.15,  # 抑制重复
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=False  # 🔧 关闭缓存，防止上下文泄漏
                )

            # ───────────────────────────────────────────────────────
            # 🔧 关键修复：只解码新生成的部分（不包含输入）
            # ───────────────────────────────────────────────────────
            # outputs[0]: 完整序列 [输入token + 生成token]
            # 需要切片：只取生成的部分
            # ───────────────────────────────────────────────────────
            input_length = inputs['input_ids'].shape[1]  # 输入token数量
            generated_tokens = outputs[0][input_length:]  # 只取生成部分

            generated_text = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            # ───────────────────────────────────────────────────────
            # 🔧 调试日志：查看模型原始输出
            # ───────────────────────────────────────────────────────
            # self.logger.info(f"【模型原始输出】长度: {len(generated_text)} 字符")
            # self.logger.info(f"【模型原始内容】{generated_text}")

            # 提取分数
            score = self._extract_score(generated_text)

            # ───────────────────────────────────────────────────────
            # 关键改进：分数提取失败时返回 None，不使用默认值
            # ───────────────────────────────────────────────────────
            if score is None:
                # 样本标记为 INVALID，不参与后续筛选
                self.logger.warning(
                    f"模型打分输出无有效分数，样本标记为INVALID，丢弃。"
                    f"模型输出: {generated_text[:100]}..."
                )
                return {
                    'model_score': None,  # 明确标记无效
                    'dimensions': {},
                    'reasoning': generated_text,
                    'is_valid': False,  # 标记为无效样本
                    'invalid_reason': '无法从模型输出中提取有效分数'
                }

            return {
                'model_score': score,
                'dimensions': {},  # 简化版本不返回维度分数
                'reasoning': generated_text,
                'is_valid': True  # 成功提取分数
            }

        except Exception as e:
            self.logger.warning(f"模型打分失败: {e}")
            # ───────────────────────────────────────────────────────
            # 失败时返回 None，不使用默认分数
            # ───────────────────────────────────────────────────────
            return {
                'model_score': None,
                'dimensions': {},
                'reasoning': f'模型打分异常: {e}',
                'is_valid': False,
                'invalid_reason': f'模型打分异常: {e}'
            }

    def _extract_score(self, text: str) -> Optional[float]:
        """
        从生成的文本中提取分数（0-100）

        ⚠️ 分数范围：0-100（与 prompt 一致，无需映射）

        Args:
            text: 生成的文本

        Returns:
            分数（0-100），无法提取时返回 None
        """
        import re

        # ───────────────────────────────────────────────────────
        # 策略1：优先匹配 "Score: X" 格式（最准确）
        # ───────────────────────────────────────────────────────
        match = re.search(r'Score:\s*(\d{1,3}(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 100:
                self.logger.info(f"✓ 提取到分数（Score格式）: {score}")
                return score

        # ───────────────────────────────────────────────────────
        # 策略1.5：匹配Markdown格式的数字（如 **90**、*85*）
        # ───────────────────────────────────────────────────────
        # 模型可能输出带Markdown格式的分数
        match = re.search(r'\*{1,2}(\d{1,3})\*{1,2}', text)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 100:
                self.logger.info(f"✓ 提取到分数（Markdown格式）: {score}")
                return score

        # ───────────────────────────────────────────────────────
        # 策略2：匹配文本中的独立数字（宽松策略）
        # ───────────────────────────────────────────────────────
        # 尝试提取文本中所有符合条件的数字，取最后一个
        all_numbers = re.findall(r'\b(\d{1,3})\b', text)
        if all_numbers:
            # 从后往前找，最后一个符合条件的数字通常是分数
            for num_str in reversed(all_numbers):
                score = float(num_str)
                if 0 <= score <= 100:
                    self.logger.info(f"✓ 提取到分数（文本数字）: {score}")
                    return score

        # ───────────────────────────────────────────────────────
        # 策略3：查找行首的独立数字（如 "90\n"）
        # ───────────────────────────────────────────────────────
        lines = text.strip().split('\n')
        for line in reversed(lines):  # 从最后一行开始检查
            line = line.strip()
            if line.isdigit():
                score = float(line)
                if 0 <= score <= 100:
                    self.logger.info(f"✓ 提取到分数（独立行）: {score}")
                    return score

        # ───────────────────────────────────────────────────────
        # 完全无法提取 - 返回 None
        # ───────────────────────────────────────────────────────
        self.logger.warning(
            f"⚠️ 无法从输出中提取有效分数。"
            f"输出内容: {text[:200]}"
        )
        return None

    def is_available(self) -> bool:
        """
        检查模型是否可用

        Returns:
            模型是否可用
        """
        try:
            self._load_model()
            return self._model_loaded
        except:
            return False
