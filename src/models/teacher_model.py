"""
Teacher Model Interface
=======================

Wraps Qwen2.5-VL teacher model for multi-task distillation.
Supports AWQ quantized models via autoawq.
"""

# 🔧 关键：在导入 transformers 之前设置离线模式和禁用 Intel 优化
import os
import sys
from pathlib import Path

# ✅ 禁用 Intel 优化（如果不需要 Intel GPU）
os.environ['DISABLE_IPEX'] = '1'
os.environ['INTEL_IGCL'] = '0'

# 检查是否有本地模型配置
_config_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
if _config_path.exists():
    import yaml
    try:
        with open(_config_path, 'r', encoding='utf-8') as f:
            _cfg = yaml.safe_load(f)
            _model_name = _cfg.get('teacher', {}).get('model_name', '')
            if _model_name:
                # 检查模型路径是否存在
                _model_path = Path(_model_name)
                if not _model_path.is_absolute():
                    _model_path = Path(__file__).parent.parent.parent / _model_name
                if _model_path.exists():
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    except Exception:
        pass

import torch
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from PIL import Image

from ..utils.config import ConfigManager
from ..utils.logger import get_logger

# Lazy imports for backends
_transformers_loaded = False


class TeacherModel:
    """
    Wrapper for Qwen2.5-VL teacher model.

    Supports AWQ quantized models via autoawq.
    Provides multi-task inference capabilities:
    - Visual Question Answering (VQA)
    - Image Captioning
    - Object Detection
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None
    ):
        """
        Initialize Teacher Model.

        Args:
            config: Configuration manager instance
            model_name: Model name or path (default: from config)
            device: Device to load model (cuda/cpu)
            precision: Model precision (fp32/fp16/bf16/4bit)
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Model settings
        self.model_name = model_name or self.config.get("teacher.model_name", "models/Qwen2.5-VL-32B-Instruct-AWQ")
        self.device = device or self.config.get("teacher.device", "cuda")
        self.precision = precision or self.config.get("teacher.precision", "auto")

        # Generation parameters (32B optimized)
        self.max_new_tokens = self.config.get("teacher.max_new_tokens", 768)  # 🔧 32B: 768 (from 512)
        self.temperature = self.config.get("teacher.temperature", 0.0)  # 贪婪解码
        self.top_p = self.config.get("teacher.top_p", 1.0)
        self.top_k = self.config.get("teacher.top_k", 1)

        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None

        # 🔧 视觉特征缓存（性能优化）
        self._visual_cache = {}  # {image_id: visual_features}
        self._cache_enabled = self.config.get("teacher.cache_visual_features", True)
        self._cache_max_size = self.config.get("teacher.cache_max_size", 1000)

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """
        Load Qwen2.5-VL model using transformers + autoawq.

        🔧 注意：Qwen2.5-VL 的 AWQ 模型需要 autoawq 库支持
        transformers 在检测到 quant_method: "awq" 后会调用 autoawq 进行反量化
        """
        self.logger.info(f"Loading teacher model: {self.model_name}")

        # 🔧 GPU选择：从配置读取cuda_devices并设置环境变量
        cuda_devices = self.config.get("teacher.cuda_devices", None)
        if cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
            self.logger.info(f"Using GPU device(s): {cuda_devices}")

        # 🔧 设置随机种子，确保推理结果可复现
        import random
        import numpy as np

        seed = self.config.get("data.seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            self.logger.info(f"CUDA seed set to {seed} for reproducibility")

        # Check if using HuggingFace mirror (for network issues)
        hf_mirror = self.config.get("teacher.hf_mirror", None)
        if hf_mirror:
            self.logger.info(f"Using HuggingFace mirror: {hf_mirror}")
            os.environ['HF_ENDPOINT'] = hf_mirror

        # 🔧 检查模型路径（支持本地路径和 HuggingFace Hub）
        model_path = Path(self.model_name)
        if not model_path.is_absolute():
            model_path = Path.cwd() / self.model_name

        # 设置离线模式（本地模型）
        if model_path.exists():
            self.logger.info(f"✓ Found local model at: {model_path.absolute()}")
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            self.logger.warning(f"Model path not found locally: {self.model_name}")
            self.logger.warning(f"Will try to download from HuggingFace Hub")

        # ==================== Transformers 原生加载 ====================
        # ✅ Qwen 官方标准加载方式，支持 AWQ 模型（无需 autoawq）
        self._load_with_transformers()

    # ==================== Transformers 加载 ====================
        # 🔧 Qwen2.5-VL AWQ 模型需要 autoawq 支持
        self._load_with_transformers()

    def _load_with_transformers(self) -> None:
        """
        Load model using transformers + autoawq.

        🔧 Qwen2.5-VL 的 AWQ 模型：
        - transformers 在检测到 quant_method: "awq" 后会调用 autoawq
        - 需要 autoawq>=0.2.0 版本
        - 使用 device_map="auto" 自动分配 GPU
        """
        self.logger.info("Loading model with transformers + autoawq...")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # ✅ 加载模型（AWQ 模型会自动调用 autoawq 反量化）
        self.logger.info(f"Loading model from: {self.model_name}")
        self.logger.info("Note: AWQ model will use autoawq for dequantization")

        # 🔧 检测是否是AWQ模型
        is_awq = 'awq' in self.model_name.lower()

        if is_awq:
            # AWQ 模型：不支持 device_map="auto" 包含 CPU/disk
            # 使用强制全GPU加载
            self.logger.info("Detected AWQ model, using full GPU loading")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1'
            ).cuda()  # 直接加载到 GPU
        else:
            # 非AWQ模型：可以使用 device_map="auto"
            self.logger.info("Using device_map='auto' for non-AWQ model")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1'
            )

        # 加载 processor
        self.logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1'
        )

        # 加载 tokenizer（从 processor 获取）
        self.tokenizer = self.processor.tokenizer

        # ✅ 关键：验证模型加载成功
        assert self.model is not None, "Model loading failed!"

        self.logger.info("✓ Model loaded successfully")

        # 显示实际设备分配
        if hasattr(self.model, 'hf_device_map'):
            self.logger.info(f"Device map: {self.model.hf_device_map}")
        else:
            # AWQ模型使用 .cuda()，没有 hf_device_map
            self.logger.info(f"Model loaded on: {self.model.device}")

        # 显示模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params / 1e9:.2f}B")

        # 显示显存占用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    def inference_vqa(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        return_logits: bool = False,
        generate_cot: bool = False,
        primary_answer: Optional[str] = None,
        allowed_answers: Optional[List[str]] = None,
        candidate_answers: Optional[List[str]] = None,  # 候选答案集（用于硬标签阶段）
        cache_visual: bool = True,  # 是否缓存视觉特征
        use_cached_visual: bool = False,  # 是否使用缓存的视觉特征
        image_id: Optional[str] = None,  # 图像ID（用于缓存）
        custom_prompt: Optional[str] = None,  # 自定义 prompt（用于开放推理）
        is_open_question: bool = False,  # 是否为开放问题（返回完整答案）
        is_strong_pool: bool = True,  # 是否为强候选池（MUST pick from list）
        question_type: Optional[str] = None  # 🔧 新增：问题类型
    ) -> Dict[str, Any]:
        """
        Perform VQA inference.

        Args:
            image: PIL Image or image path
            question: Question string
            return_logits: Whether to return logits for soft labels
            generate_cot: Whether to generate Chain-of-Thought reasoning
            primary_answer: Reference answer from hard_label (for CoT prompt)
            allowed_answers: List of allowed answers from soft_label (for CoT prompt)
            candidate_answers: List of candidate answers from VQA vocabulary (for hard_label prompt)
            cache_visual: Whether to cache visual features (default: True)
            use_cached_visual: Whether to use cached visual features (default: False)
            image_id: Image ID for caching (auto-extracted from path if not provided)
            custom_prompt: Custom prompt for open inference (optional)
            is_open_question: Whether this is an open question (return full answer instead of first word)
            is_strong_pool: Whether to use strong pool constraint (MUST pick from list) vs weak pool (MAY consider list)
            question_type: Question type (open_descriptive, closed_choice, etc.)

        Returns:
            Dictionary with answer, confidence, and optionally logits/cot
        """
        # Construct prompt
        if custom_prompt:
            # 使用自定义 prompt（开放推理）
            system_prompt = None
            user_prompt = custom_prompt.format(question=question) if '{question}' in custom_prompt else custom_prompt
        elif generate_cot:
            # CoT阶段：使用allowed_answers（从软标签分布中提取）
            system_prompt, user_prompt = self._construct_cot_prompt(
                question, task="vqa", primary_answer=primary_answer,
                allowed_answers=allowed_answers, is_strong_pool=is_strong_pool,
                question_type=question_type  # 🔧 新增：传递问题类型
            )
        else:
            # 硬标签阶段：使用candidate_answers（从VQA词表得到）
            system_prompt = None
            user_prompt = self._construct_prompt(question, task="vqa", candidate_answers=candidate_answers, is_strong_pool=is_strong_pool)

        # Transformers 推理
        return self._inference_vqa_transformers(
            image, user_prompt, system_prompt,
            return_logits, generate_cot,
            cache_visual=cache_visual,
            use_cached_visual=use_cached_visual,
            image_id=image_id,
            is_open_question=is_open_question  # 传递开放问题标志
        )

    def inference_vqa_with_teacher_forcing(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        ground_truth: str,
        top_k_per_position: int = 5,
        max_sequences: int = 20,
        min_prob_threshold: float = 0.01,
        image_id: Optional[str] = None,
        use_cached_visual: bool = False,
        # 🔧 新增参数
        question_type: str = "open",
        candidate_pool: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        使用Teacher Forcing计算多token答案的序列级概率分布。

        🔧 解决方案：方案3 - Teacher Forcing（推荐用于有GT的情况）

        核心思想：
        - 使用ground_truth作为提示
        - 强制模型生成正确答案
        - 计算序列级概率分布

        适用场景：
        - ✅ 有ground_truth的样本
        - ✅ 需要计算多token联合概率（如"hot dog"、"elephant abuse"）
        - ✅ 需要高质量软标签

        Args:
            image: PIL Image or image path
            question: Question text
            ground_truth: Ground truth answer (e.g., "hotdog", "elephant abuse")
            top_k_per_position: Number of top tokens to keep per position (default: 5)
            max_sequences: Maximum number of sequences to return (default: 20)
            min_prob_threshold: Minimum probability threshold for pruning (default: 0.01)
            image_id: Image ID for caching
            use_cached_visual: Whether to use cached visual features

        Returns:
            {
                'sequence_distribution': Dict[str, float],  # 序列级概率分布
                'gt_joint_prob': float,  # GT的联合概率
                'num_tokens': int,  # GT的token数量
                'position_distributions': List[Dict],  # 每个位置的分布（可选）
            }

        Example:
            >>> result = model.inference_vqa_with_teacher_forcing(
            ...     image="image.jpg",
            ...     question="What is the man eating?",
            ...     ground_truth="hotdog"
            ... )
            >>> print(result['sequence_distribution'])
            {
                "hotdog": 0.72,
                "hot cake": 0.04,
                "warm dog": 0.09,
                ...
            }
        """
        import torch.nn.functional as F
        from itertools import product

        self.logger.info(f"[Teacher Forcing] 开始计算序列概率，GT: '{ground_truth}'")

        # ───────────────────────────────────────────────────────
        # Step 1: Tokenize ground_truth
        # ───────────────────────────────────────────────────────
        tokens_gt = self.tokenizer.encode(ground_truth, add_special_tokens=False)
        seq_length = len(tokens_gt)

        self.logger.info(f"[Teacher Forcing] GT token数: {seq_length}, tokens: {tokens_gt}")

        # ───────────────────────────────────────────────────────
        # Step 2: 构建Prompt（简化版，直接预测答案）
        # ───────────────────────────────────────────────────────
        # 🔧 关键：Teacher Forcing只预测答案部分，不需要[Reasoning]
        # 原因：逐位置预测答案token，不需要生成推理过程
        # ───────────────────────────────────────────────────────

        # 根据问题类型构建不同的prompt（简化版）
        if question_type in ["closed_choice", "closed_yesno"]:
            # 闭合问题（有候选池）
            if not candidate_pool:
                self.logger.warning(f"[Teacher Forcing] ⚠️ {question_type} 需要候选池，但未提供")
                candidate_pool = []

            user_prompt = (
                f"Question: {question}\n"
                f"CANDIDATE LIST: {', '.join(candidate_pool)}\n"
                f"Answer: "  # 🔧 修复：添加空格，明确告诉模型开始生成答案
            )
        else:
            # 开放问题或枚举类闭合问题
            user_prompt = (
                f"Question: {question}\n"
                f"Answer: "  # 🔧 修复：添加空格，明确告诉模型开始生成答案
            )

        system_prompt = None  # 不使用system prompt

        # ───────────────────────────────────────────────────────
        # Step 3: Teacher Forcing推理（逐位置）
        # ───────────────────────────────────────────────────────
        position_distributions = []
        position_distributions_ids = []  # 🔧 修复：初始化变量

        for pos in range(seq_length):
            self.logger.debug(f"[Teacher Forcing] 处理位置 {pos}/{seq_length-1}")

            # Teacher forcing: 使用GT的前pos个token作为prefix
            # 🔧 注意：user_prompt已经包含空格，直接拼接prefix即可
            if pos > 0:
                prefix_tokens = tokens_gt[:pos]
                prefix_text = self.tokenizer.decode(prefix_tokens)
                # 🔧 修复：user_prompt已经有空格，直接拼接，不要再加空格
                current_prompt = f"{user_prompt}{prefix_text}"
            else:
                # 位置0：直接使用user_prompt（已包含空格）
                current_prompt = user_prompt

            # 🔧 DEBUG: 打印实际使用的prompt
            if pos == 0:
                self.logger.info(f"[Teacher Forcing] 位置0的完整Prompt:\n{current_prompt}")
                self.logger.info(f"[Teacher Forcing] Prompt长度: {len(self.tokenizer.encode(current_prompt))} tokens")

            # 推理得到logits
            # ⚠️ 注意：不使用缓存，每次都重新处理图像
            # 原因：缓存的视觉特征需要图像占位符token，但文本输入中没有
            # Teacher Forcing本身就是慢速高质量方案，这个性能损失可接受
            inputs = self._prepare_inputs(
                image, current_prompt, system_prompt=system_prompt,
                use_cached_visual=False,  # ← 不使用缓存
                image_id=image_id
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 提取最后一个位置的logits
            logits = outputs.logits[0, -1, :]  # [vocab_size]

            # ───────────────────────────────────────────────────────
            # Step 4: 提取top-k token
            # ───────────────────────────────────────────────────────
            # 🔧 修复：直接记录 token_id 分布，而不是 token_text
            # ───────────────────────────────────────────────────────
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k_per_position)

            # ✅ 修复：构建 token_id 分布（而不是token文本）
            position_dist_ids = {}
            for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
                token_id = int(idx)
                position_dist_ids[token_id] = prob

            position_distributions_ids.append(position_dist_ids)

            # 保留文本版本用于日志
            position_dist_text = {self.tokenizer.decode([tid]): prob for tid, prob in position_dist_ids.items()}
            position_distributions.append(position_dist_text)

            # 🔧 DEBUG: 打印每个位置的top-k token
            self.logger.info(f"[Teacher Forcing] 位置{pos} top-{top_k_per_position} tokens:")
            for i, (token_id, prob) in enumerate(list(position_dist_ids.items())[:5], 1):
                token_text = self.tokenizer.decode([token_id])
                self.logger.info(f"  {i}. ID:{token_id} → '{token_text}' (prob={prob:.4f})")

            self.logger.debug(
                f"[Teacher Forcing] 位置{pos} top-{top_k_per_position}: "
                f"{list(position_dist_text.keys())[:3]}..."
            )

        # ───────────────────────────────────────────────────────
        # Step 5: 组合所有位置，生成序列分布
        # ───────────────────────────────────────────────────────
        sequence_distribution = {}
        combinations_tried = 0

        for combination in product(*[dist.keys() for dist in position_distributions_ids]):
            combinations_tried += 1

            # 计算联合概率
            joint_prob = 1.0
            for pos, token_id in enumerate(combination):
                joint_prob *= position_distributions_ids[pos].get(token_id, 0.0)

            # 剪枝：跳过低概率组合
            if joint_prob < min_prob_threshold:
                continue

            # ✅ 修复：使用tokenizer解码token ID序列
            sequence_text = self.tokenizer.decode(list(combination), skip_special_tokens=True).strip()

            # 合并同义序列
            if sequence_text not in sequence_distribution:
                sequence_distribution[sequence_text] = joint_prob
            else:
                sequence_distribution[sequence_text] += joint_prob

        # ───────────────────────────────────────────────────────
        # Step 6: 归一化 + Top-K截断
        # ───────────────────────────────────────────────────────
        total = sum(sequence_distribution.values())
        if total > 0:
            sequence_distribution = {
                k: v / total for k, v in sequence_distribution.items()
            }

        # 按概率排序，保留top-k
        sorted_sequences = sorted(
            sequence_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_sequences]

        sequence_distribution = dict(sorted_sequences)

        # ───────────────────────────────────────────────────────
        # Step 7: 计算GT的联合概率
        # ───────────────────────────────────────────────────────
        gt_joint_prob = 1.0
        for pos in range(seq_length):
            gt_token_id = tokens_gt[pos]  # GT的token ID
            # 在位置分布中查找GT token的概率
            gt_prob = position_distributions_ids[pos].get(gt_token_id, 0.0)
            gt_joint_prob *= gt_prob

            if gt_prob == 0.0:
                self.logger.warning(
                    f"[Teacher Forcing] GT token '{self.tokenizer.decode([gt_token_id])}' "
                    f"(ID: {gt_token_id}) 不在位置 {pos} 的top-{top_k_per_position}中"
                )

        self.logger.info(
            f"[Teacher Forcing] 完成: {len(sequence_distribution)}个序列, "
            f"GT联合概率: {gt_joint_prob:.4f}, "
            f"组合数: {combinations_tried}"
        )

        return {
            'sequence_distribution': sequence_distribution,
            'gt_joint_prob': gt_joint_prob,
            'num_tokens': seq_length,
            'position_distributions': position_distributions
        }

    def _inference_vqa_transformers(
        self,
        image: Union[Image.Image, str, Path],
        user_prompt: str,
        system_prompt: Optional[str],
        return_logits: bool,
        generate_cot: bool,
        cache_visual: bool = True,  # 🔧 新增参数
        use_cached_visual: bool = False,  # 🔧 新增参数
        image_id: Optional[str] = None,  # 🔧 新增参数
        is_open_question: bool = False  # 🔧 新增：是否为开放问题
    ) -> Dict[str, Any]:
        """Transformers 后端的 VQA 推理"""
        # Prepare inputs
        inputs = self._prepare_inputs(
            image, user_prompt, system_prompt=system_prompt,
            use_cached_visual=use_cached_visual,  # 🔧 使用缓存参数
            image_id=image_id if cache_visual or use_cached_visual else None
        )

        # Generate
        outputs = self._generate(inputs, return_logits=return_logits)

        # Process outputs
        result = self._process_vqa_outputs(outputs, return_logits, is_open_question)

        return result

    def _construct_prompt(self, question: str, task: str, candidate_answers: Optional[List[str]] = None, is_strong_pool: bool = True) -> str:
        """
        Construct task-specific prompt from configuration file.

        🔧 改进：支持候选答案集（candidate_answers），用于硬标签生成阶段
        🔧 新增：根据候选池类型选择不同的prompt（强候选池 vs 弱候选池）

        概念区分：
        - candidate_answers: 从VQA词表/训练集得到的预定义答案集，用于引导模型
        - allowed_answers: 从软标签分布中提取的可能答案，用于CoT生成阶段
        - is_strong_pool: 强候选池（MUST pick from list）vs 弱候选池（MAY consider list）

        Args:
            question: Question for VQA (empty for other tasks)
            task: Task type (vqa/captioning/detection/keypoints)
            candidate_answers: List of candidate answers from VQA vocabulary (optional)
            is_strong_pool: Whether to use strong pool constraint (default: True)

        Returns:
            Formatted prompt string
        """
        # 🔧 根据候选池类型选择不同的prompt模板
        if is_strong_pool:
            # 强候选池：closed_choice / closed_yesno
            prompt_key = f'prompts.standard.{task}_strong'
            fallback_key = f'prompts.standard.{task}'
            pool_type_log = "强候选池（MUST pick from list）"
        else:
            # 弱候选池：closed_enumerate (counting/color/location)
            prompt_key = f'prompts.standard.{task}_weak'
            fallback_key = f'prompts.standard.{task}'
            pool_type_log = "弱候选池（MAY consider list）"

        # 从配置文件读取 prompt
        prompt_template = self.config.get(
            prompt_key,
            self.config.get(fallback_key, "Analyze this image.")
        )

        # 调试日志：显示实际使用的 prompt
        self.logger.debug(f"Loading prompt for task '{task}' from config ({pool_type_log})")
        self.logger.debug(f"Prompt template (first 100 chars): {prompt_template[:100]}")

        # 支持变量插值 - 使用replace代替format避免大括号冲突
        try:
            prompt = prompt_template
            if '{question}' in prompt:
                prompt = prompt.replace('{question}', question)
                self.logger.debug(f"Formatted prompt with question: {question}")

            # 🔧 新增：支持候选答案集（candidate_answers）
            if '{candidate_answers}' in prompt and candidate_answers:
                candidate_answers_str = ', '.join(candidate_answers[:20])  # 只显示前20个，避免prompt过长
                prompt = prompt.replace('{candidate_answers}', candidate_answers_str)
                self.logger.debug(f"Formatted prompt with candidate_answers: {len(candidate_answers)} candidates")
        except Exception as e:
            self.logger.warning(f"Prompt template error: {e}")
            prompt = prompt_template

        return prompt.strip()

    def _construct_cot_prompt(self, question: str, task: str, primary_answer: Optional[str] = None, allowed_answers: Optional[List[str]] = None, is_strong_pool: bool = True, question_type: Optional[str] = None) -> tuple:
        """
        Construct Chain-of-Thought prompt with system/user role separation.

        🔧 新方案：一次推理同时生成logits和CoT
        根据问题类型选择统一的prompt：
        - 开放问题：prompts.open.vqa_system/user
        - 有候选集闭合：prompts.closed_with_candidates.vqa_system/user
        - 枚举类闭合：prompts.closed_enumerate.vqa_system/user

        Args:
            question: Question for VQA
            task: Task type
            primary_answer: Reference answer from hard_label (optional)
            allowed_answers: List of allowed answers from soft_label (optional)
            is_strong_pool: Whether to use strong pool constraint (default: True)
            question_type: Question type (open_descriptive, closed_choice, etc.)

        Returns:
            (system_prompt, user_prompt) tuple
        """
        # ───────────────────────────────────────────────────────
        # 🔧 新方案：根据问题类型选择统一prompt
        # ───────────────────────────────────────────────────────
        # 问题类型映射：
        # - open_descriptive → 开放问题
        # - closed_choice / closed_yesno → 有候选集闭合
        # - closed_enumerate (counting/color/location) → 枚举类闭合
        # ───────────────────────────────────────────────────────

        # 标准化问题类型
        if question_type == 'open_descriptive' or question_type == 'open':
            # 开放问题
            system_key = 'prompts.open.vqa_system'
            user_key = 'prompts.open.vqa_user'
            pool_type_log = "开放问题（无候选池）"
        elif question_type in ['closed_choice', 'closed_yesno', 'choice', 'binary', 'yes_no']:
            # 有候选集的闭合问题
            system_key = 'prompts.closed_with_candidates.vqa_system'
            user_key = 'prompts.closed_with_candidates.vqa_user'
            pool_type_log = "有候选集闭合（必须从列表选择）"
        else:
            # 枚举类闭合问题（counting/color/location）或未知类型
            system_key = 'prompts.closed_enumerate.vqa_system'
            user_key = 'prompts.closed_enumerate.vqa_user'
            pool_type_log = f"枚举类闭合（参考答案可自由输出）"

        # 🔧 从配置读取 system 规则和 user 模板（分离存储）
        system_template = self.config.get(
            system_key,
            self.config.get('prompts.cot.vqa_system', "Analyze this image step by step.")
        )
        user_template = self.config.get(
            user_key,
            self.config.get('prompts.cot.vqa_user', "{question}")
        )

        self.logger.debug(f"Loading unified prompt from config ({pool_type_log})")
        self.logger.debug(f"  - system_key: {system_key}")
        self.logger.debug(f"  - user_key: {user_key}")

        # 🔧 system 消息：通用硬性规则（无需变量替换）
        system_prompt = system_template.strip()

        # 🔧 user 消息：单条样本专属参数（需要变量替换）
        user_prompt = user_template
        try:
            # 替换问题
            if '{question}' in user_prompt:
                user_prompt = user_prompt.replace('{question}', question)

            # 替换候选答案（用于有候选集的闭合问题）
            if '{candidate_answers}' in user_prompt:
                if allowed_answers:
                    candidates_str = ', '.join([str(a) for a in allowed_answers[:20]])
                else:
                    candidates_str = 'N/A'
                user_prompt = user_prompt.replace('{candidate_answers}', candidates_str)

            # 替换参考答案（用于开放问题）
            if '{primary_answer}' in user_prompt and primary_answer:
                user_prompt = user_prompt.replace('{primary_answer}', primary_answer)

            # 替换答案分布（可选）
            if '{answer_distribution}' in user_prompt:
                user_prompt = user_prompt.replace('{answer_distribution}', 'See reasoning')

            self.logger.debug(f"Formatted user prompt for task '{task}'")
        except Exception as e:
            self.logger.warning(f"CoT user prompt error: {e}")

        return system_prompt, user_prompt.strip()

    def _prepare_inputs(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cached_visual: bool = False,
        image_id: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model inference.

        Args:
            image: PIL Image or path
            prompt: Text prompt
            system_prompt: Optional system prompt
            use_cached_visual: Whether to use cached visual features (default: False)
            image_id: Image ID for caching (optional)

        Returns:
            Dictionary of input tensors
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            # 🔧 自动提取image_id（如果未提供）
            if not image_id:
                image_id = Path(str(image)).stem

            # 加载图像
            image = Image.open(image).convert('RGB')

        # 🔧 检查缓存（如果启用）
        if use_cached_visual and image_id and image_id in self._visual_cache:
            self.logger.debug(f"[Cache] Using cached visual features for image {image_id}")
            cached_features = self._visual_cache[image_id]

            # 使用缓存的视觉特征，只处理文本
            inputs = self._prepare_text_inputs(prompt, system_prompt)
            inputs.update(cached_features)

            return inputs

        # 🔧 正常处理（无缓存或缓存未命中）
        messages = []

        # 添加 system 消息（如果提供）
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 添加 user 消息（包含图片和问题）
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # 🔧 缓存视觉特征（如果启用且未命中）
        if self._cache_enabled and image_id and image_id not in self._visual_cache:
            visual_features = self._extract_visual_features(inputs)
            self._add_to_cache(image_id, visual_features)
            self.logger.debug(f"[Cache] Cached visual features for image {image_id}")

        return inputs

    def _prepare_text_inputs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare text-only inputs (without image).

        Used when visual features are cached.

        Args:
            prompt: Text prompt
            system_prompt: Optional system prompt

        Returns:
            Dictionary of text input tensors
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process text inputs only
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        return inputs

    def _extract_visual_features(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract visual features from inputs for caching.

        Args:
            inputs: Full input tensors

        Returns:
            Dictionary containing only visual features:
            - pixel_values: Image pixel values
            - image_grid_thw: Image grid dimensions
            - image_sizes: Image sizes (if available)
        """
        visual_keys = ['pixel_values', 'image_grid_thw', 'image_sizes']
        visual_features = {}

        for key in visual_keys:
            if key in inputs:
                visual_features[key] = inputs[key]

        return visual_features

    def _add_to_cache(self, image_id: str, visual_features: Dict[str, torch.Tensor]) -> None:
        """
        Add visual features to cache with LRU eviction.

        Args:
            image_id: Image identifier
            visual_features: Visual features to cache
        """
        # Check cache size limit
        if len(self._visual_cache) >= self._cache_max_size:
            # Evict oldest entry (FIFO strategy)
            oldest_key = next(iter(self._visual_cache))
            del self._visual_cache[oldest_key]
            self.logger.debug(f"[Cache] Evicted entry: {oldest_key} (LRU)")

        # Add to cache
        self._visual_cache[image_id] = visual_features

    def clear_cache(self) -> None:
        """Clear visual features cache."""
        self._visual_cache.clear()
        self.logger.info("[Cache] Visual features cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache info:
            - enabled: Whether cache is enabled
            - size: Current cache size
            - max_size: Maximum cache size
            - usage_percent: Cache usage percentage
        """
        return {
            'enabled': self._cache_enabled,
            'size': len(self._visual_cache),
            'max_size': self._cache_max_size,
            'usage_percent': len(self._visual_cache) / self._cache_max_size * 100 if self._cache_max_size > 0 else 0
        }

    def _generate(
        self,
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate outputs from model.

        改进：当return_logits=True时，不应用温度缩放
        让soft_label在提取后手动应用温度，避免分布过于尖锐

        Args:
            inputs: Input tensors
            return_logits: Whether to return logits
            max_new_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            top_p: Top-p sampling parameter (overrides default)

        Returns:
            Generation outputs
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p

        # Generation config
        # 🔧 关键修复：当return_logits=True时，强制使用temperature=0（确定性生成）
        # 原因：如果temperature > 0，模型可能通过采样选择非top-1的token
        #      导致生成答案和logits top-1不一致（硬标签≠logits的bug）
        # 解决：使用temperature=0，确保生成答案一定来自logits的argmax
        if return_logits:
            # 强制使用确定性生成
            temperature = 0.0  # 或使用极小值如0.01
            self.logger.debug(f"Using temperature={temperature} for deterministic generation (return_logits=True)")

        gen_config = {
            'max_new_tokens': max_new_tokens,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        # 🔧 关键修复：只在采样时传递temperature和top_p参数
        # 当temperature=0时，使用贪婪解码（do_sample=False），不需要这些参数
        if temperature > 0:
            gen_config['temperature'] = temperature
            gen_config['top_p'] = top_p
            gen_config['top_k'] = self.top_k
            gen_config['do_sample'] = True
        else:
            # temperature=0时，使用贪婪解码，不传递temperature/top_p避免警告
            gen_config['do_sample'] = False  # 贪婪解码

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config,
                output_scores=return_logits,
                return_dict_in_generate=True,
            )

        return outputs

    def _process_vqa_outputs(
        self,
        outputs: Dict[str, Any],
        return_logits: bool,
        is_open_question: bool = False  # 🔧 新增：是否为开放问题
    ) -> Dict[str, Any]:
        """
        Process VQA generation outputs.

        Args:
            outputs: Model outputs
            return_logits: Whether logits are included
            is_open_question: Whether this is an open question (return full answer)

        Returns:
            Processed VQA result
        """
        # transformers 输出格式
        generated_ids = outputs.sequences
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        answer = self._extract_answer(generated_text, is_open_question)

        # 🔧 计算置信度
        confidence = 0.8  # 默认值
        if hasattr(outputs, 'scores') and outputs.scores:
            try:
                # 从第一个生成的 token 计算 confidence
                first_token_logits = outputs.scores[0]
                probs = torch.softmax(first_token_logits[0], dim=-1)
                max_prob = probs.max().item()
                confidence = max_prob
            except Exception as e:
                self.logger.debug(f"Failed to compute confidence: {e}")

        result = {
            'full_response': generated_text,
            'answer': answer,
            'sequences': generated_ids[0].cpu().tolist(),
            'confidence': confidence,
        }

        if return_logits:
            # Process logits for soft labels
            logits = self._process_logits(outputs.scores)
            result['logits'] = logits

        return result

    def _extract_answer(self, text: str, is_open_question: bool = False) -> str:
        """
        从VQA响应中提取答案（改进版）

        改进：
        1. 区分开放/闭合问题
        2. 闭合问题：提取第一个词作为简短答案
        3. 开放问题：返回完整的自然语言答案

        Args:
            text: 生成的文本
            is_open_question: 是否为开放问题（返回完整答案）

        Returns:
            清理后的答案（开放：完整文本，闭合：第一个单词）
        """
        import re

        # 去掉常见的系统提示前缀
        prefixes_to_remove = [
            "assistant\n",
            "Assistant\n",
            "ASSISTANT\n",
            "Assistant:",
            "assistant:",
            "Answer:",
            "answer:",
        ]

        cleaned_text = text.strip()

        # 去掉前缀
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
            elif prefix in cleaned_text:
                # 如果前缀在中间，取后面的部分
                parts = cleaned_text.split(prefix)
                if len(parts) > 1:
                    cleaned_text = parts[-1].strip()

        # 🔧 关键：区分开放/闭合问题
        if is_open_question:
            # 开放问题：返回完整文本，不做截断
            return cleaned_text

        # 闭合问题：提取第一个词（最可能的简短答案）
        words = cleaned_text.split()

        if words:
            # 取第一个词
            answer = words[0]

            # 去掉标点符号
            answer = re.sub(r'[^\w]', '', answer)

            # 转小写
            answer = answer.lower()

            return answer

        # 如果没有词，返回空字符串
        return ""

    def _extract_caption(self, text: str) -> str:
        """Extract caption from captioning response."""
        # Remove any meta-commentary
        lines = text.split("\n")
        caption_lines = [l for l in lines if not l.startswith("Let's") and not l.startswith("First")]
        caption = " ".join(caption_lines).strip()
        return caption

    def _clean_json_content(self, json_str: str) -> str:
        """
        Clean JSON content by removing extra text after the JSON structure.

        🔧 新增方法：清理JSON后面的多余文本

        Args:
            json_str: JSON string that may contain extra text

        Returns:
            Cleaned JSON string
        """
        import re

        # 去掉前后空白
        json_str = json_str.strip()

        # 策略：找到完整的JSON结构（从第一个 { 或 [ 到最后一个匹配的 } 或 ]）

        # 判断JSON是对象还是数组
        if json_str.startswith('{'):
            # 对象格式：找到最后一个匹配的 }
            brace_count = 0
            last_valid_pos = 0

            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
                        break

            if last_valid_pos > 0:
                return json_str[:last_valid_pos]

        elif json_str.startswith('['):
            # 数组格式：找到最后一个匹配的 ]
            bracket_count = 0
            last_valid_pos = 0

            for i, char in enumerate(json_str):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        last_valid_pos = i + 1
                        break

            if last_valid_pos > 0:
                return json_str[:last_valid_pos]

        # Fallback: 使用正则表达式匹配完整的JSON
        # 匹配 {...} 或 [...]
        json_patterns = [
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',  # 匹配嵌套的 {}
            r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]',  # 匹配嵌套的 []
        ]

        for pattern in json_patterns:
            match = re.search(pattern, json_str, re.DOTALL)
            if match:
                return match.group(0)

        # 如果都无法匹配，返回原始字符串（让后续逻辑处理）
        return json_str

    def _repair_json(self, json_str: str) -> Optional[str]:
        """
        Attempt to repair common JSON syntax errors in detection responses.

        Handles common JSON syntax errors.

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            Repaired JSON string, or None if repair failed
        """
        import re

        try:
            repaired = json_str

            # 🔧 新增：先清理多余文本
            repaired = self._clean_json_content(repaired)

            # Error 0: Missing key name before array - {"category": "person", [0, 150, 38, 387], "confidence": 0.8}
            # Pattern: comma followed by array without key name
            if re.search(r',\s*\[', repaired):
                self.logger.debug("Detected: missing 'bbox' key before array")
                # Fix: Insert "bbox": before arrays that follow a comma
                # Pattern: ", [...]" → ", "bbox": [...]"
                repaired = re.sub(r',\s*\[', ', "bbox": [', repaired)
                self.logger.debug("Added missing 'bbox' key before array")

            # Error 00: Missing key name after comma - {"category": "person", "bbox": [18, 195, 154, 387], "confidence": 0.9}[0, 150, 38, 387]
            # Pattern: } followed by [ without key
            if re.search(r'\}\s*\[', repaired):
                self.logger.debug("Detected: array after object without key")
                # This is malformed: two objects merged incorrectly
                # Try to split them
                repaired = re.sub(r'\}\s*\[', '},\n{', repaired)
                self.logger.debug("Split merged objects")
                self.logger.debug("Detected: objects outside array (missing bracket)")
                # Try to extract all objects and rebuild
                object_pattern = r'\{[^{}]*"category":\s*"[^"]*"[^{}]*\}'
                objects = re.findall(object_pattern, repaired)
                if objects:
                    repaired = '{"objects": [' + ', '.join(objects) + ']}'
                    self.logger.debug(f"Repaired: extracted {len(objects)} objects")

            # Error 2: Truncated confidence values - "confidence": 0.
            # Pattern: incomplete number at the end
            if re.search(r'"confidence":\s*0\.$', repaired):
                self.logger.debug("Detected: truncated confidence value")
                repaired = re.sub(r'"confidence":\s*0\.$', '"confidence": 0.5}', repaired)

            # Error 3: Truncated JSON - missing closing brackets at the end
            open_brackets = repaired.count('{') - repaired.count('}')
            open_array = repaired.count('[') - repaired.count(']')

            if open_brackets > 0 or open_array > 0:
                self.logger.debug(f"Detected: missing closing brackets (braces:{open_brackets}, arrays:{open_array})")
                # Add missing closing brackets/braces
                repaired += ']' * open_array + '}' * open_brackets

            # Error 4: Extra trailing comma before closing bracket - [obj1, obj2, ]
            repaired = re.sub(r',\s*]', ']', repaired)
            repaired = re.sub(r',\s*}', '}', repaired)

            # Error 5: Missing quotes on property names - {category: "cat"}
            # Pattern: property name without quotes
            if re.search(r'\{[a-z_]+:', repaired):
                self.logger.debug("Detected: missing quotes on property names")
                # Add quotes to property names
                property_pattern = r'(\{|\,)\s*([a-z_]+)\s*:'
                repaired = re.sub(property_pattern, r'\1"\2":', repaired)

            # Error 6: Mixed bracket types - [obj1}, obj2]
            # Replace mismatched brackets in arrays
            if '[' in repaired:
                # Find array boundaries and fix internal brackets
                array_start = repaired.find('[')
                array_end = repaired.rfind(']')
                if array_start < array_end:
                    array_content = repaired[array_start:array_end+1]
                    # Replace } with ] inside arrays (except for object boundaries)
                    # This is tricky - need to preserve object {}
                    # Simple heuristic: count depth and fix mismatches
                    depth = 0
                    fixed_array = []
                    for char in array_content:
                        if char == '[':
                            depth += 1
                            fixed_array.append(char)
                        elif char == ']':
                            depth -= 1
                            fixed_array.append(char)
                        elif char == '{':
                            fixed_array.append(char)
                        elif char == '}' and depth > 0:
                            # Might be mismatched - check context
                            fixed_array.append(char)
                        else:
                            fixed_array.append(char)
                    repaired = repaired[:array_start] + ''.join(fixed_array) + repaired[array_end+1:]

            # Error 7: Multiple "objects" wrappers - {"objects": [obj1], ["objects": [obj2], ...]
            # This is a special format where teacher model repeats "objects" wrapper for each object
            if repaired.count('"objects"') > 1 or repaired.count('"objects":') > 1:
                self.logger.debug("Detected: multiple 'objects' wrappers (repeated structure)")

                # Strategy: Extract all complete object definitions and rebuild
                # Pattern to match individual objects: {"category": "...", "bbox_2d": [...], "confidence": ...}
                object_pattern = r'\{[^{}]*"category":\s*"[^"]*"[^{}]*"bbox(?:_2d)?":\s*\[[^\]]+\][^{}]*"confidence":\s*[\d.]+[^{}]*\}'

                # Try to find all complete object definitions
                extracted_objects = re.findall(object_pattern, repaired)

                if extracted_objects:
                    self.logger.debug(f"Extracted {len(extracted_objects)} objects from repeated wrappers")
                    # Rebuild as proper format
                    repaired = '{"objects": [' + ', '.join(extracted_objects) + ']}'
                    self.logger.debug(f"Rebuilt JSON with {len(extracted_objects)} objects")
                else:
                    # Fallback: Try simpler extraction
                    # Look for patterns like {"category": "name", ...}
                    simple_pattern = r'\{"category":\s*"([^"]+)"[^}]+\}'
                    matches = re.finditer(simple_pattern, repaired)

                    extracted_objects = []
                    for match in matches:
                        obj_str = match.group(0)
                        # Validate it has bbox
                        if 'bbox' in obj_str or 'bbox_2d' in obj_str:
                            extracted_objects.append(obj_str)

                    if extracted_objects:
                        repaired = '{"objects": [' + ', '.join(extracted_objects) + ']}'
                        self.logger.debug(f"Rebuilt JSON using simple extraction with {len(extracted_objects)} objects")

            # Pattern: comma followed directly by array (missing key name)
            if re.search(r',\s*\[\d+', repaired):
                self.logger.debug("Detected: missing 'bbox' key name before array")
                # Fix: Insert "bbox": before arrays that follow a comma
                # Pattern: ", [...]" → ", "bbox": [...]"
                repaired = re.sub(r',\s*\[', ', "bbox": [', repaired)
                self.logger.debug("Added missing 'bbox' key name")

            # Pattern: "bbox" followed by space and array (missing colon)
            if re.search(r'"bbox"\s*\[', repaired):
                self.logger.debug("Detected: missing colon after 'bbox'")
                # Fix: Add colon between "bbox" and array
                # Pattern: "bbox [...]" → "bbox": [...]"
                repaired = re.sub(r'"bbox"\s*\[', '"bbox": [', repaired)
                self.logger.debug("Added missing colon after 'bbox'")

            # Pattern: "bbox=" followed by quoted coordinates
            if re.search(r'"bbox="', repaired):
                self.logger.debug("Detected: wrong bbox format 'bbox=\"...\"'")
                # Fix: Replace "bbox="..." with "bbox": [...]
                # Pattern: "bbox="169, 172, 194, 277" → "bbox": [169, 172, 194, 277]
                # Use non-greedy match to capture coordinates between quotes
                repaired = re.sub(r'"bbox="([^"]+)"', r'"bbox": [\1]', repaired)
                self.logger.debug("Fixed bbox format from 'bbox=\"...\"' to 'bbox\": [...]'")

            # Error 8: Malformed bbox format - "bbox="bbox_2d": or similar nested errors
            if re.search(r'"bbox="?bbox', repaired):
                self.logger.debug("Detected: malformed nested bbox format 'bbox=\"bbox_2d\"' or 'bbox=bbox'")
                # Fix: Replace "bbox="bbox_2d": or "bbox=bbox_2d": with "bbox": or "bbox_2d":
                repaired = re.sub(r'"bbox="?bbox_2d":', '"bbox":', repaired)
                repaired = re.sub(r'"bbox="?bbox":', '"bbox":', repaired)
                self.logger.debug("Fixed malformed nested bbox format")

            # Error 9: Unclosed bbox array followed by other fields
            # Pattern: "bbox": [0, 56, 83, 311, \n "confidence": 0.95
            if re.search(r'"bbox":\s*\[[^\]]*\n\s*"[^"]+":', repaired):
                self.logger.debug("Detected: unclosed bbox array with following fields")
                # Fix: Find all bbox arrays and close them before the next field
                # Pattern: "bbox": [numbers,\n → "bbox": [numbers],\n
                repaired = re.sub(
                    r'"bbox":\s*\[([^\]]*?)\n(\s*)"([^"]+)":',
                    r'"bbox": [\1],\n\2"\3":',
                    repaired
                )
                self.logger.debug("Added closing bracket to bbox array before next field")

            # Error 10: Unclosed bbox array followed by newline and confidence
            # Pattern: "bbox": [0, 56, 83, 311,\n "confidence": 0.95
            if re.search(r'"bbox":\s*\[[^\]]+,\s*\n\s*"confidence":', repaired):
                self.logger.debug("Detected: unclosed bbox array before confidence")
                # Fix: Close the bbox array before confidence
                repaired = re.sub(
                    r'"bbox":\s*\[([^\]]+),\s*\n(\s*)"confidence":',
                    r'"bbox": [\1],\n\2"confidence":',
                    repaired
                )
                self.logger.debug("Closed bbox array before confidence field")

            if repaired != json_str:
                self.logger.info(f"JSON repaired successfully")
                self.logger.debug(f"Original: {json_str[:100]}")
                self.logger.debug(f"Repaired: {repaired[:100]}")
                return repaired

            return None  # No repair needed or no errors detected

        except Exception as e:
            self.logger.error(f"Error during JSON repair: {e}")
            return None

    def _extract_objects_from_malformed_json(self, json_str: str) -> List[Dict]:
        """
        Extract objects from severely malformed JSON using regex patterns.

        This is a fallback method when JSON repair fails. It attempts to
        extract object information using pattern matching.

        Args:
            json_str: Malformed JSON string

        Returns:
            List of extracted objects (may be incomplete)
        """
        import re

        objects = []

        try:
            # Strategy 1: Extract objects with all three required fields
            # Pattern: {"category": "...", "bbox": [...], "confidence": ...}
            # Allow for various formats and missing brackets

            # Find all category names
            category_pattern = r'"category":\s*"([^"]+)"'
            categories = re.findall(category_pattern, json_str)

            # Find all bbox arrays (may be malformed)
            # Try to extract numbers from bbox arrays
            bbox_pattern = r'"bbox(?:_2d)?":\s*\[([\d,\s]+)'
            bbox_matches = re.findall(bbox_pattern, json_str)

            # Find all confidence values
            confidence_pattern = r'"confidence":\s*([\d.]+)'
            confidences = re.findall(confidence_pattern, json_str)

            # If we have matching counts, try to build objects
            if len(categories) == len(bbox_matches) == len(confidences):
                for i in range(len(categories)):
                    try:
                        # Parse bbox numbers
                        bbox_str = bbox_matches[i].strip()
                        if bbox_str.endswith(','):
                            bbox_str = bbox_str[:-1]
                        bbox_numbers = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]

                        if len(bbox_numbers) >= 4:
                            obj = {
                                'category': categories[i],
                                'bbox': bbox_numbers[:4],
                                'confidence': float(confidences[i])
                            }
                            objects.append(obj)
                    except Exception as e:
                        self.logger.debug(f"Failed to parse object {i}: {e}")
                        continue

                if objects:
                    self.logger.info(f"Extracted {len(objects)} objects using field-by-field matching")
                    return objects

            # Strategy 2: Extract using object pattern with flexible bbox
            # Pattern matches objects even with malformed bbox arrays
            object_pattern = r'\{[^{}]*"category":\s*"([^"]+)"[^{}]*"bbox(?:_2d)?":\s*\[([\d,\s]+)[^\}]*"confidence":\s*([\d.]+)[^{}]*\}'

            matches = re.finditer(object_pattern, json_str, re.DOTALL)
            for match in matches:
                try:
                    category = match.group(1)
                    bbox_str = match.group(2).strip()
                    if bbox_str.endswith(','):
                        bbox_str = bbox_str[:-1]
                    bbox_numbers = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]
                    confidence = float(match.group(3))

                    if len(bbox_numbers) >= 4:
                        obj = {
                            'category': category,
                            'bbox': bbox_numbers[:4],
                            'confidence': confidence
                        }
                        objects.append(obj)
                except Exception as e:
                    self.logger.debug(f"Failed to parse matched object: {e}")
                    continue

            if objects:
                self.logger.info(f"Extracted {len(objects)} objects using flexible pattern")
                return objects

            # Strategy 3: Last resort - try to extract any partial information
            # Find individual objects even if some fields are missing
            partial_pattern = r'\{"category":\s*"([^"]+)"[^}]*\}'
            partial_matches = re.findall(partial_pattern, json_str)

            for category in partial_matches:
                # Try to find corresponding bbox and confidence near this category
                # This is a very rough heuristic
                self.logger.warning(f"Found partial object with category '{category}' but incomplete data")

            return objects

        except Exception as e:
            self.logger.error(f"Error extracting objects from malformed JSON: {e}")
            return objects

    def _parse_detection_response(self, text: str) -> List[Dict]:
        """
        Parse detected objects from response.

        🔧 重要：此方法仅用于 detection hard_label 解析 JSON 输出
        CoT 分支不应调用此方法，CoT 只输出纯文本推理

        Handles multiple formats:
        - Markdown code blocks: ```json [...] ```
        - Single JSON object: {"label": "...", "bbox": [...]}
        - JSON array: [{"label": "...", "bbox": [...]}, ...]
        - Multiple JSON objects on separate lines
        - JSON with 'objects' key: {"objects": [...]}
        - Different field names: bbox_2d/bbox/box, label/category
        """
        objects = []

        # Try to parse JSON if present
        try:
            import json
            import re

            self.logger.debug(f"[Detection JSON解析] 尝试解析: {text[:200]}")

            # Method 1: Extract from markdown code blocks (most common for VLM outputs)
            # Pattern: ```json ... ``` or ``` ... ```
            # 🔧 改进：更健壮的提取，避免提取到多余的prompt文本
            markdown_patterns = [
                r'```json\s*\n(.*?)\n```',  # 标准格式，严格要求换行
                r'```json\s+(.*?)\s+```',   # 空格分隔
                r'```\s*\n(.*?)\n```',      # 无语言标记，要求换行
                r'```json\s*(.*?)\s*```',   # 无换行（fallback）
            ]

            json_content = None
            for pattern in markdown_patterns:
                matches = list(re.finditer(pattern, text, re.DOTALL))
                if matches:
                    # 使用最后一个匹配（通常是最完整的JSON）
                    json_content = matches[-1].group(1).strip()
                    self.logger.debug(f"Found JSON with pattern: {pattern}")
                    self.logger.debug(f"Extracted content (first 200 chars): {json_content[:200]}")
                    break

            if json_content:
                # 🔧 关键修复：清理JSON后面的多余文本
                # 策略：找到最后一个闭合的 } 或 ]，截断后面的内容
                json_content = self._clean_json_content(json_content)

                # 🔧 调试：显示清理后的内容
                self.logger.debug(f"Cleaned JSON content (first 200 chars): {json_content[:200]}")

                # 🔧 修复：检查是否提取到的是占位符或截断的内容
                if json_content == '{"objects": [...]}' or json_content == '{"objects": [...]}':
                    self.logger.warning("Extracted placeholder content, trying alternative extraction")
                    # 尝试提取完整的markdown块（包括换行）
                    full_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                    if full_match:
                        json_content = full_match.group(1).strip()
                        self.logger.debug(f"Re-extracted content: {json_content[:500]}")

                try:
                    parsed = json.loads(json_content)
                    if isinstance(parsed, list):
                        objects = parsed
                    elif isinstance(parsed, dict) and 'objects' in parsed:
                        objects = parsed['objects']

                    # Normalize field names
                    for obj in objects:
                        # Rename bbox_2d to bbox
                        if 'bbox_2d' in obj:
                            obj['bbox'] = obj.pop('bbox_2d')
                        # Rename label to category
                        if 'label' in obj and 'category' not in obj:
                            obj['category'] = obj.pop('label')
                        # Ensure confidence field exists
                        if 'confidence' not in obj:
                            obj['confidence'] = 0.5

                    return objects
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON from markdown: {e}")
                    self.logger.warning(f"Markdown content: {repr(json_content)}")

                    # Try to repair common JSON errors
                    repaired_json = self._repair_json(json_content)
                    if repaired_json:
                        try:
                            parsed = json.loads(repaired_json)
                            if isinstance(parsed, list):
                                objects = parsed
                            elif isinstance(parsed, dict) and 'objects' in parsed:
                                objects = parsed['objects']

                            # Normalize field names
                            for obj in objects:
                                if 'bbox_2d' in obj:
                                    obj['bbox'] = obj.pop('bbox_2d')
                                if 'label' in obj and 'category' not in obj:
                                    obj['category'] = obj.pop('label')
                                if 'confidence' not in obj:
                                    obj['confidence'] = 0.5

                            return objects
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse repaired JSON")

                            # Try to extract objects manually using regex
                            extracted_objects = self._extract_objects_from_malformed_json(json_content)
                            if extracted_objects:
                                self.logger.info(f"Manually extracted {len(extracted_objects)} objects from malformed JSON")
                                return extracted_objects

            # Method 2: Try to parse each line as separate JSON object
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

            for line in lines:
                if not (line.startswith('{') or line.startswith('[')):
                    continue

                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, list):
                        for obj in parsed:
                            # Normalize fields
                            if 'bbox_2d' in obj:
                                obj['bbox'] = obj.pop('bbox_2d')
                            if 'label' in obj and 'category' not in obj:
                                obj['category'] = obj.pop('label')
                            if 'confidence' not in obj:
                                obj['confidence'] = 0.5
                        objects.extend(parsed)
                    elif isinstance(parsed, dict):
                        if 'objects' in parsed:
                            for obj in parsed['objects']:
                                if 'bbox_2d' in obj:
                                    obj['bbox'] = obj.pop('bbox_2d')
                                if 'label' in obj and 'category' not in obj:
                                    obj['category'] = obj.pop('label')
                                if 'confidence' not in obj:
                                    obj['confidence'] = 0.5
                            objects.extend(parsed['objects'])
                        elif 'bbox_2d' in parsed or 'bbox' in parsed or 'box' in parsed:
                            # Normalize single object
                            if 'bbox_2d' in parsed:
                                parsed['bbox'] = parsed.pop('bbox_2d')
                            if 'label' in parsed and 'category' not in parsed:
                                parsed['category'] = parsed.pop('label')
                            if 'confidence' not in parsed:
                                parsed['confidence'] = 0.5
                            objects.append(parsed)
                except json.JSONDecodeError:
                    continue

            if objects:
                self.logger.debug(f"Successfully parsed {len(objects)} objects from multi-line response")
                return objects

            # Method 3: Try to find JSON array
            array_match = re.search(r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]', text)
            if array_match:
                json_str = array_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        for obj in parsed:
                            if 'bbox_2d' in obj:
                                obj['bbox'] = obj.pop('bbox_2d')
                            if 'label' in obj and 'category' not in obj:
                                obj['category'] = obj.pop('label')
                            if 'confidence' not in obj:
                                obj['confidence'] = 0.5
                        objects = parsed
                        self.logger.debug(f"Successfully parsed {len(objects)} objects from JSON array")
                        return objects
                except json.JSONDecodeError:
                    pass

            # Method 4: Try to find all JSON objects in text
            object_matches = re.finditer(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text)
            for match in object_matches:
                json_str = match.group(0)
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        if 'objects' in parsed:
                            for obj in parsed['objects']:
                                if 'bbox_2d' in obj:
                                    obj['bbox'] = obj.pop('bbox_2d')
                                if 'label' in obj and 'category' not in obj:
                                    obj['category'] = obj.pop('label')
                                if 'confidence' not in obj:
                                    obj['confidence'] = 0.5
                            objects.extend(parsed['objects'])
                        elif 'bbox_2d' in parsed or 'bbox' in parsed or 'box' in parsed or 'label' in parsed or 'category' in parsed:
                            if 'bbox_2d' in parsed:
                                parsed['bbox'] = parsed.pop('bbox_2d')
                            if 'label' in parsed and 'category' not in parsed:
                                parsed['category'] = parsed.pop('label')
                            if 'confidence' not in parsed:
                                parsed['confidence'] = 0.5
                            objects.append(parsed)
                except json.JSONDecodeError:
                    continue

            if objects:
                self.logger.debug(f"Successfully parsed {len(objects)} objects from JSON objects search")
                return objects

            # Method 5: Balanced braces extraction
            if "{" in text or "[" in text:
                start = text.find("{")
                if start == -1:
                    start = text.find("[")
                if start == -1:
                    return objects

                depth = 0
                end = start
                open_char = text[start]
                close_char = '}' if open_char == '{' else ']'

                for i in range(start, len(text)):
                    if text[i] == open_char:
                        depth += 1
                    elif text[i] == close_char:
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break

                if end > start:
                    json_str = text[start:end]
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, list):
                            for obj in parsed:
                                if 'bbox_2d' in obj:
                                    obj['bbox'] = obj.pop('bbox_2d')
                                if 'label' in obj and 'category' not in obj:
                                    obj['category'] = obj.pop('label')
                                if 'confidence' not in obj:
                                    obj['confidence'] = 0.5
                            objects = parsed
                        elif isinstance(parsed, dict) and 'objects' in parsed:
                            for obj in parsed['objects']:
                                if 'bbox_2d' in obj:
                                    obj['bbox'] = obj.pop('bbox_2d')
                                if 'label' in obj and 'category' not in obj:
                                    obj['category'] = obj.pop('label')
                                if 'confidence' not in obj:
                                    obj['confidence'] = 0.5
                            objects = parsed['objects']
                        self.logger.debug(f"Successfully parsed {len(objects)} objects from balanced braces")
                    except json.JSONDecodeError as e:
                        # JSON解析失败，可能是模型输出格式问题
                        self.logger.debug(f"[Detection JSON解析] 平衡括号解析失败: {e}")
                        self.logger.debug(f"[Detection JSON解析] 失败内容: {repr(text[start:end])}")

            # Method 6: Manual extraction fallback
            if not objects:
                self.logger.debug(f"[Detection JSON解析] 未找到JSON对象，尝试手动提取")
                self.logger.debug(f"[Detection JSON解析] 原始文本: {repr(text[:500])}")

                # Try to extract bbox patterns like [x, y, x, y] or (x, y, x, y)
                bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                bbox_matches = re.finditer(bbox_pattern, text)

                for match in bbox_matches:
                    bbox = [int(match.group(i)) for i in range(1, 5)]
                    # Try to find category name near the bbox
                    context_before = text[:match.start()]
                    # Look for common object names
                    category_match = re.search(r'(\w+)\s*(?:at|in|with|:)\s*', context_before[-50:])
                    category = category_match.group(1) if category_match else "unknown"

                    objects.append({
                        'category': category,
                        'bbox': bbox,
                        'confidence': 0.5
                    })
                    self.logger.debug(f"[Detection JSON解析] 手动提取对象: {category} at {bbox}")

                if objects:
                    self.logger.debug(f"[Detection JSON解析] 手动提取到 {len(objects)} 个对象")

        except Exception as e:
            self.logger.error(f"[Detection JSON解析] 解析过程发生错误: {e}")
            self.logger.debug(f"[Detection JSON解析] 原始文本: {repr(text[:500])}")

        return objects

    def _parse_keypoints_response(self, text: str) -> List[Dict]:
        """
        Parse keypoints from model response.

        Expected format: JSON with persons and their 17 keypoints.
        Each keypoint has name, x, y coordinates and visibility.
        """
        persons = []

        # Try to parse JSON if present
        try:
            import json
            import re

            # Method 1: Try to find complete JSON object
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text)

            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)

                    if isinstance(parsed, list):
                        persons = parsed
                    elif isinstance(parsed, dict):
                        if 'persons' in parsed:
                            persons = parsed['persons']
                        elif 'people' in parsed:
                            persons = parsed['people']
                        elif 'keypoints' in parsed:
                            persons = [parsed]

                    self.logger.debug(f"Successfully parsed {len(persons)} persons from keypoints response")
                    return persons
                except json.JSONDecodeError:
                    pass

            # Method 2: Extract from larger JSON structure
            if "{" in text or "[" in text:
                start = text.find("{")
                if start == -1:
                    start = text.find("[")

                # Find balanced braces
                depth = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] in '{[':
                        depth += 1
                    elif text[i] in '}]':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break

                json_str = text[start:end]
                parsed = json.loads(json_str)

                if isinstance(parsed, list):
                    persons = parsed
                elif isinstance(parsed, dict):
                    if 'persons' in parsed:
                        persons = parsed['persons']
                    elif 'people' in parsed:
                        persons = parsed['people']
                    elif 'keypoints' in parsed:
                        persons = [parsed]

                self.logger.debug(f"Successfully parsed {len(persons)} persons from keypoints response (method 2)")
            else:
                self.logger.warning(f"No JSON found in keypoints response: {text[:200]}...")

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from keypoints response: {e}")
            self.logger.debug(f"Raw response text: {text[:500]}")
            # Fallback: parse manually from text
            # Look for coordinate patterns like [x, y] or "nose: (123, 456)"
            pass
        except Exception as e:
            self.logger.error(f"Unexpected error parsing keypoints response: {e}")
            self.logger.debug(f"Raw response text: {text[:500]}")

        return persons

    def _process_logits(self, scores: tuple) -> Dict[str, torch.Tensor]:
        """
        Process generation scores into probability distribution.

        ✅ 官方标准：返回原始logits，不应用温度缩放
        温度缩放应该在软标签生成器中手动应用（T=3.0）

        Args:
            scores: Tuple of score tensors from generation

        Returns:
            Dictionary with raw logits (NOT probabilities)
        """
        # 🔧 添加调试日志
        self.logger.info(f"[Teacher Model DEBUG] scores类型: {type(scores)}")
        if isinstance(scores, (list, tuple)):
            self.logger.info(f"[Teacher Model DEBUG] scores长度: {len(scores)}")
            if len(scores) > 0:
                self.logger.info(f"[Teacher Model DEBUG] scores[0]形状: {scores[0].shape if hasattr(scores[0], 'shape') else 'N/A'}")
                if len(scores) > 1:
                    self.logger.info(f"[Teacher Model DEBUG] scores[1]形状: {scores[1].shape if hasattr(scores[1], 'shape') else 'N/A'}")
                # 检查是否所有scores形状相同
                if len(scores) > 2:
                    first_shape = scores[0].shape if hasattr(scores[0], 'shape') else None
                    all_same = all(
                        (s.shape == first_shape if hasattr(s, 'shape') else False)
                        for s in scores[1:]
                    )
                    self.logger.info(f"[Teacher Model DEBUG] 所有scores形状相同: {all_same}")

        # Stack scores - 这是原始logits
        logits_stack = torch.stack(scores) if isinstance(scores, (list, tuple)) else scores

        self.logger.info(f"[Teacher Model DEBUG] logits_stack形状: {logits_stack.shape}")
        self.logger.info(f"[Teacher Model DEBUG] logits_stack维度: {logits_stack.dim()}")

        # 详细分析维度含义
        if logits_stack.dim() == 3:
            self.logger.info(f"[Teacher Model DEBUG] raw_logits格式: [num_tokens={logits_stack.shape[0]}, batch={logits_stack.shape[1]}, vocab_size={logits_stack.shape[2]}]")
        elif logits_stack.dim() == 2:
            self.logger.info(f"[Teacher Model DEBUG] raw_logits格式: [num_tokens={logits_stack.shape[0]}, vocab_size={logits_stack.shape[1]}]")
        else:
            self.logger.info(f"[Teacher Model DEBUG] raw_logits格式: 未知维度 {logits_stack.dim()}")

        # ───────────────────────────────────────────────────────
        # ✅ 官方标准：返回原始logits，不应用温度缩放
        # 温度缩放在软标签生成器中手动应用（T=3.0）
        # ───────────────────────────────────────────────────────

        # 提取top-k logits（原始logits，未缩放）
        top_k = self.config.get('distillation.soft_labels.top_k_logits', 50)
        top_k = min(top_k, logits_stack.size(-1))

        # 提取top-k原始logits
        if logits_stack.dim() == 3:
            # [batch, num_tokens, vocab_size]
            top_logits, top_indices = torch.topk(logits_stack, top_k, dim=-1)
        else:
            # [num_tokens, vocab_size]
            top_logits, top_indices = torch.topk(logits_stack, top_k, dim=-1)

        return {
            'raw_logits': logits_stack,        # ✅ 原始logits（完整vocab_size维度）
            'top_k_indices': top_indices,      # token IDs
            'top_k_values': top_logits,        # ✅ 原始logits（top-k）
            'temperature': 1.0,                # ✅ 标记为原始logits（未缩放）
            'top_k': top_k,
        }

    def _compute_confidence(self, logits_data: Dict) -> float:
        """
        Compute confidence score from logits.

        改进：兼容新的 top-k 存储格式

        Args:
            logits_data: Processed logits dictionary

        Returns:
            Confidence score (0-1)
        """
        # 🔧 兼容新格式（top-k only）和旧格式（完整概率）
        if 'probabilities' in logits_data:
            # 旧格式：有完整的概率分布
            probs = logits_data['probabilities']
            max_probs = probs.max(dim=-1).values
            confidence = max_probs.mean().item()
            return confidence

        elif 'top_k_values' in logits_data:
            # 🔧 修复：优先使用原始概率值（temperature=1.0）计算硬标签置信度
            # raw_top_k_values 是未经温度缩放的原始概率，代表模型的真实置信度

            # ✅ 优先使用 raw_top_k_values（temperature=1.0）
            if 'raw_top_k_values' in logits_data:
                top_k_probs = logits_data['raw_top_k_values']  # 原始概率（temperature=1.0）
            else:
                # 兼容旧数据
                top_k_probs = logits_data['top_k_values']  # 可能是温度缩放后的概率

            # 直接使用概率值（不需要softmax）
            if top_k_probs.dim() == 2:
                # [num_tokens, top_k]
                max_probs = top_k_probs[:, 0]  # top-k 已排序，第一个是最大的
            elif top_k_probs.dim() == 3:
                # [num_tokens, batch_size, top_k]
                max_probs = top_k_probs[:, 0, 0]  # 第一个batch的第一个位置
            else:
                # 单个位置
                max_probs = top_k_probs[0]

            # 🔧 置信度 = 第一个token（答案）的最大概率
            confidence = max_probs[0].item() if len(max_probs) > 0 else 0.5

            return confidence

        else:
            # 既没有 probabilities 也没有 top_k_values
            self.logger.warning("logits_data missing both 'probabilities' and 'top_k_values'")
            return 0.5  # 默认值

    def _compute_confidence_from_tokens(self, logits_data: Dict, token_ids: List[int]) -> float:
        """
        Compute confidence based on actual generated tokens.

        改进：
        1. 兼容新的 top-k 存储格式
        2. 处理数字和文字的映射（如 "2" vs "two"）

        Args:
            logits_data: Processed logits dictionary (新格式：top-k only，或旧格式：完整概率)
            token_ids: List of generated token IDs

        Returns:
            Confidence score (0-1)
        """
        if not token_ids:
            return 0.0

        # 🔧 兼容新格式（top-k only）和旧格式（完整概率）
        if 'probabilities' in logits_data:
            # 旧格式：有完整的概率分布
            probs = logits_data['probabilities']

            token_probs = []
            for i, token_id in enumerate(token_ids):
                if i < probs.shape[0]:
                    prob = probs[i, 0, token_id].item()  # batch_size=0
                    token_probs.append(prob)

            # 🔧 修复：返回第一个token的概率，而不是平均值
            if token_probs:
                return token_probs[0]
            else:
                return 0.0

        elif 'top_k_indices' in logits_data:
            # 🔧 修复：使用原始概率值（temperature=1.0）计算硬标签置信度
            # raw_top_k_values 是未经温度缩放的原始概率，代表模型的真实置信度
            top_k_indices = logits_data['top_k_indices']

            # ✅ 优先使用 raw_top_k_values（temperature=1.0），否则回退到 top_k_values
            if 'raw_top_k_values' in logits_data:
                top_k_probs = logits_data['raw_top_k_values']  # 原始概率（temperature=1.0）
            else:
                # 兼容旧数据
                top_k_probs = logits_data['top_k_values']  # 可能是温度缩放后的概率

            token_probs = []
            for i, token_id in enumerate(token_ids):
                # 检查 token 是否在 top-k 中
                if i < top_k_indices.shape[0]:  # 确保不越界
                    # 获取该位置的 top-k indices 和概率
                    if top_k_indices.dim() == 2:
                        # [num_tokens, top_k]
                        position_indices = top_k_indices[i]
                        position_probs = top_k_probs[i]
                    elif top_k_indices.dim() == 3:
                        # [num_tokens, batch_size, top_k]
                        position_indices = top_k_indices[i, 0]
                        position_probs = top_k_probs[i, 0]
                    else:
                        continue

                    # 🔧 调试：打印详细信息（第一个token）
                    if i == 0:
                        self.logger.info(f"[Confidence Debug] Position {i}:")
                        self.logger.info(f"  Looking for token_id: {token_id}")
                        self.logger.info(f"  Top-5 indices: {position_indices[:5].tolist()}")
                        self.logger.info(f"  Top-5 probs (temperature=1.0): {position_probs[:5].tolist()}")

                    # 查找 token_id 是否在 top-k 中
                    mask = position_indices == token_id
                    if mask.any():
                        # 找到了，获取对应的概率
                        prob = position_probs[mask].item()
                        token_probs.append(prob)

                        if i == 0:
                            # 找到匹配的位置
                            match_idx = mask.nonzero(as_tuple=True)[0][0].item()
                            self.logger.info(f"  ✓ Found at position {match_idx} in top-k")
                            self.logger.info(f"  Probability: {prob:.4f}")
                    else:
                        # 🔧 调试：token不在top-k中
                        if i == 0:
                            self.logger.warning(f"  ✗ Token ID {token_id} not in top-k!")
                            self.logger.warning(f"  Will try synonym/text matching...")

                        # 🔧 改进：检查同义词和不同形式的token
                        # 解码token看是什么
                        token_text = self.tokenizer.decode([token_id]).strip().lower()

                        # 数字和文字的映射
                        number_word_map = {
                            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
                            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
                        }

                        # 在top-k中查找匹配的token
                        found_match = False

                        # 方法1：检查同义词
                        if token_text in number_word_map:
                            synonym = number_word_map[token_text]
                            self.logger.debug(f"Token {i}: '{token_text}' -> checking synonym '{synonym}'")

                            # 在top-k中查找同义词
                            for j, idx in enumerate(position_indices.tolist()):
                                word = self.tokenizer.decode([int(idx)]).strip().lower()
                                # 去掉前缀字符（▁）
                                word_clean = word.replace('▁', '').strip()

                                if word_clean == synonym or word_clean == token_text:
                                    prob = position_probs[j].item()
                                    token_probs.append(prob)
                                    found_match = True
                                    self.logger.debug(f"Found synonym: '{token_text}' -> '{word_clean}' (ID={idx}) with prob {prob:.4f}")
                                    break

                        # 方法2：检查文本匹配（去掉前缀字符）
                        if not found_match:
                            token_text_clean = token_text.replace('▁', '').strip()

                            for j, idx in enumerate(position_indices.tolist()):
                                word = self.tokenizer.decode([int(idx)]).strip().lower()
                                word_clean = word.replace('▁', '').strip()

                                # 检查文本是否匹配
                                if word_clean == token_text_clean:
                                    prob = position_probs[j].item()
                                    token_probs.append(prob)
                                    found_match = True
                                    self.logger.debug(f"Found text match: '{token_text_clean}' (ID={idx}) with prob {prob:.4f}")
                                    break

                        if not found_match:
                            # 不在 top-k 中，说明概率很低（< 1/top_k）
                            # 使用一个很小的默认值
                            token_probs.append(0.001)
                            self.logger.warning(f"Token {i} (ID={token_id}, text='{token_text}') not found in top-k")

            # 🔧 修复：置信度 = 第一个token（答案开始）的概率
            # 不使用平均值，因为后续token的概率不代表答案的置信度
            if token_probs:
                confidence = token_probs[0]  # 第一个token的概率
                
                self.logger.error(f"Confidence (first token): {confidence:.4f}")
                return confidence
            else:
                return 0.0

        else:
            # 既没有 probabilities 也没有 top_k_indices
            self.logger.warning("logits_data missing both 'probabilities' and 'top_k_indices'")
            return 0.0

    def batch_inference(
        self,
        batch_data: Dict[str, Any],
        task: str
    ) -> List[Dict[str, Any]]:
        """
        Process batch of images for a specific task.

        Args:
            batch_data: Dictionary with batch images and annotations
            task: Task type (vqa/captioning/detection)

        Returns:
            List of inference results for each image
        """
        results = []

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image = img_data['image']

            self.logger.info(f"Processing image {image_id} for task {task}")

            # Task-specific inference
            if task == 'vqa':
                # Process VQA questions
                questions = batch_data['annotations']['vqa'].get(image_id, [])
                for q_data in questions:
                    question = q_data.get('question', '')
                    result = self.inference_vqa(image, question, return_logits=True, generate_cot=True)
                    result['image_id'] = image_id
                    result['question'] = question
                    results.append(result)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.

        Returns:
            Dictionary with model info
        """
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'precision': self.precision,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
        }

        if self.model:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['model_dtype'] = str(self.model.dtype)

        return info

    def __repr__(self) -> str:
        """String representation."""
        return f"TeacherModel(name={self.model_name}, device={self.device}, precision={self.precision})"