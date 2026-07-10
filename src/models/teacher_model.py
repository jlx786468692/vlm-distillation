"""
Teacher Model Interface
=======================

Wraps Qwen2.5-VL-7B-Instruct for multi-task distillation.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from PIL import Image

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class TeacherModel:
    """
    Wrapper for Qwen2.5-VL-7B-Instruct teacher model.

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
            precision: Model precision (fp32/fp16/bf16)
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Model settings
        self.model_name = model_name or self.config.get("teacher.model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.device = device or self.config.get("teacher.device", "cuda")
        self.precision = precision or self.config.get("teacher.precision", "bf16")

        # Generation parameters
        self.max_new_tokens = self.config.get("model.max_new_tokens", 512)
        self.max_detection_tokens = self.config.get("model.max_detection_tokens", 1024)  # Detection needs more tokens
        self.temperature = self.config.get("model.temperature", 0.7)
        self.detection_temperature = self.config.get("model.detection_temperature", 0.3)  # Lower temp for detection
        self.top_p = self.config.get("model.top_p", 0.9)
        self.detection_top_p = self.config.get("model.detection_top_p", 0.95)  # Higher for deterministic output
        self.top_k = self.config.get("model.top_k", 50)

        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load Qwen2.5-VL-7B-Instruct model and components."""
        self.logger.info(f"Loading teacher model: {self.model_name}")

        try:
            # Check if using HuggingFace mirror (for network issues)
            hf_mirror = self.config.get("teacher.hf_mirror", None)
            if hf_mirror:
                import os
                self.logger.info(f"Using HuggingFace mirror: {hf_mirror}")
                os.environ['HF_ENDPOINT'] = hf_mirror

            # Determine dtype
            dtype_map = {
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.precision, torch.bfloat16)

            # Load processor
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            self.logger.info(f"Loading model on {self.device} with {self.precision} precision...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            self.logger.info("Teacher model loaded successfully")
            self.logger.info(f"Model device: {self.device}")
            self.logger.info(f"Model precision: {self.precision}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.error(f"Possible solutions:")
            self.logger.error(f"  1. Use local model path in config: teacher.model_name")
            self.logger.error(f"  2. Use HuggingFace mirror: teacher.hf_mirror: 'https://hf-mirror.com'")
            self.logger.error(f"  3. Download model manually and use local path")
            raise

    def inference_vqa(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        return_logits: bool = False,
        generate_cot: bool = False
    ) -> Dict[str, Any]:
        """
        Perform VQA inference.

        Args:
            image: PIL Image or image path
            question: Question string
            return_logits: Whether to return logits for soft labels
            generate_cot: Whether to generate Chain-of-Thought reasoning

        Returns:
            Dictionary with answer, confidence, and optionally logits/cot
        """
        # Construct prompt
        if generate_cot:
            prompt = self._construct_cot_prompt(question, task="vqa")
        else:
            prompt = self._construct_prompt(question, task="vqa")

        # Prepare inputs
        inputs = self._prepare_inputs(image, prompt)

        # Generate
        outputs = self._generate(inputs, return_logits=return_logits)

        # Process outputs
        result = self._process_vqa_outputs(outputs, return_logits)

        return result

    def inference_captioning(
        self,
        image: Union[Image.Image, str, Path],
        return_logits: bool = False,
        generate_cot: bool = False,
        num_captions: int = 1
    ) -> Dict[str, Any]:
        """
        Perform image captioning inference.

        Args:
            image: PIL Image or image path
            return_logits: Whether to return logits
            generate_cot: Whether to generate CoT
            num_captions: Number of caption variations to generate

        Returns:
            Dictionary with captions and metadata
        """
        # Construct prompt
        if generate_cot:
            prompt = self._construct_cot_prompt("", task="captioning")
        else:
            prompt = self._construct_prompt("", task="captioning")

        # Prepare inputs
        inputs = self._prepare_inputs(image, prompt)

        # Generate multiple captions if requested
        captions = []
        all_logits = []

        for i in range(num_captions):
            outputs = self._generate(inputs, return_logits=return_logits)
            caption_result = self._process_captioning_outputs(outputs, return_logits)
            captions.append(caption_result['caption'])

            if return_logits and 'logits' in caption_result:
                all_logits.append(caption_result['logits'])

        result = {
            'captions': captions,
            'num_captions': num_captions,
        }

        if return_logits:
            result['logits'] = all_logits

        if generate_cot:
            # Generate separate CoT
            cot_result = self._generate_cot(image, task="captioning")
            result['cot'] = cot_result

        return result

    def inference_detection(
        self,
        image: Union[Image.Image, str, Path],
        return_logits: bool = False,
        generate_cot: bool = False,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Perform object detection inference.

        Args:
            image: PIL Image or image path
            return_logits: Whether to return logits
            generate_cot: Whether to generate CoT
            max_retries: Maximum retries if JSON parsing fails (default: 2)

        Returns:
            Dictionary with detected objects and metadata
        """
        # Construct prompt
        if generate_cot:
            prompt = self._construct_cot_prompt("", task="detection")
        else:
            prompt = self._construct_prompt("", task="detection")

        # Prepare inputs
        inputs = self._prepare_inputs(image, prompt)

        # Try generation with retries
        for attempt in range(max_retries + 1):
            # Generate with optimized parameters for detection
            # Increase temperature slightly on retries for more diversity
            current_temp = self.detection_temperature if attempt == 0 else max(self.detection_temperature + 0.05, 0.3)

            outputs = self._generate(
                inputs,
                return_logits=return_logits,
                max_new_tokens=self.max_detection_tokens,
                temperature=current_temp,
                top_p=self.detection_top_p
            )

            # Process outputs
            result = self._process_detection_outputs(outputs, return_logits)

            # Check if parsing was successful
            objects = result.get('objects', [])
            full_response = result.get('full_response', '')

            # Success criteria: has objects OR no parsing errors
            if objects or not ('Failed to parse' in full_response or 'malformed' in full_response.lower()):
                if attempt > 0:
                    self.logger.info(f"Detection successful on retry attempt {attempt}")
                return result

            # If failed and this is not the last attempt, retry
            if attempt < max_retries:
                self.logger.warning(f"Detection output parsing failed (attempt {attempt + 1}), retrying...")
                # On retry, try with even lower temperature
                if attempt == max_retries - 1:
                    self.logger.info("Final retry with minimal temperature")

        # Return last result even if parsing had issues (fallback extraction will handle it)
        self.logger.warning(f"Detection completed after {max_retries + 1} attempts with potential parsing issues")
        return result

    def inference_keypoints(
        self,
        image: Union[Image.Image, str, Path],
        return_logits: bool = False,
        generate_cot: bool = False
    ) -> Dict[str, Any]:
        """
        Perform human pose estimation (keypoints) inference.

        Args:
            image: PIL Image or image path
            return_logits: Whether to return logits for soft labels
            generate_cot: Whether to generate Chain-of-Thought reasoning

        Returns:
            Dictionary with detected persons and their keypoints
        """
        # Construct prompt
        if generate_cot:
            prompt = self._construct_cot_prompt("", task="keypoints")
        else:
            prompt = self._construct_prompt("", task="keypoints")

        # Prepare inputs
        inputs = self._prepare_inputs(image, prompt)

        # Generate
        outputs = self._generate(inputs, return_logits=return_logits)

        # Process outputs
        result = self._process_keypoints_outputs(outputs, return_logits)

        return result

    def _construct_prompt(self, question: str, task: str) -> str:
        """
        Construct task-specific prompt from configuration file.

        Args:
            question: Question for VQA (empty for other tasks)
            task: Task type (vqa/captioning/detection/keypoints)

        Returns:
            Formatted prompt string
        """
        # 从配置文件读取 prompt
        prompt_template = self.config.get(
            f'prompts.standard.{task}',
            self.config.get('prompts.default.standard', "Analyze this image.")
        )

        # 调试日志：显示实际使用的 prompt
        self.logger.debug(f"Loading prompt for task '{task}' from config")
        self.logger.debug(f"Prompt template (first 100 chars): {prompt_template[:100]}")

        # 支持变量插值（如 {question}）
        try:
            if '{question}' in prompt_template:
                prompt = prompt_template.format(question=question)
                self.logger.debug(f"Formatted prompt with question: {question}")
            else:
                prompt = prompt_template
        except KeyError as e:
            self.logger.warning(f"Prompt template missing variable: {e}")
            prompt = prompt_template

        return prompt.strip()

    def _construct_cot_prompt(self, question: str, task: str) -> str:
        """
        Construct Chain-of-Thought prompt from configuration file.

        Args:
            question: Question for VQA
            task: Task type

        Returns:
            CoT-formatted prompt
        """
        # 从配置文件读取 CoT prompt
        cot_template = self.config.get(
            f'prompts.cot.{task}',
            self.config.get('prompts.default.cot', "Analyze this image step by step.")
        )

        # 调试日志：显示实际使用的 prompt
        self.logger.debug(f"Loading CoT prompt for task '{task}' from config")
        self.logger.debug(f"CoT template (first 100 chars): {cot_template[:100]}")

        # 支持变量插值（如 {question}）
        try:
            if '{question}' in cot_template:
                prompt = cot_template.format(question=question)
                self.logger.debug(f"Formatted CoT prompt with question: {question}")
            else:
                prompt = cot_template
        except KeyError as e:
            self.logger.warning(f"CoT prompt template missing variable: {e}")
            prompt = cot_template

        return prompt.strip()

    def _prepare_inputs(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model inference.

        Args:
            image: PIL Image or path
            prompt: Text prompt

        Returns:
            Dictionary of input tensors
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Construct message format for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

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

        return inputs

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
        gen_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': self.top_k,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

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
        return_logits: bool
    ) -> Dict[str, Any]:
        """
        Process VQA generation outputs.

        Args:
            outputs: Model outputs
            return_logits: Whether logits are included

        Returns:
            Processed VQA result
        """
        # Decode generated text
        generated_ids = outputs.sequences
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract answer (remove prompt part)
        answer = self._extract_answer(generated_text)

        result = {
            'full_response': generated_text,
            'answer': answer,
        }

        if return_logits:
            # Process logits for soft labels
            logits = self._process_logits(outputs.scores)
            result['logits'] = logits
            result['confidence'] = self._compute_confidence(logits)

        return result

    def _process_captioning_outputs(
        self,
        outputs: Dict[str, Any],
        return_logits: bool
    ) -> Dict[str, Any]:
        """
        Process captioning outputs.

        Args:
            outputs: Model outputs
            return_logits: Whether logits are included

        Returns:
            Processed caption result
        """
        # Decode
        generated_ids = outputs.sequences
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract caption
        caption = self._extract_caption(generated_text)

        result = {
            'full_response': generated_text,
            'caption': caption,
        }

        if return_logits:
            logits = self._process_logits(outputs.scores)
            result['logits'] = logits

        return result

    def _process_detection_outputs(
        self,
        outputs: Dict[str, Any],
        return_logits: bool
    ) -> Dict[str, Any]:
        """
        Process detection outputs.

        Args:
            outputs: Model outputs
            return_logits: Whether logits are included

        Returns:
            Processed detection result
        """
        # Decode
        generated_ids = outputs.sequences
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Parse detected objects
        objects = self._parse_detection_response(generated_text)

        result = {
            'full_response': generated_text,
            'objects': objects,
        }

        if return_logits:
            logits = self._process_logits(outputs.scores)
            result['logits'] = logits

        return result

    def _process_keypoints_outputs(
        self,
        outputs: Dict[str, Any],
        return_logits: bool
    ) -> Dict[str, Any]:
        """
        Process keypoints (pose estimation) outputs.

        Args:
            outputs: Model outputs
            return_logits: Whether logits are included

        Returns:
            Processed keypoints result with persons and their keypoint coordinates
        """
        # Decode
        generated_ids = outputs.sequences
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Parse keypoints from response
        persons = self._parse_keypoints_response(generated_text)

        result = {
            'full_response': generated_text,
            'persons': persons,
            'num_persons': len(persons),
        }

        if return_logits:
            logits = self._process_logits(outputs.scores)
            result['logits'] = logits

        return result

    def _extract_answer(self, text: str) -> str:
        """
        从VQA响应中提取简短答案（改进版）

        改进：
        1. 去掉 assistant\n 前缀
        2. 去掉 Answer: 等提示词
        3. 提取第一个词作为简短答案
        4. 清理标点和格式
        5. 转小写

        Args:
            text: 生成的文本

        Returns:
            清理后的简短答案
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

        # 提取第一个词（最可能的简短答案）
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

            # Error 1: Objects outside array - {"objects": [obj1], obj2}
            # Pattern: closing bracket followed by comma and another object
            if ']}, {' in repaired or ']}, {' in repaired:
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
        """Parse detected objects from response.

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

            self.logger.debug(f"Attempting to parse detection response: {text[:200]}")

            # Method 1: Extract from markdown code blocks (most common for VLM outputs)
            # Pattern: ```json ... ``` or ``` ... ```
            markdown_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if markdown_match:
                json_content = markdown_match.group(1).strip()
                self.logger.debug(f"Extracted from markdown: {json_content[:200]}")

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
                        self.logger.warning(f"Failed to parse balanced JSON: {e}")
                        self.logger.warning(f"Raw response that failed: {repr(text[start:end])}")

            # Method 6: Manual extraction fallback
            if not objects:
                self.logger.warning(f"No JSON objects found, trying manual extraction")
                self.logger.warning(f"Full raw response text: {repr(text[:500])}")

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
                    self.logger.debug(f"Manually extracted object: {category} at {bbox}")

                if objects:
                    self.logger.info(f"Extracted {len(objects)} objects using manual fallback")

        except Exception as e:
            self.logger.error(f"Unexpected error parsing detection response: {e}")
            self.logger.warning(f"Raw response text: {repr(text[:500])}")

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
        Process generation scores into logits.

        Args:
            scores: Tuple of score tensors from generation

        Returns:
            Dictionary with processed logits
        """
        # Stack scores
        logits_stack = torch.stack(scores)

        # Apply softmax to get probabilities
        probs = torch.softmax(logits_stack, dim=-1)

        return {
            'raw_logits': logits_stack,
            'probabilities': probs,
        }

    def _compute_confidence(self, logits_data: Dict) -> float:
        """
        Compute confidence score from logits.

        Args:
            logits_data: Processed logits dictionary

        Returns:
            Confidence score (0-1)
        """
        probs = logits_data['probabilities']
        # Take max probability for each position
        max_probs = probs.max(dim=-1).values
        # Average across positions
        confidence = max_probs.mean().item()
        return confidence

    def _generate_cot(
        self,
        image: Union[Image.Image, str, Path],
        task: str
    ) -> str:
        """Generate Chain-of-Thought reasoning."""
        cot_prompt = self._construct_cot_prompt("", task)
        inputs = self._prepare_inputs(image, cot_prompt)

        outputs = self._generate(inputs, max_new_tokens=512)
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return generated_text

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

            elif task == 'captioning':
                result = self.inference_captioning(image, return_logits=True, generate_cot=True, num_captions=3)
                result['image_id'] = image_id
                results.append(result)

            elif task == 'detection':
                result = self.inference_detection(image, return_logits=True, generate_cot=True)
                result['image_id'] = image_id
                results.append(result)

            elif task == 'keypoints':
                result = self.inference_keypoints(image, return_logits=True, generate_cot=True)
                result['image_id'] = image_id
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