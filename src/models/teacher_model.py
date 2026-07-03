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
        self.temperature = self.config.get("model.temperature", 0.7)
        self.top_p = self.config.get("model.top_p", 0.9)
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
        generate_cot: bool = False
    ) -> Dict[str, Any]:
        """
        Perform object detection inference.

        Args:
            image: PIL Image or image path
            return_logits: Whether to return logits
            generate_cot: Whether to generate CoT

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

        # Generate
        outputs = self._generate(inputs, return_logits=return_logits)

        # Process outputs
        result = self._process_detection_outputs(outputs, return_logits)

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
        Construct task-specific prompt.

        Args:
            question: Question for VQA (empty for other tasks)
            task: Task type (vqa/captioning/detection/keypoints)

        Returns:
            Formatted prompt string
        """
        prompts = {
            'vqa': f"Look at the image and answer the following question:\nQuestion: {question}\nAnswer:",
            'captioning': "Describe this image in detail, including all objects, their attributes, and the overall scene.",
            'detection': "Detect all objects in this image. For each object, provide the bounding box coordinates and category.\n\nRespond ONLY with valid JSON in this exact format:\n{\"objects\": [{\"category\": \"object_name\", \"bbox\": [x_min, y_min, x_max, y_max], \"confidence\": 0.95}]}\n\nExample response:\n{\"objects\": [{\"category\": \"person\", \"bbox\": [100, 50, 300, 400], \"confidence\": 0.98}, {\"category\": \"car\", \"bbox\": [400, 200, 600, 350], \"confidence\": 0.92}]}",
            'keypoints': "Detect all persons in this image and estimate their body pose with 17 keypoints.\n\nRespond ONLY with valid JSON in this exact format:\n{\"persons\": [{\"bbox\": [x, y, width, height], \"keypoints\": [{\"name\": \"nose\", \"x\": 123, \"y\": 456, \"visibility\": 2}]}]}\n\nKeypoint names: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle\n\nVisibility: 0=not visible, 1=occluded, 2=visible\n\nExample response:\n{\"persons\": [{\"bbox\": [50, 100, 200, 300], \"keypoints\": [{\"name\": \"nose\", \"x\": 150, \"y\": 120, \"visibility\": 2}, ...]}]}",
        }

        return prompts.get(task, "Analyze this image.")

    def _construct_cot_prompt(self, question: str, task: str) -> str:
        """
        Construct Chain-of-Thought prompt.

        Args:
            question: Question for VQA
            task: Task type

        Returns:
            CoT-formatted prompt
        """
        cot_prompts = {
            'vqa': f"Analyze this image step by step to answer the following question.\nQuestion: {question}\n\nPlease think through this systematically:\n1. First, identify all visual elements in the image.\n2. Next, analyze their attributes and relationships.\n3. Then, consider the context and scene.\n4. Finally, based on your analysis, provide the answer.\n\nLet's start:",
            'captioning': "Describe this image systematically. Think through the following steps:\n1. Identify the main subjects and objects.\n2. Describe their attributes and positions.\n3. Note the scene and setting.\n4. Describe any actions or activities.\n5. Combine everything into a comprehensive caption.\n\nLet's analyze:",
            'detection': "Detect objects in this image methodically:\n1. Scan the image systematically.\n2. Identify each object and its location.\n3. Determine the bounding box coordinates.\n4. Classify each object.\n\nAfter analysis, provide results ONLY as valid JSON:\n{\"objects\": [{\"category\": \"name\", \"bbox\": [x_min, y_min, x_max, y_max], \"confidence\": 0.95}]}\n\nLet's start the detection:",
            'keypoints': "Estimate human poses in this image systematically:\n1. Identify all persons in the image and their approximate locations.\n2. For each person, locate head and face keypoints (nose, eyes, ears).\n3. Identify upper body keypoints (shoulders, elbows, wrists).\n4. Locate lower body keypoints (hips, knees, ankles).\n5. For each keypoint, provide coordinates [x, y] and visibility (0/1/2).\n\nAfter analysis, provide results ONLY as valid JSON:\n{\"persons\": [{\"bbox\": [x, y, w, h], \"keypoints\": [{\"name\": \"nose\", \"x\": 123, \"y\": 456, \"visibility\": 2}]}]}\n\nLet's start the pose estimation:",
        }

        return cot_prompts.get(task, "Analyze this image step by step.")

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
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate outputs from model.

        Args:
            inputs: Input tensors
            return_logits: Whether to return logits
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generation outputs
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        # Generation config
        gen_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
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
        """Extract final answer from VQA response."""
        # Simple extraction - last sentence or after "Answer:"
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            # Take last sentence
            sentences = text.split(".")
            answer = sentences[-1].strip() if sentences else text.strip()

        return answer

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

        Common errors handled:
        1. Missing closing brackets: {"objects": [obj1], obj2} → {"objects": [obj1, obj2]}
        2. Truncated JSON: {"category": "cat", "bbox": [1,2, → auto-complete
        3. Extra commas: [obj1, obj2, ] → [obj1, obj2]
        4. Missing quotes: {category: "cat"} → {"category": "cat"}
        5. Mixed bracket types: {"objects": [obj1}, obj2] → {"objects": [obj1, obj2]}

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

            if repaired != json_str:
                self.logger.info(f"JSON repaired successfully")
                self.logger.debug(f"Original: {json_str[:100]}")
                self.logger.debug(f"Repaired: {repaired[:100]}")
                return repaired

            return None  # No repair needed or no errors detected

        except Exception as e:
            self.logger.error(f"Error during JSON repair: {e}")
            return None

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

                    self.logger.info(f"Successfully parsed {len(objects)} objects from markdown code block")
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

                            self.logger.info(f"Successfully parsed {len(objects)} objects after JSON repair")
                            return objects
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse repaired JSON")

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