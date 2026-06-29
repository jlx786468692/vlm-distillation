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

    def _construct_prompt(self, question: str, task: str) -> str:
        """
        Construct task-specific prompt.

        Args:
            question: Question for VQA (empty for other tasks)
            task: Task type (vqa/captioning/detection)

        Returns:
            Formatted prompt string
        """
        prompts = {
            'vqa': f"Look at the image and answer the following question:\nQuestion: {question}\nAnswer:",
            'captioning': "Describe this image in detail, including all objects, their attributes, and the overall scene.",
            'detection': "Detect all objects in this image. For each object, provide the bounding box coordinates in format [x_min, y_min, x_max, y_max] and the object category. Format your response as JSON.",
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
            'detection': "Detect objects in this image methodically:\n1. Scan the image systematically.\n2. Identify each object and its location.\n3. Determine the bounding box coordinates.\n4. Classify each object.\n\nProvide results in JSON format with bounding boxes as [x_min, y_min, x_max, y_max].\n\nLet's start the detection:",
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

    def _parse_detection_response(self, text: str) -> List[Dict]:
        """Parse detected objects from response."""
        objects = []

        # Try to parse JSON if present
        try:
            import json
            # Find JSON-like content
            if "{" in text or "[" in text:
                start = text.find("{")
                if start == -1:
                    start = text.find("[")
                json_str = text[start:]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    objects = parsed
                elif isinstance(parsed, dict) and 'objects' in parsed:
                    objects = parsed['objects']
        except:
            # Fallback: parse manually
            # Look for patterns like [x, y, x, y] or bbox descriptions
            pass

        return objects

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
