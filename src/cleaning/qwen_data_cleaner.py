"""
数据清洗增强模块（Qwen-VL官方方案）
====================================

实现Qwen-VL官方的四层清洗策略：
1. 置信度过滤（已有）
2. CoT一致性校验（Conclusion vs hard_label）
3. 语义过滤（Observation/Analysis相似度、遮挡规则）
4. 图像哈希去重

新增功能：
- 兜底机制：OOM捕获、空输出重试
- 幻觉兜底：遮挡幻觉检测、闭合集校验
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


class QwenDataCleaner:
    """
    Qwen-VL官方数据清洗策略实现

    四层清洗逻辑：
    - 第一层：置信度过滤（已有）
    - 第二层：CoT一致性校验
    - 第三层：语义过滤（相似度检测、遮挡规则）
    - 第四层：图像哈希去重
    """

    # COCO 80类别列表（用于幻觉检测）
    COCO_CATEGORIES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, config: Optional[Any] = None):
        """
        初始化清洗器

        Args:
            config: 配置管理器
        """
        self.config = config

        # 清洗参数
        self.similarity_threshold = 0.9  # Observation/Analysis相似度阈值
        self.min_observation_tokens = 15  # Observation最小token数

        # 图像哈希缓存
        self._image_hashes: Dict[str, str] = {}
        self._hash_to_images: Dict[str, List[str]] = {}

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'consistency_passed': 0,
            'consistency_failed': 0,
            'similarity_passed': 0,
            'similarity_failed': 0,
            'occlusion_passed': 0,
            'occlusion_failed': 0,
            'duplicate_removed': 0,
            'valid_samples': 0
        }

    # ===== 第二层：CoT一致性校验 =====

    def extract_conclusion_answer(self, cot_text: str) -> Optional[str]:
        """
        从CoT文本中提取Conclusion部分的答案

        Qwen-VL官方方案：抽取CoT内"Final Answer"后的词汇

        Args:
            cot_text: CoT文本

        Returns:
            提取的答案，如果未找到返回None
        """
        if not cot_text:
            return None

        # 尝试多种模式匹配
        patterns = [
            r'Final Answer\s*:\s*(\w+)',  # Final Answer: one
            r'Conclusion\s*:\s*.*?Final Answer\s*:\s*(\w+)',  # Conclusion: ... Final Answer: one
            r'Conclusion\s*:\s*(\w+)',  # Conclusion: one
            r'Answer\s*:\s*(\w+)',  # Answer: one
        ]

        for pattern in patterns:
            match = re.search(pattern, cot_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).lower().strip()

        return None

    def check_consistency(
        self,
        cot_text: str,
        hard_label: str,
        allowed_answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        校验CoT结论与hard_label的一致性

        Qwen-VL官方方案：
        - 抽取CoT内Final Answer后的词汇
        - 与hard_label字符串严格比对
        - 不一致则标记冲突脏样本，直接剔除

        Args:
            cot_text: CoT文本
            hard_label: 硬标签答案
            allowed_answers: 允许的答案列表（可选）

        Returns:
            校验结果字典，包含：
            - is_consistent: 是否一致
            - extracted_answer: 提取的答案
            - hard_label: 硬标签
            - is_in_allowed_set: 是否在允许答案集内
        """
        self.stats['total_processed'] += 1

        # 提取Conclusion答案
        extracted_answer = self.extract_conclusion_answer(cot_text)

        if not extracted_answer:
            # 无法提取Conclusion，视为不一致
            self.stats['consistency_failed'] += 1
            return {
                'is_consistent': False,
                'extracted_answer': None,
                'hard_label': hard_label,
                'is_in_allowed_set': False,
                'reason': 'Cannot extract conclusion from CoT'
            }

        # 标准化答案（小写、去空格）
        hard_label_normalized = hard_label.lower().strip()
        extracted_normalized = extracted_answer.lower().strip()

        # 一致性比对
        is_consistent = (extracted_normalized == hard_label_normalized)

        # 检查是否在允许答案集内
        is_in_allowed_set = True
        if allowed_answers:
            allowed_set = set(a.lower().strip() for a in allowed_answers)
            is_in_allowed_set = extracted_normalized in allowed_set

        if is_consistent:
            self.stats['consistency_passed'] += 1
        else:
            self.stats['consistency_failed'] += 1

        return {
            'is_consistent': is_consistent,
            'extracted_answer': extracted_answer,
            'hard_label': hard_label,
            'is_in_allowed_set': is_in_allowed_set,
            'reason': 'Consistent' if is_consistent else 'Answer mismatch'
        }

    # ===== 第三层：语义过滤 =====

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的余弦相似度

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            相似度分数（0-1）
        """
        if not NUMPY_AVAILABLE:
            # 简单回退：基于词重叠率
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

        # 使用TF-IDF风格的词频向量
        def text_to_vector(text: str) -> Dict[str, int]:
            words = re.findall(r'\b\w+\b', text.lower())
            vector = {}
            for word in words:
                vector[word] = vector.get(word, 0) + 1
            return vector

        vec1 = text_to_vector(text1)
        vec2 = text_to_vector(text2)

        # 计算余弦相似度
        all_words = set(vec1.keys()) | set(vec2.keys())

        if not all_words:
            return 0.0

        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in all_words)
        norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def extract_cot_sections(self, cot_text: str) -> Dict[str, str]:
        """
        提取CoT的三段内容

        Args:
            cot_text: CoT文本

        Returns:
            包含observation、analysis、conclusion的字典
        """
        sections = {
            'observation': '',
            'analysis': '',
            'conclusion': ''
        }

        if not cot_text:
            return sections

        # 定义标签模式
        patterns = {
            'observation': r'Observation\s*:(.*?)(?=Analysis\s*:|Conclusion\s*:|Final Answer\s*:|$)',
            'analysis': r'Analysis\s*:(.*?)(?=Conclusion\s*:|Final Answer\s*:|$)',
            'conclusion': r'(?:Conclusion|Final Answer)\s*:(.*?)$'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, cot_text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()

        return sections

    def check_observation_analysis_similarity(
        self,
        cot_text: str,
        threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        检查Observation与Analysis的相似度

        Qwen-VL官方方案：余弦相似度 > 0.9，丢弃

        Args:
            cot_text: CoT文本
            threshold: 相似度阈值

        Returns:
            检查结果字典
        """
        sections = self.extract_cot_sections(cot_text)
        observation = sections['observation']
        analysis = sections['analysis']

        if not observation or not analysis:
            return {
                'is_valid': True,  # 缺少段落时不标记为无效
                'similarity': 0.0,
                'reason': 'Missing observation or analysis section'
            }

        similarity = self.calculate_text_similarity(observation, analysis)
        is_valid = similarity <= threshold

        if is_valid:
            self.stats['similarity_passed'] += 1
        else:
            self.stats['similarity_failed'] += 1

        return {
            'is_valid': is_valid,
            'similarity': similarity,
            'reason': 'Similarity too high' if not is_valid else 'Valid'
        }

    def check_occlusion_rule(
        self,
        cot_text: str,
        detected_objects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检查遮挡规则

        Qwen-VL官方方案：
        - 遮挡物体必须写"cannot recognize"
        - 如果目标被遮挡但文本写"fully exposed"，过滤

        Args:
            cot_text: CoT文本
            detected_objects: 检测到的物体列表（可选）

        Returns:
            检查结果字典
        """
        if not cot_text:
            return {'is_valid': True, 'reason': 'No CoT text'}

        # 遮挡相关关键词
        occlusion_keywords = ['blocked', 'occluded', 'hidden', 'covered', 'obscured', 'partially visible']
        clear_keywords = ['fully visible', 'completely visible', 'clearly visible', 'fully exposed']

        cot_lower = cot_text.lower()

        # 检查是否有遮挡描述
        has_occlusion = any(kw in cot_lower for kw in occlusion_keywords)
        has_clear = any(kw in cot_lower for kw in clear_keywords)

        # 检查是否有"cannot recognize"标注
        has_cannot_recognize = 'cannot recognize' in cot_lower or 'unable to recognize' in cot_lower

        # 规则检查
        is_valid = True
        reason = 'Valid'

        if has_occlusion and has_clear and not has_cannot_recognize:
            # 存在遮挡但写了"fully visible"，且没有标注"cannot recognize"
            is_valid = False
            reason = 'Occlusion conflict: mentions occlusion but also says fully visible'

        if is_valid:
            self.stats['occlusion_passed'] += 1
        else:
            self.stats['occlusion_failed'] += 1

        return {
            'is_valid': is_valid,
            'has_occlusion': has_occlusion,
            'has_clear': has_clear,
            'has_cannot_recognize': has_cannot_recognize,
            'reason': reason
        }

    # ===== 第四层：图像哈希去重 =====

    def compute_image_hash(self, image_path: str) -> Optional[str]:
        """
        计算图像哈希值

        Args:
            image_path: 图像路径

        Returns:
            图像哈希值，失败返回None
        """
        if not PIL_AVAILABLE or not IMAGEHASH_AVAILABLE:
            # 回退到文件内容哈希
            try:
                with open(image_path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
            except Exception:
                return None

        try:
            img = Image.open(image_path)
            # 使用感知哈希（pHash）
            img_hash = imagehash.phash(img)
            return str(img_hash)
        except Exception:
            return None

    def check_image_duplicate(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        检查图像是否重复

        Qwen-VL官方方案：重复图片只保留1条标注

        Args:
            image_path: 图像路径
            image_id: 图像ID（可选）

        Returns:
            检查结果字典，包含：
            - is_duplicate: 是否重复
            - hash_value: 哈希值
            - first_occurrence: 首次出现的图像ID
        """
        # 计算哈希
        img_hash = self.compute_image_hash(image_path)

        if not img_hash:
            return {
                'is_duplicate': False,
                'hash_value': None,
                'first_occurrence': None,
                'reason': 'Failed to compute hash'
            }

        # 使用image_id或image_path作为标识
        img_id = image_id or image_path

        # 检查是否已存在
        is_duplicate = img_hash in self._hash_to_images

        first_occurrence = None
        if is_duplicate:
            first_occurrence = self._hash_to_images[img_hash][0]
            self.stats['duplicate_removed'] += 1

        # 记录哈希
        if not is_duplicate:
            self._hash_to_images[img_hash] = [img_id]
            self._image_hashes[img_id] = img_hash
        else:
            self._hash_to_images[img_hash].append(img_id)

        return {
            'is_duplicate': is_duplicate,
            'hash_value': img_hash,
            'first_occurrence': first_occurrence,
            'reason': 'Duplicate detected' if is_duplicate else 'Unique'
        }

    # ===== 幻觉兜底检测 =====

    def detect_hallucination(
        self,
        cot_text: str,
        allowed_answers: Optional[List[str]] = None,
        coco_objects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检测CoT文本中的幻觉

        Args:
            cot_text: CoT文本
            allowed_answers: 允许的答案集（闭合集校验）
            coco_objects: 图像中存在的COCO物体列表

        Returns:
            检测结果字典
        """
        if not cot_text:
            return {'has_hallucination': False, 'issues': []}

        issues = []
        cot_lower = cot_text.lower()

        # 1. 闭合集校验：检查CoT中的答案是否超出allowed_answers
        if allowed_answers:
            allowed_set = set(a.lower() for a in allowed_answers)
            extracted_answer = self.extract_conclusion_answer(cot_text)
            if extracted_answer and extracted_answer not in allowed_set:
                issues.append({
                    'type': 'closed_set_violation',
                    'detail': f'Answer "{extracted_answer}" not in allowed set'
                })

        # 2. COCO物体幻觉检测：检查是否提到不存在的物体
        if coco_objects:
            existing_objects = set(o.lower() for o in coco_objects)
            mentioned_objects = []

            for category in self.COCO_CATEGORIES:
                if category.lower() in cot_lower:
                    mentioned_objects.append(category)

            # 检查是否有不存在的物体被提及
            hallucinated_objects = [obj for obj in mentioned_objects if obj not in existing_objects]
            if hallucinated_objects:
                issues.append({
                    'type': 'object_hallucination',
                    'detail': f'Mentioned non-existing objects: {hallucinated_objects}'
                })

        # 3. 推测词检测
        speculative_words = ['appear', 'seem', 'probably', 'likely', 'maybe', 'might', 'could be']
        found_speculative = [w for w in speculative_words if w in cot_lower]
        if found_speculative:
            issues.append({
                'type': 'speculative_language',
                'detail': f'Contains speculative words: {found_speculative}'
            })

        return {
            'has_hallucination': len(issues) > 0,
            'issues': issues,
            'issue_count': len(issues)
        }

    # ===== 综合清洗方法 =====

    def clean_sample(
        self,
        sample: Dict[str, Any],
        image_path: Optional[str] = None,
        check_duplicate: bool = True
    ) -> Dict[str, Any]:
        """
        对单个样本进行完整清洗检查

        Args:
            sample: 样本数据字典，应包含：
                - hard_label: 硬标签答案
                - cot: CoT文本（包含structured_reasoning）
                - soft_label: 软标签分布（包含allowed_answers）
            image_path: 图像路径（用于去重）
            check_duplicate: 是否检查重复

        Returns:
            清洗结果字典，包含：
            - is_valid: 是否有效
            - consistency_check: 一致性检查结果
            - similarity_check: 相似度检查结果
            - occlusion_check: 遮挡规则检查结果
            - duplicate_check: 重复检查结果
            - hallucination_check: 幻觉检查结果
        """
        results = {
            'is_valid': True,
            'checks_passed': 0,
            'checks_failed': 0
        }

        # 提取数据
        hard_label = sample.get('hard_label', {}).get('answer', '')
        cot_data = sample.get('cot', {})
        cot_text = cot_data.get('structured_reasoning', {}).get('conclusion', '')
        if not cot_text:
            cot_text = cot_data.get('full_response', '')

        soft_label = sample.get('soft_label', {})
        allowed_answers = soft_label.get('allowed_answers', [])

        # 第二层：一致性校验
        results['consistency_check'] = self.check_consistency(
            cot_text, hard_label, allowed_answers
        )
        if not results['consistency_check']['is_consistent']:
            results['is_valid'] = False
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1

        # 第三层：语义过滤
        # 相似度检查
        results['similarity_check'] = self.check_observation_analysis_similarity(cot_text)
        if not results['similarity_check']['is_valid']:
            results['is_valid'] = False
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1

        # 遮挡规则检查
        results['occlusion_check'] = self.check_occlusion_rule(cot_text)
        if not results['occlusion_check']['is_valid']:
            results['is_valid'] = False
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1

        # 第四层：图像去重
        if check_duplicate and image_path:
            results['duplicate_check'] = self.check_image_duplicate(image_path)
            if results['duplicate_check']['is_duplicate']:
                results['is_valid'] = False
                results['checks_failed'] += 1
            else:
                results['checks_passed'] += 1
        else:
            results['duplicate_check'] = {'is_duplicate': False, 'reason': 'Skipped'}

        # 幻觉检测
        results['hallucination_check'] = self.detect_hallucination(
            cot_text, allowed_answers
        )
        if results['hallucination_check']['has_hallucination']:
            # 幻觉检测作为警告，不直接标记为无效
            results['warnings'] = results.get('warnings', [])
            results['warnings'].append(results['hallucination_check'])

        if results['is_valid']:
            self.stats['valid_samples'] += 1

        return results

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        获取清洗统计信息

        Returns:
            统计信息字典
        """
        return {
            **self.stats,
            'consistency_pass_rate': (
                self.stats['consistency_passed'] / self.stats['total_processed'] * 100
                if self.stats['total_processed'] > 0 else 0
            ),
            'similarity_pass_rate': (
                self.stats['similarity_passed'] / self.stats['total_processed'] * 100
                if self.stats['total_processed'] > 0 else 0
            ),
            'occlusion_pass_rate': (
                self.stats['occlusion_passed'] / self.stats['total_processed'] * 100
                if self.stats['total_processed'] > 0 else 0
            )
        }

    def clear_cache(self):
        """清除缓存"""
        self._image_hashes.clear()
        self._hash_to_images.clear()


# ===== 推理兜底机制 =====

class InferenceFallback:
    """
    推理兜底机制

    功能：
    - OOM捕获：捕获显存溢出，跳过当前图片
    - 空输出重试：重试最多2次
    """

    def __init__(self, max_retries: int = 2):
        """
        初始化兜底机制

        Args:
            max_retries: 最大重试次数
        """
        self.max_retries = max_retries
        self.stats = {
            'oom_caught': 0,
            'empty_output_retries': 0,
            'empty_output_success': 0,
            'empty_output_failed': 0
        }

    def handle_oom(self, func, *args, **kwargs) -> Tuple[bool, Any]:
        """
        处理OOM异常

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            (success, result) 元组
        """
        import torch

        try:
            result = func(*args, **kwargs)
            return True, result
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats['oom_caught'] += 1
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False, None
            raise
        except Exception as e:
            raise

    def retry_on_empty_output(
        self,
        func,
        *args,
        check_empty_func=None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        空输出重试机制

        Args:
            func: 要执行的函数
            *args: 位置参数
            check_empty_func: 检查结果是否为空的函数
            **kwargs: 关键字参数

        Returns:
            (success, result) 元组
        """
        if check_empty_func is None:
            # 默认检查函数：检查结果是否为空字符串或None
            check_empty_func = lambda x: x is None or (isinstance(x, str) and not x.strip())

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                if not check_empty_func(result):
                    if attempt > 0:
                        self.stats['empty_output_success'] += 1
                    return True, result

                # 结果为空，尝试重试
                if attempt < self.max_retries:
                    self.stats['empty_output_retries'] += 1
                    continue
                else:
                    self.stats['empty_output_failed'] += 1
                    return False, None

            except Exception as e:
                raise

        return False, None

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()


# ===== 使用示例 =====
if __name__ == '__main__':
    # 创建清洗器
    cleaner = QwenDataCleaner()

    # 示例1：CoT一致性校验
    cot_text = """
    Observation: I can see one person wearing headphones clearly.
    Analysis: The person is on the left side of the image.
    Conclusion: Final Answer: one
    """
    hard_label = "one"

    result = cleaner.check_consistency(cot_text, hard_label)
    print(f"Consistency check: {result}")

    # 示例2：相似度检查
    similarity_result = cleaner.check_observation_analysis_similarity(cot_text)
    print(f"Similarity check: {similarity_result}")

    # 示例3：完整样本清洗
    sample = {
        'hard_label': {'answer': 'one'},
        'cot': {'structured_reasoning': {'conclusion': cot_text}},
        'soft_label': {'allowed_answers': ['zero', 'one', 'two']}
    }

    cleaning_result = cleaner.clean_sample(sample)
    print(f"Full cleaning result: {cleaning_result}")

    # 打印统计信息
    print(f"\nCleaning stats: {cleaner.get_cleaning_stats()}")