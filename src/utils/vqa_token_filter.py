"""
VQA Token 过滤器
==================

用于剔除VQA软标签中的噪音token，只保留有效答案。
配置路径从主配置文件(default.yaml)读取，配置内容在独立的YAML文件中。

新增功能：
- 词干归一化（Stemming）：将 animal/animals 视为同一候选
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Union
import yaml


class VQATokenFilter:
    """
    VQA Token过滤器

    根据问题类型和答案特征，智能过滤噪音token
    配置路径从主配置文件读取，配置内容在独立的YAML文件中

    新增功能：
    - 词干归一化（Stemming）：统一处理单复数形式
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化过滤器

        Args:
            config_path: 主配置文件路径，默认为 configs/default.yaml
                        如果为None，则自动查找
        """
        # 加载主配置文件，获取token过滤器配置路径
        self.main_config_path = config_path
        self.config = self._load_config()

        # 初始化所有配置
        self._init_all_configs()

        # 🔧 新增：词干归一化配置
        self.enable_stemming = self.config.get('stemming', {}).get('enabled', True)

    def _load_config(self) -> Dict:
        """
        加载配置：先从default.yaml读取路径，再加载独立配置文件

        Returns:
            配置字典
        """
        # Step 1: 确定主配置文件路径
        if self.main_config_path is None:
            self.main_config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"

        self.main_config_path = Path(self.main_config_path)

        # Step 2: 读取主配置文件，获取token过滤器配置路径
        try:
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                main_config = yaml.safe_load(f)

            # 从主配置中读取配置文件路径
            vqa_filter_config = main_config.get('vqa_token_filter', {})
            config_file = vqa_filter_config.get('config_file')

            if not config_file:
                raise ValueError("主配置文件中未找到 vqa_token_filter.config_file 路径")

            # Step 3: 构建完整路径并加载独立配置文件
            config_file_path = Path(__file__).parent.parent.parent / config_file

            if not config_file_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_file_path}")

            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            print(f"✓ 加载VQA Token过滤器配置: {config_file_path}")
            return config

        except FileNotFoundError as e:
            raise RuntimeError(f"配置文件不存在: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")

    def _init_all_configs(self):
        """初始化所有配置项"""
        # 初始化任务白名单
        self._init_task_whitelists()

        # 初始化等价token映射
        self._init_equivalent_tokens()

        # 初始化黑名单
        self._init_blacklists()

        # 初始化问题关键词
        self._init_question_keywords()

    def _init_task_whitelists(self):
        """初始化任务白名单"""
        whitelists_config = self.config.get('task_whitelists', {})

        self.task_whitelists = {}
        for task_type, tokens in whitelists_config.items():
            self.task_whitelists[task_type] = set(tokens)

        # 兼容旧代码
        self.number_answers = self.task_whitelists.get('count', set())
        self.color_answers = self.task_whitelists.get('color', set())
        self.binary_answers = self.task_whitelists.get('binary', set())
        self.valid_answers = self.number_answers | self.color_answers | self.binary_answers

        # 添加位置和大小答案（保留兼容）
        self.location_answers = {
            'left', 'right', 'top', 'bottom', 'center', 'middle',
            'front', 'back', 'side', 'corner', 'edge'
        }
        self.size_answers = {
            'big', 'small', 'large', 'tiny', 'huge', 'medium',
            'tall', 'short', 'long', 'wide', 'narrow'
        }
        self.valid_answers |= self.location_answers | self.size_answers

    def _init_equivalent_tokens(self):
        """初始化等价token映射"""
        equiv_config = self.config.get('equivalent_tokens', {})

        self.equivalent_tokens = {}

        # 处理配置格式：标准形式 -> [变体列表]
        for canonical, variants in equiv_config.items():
            # 添加标准形式到自身的映射
            self.equivalent_tokens[canonical] = canonical
            # 添加所有变体到标准形式的映射
            if isinstance(variants, list):
                for variant in variants:
                    self.equivalent_tokens[variant] = canonical

        # 创建反向映射
        self.canonical_to_variants = {}
        for variant, canonical in self.equivalent_tokens.items():
            if canonical not in self.canonical_to_variants:
                self.canonical_to_variants[canonical] = []
            self.canonical_to_variants[canonical].append(variant)

    def _init_blacklists(self):
        """初始化黑名单"""
        blacklists_config = self.config.get('blacklists', {})

        # 特殊token黑名单
        self.special_token_blacklist = set(blacklists_config.get('special_tokens', []))

        # 空格黑名单
        self.whitespace_blacklist = set(blacklists_config.get('whitespace', []))

        # 标点符号黑名单
        self.punctuation_blacklist = set(blacklists_config.get('punctuation', []))

        # 纯符号组合黑名单
        self.symbol_only_blacklist = set(blacklists_config.get('symbol_only', []))

        # 极短碎片黑名单
        self.short_fragment_blacklist = set(blacklists_config.get('short_fragments', []))

        # 噪音词汇（合并多个分类）
        self.noise_words = set()
        noise_config = blacklists_config.get('noise_words', [])
        if isinstance(noise_config, list):
            self.noise_words = set(noise_config)
        elif isinstance(noise_config, dict):
            for category, words in noise_config.items():
                if isinstance(words, list):
                    self.noise_words.update(words)

        # 多语言噪音词
        multilingual_noise = blacklists_config.get('multilingual_noise', [])
        if isinstance(multilingual_noise, list):
            self.noise_words.update(multilingual_noise)

        # 特殊Unicode字符（保留硬编码，因为难以在YAML中表示）
        self.special_unicode_tokens = {
            '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',
            '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',
        }

        # 子词模式（保留硬编码，因为是正则表达式）
        self.subword_patterns = [
            r'^[_\-\.\(]',
            r'[_\-\.\)]$',
            r'^\[',
        ]

        self.fragment_patterns = [
            r'^[A-Z][a-z]$',
            r'^##',
            r'^@@',
        ]

        # 🔧 新增：截断词黑名单（基于观察到的BPE碎片模式）
        # 这些词通常是完整单词被tokenizer切分后的前缀部分
        self.truncated_word_blacklist = {
            # 2-4字符的常见截断词（观察到的实际案例）
            'fil', 'phot', 'rec', 'dj', 'sk', 'dr', 'mus', 'bl', 'sh', 'st',
            'obs', 'jam', 'box', 'vid', 'lis', 'wat', 'lis', 'hol',
            'play', 'sing', 'stan', 'walk', 'take', 'hold', 'list', 'watch',
            'per', 'for', 'back', 'groun', 'grou', 'ckgroun',

            # 可能的动词截断
            'stand', 'sitt', 'lyi', 'runn', 'jump', 'clim',
        }

        # 🔧 新增：有效短词白名单（即使长度短也允许）
        # 这些是常见的有效答案，不是截断词
        self.valid_short_words = {
            # 数字答案
            'one', 'two', 'six', 'ten',

            # 颜色答案
            'red', 'blue',

            # 是非答案
            'yes', 'no',

            # 常见名词（3-4字符）
            'cat', 'dog', 'car', 'bus', 'cup', 'jar', 'sky', 'sun',
            'hat', 'pen', 'key', 'map', 'bag', 'box', 'bed', 'door',

            # 常见动词（完整形式）
            'sit', 'run', 'eat', 'cry', 'walk', 'talk', 'play',

            # 常见缩写
            'tv', 'pc', 'id',

            # 常见介词/副词（可能作为答案）
            'left', 'right', 'top', 'down', 'near', 'far',
        }

    def _init_question_keywords(self):
        """初始化问题类型关键词"""
        keywords_config = self.config.get('question_keywords', {})

        self.number_question_keywords = set(keywords_config.get('count', []))
        self.color_question_keywords = set(keywords_config.get('color', []))
        self.binary_question_keywords = set(keywords_config.get('binary', []))

    # ===== 过滤方法 =====

    def is_valid_token(self, token: str, question: Optional[str] = None) -> bool:
        """
        判断token是否为有效答案

        Args:
            token: 要检查的token
            question: 问题文本（保留参数兼容，但不用于类型限制）

        Returns:
            True如果token有效，False如果是噪音
        """
        token_lower = token.lower().strip()

        # 第一层：黑名单（核心防线）

        # 1. 空token
        if not token_lower:
            return False

        # 2. 检查纯空格/空字符串
        if token_lower in self.whitespace_blacklist:
            return False

        # 3. 检查特殊Token
        if token_lower in self.special_token_blacklist:
            return False

        # 4. 检查标点符号黑名单
        if token_lower in self.punctuation_blacklist:
            return False

        # 5. 检查纯符号组合黑名单
        if token_lower in self.symbol_only_blacklist:
            return False

        # 6. 检查BPE子词碎片
        if token.startswith('Ġ') or token.startswith('Ġ'):
            return False

        # 7. 检查子词片段
        for pattern in self.subword_patterns:
            if re.search(pattern, token_lower):
                return False

        # 8. 检查碎片token特征
        for pattern in self.fragment_patterns:
            if re.search(pattern, token):
                return False

        # 9. 检查极短碎片黑名单
        if token_lower in self.short_fragment_blacklist:
            return False

        # 10. 检查噪音词汇
        if token_lower in self.noise_words:
            return False

        # 11. 检查特殊Unicode字符
        if token_lower in self.special_unicode_tokens:
            return False

        # 🔧 新增：12. 检查截断词黑名单
        if token_lower in self.truncated_word_blacklist:
            return False

        # 🔧 新增：13. 检查特殊符号组合模式（如 "*t", ">t", "**"等）
        # 这些通常是BPE切分产生的噪音token
        if len(token_lower) <= 3:
            # 检查是否包含特殊符号+字母的组合
            special_chars = {'*', '>', '<', '`', "'", '"', '#', '@', '$', '%', '^', '&', '|', '\\', '/', '~'}
            has_special = any(c in special_chars for c in token_lower)
            has_alpha = any(c.isalpha() for c in token_lower)
            # 如果同时包含特殊符号和字母，很可能是噪音
            if has_special and has_alpha:
                return False
            # 纯特殊符号组合（如 "**"）
            if has_special and not has_alpha:
                return False

        # 第二层：有效答案检查

        # 检查是否在有效答案列表中
        if token_lower in self.valid_answers:
            return True

        # 检查是否是纯数字
        if token_lower.isdigit():
            return True

        # 🔧 新增：检查是否在有效短词白名单中
        if token_lower in self.valid_short_words:
            return True

        # 🔧 改进：长度检查（更严格，针对可能的截断词）
        if len(token_lower) < 2 or len(token_lower) > 20:
            return False

        # 🔧 新增：启发式规则 - 对短词（<=4字符）进行更严格的检查
        if len(token_lower) <= 4:
            # 如果已经通过前面的黑名单检查，且在有效短词列表中，则允许
            if token_lower in self.valid_short_words:
                return True

            # 如果是数字，允许
            if token_lower.isdigit():
                return True

            # 否则，保守地认为可能是截断词，需要进一步验证
            # 这里我们可以检查词频词典或其他特征
            # 为了安全起见，暂时拒绝
            return False

        # 默认：长度>4的token更可能是完整单词，允许通过
        return True

    def filter_distribution(
        self,
        distribution: Dict[str, float],
        question: Optional[str] = None,
        primary_answer: Optional[str] = None,
        min_prob: float = 0.001,
        max_answers: int = 50
    ) -> Dict[str, float]:
        """
        过滤答案分布，剔除噪音token

        Args:
            distribution: 原始答案分布
            question: 问题文本
            primary_answer: 主答案（确保保留）
            min_prob: 最小概率阈值
            max_answers: 最大保留答案数

        Returns:
            过滤后的答案分布
        """
        filtered = {}
        primary_answer_lower = primary_answer.lower() if primary_answer else None

        # 第一轮：应用过滤器
        for token, prob in distribution.items():
            if prob < min_prob:
                continue

            token_lower = token.lower().strip()

            # 强制保留主答案
            if primary_answer_lower and token_lower == primary_answer_lower:
                filtered[token_lower] = prob
                continue

            # 应用过滤器
            if self.is_valid_token(token_lower, question):
                filtered[token_lower] = prob

        # 第二轮：如果过滤后为空，至少保留主答案
        if not filtered and primary_answer_lower:
            for token, prob in distribution.items():
                if token.lower() == primary_answer_lower:
                    filtered[primary_answer_lower] = prob
                    break

        # 第三轮：限制答案数量
        if len(filtered) > max_answers:
            sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_items[:max_answers])

        # 第四轮：归一化
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}

        return filtered

    def get_answer_type(self, question: str) -> str:
        """
        判断问题类型

        Args:
            question: 问题文本

        Returns:
            问题类型：'count', 'color', 'binary', 'other'
        """
        question_lower = question.lower()

        if any(kw in question_lower for kw in self.number_question_keywords):
            return 'count'
        elif any(kw in question_lower for kw in self.color_question_keywords):
            return 'color'
        elif any(kw in question_lower for kw in self.binary_question_keywords):
            return 'binary'
        else:
            return 'other'

    def infer_task_type(self, question: str, hard_label: str) -> str:
        """
        推断任务类型（优先基于hard_label，其次基于question关键词）

        Args:
            question: 问题文本
            hard_label: 硬标签答案

        Returns:
            任务类型：'count', 'color', 'binary', 'unknown'
        """
        hard_lower = hard_label.lower().strip()

        # 优先：基于hard_label判断
        if hard_lower in self.task_whitelists.get('count', set()) or hard_lower.isdigit():
            return 'count'
        if hard_lower in self.task_whitelists.get('color', set()):
            return 'color'
        if hard_lower in self.task_whitelists.get('binary', set()):
            return 'binary'

        # 次要：基于question关键词判断
        task_type = self.get_answer_type(question)
        return task_type

    def filter_by_task_type(
        self,
        distribution: Dict[str, float],
        task_type: str,
        hard_label: str,
        preserve_hard_label: bool = True
    ) -> Dict[str, float]:
        """
        根据任务类型应用白名单过滤

        Args:
            distribution: 原始分布
            task_type: 任务类型
            hard_label: 硬标签答案
            preserve_hard_label: 是否强制保留hard_label

        Returns:
            过滤后的分布
        """
        whitelist = self.task_whitelists.get(task_type, set())

        if not whitelist:
            return distribution

        filtered = {}
        hard_lower = hard_label.lower().strip()

        for token, prob in distribution.items():
            token_lower = token.lower().strip()

            # 强制保留hard_label
            if preserve_hard_label and token_lower == hard_lower:
                filtered[token_lower] = prob
                continue

            # 应用白名单
            if token_lower in whitelist:
                filtered[token_lower] = prob

        # 归一化
        if filtered:
            total = sum(filtered.values())
            if total > 0:
                filtered = {k: v/total for k, v in filtered.items()}

        return filtered

    def _stem_word(self, word: str) -> str:
        """
        词干归一化（Stemming）

        将单词转换为词干形式，统一处理单复数等变体。
        使用简单规则，避免引入nltk等重量级依赖。

        Args:
            word: 输入单词

        Returns:
            词干形式

        Examples:
            animals -> animal
            dogs -> dog
            cats -> cat
            watches -> watch
            boxes -> box
        """
        if not word or len(word) < 3:
            return word

        # 常见复数规则（按优先级排序）
        # 1. -ies -> -y (babies -> baby)
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'

        # 2. -es -> 去掉es (boxes -> box, watches -> watch)
        if word.endswith('es') and len(word) > 3:
            # 但要避免误处理如: cheese, geese
            if word.endswith('ses') or word.endswith('xes') or word.endswith('zes') or \
               word.endswith('ches') or word.endswith('shes'):
                return word[:-2]

        # 3. -s -> 去掉s (dogs -> dog, cats -> cat)
        if word.endswith('s') and not word.endswith('ss'):
            return word[:-1]

        return word

    def _get_stemmed_token(self, token: str) -> str:
        """
        获取token的词干形式（带缓存）

        Args:
            token: 输入token

        Returns:
            词干形式
        """
        if not self.enable_stemming:
            return token

        # 简单缓存（可选）
        token_lower = token.lower().strip()
        return self._stem_word(token_lower)

    def get_canonical_token(self, token: str) -> str:
        """
        获取token的标准形式（用于合并等价token）

        Args:
            token: 输入token

        Returns:
            标准形式的token
        """
        token_lower = token.lower().strip()

        # 如果在等价映射中，返回标准形式
        if token_lower in self.equivalent_tokens:
            return self.equivalent_tokens[token_lower]

        # 自动去除引号
        if token_lower and token_lower[0] in ['"', "'", '`']:
            unquoted = token_lower[1:]
            if unquoted in self.equivalent_tokens:
                return self.equivalent_tokens[unquoted]
            return unquoted

        return token_lower

    def merge_equivalent_tokens(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """
        合并等价token的概率

        新增功能：应用词干归一化，将 animal/animals 视为同一候选

        Args:
            distribution: 原始答案分布

        Returns:
            合并后的答案分布
        """
        merged = {}

        for token, prob in distribution.items():
            # Step 1: 获取标准形式（等价token映射）
            canonical = self.get_canonical_token(token)

            # Step 2: 应用词干归一化（如果启用）
            if self.enable_stemming:
                stemmed = self._get_stemmed_token(canonical)
            else:
                stemmed = canonical

            if stemmed in merged:
                merged[stemmed] += prob
            else:
                merged[stemmed] = prob

        # 归一化
        total = sum(merged.values())
        if total > 0:
            merged = {k: v / total for k, v in merged.items()}

        return merged

    def suggest_valid_answers(self, question: str) -> Set[str]:
        """
        根据问题类型，建议有效答案集

        Args:
            question: 问题文本

        Returns:
            建议的有效答案集合
        """
        answer_type = self.get_answer_type(question)

        if answer_type == 'count':
            return self.number_answers
        elif answer_type == 'color':
            return self.color_answers
        elif answer_type == 'binary':
            return self.binary_answers
        else:
            return self.valid_answers

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'VQATokenFilter':
        """
        从配置文件创建过滤器实例

        Args:
            config_path: 配置文件路径

        Returns:
            VQATokenFilter实例
        """
        return cls(config_path=config_path)


# ===== 使用示例 =====
if __name__ == '__main__':
    filter_tool = VQATokenFilter()

    # 示例1：数字问题
    question = "How many people are wearing headphones?"
    distribution = {
        'one': 0.0815,
        'two': 0.0738,
        'yes': 0.0361,
        'no': 0.0339,
        '-one': 0.0073,
        '_one': 0.0070,
        'uno': 0.0069,
        'black': 0.0035,
        'white': 0.0034,
    }

    filtered = filter_tool.filter_distribution(
        distribution=distribution,
        question=question,
        primary_answer='one'
    )

    print("问题:", question)
    print("原始分布:", distribution)
    print("过滤后:", filtered)

    # 示例2：颜色问题
    question2 = "What is the color of the water?"
    distribution2 = {
        'green': 0.588,
        'blue': 0.411,
        'one': 0.05,
        'two': 0.04,
        '_green': 0.03,
    }

    filtered2 = filter_tool.filter_distribution(
        distribution=distribution2,
        question=question2,
        primary_answer='green'
    )

    print("\n问题:", question2)
    print("原始分布:", distribution2)
    print("过滤后:", filtered2)