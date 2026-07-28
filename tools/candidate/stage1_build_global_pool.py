"""
阶段1：构建全局超大候选池 C_all
================================

千问官方三层答案来源，逐层合并去重：
- 底层：VQA人工标注答案（最高优先级）
- 中层：COCO Caption关键词提取（补充长尾视觉答案）
- 顶层：千问大教师模型零样本扩充答案（关键增强步骤）

核心修复：全局前置分流（从源头杜绝脏词混入）
- 采集答案 → 全局清洗 → 全局前置分流 → 频次过滤
- 输出分类候选集：color / location / number / yesno / object

使用方式：
    python tools/candidate/stage1_build_global_pool.py
    python tools/candidate/stage1_build_global_pool.py --use-teacher-model --teacher-model models/Qwen2.5-VL-32B-Instruct-AWQ
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Tuple
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_vqa_annotations(annotations_dir: Path) -> Tuple[Set[str], Counter]:
    """
    底层：从VQA人工标注中提取答案

    Returns:
        answers: 答案集合
        answer_counter: 答案频次统计
    """
    print("\n" + "="*60)
    print("【底层】VQA人工标注答案提取")
    print("="*60)

    answers = set()
    answer_counter = Counter()

    # VQA标注文件路径（只使用训练数据，避免数据泄露）
    vqa_files = [
        annotations_dir / "v2_mscoco_train2014_annotations.json",
        # 注意：不使用 val 数据，候选集应该只基于训练数据构建
    ]

    total_questions = 0
    total_raw_answers = 0
    filtered_by_clean = 0

    for vqa_file in vqa_files:
        if not vqa_file.exists():
            print(f"⚠️  文件不存在: {vqa_file}")
            continue

        print(f"📖 加载VQA标注: {vqa_file}")

        try:
            with open(vqa_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = data.get('annotations', [])

            for ann in annotations:
                total_questions += 1

                # 多个众包答案
                if 'answers' in ann:
                    for ans_obj in ann['answers']:
                        raw_answer = ans_obj.get('answer', '').strip()
                        total_raw_answers += 1

                        # 🔧 应用全局标准清洗
                        cleaned_answer = global_standard_clean(raw_answer)
                        if cleaned_answer:
                            answers.add(cleaned_answer)
                            answer_counter[cleaned_answer] += 1
                        else:
                            filtered_by_clean += 1

                # 单个答案
                elif 'answer' in ann:
                    raw_answer = ann['answer'].strip()
                    total_raw_answers += 1

                    # 🔧 应用全局标准清洗
                    cleaned_answer = global_standard_clean(raw_answer)
                    if cleaned_answer:
                        answers.add(cleaned_answer)
                        answer_counter[cleaned_answer] += 1
                    else:
                        filtered_by_clean += 1

            print(f"  ✓ 加载 {len(annotations)} 条标注")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")

    print(f"\n✓ 底层提取完成：")
    print(f"  总问题数: {total_questions}")
    print(f"  原始答案数: {total_raw_answers}")
    print(f"  清洗过滤: {filtered_by_clean} 个答案")
    print(f"  有效答案数: {len(answers)}")
    print(f"  总答案频次: {sum(answer_counter.values())}")

    return answers, answer_counter


def extract_caption_keywords(annotations_dir: Path) -> Tuple[Set[str], Counter]:
    """
    中层：从COCO Captions中提取关键词（补充长尾视觉答案）

    提取：名词、颜色、数字、位置短语
    """
    print("\n" + "="*60)
    print("【中层】COCO Caption关键词提取")
    print("="*60)

    keywords = set()
    keyword_counter = Counter()

    caption_files = [
        annotations_dir / "captions_train2014.json",
        # 注意：不使用 val 数据，候选集应该只基于训练数据构建
    ]

    # 颜色词表
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'black', 'white', 'gray', 'grey', 'cyan', 'magenta'
    ]

    # 数字词表
    numbers = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
        'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'
    ]

    # 位置词表
    positions = [
        'left', 'right', 'top', 'bottom', 'center', 'middle', 'front',
        'back', 'side', 'corner', 'above', 'below', 'behind', 'near'
    ]

    total_captions = 0
    total_raw_keywords = 0
    filtered_by_clean = 0

    for caption_file in caption_files:
        if not caption_file.exists():
            print(f"⚠️  文件不存在: {caption_file}")
            continue

        print(f"📖 加载COCO Captions: {caption_file}")

        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = data.get('annotations', [])

            for ann in annotations:
                total_captions += 1
                caption = ann.get('caption', '').lower().strip()

                if not caption:
                    continue

                # 简单分词
                words = caption.split()

                for word in words:
                    # 清理标点符号
                    word = re.sub(r'[^\w]', '', word).strip()

                    if not word:
                        continue

                    total_raw_keywords += 1

                    # 🔧 应用全局标准清洗
                    cleaned_word = global_standard_clean(word)
                    if not cleaned_word:
                        filtered_by_clean += 1
                        continue

                    # 提取颜色
                    if cleaned_word in colors:
                        keywords.add(cleaned_word)
                        keyword_counter[cleaned_word] += 1

                    # 提取数字
                    elif cleaned_word in numbers:
                        keywords.add(cleaned_word)
                        keyword_counter[cleaned_word] += 1

                    # 提取位置
                    elif cleaned_word in positions:
                        keywords.add(cleaned_word)
                        keyword_counter[cleaned_word] += 1

                    # 提取名词（简单规则：长度>2的单词）
                    elif len(cleaned_word) > 2 and cleaned_word.isalpha():
                        keywords.add(cleaned_word)
                        keyword_counter[cleaned_word] += 1

            print(f"  ✓ 加载 {len(annotations)} 条描述")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")

    print(f"\n✓ 中层提取完成：")
    print(f"  总描述数: {total_captions}")
    print(f"  原始关键词数: {total_raw_keywords}")
    print(f"  清洗过滤: {filtered_by_clean} 个关键词")
    print(f"  有效关键词数: {len(keywords)}")
    print(f"  总关键词频次: {sum(keyword_counter.values())}")

    return keywords, keyword_counter


def expand_with_teacher_model(
    answers: Set[str],
    annotations_dir: Path,
    teacher_model: str,
    max_samples: int = 1000,
    prompt_config: Dict = None,
    generation_params: Dict = None
) -> Tuple[Set[str], Counter]:
    """
    顶层：使用千问大教师模型零样本扩充答案

    ⚠️ 重要约束：
    1. 大模型扩充仅针对标准物体名词（person/vehicle/food等白名单内词汇）
    2. color/location/number/yesno 四类分支禁止使用大模型扩充
    3. 扩充输出规则：
       - 只输出同义标准名词、规范单复数
       - 禁止输出带颜色、方位、数字、形容词修饰的复合短语
       - 禁止输出否定词、位置描述词

    冻结教师模型，输入「物体名词」，生成补充同义答案、单复数变体

    Args:
        answers: 答案集合（应该是物体名词，不应包含color/location/number/yesno）
        annotations_dir: 标注文件目录
        teacher_model: 教师模型路径
        max_samples: 扩充样本数
        prompt_config: Prompt配置（从tools.yaml读取）
        generation_params: 生成参数（从tools.yaml读取）
    """
    print("\n" + "="*60)
    print("【顶层】千问大教师模型零样本扩充答案")
    print("="*60)
    print("⚠️  扩充范围：仅针对标准物体名词")
    print("   - color/location/number/yesno 分支使用固定词库，不进行扩充")
    print("   - 扩充输出严格限制：同义名词 + 规范单复数")

    expanded_answers = set()
    expanded_counter = Counter()

    # 🔧 默认配置（如果未传入）
    if prompt_config is None:
        prompt_config = {}
    if generation_params is None:
        generation_params = {}

    # 输出配置信息
    if prompt_config:
        print("✓ 使用自定义Prompt配置")
    else:
        print("ℹ️  使用默认Prompt配置")

    if generation_params:
        print("✓ 使用自定义生成参数")
        print(f"  temperature: {generation_params.get('temperature', 0.7)}")
        print(f"  max_new_tokens: {generation_params.get('max_new_tokens', 50)}")
    else:
        print("ℹ️  使用默认生成参数")

    try:
        # 🔧 设置离线模式（本地模型）
        import os
        model_path = Path(teacher_model)
        if not model_path.is_absolute():
            model_path = Path.cwd() / teacher_model

        if model_path.exists():
            print(f"✓ 找到本地模型: {model_path.absolute()}")
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # 导入模型
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        print(f"🔧 加载教师模型: {teacher_model}")

        # 🔧 检测是否是AWQ模型
        is_awq = 'awq' in teacher_model.lower()

        if is_awq:
            # AWQ 模型：直接加载到 GPU（不支持 device_map="auto" 包含 CPU/disk）
            print("  检测到 AWQ 模型，使用全 GPU 加载")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                teacher_model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1'
            ).cuda()
        else:
            # 非 AWQ 模型：可以使用 device_map="auto"
            print("  使用 device_map='auto'")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                teacher_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1'
            )

        processor = AutoProcessor.from_pretrained(
            teacher_model,
            trust_remote_code=True,
            local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1'
        )

        print(f"✓ 模型加载成功")

        # 🔧 检查模型设备位置
        if hasattr(model, 'device'):
            print(f"  模型设备: {model.device}")
        else:
            print(f"  警告: 无法获取模型设备信息")

        # 🔧 检查显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  显存占用: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        # 加载VQA问题和图像（示例）
        # 这里简化实现，只对部分答案进行扩充
        print(f"\n扩充答案（前{min(len(answers), max_samples)}个）...")
        print(f"💡 提示: 前5个答案会显示详细调试信息")

        sample_answers = list(answers)[:max_samples]

        for i, answer in enumerate(sample_answers):
            if i % 100 == 0:
                print(f"  处理进度: {i}/{len(sample_answers)}")

            # 🔧 使用教师模型生成答案变体
            try:
                # 🔧 前5个答案启用debug模式，帮助诊断问题
                enable_debug = (i < 5)
                variants = generate_answer_variants_with_model(
                    answer,
                    model,
                    processor,
                    prompt_config=prompt_config,
                    generation_params=generation_params,
                    debug=enable_debug  # 启用debug模式
                )
            except Exception as e:
                # 如果模型生成失败，使用规则生成
                if i == 0:  # 只在第一次失败时打印
                    print(f"  ⚠️ 模型生成失败，使用规则生成: {e}")
                variants = generate_answer_variants(answer)

            for variant in variants:
                expanded_answers.add(variant)
                expanded_counter[variant] += 1

        print(f"\n✓ 顶层扩充完成：")
        print(f"  原始答案数: {len(answers)}")
        print(f"  扩充后答案数: {len(expanded_answers)}")

    except Exception as e:
        print(f"⚠️  教师模型扩充失败: {e}")
        print("  使用简单规则扩充")

        # 简单规则扩充
        for answer in list(answers)[:max_samples]:
            variants = generate_answer_variants(answer)
            for variant in variants:
                expanded_answers.add(variant)
                expanded_counter[variant] += 1

    return expanded_answers, expanded_counter


def generate_answer_variants_with_model(
    answer: str,
    model,
    processor,
    prompt_config: Dict = None,
    generation_params: Dict = None,
    debug: bool = True  # 🔧 添加debug参数
) -> List[str]:
    """
    使用千问教师模型生成答案变体

    Prompt要求模型输出「所有简短答案、同义词、单复数，逗号分隔」

    Args:
        answer: 原答案
        model: 教师模型
        processor: 处理器
        prompt_config: Prompt配置（从tools.yaml读取）
        generation_params: 生成参数（从tools.yaml读取）
        debug: 是否打印详细调试信息
    """
    import time
    start_time = time.time()

    if debug:
        print(f"\n    [DEBUG] 开始处理答案: {answer}")

    # 🔧 从配置构建prompt
    if prompt_config and 'template' in prompt_config:
        template = prompt_config['template']
        prompt = template.format(answer=answer)
    else:
        # 默认prompt（后备）
        prompt = f"""请列出答案"{answer}"的所有可能的同义答案、单复数变体、近似表达。

要求：
1. 输出所有简短答案、同义词、单复数变体
2. 用逗号分隔
3. 不要有多余文字
4. 包含原文答案

示例：
输入：dog
输出：dog, dogs, puppy, puppies, canine, hound

输入：three
输出：three, 3

输入答案：{answer}
输出："""

    try:
        import torch

        # 🔧 构建messages格式（Qwen2.5-VL要求）
        messages = [
            {"role": "user", "content": prompt}
        ]

        if debug:
            print(f"    [DEBUG] Prompt长度: {len(prompt)} 字符")

        # Apply chat template
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if debug:
                print(f"    [DEBUG] Chat template应用成功")
        except Exception as e:
            print(f"  ⚠️ apply_chat_template失败: {e}")
            # 后备方案：直接使用prompt
            text = prompt
            if debug:
                print(f"    [DEBUG] 使用后备prompt方案")

        # 编码输入（纯文本，无图像）
        if debug:
            print(f"    [DEBUG] 开始编码输入...")

        try:
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            if debug:
                print(f"    [DEBUG] Processor编码成功")
        except Exception as e:
            print(f"  ⚠️ processor编码失败: {e}")
            # 后备方案：使用tokenizer
            inputs = processor.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in inputs.items()}
            if debug:
                print(f"    [DEBUG] Tokenizer编码成功（后备方案）")

        # 移动到设备
        if debug:
            print(f"    [DEBUG] 移动输入到GPU...")

        try:
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            if debug:
                print(f"    [DEBUG] 输入已移动到GPU")
        except Exception as e:
            print(f"  ⚠️ 移动到设备失败: {e}")
            # 继续执行，可能已经在正确的设备上

        # 🔧 从配置获取生成参数（只设置必要的默认值）
        gen_params = {
            'max_new_tokens': 128,  # 提高默认值
            'pad_token_id': processor.tokenizer.pad_token_id,
            'eos_token_id': processor.tokenizer.eos_token_id
        }

        # 默认使用贪婪解码（确定性输出）
        if not generation_params or 'do_sample' not in generation_params:
            gen_params['do_sample'] = False  # 贪婪解码

        # 如果有配置，使用配置参数（覆盖默认值）
        if generation_params:
            # 🔧 过滤掉注释（YAML可能把注释也解析进去了）
            filtered_params = {
                k: v for k, v in generation_params.items()
                if not k.startswith('#') and v is not None
            }
            gen_params.update(filtered_params)

        # 🔧 确保贪婪解码模式下不包含采样参数
        if gen_params.get('do_sample') == False:
            # 移除采样参数，避免警告
            gen_params.pop('temperature', None)
            gen_params.pop('top_p', None)
            gen_params.pop('top_k', None)

        # 🔧 关键：生成前打印
        if debug:
            print(f"    [DEBUG] 开始模型生成...")
            print(f"    [DEBUG] 生成参数: max_new_tokens={gen_params.get('max_new_tokens')}, do_sample={gen_params.get('do_sample')}")
            gen_start = time.time()

        # 🔧 添加超时检测（仅用于debug）
        max_wait_time = 60  # 最长等待60秒

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_params
            )

        # 🔧 关键：生成后打印
        if debug:
            gen_time = time.time() - gen_start
            print(f"    [DEBUG] ✓ 模型生成完成（耗时: {gen_time:.2f}秒）")
            if gen_time > 30:
                print(f"    [WARNING] 生成耗时过长（>{gen_time:.1f}秒），可能存在问题！")

        # 解码输出
        if debug:
            print(f"    [DEBUG] 开始解码输出...")

        generated_text = processor.decode(
            outputs[0][len(inputs['input_ids'][0]):],  # 只解码新生成的部分
            skip_special_tokens=True
        )

        if debug:
            print(f"    [DEBUG] 解码完成: {generated_text[:50]}...")

        # 提取生成的答案（去掉可能的对话标记）
        generated_text = generated_text.strip()

        # 如果输出中包含对话标记，提取assistant部分
        if '<|im_end|>' in generated_text:
            generated_text = generated_text.split('<|im_end|>')[0].strip()

        # 提取逗号分隔的答案
        # 处理可能的输出格式
        generated_part = generated_text

        # 尝试找到实际的答案列表
        if '输出：' in generated_part:
            generated_part = generated_part.split('输出：')[-1].strip()
        elif '输出:' in generated_part:
            generated_part = generated_part.split('输出:')[-1].strip()

        # 如果还有其他格式标记，尝试清理
        # 例如：去掉可能的新行、多余空格
        generated_part = generated_part.split('\n')[0].strip()

        # 分割答案
        variants = []
        for item in generated_part.split(','):
            item = item.strip().lower()
            # 过滤无效答案
            if item and len(item) > 0 and len(item) < 50:  # 简短答案
                # 🔧 更严格地清理特殊字符
                # 只保留字母、数字、空格、连字符
                import re
                # 如果包含特殊字符（除了连字符），过滤掉
                if re.search(r'[\\/@#$%^&*()+=\[\]{}|;:<>]', item):
                    if debug:
                        print(f"    [DEBUG] 过滤含特殊字符: {item}")
                    continue
                # 如果是数字范围（如 82-2900），过滤掉
                if re.search(r'\d+-\d+', item):
                    if debug:
                        print(f"    [DEBUG] 过滤数字范围: {item}")
                    continue
                # 去掉可能的前后符号
                item = item.strip('0123456789.- ')
                if item:
                    variants.append(item)

        # 确保原答案在列表中
        if answer.lower() not in variants:
            variants.insert(0, answer.lower())

        # 🔧 新增：输出结果过滤（移除不符合约束的答案）
        filtered_variants = []
        for variant in variants:
            # 跳过空答案
            if not variant or len(variant) == 0:
                continue

            # 跳过过长答案（>2个单词）
            if len(variant.split()) > 2:
                if debug:
                    print(f"    [DEBUG] 过滤过长答案: {variant}")
                continue

            # 跳过带颜色修饰的答案
            color_words = {"white", "black", "red", "blue", "green", "yellow", "orange",
                           "purple", "pink", "brown", "gray", "grey"}
            if any(color in variant for color in color_words) and variant != answer.lower():
                if debug:
                    print(f"    [DEBUG] 过滤带颜色修饰: {variant}")
                continue

            # 跳过带方位修饰的答案
            location_words = {"left", "right", "front", "back", "top", "bottom",
                              "center", "side", "corner"}
            if any(loc in variant for loc in location_words):
                if debug:
                    print(f"    [DEBUG] 过滤带方位修饰: {variant}")
                continue

            # 跳过带数字修饰的答案
            import re
            if re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+\w+', variant):
                if debug:
                    print(f"    [DEBUG] 过滤带数字修饰: {variant}")
                continue

            # 跳过带形容词修饰的答案
            adj_words = {"old", "young", "big", "small", "tall", "short", "long",
                         "little", "large", "huge", "tiny", "fat", "thin"}
            if any(adj in variant.split() for adj in adj_words):
                if debug:
                    print(f"    [DEBUG] 过滤带形容词修饰: {variant}")
                continue

            # 跳过否定词
            neg_words = {"no", "none", "not", "nothing", "never", "neither", "nobody"}
            if variant in neg_words:
                if debug:
                    print(f"    [DEBUG] 过滤否定词: {variant}")
                continue

            # 🔧 跳过动词形式（时态、分词）
            # VQA答案应该是名词或形容词，动词形式不符合候选集要求
            verb_forms = {
                # -ing 形式（现在分词）
                "holding", "riding", "sitting", "standing", "walking", "running",
                "eating", "drinking", "wearing", "carrying", "playing", "sleeping",
                "flying", "swimming", "driving", "reading", "writing", "talking",
                "looking", "watching", "listening", "cooking", "cleaning", "washing",
                "tapped", "wrapped", "marked", "filled", "covered", "painted",
                "attached", "connected", "opened", "closed", "folded", "rolled",
                # -ed 形式（过去式/过去分词）
                "worn", "eaten", "ridden", "driven", "flown", "swum", "written",
                "spoken", "taken", "given", "seen", "heard", "known", "shown",
                # 其他动词形式
                "does", "doing", "did", "done", "goes", "going", "went", "gone",
                "comes", "coming", "came", "gets", "getting", "got", "gotten",
                "puts", "putting", "takes", "taking", "took", "taken"
            }
            # 检查是否是动词形式（包括单复数变形）
            if variant in verb_forms or variant.rstrip('s') in verb_forms or variant.rstrip('es') in verb_forms:
                if debug:
                    print(f"    [DEBUG] 过滤动词形式: {variant}")
                continue

            # 🔧 跳过包含特殊字符的答案
            if re.search(r'[\\/@#$%^&*()+=\[\]{}|;:<>]', variant):
                if debug:
                    print(f"    [DEBUG] 过滤特殊字符: {variant}")
                continue

            # 🔧 跳过数字范围（如 82-2900）
            if re.search(r'\d+-\d+', variant):
                if debug:
                    print(f"    [DEBUG] 过滤数字范围: {variant}")
                continue

            filtered_variants.append(variant)

        if debug:
            total_time = time.time() - start_time
            print(f"    [DEBUG] ✓ 处理完成（总耗时: {total_time:.2f}秒）")
            print(f"    [DEBUG] 原始变体数: {len(variants)}, 过滤后: {len(filtered_variants)}")
            if filtered_variants:
                print(f"    [DEBUG] 最终变体: {', '.join(filtered_variants[:10])}")

        return filtered_variants

    except Exception as e:
        if debug:
            print(f"    [DEBUG] ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
        # 如果失败，返回原答案和基本变体
        return [answer.lower(), answer.lower() + 's']


def generate_answer_variants(answer: str) -> List[str]:
    """
    生成答案变体（单复数、同义词等）

    简化版本：使用规则生成
    """
    variants = [answer]

    # 单复数变体
    if answer.endswith('s'):
        variants.append(answer[:-1])  # 复数 → 单数
    else:
        variants.append(answer + 's')  # 单数 → 复数

    # 数字变体
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    if answer in number_map:
        variants.append(number_map[answer])
    for word, digit in number_map.items():
        if answer == digit:
            variants.append(word)

    return variants


def global_standard_clean(raw_ans: str) -> str:
    """
    全局标准清洗函数（千问官方蒸馏标准）

    在全局候选池去重前执行，拦截90%基础噪声

    三层防御机制：
    1. 全局基础规则：拦截多词短语，只放行单单词和固定复合词白名单
    2. 全局修饰词黑名单兜底：包含修饰词的多词短语直接丢弃
    3. 场景白名单强匹配：剩余答案必须完全匹配场景白名单（在stage3执行）

    Args:
        raw_ans: 原始答案字符串

    Returns:
        清洗后的答案，如果需要过滤则返回 None
    """
    ans = raw_ans.strip().lower()
    words = ans.split()
    word_set = set(words)

    # ========== 逻辑1：全局基础规则 ==========
    # 全局固定复合标准答案白名单（COCO VQA高频标准多词答案）
    global_compound_whitelist = {
        "hot dog",
        "ice cream",
        "french fries",
        "cell phone",
        "mobile phone",
        "base ball",
        "soft drink",
        "banana split",
        "teddy bear",
        "dining table",
        "traffic light",
        "fire hydrant",
        "parking meter",
        "stop sign",
        "potted plant"
    }

    # 基础判断：不是1个单词，进入复合词白名单校验
    if len(words) != 1:
        # 只有完全匹配固定复合词白名单才保留，其余多词全部丢弃
        if ans not in global_compound_whitelist:
            return None

    # ========== 逻辑2：全局修饰词黑名单兜底（双重保险）==========
    # 方位介词，出现直接丢弃
    position_words = {"on", "at", "under", "behind", "in", "front", "next", "near",
                      "above", "below", "beside", "between", "behind", "across"}

    # 形容词修饰词，拦截 old man / white car / big bus
    adj_words = {"old", "young", "big", "small", "white", "black", "red", "blue", "green",
                 "public", "tour", "police", "fishing", "pickup", "toy", "speed", "party",
                 "tall", "short", "long", "little", "large", "huge", "tiny", "fat", "thin"}

    # 附属/限定名词：driver、show、seat、bed、platform
    suffix_modify = {"driver", "show", "seat", "bed", "platform", "depot", "crossing",
                     "yard", "track", "route", "terminal", "stand", "station", "stop"}

    # 无关专有名词
    proper_nonsense = {"wonder", "pac", "thomas", "magic", "metro", "queilen",
                       "batman", "spiderman", "superman", "hulk", "iron"}

    all_modify_black = position_words | adj_words | suffix_modify | proper_nonsense
    if word_set & all_modify_black:
        return None
    # ==================================================================

    # ========== 原有过滤逻辑保留 ==========
    # 1. 过滤and复合短语
    if "and" in word_set:
        return None

    # 2. 过滤数字开头数量短语
    num_tokens = {"1", "2", "3", "4", "two", "three", "four", "five"}
    if words and words[0] in num_tokens:
        return None

    # 3. 过滤否定词汇
    neg_tokens = {"no", "none", "not", "zero"}
    if word_set & neg_tokens:
        return None

    # 🔧 4. 过滤动词形式（时态、分词）
    # VQA答案应该是名词或形容词，动词形式不符合候选集要求
    verb_forms = {
        # -ing 形式（现在分词）
        "holding", "riding", "sitting", "standing", "walking", "running",
        "eating", "drinking", "wearing", "carrying", "playing", "sleeping",
        "flying", "swimming", "driving", "reading", "writing", "talking",
        "looking", "watching", "listening", "cooking", "cleaning", "washing",
        "tapped", "wrapped", "marked", "filled", "covered", "painted",
        "attached", "connected", "opened", "closed", "folded", "rolled",
        # -ed 形式（过去式/过去分词）
        "worn", "eaten", "ridden", "driven", "flown", "swum", "written",
        "spoken", "taken", "given", "seen", "heard", "known", "shown",
        # 其他动词形式
        "does", "doing", "did", "done", "goes", "going", "went", "gone",
        "comes", "coming", "came", "gets", "getting", "got", "gotten",
        "puts", "putting", "takes", "taking", "took", "taken"
    }
    if ans in verb_forms or ans.rstrip('s') in verb_forms or ans.rstrip('es') in verb_forms:
        return None

    # 🔧 5. 过滤特殊字符（-, \, /, @, #, $等）
    # 答案应该是纯单词或数字，不应包含特殊符号
    import re
    # 检测是否包含特殊字符（除了空格和连字符）
    if re.search(r'[^a-z0-9\s\-]', ans):
        # 允许连字符在特定复合词中（如 hot-dog）
        # 但不允许单独出现的特殊字符组合（如 82-2900, a\b, c/d）
        if re.search(r'[\d]+-[\d]+', ans):  # 数字范围（如 82-2900）
            return None
        if re.search(r'[\\/@#$%^&*()+=\[\]{}|;:<>]', ans):  # 其他特殊字符
            return None
        # 连字符连接的复合词，检查是否在白名单中
        if '-' in ans and ans not in global_compound_whitelist:
            # 检查是否是合理的复合词（如 hot-dog）
            # 允许：字母-字母
            # 拒绝：数字-数字，字母-数字混合
            if re.search(r'\d+-\d+', ans):  # 数字-数字
                return None
            if re.search(r'\d+-[a-z]+', ans):  # 数字-字母
                return None
            if re.search(r'[a-z]+-\d+', ans):  # 字母-数字
                return None
            # 连字符连接的复合词，只允许白名单中的
            if '-' in ans:
                return None

    # 6. 过滤所有格's
    if "'s" in ans:
        return None

    # 5. 过滤动作动词
    verb_tokens = {"hold", "holding", "ride", "riding", "cut", "cutting", "eat", "eating"}
    if word_set & verb_tokens:
        return None

    # 6. 过滤错误复数变形
    wrong_plural = {"mans", "womans", "childs", "childrens", "peoples"}
    if ans in wrong_plural:
        return None

    # 单复数归一（官方标准操作，合并重复条目）
    plural_norm = {
        "cars": "car", "trains": "train", "children": "child", "people": "person",
        "bicycles": "bicycle", "boats": "boat", "motorcycles": "motorcycle",
        "trucks": "truck", "airplanes": "airplane", "buses": "bus"
    }
    if ans in plural_norm:
        ans = plural_norm[ans]

    return ans


def categorize_answers(answers: Set[str]) -> Dict[str, List[str]]:
    """
    步骤1：全局前置分流（从源头杜绝脏词进入 object）

    将候选集分为4类：color、location、number、object

    Args:
        answers: 原始答案集合

    Returns:
        分类后的候选集字典
    """
    print("\n" + "="*60)
    print("【步骤1】全局前置分流")
    print("="*60)

    # 预定义三类脏词集合（分流标准）
    color_set = {
        "black", "white", "red", "blue", "green", "orange", "brown",
        "gray", "grey", "purple", "pink", "gold", "silver", "tan",
        "yellow", "cyan", "magenta", "turquoise", "beige", "cream"
    }

    location_set = {
        "left", "right", "middle", "front", "back", "top", "bottom",
        "center", "side", "corner", "above", "below", "behind", "near",
        "outside", "inside", "foreground", "background"
    }

    number_set = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
    }

    yesno_set = {"yes", "no"}

    # 初始化分类容器
    candidates = {
        "color": [],
        "location": [],
        "number": [],
        "yesno": [],
        "object": []  # 纯物体答案
    }

    stats = {
        "total": len(answers),
        "color": 0,
        "location": 0,
        "number": 0,
        "yesno": 0,
        "object": 0
    }

    for ans in answers:
        ans_lower = ans.strip().lower()

        # 分流逻辑：按优先级分流
        if ans_lower in color_set:
            candidates["color"].append(ans_lower)
            stats["color"] += 1
        elif ans_lower in location_set:
            candidates["location"].append(ans_lower)
            stats["location"] += 1
        elif ans_lower in number_set:
            candidates["number"].append(ans_lower)
            stats["number"] += 1
        elif ans_lower in yesno_set:
            candidates["yesno"].append(ans_lower)
            stats["yesno"] += 1
        else:
            # 剩下的进入 object 候选池
            candidates["object"].append(ans_lower)
            stats["object"] += 1

    print(f"\n✓ 分流完成：")
    print(f"  总答案数: {stats['total']}")
    print(f"  颜色答案: {stats['color']} 个 → candidates['color']")
    print(f"  方位答案: {stats['location']} 个 → candidates['location']")
    print(f"  数字答案: {stats['number']} 个 → candidates['number']")
    print(f"  二元答案: {stats['yesno']} 个 → candidates['yesno']")
    print(f"  物体答案: {stats['object']} 个 → candidates['object']")

    return candidates


def clean_global_pool(
    answers: Set[str],
    answer_counter: Counter,
    min_freq: int = 2,
    max_words: int = 8
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    全局池清洗过滤 + 分流

    流程（修正后）：
    1. 全局清洗（过滤噪声）
    2. 全局前置分流（分出 color/location/number/yesno/object）
    3. 频次过滤

    Args:
        answers: 答案集合
        answer_counter: 答案频次统计
        min_freq: 最低频次阈值
        max_words: 最大单词数

    Returns:
        categorized_candidates: 分类后的候选集
        freq_map: 答案频次映射
    """
    print("\n" + "="*60)
    print("全局池清洗过滤 + 分流")
    print("="*60)

    # 🔧 关键词黑名单（否定词、厨具、品牌、无关物品）
    BLACKLIST = {
        # 否定词
        'no', 'not', 'none', 'nothing', 'never', 'neither', 'nobody', 'nowhere',
        'cannot', 'cant', 'dont', 'wont', 'shouldnt', 'couldnt', 'wouldnt',
        'unknown', 'unclear', 'undetermined', 'n/a', 'na', 'null', 'nil',

        # 厨具（kitchen场景相关，但不应作为VQA答案）
        'utensil', 'cutlery', 'silverware', 'tableware', 'cookware', 'bakeware',
        'appliance', 'stove', 'range', 'hob', 'oven', 'microwave', 'blender',
        'mixer', 'toaster', 'kettle', 'coffeemaker', 'cooker',

        # 品牌（不应出现在通用VQA答案中）
        'nike', 'adidas', 'apple', 'samsung', 'sony', 'lg', 'dell', 'hp',
        'lenovo', 'asus', 'acer', 'microsoft', 'google', 'amazon', 'facebook',
        'twitter', 'instagram', 'youtube', 'netflix', 'spotify', 'uber', 'tesla',
        'ford', 'toyota', 'bmw', 'mercedes', 'audi', 'honda', 'nissan', 'volkswagen',
        'starbucks', 'mcdonalds', 'subway', 'kfc', 'pizzahut', 'dominos',
        'cocacola', 'pepsi', 'nivea', 'dove', 'olay', 'louis', 'vuitton', 'gucci',
        'prada', 'chanel', 'dior', 'hermes', 'zara', 'h&m', 'uniqlo',

        # 无关物品（与COCO场景无关的通用名词）
        'thing', 'stuff', 'item', 'object', 'product', 'device', 'widget',
        'gadget', 'tool', 'equipment', 'material', 'substance', 'element',
        'component', 'part', 'piece', 'fragment', 'segment', 'section',
        'structure', 'formation', 'arrangement', 'configuration',

        # 其他噪声词
        'etc', 'or', 'but', 'yet', 'so', 'because', 'although',
        'however', 'therefore', 'moreover', 'furthermore', 'meanwhile'
    }

    print(f"原始答案数: {len(answers)}")
    print(f"黑名单词数: {len(BLACKLIST)}")

    # 步骤1：全局清洗（过滤噪声）
    cleaned_answers = set()
    filtered_by_blacklist = 0
    filtered_by_and = 0

    for answer in answers:
        # 长度过滤
        word_count = len(answer.split())
        if word_count > max_words:
            continue

        # 噪声过滤
        if not answer.strip():
            continue

        if len(answer) == 1 and not answer.isdigit():
            continue

        if not any(c.isalpha() or c.isdigit() for c in answer):
            continue

        answer_lower = answer.lower()
        answer_words = set(answer_lower.split())

        # 黑名单过滤
        if answer_words & BLACKLIST:
            filtered_by_blacklist += 1
            continue

        # "and" 过滤
        if ' and ' in answer_lower:
            filtered_by_and += 1
            continue

        cleaned_answers.add(answer_lower)

    print(f"\n✓ 全局清洗完成：")
    print(f"  清洗后答案数: {len(cleaned_answers)}")
    print(f"  黑名单过滤: {filtered_by_blacklist} 个")
    print(f"  'and' 过滤: {filtered_by_and} 个")

    # 步骤2：全局前置分流
    categorized_candidates = categorize_answers(cleaned_answers)

    # 步骤3：频次过滤（构建频次映射）
    freq_map = {}
    for ans in cleaned_answers:
        freq = answer_counter.get(ans, 0)
        if freq >= min_freq or ans in (categorized_candidates["color"] +
                                       categorized_candidates["location"] +
                                       categorized_candidates["number"] +
                                       categorized_candidates["yesno"]):
            # 保留频次>=min_freq的，或者基础答案（颜色/方位/数字/二元）
            freq_map[ans] = freq

    print(f"\n✓ 频次过滤完成：")
    print(f"  频次>={min_freq}的答案数: {len(freq_map)}")

    return categorized_candidates, freq_map


def main():
    parser = argparse.ArgumentParser(description="阶段1：构建全局超大候选池")

    parser.add_argument(
        '--config',
        default='configs/tools.yaml',
        help='配置文件路径（默认：configs/tools.yaml）'
    )

    parser.add_argument(
        '--annotations-dir',
        default='data/coco/annotations',
        help='COCO标注文件目录'
    )

    parser.add_argument(
        '--output',
        default='data/global_candidate_pool.json',
        help='输出文件路径'
    )

    parser.add_argument(
        '--use-teacher-model',
        action='store_true',
        help='使用千问教师模型扩充答案'
    )

    parser.add_argument(
        '--teacher-model',
        default='models/Qwen2.5-VL-32B-Instruct-AWQ',
        help='千问教师模型路径'
    )

    parser.add_argument(
        '--max-expand-samples',
        type=int,
        default=1000,
        help='教师模型扩充样本数'
    )

    parser.add_argument(
        '--min-freq',
        type=int,
        default=2,
        help='最低频次阈值'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("阶段1：构建全局超大候选池 C_all")
    print("="*60)
    print("\n千问官方三层答案来源：")
    print("  底层：VQA人工标注答案（最高优先级）")
    print("  中层：COCO Caption关键词提取（补充长尾视觉答案）")
    print("  顶层：千问大教师模型零样本扩充答案（关键增强步骤）")

    annotations_dir = Path(args.annotations_dir)

    # 🔧 加载配置文件
    config = None
    prompt_config = None
    generation_params = None

    if args.config:
        try:
            import yaml
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
                    config = full_config.get('candidate_closure', {})

                    # 从配置读取prompt和生成参数
                    prompt_config = config.get('teacher_prompt', {})
                    generation_params = config.get('generation_params', {})

                    print(f"\n✓ 加载配置文件: {config_path}")
                    if prompt_config:
                        print(f"  ✓ 使用配置中的Prompt模板")
                    if generation_params:
                        print(f"  ✓ 使用配置中的生成参数")

        except Exception as e:
            print(f"⚠️  配置文件加载失败: {e}")
            print("  使用默认配置")

    # 底层：VQA人工标注
    vqa_answers, vqa_counter = load_vqa_annotations(annotations_dir)

    # 中层：COCO Caption关键词
    caption_keywords, caption_counter = extract_caption_keywords(annotations_dir)

    # 🔧 步骤1：合并底层和中层答案
    print("\n" + "="*60)
    print("合并底层和中层答案")
    print("="*60)

    base_answers = vqa_answers | caption_keywords
    base_counter = Counter()
    base_counter.update(vqa_counter)
    base_counter.update(caption_counter)

    print(f"✓ 底层（VQA）: {len(vqa_answers)} 个答案")
    print(f"  中层（Caption）: {len(caption_keywords)} 个关键词")
    print(f"  合并后: {len(base_answers)} 个唯一答案")

    # 🔧 步骤2：全局基础清洗（过滤and短语、所有格、错误复数等）
    print("\n" + "="*60)
    print("【步骤2】全局基础清洗")
    print("="*60)

    cleaned_answers = set()
    cleaned_counter = Counter()

    for ans in base_answers:
        cleaned = global_standard_clean(ans)
        if cleaned:
            cleaned_answers.add(cleaned)
            cleaned_counter[cleaned] = base_counter.get(ans, 0)

    print(f"✓ 清洗完成: {len(base_answers)} → {len(cleaned_answers)} 个答案")

    # 🔧 步骤3：全局分流拆分（color/location/number/yesno/object）
    print("\n" + "="*60)
    print("【步骤3】全局分流拆分")
    print("="*60)

    categorized_candidates = categorize_answers(cleaned_answers)

    # 🔧 步骤4：教师模型扩充（仅对object分支）
    if args.use_teacher_model:
        print("\n" + "="*60)
        print("【步骤4】教师模型扩充（仅针对物体名词）")
        print("="*60)

        # 只对object分支进行扩充
        object_answers = set(categorized_candidates["object"])

        print(f"ℹ️  扩充范围：仅对 {len(object_answers)} 个物体名词进行扩充")
        print(f"  color/location/number/yesno 分支使用固定词库，不进行扩充")

        expanded_object_answers, expanded_object_counter = expand_with_teacher_model(
            object_answers,
            annotations_dir,
            args.teacher_model,
            args.max_expand_samples,
            prompt_config=prompt_config,
            generation_params=generation_params
        )

        # 将扩充后的答案添加到object分支
        original_object_count = len(categorized_candidates["object"])
        for ans in expanded_object_answers:
            if ans not in object_answers:  # 去重
                categorized_candidates["object"].append(ans)

        print(f"\n✓ 物体名词扩充完成:")
        print(f"  原始物体名词: {original_object_count} 个")
        print(f"  扩充后物体名词: {len(categorized_candidates['object'])} 个")
        print(f"  新增同义词: {len(categorized_candidates['object']) - original_object_count} 个")

        # 更新计数器
        all_counter = Counter()
        all_counter.update(cleaned_counter)
        all_counter.update(expanded_object_counter)
    else:
        print("\nℹ️  教师模型扩充未启用")
        all_counter = cleaned_counter

    # 🔧 步骤5：频次过滤
    print("\n" + "="*60)
    print("【步骤5】频次过滤")
    print("="*60)

    # 对object分支进行频次过滤
    min_freq = args.min_freq
    object_filtered = [
        ans for ans in categorized_candidates["object"]
        if all_counter.get(ans, 0) >= min_freq
    ]

    # color/location/number/yesno保持不变（使用固定词库）
    print(f"✓ 物体名词频次过滤完成:")
    print(f"  过滤前: {len(categorized_candidates['object'])} 个")
    print(f"  过滤后（频次>={min_freq}）: {len(object_filtered)} 个")

    categorized_candidates["object"] = object_filtered

    # 🔧 构建全局候选池（合并所有类别，但保留分类信息）
    global_pool = []
    global_pool.extend(categorized_candidates["color"])
    global_pool.extend(categorized_candidates["location"])
    global_pool.extend(categorized_candidates["number"])
    global_pool.extend(categorized_candidates["yesno"])
    global_pool.extend(categorized_candidates["object"])

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建频次映射
    freq_map = {ans: all_counter.get(ans, 0) for ans in global_pool}

    output_data = {
        'metadata': {
            'source': 'Qwen Official Pipeline Stage 1 (Fixed v2.1)',
            'description': '严格约束版：大模型仅扩充物体名词',
            'layers': {
                'vqa': len(vqa_answers),
                'caption': len(caption_keywords),
                'expansion': 'object_only' if args.use_teacher_model else 'none'
            },
            'total_unique': len(global_pool),
            'categorized': {
                'color': len(categorized_candidates["color"]),
                'location': len(categorized_candidates["location"]),
                'number': len(categorized_candidates["number"]),
                'yesno': len(categorized_candidates["yesno"]),
                'object': len(categorized_candidates["object"])
            },
            'global_pool_size': len(global_pool),
            'config': {
                'min_freq': args.min_freq,
                'use_teacher_model': args.use_teacher_model,
                'expansion_scope': 'object_branch_only'
            }
        },
        'global_pool': global_pool,
        'categorized_candidates': categorized_candidates,
        'freq_map': freq_map
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✓ 阶段1完成")
    print("="*60)
    print(f"\n输出文件: {output_path}")
    print(f"\n分类统计：")
    print(f"  颜色答案: {len(categorized_candidates['color'])} 个")
    print(f"  方位答案: {len(categorized_candidates['location'])} 个")
    print(f"  数字答案: {len(categorized_candidates['number'])} 个")
    print(f"  二元答案: {len(categorized_candidates['yesno'])} 个")
    print(f"  物体答案: {len(categorized_candidates['object'])} 个")
    print(f"  总计: {len(global_pool)} 个答案")

    print(f"\n前20个高频物体答案：")
    object_sorted = sorted(
        categorized_candidates['object'],
        key=lambda x: freq_map.get(x, 0),
        reverse=True
    )
    for i, ans in enumerate(object_sorted[:20], 1):
        freq = freq_map.get(ans, 0)
        print(f"  {i:2d}. {ans:20s} (频次: {freq})")


if __name__ == "__main__":
    main()