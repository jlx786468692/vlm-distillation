"""
阶段3：分场景过滤生成局部小闭合集
==================================

从全局池C_all中，为每个场景筛选场景强相关答案。

核心修复：
- 步骤2：分组后增加场景白名单强过滤（兜底清理残留脏词）
- 步骤3：补全全局黑名单过滤（二次拦截复合短语）
- 步骤4：修正流水线执行顺序（只对 object 候选集进行场景分组）

流程：
1. 加载 stage1 输出的分类候选集（color/location/number/yesno/object）
2. 基础答案直接作为独立场景
3. object 候选集进行场景分组 + 白名单强过滤
4. 输出分场景候选集

使用方式：
    python tools/candidate/stage3_scene_closure.py
    python tools/candidate/stage3_scene_closure.py --use-semantic-filter
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 🔧 导入全局清洗函数
from tools.candidate.stage1_build_global_pool import global_standard_clean


# COCO 12大场景的标准物体类别（简化版）
SCENE_OBJECTS = {
    'person': ['person', 'man', 'woman', 'child', 'baby', 'kid', 'people'],
    'vehicle': ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'train', 'airplane', 'boat'],
    'animal': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear'],
    'food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'cake'],
    'furniture': ['chair', 'couch', 'bed', 'dining table', 'toilet', 'desk', 'shelf'],
    'electronic': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave'],
    'appliance': ['oven', 'toaster', 'sink', 'refrigerator', 'blender', 'fan'],
    'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'skateboard'],
    'kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'plate'],
    'accessory': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'purse'],
    'indoor': ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier'],
    'outdoor': ['potted plant', 'bench', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter']
}

# 🔧 12类COCO标准场景纯净白名单（千问官方蒸馏标准）
# 规则：白名单内只有纯物体名词/规范复数/固定复合词，不包含任何形容词、方位、附属词汇
SCENE_FILTER_CONFIG = {
    "person": {
        "white_list": {"man", "woman", "person", "people", "child", "children", "baby", "kid", "kids", "human", "humans", "male", "female", "boy", "girl", "adult", "adults"},
        "black_list": {"left", "right", "middle", "front", "back", "behind", "another", "mother", "father", "parent", "holding", "wearing", "looking", "old", "young", "tall", "short"}
    },
    "vehicle": {
        "white_list": {"car", "cars", "bus", "buses", "bicycle", "bicycles", "motorcycle", "motorcycles", "truck", "trucks", "train", "trains", "boat", "boats", "airplane", "airplanes", "plane", "planes"},
        "black_list": {"station", "stop", "tracks", "fire", "school", "tow", "dump", "food", "ride", "driver", "public", "tour", "police", "taxi", "parking"}
    },
    "animal": {
        "white_list": {"dog", "dogs", "cat", "cats", "bird", "birds", "horse", "horses", "cow", "cows", "sheep", "elephant", "elephants", "bear", "bears"},
        "black_list": {"statue", "toy", "fake", "picture", "photo", "drawing", "plastic", "small", "big", "wild", "domestic"}
    },
    "food": {
        "white_list": {"pizza", "pizza", "cake", "cakes", "orange", "oranges", "apple", "apples", "banana", "bananas", "broccoli", "carrot", "carrots", "sandwich", "sandwiches",
                      "hot dog", "ice cream", "french fries", "soft drink", "banana split"},
        "black_list": {"knife", "cutter", "box", "oven", "hut", "restaurant", "store", "shop", "market", "slice", "piece", "half"}
    },
    "furniture": {
        "white_list": {"chair", "chairs", "couch", "sofa", "bed", "beds", "table", "tables", "desk", "desks", "shelf", "shelves", "cabinet", "cabinets",
                      "dining table"},
        "black_list": {"floor", "wall", "room", "house", "home", "office", "building", "wooden", "metal", "small", "large"}
    },
    "electronic": {
        "white_list": {"tv", "television", "laptop", "laptops", "computer", "computers", "phone", "phones", "cellphone", "mobile", "keyboard", "keyboards", "mouse",
                      "cell phone", "mobile phone"},
        "black_list": {"store", "shop", "office", "desk", "table", "room", "screen", "monitor", "small", "big"}
    },
    "appliance": {
        "white_list": {"oven", "ovens", "toaster", "toasters", "sink", "sinks", "refrigerator", "refrigerators", "fridge", "blender", "blenders", "fan", "fans", "microwave", "microwaves"},
        "black_list": {"store", "shop", "kitchen", "room", "house", "home", "small", "large", "electric"}
    },
    "sports": {
        "white_list": {"frisbee", "frisbees", "skis", "ski", "snowboard", "snowboards", "ball", "balls", "kite", "kites", "skateboard", "skateboards",
                      "base ball"},
        "black_list": {"store", "shop", "outside", "outdoor", "park", "field", "playing", "game", "small", "big"}
    },
    "kitchen": {
        "white_list": {"bottle", "bottles", "wine", "glass", "glasses", "cup", "cups", "fork", "forks", "knife", "knives", "spoon", "spoons", "bowl", "bowls", "plate", "plates"},
        "black_list": {"store", "shop", "restaurant", "cafe", "dining", "room", "small", "large", "plastic"}
    },
    "accessory": {
        "white_list": {"backpack", "backpacks", "umbrella", "umbrellas", "handbag", "handbags", "bag", "bags", "tie", "ties", "suitcase", "suitcases", "purse", "purses"},
        "black_list": {"store", "shop", "wearing", "holding", "carrying", "person", "man", "woman", "small", "large"}
    },
    "indoor": {
        "white_list": {"book", "books", "clock", "clocks", "vase", "vases", "scissors", "teddy", "bear", "bears", "doll", "dolls",
                      "teddy bear"},
        "black_list": {"store", "shop", "room", "house", "home", "office", "building", "small", "large"}
    },
    "outdoor": {
        "white_list": {"plant", "plants", "bench", "benches", "hydrant", "hydrants", "sign", "signs", "meter", "meters",
                      "traffic light", "fire hydrant", "parking meter", "stop sign", "potted plant"},
        "black_list": {"store", "shop", "building", "house", "home", "street", "road", "traffic", "light", "small", "large"}
    }
}


def load_global_pool(global_pool_file: Path) -> tuple:
    """
    加载全局候选池（包含分类候选集）

    Returns:
        global_pool: 全局候选池（合并）
        categorized_candidates: 分类候选集（color/location/number/yesno/object）
        freq_map: 频次映射
    """
    print(f"📖 加载全局候选池: {global_pool_file}")

    with open(global_pool_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    global_pool = data.get('global_pool', [])
    freq_map = data.get('freq_map', {})

    # 🔧 新增：加载分类候选集
    categorized_candidates = data.get('categorized_candidates', {
        'color': [],
        'location': [],
        'number': [],
        'yesno': [],
        'object': global_pool  # 如果没有分类信息，使用全局池作为 object
    })

    print(f"  ✓ 加载 {len(global_pool)} 个候选答案")
    print(f"  ✓ 分类候选集：")
    print(f"    - color: {len(categorized_candidates.get('color', []))} 个")
    print(f"    - location: {len(categorized_candidates.get('location', []))} 个")
    print(f"    - number: {len(categorized_candidates.get('number', []))} 个")
    print(f"    - yesno: {len(categorized_candidates.get('yesno', []))} 个")
    print(f"    - object: {len(categorized_candidates.get('object', []))} 个")

    return global_pool, categorized_candidates, freq_map


def load_scene_mapping(scene_mapping_file: Path) -> Dict:
    """加载图像-场景映射"""
    print(f"📖 加载场景映射: {scene_mapping_file}")

    with open(scene_mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    imgid2scene = data.get('imgid2scene', {})

    print(f"  ✓ 加载 {len(imgid2scene)} 张图像的场景映射")

    return imgid2scene


def is_color_combination(answer: str) -> bool:
    """
    检查是否是颜色组合短语（如 "orange and white"）

    Args:
        answer: 答案字符串

    Returns:
        True 如果是颜色组合，False 否则
    """
    # 常见颜色词
    colors = {
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'black', 'white', 'gray', 'grey', 'cyan', 'magenta'
    }

    # 检查是否包含 "and" 且两端都是颜色词
    if ' and ' in answer.lower():
        words = set(answer.lower().split())
        # 如果包含两个或以上颜色词，判定为颜色组合
        if len(words & colors) >= 2:
            return True

    return False


def is_brand_or_metaphor(answer: str, scene: str) -> bool:
    """
    检查是否是品牌或比喻用法（如 "apple laptop"）

    Args:
        answer: 答案字符串
        scene: 场景名称

    Returns:
        True 如果是品牌或比喻用法，False 否则
    """
    # 品牌词列表
    brands = {
        'nike', 'adidas', 'apple', 'samsung', 'sony', 'lg', 'dell', 'hp',
        'lenovo', 'asus', 'acer', 'microsoft', 'google', 'amazon', 'tesla'
    }

    # 检查是否包含品牌词
    answer_lower = answer.lower()
    for brand in brands:
        if brand in answer_lower:
            # 如果是food场景且包含apple，检查是否是水果apple
            if scene == 'food' and brand == 'apple':
                # 如果答案只是"apple"或"apples"，不是品牌
                if answer_lower.strip() in ['apple', 'apples']:
                    return False
                # 如果是"apple laptop"等，是品牌
                return True
            # 其他情况，包含品牌词即为品牌用法
            return True

    return False


def filter_by_objects(object_candidates: List[str], scene: str) -> List[str]:
    """
    过滤1：场景白名单强过滤（步骤2：兜底清理残留脏词）

    核心规则：
    1. 答案必须完全等于白名单内标准单词（不允许带任何修饰词）
    2. 答案不能包含黑名单词汇
    3. 不在白名单内的颜色/方位/数字脏词全部删除

    Args:
        object_candidates: object 类别的候选答案列表
        scene: 场景名称

    Returns:
        过滤后的候选答案列表
    """
    import re

    scene_filter_cfg = SCENE_FILTER_CONFIG.get(scene, {})
    white_list = scene_filter_cfg.get('white_list', set())
    black_list = scene_filter_cfg.get('black_list', set())

    # 🔧 步骤3：全局黑名单过滤（二次拦截复合短语）
    # 颜色、方位、数字脏词集合
    color_set = {
        "black", "white", "red", "blue", "green", "orange", "brown",
        "gray", "grey", "purple", "pink", "gold", "silver", "tan",
        "yellow", "cyan", "magenta"
    }

    location_set = {
        "left", "right", "middle", "front", "back", "top", "bottom",
        "center", "side", "corner", "above", "below", "behind", "near"
    }

    number_set = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "1", "2", "3", "4", "5"
    }

    # 过滤答案
    filtered = []

    for answer in object_candidates:
        answer_lower = answer.lower().strip()

        # 🔧 黑名单过滤：包含场景黑名单任意词汇 → 直接丢弃
        answer_words = set(answer_lower.split())
        if answer_words & black_list:
            continue

        # 🔧 白名单强过滤：答案必须完全等于白名单内标准单词
        if white_list and answer_lower not in white_list:
            # 不在白名单内，检查是否是脏词
            # 如果是颜色/方位/数字，直接丢弃（兜底清理）
            if answer_lower in color_set or answer_lower in location_set or answer_lower in number_set:
                print(f"  ⚠️  清理脏词: '{answer_lower}' 不在场景 '{scene}' 白名单内")
                continue
            # 如果不是脏词，也不在白名单，丢弃
            continue

        filtered.append(answer)

    return filtered


def filter_by_frequency(
    global_pool: List[str],
    scene: str,
    imgid2scene: Dict,
    vqa_file: Path,
    min_freq: int = 1
) -> List[str]:
    """
    过滤2：场景内VQA答案频次过滤

    仅保留全局池中场景内频次≥min_freq的答案
    """
    print(f"\n  加载场景内VQA答案...")

    # 找出该场景的所有图像
    scene_images = set([
        img_id for img_id, scene_data in imgid2scene.items()
        if scene_data.get('primary_scene') == scene
    ])

    print(f"  场景 '{scene}' 包含 {len(scene_images)} 张图像")

    # 统计场景内答案频次
    scene_answer_counter = Counter()

    # 🔧 如果有VQA标注文件，从标注中统计
    if vqa_file and vqa_file.exists():
        try:
            import json
            with open(vqa_file, 'r', encoding='utf-8') as f:
                vqa_data = json.load(f)

            annotations = vqa_data.get('annotations', [])

            for ann in annotations:
                # 检查是否属于该场景
                image_id = ann.get('image_id')
                if image_id not in scene_images:
                    continue

                # 统计答案
                if 'answers' in ann:
                    for ans_obj in ann['answers']:
                        answer = ans_obj.get('answer', '').lower().strip()
                        if answer:
                            confidence = ans_obj.get('answer_confidence', 'yes')
                            weight = {'yes': 1.0, 'maybe': 0.5, 'no': 0.1}.get(confidence, 1.0)
                            scene_answer_counter[answer] += weight

                elif 'answer' in ann:
                    answer = ann['answer'].lower().strip()
                    if answer:
                        scene_answer_counter[answer] += 1

            print(f"  ✓ 从VQA标注统计到 {len(scene_answer_counter)} 个不同答案")

        except Exception as e:
            print(f"  ⚠️  VQA标注加载失败: {e}")

    # 🔧 如果没有VQA标注或加载失败，使用全局频次作为后备
    if not scene_answer_counter:
        print(f"  ⚠️  使用全局频次作为后备")
        # 假设全局池中的高频答案在该场景内也高频
        # 这是一个简化的假设，实际应该从VQA标注中统计
        for answer in global_pool:
            scene_answer_counter[answer] = 1

    # 过滤低频答案
    filtered = [
        ans for ans in global_pool
        if scene_answer_counter.get(ans, 0) >= min_freq
    ]

    print(f"  ✓ 过滤后: {len(filtered)} 个答案 (min_freq={min_freq})")

    return filtered


def filter_by_semantic_similarity(
    global_pool: List[str],
    scene: str,
    caption_file: Path,
    threshold: float = 0.3,
    use_lightweight: bool = True
) -> List[str]:
    """
    过滤3：Caption语义相似度软过滤

    使用文本encoder计算候选答案与场景caption的向量相似度

    Args:
        global_pool: 候选答案列表
        scene: 场景名称
        caption_file: Caption文件路径
        threshold: 相似度阈值（默认0.3）
        use_lightweight: 是否使用轻量级实现（默认True，基于关键词匹配）

    Returns:
        过滤后的候选答案列表
    """
    print(f"\n  语义相似度过滤（阈值: {threshold}）...")

    if not caption_file.exists():
        print(f"  ⚠️  Caption文件不存在，跳过语义过滤")
        return global_pool

    try:
        # 🔧 轻量级实现：基于场景关键词匹配（推荐，无需额外模型）
        if use_lightweight:
            return _filter_by_scene_keywords(global_pool, scene, threshold)

        # 🔧 完整实现：基于Sentence-Transformers（需要安装额外依赖）
        else:
            return _filter_by_sentence_transformers(global_pool, scene, caption_file, threshold)

    except Exception as e:
        print(f"  ⚠️  语义相似度过滤失败: {e}")
        print(f"  使用轻量级关键词匹配作为后备")
        return _filter_by_scene_keywords(global_pool, scene, threshold)


def _filter_by_scene_keywords(global_pool: List[str], scene: str, threshold: float) -> List[str]:
    """
    轻量级语义过滤：基于场景关键词匹配

    保留与场景强相关的答案，过滤明显不相关的噪声

    Args:
        global_pool: 候选答案列表
        scene: 场景名称
        threshold: 保留比例阈值（0-1）

    Returns:
        过滤后的候选答案列表
    """
    print(f"  使用轻量级关键词匹配")

    # 获取场景的黑白名单
    scene_filter_cfg = SCENE_FILTER_CONFIG.get(scene, {})
    white_list = scene_filter_cfg.get('white_list', set())

    # 🔧 定义场景相关的关键词（扩展白名单）
    scene_keywords = {
        'person': {'person', 'man', 'woman', 'child', 'baby', 'kid', 'people', 'human', 'male', 'female', 'boy', 'girl', 'adult', 'standing', 'sitting', 'walking', 'running', 'face', 'hand', 'head', 'body'},
        'vehicle': {'car', 'bus', 'truck', 'train', 'bicycle', 'motorcycle', 'boat', 'airplane', 'vehicle', 'driving', 'parking', 'riding', 'wheel', 'door', 'window', 'seat'},
        'animal': {'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'animal', 'pet', 'wild', 'tail', 'fur', 'wing', 'leg'},
        'food': {'pizza', 'cake', 'apple', 'banana', 'orange', 'broccoli', 'carrot', 'sandwich', 'food', 'meal', 'dish', 'fruit', 'vegetable', 'eat', 'plate', 'bowl'},
        'furniture': {'chair', 'couch', 'bed', 'table', 'desk', 'shelf', 'cabinet', 'furniture', 'seat', 'sit', 'wood', 'metal'},
        'electronic': {'tv', 'laptop', 'computer', 'phone', 'keyboard', 'mouse', 'screen', 'monitor', 'electronic', 'device', 'digital'},
        'appliance': {'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'fan', 'microwave', 'appliance', 'kitchen', 'electric'},
        'sports': {'ball', 'kite', 'skateboard', 'frisbee', 'ski', 'snowboard', 'sports', 'game', 'play', 'field', 'court'},
        'kitchen': {'bottle', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'plate', 'kitchen', 'utensil', 'drink', 'wine'},
        'accessory': {'backpack', 'umbrella', 'handbag', 'bag', 'tie', 'suitcase', 'purse', 'accessory', 'carry', 'wear'},
        'indoor': {'book', 'clock', 'vase', 'scissors', 'teddy', 'bear', 'doll', 'indoor', 'inside', 'room', 'decoration'},
        'outdoor': {'plant', 'bench', 'traffic', 'light', 'hydrant', 'sign', 'meter', 'outdoor', 'outside', 'tree', 'flower', 'street'}
    }

    # 获取当前场景的关键词
    keywords = scene_keywords.get(scene, set())

    # 如果没有定义关键词，保留所有答案
    if not keywords:
        print(f"  ⚠️  未定义场景 '{scene}' 的关键词，保留所有答案")
        return global_pool

    # 🔧 过滤逻辑：保留与场景关键词相关的答案
    filtered = []
    for answer in global_pool:
        answer_words = set(answer.lower().split())

        # 如果答案包含场景关键词，保留
        if answer_words & keywords:
            filtered.append(answer)
        # 如果答案在白名单中，保留
        elif white_list and (answer_words & white_list):
            filtered.append(answer)
        # 颜色、数字、位置等通用词，保留（由频次过滤决定）
        elif any(word in answer_words for word in ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
                                                     'brown', 'black', 'white', 'gray', 'grey',
                                                     'one', 'two', 'three', 'four', 'five',
                                                     'left', 'right', 'top', 'bottom', 'center']):
            filtered.append(answer)

    print(f"  ✓ 语义过滤完成：{len(filtered)}/{len(global_pool)} 个答案保留")
    return filtered


def _filter_by_sentence_transformers(
    global_pool: List[str],
    scene: str,
    caption_file: Path,
    threshold: float
) -> List[str]:
    """
    完整语义过滤：基于Sentence-Transformers（需要额外依赖）

    使用预训练模型计算候选答案与场景caption的语义相似度

    Args:
        global_pool: 候选答案列表
        scene: 场景名称
        caption_file: Caption文件路径
        threshold: 相似度阈值

    Returns:
        过滤后的候选答案列表
    """
    print(f"  使用Sentence-Transformers模型")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # 加载模型（首次运行会下载）
        print(f"  加载Sentence-Transformers模型...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 加载caption
        with open(caption_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取该场景的caption
        scene_captions = []
        for ann in data.get('annotations', []):
            # 简化：使用所有caption（实际应根据image_id过滤）
            caption = ann.get('caption', '').strip()
            if caption:
                scene_captions.append(caption)

        if not scene_captions:
            print(f"  ⚠️  未找到caption，保留所有答案")
            return global_pool

        # 计算场景caption的语义向量（取平均）
        scene_embedding = model.encode(scene_captions[:100])  # 限制数量
        scene_embedding = np.mean(scene_embedding, axis=0)

        # 计算候选答案的语义向量
        answer_embeddings = model.encode(global_pool)

        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([scene_embedding], answer_embeddings)[0]

        # 过滤低相似度的答案
        filtered = [
            ans for ans, sim in zip(global_pool, similarities)
            if sim >= threshold
        ]

        print(f"  ✓ 语义过滤完成：{len(filtered)}/{len(global_pool)} 个答案保留（阈值={threshold}）")
        return filtered

    except ImportError:
        print(f"  ⚠️  缺少依赖库（sentence-transformers, sklearn），使用轻量级实现")
        return _filter_by_scene_keywords(global_pool, scene, threshold)

    except Exception as e:
        print(f"  ⚠️  Sentence-Transformers失败: {e}，使用轻量级实现")
        return _filter_by_scene_keywords(global_pool, scene, threshold)


def generate_scene_candidates(
    global_pool: List[str],
    freq_map: Dict[str, int],
    imgid2scene: Dict,
    annotations_dir: Path,
    min_freq: int = 2,
    max_candidates: int = 1600,
    use_semantic_filter: bool = True,
    semantic_use_lightweight: bool = True
) -> Dict[str, Dict]:
    """
    为每个场景生成独立候选集

    执行顺序（千问官方标准）：
    1. 读取全局干净候选池
    2. 按图像场景分组，得到该场景全部原始候选
    3. 第一步：场景黑白名单强过滤（只保留完全匹配白名单的标准单词）
    4. 对过滤后剩余干净答案，统计场景内出现频次
    5. 频次过滤 scene_min_frequency: 2
    6. 语义相似度软过滤
    7. 频次降序排序，截断 scene_max_candidates:1600
    8. 输出最终闭合集
    """
    print("\n" + "="*60)
    print("为每个场景生成独立候选集")
    print("="*60)

    scene_candidates = {}

    scenes = ['person', 'vehicle', 'animal', 'food', 'furniture',
              'electronic', 'appliance', 'sports', 'kitchen',
              'accessory', 'indoor', 'outdoor']

    for scene in scenes:
        print(f"\n处理场景: {scene}")
        print("-" * 40)

        # 🔧 步骤3：场景黑白名单强过滤（第一步，核心修复点）
        candidates = filter_by_objects(global_pool, scene)
        print(f"  步骤1（黑白名单强过滤）: {len(candidates)} 个答案")

        # 🔧 步骤4：统计场景内答案频次（在黑白名单过滤后）
        # 找出该场景的所有图像
        scene_images = set([
            img_id for img_id, scene_data in imgid2scene.items()
            if scene_data.get('primary_scene') == scene
        ])

        print(f"  场景 '{scene}' 包含 {len(scene_images)} 张图像")

        # 统计场景内答案频次
        scene_answer_counter = Counter()
        vqa_file = annotations_dir / "v2_Annotations_Train_mscoco.json"

        if vqa_file.exists():
            try:
                with open(vqa_file, 'r', encoding='utf-8') as f:
                    vqa_data = json.load(f)

                annotations = vqa_data.get('annotations', [])

                for ann in annotations:
                    # 检查是否属于该场景
                    image_id = ann.get('image_id')
                    if str(image_id) not in scene_images:
                        continue

                    # 统计答案（对VQA标注中的答案进行清洗后再统计）
                    if 'answers' in ann:
                        for ans_obj in ann['answers']:
                            raw_answer = ans_obj.get('answer', '').strip()

                            # 🔧 关键修复：对VQA原始答案应用全局清洗
                            cleaned_answer = global_standard_clean(raw_answer)
                            if not cleaned_answer:
                                continue  # 清洗后为空，跳过

                            # 🔧 只统计在黑白名单过滤后的候选答案
                            if cleaned_answer in candidates:
                                confidence = ans_obj.get('answer_confidence', 'yes')
                                weight = {'yes': 1.0, 'maybe': 0.5, 'no': 0.1}.get(confidence, 1.0)
                                scene_answer_counter[cleaned_answer] += weight

                    elif 'answer' in ann:
                        raw_answer = ann['answer'].strip()

                        # 🔧 关键修复：对VQA原始答案应用全局清洗
                        cleaned_answer = global_standard_clean(raw_answer)
                        if not cleaned_answer:
                            continue

                        # 🔧 只统计在黑白名单过滤后的候选答案
                        if cleaned_answer in candidates:
                            scene_answer_counter[cleaned_answer] += 1

                print(f"  ✓ 从VQA标注统计到 {len(scene_answer_counter)} 个不同答案")

            except Exception as e:
                print(f"  ⚠️  VQA标注加载失败: {e}")
                # 后备：使用全局频次
                for answer in candidates:
                    scene_answer_counter[answer] = freq_map.get(answer, 1)

        else:
            print(f"  ⚠️  VQA标注文件不存在，使用全局频次作为后备")
            for answer in candidates:
                scene_answer_counter[answer] = freq_map.get(answer, 1)

        # 🔧 步骤5：频次过滤（min_freq=2，过滤只出现一次的小众修饰短语）
        candidates = [
            ans for ans in candidates
            if scene_answer_counter.get(ans, 0) >= min_freq
        ]
        print(f"  步骤2（频次过滤 min_freq={min_freq}）: {len(candidates)} 个答案")

        # 🔧 步骤6：语义相似度软过滤（可选）
        if use_semantic_filter:
            caption_file = annotations_dir / "captions_train2014.json"
            if caption_file.exists():
                candidates = filter_by_semantic_similarity(
                    candidates, scene, caption_file, threshold=0.3,
                    use_lightweight=semantic_use_lightweight
                )
                print(f"  步骤3（语义过滤）: {len(candidates)} 个答案")

        # 🔧 步骤7：频次降序排序，截断
        if len(candidates) > max_candidates:
            # 按场景内频次排序，取Top-K
            candidates = sorted(
                candidates,
                key=lambda x: scene_answer_counter.get(x, 0),
                reverse=True
            )[:max_candidates]
            print(f"  步骤4（截断 max_candidates={max_candidates}）: {len(candidates)} 个答案")

        # 🔧 步骤8：输出最终闭合集
        scene_candidates[scene] = {
            'candidates': candidates,
            'count': len(candidates),
            'source': 'Qwen Pipeline Stage 3',
            'total_images': len(scene_images),
            'filters': {
                'object_filter': True,
                'frequency_filter': True,
                'semantic_filter': use_semantic_filter,
                'execution_order': 'correct'  # 标记执行顺序已修正
            }
        }

        print(f"  ✓ 最终候选集: {len(candidates)} 个答案")

    return scene_candidates


def main():
    parser = argparse.ArgumentParser(description="阶段3：分场景过滤生成局部小闭合集")

    parser.add_argument(
        '--global-pool',
        default='data/global_candidate_pool.json',
        help='全局候选池文件'
    )

    parser.add_argument(
        '--scene-mapping',
        default='data/imgid2scene.json',
        help='图像-场景映射文件'
    )

    parser.add_argument(
        '--annotations-dir',
        default='data/coco/annotations',
        help='COCO标注目录'
    )

    parser.add_argument(
        '--output',
        default='data/scene_candidates.json',
        help='输出文件路径'
    )

    parser.add_argument(
        '--min-freq',
        type=int,
        default=2,
        help='最低频次阈值（默认：2，过滤只出现一次的小众修饰短句）'
    )

    parser.add_argument(
        '--max-candidates',
        type=int,
        default=1600,
        help='单场景最大候选数（默认：1600，千问官方标准）'
    )

    parser.add_argument(
        '--use-semantic-filter',
        action='store_true',
        default=True,
        help='使用语义相似度过滤（默认：True，千问官方标准）'
    )

    parser.add_argument(
        '--semantic-use-lightweight',
        action='store_true',
        default=True,
        help='使用轻量级语义过滤（默认：True，基于关键词匹配，无需额外依赖）'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("阶段3：分场景过滤生成局部小闭合集")
    print("="*60)
    print("\n三层递进过滤规则：")
    print("  过滤1：物体词汇硬匹配（强约束第一层）")
    print("  过滤2：场景内VQA答案频次过滤（次约束第二层）")
    print("  过滤3：Caption语义相似度软过滤（可选提升精度）")

    global_pool_file = Path(args.global_pool)
    scene_mapping_file = Path(args.scene_mapping)
    annotations_dir = Path(args.annotations_dir)

    # 检查文件
    if not global_pool_file.exists():
        print(f"\n❌ 全局候选池不存在: {global_pool_file}")
        print("请先运行阶段1：python tools/candidate/stage1_build_global_pool.py")
        return

    if not scene_mapping_file.exists():
        print(f"\n❌ 场景映射不存在: {scene_mapping_file}")
        print("请先运行阶段2：python tools/candidate/stage2_scene_mapping.py")
        return

    # 加载数据
    global_pool, categorized_candidates, freq_map = load_global_pool(global_pool_file)
    imgid2scene = load_scene_mapping(scene_mapping_file)

    # 🔧 生成场景候选集（只对 object 候选集进行场景分组）
    # 基础答案（color/location/number/yesno）直接作为独立场景
    scene_candidates = {}

    # 添加基础场景
    if categorized_candidates.get('color'):
        scene_candidates['color'] = {
            'candidates': categorized_candidates['color'],
            'count': len(categorized_candidates['color']),
            'source': 'stage1_categorized'
        }

    if categorized_candidates.get('location'):
        scene_candidates['location'] = {
            'candidates': categorized_candidates['location'],
            'count': len(categorized_candidates['location']),
            'source': 'stage1_categorized'
        }

    if categorized_candidates.get('number'):
        scene_candidates['count'] = {
            'candidates': categorized_candidates['number'],
            'count': len(categorized_candidates['number']),
            'source': 'stage1_categorized'
        }

    if categorized_candidates.get('yesno'):
        scene_candidates['binary'] = {
            'candidates': categorized_candidates['yesno'],
            'count': len(categorized_candidates['yesno']),
            'source': 'stage1_categorized'
        }

    # 🔧 对 object 候选集进行场景分组和过滤
    object_pool = categorized_candidates.get('object', [])
    print(f"\n对 {len(object_pool)} 个 object 候选进行场景分组...")

    object_scene_candidates = generate_scene_candidates(
        object_pool,
        freq_map,
        imgid2scene,
        annotations_dir,
        min_freq=args.min_freq,
        max_candidates=args.max_candidates,
        use_semantic_filter=args.use_semantic_filter,
        semantic_use_lightweight=args.semantic_use_lightweight
    )

    # 合并到 scene_candidates
    for scene, data in object_scene_candidates.items():
        # 如果场景已存在（如 color/count），不覆盖
        if scene not in scene_candidates:
            scene_candidates[scene] = data

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'source': 'Qwen Official Pipeline Stage 3',
            'global_pool_file': str(global_pool_file),
            'scene_mapping_file': str(scene_mapping_file),
            'total_scenes': len(scene_candidates),
            'config': {
                'min_freq': args.min_freq,
                'max_candidates': args.max_candidates,
                'use_semantic_filter': args.use_semantic_filter
            }
        },
        'scenes': scene_candidates
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✓ 阶段3完成")
    print("="*60)
    print(f"\n输出文件: {output_path}")

    print(f"\n场景候选集统计：")
    total_candidates = 0
    for scene, data in scene_candidates.items():
        count = data['count']
        total_candidates += count
        print(f"  {scene:15s}: {count:4d} 个候选")

    print(f"\n总候选数: {total_candidates}")
    print(f"平均候选数: {total_candidates / len(scene_candidates):.0f}")


if __name__ == "__main__":
    main()