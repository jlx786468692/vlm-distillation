"""
生成分场景独立小闭合集
========================

从VQA训练集标注中，使用零样本分类路由，为每个问题类型生成独立的候选集。

核心流程：
1. 加载VQA训练集标注
2. 零样本分类问题类型（count/color/binary/other）
3. 分别统计每个场景的答案
4. 输出分场景独立小闭合集

输出格式：
{
    "count": {"candidates": ["zero", "one", ...], "count": 21, "source": "VQA train"},
    "color": {"candidates": ["red", "green", ...], "count": 24, "source": "VQA train"},
    "binary": {"candidates": ["yes", "no"], "count": 2, "source": "VQA train"},
    "other": {"candidates": [...], "count": 1873, "source": "VQA train"}
}

使用方式：
    python tools/candidate/generate_vqa_vocab.py
    python tools/candidate/generate_vqa_vocab.py --vqa-annotations data/vqa/v2_Annotations_Train_mscoco.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.candidate.question_type_classifier import QuestionTypeClassifier
except ImportError:
    print("⚠️  无法导入 QuestionTypeClassifier，使用基于关键词的分类")
    QuestionTypeClassifier = None


def load_vqa_train_annotations(annotations_path: Path) -> List[Dict]:
    """
    加载VQA训练集标注

    VQA v2标注格式：
    {
        "annotations": [
            {
                "question_id": 1,
                "question": "How many people are in the image?",
                "answers": [
                    {"answer": "net", "answer_confidence": "yes"},
                    ...
                ]
            },
            ...
        ]
    }

    或者分开的文件：
    - v2_Questions_Train_mscoco.json: {"questions": [...]}
    - v2_Annotations_Train_mscoco.json: {"annotations": [...]}
    """
    annotations = []

    # 尝试不同的文件格式
    possible_files = [
        annotations_path / "v2_Annotations_Train_mscoco.json",
        annotations_path / "vqa_train2014.json",
        annotations_path / "mscoco_train2014_annotations.json",
    ]

    # 也尝试问题文件（需要合并）
    questions_files = [
        annotations_path / "v2_Questions_Train_mscoco.json",
        annotations_path / "vqa_questions_train.json",
    ]

    # 首先尝试加载标注文件
    for ann_file in possible_files:
        if ann_file.exists():
            print(f"📖 加载VQA训练集标注: {ann_file}")
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'annotations' in data:
                    annotations = data['annotations']
                elif isinstance(data, list):
                    annotations = data

                print(f"  ✓ 加载 {len(annotations)} 条标注")
                return annotations

            except Exception as e:
                print(f"  ✗ 加载失败: {e}")

    # 如果没有找到标注文件，尝试从问题和答案文件合并
    print("⚠️  未找到完整标注文件，尝试加载问题和答案文件...")

    questions_data = None
    annotations_data = None

    for q_file in questions_files:
        if q_file.exists():
            print(f"📖 加载问题文件: {q_file}")
            with open(q_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            break

    for a_file in possible_files:
        if a_file.exists():
            print(f"📖 加载答案文件: {a_file}")
            with open(a_file, 'r', encoding='utf-8') as f:
                annotations_data = json.load(f)
            break

    if questions_data and annotations_data:
        # 合并问题和答案
        questions_dict = {q['question_id']: q for q in questions_data.get('questions', [])}

        merged_annotations = []
        for ann in annotations_data.get('annotations', []):
            question_id = ann.get('question_id')
            if question_id in questions_dict:
                merged_ann = ann.copy()
                merged_ann['question'] = questions_dict[question_id].get('question', '')
                merged_annotations.append(merged_ann)

        print(f"  ✓ 合并 {len(merged_annotations)} 条标注")
        return merged_annotations

    return annotations


def classify_questions_by_type(
    annotations: List[Dict],
    use_model: bool = True
) -> Dict[str, List[Dict]]:
    """
    对问题进行零样本分类，按类型分组

    Args:
        annotations: VQA标注列表
        use_model: 是否使用模型分类（否则使用关键词）

    Returns:
        分场景标注字典 {type: [annotations]}
    """
    print(f"\n📊 对 {len(annotations)} 个问题进行分类...")

    # 初始化分类器
    classifier = None
    if use_model and QuestionTypeClassifier:
        try:
            print("🔧 初始化零样本分类器...")
            classifier = QuestionTypeClassifier(model_name="models/bart-large-mnli")
            print("✓ 分类器初始化成功")
        except Exception as e:
            print(f"⚠️  分类器初始化失败: {e}")
            print("   使用基于关键词的分类")

    # 按类型分组
    grouped_annotations = defaultdict(list)

    for i, ann in enumerate(annotations):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(annotations)} ({i/len(annotations)*100:.1f}%)")

        question = ann.get('question', '')

        if not question:
            continue

        # 问题类型分类
        if classifier:
            result = classifier.classify(question)
            question_type = result['type']
        else:
            # 基于关键词的分类
            question_type = _classify_by_keywords(question)

        grouped_annotations[question_type].append(ann)

    print(f"\n✓ 分类完成：")
    for q_type, anns in grouped_annotations.items():
        print(f"  {q_type}: {len(anns)} 个问题")

    return grouped_annotations


def _classify_by_keywords(question: str) -> str:
    """基于关键词的问题分类"""
    question_lower = question.lower()

    if any(kw in question_lower for kw in ['how many', 'how much', 'count', 'number']):
        return 'count'
    elif any(kw in question_lower for kw in ['what color', 'color is', 'what colour']):
        return 'color'
    elif any(kw in question_lower for kw in ['is there', 'are there', 'is it', 'are they', 'does', 'do you', 'can you']):
        return 'binary'
    else:
        return 'other'


def generate_scene_specific_candidates(
    grouped_annotations: Dict[str, List[Dict]],
    min_frequency: int = 5,
    max_candidates: int = 100
) -> Dict[str, Dict]:
    """
    为每个场景生成独立候选集

    Args:
        grouped_annotations: 分场景标注
        min_frequency: 最低频率阈值
        max_candidates: 每个场景最大候选数

    Returns:
        分场景候选集
    """
    print(f"\n🔨 生成分场景独立小闭合集...")

    scene_candidates = {}

    for question_type, annotations in grouped_annotations.items():
        print(f"\n处理 {question_type} 场景 ({len(annotations)} 个问题)：")

        # 统计答案频率
        answer_counter = Counter()

        for ann in annotations:
            if 'answers' in ann:
                # VQA格式：10个标注者答案
                for ans_obj in ann['answers']:
                    answer = ans_obj.get('answer', '').lower().strip()
                    if answer:
                        # 使用置信度加权
                        confidence = ans_obj.get('answer_confidence', 'yes')
                        weight = {'yes': 1.0, 'maybe': 0.5, 'no': 0.1}.get(confidence, 1.0)
                        answer_counter[answer] += weight

            elif 'answer' in ann:
                answer = ann['answer'].lower().strip()
                if answer:
                    answer_counter[answer] += 1

        print(f"  统计到 {len(answer_counter)} 个不同答案")

        # 过滤并选择候选
        filtered = [
            (ans, freq) for ans, freq in answer_counter.items()
            if freq >= min_frequency
        ]

        # 按频率排序
        sorted_answers = sorted(filtered, key=lambda x: x[1], reverse=True)

        # 取Top-K
        candidates = [ans for ans, freq in sorted_answers[:max_candidates]]

        # 确保预定义答案
        predefined = _get_predefined_answers(question_type)
        for ans in predefined:
            if ans not in candidates:
                candidates.insert(0, ans)

        # 限制数量
        candidates = candidates[:max_candidates]

        # 计算覆盖率
        total = sum(answer_counter.values())
        covered = sum(freq for ans, freq in answer_counter.items() if ans in candidates)
        coverage = covered / total if total > 0 else 0

        scene_candidates[question_type] = {
            'candidates': candidates,
            'count': len(candidates),
            'source': 'VQA train',
            'total_questions': len(annotations),
            'unique_answers': len(answer_counter),
            'coverage': f"{coverage:.2%}",
            'metadata': {
                'min_frequency': min_frequency,
                'max_candidates': max_candidates,
                'top_answers': [(ans, freq) for ans, freq in sorted_answers[:10]]
            }
        }

        print(f"  ✓ 候选集大小: {len(candidates)}")
        print(f"  ✓ 覆盖率: {coverage:.2%}")
        print(f"  ✓ Top 5: {candidates[:5]}")

    return scene_candidates


def _get_predefined_answers(question_type: str) -> List[str]:
    """获取预定义答案（确保包含）"""
    if question_type == 'count':
        return [
            "zero", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
        ]
    elif question_type == 'color':
        return [
            "red", "green", "blue", "yellow", "orange", "purple",
            "pink", "brown", "black", "white", "gray", "grey"
        ]
    elif question_type == 'binary':
        return ["yes", "no"]
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="生成分场景独立小闭合集")

    parser.add_argument(
        '--vqa-annotations',
        default='data/coco/annotations',
        help='VQA训练集标注目录（默认：data/coco/annotations）'
    )

    parser.add_argument(
        '--output',
        default='data/scene_candidates.json',
        help='输出文件路径（默认：data/scene_candidates.json）'
    )

    parser.add_argument(
        '--min-frequency',
        type=int,
        default=5,
        help='最低频率阈值（默认：5）'
    )

    parser.add_argument(
        '--max-candidates',
        type=int,
        default=100,
        help='每个场景最大候选数（默认：100）'
    )

    parser.add_argument(
        '--use-model',
        action='store_true',
        default=True,
        help='使用零样本模型分类（默认：是）'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("生成分场景独立小闭合集")
    print("="*60)
    print("\n核心流程：")
    print("  1. 加载VQA训练集标注")
    print("  2. 零样本分类问题类型")
    print("  3. 为每个场景生成候选集")
    print("  4. 输出分场景独立小闭合集")

    # Step 1: 加载VQA训练集标注
    annotations_path = Path(args.vqa_annotations)
    annotations = load_vqa_train_annotations(annotations_path)

    if not annotations:
        print("\n❌ 未找到VQA训练集标注")
        print("\n请下载VQA v2训练集：")
        print("  1. 访问 https://visualqa.org/download.html")
        print("  2. 下载 Trainable data: v2_Annotations_Train_mscoco.zip")
        print("  3. 解压到 data/coco/annotations/")
        return

    # Step 2: 零样本分类
    grouped_annotations = classify_questions_by_type(
        annotations,
        use_model=args.use_model
    )

    # Step 3: 生成分场景候选集
    scene_candidates = generate_scene_specific_candidates(
        grouped_annotations,
        min_frequency=args.min_frequency,
        max_candidates=args.max_candidates
    )

    # Step 4: 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 添加元数据
    output_data = {
        'metadata': {
            'source': 'VQA v2 train',
            'total_questions': sum(len(anns) for anns in grouped_annotations.values()),
            'scenes': list(scene_candidates.keys()),
            'generated_at': str(Path(__file__).stat().st_mtime),
            'config': {
                'min_frequency': args.min_frequency,
                'max_candidates': args.max_candidates,
                'use_model': args.use_model
            }
        },
        'scenes': scene_candidates
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✓ 分场景独立小闭合集已生成")
    print("="*60)
    print(f"\n输出文件: {output_path}")
    print("\n场景统计：")
    for scene, data in scene_candidates.items():
        print(f"  {scene:10s}: {data['count']:3d} 个候选 (覆盖率: {data['coverage']})")

    print("\n💡 使用方式：")
    print("  from tools.candidate.candidate_closure import CandidateClosure")
    print("  closure = CandidateClosure(config)")
    print("  candidates = closure.get_candidates_for_question(question, primary_answer)")


if __name__ == "__main__":
    main()