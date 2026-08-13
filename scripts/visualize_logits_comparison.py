#!/usr/bin/env python
"""
Logits处理流程可视化（对比表格）
===============================

将原始logits、温度缩放、softmax概率、过滤后结果放在同一表格中对比
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from pathlib import Path
from src.models.teacher_model import TeacherModel
from src.utils.config import ConfigManager
from src.utils.vqa_token_filter import VQATokenFilter


def visualize_logits_comparison(image_path: str, question: str):
    """
    可视化logits处理流程（对比表格）

    Args:
        image_path: 图像路径
        question: 问题
    """
    print("\n" + "="*140)
    print("LOGITS处理流程对比表格".center(140))
    print("="*140)

    # 加载模型和过滤器
    config = ConfigManager()
    teacher = TeacherModel(config)
    token_filter = VQATokenFilter()  # 添加token过滤器

    temperature = config.get("distillation.soft_labels.temperature", 4)

    print(f"\n配置: 图像={Path(image_path).name} | 问题='{question}' | 温度T={temperature}")

    # 🔧 新增：显示白名单过滤策略说明
    print("\n" + "="*140)
    print("【处理策略说明】".center(140))
    print("="*140)
    print("\n优先级策略：")
    print("  1️⃣  【白名单过滤】（优先）：根据问题类型过滤无效token")
    print("      - 数字问题：只保留数字答案（one, two, 1, 2, ...）")
    print("      - 颜色问题：只保留颜色答案（green, blue, ...）")
    print("      - 二元问题：只保留 yes/no")
    print("  2️⃣  【Topk兜底】（备选）：当白名单过滤后为空时，保留top-k个token")
    print(f"      - top_k = {config.get('distillation.soft_labels.top_k_logits', 50)} (来自配置)")

    print("\n正在运行模型推理...")

    # 获取模型输出
    result = teacher.inference_vqa(
        image=image_path,
        question=question,
        return_logits=True,
        generate_cot=False
    )

    answer = result.get('answer', '')
    confidence = result.get('confidence', 0.0)

    print(f"模型输出: 答案='{answer}' | 置信度={confidence:.4f}")

    if 'logits' not in result:
        print("\n❌ 错误: 模型没有返回logits")
        return

    logits_data = result['logits']

    # 提取logits
    if 'top_k_indices' in logits_data and 'top_k_values' in logits_data:
        token_indices = logits_data['top_k_indices']
        token_logits = logits_data['top_k_values']

        # 取第一个token位置
        if token_indices.dim() == 1:
            first_indices = token_indices
            first_logits = token_logits
        elif token_indices.dim() == 2:
            first_indices = token_indices[0]
            first_logits = token_logits[0]
        else:
            first_indices = token_indices[0, 0]
            first_logits = token_logits[0, 0]

        # 解码token
        def decode_token(idx):
            try:
                token = teacher.tokenizer.decode([idx.item()])
                return token.strip().lower()
            except:
                return f"[ID:{idx}]"

        # 温度缩放
        scaled_logits = first_logits / temperature

        # ===== 🔧 修复：与soft_label_gen.py保持一致 =====
        # 原有问题：先对所有token计算softmax，然后过滤，导致概率分布错误
        # 正确逻辑：先判断token有效性，将无效token的logits设为-1e9，然后计算softmax
        # 这样无效token的概率接近0，有效token能获得更高的概率
        # ==============================================================
        print("\n正在进行白名单过滤...")
        valid_token_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)

        for i, idx in enumerate(first_indices):
            token = decode_token(idx)
            is_valid = token_filter.is_valid_token(token, question)
            is_primary = token == answer.lower()

            if is_valid or is_primary:
                valid_token_mask[i] = True

        num_valid = valid_token_mask.sum().item()
        print(f"白名单过滤结果: {num_valid}/{len(first_indices)} tokens有效")

        # ===== 🔧 Step 2: 应用mask到logits =====
        if num_valid > 0:
            scaled_logits_filtered = scaled_logits.clone()
            scaled_logits_filtered[~valid_token_mask] = -1e9
            # 对过滤后的logits计算softmax
            probs = F.softmax(scaled_logits_filtered, dim=-1)
        else:
            # Topk兜底（当白名单过滤后为空时）
            print("⚠️  白名单过滤后为空，使用top-k兜底策略")
            probs = F.softmax(scaled_logits, dim=-1)
            config_top_k = config.get("distillation.soft_labels.top_k_logits", 50)
            top_k = min(config_top_k, len(first_indices))
            top_k_indices = torch.topk(probs, top_k).indices
            valid_token_mask = torch.zeros_like(probs, dtype=torch.bool)
            valid_token_mask[top_k_indices] = True
            scaled_logits_filtered = scaled_logits.clone()
            scaled_logits_filtered[~valid_token_mask] = -1e9
            probs = F.softmax(scaled_logits_filtered, dim=-1)

        # 取top-50用于对比
        # 🔧 修复：使用配置文件中的top_k_logits值（作为兜底策略）
        config_top_k = config.get("distillation.soft_labels.top_k_logits", 50)
        top_k = min(config_top_k, len(first_indices))  # 不超过实际长度

        print(f"\n提取 top-{top_k} logits用于分析")
        print(f"  （配置: distillation.soft_labels.top_k_logits = {config_top_k}）")
        print(f"  （说明：白名单过滤优先，top-k仅作兜底）")

        top_indices = first_indices[:top_k]
        top_logits = first_logits[:top_k]
        top_scaled = scaled_logits[:top_k]
        top_probs = probs[:top_k]

        # 构建数据（过滤前）
        data_all = []
        for idx, raw_logit, scaled_logit, prob in zip(top_indices, top_logits, top_scaled, top_probs):
            token = decode_token(idx)
            # 🔧 注意：有效性已经在前面白名单过滤时判断过了
            # 这里再次判断是为了显示状态，但实际概率已经在softmax时考虑了过滤
            is_valid = token_filter.is_valid_token(token, question)
            is_primary = token == answer.lower()

            data_all.append({
                'token': token,
                'raw_logit': raw_logit.item(),
                'scaled_logit': scaled_logit.item(),
                'prob': prob.item(),
                'prob_pct': prob.item() * 100,
                'token_id': idx.item(),
                'is_valid': is_valid,
                'is_primary': is_primary
            })

        # ===== 🔧 新增：合并等价token（如 '1' 和 'one'） =====
        print("\n正在合并等价token...")
        merged_data = {}
        for d in data_all:
            # 获取标准形式
            canonical = token_filter.get_canonical_token(d['token'])

            if canonical not in merged_data:
                # 第一次遇到这个标准形式
                merged_data[canonical] = {
                    'token': canonical,
                    'raw_logit': d['raw_logit'],  # 保留最大的logit
                    'scaled_logit': d['scaled_logit'],
                    'prob': d['prob'],
                    'prob_pct': d['prob_pct'],
                    'token_id': d['token_id'],
                    'is_valid': d['is_valid'],
                    'is_primary': d['is_primary'],
                    'variants': [d['token']],  # 记录所有变体
                    'count': 1
                }
            else:
                # 合并概率
                merged_data[canonical]['prob'] += d['prob']
                merged_data[canonical]['prob_pct'] += d['prob_pct']
                merged_data[canonical]['variants'].append(d['token'])
                merged_data[canonical]['count'] += 1

                # 如果是主答案，标记
                if d['is_primary']:
                    merged_data[canonical]['is_primary'] = True

        # 转换为列表
        data_merged = list(merged_data.values())

        # 按概率排序
        data_merged.sort(key=lambda x: x['prob'], reverse=True)

        # 统计合并信息
        merge_stats = {}
        for d in data_merged:
            if d['count'] > 1:
                merge_stats[d['token']] = d['variants']

        if merge_stats:
            print(f"合并了 {len(merge_stats)} 组等价token:")
            for canonical, variants in merge_stats.items():
                print(f"  '{canonical}' <- {variants}")
        else:
            print("没有需要合并的等价token")

        # 使用合并后的数据替换原始数据
        data_all = data_merged

        # 🔧 说明：由于我们在softmax前已经对无效token设为了-1e9
        # 所以无效token的概率已经接近0，softmax后的概率总和应该接近1.0
        # 这里的归一化是为了进一步确保概率和为1.0（消除数值误差）

        # 计算有效token的归一化概率
        valid_data = [d for d in data_all if d['is_valid'] or d['is_primary']]
        total_valid_prob = sum(d['prob'] for d in valid_data)

        # 为所有数据计算归一化概率
        for d in data_all:
            if d in valid_data and total_valid_prob > 0:
                # 🔧 重新归一化：确保有效token的概率总和为1.0
                d['prob_normalized'] = d['prob'] / total_valid_prob
                d['prob_pct_normalized'] = d['prob_normalized'] * 100
            else:
                # 无效token的概率已经在softmax时设为接近0了
                d['prob_normalized'] = 0.0
                d['prob_pct_normalized'] = 0.0

        # 打印表格1：过滤前（Top-50所有token）
        print("\n" + "="*160)
        print(f"表格1: 过滤前 {len(data_all)} Tokens (已合并等价token，含噪音)")
        print("="*160)

        # 表头
        header1 = (
            f"{'#':<3} │ "
            f"{'Token':<10} │ "
            f"{'原始Logits':>12} │ "
            f"{'缩放后÷T':>12} │ "
            f"{'概率':>10} │ "
            f"{'百分比':>7} │ "
            f"{'合并来源':<20} │ "
            f"{'状态':<8}"
        )
        print(header1)
        print("─" * 160)

        # 数据行
        for i, item in enumerate(data_all, 1):
            # 标记状态
            if item['is_primary']:
                status = "★答案"
                marker = "★"
            elif item['is_valid']:
                status = "✓有效"
                marker = " "
            else:
                status = "✗噪音"
                marker = "✗"

            # 显示合并来源
            if item['count'] > 1:
                merge_info = f"+{item['count']-1}项"
                variants_str = f"{merge_info} ({', '.join(item['variants'][:3])})"
            else:
                variants_str = "-"

            row = (
                f"{i:>3} │ "
                f"{item['token']:<9} │ "
                f"{item['raw_logit']:>12.4f} │ "
                f"{item['scaled_logit']:>12.4f} │ "
                f"{item['prob']:>10.6f} │ "
                f"{item['prob_pct']:>6.2f}% │ "
                f"{variants_str:<20} │ "
                f"{status:<8}"
            )
            print(row)

        print("="*160)

        # 打印表格2：过滤后（只有有效token）
        print("\n" + "="*160)
        print(f"表格2: 过滤后 {len(valid_data)} 个有效Tokens (已归一化)")
        print("="*160)

        # 表头
        header2 = (
            f"{'#':<3} │ "
            f"{'Token':<10} │ "
            f"{'原始Logits':>12} │ "
            f"{'缩放后÷T':>12} │ "
            f"{'原概率':>10} │ "
            f"{'归一化概率':>12} │ "
            f"{'百分比':>7} │ "
            f"{'合并来源':<15}"
        )
        print(header2)
        print("─" * 160)

        # 数据行
        cumulative = 0.0
        for i, item in enumerate(valid_data, 1):
            cumulative += item['prob_normalized']

            # 标记答案
            marker = "★" if item['is_primary'] else " "

            # 显示合并来源
            if item['count'] > 1:
                merge_info = f"+{item['count']-1}项"
            else:
                merge_info = "-"

            row = (
                f"{i:>3} │ "
                f"{item['token']:<9} │ "
                f"{item['raw_logit']:>12.4f} │ "
                f"{item['scaled_logit']:>12.4f} │ "
                f"{item['prob']:>10.6f} │ "
                f"{item['prob_normalized']:>12.6f} │ "
                f"{item['prob_pct_normalized']:>6.2f}% │ "
                f"{merge_info:<15}"
            )
            print(row)

        print("="*160)

        # 统计信息
        print("\n过滤统计:")
        print("-" * 160)

        noise_count = len([d for d in data_all if not d['is_valid'] and not d['is_primary']])
        print(f"总Token数: {len(data_all)}")
        print(f"有效Token: {len(valid_data)} (答案 + {len(valid_data)-1}个有效答案)")
        print(f"噪音Token: {noise_count} (被过滤)")
        print(f"过滤率:   {noise_count/len(data_all)*100:.1f}%")

        # 概率对比
        probs_all = torch.tensor([d['prob'] for d in data_all])
        probs_valid = torch.tensor([d['prob'] for d in valid_data])
        probs_normalized = torch.tensor([d['prob_normalized'] for d in valid_data])

        print(f"\n概率分布对比:")
        print(f"  所有Top-{top_k}概率和: {probs_all.sum():.4f} ({probs_all.sum()*100:.2f}%)")
        print(f"  有效Token概率和:     {probs_valid.sum():.4f} ({probs_valid.sum()*100:.2f}%)")
        print(f"  归一化后概率和:      {probs_normalized.sum():.4f} ({probs_normalized.sum()*100:.2f}%)")

        # 主答案概率对比
        primary_data = next((d for d in data_all if d['is_primary']), None)
        if primary_data:
            print(f"\n主答案 '{answer}' 概率变化:")
            print(f"  原始概率:  {primary_data['prob']:.6f} ({primary_data['prob_pct']:.2f}%)")
            print(f"  归一化后:  {primary_data['prob_normalized']:.6f} ({primary_data['prob_pct_normalized']:.2f}%)")
            increase = primary_data['prob_normalized'] - primary_data['prob']
            print(f"  提升:      {increase:.6f} ({increase*100:.2f}%)")

        # 熵对比（修复警告：使用clone().detach()）
        entropy_before = -sum(p.clone().detach() * torch.log(p.clone().detach() + 1e-10) for p in probs_all if p > 0)
        entropy_after = -sum(p.clone().detach() * torch.log(p.clone().detach() + 1e-10) for p in probs_normalized if p > 0)

        print(f"\n信息熵对比:")
        print(f"  过滤前: {entropy_before:.4f} (分布更分散)")
        print(f"  过滤后: {entropy_after:.4f} (分布更集中)")
        print(f"  变化:   {entropy_after - entropy_before:.4f} (减少{abs(entropy_after - entropy_before):.2f})")

        # 噪音token列表
        noise_tokens = [d['token'] for d in data_all if not d['is_valid'] and not d['is_primary']]
        print(f"\n被过滤的噪音Token ({len(noise_tokens)}个):")

        # 按类型分组显示
        color_tokens = [t for t in noise_tokens if t in token_filter.color_answers]
        binary_tokens = [t for t in noise_tokens if t in token_filter.binary_answers]
        other_noise = [t for t in noise_tokens if t not in color_tokens and t not in binary_tokens]

        if color_tokens:
            print(f"  颜色答案: {', '.join(color_tokens)}")
        if binary_tokens:
            print(f"  二元答案: {', '.join(binary_tokens)}")
        if other_noise:
            print(f"  其他噪音: {', '.join(other_noise[:15])}")
            if len(other_noise) > 15:
                print(f"           ... 还有 {len(other_noise)-15} 个")

    else:
        print("\n❌ 错误: logits格式不正确")


def main():
    """主函数"""
    # 测试参数（可修改）
    image_path = "data/coco/val2014/COCO_val2014_000000051314.jpg"
    question = "What is the color of the water?"
    ground_truth = "green"

    if Path(image_path).exists():
        visualize_logits_comparison(
            image_path=image_path,
            question=question
        )
    else:
        print(f"⚠️  图像不存在: {image_path}")

    # ====================
    # 测试用例2：是否问题
    # ====================
    print("\n" + "="*160)
    print("测试用例2: 是否问题".center(160))
    print("="*160)

    image_path = "data/coco/val2014/COCO_val2014_000000545000.jpg"
    question = "Is the fire hydrant red?"
    ground_truth = "yes"

    if Path(image_path).exists():
        visualize_logits_comparison(
            image_path=image_path,
            question=question
        )
    else:
        print(f"⚠️  图像不存在: {image_path}")

    # ====================
    # 测试用例3：选择问题
    # ====================
    print("\n" + "="*160)
    print("测试用例3: 选择问题".center(160))
    print("="*160)

    image_path = "data/coco/val2014/COCO_val2014_000000051314.jpg"
    question = "Is it day or night?"
    ground_truth = "day"
    candidate_pool = ["day", "night"]

    if Path(image_path).exists():
        visualize_logits_comparison(
            image_path=image_path,
            question=question
        )
    else:
        print(f"⚠️  图像不存在: {image_path}")


if __name__ == "__main__":
    main()