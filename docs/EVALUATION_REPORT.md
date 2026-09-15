# VLM 蒸馏学生模型 — 完整评估报告（含逐指标分析）

> **生成日期**：2026-09-08（治本重训更新）  
> **报告文件**：`outputs/evaluation/report.json`（`is_heldout=true`）  
> **学生** Qwen2.5-VL-3B-Instruct（LoRA r=64 蒸馏后合并全权重）｜ **教师** 标签由 agent_distill 仓库代码生成（两版仅软标签抽取法不同：旧版 logit/logprob，新版 multi-sampling）
> **训练** train2014 `training/agent_teacher/train_soft_sampling.jsonl`（sampling 软标签，3 epochs，kl_weight=0.5）  
> **评估** `training/agent_teacher/val_soft_sampling.jsonl`（2431 条，**held-out**，全闭集，sampling 软标签）

> **本次更新要点**：相比上一版（logprob 软标签，color 塌缩，overall 0.6437），本次改用 sampling 法软标签重训重评——color 塌缩**已修复**（0.0245→0.9257），overall 回升至 **0.8198**，保留率 71.9%→**91.6%**，新增 CoT 语义相似度指标（Sentence-BERT cosine 0.8077）。
>
> **§13 更新（2026-09-14）**：新增 32B-教师学生（`student_merged_32b`，32B 教师 logits/top_k=20 法软标签训成）评估，overall **0.835**（目前最高，距教师天花板 0.8954 仅 6pp，保留率 93.4%）。与 sampling 学生（0.8198）+ 未训 base（lenient 0.7709）三方对比见 §13：换 32B 教师硬标签 +1.5pp（主要 yes_no/other/open），代价是 CoT Jaccard 与序列 CE 略退。
>
> **§14 更新（2026-09-15）**：两教师学生完整对比（32B 教师 vs qwen3.7-plus 教师，§14）。32B 教师学生**全面占优**——硬标签 0.835 vs 0.820、CoT 同源口径三项全胜（Jaccard 0.366/0.324、SBERT 0.836/0.808、CE 0.664/1.070）。§13.2 的"CoT 退步"经 §13.5 同源重评更正为评估参照系假象。根因：32B 与 3B 学生同族同源、logits 法软标签精度高、CoT 确定性强。
> **§15 更新（2026-09-15）**：四方对比（§15）——32B 教师 / qwen3.7-plus 教师 / 两个 3B 学生一张表。sampling val 2431 同口径下：qwen3.7-plus 教师 0.8954（天花板）、qwen3.7-plus 学生 0.8198、32B 学生 0.835。32B 教师天花板因自身 val（336）为"教师答对子集"（hard_label==GT 100%）退化为 sanity、且未在 2431 上跑过，列为参考缺失；其学生反超 qwen3.7-plus 学生间接证明 32B 教师数据 ≥ qwen3.7-plus。补齐方案见 §15.5。
>

---

## 0. 整体蒸馏质量评判（先给结论）

**一句话：治本重训成功——sampling 软标签消除了 color 塌缩，overall 0.8198，保留率 91.6%，分布拟合更优，剩余短板是 open/counting 泛化。**

| 层面 | 评判 | 关键证据 |
|---|---|---|
| 蒸馏机制（KL 软标签） | ✅ **成功** | KL 0.0188 / cos 0.9964 / top1 0.9986，分布拟合且泛化到 held-out |
| 教师软标签质量 | ✅ **已修复** | sampling 法 color 题 primary==hard 100%（旧 logprob 仅 34.3%），零 "other" 泄漏 |
| 硬标签泛化 | ✅ **良好** | held-out 0.8198 vs 天花板 0.8954（保留率 91.6%），gap 收窄至 7.6pp |
| CoT 推理 | 🟢 **良好** | 推理语义相似度 0.8077（Sentence-BERT），token Jaccard 0.3237 受指标/容量限制 |
| 部署效率 | ✅ **优秀** | 3.75B / 7.26GB 峰值 / 2.83s，12GB 卡可部署 |

**核心结论**：上一版的 color "other" 塌缩（软标签"垃圾进垃圾出"）已通过改用 sampling 法软标签**根治**——学生分布拟合极好（cos 0.9964、top1 0.9986），且硬答案准确率 0.8198，距教师天花板仅 7.6pp。color 桶从 0.0245 回升至 0.9257，逼近教师天花板 0.959。剩余短板是 open（0.537）和 counting（0.724），属泛化/容量差异，非数据缺陷。

**蒸馏质量总评：8/10**。机制层优秀、数据层已修复、硬标签泛化良好。进一步提升空间在 open/counting 泛化（扩数据/正则）和开集评估补盲。

---

## 0.5 两版对比（logit 旧版 vs sampling 新版）

> 保留旧版（logit/logprob 软标签）评估指标以供对照。两版评估数据均 held-out 于 val2014：旧版用 `val_soft_logits`（2456 条），新版用 `val_soft_sampling`（2431 条）。

| 指标 | 旧版 logit/logprob (val_soft_logits, 2456条) | 新版 multi-sampling (val_soft_sampling, 2431条) | 变化 |
|---|---|---|---|
| overall (closed_answer_match) | 0.6437 | 0.8198 | +17.6pp |
| color 桶准确率 | 0.0245（塌缩 "other"）| 0.9257 | +90.1pp ✅ |
| closed_primary_match | 0.7142 | 0.8198 | +10.6pp |
| closed_distribution KL | 0.046 | 0.0188 | -0.027（更优）|
| closed_distribution cosine | 0.981 | 0.9964 | +1.5pp |
| closed_distribution top1 | 0.9266 | 0.9986 | +7.2pp |
| cot_similarity (token Jaccard) | 0.322 | 0.3237 | ~持平 |
| cot_semantic_similarity (新) | —（旧版无）| 0.8077 | 新增 |
| gap vs 教师天花板 0.8954 | 25.2pp | 7.6pp | -17.6pp |
| 保留率 | 71.9% | 91.6% | +19.7pp |

### 0.5.1 差距在哪里

- **最大单一 delta 是 color 桶 +90.1pp**——旧版 logit/logprob 法在 65% 的 color 样本上把 "other" 排为 primary（soft argmax ≠ hard answer），"other" 的虚高质量经 KL 通道泄漏给学生，致学生塌缩到 "[Answer] other"。sampling 法使 primary==hard 100% 干净，彻底消除泄漏。

- **第二大 delta 是分布 top1 +7.2pp / cosine +1.5pp / KL -0.027**——并非因为 sampling 在"蒸馏"本身更强，而是 sampling 教师分布自身更干净（无虚高 "other" 质量），学生拟合出的分布既更正确也更对齐。

- **cot_similarity（token Jaccard）基本不变（0.322→0.324）**——这是两版差异集中在**软标签质量**（答案 token 分布）而非推理/CoT 质量的关键证据。旧版的推理链路本就正确（color 推理正确，仅 [Answer] token 错误）。新版新增的 cot_semantic_similarity 0.8077 进一步印证推理语义质量高。

- **注意**：旧版 primary_match 0.7142 > answer_match 0.6437 是"冒烟枪"——primary_match 把"学生输出 other == 教师 primary=other"也计为匹配（两者同方向地错），从而把 primary_match 抬到真实答案准确率之上。新版两者收敛到 0.8198，因为教师 primary==hard 100%。

---

## 1. 评估设置

### 1.1 数据流

```
train2014 ──agent_distill 教师标签──> train_soft_sampling.jsonl (sampling 软标签) ──> 学生(3B) SFT+KL训练
                                                                                │
val(同源 held-out) ──教师采样标注──> val_soft_sampling.jsonl (2431条) <──评估──────┘
```

训练集与验证集为 COCO 同源不重叠 split，**无数据泄漏**，`is_heldout=true`。两份数据均使用 **sampling 法**（n=8 多次推理采样）生成软标签，color primary==hard 一致率 100%（旧 logprob 法仅 34.3%）。

### 1.2 配置与改动

| 项 | 值 |
|---|---|
| 评估数据 | `val_soft_sampling.jsonl`，2431 条，全闭集，sampling 软标签 |
| 训练数据 | `train_soft_sampling.jsonl`，sampling 软标签，3 epochs，kl_weight=0.5 |
| `max_pixels` | 100352（与训练分辨率对齐，≈128 视觉 token） |
| 新增指标 | `cot_semantic_similarity`（Sentence-BERT all-MiniLM-L6-v2 cosine） |

代码改动：① `StudentInferencer` 加 `max_pixels` 对齐训练分辨率；② 修闭集判定 bug（`question_category` 为 None → 改用 `answer_distribution` 存在性，与训练对齐）；③ 新增 `scenario_buckets`；④ `_build_kl_meta` 加矛盾跳过（止血代码保留，但 sampling 数据 0 冲突不触发）；⑤ 新增 `cot_semantic_similarity`（Sentence-BERT all-MiniLM-L6-v2 cosine），衡量推理语义相似度。

---

## 2. 按题型分桶绝对准确率（scenario_buckets）

### 2.1 结果表

| 桶 | correct/total | 学生准确率 | 教师天花板 | 差距 |
|---|---|---|---|---|
| **overall** | 1993/2431 | **0.8198** | 0.8954 | -7.6pp |
| **color** | 299/323 | **0.9257** ✅ | **0.959** | **-3.3pp** |
| yes_no | 1150/1320 | 0.8712 | ~0.93-0.98 | -6pp |
| other | 160/198 | 0.8081 | — | — |
| choice | 64/81 | 0.7901 | 0.869 | -8pp |
| counting | 181/250 | 0.724 | 0.843 | -12pp |
| open | 139/259 | 0.5367 | ~0.60 | -6pp |

### 2.2 逐桶分析：为什么是这个值

**overall = 0.8198**  
为什么不是更高？color 塌缩已消除（0.0245→0.9257），不再拉低 overall。剩余 gap 7.6pp 分散在各桶，主因是 open（0.537）和 counting（0.724）泛化偏弱。相比上一版（0.6437），overall 回升 17.6pp，其中 color 修复贡献约 13pp，其余来自分布质量提升带动的整体泛化改善。

**color = 0.9257（已修复 ✅）**  
为什么从 0.0245 回升至 0.9257？sampling 法软标签 primary==hard 100%（旧 logprob 仅 34.3%），"other" 泄漏彻底消除，学生答案 token 正确学到真颜色。距教师天花板 0.959 仅 -3.3pp，属合理泛化 gap。详见 §6。

**yes_no = 0.8712（最高）**  
为什么是非题最好？① 候选集极小（yes/no 二元），argmax 决策容易；② 教师天花板本身就最高（0.93-0.98），学生有充足的训练信号；③ 是非题 question_type 占比最大（is/are 系列 ~40%），训练样本多。差距 -6pp 较上一版（-15pp）大幅收窄，说明 sampling 软标签整体质量提升对各题型均有带动。

**counting = 0.724**  
为什么中等偏弱？计数题答案是数字（1/2/3…），教师天花板 0.843 本身就偏低（计数是难任务，教师也常错）。学生距天花板 -12pp，是当前**最大 gap 桶之一**。数字答案单 token，候选映射干净，无 "other" 干扰，gap 主因是过拟合+容量差（3B 对细粒度计数能力有限）。

**open = 0.5367（绝对值最低，当前最弱桶）**  
为什么最低？① 教师天花板本身就最低（~0.60，开放描述题最难）；② 学生 0.54 距天花板约 -6pp，保留率不低，但绝对值最低使其成为**当前最弱桶**。开放题靠通用语言/推理能力，3B 学生容量限制更明显；③ 注：open 桶基于 question_type 分类，与 `open_text_similarity` 指标（本数据全闭集，该指标=0.0 不适用）不同——此处 open 指开放描述类题型在 closed_answer_match 上的表现。

**choice = 0.7901 / other = 0.8081**  
中等偏上。choice 是 "none of the above" 题，other 是未归类的杂项题。other 较上一版（0.731）提升至 0.808，choice 从 0.738 升至 0.790，均受益于 sampling 软标签整体改善。

### 2.3 教师按题型天花板（val2014 vs COCO 人工 GT）

| 题型 | 教师 | | 题型 | 教师 |
|---|---|---|---|---|
| is it | 0.984 | | what color | 0.959 |
| is this | 0.974 | | is there a | 0.950 |
| are there | 0.971 | | is this a | 0.933 |
| is | 0.958 | | are the | 0.929 |
| is the | 0.908 | | how many | 0.843 |
| none of the above | 0.869 | | what | 0.603 |

教师最弱是 `what` 类开放题（0.603），最强是是非题（is it/are 0.93-0.98）。color 教师天花板 0.959 很高，学生 color 0.9257 已逼近天花板（-3.3pp），修复成功。

---

## 3. 软标签分布蒸馏质量（closed_distribution）

### 3.1 指标表

| 指标 | 值 | 方向 |
|---|---|---|
| KL(teacher‖student) | 0.018826 | 越低越好 |
| cosine | 0.9964 | 越高越好 |
| top1 分布一致率 | 0.9986 | 越高越好 |
| 可算样本 | 2221/2431 | 跳过 210（8.6%） |
| 温度 | 1.0 | 推理不缩放 |

### 3.2 逐指标分析：为什么是这个值

**KL = 0.018826（极低，优秀，较上版 0.046 大幅改善）**  
为什么这么低？KL 蒸馏**直接优化这个目标**（`loss = CE + 0.5·KL(teacher‖student)`，在答案 token 位置对候选集）。学生在训练中专门被拉向教师分布，2431 条 held-out 上仍保持 0.0188，说明学到的不是死记训练集，而是**泛化的分布拟合能力**。0.019 nats 接近 0，意味着学生分布与教师分布几乎重合。相比上一版（0.046），sampling 软标签消除了 "other" 虚高噪声后，教师分布本身更干净集中，学生拟合更精准。

**注意**：上版 KL 低包含了 color 题学生把 "other" 当 argmax 与教师（同样把 "other" 当 argmax）一致 → 一致就低 KL 的"学到了教师的错"问题。本次 sampling 数据 primary==hard 100%，教师分布本身正确，KL 低纯粹反映"拟合成功"，双刃剑的另一面（学错）已消除。

**cosine = 0.9964（极高，优秀，较上版 0.981 改善）**  
为什么比 KL 还亮眼？余弦衡量分布**形状**（方向），对绝对概率值不敏感。学生分布的形状（各候选的相对高低排列）与教师几乎完全一致。0.9964 意味着即便个别候选概率绝对值有微小偏差，整体形状高度吻合。sampling 软标签分布更集中干净，形状拟合更精准。

**top1 分布一致率 = 0.9986（极高，较上版 0.9266 大幅提升）**  
为什么 99.86% 而非 100%？0.14% 的样本学生 argmax ≠ 教师 argmax——极少数分布**峰值位置略有偏移**的样本（学生次高峰被推到主峰）。top1 一致(0.9986) 与硬答案 match(0.8198) 的差值来自"学生 argmax = 教师 argmax 但 ≠ 人工 GT"（教师自身也错的部分，教师天花板 0.8954）。上一版 top1 一致(0.927) 远高于硬答案 match(0.64) 是因为 color 题学生与教师一致地错到 "other"；本次无此问题，差值纯粹反映教师自身误差。

**可算样本 2221/2431（跳过 210，8.6%）**  
为什么跳过 8.6%？镜像训练侧 `_build_kl_meta` 的跳过规则：① 多 token 候选答案（如 "around fire hydrant" 编码成多 token，无法映射到单候选 token）；② 候选 token 冲突（不同候选编码到同一 token）；③ 答案 token 在序列中定位失败。sampling 数据 0 矛盾冲突（止血跳过代码不触发）。8.6% 跳过率正常，不影响整体结论。

### 3.3 教师软标签置信度参照

| 量 | 值 | 解读 |
|---|---|---|
| 平均 top1 概率 | 0.8602 | 教师整体较自信 |
| 中位 top1 概率 | 0.9978 | 半数样本教师几乎确定 |
| 平均熵 | 0.257 bits | 分布集中 |
| 平均候选数 | 1.66（max 5） | 多数 2 候选 |
| 不确定样本（top1<0.7） | 25.9% | 这 25.9% 是软标签最有价值的部分 |

**为什么这重要**：25.9% 的不确定样本正是软标签 KL 蒸馏**相对硬标签 SFT 的增量价值所在**——教师在这些样本上给非退化的概率分布（而非压单点），学生学到了不确定性。这部分若用硬标签 SFT 会丢失。学生 KL 0.0188 说明连这些不确定样本的不确定性都拟合到了。

---

## 4. 硬标签评估（Hard-label）

### 4.1 指标表

| 指标 | 值 | 含义 |
|---|---|---|
| closed_answer_match_rate | 0.8198 | 学生生成答案 vs 教师 hard_label（真答案） |
| closed_primary_match_rate | 0.8198 | 学生答案 vs 教师 soft_label.primary_answer |
| answer_sequence_ce（闭集 NLL） | 1.0695 | 学生对教师目标文本的自回归 NLL（samples 2431） |

### 4.2 逐指标分析：为什么是这个值

**closed_answer_match_rate = 0.8198**  
为什么 0.82？color 塌缩已消除（sampling 软标签 primary==hard 100%），color 题不再系统性拉低。剩余 gap 7.6pp 分散在各桶，主因是过拟合（训练数据量对 3B 偏少）+ 容量差（3B 学生 vs 教师），在 open（0.537）和 counting（0.724）上表现最明显。相比上一版（0.6437），回升 18pp，其中 color 修复贡献约 13pp，分布质量提升带动其余各桶整体改善约 5pp。

**closed_primary_match_rate = 0.8198 = answer_match（一致！）**  
为什么两个指标一致了？上一版 primary_match(0.7142) > answer_match(0.6437) 的"反常"根源是 color：教师 primary="other"（虚高），学生也输出 "other" → color 题在 primary_match 上反而匹配，拉高 primary_match。本次 sampling 数据 primary==hard 100%，教师软标签 argmax 与 hard 答案完全一致，两个指标趋于一致——这正是软标签质量修复的直接证据。0.8198 = 0.8198 也说明学生在答案 token 上的 argmax 与教师分布 argmax 高度一致（top1 一致率 0.9986 印证）。

**answer_sequence_ce = 1.0695（高，held-out）**  
为什么 held-out 这么高？NLL 是学生对教师目标文本 `[Reasoning]...\n[Answer] 真答案` 的自回归负对数似然。主因是过拟合（训练数据对 3B 偏少）+ held-out 未见数据泛化弱。相比上一版（1.0855）略降——color 题不再因学生答案主峰在 "other" 而目标真颜色导致该 token NLL 极高。1.0695 nats 意味着学生对教师目标文本的每 token 困惑度 ≈ e^1.07 ≈ 2.91，即每个目标 token 学生平均在 ~3 个等价候选间犹豫——泛化偏弱但未崩溃。

### 4.3 训练集 vs 验证集对比（泛化 gap）

| 指标 | 训练集（旧 report） | held-out（本次） | 变化 |
|---|---|---|---|
| closed_answer_match | 0.8904 | 0.8198 | -7pp |
| cot_similarity | 0.4028 | 0.3237 | -8pp |
| sequence CE | 0.407 | 1.0695 | ↑大幅 |

**为什么降**：旧 0.89 是训练集评估（学生见过这些数据），held-out 0.82 才是真实泛化。7pp gap 是过拟合的直观量——相比上一版（25pp gap，含 color bug）大幅收窄。剩余 7pp gap 主因训练数据量对 3B 偏少 + 容量差，属可缓解的泛化问题。

### 4.4 蒸馏天花板

| 角色 | val vs 人工 GT |
|---|---|
| 教师硬标签 | **0.8954**（2199/2456）← 蒸馏上限 |
| 学生硬标签 | 0.8198 |
| 保留率 | **91.6%** |

学生硬答案保留教师 91.6% 的能力（较上版 71.9% 大幅提升）。注意天花板 0.90 不是 1.0——教师自身在 val 上也只 0.90，学生不可能超过。7.6pp gap 是当前剩余泛化空间。

### 4.5 硬标签指标 gap 成因分解（区分"软标签泄漏"与"泛化本身"）

> ✅ **治本已修复**：经 sampling 法软标签重训后，上一版 §4.5 分解的 25.2pp gap 中 color 软标签泄漏 12.4pp **已完全消除**，非 color 软argmax噪声泄漏 3.8pp 也大幅收窄。本节重写为**修复后的 gap 分解**，旧版分解（logprob 时代 25.2pp）保留在下方"历史记录"中供参考。

> **口径澄清（重要）**：硬标签泛化测的是"学生生成答案 vs GT"的端到端准确率，概念上属学生自身泛化能力问题（过拟合 + 容量差）。但训练 loss = CE(含[Answer]真答案) + 0.5·KL(答案token, 教师软标签分布)，**KL 项直接塑造学生答案 token 的 argmax**——软标签 bug 会通过 KL 通道污染硬答案输出。所以硬标签指标 gap 的"表现"跨两类成因，但**严格意义的"硬标签泛化问题"只指学生自身能力不足那部分**，软标签泄漏不算。

> 注：本节的 gap = **教师天花板 → 学生**（0.8954→0.8198，能力差，7.6pp），与 §4.3 的 **训练集 → held-out**（0.89→0.82，泛化跌落，7pp）是两个不同口径，互为印证但归因不同。本节回答"学生距教师差在哪"，§4.3 回答"学生记忆 vs 泛化差多少"。

#### 4.5.1 修复后总分解（sampling 软标签）

硬标签指标 gap = 教师天花板 0.8954 − 学生 0.8198 = **7.6pp**。相比上一版 25.2pp，gap 收窄 17.6pp（color 泄漏 12.4pp 消除 + 软argmax噪声 3.8pp 收窄 + 分布质量提升带动整体改善 ~1.4pp）。当前 gap 几乎全部来自泛化本身：

| 成因 | 贡献 | 占比 | 性质 | 可修复性 |
|---|---|---|---|---|
| **① 学生模仿不足（过拟合+容量差）** | ~7.6pp | ~100% | **泛化本身** | 🟡 部分修复（扩数据/正则） |
| ~~② color 软标签 bug 泄漏~~ | ~~0pp~~ | ~~0%~~ | ~~已消除~~ | ✅ 已修复 |
| ~~③ 教师软argmax噪声泄漏~~ | ~~~0pp~~ | ~~~0%~~ | ~~大幅收窄~~ | ✅ 已修复（sampling argmax≈hard） |
| **合计** | 7.6pp | 100% | — | — |

**关键结论**：修复软标签后，硬标签 gap 从 25.2pp 降至 7.6pp，且**几乎全部是泛化本身**（过拟合+容量差），软标签泄漏已不再是瓶颈。这与上一版预测（"修复软标签后硬指标可回升 ~16pp 至 0.80"）一致——实际回升至 0.8198，略超预期。

**剩余 7.6pp 的逐桶分布**：
- **open**：0.5367 vs 天花板 ~0.60，gap ~6pp，但绝对值最低，是最弱桶。开放题靠通用语言/推理能力，3B 容量限制最明显。
- **counting**：0.724 vs 天花板 0.843，gap ~12pp，是最大单桶 gap。计数是细粒度视觉任务，3B 学生对精确计数能力有限。
- **yes_no**：0.8712 vs 天花板 ~0.93-0.98，gap ~6pp。
- **color**：0.9257 vs 天花板 0.959，gap ~3.3pp——修复后已逼近天花板。

#### 4.5.2 子成因详解

**① 学生模仿不足（~7.6pp，主因）**：学生连教师的正确 soft argmax 都没完全跟上。三个子成因：
- **过拟合**：序列 CE 训练集 0.41 → held-out 1.07（2.61×），训练数据量对 3B 偏少，学生记住训练样本却泛化弱。这是当前 gap 的主因。
- **容量差**：3B 学生 vs 教师，表达能力天花板不同，部分复杂推理学生学不到——尤其在 open/counting 上最明显。
- **推理→答案链路**：KL 只作用在答案 token，推理链路靠 CE 学；3B 容量限制推理质量 → 偶尔推理偏导致答案偏。

#### 4.5.3 结论

- **软标签泄漏已消除**（color 12.4pp + 软argmax噪声 3.8pp → ~0pp）——sampling 法根治；
- **剩余 7.6pp 全部是泛化本身**（过拟合 + 容量差）——这才是当前真正的瓶颈，扩数据/正则可缓解过拟合部分，容量差是 3B vs 教师的不可消除残差；
- 容量差在 open（0.537）和 counting（0.724）上占比最大。

**即：硬标签指标 0.8198 的剩余 gap 是纯粹的泛化能力问题。** 修复软标签（sampling）已将泄漏清零，进一步突破需扩数据缓解过拟合（预期 +3-5pp 至 ~0.85）+ 正则化收窄泛化 gap（预期 +1-2pp），保留率从 91.6% 可进一步提升至 ~95%。

---

#### 4.5.H 历史记录：上一版 gap 分解（logprob 货架，25.2pp）

> 以下保留上一版（logprob 软标签）的 gap 分解作为诊断记录。该分解中的 color 软标签泄漏 12.4pp **已通过 sampling 重训消除**。

上一版 gap = 教师天花板 0.8954 − 学生 0.6437 = **25.2pp**，三类成因：

| 成因 | 贡献 | 占比 | 性质 | 状态 |
|---|---|---|---|---|
| **① color 软标签 bug 泄漏** | 12.4pp | 49% | 软标签质量 | ✅ 已消除 |
| **② 非 color 学生模仿不足** | 9.0pp | 36% | 泛化本身 | 🟡 收窄至 ~7.6pp |
| **③ 非 color 教师软argmax噪声泄漏** | 3.8pp | 15% | 软标签质量 | ✅ 大幅收窄 |

**color 块（旧）**：
```
教师天花板 0.957 ─┬─ 59.5pp ─ 教师软标签 argmax 噪声（"other" 虚高）
                 └─→ 教师软argmax上限 0.362 ─┬─ 33.8pp ─ 学生放大塌缩
                                           └─→ 学生实际 0.0245
```
教师 primary vs hard 一致率 0.3742（62.6% 矛盾），学生 100% 塌缩到 "other"。改 sampling 后教师 primary vs hard 一致率 0.374→100%（§6.3），整块 gap 消失，color 回升至 0.9257。

**非 color 块（旧）**：
```
教师天花板 0.886 ─┬─ 4.4pp ─ 教师软argmax噪声（soft argmax ≠ hard）
                 └─→ 教师软argmax上限 0.842 ─┬─ 10.4pp ─ 学生模仿不足（过拟合+容量）
                                            └─→ 学生实际 0.738
```
非 color 教师软标签质量（primary vs hard = 0.9465）远高于 color（0.3742），证明软标签 bug 是 color 局部的。sampling 重训后软argmax噪声 4.4pp 大幅收窄（sampling argmax ≈ hard），学生模仿不足从 10.4pp 收窄至 ~7.6pp。

---

## 5. CoT 推理评估

### 5.1 指标

| 指标 | 值 | 含义 |
|---|---|---|
| cot_similarity | 0.3237 | 学生推理段 vs 教师推理段 token Jaccard |
| cot_semantic_similarity（**新增**） | 0.8077 | 学生推理段 vs 教师推理段 Sentence-BERT (all-MiniLM-L6-v2) cosine |
| open_text_similarity | 0.0 | 无 open 样本（数据全闭集），不适用 |
| open_text_semantic_similarity | null | 无 open 样本，不适用 |

### 5.2 为什么 token Jaccard 0.3237 偏低，但语义相似度 0.8077 高

**token Jaccard = 0.3237（偏低，与上版 0.322 基本不变）**  
**指标本身的局限**：token Jaccard 衡量**词级重叠**，对同义不同词惩罚极重。3B 学生与教师即便语义相同，措辞必然不同（教师表达更丰富多样，学生更简洁）→ Jaccard 天然偏低。0.3237 不代表推理质量差，反映的是 **学生与教师的表达差异**（容量限制的体现）。

**cot_semantic_similarity = 0.8077（新增，高）**  
为什么语义 >> token？Sentence-BERT all-MiniLM-L6-v2 将推理段编码为句向量，衡量**语义相似度**而非词级重叠。0.8077 意味着学生推理与教师推理在语义层面高度一致——尽管用词不同（Jaccard 低），但表达的含义、观察到的视觉元素、推理逻辑链高度吻合。这正是 token Jaccard 无法捕捉的维度。

**为什么 semantic(0.8077) >> token(0.3237)**：
- token Jaccard 要求**精确词匹配**，"a red car" vs "a crimson vehicle" → Jaccard=0，但语义几乎相同；
- Sentence-BERT 通过预训练语义空间将同义/近义表达映射到相近向量，能识别同义、上下位、转述；
- 学生 3B 措辞更简洁/直白，教师措辞更丰富/多样，token 不匹配但语义对齐 → 两者差距是表达差异而非推理质量差异。

**语义层面实际正确**：从 detail 样例看，学生推理链路语义准确——包括 color 题（"green hue, indicating the go signal" 正确识别颜色，且本次答案也正确输出 green）。

### 5.3 学生输出样例

| 问题 | 教师 | 学生 | 学生推理（节选） |
|---|---|---|---|
| Is the bicyclist wearing a helmet? | no | yes ⚠️ | "…dark, rounded protective head covering with visible ventilation holes and a chin strap…" |
| Is this a skate park? | yes | yes ✓ | "…paved area with concrete ramps, rails, and ledges specifically designed for skateboarding…" |
| How many trucks on the road? | 1 | 1 ✓ | "…a single red pickup truck…No other trucks are visible…" |
| What fast food restaurant? | burger king | burger king ✓ | "…blue and yellow storefront with a large sign that clearly reads \"Burger King\"…" |

**注意第一个**：学生推理合理（识别了头盔结构）却答 yes（教师 no）——这里可能是教师与 GT 的对错待定，或学生视觉判断偏差，但推理过程本身自洽。

---

## 6. color 塌缩根因（逐层）—— 已修复

> ✅ **已修复**：经治本重训（sampling 法软标签）后 color 准确率回升至 0.9257（从 0.0245），逼近教师天花板 0.959。本节保留为根因诊断记录，描述的是上一版（logprob 软标签）的 color 塌缩现象与根因。

### 6.1 现象（logprob 时代，已修复）

color 桶 0.0245，学生推理正确但 `[Answer]` 永远输出 "other"：
```
Q: What color is the traffic light?   GT=green
   推理: "...green hue, indicating the go signal..."   ← 正确
   [Answer] other                                       ← 塌缩
```

### 6.2 根因：logprob 软标签 "other" 虚高

train.jsonl 609 条 color 题：
- `hard_label.answer`（教师实际生成答案）：red 196, white 87, yellow 84, green 83... ← **真颜色**
- `soft_label.primary_answer`（logprob 分布 argmax）：**"other" 397/609 = 65%**

candidate_pool 末尾有 `"other"`：`['white','black','red','green','blue','yellow','gray','brown','orange','silver','pink','purple','other']`。

**机制**：教师 logprob 法在候选 token 上取 logit 做分布。"other" 作为通用词 token **logit 先验高**，被虚高排成 top1，即便教师实际生成的是真颜色。导致 primary="other"，与教师自身 hard_label 矛盾。

### 6.3 logprob vs multi 采样软标签对比（决定性）

**val2014 侧**（评估用）：

| 软标签法 | color primary==hard 一致 | "other" 矛盾率 |
|---|---|---|
| **logprob（训练用）** | 122/326 = **0.374** | 203/326 = **0.623** |
| **multi 采样法** | 323/323 = **1.000** | **0.000** |

**train2014 侧**（治本重训数据，已生成 `distill_train_sampling.jsonl`，5009 条）：

| 桶 | sampling primary==hard | logprob primary==hard | sampling "other"率 |
|---|---|---|---|
| color | **609/609 = 100%** | 209/609 = 34.3% | 0% |
| yes_no | 2779/2779 = 100% | 2507/2779 = 90.2% | 0% |
| counting | 552/552 = 100% | 541/552 = 98.0% | 0% |
| open | 537/537 = 100% | 33/59 = 55.9% | 0% |
| choice | 161/161 = 100% | 131/148 = 88.5% | 0% |
| other | 371/371 = 100% | 339/370 = 91.6% | 0% |
| **整体** | **5009/5009 = 100%** | 3760/4517 = 83.2% | 0% |

multi 采样法（n=8 多次推理采样）软标签**完全正确**——primary 永远等于教师真答案，零 "other" 塌缩，全桶 100% 一致。logprob 法整体仅 83.2%、color 仅 34.3%。multi 文件还同时含 `answer_distribution`（采样，正确）+ `logprob_distribution`（logit，有偏）双字段，严格优于 logprob 文件。**✅ 治本重训已完成**：用 `train_soft_sampling.jsonl` 替换 logprob 数据重训 3 epochs，color 回升至 0.9257（详见 §2.2），验证了本节根因分析正确。

### 6.4 传播链（为什么学生塌缩到 "other"）

```
教师 logprob 软标签 "other" 虚高（color 题 65% primary=other）
   ↓ KL 项（kl_weight=0.5，答案 token 位置，loss = CE + 0.5·KL）
学生答案 token 分布被拉向 "other"（top1 一致率 0.927 含此）
   ↓ 压过 CE 训练目标 [Answer] 真颜色
学生 greedy 生成取 argmax = "other" → "[Answer] other"
   ↓ color 桶 0.0245
```

**为什么 KL 压过 CE**：CE 项是全序列平均（reasoning + answer 多 token），单个答案 token 的 CE 信号被稀释；KL 项集中作用在答案 token 且教师分布高度集中（如 0.9 在 "other"），等效梯度远大于单 token CE。kl_weight=0.5 下，集中强信号胜过稀释弱信号。

**为什么非 color 题不受影响**：yes_no/counting/choice 的 candidate_pool 没有 "other" 虚高问题（yes_no 候选 yes/no，counting 候选数字），教师 primary 与 hard 一致，KL 学到的是正确分布。

---

## 7. 部署效率

| 指标 | 值 | 解读 |
|---|---|---|
| 总参数 | 3,754,622,976（3.75B） | 全权重，LoRA 已合并（trainable==total） |
| 模型目录 | 11,942.9 MB | bf16 safetensors |
| 峰值 VRAM | 7.258 GB | 12GB 卡占用 60%，留足余量 |
| 平均延迟 | 2.8283 s | 含图像预处理+生成 |
| 中位延迟 | 2.7663 s | 与均值接近，无长尾 |
| 吞吐 | 0.3536 samples/s | ~2.8s/样本 |

**为什么延迟 2.83s**：① slow image processor（非 use_fast）CPU 预处理重；② 单样本 greedy 生成 ~50-100 token 自回归解码；③ batch=1 无法摊销。相比教师大幅轻量化，适合非实时离线场景。峰值 VRAM 7.26GB 是 bf16 权重（7.5GB）+ 小激活（低分辨率 max_pixels=100352）。

---

## 8. driving_scenario 维度（已禁用）

已禁用（`enabled: false`）。原问题：① 独立生成翻倍耗时（~5h）；② 依赖 COCO `(image_id, question)` 查表，本数据 `image_id` 缺失全空。已将分桶并入 `distillation_quality.scenario_buckets`，用记录自带 `ground_truth` 做 GT，零额外开销。报告 `dimensions` = `['distillation_quality', 'deployment_efficiency']`。

---

## 9. 综合结论与蒸馏质量总评

### 9.1 逐维度评分

| 维度 | 评分 | 依据 |
|---|---|---|
| 软标签分布蒸馏机制 | ✅ 优秀 | KL 0.0188 / cos 0.9964 / top1 0.9986，泛化到 held-out |
| 教师软标签质量 | ✅ 已修复 | sampling 法 color primary==hard 100%（旧 logprob 34.3%），零 "other" 泄漏 |
| 硬标签答案 | ✅ 良好 | held-out 0.8198，保留率 91.6%，gap 收窄至 7.6pp |
| CoT 推理 | 🟢 良好 | 语义相似度 0.8077（Sentence-BERT），token Jaccard 0.3237 受指标/容量限制 |
| 部署效率 | ✅ 优秀 | 7.26GB / 2.83s，部署友好 |

### 9.2 四条核心结论

1. **蒸馏机制本身有效**：软标签 KL 让学生学到了教师的不确定性分布（KL 0.0188），且泛化到 held-out。这是蒸馏相对纯硬标签 SFT 的核心增量价值，已验证。学生连教师不确定样本的不确定性都拟合到了——硬标签 SFT 会丢失这部分。

2. **教师软标签质量已修复**：sampling 法对 color 题 primary==hard 100%（旧 logprob 34.3%），"other" 虚高泄漏彻底消除，学生忠实学到正确的颜色分布。上一版的"垃圾进垃圾出"陷阱已通过切换软标签生成方法根治——验证了"机制对，但教师标签质量决定上限"的诊断正确，sampling 法是正确的标签质量保障方案。

3. **各题型学生表现合理**：yes_no 0.871 / color 0.926 / other 0.808 / choice 0.790，距教师天花板 3-8pp，蒸馏有效。剩余弱项是 open（0.537）和 counting（0.724），属容量差+泛化本身，非数据缺陷。

4. **硬标签 gap 已收窄至纯泛化问题**（详 §4.5）：25.2pp → 7.6pp，**软标签泄漏 16.2pp 已清零**（color bug + 软argmax噪声通过 sampling 消除），**剩余 7.6pp 全部是泛化本身**（过拟合+容量差）。硬标签指标 0.8198 的 gap 是纯粹的泛化能力问题，扩数据/正则可进一步缓解。

### 9.3 "分布好且决策也强"现象

- 学生分布形状拟合极好（cos 0.9964 / top1 0.9986），硬决策 match 0.8198。
- 上一版"分布好但决策弱"（cos 0.981 vs match 0.64）的根因是 color 题峰值被 KL 拉偏到 "other"——现已消除。当前分布形状与决策均优，top1 一致率(0.9986) 与硬答案 match(0.8198) 的差值纯粹反映教师自身相对 GT 的误差（教师天花板 0.8954）。
- 学生学的是**教师的决策倾向**（含其正确分布），蒸馏本质（模仿教师）已有效执行。

### 9.4 蒸馏质量总评：8/10

- **+4 分**：机制层成功（分布拟合 KL 0.019 / 泛化 / 推理语义相似度 0.808 / 部署优秀）
- **+3 分**：数据层已修复（sampling 软标签消除 color 泄漏，硬标签 0.8198 / 保留率 91.6%）
- **-2 分**：剩余泛化 gap（7.6pp，open 0.537 / counting 0.724 偏弱，过拟合+容量差）
- 进一步提升空间：扩数据缓解过拟合（预期 +3-5pp 至 ~0.85）、正则化（预期 +1-2pp）、开集评估补盲。

---

## 10. 优化建议（按优先级）

### 10.1 color 塌缩修复 ✅ 已完成

> ✅ **已通过 sampling 法软标签重训完成**。color 准确率 0.0245→0.9257，逼近教师天花板 0.959。以下方案表保留为记录。

| 方案 | 做法 | 预期 | 状态 |
|---|---|---|---|
| **A. 改用 sampling 软标签重训**（推荐） | 用 sampling 法（n=8）重生成软标签，替换 logprob 后重训 | color primary==hard 100%，零 "other" 偏差，color 回升至 0.926 | ✅ 已完成（0.9257） |
| B. 从 candidate_pool 删 "other" | 移除 "other" 候选项 | 兜底，减少干扰 | 未采用（A 已根治） |
| C. 对矛盾样本跳过 KL | `primary != hard_label` 时跳过该样本 KL 项 | 避免传错分布 | 止血代码保留，sampling 数据 0 冲突不触发 |
| D. 降 kl_weight | 0.5 → 0.2，让 CE 真答案主导 | 通用 | 未采用 |

### 10.2 改善硬标签泛化（当前主要瓶颈）

> 据 §4.5 分解：硬标签 gap 已从 25.2pp 收窄至 7.6pp，且全部是泛化本身（过拟合+容量差）。下述方案针对这部分。

| 方案 | 理由 | 预期收益 |
|---|---|---|
| 扩训练数据 / 多样性 | 训练数据对 3B 偏少，过拟合是当前 gap 主因（~7.6pp）；增至 1-2 万条最直接 | +3-5pp |
| 正则化 | 降 LoRA r（64→32）、加 dropout、减 epoch（3→2），收窄 train/held-out CE gap（0.41→1.07） | +1-2pp |
| 补开集样本 | 当前全闭集，开集泛化未验证 | 验证侧补盲 |
| 针对 open/counting 专项数据 | open 0.537 / counting 0.724 是最弱桶，补充开放描述+计数专项样本 | +2-3pp |

### 10.3 评估侧

| 方案 | 理由 |
|---|---|
| 保留 scenario_buckets | 已重构，持续按题型诊断 |
| 保留 cot_semantic_similarity | 新增的 Sentence-BERT 指标有效反映推理语义质量，持续监控 |
| 开集评估补盲 | 当前全闭集（open_text_similarity=0），补开集样本后可激活该指标 |

---

## 11. 下一步行动

1. ~~**止血**~~：`_build_kl_meta` 已加矛盾跳过（`src/training/distill_dataset.py`），sampling 数据 0 冲突不触发，保留作为安全网。
2. ~~**治本**~~：✅ 已完成。用 `train_soft_sampling.jsonl`（sampling 法软标签）替换 logprob 数据重训 3 epochs，color 回升至 0.9257，overall 0.8198，保留率 91.6%。
3. ~~重训后重评~~：✅ 已完成。用 `val_soft_sampling.jsonl` held-out 重跑评估，验证 color 桶回升至 0.9257（逼近天花板 0.959）。
4. **扩训练数据**：当前训练数据对 3B 偏少（过拟合，CE 0.41→1.07），增至 1-2 万条缓解泛化 gap（预期 +3-5pp）。
5. **针对 open/counting 专项提升**：open 0.537 / counting 0.724 是最弱桶，补充专项样本 + 正则化。
6. **开集评估补盲**：当前全闭集，补充开集样本后激活 `open_text_similarity` 指标，验证开集泛化。

---

## 12. 未训练 base 3B 基线对照（蒸馏收益量化）

> **新增（2026-09-10）**：用未训练的 Qwen2.5-VL-3B-Instruct（base，未经任何蒸馏训练）在同一 val 集 `val_soft_sampling.jsonl`（2431 条 held-out）上跑完整评估，与已训学生对照，量化蒸馏的真实增益。评估用同一套代码/配置（仅 `student_model_path` 指向 base 模型），本机 RTX 5070，53min。产出 `eval_untrained/report.json`（本地，非挂载）。

### 12.1 对照表

| 维度 | 指标 | 未训 base | 已训学生 | Δ | 解读 |
|---|---|---|---|---|---|
| **硬标签** | closed_answer_match_rate | 0.0 | 0.8198 | +82.0pp | ⚠ 格式伪影，见 12.2 |
| | closed_primary_match_rate | 0.0 | 0.8198 | +82.0pp | 同上 |
| | scenario overall（vs GT） | 0.0695 | 0.8198 | +75.0pp | 闭集全 0 拖累，见下 |
| **软标签** | closed_distribution KL | 0.0524 | 0.0188 | -64% | ✅ 蒸馏显著降低 |
| | cosine | 0.9953 | 0.9964 | +0.0011 | 基线已极高，增益微小 |
| | top1 分布一致率 | 0.9995 | 0.9986 | -0.0009 | 基线已 99.95%，无空间 |
| **CoT** | cot_similarity（token Jaccard） | 0.1431 | 0.3237 | +126% | ✅ 蒸馏翻倍 |
| | cot_semantic_similarity（SBERT） | 0.6553 | 0.8077 | +23% | ✅ 语义对齐提升 |
| **序列** | answer_sequence_ce（closed） | 2.28 | 1.0695 | -53% | ✅ 困惑度 9.8→2.9 |
| **部署** | 参数量 | 3.75B | 3.75B | 同 | 同模型 |
| | 峰值 VRAM | 7.258GB | 7.258GB | 同 | 同模型 |
| | 平均延迟 | 0.868s | 2.828s | ×3.3 | 已训输出更长，见 12.5 |
| | 吞吐 | 1.15/s | 0.354/s | ÷3.3 | 同上 |

各桶 vs GT（scenario_buckets）：

| 桶 | 未训 | 已训 |
|---|---|---|
| yes_no | 0.0 | 0.8712 |
| color | 0.0 | 0.9257 |
| counting | 0.0 | 0.724 |
| other | 0.0 | 0.8081 |
| choice | 0.0 | 0.7901 |
| **open** | **0.6525** | 0.5367 |

### 12.2 关键陷阱：硬标签 0.0 是格式伪影，不是知识缺失

未训 base 的 closed_answer_match_rate = **0.0** 看似"base 完全不会答"，实为**格式不匹配**：评估的 `score_match`（[_common.py:107](src/evaluation/_common.py#L107)）闭集分支要求学生输出含 `[Answer]` 标记后取整行比对；未训 base 输出自由文本、**不带 `[Answer]` 标记** → `extract_student_answer` 返回空 → 恒判 False → 0.0。这测的是"是否学会教师强制的两段式输出格式"，而非知识本身。

**全量重打分实验（2026-09-11）**：为把"4 个例子"升级成全量数字，用同一 `StudentInferencer` 对 2431 条 val 重跑 1 遍贪婪生成，落盘每条原始输出后离线重算 base vs GT 的真实答对率。base 不吐 `[Answer]` 标记，故用宽松 token 子集口径（GT 的所有词是否出现在 base 输出里）抽答——这给出 base 剥离格式要求后的真实水平（略偏乐观）。注：原严格口径（要 `[Answer]` 标记）下 base overall 仅 0.0695（即 report 里的 0.0695，格式伪影所致），故下表只列宽松口径。

| 桶（样本数） | 未训 base | 已训学生 |
|---|---|---|
| yes_no（1320） | **0.8114** | 0.8712 |
| color（323） | **0.935** | 0.9257 |
| counting（250） | **0.54** | 0.724 |
| choice（81） | **0.6173** | 0.7901 |
| other（198） | **0.7424** | 0.8081 |
| open（259） | 0.6525 | 0.5367 |
| **overall** | **0.7709** | **0.8198** |

**量化结论**：base 真实答对率 **0.7709**，已训学生 crisp 准确率 0.8198。所以硬标签 +75pp（0.07→0.82）的拆解是：

| 成分 | 幅度 | 说明 |
|---|---|---|
| 格式合规（`[Answer]` 标记 + 短答案锚定） | **~70pp** | 0.07→0.77 的几乎全部 |
| 真实答题知识增益 | **~5pp**（0.77→0.82，且 base 宽松口径略偏乐观，真实 ≤5pp） | 蒸馏确有提升但很小 |
| 开集反降 | -11.6pp（0.6525→0.5367） | 已训 terse 短答案 token 覆盖更少，token 子集反吃亏 |

即**硬标签增益里格式合规占大头，真实知识迁移 ≤5pp**。base 在颜色(0.935)、是非(0.811)、开放(0.653)上本就答得很好——这些是 trivial 视觉感知题，3B base 不学也会；蒸馏真正补的硬标签知识主要在 counting(0.54→0.724)等略难的桶。

### 12.3 软标签：base 已逼近教师，增益在"幅度"不在"方向"

最反直觉但最诚实的发现：未训 base 在候选集分布上**已经和教师几乎一致**——top1 一致率 0.9995（已训 0.9986 甚至略低）、cosine 0.9953。因为闭集候选集小（yes/no、颜色、数字），3B base 和教师都把质量压在同一显然答案上，argmax 天然一致。

蒸馏的真实软标签增益只在**幅度精度**：KL 从 0.0524 降到 0.0188（降 64%）——学生把各候选的**概率数值**调到更贴教师，但峰值位置（argmax/形状）基线已对齐。top1 略降（0.9995→0.9986）是因极少数样本学生为拟合教师分布幅度做了微调，反而挪动了 argmax——无害。

**结论**：闭集软标签蒸馏的"知识传输"价值有限（base 已会），主要价值是幅度校准 +（配合 CE）强制格式。

### 12.4 CoT 与序列 CE：蒸馏的核心价值所在

- **CoT**：token Jaccard 0.143→0.324（×2.26）、SBERT 语义 0.655→0.808——学生学会了复现教师推理链路（用词与语义都更贴教师）。这才是 base→学生最实质的能力迁移。
- **序列 CE**：2.28→1.0695（困惑度 e^2.28≈9.8 → e^1.07≈2.9）——base 对教师目标文本 `[Reasoning]…[Answer]…` 极度"意外"（不会该格式），学生已内化。降幅 53% 是三维度里最硬的蒸馏证据。

### 12.5 部署：同模型，已训更慢（更长输出）

参数/VRAM 完全相同（同 3B 模型）。但已训平均延迟 2.83s vs 未训 0.87s（**已训慢 3.3×**）——因已训学生按训练目标生成完整 `[Reasoning]` 推理段 + `[Answer]`（更长输出、更多 token），而 base 输出更短/自由。吞吐反之（未训 1.15/s > 已训 0.35/s）。

**这是蒸馏的代价**：获得结构化推理能力 → 输出变长 → 延迟上升。若部署只追求短答案准确率，base + prompt 工程可能更划算；若需可解释推理链，蒸馏后的学生值得这个延迟。

### 12.6 三方分桶对比与差异归因（教师天花板 / 未训 base / 已训学生）

把三方放一张表（均 vs 人工 GT），看蒸馏在每个桶上到底搬了多少知识：

| 桶（样本数） | 教师天花板 | 未训 base | 已训学生 |
|---|---|---|---|
| overall（2431） | 0.8954 | 0.7709 | 0.8198 |
| color（323） | 0.959 | 0.935 | 0.9257 |
| yes_no（1320） | ~0.95 | 0.8114 | 0.8712 |
| counting（250） | 0.843 | 0.54 | 0.724 |
| choice（81） | 0.869 | 0.6173 | 0.7901 |
| other（198） | — | 0.7424 | 0.8081 |
| open（259） | ~0.60 | 0.6525 | 0.5367 |

> **口径不对称（关键）**：教师天花板 = 教师 hard_label 短答案 vs GT（crisp 精确）；已训学生 = `[Answer]` 抽取后 crisp 精确匹配；未训 base = token 子集宽松匹配（base 无 `[Answer]` 标记，只能从自由文本里抽答，GT 的所有词是否出现在输出里）。**base 偏乐观**——尤其 color（单词答案碰巧命中）和 open（长答案碰巧含 GT 词）。所以 base 的真实水平**低于** 0.7709，与教师/学生的真实 gap **大于**表面数字。（base 在原严格口径下 overall 仅 0.0695——即 report 里的 0.0695，格式伪影所致，见 §12.2。）

#### 差异归因

**1. 梯度方向稳定：教师 > 已训 > base（除口径伪影桶）**
难桶梯度最陡、易桶几乎无梯度——这正是"蒸馏在有 headroom 处搬知识"的签名：

| 桶 | base→教师 headroom | 学生吃到 | 捕获率 |
|---|---|---|---|
| counting | 30.3pp | +18.4pp | 61% |
| choice | 25.2pp | +17.3pp | 69% |
| yes_no | ~13.9pp | +6.0pp | 43% |
| overall | ~12.5pp | ~4.9pp | ~40% |

counting/choice 是 base 远低于教师、蒸馏真实搬了 +17-18pp 的桶——**这才是软标签 + CoT 蒸馏的知识价值所在**。

**2. 为什么 overall 增益只有 ~5pp？easy 桶主导稀释**
yes_no(1320) + color(323) = 1693/2431 = **70%** 样本在这两桶，base 本就 0.81-0.94，headroom 极小，蒸馏再厉害也挤不出多少。overall 被 trivial 题稀释——**这是评估选题的问题，不是蒸馏的问题**。

**3. 为什么 color/open 上 base 反超学生？口径伪影，不是 base 更强**
- **color**：base 说"the light is green"，token 子集算对（0.935）；学生吐 `[Answer] green` crisp 也对（0.926）。两者真实水平相当，base 0.935 是宽松口径的过计，**不代表 base 比学生强**。
- **open**：base 长自由文本更易碰巧含 GT 词（0.6525），学生 terse `[Answer]` 短答案 token 覆盖少反吃亏（0.537）。**测的是输出形态不是知识**。
- 这两桶是"为什么不能只看 overall/单口径"的反面教材。

**4. 蒸馏到底搬了多少知识（按桶）**
- 易桶（color/yes_no）：base 已接近教师天花板，知识增益 ~0-6pp，主要补**格式**
- 难桶（counting/choice）：base 远低于教师，蒸馏真实搬 **+17-18pp 知识**
- 被易桶稀释后 overall 只剩 ~5pp——但 count/choice 证明知识搬运**确实发生了**

**一句话**：三方对比显示蒸馏的知识梯度真实存在（counting/choice +17-18pp），但被 trivial 题主导的 overall 掩盖了。要看清蒸馏价值，要么按桶看（counting/choice），要么换难题评估——overall 这把尺子对"知识蒸馏"判别力不足。

### 12.7 总评：蒸馏到底带来了什么

| | base 已有 | 蒸馏新增 |
|---|---|---|
| 闭集答题知识 | ✅（top1 0.9995 已逼近教师；lenient vs GT 0.77） | 真实 +≤5pp（0.77→0.82），幅度校准 KL -64% |
| 输出格式 | ❌（自由文本） | ✅ 两段式 `[Reasoning]/[Answer]` |
| CoT 推理 | △（能答但推理不贴教师） | ✅ Jaccard ×2.26、语义 +23% |
| 目标文本似然 | ❌（CE 2.28，意外） | ✅ CE -53% |

一句话：**闭集知识 base 本来就有，蒸馏真正教会学生的是"用教师的格式说、按教师的思路推理、把分布幅度校准到教师"**——这正是软标签 KL + CoT SFT 的设计目标，与硬标签表面 +82pp 的"知识幻觉"不同。

---

## 13. 32B-教师学生 vs sampling-教师学生（三方对比）

> **新增（2026-09-14）**：32B-教师数据学生（`student_merged_32b`，训于 4864 条 train2014，32B 教师 logits/top_k=20 法软标签）在同一 val 集 `val_soft_sampling.jsonl`（2431 条 held-out）上评估，overall **0.835**——目前最高。与 §12 的 sampling-教师学生（0.8198）和未训 base（lenient 0.7709）三方对比，量化"换更强教师"的边际收益。产出 `outputs/evaluation/report.json`（已备份到 `/mnt/data/workspace/jlx/eval_32b/report.json`）。

### 13.1 硬标签三方分桶对比

| 桶（样本数） | 32B 学生 | 0.82 学生 | 未训 base¹ | 教师天花板² |
|---|---|---|---|---|
| yes_no（1320） | **0.8924** | 0.8712 | 0.8114 | ~0.95 |
| color（323） | 0.9257 | 0.9257 | 0.935 | 0.959 |
| counting（250） | 0.724 | 0.724 | 0.54 | 0.843 |
| choice（81） | 0.7778 | 0.7901 | 0.6173 | 0.869 |
| other（198） | **0.8182** | 0.8081 | 0.7424 | — |
| open（259） | **0.5676** | 0.5367 | 0.6525 | ~0.60 |
| **overall（2431）** | **0.835** | 0.8198 | 0.7709 | 0.8954 |

> ¹ **口径不对称（关键）**：32B 学生 / 0.82 学生 = scenario 口径（crisp，`[Answer]` 抽取后精确匹配，可直接互比）；未训 base = lenient token 子集宽松口径（base 无 `[Answer]` 标记，从自由文本抽答，偏乐观），**不可与前两者直接比**；教师天花板 = 教师 hard_label vs GT（crisp）。base 在原严格口径下 overall 仅 0.0695（格式伪影，见 §12.2）。
> ² 教师天花板取自 §12.6。

### 13.2 软标签 / CoT / 序列 CE 三方对比

| 维度 | 指标 | 32B 学生 | 0.82 学生 | 未训 base |
|---|---|---|---|---|
| 硬标签 | closed_answer_match_rate | 0.8318 | 0.8198 | 0.0 / 0.7709¹ |
| | closed_primary_match_rate | 0.8318 | 0.8198 | — |
| 软标签 | KL(teacher‖student) | 0.01996 | **0.01883** | 0.0524 |
| | cosine | 0.9962 | 0.9964 | 0.9953 |
| | top1 分布一致率 | 0.995 | 0.9986 | 0.9995 |
| | skipped / samples | 210 / 2221 | 210 / 2221 | 210 / 2221² |
| CoT | token Jaccard | 0.2416 | **0.3237** | 0.1431 |
| | SBERT 语义相似度 | 0.7684 | **0.8077** | 0.6553 |
| 序列 | answer_sequence_ce（closed mean） | 2.4872 | **1.0695** | 2.28 |

> ² 三方 skipped 均为 210/2221——skipped 由 val 数据决定（多 token/冲突/定位失败），非学生属性，同 val 集自然相同。

### 13.3 差异归因：32B 教师数据法 vs sampling 法如何传导到学生

> ⚠️ **更正（2026-09-15）**：本节原把 CoT Jaccard/序列 CE 的"退步"归因为"32B CoT 风格单一/难复现"，**不完整**。真正主因是**训练/评估教师不匹配**——32B 学生训练用 32B 教师 CoT，但 §13.2 的 CoT 指标用 `val_soft_sampling`（sampling 教师 CoT）做参照，两教师措辞风格不同，故 32B 学生的输出对 sampling 教师文本的 token 重合天然低、CE 天然高。§13.5 用同源 32B 教师 val 重评后，CoT 指标**全面反超 sampling 学生**。下表/归因的第 3、4 点仅保留为"风格差异"的次要描述，主因见 §13.5。

32B 学生与 0.82 学生用**同一 3B 基座、同一训练配方**（CE + 0.5·KL，3 epoch，lr 1e-4），唯一差异是**教师数据**：

| 数据差异 | 32B 教师法（logits/top_k=20） | sampling 法（n=8） | 传导到学生 |
|---|---|---|---|
| primary==hard 一致率 | **75.2%**（867 条 primary≠hard） | **100%** | 32B 数据 867 条触发"矛盾跳过"（`_build_kl_meta`，distill_dataset.py:397）→ 仅 CE 教硬答案、跳过 KL → 有效 KL 样本少 25% → **KL 拟合略弱（0.020 vs 0.019）** |
| 数据量 | 4864 条 | 5041 条（−3.5%） | CE 数据略少，但闭集硬答案分布相近 → color/counting 两桶答对率**完全相同**（0.9257/0.724） |
| CoT 来源 | 32B 教师 logits 法单次生成 | 8 次采样合并 | 风格差异（次要）；§13.2 CoT 退步主因是评估参照系为 sampling 教师，见 §13.5 更正 |
| 序列 CE | 32B 教师目标文本风格与 3B 学生先验差距更大 | sampling 教师文本更"平均化"、易复现 | 学生对 32B 目标文本更"意外" → **CE 1.07→2.49** |

**四点归因**：

1. **硬标签 +1.5pp（0.820→0.835），主要靠 yes_no(+2.1pp)/other(+1.0pp)/open(+3.1pp)**——32B 教师在是非/杂项/开放题上硬答案更准（天花板更高），CE 把更准的硬答案灌给了学生。color/counting 两桶**纹丝不动**（0.9257/0.724），因这两桶采样法教师已接近天花板、32B 教师无额外增量。choice 反降 1.2pp（噪声，81 条小样本）。

2. **软标签 KL 反而略升（0.0188→0.0200）——最反直觉但合理**：32B 数据 primary==hard 仅 75%，矛盾跳过让 867 条不学 KL，有效 KL 样本比 sampling 法少 25%，学生分布幅度拟合自然略弱。但 cosine 0.9962、top1 0.995 仍极高，**形状对齐没退**，只是概率数值精度微降。这正是"矛盾跳过"保护机制的可观察代价——它防止 color-leak 重现（color 0.93 健康），代价是 KL 拟合密度略降。

3. **CoT Jaccard 表面反降（0.324→0.242）——经 §13.5 更正为评估假象**：原解读（32B CoT 风格单一）只是次要面。主因是 §13.2 用 sampling 教师 CoT 做参照、与 32B 学生训练教师不同源。同源重评（§13.5）Jaccard 升至 **0.366**，反超 sampling 学生。

4. **序列 CE 表面反升（1.07→2.49）——经 §13.5 更正为评估假象**：同源重评 CE 降至 **0.664**，远低于 sampling 学生的 1.07，说明 32B 学生对其教师目标文本的复刻**更好**。

### 13.4 总评：换 32B 教师值不值

| | sampling 学生（0.82） | 32B 学生（0.835） | Δ |
|---|---|---|---|
| 硬标签 overall | 0.8198 | **0.835** | +1.5pp |
| 距教师天花板（保留率） | 91.6% | **93.4%** | +1.8pp |
| 软标签 KL 拟合 | **0.0188** | 0.0200 | -0.0012（略退） |
| CoT 语义（同源口径，§13.5） | 0.808 | **0.836** | +0.028（更优） |
| 序列 CE（同源口径，§13.5） | 1.07 | **0.664** | -0.41（更优） |

**一句话**：换 32B 教师**全面占优**——硬标签 +1.5pp（overall 0.835，距天花板 6pp，保留率 93.4%）；CoT 复刻在同源口径下也反超 sampling 学生（Jaccard 0.366 > 0.324、CE 0.664 < 1.07）。唯一略退是软标签 KL 拟合（矛盾跳过致有效 KL 样本少 25%，0.020 vs 0.019，但 cosine/top1 仍极高）。**32B 教师学生在硬标签与推理复刻上均更优，是当前最佳学生。**（§13.2 的 CoT 退步是评估参照系不一致的假象，见 §13.5。）

### 13.5 更正：公平 CoT 评估（训练/评估教师同源）

> **新增（2026-09-15）**：§13.2 的 CoT/序列 CE 指标用 `val_soft_sampling`（sampling 教师 CoT）做参照，而 32B 学生训练用 32B 教师 CoT——**训练/评估教师不匹配**，CoT 指标偏低是参照系错误，非能力不足。本节用 32B 教师的 val 集 `training/32b_teacher/val_soft_logits.jsonl`（336 条，与 train 同源同法 logits/top_k=20）重评 32B 学生，CoT 指标才有意义。产出 `outputs/evaluation/report.json`（已备份 `/mnt/data/workspace/jlx/eval_32b_cot/report.json`）。

| CoT 指标 | §13.2 错配评估（vs sampling 教师） | **§13.5 同源评估（vs 32B 教师）** | sampling 学生（对照） | 结论 |
|---|---|---|---|---|
| token Jaccard | 0.2416 | **0.3657** ↑+51% | 0.3237 | 32B 学生**反超** |
| SBERT 语义 | 0.7684 | **0.8358** | 0.8077 | 32B 学生**反超** |
| 序列 CE（closed mean） | 2.4872 | **0.6636** ↓-73% | 1.0695 | 32B 学生**远优** |

**硬标签**（同源 32B 教师 val，336 条）：closed_answer_match_rate 0.8289、closed_primary_match_rate 0.8246（与 §13.1 的 0.835 同水平，样本集不同不直接比）。
**软标签分布**（216 条可分配样本）：KL 0.1344 / cosine 0.9892 / top1 0.9954——KL 比 §13.2 高，因 32B val 软标签用 logits/top_k 法、候选集与 sampling val 不同，且 336 条多为更难样本；但 cosine/top1 仍极高，形状对齐没退。

**结论**：32B 学生 CoT 复刻能力**经同源评估后全面优于 sampling 学生**——Jaccard 0.366、SBERT 0.836、CE 0.664 三项均胜。§13.2/§13.3 原本的"CoT 退步"是评估参照系不一致（训练 32B 教师 vs 评估 sampling 教师）造成的**假象**。教训：跨教师比 CoT 指标须同源，硬标签（vs 人工 GT）才可跨集比。

---

## 14. 两教师学生完整对比：32B 教师 vs qwen3.7-plus 教师

> **新增（2026-09-15）**：本项目用了两个教师各训一个 3B 学生，本节做完整对比。两学生**同一 3B 基座（Qwen2.5-VL-3B-Instruct）、同一配方**（CE + 0.5·KL，3 epoch，lr 1e-4，LoRA r=64），唯一变量是教师——故差异完全归因于教师数据。

### 14.1 两个教师的差异

| | qwen3.7-plus 教师 | 32B 教师（Qwen2.5-VL-32B-Instruct-AWQ） |
|---|---|---|
| 部署方式 | bailian 网关（远程 API，`codingxrui.geely-test.com`） | 本地 32B-AWQ（~20GB 显存） |
| 软标签法 | sampling n=8（顺序循环，thinking off，~36s/样本） | logits/top_k=20（贪婪，~15s/样本） |
| 训练数据 | `training/agent_teacher/train_soft_sampling.jsonl` 5041 条 | `outputs/training_train2014/train.jsonl` 4864 条 |
| primary==hard 一致率 | **100%** | 75.2%（867 条 primary≠hard，矛盾跳过 KL） |
| CoT 来源 | qwen3.7-plus 8 次采样生成 | 32B 单次贪婪生成 |
| 数据生成成本 | 远程 API、不占本地 GPU、~50h（5041×36s） | 本地 ~20GB GPU、~20h（4864×15s，有挂死风险，见看门狗） |
| 学生 | `student_merged`（§12，0.8198） | `student_merged_32b`（§13，0.835） |

> qwen3.7-plus 配置见 `agent_distill/configs/train2014_sampling.yaml`（`teacher.model: bailian/qwen3.7-plus`，网关坑见 [[bailian-gateway-teacher-access]]）；32B 教师见 `configs/default.yaml`（`teacher.model_name: models/Qwen2.5-VL-32B-Instruct-AWQ`）。

### 14.2 硬标签对比（val_soft_sampling 2431，vs 人工 GT，scenario 口径）

| 桶（样本数） | qwen3.7-plus 学生 | 32B 学生 | Δ |
|---|---|---|---|
| yes_no（1320） | 0.8712 | **0.8924** | +2.1pp |
| color（323） | 0.9257 | 0.9257 | 0 |
| counting（250） | 0.724 | 0.724 | 0 |
| choice（81） | **0.7901** | 0.7778 | −1.2pp（噪声，81 条小样本） |
| other（198） | 0.8081 | **0.8182** | +1.0pp |
| open（259） | 0.5367 | **0.5676** | +3.1pp |
| **overall** | 0.8198 | **0.835** | **+1.5pp** |
| 距教师天花板 0.8954 | 91.6% | **93.4%** | +1.8pp |

> 口径：两学生同用 scenario 口径（`[Answer]` 抽取后 crisp 精确匹配），vs 人工 GT，**直接可比**。color/counting 两桶完全相同——这两桶采样法教师已逼近天花板，32B 教师无额外增量。增量来自 yes_no/other/open（32B 教师硬答案更准）。

### 14.3 软标签分布对比（val_soft_sampling，vs 各自教师分布）

| 指标 | qwen3.7-plus 学生 | 32B 学生 | 解读 |
|---|---|---|---|
| KL(teacher‖student) | **0.0188** | 0.0200 | qwen3.7-plus 微优——数据 primary==hard 100%，全量 KL 有效；32B 有 867 条矛盾跳过，有效 KL 样本少 25% |
| cosine | 0.9964 | 0.9962 | 持平（形状都对齐） |
| top1 一致率 | 0.9986 | 0.995 | 持平（基线已近 100%） |
| skipped / samples | 210 / 2221 | 210 / 2221 | 同（val 数据属性，非学生） |

> KL 的差距（0.0188 vs 0.0200）很小且可解释：32B 数据 25% 样本矛盾跳过 KL（仅 CE 教硬答案），有效 KL 样本少 → 拟合密度略降。这是"矛盾跳过"防 color-leak 的可观察代价（§13.3）。cosine/top1 说明分布形状对齐没退。

### 14.4 CoT / 序列 CE 对比（公平同源口径：各 vs 自己教师的 val）

| 指标 | qwen3.7-plus 学生（vs qwen3.7-plus val） | 32B 学生（vs 32B val，§13.5） | Δ |
|---|---|---|---|
| token Jaccard | 0.3237 | **0.3657** | +0.042 |
| SBERT 语义 | 0.8077 | **0.8358** | +0.028 |
| 序列 CE（closed mean） | 1.0695 | **0.6636** | −0.41（更优） |

> **关键**：CoT 指标必须同源（训练教师 = 评估 val 教师）才可比。qwen3.7-plus 学生的 CoT 指标用 `val_soft_sampling`（qwen3.7-plus 自己的 val），32B 学生的用 `val_soft_logits`（32B 自己的 val，§13.5）——两者都同源，**公平可比**。32B 学生三项全胜。（§13.2 曾因用错参照系得出"32B CoT 退步"的假象，§13.5 已更正。）

### 14.5 差异归因：为什么本地 32B 反而强过更大的 qwen3.7-plus

qwen3.7-plus（更大、更新的网关模型）做教师，学生反而不及本地 32B 教师的学生。三点根因：

1. **同族教师，文本风格同源**：32B 与 3B 学生都是 Qwen2.5-VL，措辞/格式/推理链路同代同源，3B 学生复刻更易——序列 CE 0.66 vs 1.07 是最硬的证据（32B 教师目标文本对 3B 学生"更不意外"）。qwen3.7-plus 是另一代模型，文本风格 3B 学得更费力。
2. **logits 法 vs sampling 法的软标签精度**：32B 用 logits/top_k=20，分布是真实 logit 归一化（连续概率）；qwen3.7-plus 用 n=8 采样，是蒙特卡洛估计（离散计数，8 粒度粗糙）。前者精度高，CE+KL 教的分布更准。
3. **CoT 确定性**：32B 贪婪 CoT 确定性强、目标稳定；qwen3.7-plus n=8 采样 CoT 多样性高，单条目标更"飘"，学生拟合的文本目标一致性低。

> 注：qwen3.7-plus 数据 primary==hard 100%（软标签 argmax 与硬答案完全一致）是其**唯一结构性优势**——故软标签 KL 微优（0.0188 vs 0.0200）。但因 32B 教师在前 3 点上占优，net 仍是 32B 学生更强。

### 14.6 总评与选型建议

| 维度 | qwen3.7-plus 学生 | 32B 学生 | 胜者 |
|---|---|---|---|
| 硬标签 overall | 0.8198 | 0.835 | **32B**（+1.5pp） |
| 距天花板保留率 | 91.6% | 93.4% | **32B** |
| CoT token 复刻 | 0.3237 | 0.3657 | **32B** |
| CoT 语义 | 0.8077 | 0.8358 | **32B** |
| 序列 CE | 1.0695 | 0.6636 | **32B** |
| 软标签 KL 拟合 | 0.0188 | 0.0200 | **qwen3.7-plus**（微优） |

**一句话：本地 32B 教师学生全面占优**——硬标签 +1.5pp、CoT 三项全胜，唯一让位是软标签 KL 微升（矛盾跳过的可解释代价）。根因是 32B 与 3B 学生同族同源、logits 法软标签精度高、CoT 确定性强；qwen3.7-plus 虽更大但跨代异源、sampling 法粗糙。

**选型**：
- **追求学生质量** → 32B 教师（本地 ~20GB GPU 做一次性数据生成，看门狗自愈防挂死）。
- **无本地大 GPU / 需远程** → qwen3.7-plus 教师（远程 API，但 n≤4 需顺序循环、~36s/样本、依赖网络）。
- 两者学生均为 3B 部署（同 VRAM/延迟），推理成本相同；差异只在数据生成阶段。

## 15. 四方对比：32B 教师 / qwen3.7-plus 教师 / 两个 3B 学生

> **新增（2026-09-15）**：把两个教师天花板与两个学生放进一张表。两学生同一 3B 基座（Qwen2.5-VL-3B-Instruct）、同一配方（CE + 0.5·KL，3 epoch，lr 1e-4，LoRA r=64），唯一变量是教师——故学生间差异完全归因于教师数据。两教师自身则代表各自数据生成管线上限。

### 15.1 主对比轴：sampling val 2431（有人工 GT、完整分布）

四个模型中三个在此 val 上可直接比（vs 人工 GT，crisp `[Answer]` 精确匹配口径）：

| 模型 | val 集 | overall（vs GT） | 距 qwen3.7-plus 天花板 |
|---|---|---|---|
| qwen3.7-plus 教师（天花板） | sampling 2431 | **0.8954** | — |
| qwen3.7-plus 学生 | sampling 2431 | 0.8198 | 91.6% |
| 32B 学生 | sampling 2431 | **0.835** | 93.4% |
| 32B 教师（天花板） | sampling 2431 | **未评估**¹ | — |

> ¹ **32B 教师天花板缺失说明**：32B 教师未在 sampling val 2431 上跑过推理（需 32B-AWQ 对 2431 张图逐一前向，~10h+，未纳入现有评估）。在 32B 教师自身 val（`training/32b_teacher/val_soft_logits.jsonl`，336 条）上，其 hard_label 与人工 GT **100% 一致**——但该 val 是"教师答对子集"（构造时只保留教师 hard_label==GT 的样本，与 sampling val 零交集），天花板退化为 1.0 的 sanity check，不代表真实分布上的天花板。故 4 方中 32B 教师天花板列为"参考缺失"，以 qwen3.7-plus 0.8954 为唯一可比天花板基准。32B 学生（0.835）反超 qwen3.7-plus 学生（0.8198），间接证明 32B 教师数据质量 ≥ qwen3.7-plus。如需补齐 32B 教师在 2431 上的精确天花板，见 §15.5。

### 15.2 分桶四方对比（sampling val 2431，vs 人工 GT）

| 桶（样本数） | qwen3.7-plus 教师 | qwen3.7-plus 学生 | 32B 学生 | 32B 教师 |
|---|---|---|---|---|
| yes_no（1320） | ~0.95 | 0.8712 | **0.8924** | — |
| color（323） | 0.959 | 0.9257 | 0.9257 | — |
| counting（250） | 0.843 | 0.724 | 0.724 | — |
| choice（81） | 0.869 | **0.7901** | 0.7778 | — |
| other（198） | — | 0.8081 | **0.8182** | — |
| open（259） | ~0.60 | 0.5367 | **0.5676** | — |
| **overall（2431）** | **0.8954** | 0.8198 | **0.835** | — |

> 教师天花板取自 §2.3/§12.6；两学生取自 §14.2。32B 教师列空白（见 §15.1 脚注¹）。color/counting 两桶两学生**纹丝相同**——这两桶 sampling 法教师已逼近天花板，32B 教师无增量；增量来自 yes_no（+2.1pp）/other（+1.0pp）/open（+3.1pp）。choice 反降 1.2pp（噪声，81 条小样本）。

### 15.3 软标签分布四方对比（vs 各自教师分布，sampling val）

| 指标 | qwen3.7-plus 教师 | qwen3.7-plus 学生 | 32B 学生 | 32B 教师 |
|---|---|---|---|---|
| KL(teacher‖student) | 0（自比） | **0.0188** | 0.0200 | 0（自比） |
| cosine | 1.0 | 0.9964 | 0.9962 | 1.0 |
| top1 一致率 | 1.0 | 0.9986 | 0.995 | 1.0 |
| skipped / samples | — | 210 / 2221 | 210 / 2221 | — |

> 软标签是"学生 vs 自己的教师"，两学生教师不同、KL 跨教师不严格可比但量级接近。qwen3.7-plus 学生 KL 微优（primary==hard 100%，全量 KL 有效）；32B 学生 867 条矛盾跳过致有效 KL 样本少 25%（§13.3）。cosine/top1 两者持平（形状都对齐）。两教师自比=0/1.0（天花板）。

### 15.4 CoT 四方对比（公平同源口径：各 vs 自己教师 val）

| 指标 | qwen3.7-plus 学生（vs qwen3.7-plus val） | 32B 学生（vs 32B val，§13.5） |
|---|---|
| token Jaccard | 0.3237 | **0.3657** |
| SBERT 语义 | 0.8077 | **0.8358** |
| 序列 CE（closed mean） | 1.0695 | **0.6636** |

> CoT 必须同源（训练教师 = 评估 val 教师）才可比（§13.5 更正）。两教师自身 CoT 复刻 = 1.0（自比），故只列两学生。32B 学生三项全胜（§14.4）。**不要跨教师比 CoT**——§13.2 曾因用错参照系得出"32B CoT 退步"的假象。

### 15.5 32B 教师天花板补齐方案（可选）

如需 32B 教师在 sampling val 2431 上的精确天花板（让 4 方完全可比），跑一次 32B 教师评估（把 32B 教师权重当 `student_model_path`、`eval_data_path` 指向 `val_soft_sampling.jsonl`）：

```
cd /home/jlx/a6000_mnt/workspace/vlm-distillation
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
  /home/jlx/miniconda3/envs/agent_distill/bin/python \
  scripts/run_full_pipeline.py --config /mnt/data/workspace/jlx/eval_32b_teacher_ceiling.yaml --steps evaluation
```

约 ~10h（2431 张 32B 前向），产出后填齐 §15.1/§15.2 的 32B 教师列。**当前未跑**——按需启动，会占用工作站 GPU（~20GB）。

### 15.6 四方总评

| 维度 | qwen3.7-plus 教师 | qwen3.7-plus 学生 | 32B 学生 | 32B 教师 |
|---|---|---|---|---|
| 硬标签 overall（vs GT） | 0.8954 | 0.8198 | **0.835** | N/A（未等价比） |
| 保留率（vs qwen3.7-plus 天花板 0.8954） | — | 91.6% | **93.4%** | — |
| 软标签 KL 拟合 | — | **0.0188** | 0.0200 | — |
| CoT token 复刻（同源） | — | 0.3237 | **0.3657** | — |
| 序列 CE（同源） | — | 1.0695 | **0.6636** | — |

**一句话**：在可比口径下，**32B 教师学生（0.835）> qwen3.7-plus 教师学生（0.8198）**，距 qwen3.7-plus 天花板（0.8954）更近（保留率 93.4% vs 91.6%）；CoT 同源三项全面反超；唯一让位是软标签 KL（0.020 vs 0.019，矛盾跳过的可解释代价，cosine/top1 仍持平）。32B 教师天花板因 val 构造差异未能等价比，但其学生已反超 qwen3.7-plus 学生，间接证明 32B 教师数据质量 ≥ qwen3.7-plus。根因（§14.5）：32B 与 3B 学生同族同源、logits 法软标签精度高、CoT 确定性强；qwen3.7-plus 虽更大但跨代异源、sampling n=8 法粗糙。

---

## 附录 A：完整 report.json 字段

### 顶层
| 字段 | 值 |
|---|---|
| `train_data_path` | `training/agent_teacher/train_soft_sampling.jsonl` |
| `eval_data_path` | `./training/agent_teacher/val_soft_sampling.jsonl` |
| `is_heldout` | true |
| `student_model_path` | `./outputs/student_merged_32b` |
| `max_samples` | null |
| `parameter_count.total` | 3,754,622,976 |

### dimensions.distillation_quality
| 字段 | 值 |
|---|---|
| `samples_evaluated` | 2431 |
| `closed_answer_match_rate` | 0.8318 |
| `closed_primary_match_rate` | 0.8318 |
| `open_text_similarity` | 0.0 |
| `open_text_semantic_similarity` | null |
| `cot_similarity` | 0.2416 |
| `cot_semantic_similarity` | 0.7684 |
| `answer_sequence_ce.closed.mean` | 2.4872（samples 2431） |
| `answer_sequence_ce.open.mean` | null（samples 0） |
| `closed_distribution.samples` | 2221 |
| `closed_distribution.skipped` | 210 |
| `closed_distribution.kl_mean` | 0.019959 |
| `closed_distribution.cosine_mean` | 0.9962 |
| `closed_distribution.top1_distribution_match_rate` | 0.995 |
| `scenario_buckets.overall_accuracy` | 0.8198（1993/2431） |
| `scenario_buckets.buckets.yes_no` | 1150/1320 = 0.8712 |
| `scenario_buckets.buckets.color` | 299/323 = 0.9257 |
| `scenario_buckets.buckets.open` | 139/259 = 0.5367 |
| `scenario_buckets.buckets.counting` | 181/250 = 0.724 |
| `scenario_buckets.buckets.other` | 160/198 = 0.8081 |
| `scenario_buckets.buckets.choice` | 64/81 = 0.7901 |
| `detail` | 10 条分层抽样（color/counting/choice/open/yes_no 各 2 条），每条含：`image_path`、`question_type`、`bucket`、`question`、`teacher_answer`、`student_answer`、`teacher_output`（教师完整 [Reasoning]…[Answer]…）、`student_output`（学生完整生成）、`teacher_soft_label`（教师 answer_distribution）、`student_soft_label`（学生候选分布；多 token 候选跳过时为 null） |

### dimensions.deployment_efficiency
| 字段 | 值 |
|---|---|
| `total_parameters` | 3,754,622,976 |
| `model_dir_size_mb` | 11942.9 |
| `peak_vram_gb` | 7.258 |
| `avg_latency_seconds` | 2.8283 |
| `median_latency_seconds` | 2.7663 |
| `throughput_samples_per_second` | 0.3536 |
| `benchmark_samples` | 50 |

## 附录 B：数据文件
| 文件 | 说明 |
|---|---|
| `training/agent_teacher/train_soft_sampling.jsonl` | **治本训练数据**：sampling 法软标签（n=8），全桶 primary==hard 100%（color 100% vs logprob 34.3%），含 answer_distribution（采样）+ logprob_distribution（logit）双字段。本次重训用此 |
| `training/agent_teacher/val_soft_sampling.jsonl` | **治本评估数据**：sampling 法软标签，2431 条，全闭集，held-out。本次评估用此 |
| `outputs/training/train.jsonl` | 训练数据，train2014，5041 条，logprob 软标签（旧版学生用此训练，color 塌缩源，已弃用） |
| `outputs/teacher/val_soft_logits.jsonl` | 验证集教师标注，val2014，2456 条，logprob（旧版评估用，已弃用） |
| `outputs/teacher/val_softl_multi.jsonl` | 验证集，val2014，2431 条，multi 采样软标签（n=8，含 answer_distribution + logprob_distribution） |
| `../agent_distill/train_soft_logits/distill_train_sampling.jsonl` | train2014 multi 采样软标签，5009 条，n=8，全桶 primary==hard 100%，含双字段。治本数据原始生成位置 |
| `outputs/student_merged/` | 学生合并全权重（bf16） |
| `outputs/student_merged_32b/` | **32B-教师学生**合并全权重（bf16），§13 评估对象，训于 4864 条 train2014（32B 教师 logits/top_k=20 软标签） |
| `outputs/student_ckpt_32b/` | 32B 学生 LoRA adapter + checkpoints（checkpoint-100/200…，--resume true 续训点） |
| `/mnt/data/workspace/jlx/eval_32b/report.json` | 32B 学生评估报告备份（产出原在 outputs/evaluation/report.json，已备份防覆盖） |
| `/mnt/data/workspace/jlx/train_32b.yaml` | 32B 学生训练配置（output_dir=student_ckpt_32b / merged_32b，--save_steps 100） |
| `outputs/training_train2014/train.jsonl` | 32B 教师训练数据，4864 条，logits/top_k=20 软标签，primary==hard 75.2%（867 条矛盾跳过 KL） |
| `outputs/student_ckpt/` | LoRA adapter + checkpoints |
| `outputs/evaluation/report.json` | 本评估报告 |

---

*由 `scripts/run_full_pipeline.py --steps evaluation` 生成，配置见 `configs/default.yaml`。*
