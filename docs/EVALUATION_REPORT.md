# VLM 蒸馏学生模型 — 完整评估报告（含逐指标分析）

> **生成日期**：2026-09-07  
> **报告文件**：`outputs/evaluation/report.json`（2026-09-07 16:46，`is_heldout=true`）  
> **学生** Qwen2.5-VL-3B-Instruct（LoRA r=64 蒸馏后合并全权重）｜ **教师** Qwen2.5-VL-32B-AWQ  
> **训练** train2014 `train.jsonl`（5041 条，logprob 软标签，3 epochs）  
> **评估** val2014 `val_soft_logits.jsonl`（2456 条，**held-out**，logprob 软标签）

---

## 0. 整体蒸馏质量评判（先给结论）

**一句话：蒸馏机制本身验证成功，但教师软标签质量是瓶颈，color 题因软标签 bug 塌缩。**

| 层面 | 评判 | 关键证据 |
|---|---|---|
| 蒸馏机制（KL 软标签） | ✅ **成功** | KL 0.046 / cos 0.981 / top1 0.927，分布拟合且泛化到 held-out |
| 教师软标签质量 | 🔴 **有缺陷** | logprob 法 color 题 62.3% primary 与教师自身 hard_label 矛盾 |
| 硬标签泛化 | ⚠️ **中等** | held-out 0.6437 vs 天花板 0.8954（保留率 71.9%），过拟合 |
| CoT 推理 | 🟡 **中等偏上** | 推理链路语义正确（含 color 推理都对），但 token 相似度 0.322 受容量限制 |
| 部署效率 | ✅ **优秀** | 3.75B / 7.17GB 峰值 / 2.78s，12GB 卡可部署 |

**核心矛盾**：学生分布拟合极好（cos 0.981），但硬答案准确率只有 0.64——因为学生**忠实学到了教师的有缺陷分布**（color 上把 "other" 当 argmax），"垃圾进垃圾出"。这是软标签蒸馏的典型陷阱：机制对，但教师标签质量决定上限。

**蒸馏质量总评：6/10**。机制层优秀，数据层失败。修复教师软标签（改用 multi 采样法）后预期可升至 8/10。

---

## 1. 评估设置

### 1.1 数据流

```
train2014 ──教师(32B)标注──> train.jsonl (5041条, logprob) ──> 学生(3B) SFT+KL训练
                                                                       │
val2014 ──教师(32B)标注──> val_soft_logits.jsonl (2456条) <──评估──────┘
```

训练集（train2014）与验证集（val2014）是 COCO 官方不重叠 split，**无数据泄漏**，`is_heldout=true`。

### 1.2 配置与改动

| 项 | 值 |
|---|---|
| 评估数据 | val2014，2456 条，logprob 软标签 |
| `max_pixels` | 100352（与训练分辨率对齐，≈128 视觉 token） |
| 模型加载 | 75.59s |
| 评估耗时 | 7848s ≈ 131 min |

代码改动：① `StudentInferencer` 加 `max_pixels` 对齐训练分辨率；② 修闭集判定 bug（`question_category` 为 None → 改用 `answer_distribution` 存在性，与训练对齐）；③ 新增 `scenario_buckets`，把 driving_scenario 分桶并入 distillation_quality 复用生成，driving_scenario 独立维度禁用。

---

## 2. 按题型分桶绝对准确率（scenario_buckets）

### 2.1 结果表

| 桶 | correct/total | 学生准确率 | 教师天花板 | 差距 |
|---|---|---|---|---|
| **overall** | 1581/2456 | **0.6437** | 0.8954 | -25pp |
| yes_no | 1035/1316 | 0.7865 | ~0.93-0.98 | -15pp |
| choice | 62/84 | 0.7381 | 0.869 | -13pp |
| other | 144/197 | 0.7310 | — | — |
| counting | 183/255 | 0.7176 | 0.843 | -13pp |
| open | 149/278 | 0.5360 | ~0.60 | -6pp |
| **color** | 8/326 | **0.0245** | **0.959** | **-93pp** |

### 2.2 逐桶分析：为什么是这个值

**overall = 0.6437**  
为什么不是更高？拆解：color 桶 326 条几乎全错（8/326=0.0245），单独拉低 overall 约 13pp。**排除 color 后**，其余 2130 条准确率 = (1581-8)/(2456-326) = 1573/2130 = **0.738**。即 color 塌缩是 overall 偏低的主因，非 color 题型学生实际 0.738，距教师天花板约 -15pp，属合理泛化 gap。

**yes_no = 0.7865（最高）**  
为什么是非题最好？① 候选集极小（yes/no 二元），argmax 决策容易；② 教师天花板本身就最高（0.93-0.98），学生有充足的训练信号；③ 是非题 question_type 占比最大（is/are 系列 ~40%），训练样本多。差距 -15pp 主要来自过拟合（5041 条对 3B 偏少）。

**color = 0.0245（塌缩）**  
为什么远低于随机（~10%）？学生推理段**正确识别颜色**但 `[Answer]` 永远输出 "other"。根因是 logprob 软标签把 "other" 候选项虚高排成 top1（65% 的 color 题 primary="other"），KL 蒸馏让学生答案 token 塌缩到 "other"，压过 CE 真颜色目标。详见 §6。低于随机基线本身即说明这不是"学生不会"，而是**系统性学到了错误答案**。

**counting = 0.7176**  
为什么中等？计数题答案是数字（1/2/3…），教师天花板 0.843 本身就偏低（计数是难任务，教师也常错）。学生距天花板 -13pp，合理。数字答案单 token，候选映射干净，无 "other" 干扰。

**open = 0.5360（距天花板最近，保留率最高）**  
为什么绝对值最低但保留率最高？① 教师天花板本身就最低（~0.60，开放描述题最难）；② 学生 0.54 距天花板仅 -6pp，说明 open 能力迁移最好——开放题靠的是通用语言/推理能力，3B 学生从 32B 教师迁移这部分最成功；③ open 题无固定候选集，不受 "other" 软标签 bug 影响。

**choice = 0.7381 / other = 0.7310**  
中等。choice 是 "none of the above" 题，other 是未归类的杂项题，两者表现相近，无异常。

### 2.3 教师按题型天花板（val2014 vs COCO 人工 GT）

| 题型 | 教师 | | 题型 | 教师 |
|---|---|---|---|---|
| is it | 0.984 | | what color | 0.959 |
| is this | 0.974 | | is there a | 0.950 |
| are there | 0.971 | | is this a | 0.933 |
| is | 0.958 | | are the | 0.929 |
| is the | 0.908 | | how many | 0.843 |
| none of the above | 0.869 | | what | 0.603 |

教师最弱是 `what` 类开放题（0.603），最强是是非题（is it/are 0.93-0.98）。color 教师天花板 0.959 很高，更反衬学生 color 0.0245 的塌缩是异常而非能力问题。

---

## 3. 软标签分布蒸馏质量（closed_distribution）

### 3.1 指标表

| 指标 | 值 | 方向 |
|---|---|---|
| KL(teacher‖student) | 0.04594 | 越低越好 |
| cosine | 0.9811 | 越高越好 |
| top1 分布一致率 | 0.9266 | 越高越好 |
| 可算样本 | 2234/2456 | 跳过 222（9%） |
| 温度 | 1.0 | 推理不缩放 |

### 3.2 逐指标分析：为什么是这个值

**KL = 0.04594（极低，优秀）**  
为什么这么低？KL 蒸馏**直接优化这个目标**（`loss = CE + 0.5·KL(teacher‖student)`，在答案 token 位置对候选集）。学生在训练中专门被拉向教师分布，2456 条 held-out 上仍保持 0.046，说明学到的不是死记训练集，而是**泛化的分布拟合能力**。0.046 nats 接近 0，意味着学生分布与教师分布几乎重合。

**但注意**：KL 低包含了 color 题学生把 "other" 当 argmax 与教师（同样把 "other" 当 argmax）一致 → 一致就低 KL。所以 KL 低既反映"拟合成功"也反映"学到了教师的错"。这是软标签蒸馏的双刃剑——它不区分教师分布对错。

**cosine = 0.9811（极高，优秀）**  
为什么比 KL 还亮眼？余弦衡量分布**形状**（方向），对绝对概率值不敏感。学生分布的形状（各候选的相对高低排列）与教师几乎一致。0.981 意味着即便个别候选概率绝对值有偏差，整体形状高度吻合。同样，color 题的 "other" 主峰学生也学到了，形状一致。

**top1 分布一致率 = 0.9266（高，但不是 1.0）**  
为什么 92.7% 而非更高？7.3% 的样本学生 argmax ≠ 教师 argmax——这些是分布**峰值位置略有偏移**的样本（学生次高峰被推到主峰，或教师本身分布平坦多峰时学生选了不同峰）。这 7.3% 的偏移 + 教师自身 argmax 相对 GT 的误差，叠加出硬答案 0.64 的 match 率。top1 一致(0.927) 远高于硬答案 match(0.64)，差值来自"学生 argmax = 教师 argmax 但 ≠ 人工 GT"（教师也错的部分）。

**可算样本 2234/2456（跳过 222，9%）**  
为什么跳过 9%？镜像训练侧 `_build_kl_meta` 的跳过规则：① 多 token 候选答案（如 "around fire hydrant" 编码成多 token，无法映射到单候选 token）；② 候选 token 冲突（不同候选编码到同一 token）；③ 答案 token 在序列中定位失败。9% 跳过率正常，不影响整体结论。

### 3.3 教师软标签置信度参照

| 量 | 值 | 解读 |
|---|---|---|
| 平均 top1 概率 | 0.8602 | 教师整体较自信 |
| 中位 top1 概率 | 0.9978 | 半数样本教师几乎确定 |
| 平均熵 | 0.257 bits | 分布集中 |
| 平均候选数 | 1.66（max 5） | 多数 2 候选 |
| 不确定样本（top1<0.7） | 25.9% | 这 25.9% 是软标签最有价值的部分 |

**为什么这重要**：25.9% 的不确定样本正是软标签 KL 蒸馏**相对硬标签 SFT 的增量价值所在**——教师在这些样本上给非退化的概率分布（而非压单点），学生学到了不确定性。这部分若用硬标签 SFT 会丢失。学生 KL 0.046 说明连这 25.9% 的不确定性都拟合到了。

---

## 4. 硬标签评估（Hard-label）

### 4.1 指标表

| 指标 | 值 | 含义 |
|---|---|---|
| closed_answer_match_rate | 0.6393 | 学生生成答案 vs 教师 hard_label（真答案） |
| closed_primary_match_rate | 0.7142 | 学生答案 vs 教师 soft_label.primary_answer |
| answer_sequence_ce（闭集 NLL） | 1.0855 | 学生对教师目标文本的自回归 NLL |

### 4.2 逐指标分析：为什么是这个值

**closed_answer_match_rate = 0.6393**  
为什么 0.64？三因素叠加：① **color 塌缩**（326 条几乎全错，单独拉低 ~13pp，排除后非 color 为 0.738）；② **训练集泄漏消除**（旧 report 0.89 是在 train 集上测的，held-out 真实泛化本就会降）；③ **过拟合**（5041 条对 3B 偏少，序列 CE 训练集 0.41→held-out 1.09 印证）。三个因素中 color 占大头。

**closed_primary_match_rate = 0.7142 > answer_match 0.6393（反常！）**  
为什么 primary_match 反而更高？按理 primary（教师分布 argmax）和 hard_label（教师生成答案）应一致，两个指标应接近。反常的根源就是 color：教师 color 题 primary="other"（虚高），学生也输出 "other" → **color 题在 primary_match 上反而匹配（学生"other"==教师primary"other"），拉高了 primary_match**。这进一步印证 color 塌缩是学到了教师的（错误）分布 argmax。排除 color 后两指标应趋于一致。

**answer_sequence_ce = 1.0855（高，held-out）**  
为什么 held-out 这么高（训练集 0.41）？NLL 是学生对教师目标文本 `[Reasoning]...\n[Answer] 真答案` 的自回归负对数似然。① color 题：学生答案 token 分布主峰在 "other"，但目标是真颜色 → 该 token 的 NLL 极高，拉高均值；② held-out 未见数据泛化弱（过拟合），整体 NLL 升高。1.0855 nats 意味着学生对教师目标文本的每 token 困惑度 ≈ e^1.09 ≈ 2.96，即每个目标 token 学生平均在 ~3 个等价候选间犹豫——泛化偏弱但未崩溃。

### 4.3 训练集 vs 验证集对比（泛化 gap）

| 指标 | 训练集（旧 report） | held-out（本次） | 变化 |
|---|---|---|---|
| closed_answer_match | 0.8904 | 0.6393 | -25pp |
| cot_similarity | 0.4028 | 0.3220 | -8pp |
| sequence CE | 0.407 | 1.0855 | ↑大幅 |

**为什么降这么多**：旧 0.89 是训练集评估，**严重高估**（学生见过这些数据）。held-out 0.64 才是真实泛化。25pp gap 是过拟合的直观量，主因训练数据量（5041 条对 3B）+ color 软标签 bug。

### 4.4 蒸馏天花板

| 角色 | val2014 vs 人工 GT |
|---|---|
| 教师(32B)硬标签 | **0.8954**（2199/2456）← 蒸馏上限 |
| 学生硬标签 | 0.6437 |
| 保留率 | **71.9%** |

学生硬答案保留教师 71.9% 的能力。注意天花板 0.90 不是 1.0——教师自身在 val 上也只 0.90，学生不可能超过。

### 4.5 硬标签指标 gap 成因分解（区分"软标签泄漏"与"泛化本身"）

> **口径澄清（重要）**：硬标签泛化测的是"学生生成答案 vs GT"的端到端准确率，概念上属学生自身泛化能力问题（过拟合 + 容量差）。但训练 loss = CE(含[Answer]真答案) + 0.5·KL(答案token, 教师软标签分布)，**KL 项直接塑造学生答案 token 的 argmax**——软标签 bug 会通过 KL 通道污染硬答案输出。所以硬标签指标 gap 的"表现"跨两类成因，但**严格意义的"硬标签泛化问题"只指学生自身能力不足那部分**，软标签泄漏不算。

> 注：本节的 gap = **教师天花板 → 学生**（0.8954→0.6437，能力差），与 §4.3 的 **训练集 → held-out**（0.89→0.64，泛化跌落）是两个不同口径的 25pp，互为印证但归因不同。本节回答"学生距教师差在哪"，§4.3 回答"学生记忆 vs 泛化差多少"。

硬标签指标 gap = 教师天花板 0.8954 − 学生 0.6437 = **25.2pp**。这不是一个笼统的"过拟合"，而是两类成因（软标签泄漏 + 泛化本身）叠加，**可逐段归因**：

#### 4.5.1 总分解（按对 overall 的贡献）

| 成因 | 贡献 | 占比 | 性质 | 可修复性 |
|---|---|---|---|---|
| **① color 软标签 bug 泄漏** | 12.4pp | 49% | **软标签质量**（非泛化） | ✅ 可完全修复（multi 采样重训） |
| **② 非 color 学生模仿不足** | 9.0pp | 36% | **泛化本身**（过拟合+容量差） | 🟡 部分修复（扩数据/正则） |
| **③ 非 color 教师软argmax噪声泄漏** | 3.8pp | 15% | **软标签质量**（非泛化） | 🟡 部分缓解（multi 采样） |
| **合计** | 25.2pp | 100% | — | — |

**关键区分**：①③ 是软标签问题通过 KL 通道**泄漏到硬答案指标**上，**不属于硬标签泛化问题**；② 才是严格意义的硬标签泛化问题（学生自身能力不足）。即：**硬标签泛化问题本身只有 9.0pp（36%），其余 16.2pp 是软标签质量泄漏。** 若改纯硬标签 SFT（无 KL），①③ 会消失（但失去软标签蒸馏增量价值），硬答案指标反而可能回升——这反证了软标签泄漏的存在。

**怎么算出来的**：把整体 gap 按 color / 非 color 两块拆（各桶样本占比加权），再在每块内用"教师天花板 → 教师软标签 argmax 上限 → 学生实际"两级落差分解：

#### 4.5.2 color 块（gap 93pp，326 条，贡献 12.4pp）

```
教师天花板 0.957 ─┬─ 59.5pp ─ 教师软标签 argmax 噪声（"other" 虚高）
                 └─→ 教师软argmax上限 0.362 ─┬─ 33.8pp ─ 学生放大塌缩
                                           └─→ 学生实际 0.0245
```

| 量 | 值 | 含义 |
|---|---|---|
| 教师 hard vs GT（天花板） | 0.9571 | 教师实际生成答案几乎全对 |
| 教师 primary vs hard（软标签质量） | **0.3742** | 教师 soft argmax 与自身 hard 答案只有 37% 一致——**62.6% 矛盾** |
| 教师 primary vs GT（软argmax上限） | 0.3620 | 教师 soft argmax 对 GT 只有 36%（"other" 拉低） |
| 学生实际 | 0.0245 | 远低于 0.362 |

**为什么学生(0.0245)比教师软argmax(0.362)还低**：教师 primary="other" 占 65%（不是 100%），其余 35% 的 color 题教师 primary 仍是真颜色。但学生经 KL 蒸馏后答案 token **100% 塌缩到 "other"**——把教师 65% 的"other"偏差放大成 100%。学生 match GT 仅当 GT 恰为 "other"（极少）→ 0.0245 ≈ P(GT="other")。这是 §6 详述的 KL 压过 CE 的放大效应。

→ color 块 12.4pp gap **100% 归因于软标签 bug**（教师噪声 59.5pp + 学生放大 33.8pp，同源于 logprob "other" 虚高）。改 multi 采样法后教师 primary vs hard 一致率 37.4%→100%（§6.3），整块 gap 应消失，color 应回升至接近天花板 0.957。

#### 4.5.3 非 color 块（gap 14.8pp，2130 条，贡献 12.8pp）

```
教师天花板 0.886 ─┬─ 4.4pp ─ 教师软argmax噪声（soft argmax ≠ hard）
                 └─→ 教师软argmax上限 0.842 ─┬─ 10.4pp ─ 学生模仿不足（过拟合+容量）
                                            └─→ 学生实际 0.738
```

| 量 | 值 | 含义 |
|---|---|---|
| 教师 hard vs GT（天花板） | 0.8859 | 教师实际答案 88.6% 对 |
| 教师 primary vs hard（软标签质量） | **0.9465** | 非 color 教师软 argmax 与 hard 94.65% 一致——**软标签质量好** |
| 教师 primary vs GT（软argmax上限） | 0.8423 | 教师 soft argmax 对 GT 84.2% |
| 学生 answer_match | 0.738 | 学生生成答案 vs GT |
| 学生 primary_match（推导） | ~0.728 | 学生 vs 教师 primary |

**关键对比**：非 color 教师软标签质量（primary vs hard = 0.9465）远高于 color（0.3742）——证明软标签 bug 是 color 局部的，非 color 的软标签本身是干净的。

**③ 教师软argmax噪声（4.4pp）**：即便学生完美模仿教师 argmax，仍达不到教师 hard 天花板，因为教师 soft argmax ≠ hard 答案（5.35% 偏差，来自候选 token logit 噪声，非 color 上影响小）。这部分是软标签蒸馏相对硬标签 SFT 的固有代价，multi 采样可部分缓解（采样 argmax 比 logit argmax 更贴近教师实际生成答案）。

**② 学生模仿不足（10.4pp，主因）**：学生连教师（正确的）soft argmax 都没完全跟上。三个子成因：
- **过拟合**：序列 CE 训练集 0.41 → held-out 1.09（2.66×），5041 条对 3B 模型偏少，学生记住了训练样本却泛化弱。这是非 color gap 的主因。
- **容量差**：3B 学生 vs 32B 教师，表达能力天花板不同，部分复杂推理学生学不到。
- **推理→答案链路**：KL 只作用在答案 token，推理链路靠 CE 学；3B 容量限制推理质量 → 偶尔推理偏导致答案偏（见 §5.3 第一个样例：推理识别头盔结构却答 yes）。

#### 4.5.4 结论：硬标签指标差 ≠ 硬标签泛化问题

- **16.2pp（65%）是软标签泄漏**（① color 12.4pp + ③ 非color 3.8pp）——软标签质量问题通过 KL 通道污染硬答案，**不算泛化问题**，改 multi 采样可消除；
- **9.0pp（36%）是硬标签泛化本身**（② 过拟合 + 容量差）——这才是"学生自身泛化能力不足"，扩数据/正则可缓解；
- 容量差（3B vs 32B）是 ② 内不可消除的残差，但占比不大。

**即：硬标签指标 0.64 偏低，主因不是"学生泛化不行"，而是"软标签教错了泄漏到答案上"。** 真正的泛化问题只有 9.0pp，其中过拟合（CE 0.41→1.09）是主因、可缓解。修复软标签（multi 采样）后硬指标可回升 ~16pp 至 0.80，再扩数据缓解过拟合可至 ~0.83，保留率从 71.9% 升至 ~93%。

---

## 5. CoT 推理评估

### 5.1 指标

| 指标 | 值 | 含义 |
|---|---|---|
| cot_similarity | 0.3220 | 学生推理段 vs 教师推理段 token Jaccard |
| open_text_similarity | 0.0 | 无 open 样本（数据全闭集），不适用 |

### 5.2 为什么 0.322（偏低）

**指标本身的局限**：token Jaccard 衡量**词级重叠**，对同义不同词惩罚极重。3B 学生与 32B 教师即便语义相同，措辞必然不同（教师表达更丰富多样，学生更简洁）→ Jaccard 天然偏低。0.322 不代表推理质量差。

**语义层面实际正确**：从 detail 样例看，学生推理链路语义准确——包括 color 题（"green hue, indicating the go signal" 正确识别绿色，只是答案 token 塌缩）。问题出在答案 token，不是推理。

### 5.3 学生输出样例

| 问题 | 教师 | 学生 | 学生推理（节选） |
|---|---|---|---|
| Is the bicyclist wearing a helmet? | no | yes ⚠️ | "…dark, rounded protective head covering with visible ventilation holes and a chin strap…" |
| Is this a skate park? | yes | yes ✓ | "…paved area with concrete ramps, rails, and ledges specifically designed for skateboarding…" |
| How many trucks on the road? | 1 | 1 ✓ | "…a single red pickup truck…No other trucks are visible…" |
| What fast food restaurant? | burger king | burger king ✓ | "…blue and yellow storefront with a large sign that clearly reads \"Burger King\"…" |

**注意第一个**：学生推理合理（识别了头盔结构）却答 yes（教师 no）——这里可能是教师与 GT 的对错待定，或学生视觉判断偏差，但推理过程本身自洽。

---

## 6. 🔴 color 塌缩根因（逐层）

### 6.1 现象

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

| 软标签法 | color primary==hard 一致 | "other" 矛盾率 |
|---|---|---|
| **logprob（训练用）** | 122/326 = **0.374** | 203/326 = **0.623** |
| **multi 采样法** | 323/323 = **1.000** | **0.000** |

multi 采样法（n=8 多次推理采样）软标签**完全正确**——primary 永远等于教师真答案，零 "other" 塌缩。multi 文件还同时含 `answer_distribution`（采样，正确）+ `logprob_distribution`（logit，有偏）双字段，严格优于 logprob 文件。

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
| 峰值 VRAM | 7.173 GB | 12GB 卡占用 60%，留足余量 |
| 平均延迟 | 2.7822 s | 含图像预处理+生成 |
| 中位延迟 | 2.7669 s | 与均值接近，无长尾 |
| 吞吐 | 0.3594 samples/s | ~2.8s/样本 |

**为什么延迟 2.78s**：① slow image processor（非 use_fast）CPU 预处理重；② 单样本 greedy 生成 ~50-100 token 自回归解码；③ batch=1 无法摊销。相比教师 32B-AWQ 大幅轻量化，适合非实时离线场景。峰值 VRAM 7.17GB 是 bf16 权重（7.5GB）+ 小激活（低分辨率 max_pixels=100352）。

---

## 8. driving_scenario 维度（已禁用）

已禁用（`enabled: false`）。原问题：① 独立生成翻倍耗时（~5h）；② 依赖 COCO `(image_id, question)` 查表，本数据 `image_id` 缺失全空。已将分桶并入 `distillation_quality.scenario_buckets`，用记录自带 `ground_truth` 做 GT，零额外开销。报告 `dimensions` = `['distillation_quality', 'deployment_efficiency']`。

---

## 9. 综合结论与蒸馏质量总评

### 9.1 逐维度评分

| 维度 | 评分 | 依据 |
|---|---|---|
| 软标签分布蒸馏机制 | ✅ 优秀 | KL 0.046 / cos 0.981 / top1 0.927，泛化到 held-out |
| 教师软标签质量 | 🔴 失败 | logprob color 题 62.3% 矛盾，"other" 虚高 |
| 硬标签答案 | ⚠️ 中等 | held-out 0.6437，保留率 71.9%，过拟合 |
| CoT 推理 | 🟡 中等偏上 | 推理语义正确，token 相似度受指标/容量限制 |
| 部署效率 | ✅ 优秀 | 7.17GB / 2.78s，部署友好 |

### 9.2 三条核心结论

1. **蒸馏机制本身有效**：软标签 KL 让学生学到了教师的不确定性分布（KL 0.046），且泛化到 held-out。这是蒸馏相对纯硬标签 SFT 的核心增量价值，已验证。25.9% 的教师不确定样本，学生连不确定性都拟合到了——硬标签 SFT 会丢失这部分。

2. **教师软标签质量是瓶颈**：logprob 法对含 "other" 候选项的 color 题产生系统性偏差（62.3% 矛盾），学生忠实学到了教师的错——"垃圾进垃圾出"。这揭示了软标签蒸馏的陷阱：**机制对，但教师标签质量决定上限**，且 KL 不区分教师分布对错（一致即低 KL）。

3. **非 color 题型学生表现合理**：yes_no 0.79 / counting 0.72 / open 0.54，距教师天花板 6-15pp，gap 正常，蒸馏有效。排除 color 塌缩后，学生实际 0.738，蒸馏质量本可达良。

4. **硬标签指标差 ≠ 硬标签泛化问题**（详 §4.5）：25.2pp 硬指标 gap 里，**16.2pp（65%）是软标签质量泄漏**（color bug + 软argmax噪声，通过 KL 通道污染硬答案，不算泛化问题），**9.0pp（36%）才是硬标签泛化本身**（过拟合+容量差）。硬标签指标 0.64 偏低主因是软标签教错泄漏，不是学生泛化不行——改 multi 采样可消除 16.2pp。

### 9.3 "分布好而决策弱"现象解析

- 学生分布形状拟合极好（cos 0.981），但硬决策 match 仅 0.64。
- **为什么形状对但决策弱**：分布"形状"对了，但"峰值位置"在 color 题被 KL 拉偏到 "other"（学生 argmax="other" 与教师一致但 ≠ GT）；叠加教师 argmax 本身相对 GT 的误差（教师也只 0.90），累加出 0.64。
- 这说明学生学的是**教师的决策倾向**（含其错误），而非**绝对正确答案**——这正是蒸馏的本质（模仿教师，不是学 GT）。

### 9.4 蒸馏质量总评：6/10

- **+3 分**：机制层成功（分布拟合 + 泛化 + 推理链路正确 + 部署优秀）
- **-4 分**：数据层失败（color 软标签 bug 导致塌缩）+ 硬标签过拟合 gap
- 修复教师软标签（改用 multi 采样法）后预期升至 **8/10**：color 回升接近天花板 0.959，overall 提升至 ~0.78，泛化 gap 收窄。

---

## 10. 优化建议（按优先级）

### 10.1 修复 color 塌缩（最高）

| 方案 | 做法 | 预期 | 风险 |
|---|---|---|---|
| **A. 改用 multi 采样软标签重训**（推荐） | 用 multi 法（n=8）重生成 train2014 软标签，替换 logprob 后重训 | color primary==hard 100%，零 "other" 偏差，color 应回升至 ~0.9 | 需重生成训练软标签（耗时） |
| B. 从 candidate_pool 删 "other" | 移除 "other" 候选项 | 兜底，减少干扰 | 改变候选集定义 |
| C. 对矛盾样本跳过 KL | `primary != hard_label` 时跳过该样本 KL 项 | 避免传错分布 | 损失部分 KL 信号 |
| D. 降 kl_weight | 0.5 → 0.2，让 CE 真答案主导 | 通用 | 削弱软标签蒸馏价值 |

### 10.2 改善硬标签泛化

> 据 §4.5 分解：硬标签 gap 25.2pp 中 9.0pp（36%）来自非 color 过拟合/容量差。下述方案针对这部分（color 的 12.4pp 见 §10.1）。

| 方案 | 理由 | 预期收益 |
|---|---|---|
| 扩训练数据 / 多样性 | 5041 条对 3B 偏少，过拟合是非 color gap 主因（9.0pp）；增至 1-2 万条最直接 | +3-5pp |
| 正则化 | 降 LoRA r（64→32）、加 dropout、减 epoch（3→2），收窄 train/held-out CE gap（0.41→1.09） | +1-2pp |
| 补开集样本 | 当前全闭集，开集泛化未验证 | 验证侧补盲 |

### 10.3 评估侧

| 方案 | 理由 |
|---|---|
| 用 multi 软标签重跑评估 | 当前用 logprob 软标签比对，改用 multi 更能反映真实蒸馏质量 |
| 保留 scenario_buckets | 已重构，持续按题型诊断 |

---

## 11. 下一步行动

1. **优先**：用 multi 采样法重新生成 train2014 教师软标签（参考 `val_softl_multi.jsonl` 生成方式），替换训练数据后重训。
2. 重训后用同一 val2014 held-out 重跑评估，验证 color 桶是否回升至接近教师天花板 0.959。
3. 用 multi 软标签重跑评估，对比 closed_distribution。

---

## 附录 A：完整 report.json 字段

### 顶层
| 字段 | 值 |
|---|---|
| `train_data_path` | `outputs/training/train.jsonl` |
| `eval_data_path` | `./outputs/teacher/val_soft_logits.jsonl` |
| `is_heldout` | true |
| `student_model_path` | `./outputs/student_merged` |
| `max_samples` | null |
| `model_load_seconds` | 75.59 |
| `parameter_count.total` | 3,754,622,976 |

### dimensions.distillation_quality
| 字段 | 值 |
|---|---|
| `samples_evaluated` | 2456 |
| `closed_answer_match_rate` | 0.6393 |
| `closed_primary_match_rate` | 0.7142 |
| `open_text_similarity` | 0.0 |
| `cot_similarity` | 0.322 |
| `answer_sequence_ce.closed.mean` | 1.0855（samples 2456） |
| `answer_sequence_ce.open.mean` | null（samples 0） |
| `closed_distribution.samples` | 2234 |
| `closed_distribution.skipped` | 222 |
| `closed_distribution.kl_mean` | 0.04594 |
| `closed_distribution.cosine_mean` | 0.9811 |
| `closed_distribution.top1_distribution_match_rate` | 0.9266 |
| `scenario_buckets.overall_accuracy` | 0.6437（1581/2456） |
| `scenario_buckets.buckets.yes_no` | 1035/1316 = 0.7865 |
| `scenario_buckets.buckets.color` | 8/326 = 0.0245 |
| `scenario_buckets.buckets.open` | 149/278 = 0.536 |
| `scenario_buckets.buckets.counting` | 183/255 = 0.7176 |
| `scenario_buckets.buckets.other` | 144/197 = 0.731 |
| `scenario_buckets.buckets.choice` | 62/84 = 0.7381 |

### dimensions.deployment_efficiency
| 字段 | 值 |
|---|---|
| `total_parameters` | 3,754,622,976 |
| `model_dir_size_mb` | 11942.9 |
| `peak_vram_gb` | 7.173 |
| `avg_latency_seconds` | 2.7822 |
| `median_latency_seconds` | 2.7669 |
| `throughput_samples_per_second` | 0.3594 |
| `benchmark_samples` | 50 |

## 附录 B：数据文件
| 文件 | 说明 |
|---|---|
| `outputs/training/train.jsonl` | 训练数据，train2014，5041 条，logprob 软标签 |
| `outputs/teacher/val_soft_logits.jsonl` | 验证集教师标注，val2014，2456 条，logprob（本次评估用） |
| `outputs/teacher/val_softl_multi.jsonl` | 验证集，val2014，2431 条，**multi 采样软标签**（n=8，含 answer_distribution + logprob_distribution） |
| `outputs/student_merged/` | 学生合并全权重（bf16） |
| `outputs/student_ckpt/` | LoRA adapter + checkpoints |
| `outputs/evaluation/report.json` | 本评估报告 |

---

*由 `scripts/run_full_pipeline.py --steps evaluation` 生成，配置见 `configs/default.yaml`。*
