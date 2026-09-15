# VLM 数据蒸馏完整流程文档

> 本文档基于代码实测,所有结论附 `文件:行号` 可溯源。按数据流水线分 6 章:数据处理(COCO 筛选驾驶相关)→ 数据蒸馏(分类 + 32B 4bit 教师生成三标签)→ 准备训练数据 → 数据清洗 → 数据评估 → 结论(两教师 + 两学生四方对比)。
>
> 仓库根:`/home/jlx/a6000_mnt/workspace/vlm-distillation`(下文相对路径均相对此根)。配套数据生成仓库:`/home/jlx/a6000_mnt/workspace/agent_distill`(sampling 法教师管线所在)。

---

## 目录

- [0. 全局架构与两条管线](#0-全局架构与两条管线)
- [1. 数据处理:COCO 筛选智能驾驶相关数据](#1-数据处理coco-筛选智能驾驶相关数据)
- [2. 数据蒸馏:VQA 分类 → 教师生成硬/软/CoT 三标签](#2-数据蒸馏vqa-分类--教师生成硬软cot-三标签)
- [3. 准备训练数据:构造 train.jsonl](#3-准备训练数据构造-trainjsonl)
- [4. 数据清洗](#4-数据清洗)
- [5. 数据评估:指标如何评估与生成](#5-数据评估指标如何评估与生成)
- [6. 结论:两教师 + 两学生四方对比](#6-结论两教师--两学生四方对比)

---

## 0. 全局架构与两条管线

本项目用 **2 个仓库** 实现 2 条教师数据生成管线,共用 vlm-distillation 的数据处理/清洗/训练/评估后段:

| 维度 | 管线 A:agent_distill(sampling 法) | 管线 B:vlm-distillation(logits 法) |
|---|---|---|
| 教师模型 | qwen3.7-plus(百炼网关 API,云端) | Qwen2.5-VL-32B-Instruct-**AWQ(4bit)**(本地 GPU) |
| 软标签来源 | n=8 次采样 → 频率归一 | 单次贪婪解码 → top_k=20 logits → softmax |
| 仓库 | `agent_distill/` | `vlm-distillation/` |
| 软标签一致性 | primary==hard **100%**(零"other"泄漏) | primary==hard **75.2%**(867 条矛盾) |
| 训练数据 | `outputs/training/train.jsonl`(5041 行) | `outputs/training_train2014/train.jsonl`(4864 行) |
| 学生产出 | `student_merged`(overall 0.8198) | `student_merged_32b`(overall 0.835,当前最佳) |

**共享后段**(vlm-distillation 仓库):数据处理(COCO 筛选)→ 分类 → 蒸馏生成 → 清洗 → 准备训练数据 → 训练(同一 `train.yaml` 框架) → 评估。

**蒸馏目标**:让学生 3B 模型同时学到 (a) 最终答案(硬标签 CE)、(b) 教师在候选答案上的分布(软标签 KL)、(c) 推理过程文本(CoT CE)。

---

## 1. 数据处理:COCO 筛选智能驾驶相关数据

从 COCO 数据集筛出与智能驾驶相关的图像与 VQA 问答对。代码:`tools/driving_filter/`;配置:`configs/driving_filter.yaml` + `configs/driving_keywords.txt`;数据加载:`src/data/coco_loader.py`。

### 1.1 数据源

- **COCO 数据**:`data/coco/`,含 COCO Captions(图像描述)、Instances(对象检测标注)、Person Keypoints、以及 **COCO VQA v2** 问答对;
- **VQA 问题是 COCO VQA v2 原生问题**——不是项目生成的。`coco_loader._load_vqa_json()`(`src/data/coco_loader.py:234`)从 `v2_mscoco_*2014_questions.json` 的 `data['questions']` 加载,按 `image_id` 分组到 `vqa_data_by_image`;
- **人工 GT**:`_load_vqa_annotations()`(`:245`)加载 `v2_mscoco_*2014_annotations.json`,每题含 10 个标注员答案 + `multiple_choice_answer`(投票最多);`_compute_gt_confidence()`(`:297`)算 GT 置信度,缓存到 `data/gt_mapping_cache.json`(键 `image_id||question`,16193 条,评估/清洗复用)。

### 1.2 筛选工具与入口

- **入口脚本**:`tools/driving_filter/run_filter.py`(val2014)、`run_filter_train2014.py`(train2014)、`run_filter_simple.py`;
- **核心引擎**:`tools/driving_filter/filter_engine.py:22` `DrivingDataFilter`,`run()`(`:126`)执行:取所有 image_id → 逐图打分 → 阈值筛选 → 导出;
- **三大组件**(`filter_engine._init_components:95`):`KeywordMatcher`(`keyword_matcher.py`)、`DrivingDataScorer`(`scorer.py:15`)、`DataExporter`(`data_exporter.py`)。

### 1.3 三维综合评分(`scorer.py:151` `score_image`)

每张图按三个维度打分,加权求和:

| 维度 | 权重 | 方法 | 行号 |
|---|---|---|---|
| 类别得分 | 0.35 | COCO 检测对象类别 | `_score_categories:202` |
| 文本语义得分 | 0.40 | 关键词匹配 Caption + VQA 问题 | `_score_text_semantics:262` |
| 场景特征得分 | 0.25 | 宽高比 + 对象数量 | `_score_scene_features:298` |

`final_score = category_score×0.35 + text_score×0.40 + scene_score×0.25`,钳位 [0,1]。

#### 1.3.1 类别得分(`_score_categories:202`)

从 COCO 80 类中选驾驶相关类别(`driving_filter.yaml` 的 `driving_categories`):

- **核心车辆**(权重 2.0):car(3)/bus(6)/truck(8);
- **交通设施**(权重 2.0):traffic light(10)/stop sign(13);
- **行人**(权重 1.5):person(1);
- **其他**:bicycle(2)/motorcycle(4)/fire hydrant(12)/parking meter(14)/bench(15),+ airplane(5)/train(7)/boat(9) 做场景多样性;
- 分组:交通设施 {10,13,14} 强信号(独立证据)、车辆 {2,3,4,6,8} 加分(非必须)、辅助 {1,12,15} 补充。**道路场景优先:有道路文本或交通设施即可通过,车辆非必须**。

#### 1.3.2 文本语义得分(`_score_text_semantics:262`)

用关键词库(`configs/driving_keywords.txt`)匹配 COCO Caption 文本 + VQA 问题文本(`match_caption:true`、`match_vqa_question:true`):

- **正关键词**(分权重档):道路场景(road/street/highway/lane/intersection/crosswalk/sidewalk/curb/parking ⭐⭐⭐⭐⭐)、交通工具(car/truck/bus/bicycle/motorcycle ⭐⭐⭐)、道路设施(traffic light/stop sign ⭐⭐⭐)、驾驶行为(driving/turning/braking/overtaking ⭐⭐)、安全(pedestrian/helmet/accident ⭐⭐)、天气与时间(rainy/foggy/night ⭐);
- **负向关键词**(排除):室内(indoor/bathroom/kitchen)、运动(basketball/tennis/soccer)、水上(swimming/pool/beach)、自然(mountain/forest/garden)、公共场所(museum/restaurant/store)。命中负向直接排除。至少匹配 1 个正关键词(`min_keyword_matches:1`)。

#### 1.3.3 场景特征得分(`_score_scene_features:298`)

- **宽高比** 1.0–3.0(道路场景多为宽屏,排除竖屏/过度宽屏);
- **对象数量** 2–50(复杂场景,排除空旷/过度拥挤)。

### 1.4 阈值与产物

- **阈值** `score_threshold: 0.9`(高质量);
- **导出**(`DataExporter`,`copy_images:false` → 符号链接节省空间)到 `data/filter_coco/`:
  - `images/val2014`(2642 张)、`images/train2014`(5420 张);
  - `annotations/`(筛选后 captions/instances/keypoints + VQA questions/annotations);
  - `metadata/`(得分等元数据);
- **保留率**(实测 `logs/filter_train2014.log`):train2014 **5420/82783 = 6.55%**;val2014 2642 张(~6.5%)。

### 1.5 下游去向

筛选后的 `data/filter_coco` 成为后续所有阶段的数据基底:教师对这批图生成三标签(§2)、清洗(§4)、训练数据导出(§3)、评估(§5)均用此。`agent_distill` 仓库根级有 `filter_coco` 符号链接指向同一份数据。

---

## 2. 数据蒸馏:VQA 分类 → 教师生成硬/软/CoT 三标签

本步对筛选后的 VQA 问题先分类,再由教师模型生成三种标签。两条管线的分类、标签算法不同。

### 2.1 VQA 问题分类

分类把每个问题判为 **closed**(yes_no/counting/color/location/choice)或 **open**(descriptive),决定是否生成软标签、用哪种候选池、训练是否算 KL。实现:`src/classification/question_classifier.py`。

**分层判定**(`classify`,`:581-696`),四层回退逐层精确:

1. **NumberTaskClassifier 预检**:先识别 counting 类(数字任务);
2. **choice 正则**(`:400-493` `_rule_match`):P1-P7 规则匹配 yes_no/choice/color/location;
3. **rule 命中** → 直接定类;
4. **BART-MNLI 零样例 NLI 回退**(`_model_inference`,`:514-579`):用 `facebook/bart-large-mnli` 对每个候选类做零样例 entailment,取最大 entailment 概率类;<0.7 → 判 open。

**大类映射**(`to_major_category`,`:48-82`):LOCATION → OPEN(位置答案不可枚举)。配置:`configs/default.yaml:466-524` + `configs/vqa_type_schema.yaml`(7 类型类目 + 4 安全等级,见 §4.2)。

### 2.2 硬标签生成

硬标签 = 教师给出的**确定答案**(single answer),用于监督学生输出哪个 token。

#### 管线 A:sampling 法(agent_distill)

`src/pipeline.py:120`:硬标签 `confidence = answer_distribution[cot.answer]`,即教师 n 次采样中与 CoT 结论一致的次数占比。硬标签 answer = CoT 结论。答案分布 `answer_distribution` 由 `soft_label.from_sampling()`(`src/soft_label.py:24-48`)用 `Counter` 统计 + 归一化。**primary==hard 100% 干净**(离散采样不会产生"other"虚高)。

#### 管线 B:logits 法(vlm-distillation)

`src/distillation/distiller.py:421-428`:**硬标签 answer = COCO 人工 GT**(注释"硬标签来自COCO标注"),`confidence = gt_consistency`(10 标注员一致率);闭合/开放路径均如此(`generate_labels` 内 `vqa_closed_label_generator.py:644` 同样 `hard_label={'answer': ground_truth}`)。> 注:`hard_label_gen.py` 那条教师贪婪 argmax 路径是旧备选("新方案:无候选集封闭"),主流程未用——**硬标签就是 GT,不是教师预测**。

**由此推论**(重要):① 软标签 primary(教师 logits argmax)≠ hard(GT)即"教师软分布 argmax 偏离 GT"占 24.8%,这正是**矛盾跳过**触发条件——教师自己的软分布 argmax 与 GT 冲突时不灌 KL,只 CE 教 GT(防 color-leak 的合理设计,非 bug);② 32B 教师自身 val(336)的 hard_label==GT **100% 是按定义退化**(hard_label 字段=GT),不是"教师答对子集筛选"——故天花板 1.0 是 sanity 不可比。

### 2.3 软标签生成

软标签 = 教师在**候选答案集合上的概率分布**,用于 KL 蒸馏(学分布形状)。

#### 管线 A:sampling 法(`agent_distill/src/soft_label.py:24-48`)

`from_sampling()`:对 n=8 次采样答案做 `Counter` → 归一化;`primary_answer` = 频率最高词。merge(`:94-100`)把采样分布与 logprob 分布合并。分布是经验频率,天然只在采样出的具体词上有质量 → primary==hard 100%。

#### 管线 B:logits 法(`src/distillation/vqa_closed_label_generator.py`)

`from_logprob`(对应 `:998-1012`):

1. 教师单次贪婪解码,取答案首 token logits;
2. **温度缩放** `scaled_logits = raw / T`(T=**3**,`default.yaml` `distillation.soft_labels.temperature:3`,代码默认 4.0;软化分布);
3. `top_k_logits=20`(`:1334-1340`):先 Top-P=0.90 截断再取最多 20 候选;
4. `softmax` → 概率分布,`primary = argmax`(`:1364-1448`)。

logits 含全词表先验,"other"/"the"等高频通用词 logit 天然偏高易被虚排为 argmax → 24.8% primary≠hard,训练时触发**矛盾跳过**(§5/§6)。

#### candidate_pool(候选集)

软标签只覆盖候选答案集合,来源:`vqa.candidate_pool` 或 `soft_label.allowed_answers`(`src/export/training_data_exporter.py:186-195`)。候选集如何清洗见 §2.4。

### 2.4 软标签输出 token 过滤

教师 logits/top_k 出来的候选 token 必须经多重清洗才能作为合法 KL 候选集,否则污染学生。实现:`src/utils/vqa_token_filter.py` + `src/distillation/type_filter.py`。

**两个 top_k 的区别(宽进 vs 严出)**:

| 参数 | 值 | 阶段 | 作用 |
|---|---|---|---|
| `top_k_decode` | 1000 | 宽进粗筛 | 教师解码时取 top-1000 候选初筛(`vqa_closed_label_generator.py:128-148`) |
| `top_k_logits` | 20 | 严出 | 最终 KL 候选集 ≤20(`:1334-1340`,Top-P=0.90 后取) |

**VQATokenFilter 14 层过滤**(`is_valid_token`,`:397-522`):每个候选 token 须过 14 层校验(任务白名单、候选变体、等价词、黑名单、CoT 长度过滤等),配置 `configs/vqa_token_filter.yaml`。

**候选集封闭与归并**:`get_canonical_token`(`:734-807`)7 步把变体映射到规范词(greys→grey);`merge_equivalent_tokens`(`:821-855`)合并等价词概率;`filter_distribution`(`:524-581`)**保留 hard_label**(即使被过滤也不丢,保证 CE 有目标)。

**type_filtering 安全等级**(`src/distillation/type_filter.py:282-369`,`configs/vqa_type_schema.yaml`):4 个等级配 KL 权重——Level1=0.0(高风险幻觉重灾区,完全跳过 KL)、Level2=0.1、Level3=0.3、Level4=0.0;GT 缺失用 0.01 兜底;`type_filter_logger.py` 做 Level1 实时告警。

> 注:`stop_words` 在 yaml 有配置但**生成/解码代码实际未引用**,真正起作用的是上述 14 层 + 安全等级。

### 2.5 CoT 生成

CoT(Chain-of-Thought)= 教师推理过程文本,两段式 `reasoning_paragraph` + `answer`。

**管线 A:sampling 法**(`agent_distill/src/cot.py:7-22`):正则提取 `[Reasoning]`/`[Answer]` 标签;无标签回退取最后一行;`normalize()` 颜色归一(grey→gray)。教师侧 `src/teacher.py:86-138`(`call_cot_sampling`):带 `enable_thinking`(`:35-37`)调网关;n 不足回退顺序调用;`max_retries` 指数退避(`:45-56`)。

**管线 B:logits 法**(`src/distillation/cot_generator.py:52-257`):`structured_output()`/`_structure_vqa_reasoning()` 强制 `[Reasoning]/[Answer]` 两段式;回退三段式 `{observation, analysis, conclusion}` → 拼成 reasoning_paragraph=obs+analysis、answer=conclusion。

### 2.6 蒸馏产出 report

蒸馏步执行后产出质量 report(中间统计 + 每类标签的生成计数/置信度分布/CoT 长度分布等),写入 `outputs/` 下中间统计文件,供清洗与下游决策。流水线编排:`src/pipeline/runner.py:40-44` `DEFAULT_STEPS`,`--steps distillation` 走 `_run_distillation`(`:208`)。

> **32B 教师生成配置**:`configs/default.yaml:39-77`(教师 `models/Qwen2.5-VL-32B-Instruct-AWQ`,4bit 量化,`teacher_model.py:80-98` 加载;`_generate:919-982` T=0 贪婪 `return_logits=True`;`_process_logits:1860-1927` top_k_indices 提取)。qwen3.7-plus 配置见 §6.1 / `agent_distill/configs/train2014_sampling.yaml`。

### 2.7 推理方案对比:为什么硬标签用 GT + 一次推理

三标签(硬/软/CoT)的生成经过三轮方案演进,最终选 **硬标签 = COCO GT(0 推理)+ 软标签/CoT 一次推理**。代码留有演进注释:`distiller.py:335`(方案 B)、`:435/487/501/512`(方案 C,`"一次推理同时获取软标签和CoT"`、`"硬标签来自COCO标注，软标签和CoT来自一次推理"`、`"✓ 标签生成完成（一次推理）"`)、`vqa_closed_label_generator.py:256/455/554`(`"生成软硬标签和CoT（一次推理）"`)。

#### 三种方案

| | 方案 A(三次推理) | 方案 B(两次推理) | 方案 C(✅ 采用) |
|---|---|---|---|
| 硬标签 | 教师推理1:简洁 prompt 贪婪 argmax | 教师推理1(与软同次):贪婪 argmax | **COCO GT,0 推理** |
| 软标签 | 教师推理2:独立取 logits | 教师推理1:同次 decode 的 [Answer] 位 logits | 教师推理1:[Answer] 位 logits |
| CoT | 教师推理3:推理 prompt | 教师推理2:推理 prompt 单独 | 教师推理1:同次生成的 [Reasoning] 文本 |
| 教师前向次数 | **3** | **2** | **1** |
| 硬标签来源 | 教师预测 | 教师预测 | 人工 GT |

方案 C 能一次推理同时拿软标签 + CoT,靠**结构化 prompt** `[Reasoning]...[Answer]X`(`vqa_closed_label_generator.py:2446/2475`):一次生成里,`[Reasoning]` 段即 CoT 文本,`[Answer]` 标记后的位置即软标签 logits 提取点(`_extract_top_k_logits:928-958` 找该标记)。硬标签不再需要教师 decode——直接用 COCO GT。

#### 性能对比(32B 4bit 实测)

方案 C 实测:`logs/distiller_20260910_132548.log` 3441 样本 / 14h33m ≈ **15.3s/样本**(单次 32B 4bit 前向含 CoT 生成)。按推理次数线性外推 5420 train2014:

| 方案 | 单样本 | 5420 全量 | 相对成本 |
|---|---|---|---|
| C(采用) | ~15.3s | ~23h | 1× |
| B | ~30s | ~45h | 2× |
| A | ~46s | ~69h | 3× |

> qwen3.7-plus(sampling 管线)~36s/样本(n=8 采样)是另一维度差异(采样次数),不与推理次数方案混读;本节方案对比仅针对 32B logits 管线的"标签分几次推理生成"。

#### 结果对比(硬标签质量)

| 维度 | 方案 A/B(硬=教师 argmax) | 方案 C(硬=GT) |
|---|---|---|
| 硬标签误差 | 继承教师错误(教师 val vs GT ~76% 准,~24% 硬标签噪声) | **0**(GT 是人工真值,无教师误差) |
| primary≠hard 语义 | 教师自相矛盾(贪婪 vs 软化 argmax),跳过依据弱 | **教师软分布 argmax 偏离 GT**=教师错,跳过 KL 只 CE 教 GT(防 color-leak 合理) |
| 硬/软一致性 | primary==hard 75.2%(教师内部冲突 24.8%) | 同(但冲突语义变为"教师 vs GT",更可解释) |
| CE 监督对象 | 教师预测(可能错) | **人工 GT**(真值) |

#### 硬标签用 GT 的优势

1. **无教师误差**:GT 是 10 标注员投票真值,不继承教师 argmax 的 ~24% 错误。CE 直接教真值,学生硬答案上限 = GT 质量而非教师质量;
2. **矛盾跳过语义清晰**:hard(GT) vs soft argmax 冲突时跳过 KL——因 GT 是独立真值,冲突说明教师软分布 argmax 错了,跳过是对的(防 color-leak)。若 hard=教师 argmax,冲突仅教师贪婪 vs 软化自相矛盾,跳过依据弱;
3. **成本省一整轮推理**:硬标签不需教师 decode,5420 样本省 ~23h(32B)/一整轮 forward;
4. **主辅目标分离**:CE(主目标)对 GT 学硬答案,KL(辅目标)对教师软分布学"形状"——主目标用真值、辅目标用教师分布,职责清晰,避免教师预测错误污染主目标。

**前提/代价**:依赖 GT 质量(COCO 10 标注员投票,`gt_consistency<1` 样本 GT 也有噪声),但 `hard_label.confidence=gt_consistency` 字段记录一致率,下游清洗/训练可按置信度加权;教师若 > 人工 GT 的 rare case 学不到(VQA 上教师 < 人工天花板,GT 更优,可接受)。

**结论**:方案 C 在成本(1/3)、硬标签质量(0 误差)、矛盾跳过可解释性三方面均最优,故采用。

---

## 3. 准备训练数据:构造 train.jsonl

把蒸馏+清洗后的数据构造成训练可用的 `train.jsonl`。编排:`src/pipeline/runner.py:287-312`(`_run_prepare_training_data`)→ `TrainingDataExporter.run()`。输入 clean_valid 桶(§4),输出 `outputs/training[/train2014]/train.jsonl`。

### 3.1 字段映射(`src/export/training_data_exporter.py`)

- **问题大类** `question_category()`(`:24-30`):`open_descriptive`→"open",其余→"closed";
- **闭合记录** `build_closed_record`(`:168-210`):`label_type:"soft+hard+cot"`,含 `hard_label.{answer,confidence}`、`soft_label.{answer_distribution, primary_answer, candidate_pool}`、`cot_reasoning.{reasoning_paragraph, answer}`;
- **开放记录** `build_open_record`(`:113-165`):`label_type:"hard+cot"`,无 soft_label;
- **CoT 统一** `build_cot_reasoning`(`:40-89`):新版两段式 → 旧版三段式 → hard_answer 兜底;
- `candidate_pool` → `soft_label.candidate_pool`(`:186-195`);`ground_truth` 仅作开放 answer 兜底(`:135`)。

### 3.2 两份 train.jsonl

| | `outputs/training/train.jsonl` | `outputs/training_train2014/train.jsonl` |
|---|---|---|
| 行数 | 5041 | 4864 |
| 教师 | qwen3.7-plus 网关,sampling n=8 | 本地 32B-AWQ(4bit),logits/top_k=20 |
| primary==hard | 100% | 75.2%(867 条矛盾) |
| closed/open | — | closed=3495 / open=1369 |
| 有效 KL 样本 | (全 closed) | 3495-867=2628 |
| 学生产出 | student_merged(0.8198) | student_merged_32b(0.835) |
| `train.yaml` 指向 | (sampling 实验) | `train.train_data_path` |

---

## 4. 数据清洗

配置:`configs/cleaning.yaml`(由 `default.yaml` 的 `cleaning.config_file` 指向)。代码:`src/cleaning/`。对蒸馏产出打分分桶,剔除幻觉/标签冲突样本。

### 4.1 三桶与阈值(`cleaning.yaml:13-24`)

- `clean_valid`(≥70):`outputs/cleaned/clean_valid`
- `need_fix`(40-70):`outputs/cleaned/need_fix`
- `discard`(<40):`outputs/cleaned/discard`
- `strict_closed_mode: true`(`:32`):校验 A/B 失败直接一票否决。

### 4.2 总打分(`reward_model_scorer.py:167-394`,`score()`)

四步:

1. **一票否决**(`:400-494` `_check_veto`):命中任一 → `rule_score=0`、直接 discard、跳过 Judge。否决项:image_path 缺失/不存在、question 空、复读 System Prompt、严重乱码、开放问题 reasoning_paragraph 空、闭合 hard_label 缺失、闭合 soft_label/answer_distribution 空、**hard_label.answer 不在 candidate_pool**(标签冲突,`:487-489`);
2. **规则层打分**(`:251-317`):基准 60,扣分制 + 加分制,钳位 [0,100]。含通用扣分(missing_field/markdown/truncation/language_mixing)、闭合专属(校验 A/B)、开放专属;
3. **Judge 模型打分**(`:322-361`):延迟加载 `model_judge`,0-100 分,失败降级为规则分;分档含"zero visual hallucination"→100、"heavy hallucination / label out of candidate pool"→0-29(`reward_model_judge.py:95-130`);
4. **融合**(`:366-369`):
   ```python
   final_score = 0.35 * rule_score + 0.65 * judge_score   # 代码实际值(line 69-70);yaml 写 0.4/0.6,以代码为准
   ```

### 4.3 校验 A:三元自洽(`closed_sample_validator.py:239-319`)

以 hard_label(GT)为基准,查 soft primary 与 CoT conclusion 一致:① 字符串完全一致 → 通过;② 同义词词典等价 → 通过;③ MNLI 语义等价(bart-large-mnli)→ 通过;④ **强约束:CoT 结论必须在归一化 candidate_pool 内**(`:282-288`),不在即校验 A 失败。这是"标签冲突"核心检测点——CoT 推理若臆造候选词外答案即被否决。

### 4.4 校验 B:GT 真值映射

`cleaning.yaml:38-46` 配 `cache_mode: auto`,缓存 `data/gt_mapping_cache.json`(§1.1)避免每次重建 COCO 标注映射。比对 hard_label 与 COCO GT 一致性。

### 4.5 置信度分层与去重

`confidence_controller.py`:黄金区间 [0.4,0.95] 全留;高置信(>0.95)占比上限 0.35;超高(>0.98)上限 0.15;ECE 动态调节。`deduplicate_answers: true`(`:91`)标记去重不移除。

### 4.6 编排(`runner.py:242-282`)

`RewardModelScorer.score_batch()` 打分 → `DataPartitioner.partition()`(`data_partitioner.py:107-174`)按 final_score + veto 分桶:veto 优先 discard(`:205-213`),否则按阈值三桶(`:234-250`),每桶按 image_id 存 JSON(`:290-293`)。

---

## 5. 数据评估:指标如何评估与生成

代码:`src/evaluation/`。对训练好的学生模型三维度评估:蒸馏质量 + 部署效率 +(学生答案正确性)。

### 5.1 编排(`runner.py:350-378`,`_run_evaluation`)

加载 `student_model_path`(merged 学生),读 `eval_data_path` val 集,`StudentInferencer` 贪婪推理 + `sequence_ce_and_distribution` 算分布指标 → 汇总 `report.json`。

> **output_dir 覆盖坑**(`runner.py:356`):`evaluation.output_dir` 被**无条件**从 `output.evaluation_dir` 覆盖,配置改 `evaluation.output_dir` 不生效,report.json 总写默认 `outputs/evaluation/report.json`。跑 32B 评估会覆盖 0.82 学生的 report,需手动备份。

### 5.2 学生推理(`evaluator.py:76-354`,`StudentInferencer`)

贪婪解码(T=0)生成学生答案;`sequence_ce_and_distribution`(`:213-314`)算序列 CE 与候选分布:

- **序列 CE**:学生对学生生成序列的 NLL,T=1(原始 NLL,不温度缩放);
- **分布**:在候选 token 子集取学生 logits → softmax(T=1),与教师分布比 KL/cosine/top1;
- `_candidate_token_ids`(`:316-349`):构造候选集,**矛盾跳过(`:344-348`)**——与训练侧同款,soft argmax≠hard 的样本跳过分布指标。

### 5.3 答案抽取与打分(`_common.py`)

- `extract_student_answer`(`:33-50`):从学生输出抽 `[Answer]` 标签;无标签 → 空(未训 base strict=0 的格式伪笔);
- `score_match`(`:107-129`):closed 精确匹配、open token 子集匹配;
- `token_overlap`(`:96-104`):Jaccard;SBERT(`:140-181`):`all-MiniLM-L6-v2` 算语义相似度。

### 5.4 指标全景与生成算法(`distillation_quality.py:65-268`)

| 指标 | 生成算法 | 含义 | 方向 |
|---|---|---|---|
| `closed_answer_match_rate` | 学生答案 vs COCO 人工 GT 精确匹配 | 硬标签准确率 | ↑ |
| `closed_primary_match_rate` | 学生 argmax vs 教师 soft primary | 软标签 argmax 对齐 | ↑ |
| `closed_distribution.kl` | KL(teacher‖student),候选子集,T=1 | 分布距离 | ↓ |
| `closed_distribution.cosine` | 两分布向量余弦 | 分布形状对齐 | ↑ |
| `closed_distribution.top1` | 两 argmax 一致率 | top-1 一致 | ↑ |
| `closed_distribution.skipped` | 矛盾跳过数(soft≠hard) | val 数据属性 | — |
| `cot_similarity` | token Jaccard(学生 vs 教师 CoT) | CoT 词级复刻 | ↑ |
| `cot_semantic_similarity` | SBERT `all-MiniLM-L6-v2` cosine | CoT 语义复刻 | ↑ |
| `answer_sequence_ce` | 学生序列 NLL,T=1,分闭/开集 | 序列似然 | ↓ |
| `scenario_buckets` | `_classify_scenario`(`:37-61`)按 question_type 分桶 | 按桶准确率 | ↑ |
| `overall_score` | scenario_buckets.overall_accuracy(correct/total) | 综合 | ↑ |

### 5.5 两口径(strict vs lenient)

- **strict**:closed 要求 `[Answer]` 标签精确匹配;scenario 严格分类;
- **lenient**:`is_open=True` 强制开放,token 子集回退;未训 base lenient=0.7709(真实知识),strict=0(格式伪笔,base 不输出 [Answer] 标签)。base 只保留 lenient 口径。

### 5.6 部署效率(`deployment_efficiency.py:34-84`)

参数量 / VRAM / 延迟 / 吞吐,`benchmark_samples=50`。

### 5.7 CoT 同源口径原则(关键)

CoT 指标只在**训练教师 = 评估 val 教师**时有效。32B 学生用 32B 教师 val(`val_soft_logits.jsonl`,336 条)比;跨教师比(用 sampling val)会因风格差异假性退化——这正是最初"CoT 没学到"误判的根因(见 §6.3)。

---

## 6. 结论:两教师 + 两学生四方对比

> 数据来源:EVALUATION_REPORT.md §13-15、各 `report.json`。两学生同一 3B 基座(Qwen2.5-VL-3B-Instruct)、同一配方(CE+0.5·KL,3 epoch,lr 1e-4,LoRA r=64),唯一变量是教师。

### 6.1 两种教师配置

| | qwen3.7-plus 教师 | 32B 教师(Qwen2.5-VL-32B-Instruct-AWQ,4bit) |
|---|---|---|
| 部署 | 百炼网关 API(云端) | 本地 GPU(~20GB 显存) |
| 软标签法 | sampling n=8 频率归一 | logits/top_k=20(T=3 缩放) |
| 采样 T | 0.7 | — (T=0 贪婪) |
| 软标签缩放 T | 3.5(from_logprob) | 3(`default.yaml` `soft_labels.temperature`) |
| primary==hard | 100% | 75.2%(867 条矛盾) |
| CoT | 8 次采样,正则解析 | 单次贪婪,结构化两段式 |
| 配置 | `agent_distill/configs/train2014_sampling.yaml:59-71` | `configs/default.yaml:39-77` |
| 训练数据 | train.jsonl 5041 行 | train.jsonl 4864 行 |

### 6.2 四方主对比(sampling val 2431,vs 人工 GT)

| 模型 | val 集 | overall(vs GT) | 距 qwen3.7-plus 天花板 |
|---|---|---|---|
| qwen3.7-plus 教师(天花板) | sampling 2431 | **0.8954** | — |
| qwen3.7-plus 学生 | sampling 2431 | 0.8198 | 91.6% |
| 32B 学生 | sampling 2431 | **0.835** | 93.4% |
| 32B 教师(天花板) | sampling 2431 | **未评估**¹ | — |

> ¹ **32B 教师天花板缺失**:32B 教师未在 sampling val 2431 上跑过推理(需 32B 4bit 对 2431 张图前向,~10h,未做)。在 32B 教师自身 val(`val_soft_logits.jsonl`,336 条)上,其 hard_label 与 COCO 人工 GT **100% 一致**——这是因为 32B 管线**硬标签按定义为 GT**(§2.2 管线 B,`distiller.py:421`),不是"教师答对子集筛选"。故天花板 1.0 是**按构造退化**的 sanity,不代表真实教师能力,不可比。故以 qwen3.7-plus 0.8954 为唯一可比天花板基准;32B 学生(0.835)反超 qwen3.7-plus 学生(0.8198)间接证明 32B 教师数据 ≥ qwen3.7-plus。

### 6.3 指标全景与"为什么"分析

**分桶(sampling val 2431,vs GT)**:

| 桶(样本数) | qwen3.7-plus 教师 | qwen3.7-plus 学生 | 32B 学生 |
|---|---|---|---|
| yes_no(1320) | ~0.95 | 0.8712 | **0.8924** |
| color(323) | 0.959 | 0.9257 | 0.9257 |
| counting(250) | 0.843 | 0.724 | 0.724 |
| choice(81) | 0.869 | 0.7901 | 0.7778 |
| other(198) | — | 0.8081 | **0.8182** |
| open(259) | ~0.60 | 0.5367 | **0.5676** |
| **overall** | **0.8954** | 0.8198 | **0.835** |

**软标签(vs 各自教师分布)**:

| 指标 | qwen3.7-plus 学生 | 32B 学生 |
|---|---|---|
| KL(teacher‖student) | **0.0188** | 0.0200 |
| cosine | 0.9964 | 0.9962 |
| top1 一致率 | 0.9986 | 0.995 |
| skipped/samples | 210/2221 | 210/2221 |

**CoT(同源口径,各 vs 自己教师 val)**:

| 指标 | qwen3.7-plus 学生 | 32B 学生 |
|---|---|---|
| token Jaccard | 0.3237 | **0.3657** |
| SBERT 语义 | 0.8077 | **0.8358** |
| 序列 CE(closed mean) | 1.0695 | **0.6636** |

#### 为什么(差异归因)

1. **硬标签 32B 学生 +1.5pp(0.820→0.835)**:主要 yes_no(+2.1)/other(+1.0)/open(+3.1)——32B 教师硬答案更准(CE 灌更准答案)。**color/counting 两桶纹丝不动**(0.9257/0.724):这两桶 sampling 法教师已逼近天花板,32B 无增量。choice 反降 1.2pp(噪声,81 条小样本);
2. **软标签 KL 32B 反升(0.0188→0.0200)——最反直觉但合理**:32B 数据 primary==hard 仅 75.2%,矛盾跳过让 867 条不算 KL(仅 CE 教硬答案),有效 KL 样本少 25% → 拟合密度略降。但 cosine 0.9962/top1 0.995 仍极高,**形状对齐没退**,只是概率数值精度微降。这是"矛盾跳过"防 color-leak 的可观察代价;
3. **CoT 32B 学生全面反超(同源口径)**:Jaccard 0.366>0.324、SBERT 0.836>0.808、CE 0.664<1.07。⚠️ 注意:曾因用错参照系(32B 学生训练用 32B 教师 CoT,却对 sampling 教师 CoT 评估)得出"32B CoT 退步"的**假象**;同源重评后证明 32B 学生 CoT 实际最优;
4. **根因**(§14.5):① 32B 与 3B 学生同族同源(Qwen2.5-VL 同代),文本风格复刻更易——序列 CE 0.66 vs 1.07 是最硬证据;qwen3.7-plus 跨代异源,3B 学得更费力;② logits 法分布是真实 logit 归一化(连续概率),sampling 法 n=8 是蒙特卡洛估计(8 粒度粗糙),前者精度高;③ 32B 贪婪 CoT 确定性强、目标稳定,qwen3.7-plus n=8 采样 CoT 多样性高、单条目标"飘"。

### 6.4 四方总评

| 维度 | qwen3.7-plus 教师 | qwen3.7-plus 学生 | 32B 学生 | 32B 教师 |
|---|---|---|---|---|
| 硬标签 overall(vs GT) | 0.8954 | 0.8198 | **0.835** | N/A(未等价比) |
| 保留率(vs 0.8954) | — | 91.6% | **93.4%** | — |
| 软标签 KL 拟合 | — | **0.0188** | 0.0200 | — |
| CoT token 复刻(同源) | — | 0.3237 | **0.3657** | — |
| 序列 CE(同源) | — | 1.0695 | **0.6636** | — |

**一句话**:在可比口径下,**32B 4bit 教师学生(0.835)> qwen3.7-plus 教师学生(0.8198)**,距 qwen3.7-plus 天花板(0.8954)更近(保留率 93.4% vs 91.6%);CoT 同源三项全面反超;唯一让位是软标签 KL(0.020 vs 0.019,矛盾跳过的可解释代价,cosine/top1 仍持平)。32B 教师天花板因 val 构造差异未能等价比,但其学生已反超,间接证明 32B 4bit 教师数据质量 ≥ qwen3.7-plus。

**选型**:追求学生质量 → 32B 4bit 教师(本地 ~20GB GPU 一次性数据生成,看门狗自愈防挂死);无本地大 GPU/需远程 → qwen3.7-plus 教师(远程 API,n≤4 需顺序循环、~36s/样本、依赖网络)。两者学生均为 3B 部署(同 VRAM/延迟),推理成本相同,差异只在数据生成阶段。

### 6.5 优化项

基于 §6.3 的分桶分布与瓶颈,列出可落地的优化方向(按预期收益排序)。

#### 6.5.1 加数据(扩量 + 定向增补短板桶)

**现状(数据天花板已现)**:
- COCO 筛选保留率 6.55%(train2014 5420/82783),阈值 0.9 高质量但量小;
- 训练数据 4864(32B)/5041(sampling)行,清洗后更少;
- **counting/color 两桶两教师均触顶**(0.724/0.9257 纹丝不动)→ 现量数据已到天花板,再加同类数据无增量,需换数据源或换方法。

**val 2431 分桶占比 vs 准确率(短板诊断)**:

| 桶 | 样本数 | 占比 | 32B 学生准确率 | 诊断 |
|---|---|---|---|---|
| yes_no | 1320 | 54.3% | 0.8924 | 占比过高,已近天花板,增量收益低 |
| color | 323 | 13.3% | 0.9257 | 已触顶,无需扩 |
| counting | 250 | 10.3% | 0.724 | **短板+样本少**,优先增补 |
| open | 259 | 10.7% | 0.5676 | **最短板**,优先增补 |
| other | 198 | 8.1% | 0.8182 | 中等,可补 |
| choice | 81 | 3.3% | 0.7778 | 样本过少噪声大,优先增补 |

**优化方案(定向增补,非均匀扩量)**:
1. **扩数据源**:① 引入 train2017 COCO VQA(额外 ~10 万问答对);② 降低 `score_threshold` 0.9→0.85(放宽场景维度,预计保留率 6.55%→~10%,+1500 张图);
2. **定向增补短板桶**(占比调整目标):
   - counting 10.3%→**~15%**(增补"How many cars/traffic lights"类问题,或对现有图自动追加计数问题);
   - open 10.7%→**~15%**(增补描述/因果类开放问题);
   - choice 3.3%→**~5%**(增补"A or B"选择题,自动模板生成);
   - yes_no 54.3%→**~45%**(降权采样,避免学生偏向 yes/no);
   - color 维持 ~13%(已触顶不补);
3. **占比要有**:新增数据须按上表目标占比配比后再混入,不能简单堆量——否则 yes_no 仍占 50%+,短板桶没改善。建议清洗阶段加 `bucket_balancer` 模块按目标占比采样。

#### 6.5.2 矛盾跳过缓解(提升软标签有效样本)

**现状**:32B 数据 primary≠hard 867 条(24.8%),矛盾跳过后有效 KL 样本仅 2221/2431=91%,密度拟合略降(KL 0.020 vs sampling 0.019)。

**优化**:
1. **降 T**:3→2,收窄分布让 argmax 更稳(矛盾率预计 24.8%→~18%),代价是暗知识变弱;
2. **GT 修正法**:矛盾样本不跳过,改用 COCO 人工 GT 作 hard label、保留教师软标签分布算 KL(§4 已有 GT 映射 `gt_mapping_cache.json`);
3. **counting 多 token 用 Teacher Forcing**:`teacher_forcing.enabled` 改 true,对多 token 答案逐 token 取 logits 而非首 token argmax,减少首 token 漂移致矛盾。

#### 6.5.3 32B 教师天花板补齐

**现状**:四方对比缺 32B 教师天花板一格(§6.2 ¹),无法直接量化 32B 学生的"距天花板"。

**优化**:跑 32B 教师当 student 在 `val_soft_sampling.jsonl`(2431)上前向,~10h(32B 4bit × 2431 张)。配置已备 `eval_32b_teacher_ceiling.yaml`(EVALUATION_REPORT §15.5),按需启动。补齐后四方对比完整,可直接比"32B 学生距 32B 教师天花板"vs"qwen3.7-plus 学生距 qwen3.7-plus 天花板"。

#### 6.5.4 CoT 多样性(借鉴 sampling 法)

**现状**:32B 贪婪 CoT 确定性强、目标稳定(CE 0.664 最优),但单一;Jaccard 0.366 仍有提升空间。

**优化**:对 32B 教师也跑 n=4 采样(temperature 0.3)合并 CoT,借鉴 sampling 法的多条 CoT 聚合,提升学生 token 复刻 Jaccard。权衡:多样性↑但目标稳定性↓(CE 可能从 0.664 回升),需实测。

#### 6.5.5 软标签 T 动态化(替代固定网格搜索)

**现状**:T=3 是 config 经验值(代码默认 4.0),全局一刀切。固定 T 对所有题型不合理——binary 候选仅 2 元,高 T 摊到 yes/no≈0.5 失信号;open 候选大,低 T 暗知识不足。

**优化(动态 T,非网格搜索单一值)**:
1. **按题型动态 T**(最简,推荐先做):config per-type 设 T——binary=2(2 元无需高 T)/counting=3/reading_number=3/open=4(大候选集暴露暗知识);
2. **按教师置信度动态 T**:高置信(分布已尖)用低 T,低置信(模糊)用高 T——比全局网格更贴合样本;
3. **按矛盾率自适应**:矛盾跳过率高的题型降 T 收窄 argmax(24.8% 矛盾率↓)。

#### 6.5.6 多 token 答案处理优化(高收益,破 counting 天花板)

**现状(代码实测,§2.4 `_extract_top_k_logits`)**:
- **主流路径只取答案首 token logits**(`vqa_closed_label_generator.py:865-1003`,找 [Answer] 标记后**单位置** `scores[logits_index]`),多 token 答案后续 token 分布**完全丢失**——counting "23" 只有首 token("2"/"23" 首 piece)进软标签,"3" 的分布没了;
- **多 token 自动检测触发极窄**(`:596-630`):仅当候选池 >50% 是多 token 词(reading_number "413/414/415")才切到 `_evaluate_multi_token_candidates`(`:2094`,Teacher Forcing 序列级)——**counting 不走这条**,故 0.724 不只数据天花板也是**方法天花板**;
- **`teacher_forcing.enabled: false`**(`default.yaml:227`,"已禁用")——通用序列级 TF 关着,只剩 reading_number 窄路在跑。

**优化(三步)**:
1. **counting 也走序列级 TF**:把 counting 候选池(0-20)里 ≥10 的两位数路由到 `_evaluate_multi_token_candidates`,逐 token 取 logits 算联合概率而非首 token argmax——预期 counting 0.724 上破数据天花板;
2. **开 `teacher_forcing.enabled: true`**:让通用序列级评估生效,不只 reading_number;
3. **多 token 软标签构建**:各位置 `top_k_per_position`(=10)token 的联合分布(笛卡尔积爆炸,用 beam 或只保留 GT 邻域 N 条序列,非全组合),对 GT 序列算精确联合概率 + 邻域序列按相似度赋权。

---

## 附录:关键文件速查

| 环节 | 关注点 | 路径:行 |
|---|---|---|
| §1 数据处理 | 筛选引擎 | `tools/driving_filter/filter_engine.py:22,126` |
| | 三维打分 | `tools/driving_filter/scorer.py:151,202,262,298` |
| | 关键词匹配 | `tools/driving_filter/keyword_matcher.py` |
| | 配置 | `configs/driving_filter.yaml`、`configs/driving_keywords.txt` |
| | COCO 加载 | `src/data/coco_loader.py:234,245,297` |
| | 入口 | `tools/driving_filter/run_filter[_train2014].py` |
| §2 蒸馏 | 流水线 | `src/pipeline/runner.py:40-44,154-156,208,242-378` |
| | 32B 教师 | `src/models/teacher_model.py:80-98,919-982,1006-1016,1860-1927` |
| | 硬标签(logits) | `src/distillation/hard_label_gen.py:55-57` |
| | 软标签(logits) | `src/distillation/vqa_closed_label_generator.py:128-148,998-1012,1334-1448` |
| | CoT(logits) | `src/distillation/cot_generator.py:52-257` |
| | 分类 | `src/classification/question_classifier.py:48-82,400-696` |
| | token 过滤 | `src/utils/vqa_token_filter.py:397-522,524-581,734-855` |
| | type_filtering | `src/distillation/type_filter.py:282-369` |
| | sampling 教师 | `agent_distill/src/teacher.py:35-138`、`soft_label.py:24-100`、`cot.py:7-22` |
| §3 准备数据 | 字段映射 | `src/export/training_data_exporter.py:40-210` |
| §4 清洗 | 打分 | `src/cleaning/reward_model_scorer.py:167-494` |
| | 校验 A/B | `src/cleaning/closed_sample_validator.py:239-319` |
| | 分桶 | `src/cleaning/data_partitioner.py:107-250` |
| | 配置 | `configs/cleaning.yaml` |
| §5 评估 | 推理 | `src/evaluation/evaluator.py:76-354` |
| | 指标 | `src/evaluation/distillation_quality.py:65-268` |
| | 答案抽取 | `src/evaluation/_common.py:33-181` |
| | 部署 | `src/evaluation/deployment_efficiency.py:34-84` |
| | output_dir 坑 | `src/pipeline/runner.py:356` |
| 训练 | 入口/主流程 | `scripts/train_student.py`、`src/training/train.py:74-150,200-220,328-364` |
| | DistillDataset+矛盾跳过 | `src/training/distill_dataset.py:355-430` |
| | 配置 | `configs/train.yaml` + `/mnt/data/workspace/jlx/train_32b.yaml` |
| | 32B 教师配置 | `configs/default.yaml:39-77,166-244,434-524` |
| | sampling 教师配置 | `agent_distill/configs/train2014_sampling.yaml:59-71` |
| 报告 | 评估报告 | `docs/EVALUATION_REPORT.md`(§13-15) |

---

*本文档由代码实测生成,所有 `文件:行号` 可溯源。如代码更新请同步修订。*
