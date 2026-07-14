下面我给你整理一版**可以直接开始搭建的 Benchmark Design 文档**。我会按照工程落地思路设计，而不是继续扩展任务数量。

设计原则：

1. **只覆盖 Text-only LLM 数据处理能力**
2. **每个任务必须对应一个明确的数据处理决策**
3. **每个任务必须有可获得的数据来源**
4. **每个任务必须有明确评测指标**
5. **尽量优先选择已有公开数据，不自己造 benchmark**

---

# Data Processing Capability Benchmark (LLM-Only)

## 1. Benchmark Goal

目标：

评估 LLM 在数据处理生命周期中的核心能力：

> 从原始数据 → 高质量训练数据 → 优化训练策略

覆盖：

* 数据清洗（Cleaning）
* 数据筛选（Filtering）
* 数据生成（Synthesis）
* 数据组织与配比（Mixture）
* 数据价值评估（Evaluation）

统一抽象：

```
Input Data
    +
Task Objective / Constraint
        |
        v
LLM Data Processing Decision
        |
        v
Evaluation
```

---

# 2. Task Overview

| Category   | Task                      | Core Ability | Decision Type      |
| ---------- | ------------------------- | ------------ | ------------------ |
| Cleaning   | Semantic Deduplication    | 判断重复数据       | Binary             |
| Cleaning   | Quality Filtering         | 判断数据价值       | Binary/Score       |
| Cleaning   | Format Repair             | 修复结构错误       | Generation         |
| Filtering  | Quality Ranking           | 样本排序         | Ranking            |
| Filtering  | Diversity Selection       | 子集选择         | Selection          |
| Synthesis  | Instruction Generation    | 生成训练样本       | Generation         |
| Synthesis  | Reasoning Data Generation | 生成推理数据       | Generation         |
| Synthesis  | Hard Sample Generation    | 难例构造         | Generation         |
| Mixture    | Dataset Mixing            | 数据配比优化       | Allocation         |
| Mixture    | Curriculum Scheduling     | 数据顺序优化       | Scheduling         |
| Evaluation | Data Utility Prediction   | 预测训练价值       | Regression/Ranking |

---

# Part I. Data Cleaning

---

# Task 1. Semantic Deduplication

## 目标

判断两个文本是否语义重复。

测试能力：

* embedding理解
* paraphrase识别
* 表面不同但语义一致判断

---

## Dataset

### Primary

**C4 duplicate detection subset**

来源：

* C4 dedup metadata

或者：

构造：

* Common Crawl文本
* 人工生成paraphrase pair

---

## Input

```
Text A:
...

Text B:
...
```

---

## Output

```
duplicate:
YES / NO
```

---

## Evaluation

### Main

F1

```
Precision
Recall
F1
```

### Additional

Hard negative accuracy:

* 同主题但不同含义
* 相似词但不同事实

---

# Task 2. Quality Filtering

## 目标

判断文本是否应该进入训练集。

---

## Dataset

### Primary

FineWeb quality filtered data

构造：

```
high-quality samples
+
low-quality samples
```

---

## Input

```
Text:
...
```

---

## Output

```
keep / remove
```

---

## Evaluation

### Classification

* Accuracy
* F1
* AUROC

### Training impact

两个数据集：

```
raw dataset
filtered dataset
```

训练同一个模型：

比较：

* validation loss
* downstream accuracy

---

# Task 3. Format Repair

## 目标

修复训练数据格式错误。

例如：

JSON instruction dataset:

Before:

```json
{
"instruction":
"xxx",
"output":
}
```

After:

```json
{
"instruction":"xxx",
"output":"xxx"
}
```

---

## Dataset

来源：

* Alpaca format
* ShareGPT format

人工注入：

* missing field
* wrong format
* inconsistent schema

---

## Evaluation

### Structural

* JSON validity
* schema accuracy

### Semantic

repair后：

* instruction meaning preservation

---

# Part II. Data Filtering

---

# Task 4. Quality Ranking

## 目标

给多个数据样本排序。

---

## Dataset

FineWeb / RedPajama samples

---

## Input

```
Sample A
Sample B
Sample C
...
```

---

## Output

ranking:

```
A > C > B
```

---

## Evaluation

* Spearman correlation
* Kendall Tau

---

# Task 5. Diversity Selection

## 目标

从大规模数据中选择覆盖最大的数据子集。

---

## Dataset

Instruction dataset:

* Alpaca
* Dolly

---

## Input

```
1000 samples

budget:
100 samples
```

---

## Output

selected subset

---

## Evaluation

### Diversity

* embedding coverage
* cluster coverage

### Training

train on:

random subset

vs

selected subset

metric:

Δ performance

---

# Part III. Data Synthesis

---

# Task 6. Instruction Generation

## 目标

生成高质量 instruction-response 数据。

---

## Dataset

参考：

* Self-Instruct
* Alpaca

---

## Input

```
Topic:
Machine Learning

Constraints:
difficulty=medium
requires reasoning
```

---

## Output

```
instruction
input
output
```

---

## Evaluation

### Quality

LLM judge:

* correctness
* usefulness
* difficulty

### Diversity

* embedding entropy

### Downstream

SFT:

generated data

vs

baseline data

---

# Task 7. Reasoning Data Generation

## 目标

生成具有可靠推理链的数据。

---

## Dataset

* GSM8K
* MATH

---

## Input

```
Target capability:
multi-step arithmetic reasoning
```

---

## Output

```
question
solution
reasoning
```

---

## Evaluation

### Correctness

* symbolic checker
* answer matching

### Reasoning

LLM judge:

* logical completeness

### Training

SFT:

generated reasoning data

→ GSM8K accuracy

---

# Task 8. Hard Sample Generation

## 目标

生成能暴露模型弱点的数据。

---

## Dataset

* GSM8K hard
* MATH

---

## Input

```
Capability:
fraction reasoning
```

---

## Output

hard samples

---

## Evaluation

Difficulty:

compare:

baseline model accuracy

```
easy set accuracy

vs

generated set accuracy
```

Quality:

human/LLM judge

---

# Part IV. Data Mixture

---

# Task 9. Dataset Mixing Optimization

## 目标

找到最佳训练数据比例。

---

## Dataset

固定：

| Data        | Size |
| ----------- | ---- |
| Instruction |      |
| Math        |      |
| Code        |      |
| Knowledge   |      |

---

## Input

```
Target:
maximize reasoning ability

Budget:
100k samples
```

---

## Output

```
Instruction 40%
Math 30%
Code 20%
...
```

---

## Evaluation

真正训练：

```
mixture A
mixture B
```

SFT后：

* MMLU
* GSM8K
* HumanEval

---

# Task 10. Curriculum Scheduling

## 目标

决定训练顺序。

---

## Dataset

GSM8K:

按照difficulty划分：

* easy
* medium
* hard

---

## Input

```
dataset

training steps
```

---

## Output

schedule:

```
easy
medium
hard
```

---

## Evaluation

训练曲线：

* convergence speed
* final accuracy
* stability

---

# Part V. Data Evaluation

---

# Task 11. Data Utility Prediction

## 目标

预测一条数据是否值得训练。

---

## Dataset

构造：

不同质量数据：

训练模型：

记录：

sample contribution

---

## Input

```
training sample
```

---

## Output

```
utility score
```

---

## Evaluation

Correlation:

prediction

vs

actual Δ performance

Metrics:

* Spearman
* Pairwise accuracy

---

# 3. Benchmark Execution Pipeline

整体流程：

```
             Dataset
                |
                v
        Task-specific Prompt
                |
                v
             LLM
                |
                v
       Structured Output
                |
                v
          Evaluation
                |
        +-------+-------+
        |               |
    Direct Metric   Training Impact
```

---

# 4. Evaluation Priority

由于不同任务成本不同，建议分三级：

## Level 1: Fast Evaluation

所有模型跑：

* F1
* Accuracy
* Ranking correlation
* Validity

---

## Level 2: Semantic Evaluation

LLM judge：

* correctness
* usefulness
* diversity

---

## Level 3: Gold Evaluation

只针对关键任务：

训练模型：

* SFT
* downstream benchmark

---

# 5. 第一版建议实现顺序（非常重要）

不要一开始全部做。

建议 MVP：

## Phase 1（1-2周）

实现：

1. Semantic Dedup
2. Quality Filtering
3. Instruction Generation
4. Dataset Mixing

原因：

覆盖：

* cleaning
* filtering
* generation
* optimization

四个核心能力。

---

## Phase 2

加入：

5. Reasoning synthesis
6. Hard sample generation
7. Utility prediction

---

## Phase 3

加入：

8. Curriculum
9. Automated mixture optimization

---

这版可以作为你开始搭建的**benchmark v0.1设计文档**。后续扩展时，只需要往 task ontology 中增加任务，而不用推翻整体结构。
