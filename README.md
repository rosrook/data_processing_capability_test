# Data Processing Capability Bench (v0.1)

多模型粗评：LLM-only 数据处理能力（Cleaning / Filtering / Synthesis / Mixture / Evaluation）。

对应设计文档 [design.md](design.md) 的 **全部 11 个任务 + Level-1** 快评（无真训练、默认无 LLM-as-judge）。

## 覆盖任务

| Task | Ability | Primary metric |
| ---- | ------- | -------------- |
| `semantic_dedup` | 成对语义去重 | F1 |
| `quality_filtering` | 单条质量过滤 keep/remove | F1 |
| `corpus_filtering` | 脏语料批次过滤（集合召回） | mean_recall |
| `corpus_dedup` | 多文档语料近重删除 | mean_pair_recall |
| `format_repair` | 修复破损 JSON 样本 | mean_score |
| `quality_ranking` | 多样本质量排序 | mean Spearman |
| `diversity_selection` | 多样性子集选择 | mean_score |
| `instruction_generation` | 指令样本生成 | mean_score |
| `reasoning_generation` | 推理链数据（GSM8K 答题） | mean_score / answer_accuracy |
| `hard_sample_generation` | 难例构造 | mean_score |
| `dataset_mixing` | 数据配比 | mean_score |
| `curriculum_scheduling` | 课程顺序 | mean_score |
| `data_utility_prediction` | 样本训练价值预测 | Spearman（归一化） |

补 / 重建 seed（从 HuggingFace 拉取或复用 `data/hf_raw/`）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 全部 11 个任务
python scripts/prepare_seed_from_hf.py --tasks all --limit-per-task 400
# 仅新任务
python scripts/prepare_remaining_seeds.py --limit-per-task 400
```

说明：种子主体来自 HF（PAWS/MRPC、FineWeb-Edu、Alpaca、GSM8K、Hendrycks MATH 等）。
仅 `format_repair` 会在 Alpaca 真样本上做本地 JSON 破坏；`curriculum` 难度用 GSM8K 解答步数启发式。

## 安装

```bash
cd data_processing_capability_test
pip install -e ".[dev]"
cp .env.example .env
cp models.example.yaml models.yaml
```

在 `.env` 里填入各厂商 API key；在 `models.yaml` 里用 `api_key_env` 引用这些变量。

## 跑多模型摸底

```bash
# 成本粗估：calls ≈ (#models) × (#tasks) × (--limit)
# 例：3 模型 × 4 任务 × 40 ≈ 480 次调用
dpc-bench run --models-file models.yaml --task all --limit 40 --output-dir output/sweep1

# 多进程并发（按 model×task 分进程；默认 resume，可断点续跑）
# 建议 jobs ≈ 模型数或 5–10；同目录再次启动会跳过已完成样本
dpc-bench run --models-file models.yaml --task all --limit 200 \
  --jobs 5 --output-dir output/sweep_memtensor_l200 -v
```

产出：

- `output/sweep1/<model>/<task>/predictions.jsonl`
- `output/sweep1/<model>/<task>/metrics.json`
- `output/sweep1/summary.json`
- `output/sweep1/comparison.md` ← 模型 × 任务主指标表

仅重算对比表：

```bash
dpc-bench compare --output-dir output/sweep1
```

离线冒烟（不调 API）：

```bash
dpc-bench run --dry-run --task all --limit 10 --output-dir output/smoke
```

单模型（用 `.env` 里的 `DPC_BENCH_LLM_*`）：

```bash
dpc-bench run --task semantic_dedup --limit 20 --output-dir output/single
```

## models.yaml 示例

```yaml
models:
  - name: gpt-mini
    backend: openai_compat
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: gpt-4.1-mini
  - name: qwen-plus
    backend: openai_compat
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: QWEN_API_KEY
    model: qwen-plus
```

`backend`：

- `memtensor`（推荐）：memtensor Chat Completions（`MODEL_gpt_memtensor_URL` + `KEY`，内部走 openai_compat）
- `openai_compat`：通用 OpenAI-compatible `/chat/completions`
- `gateway`：公司 Gateway Responses API（`model.memtensor.cn`）

默认请在 `.env` 填写：

```bash
DPC_BENCH_LLM_BACKEND=memtensor
MODEL_gpt_memtensor_URL=https://api-int.memtensor.cn/v1
MODEL_gpt_memtensor_KEY=...
GPT_API_MODEL=gpt-4o-2024-11-20
```

连通性冒烟：

```bash
python scripts/smoke_memtensor.py
dpc-bench run --task semantic_dedup --limit 2 --output-dir output/memtensor_smoke
```

## Seed 数据（公开 HF 全量）

默认拉取各任务对应 HF 数据的**完整文件/分片**，并物化为 `data/seed/*.jsonl`（详见 `data/seed/SOURCES.json`）。

| Task | Source（全量） |
| ---- | -------------- |
| `semantic_dedup` | PAWS `labeled_final` train+val+test + GLUE MRPC train+val+test |
| `quality_filtering` | FineWeb-Edu llama3 annotations **全部 8 个 shard**（score≥3 → keep） |
| `instruction_generation` | `yahma/alpaca-cleaned` 全量 JSON |
| `dataset_mixing` | 全量域池（Alpaca / GSM8K main / HumanEval / annotations≥3）+ 配比场景；域池在 `data/seed/domain_pools/` |

手写版备份：`data/seed_handwritten_backup/`。原始下载：`data/hf_raw/`。

```bash
pip install -e ".[hf,dev]"
export HF_ENDPOINT=https://hf-mirror.com
# 全量拉数 + 全量写 seed（约需数 GB 磁盘；annotations ~1.6GB）
python scripts/prepare_seed_from_hf.py

# 若只想生成较小 seed 文件（可选）
python scripts/prepare_seed_from_hf.py --limit-per-task 400
```

评测时可选择是否全量跑：

```bash
# 全量（不加 --limit）
dpc-bench run --models-file models.yaml --task all --output-dir output/full

# 抽样摸底
dpc-bench run --models-file models.yaml --task all --limit 50 --output-dir output/smoke
```

## 测试

```bash
pytest
dpc-bench run --dry-run --task all --limit 5 --output-dir output/smoke
```
