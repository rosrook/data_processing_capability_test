import pytest

from dpc_bench.metrics.allocation import evaluate_allocation, score_allocation
from dpc_bench.metrics.classification import binary_classification_metrics, normalize_yes_no
from dpc_bench.metrics.generation import evaluate_instruction_generation, score_instruction_sample
from dpc_bench.metrics.corpus import score_corpus_dedup, score_corpus_filtering
from dpc_bench.metrics.ranking import spearman_corr
from dpc_bench.metrics.reasoning import score_reasoning_sample
from dpc_bench.metrics.repair import score_format_repair


def test_binary_f1_perfect():
    m = binary_classification_metrics(["YES", "NO", "YES"], ["YES", "NO", "YES"], positive="YES")
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_normalize_yes_no():
    assert normalize_yes_no("yes") == "YES"
    assert normalize_yes_no("NO") == "NO"
    assert normalize_yes_no("maybe") is None


def test_instruction_score():
    good = score_instruction_sample(
        {
            "instruction": "Explain gradient descent for beginners.",
            "input": "",
            "output": "First define the loss. Then take steps along the negative gradient. Therefore parameters update.",
        },
        requires_reasoning=True,
    )
    assert good["schema_ok"]
    assert good["score"] >= 0.9

    bad = score_instruction_sample({"instruction": "x", "output": "y"})
    assert bad["score"] < 0.5


def test_evaluate_instruction_generation_aggregate():
    rows = [
        {
            "prediction": {
                "instruction": "Write a short tip about Python lists.",
                "input": "",
                "output": "Use list comprehensions for concise transformations.",
            },
            "meta": {"requires_reasoning": False},
        },
        {"prediction": None, "error": "boom", "meta": {}},
    ]
    m = evaluate_instruction_generation(rows)
    assert m["n"] == 2
    assert m["parse_errors"] == 1
    assert 0 < m["mean_score"] < 1


def test_allocation_score():
    domains = ["Instruction", "Math", "Code", "Knowledge"]
    ref = {"Instruction": 25, "Math": 40, "Code": 20, "Knowledge": 15}
    perfect = score_allocation({"ratios": ref}, domains=domains, reference=ref)
    assert perfect["valid"]
    assert perfect["l1_distance"] == 0.0
    assert perfect["score"] == 1.0

    bad = score_allocation({"ratios": {"Instruction": 50}}, domains=domains, reference=ref)
    assert not bad["valid"]

    m = evaluate_allocation(
        [
            {
                "prediction": {"ratios": ref},
                "label": {"domains": domains, "reference": ref},
                "meta": {},
            }
        ]
    )
    assert m["mean_score"] == 1.0


def test_spearman_monotonic():
    assert spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_format_repair_score():
    target = {"instruction": "Say hi", "input": "", "output": "Hello"}
    detail = score_format_repair(target, target=target, required_keys=list(target))
    assert detail["schema_ok"]
    assert detail["score"] == 1.0


def test_reasoning_answer_match():
    detail = score_reasoning_sample(
        {
            "question": "What is 2+2?",
            "reasoning": "Step 1: first identify the addends. Step 2: add them. Therefore the result is 4.",
            "answer": "4",
        },
        gold_answer="4",
    )
    assert detail["answer_match"]
    assert detail["score"] >= 0.9


def test_corpus_filtering_recall():
    detail = score_corpus_filtering(
        {"keep": ["a", "b"]},
        universe=["a", "b", "c"],
        gold_keep=["a", "b"],
    )
    assert detail["recall"] == 1.0
    assert detail["precision"] == 1.0


def test_corpus_dedup_pair_recall():
    detail = score_corpus_dedup(
        {"remove": ["b"]},
        universe=["a", "b", "c"],
        clusters=[["a", "b"], ["c"]],
        gold_remove=["b"],
    )
    assert detail["pair_recall"] == 1.0
    assert detail["remove_recall"] == 1.0
