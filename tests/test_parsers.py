import pytest

from dpc_bench.tasks.dataset_mixing import DatasetMixingTask
from dpc_bench.tasks.instruction_generation import InstructionGenerationTask
from dpc_bench.tasks.quality_filtering import QualityFilteringTask
from dpc_bench.tasks.semantic_dedup import SemanticDedupTask


def test_semantic_dedup_parse():
    t = SemanticDedupTask()
    assert t.parse_prediction({"duplicate": "yes"}) == {"duplicate": "YES"}
    with pytest.raises(ValueError):
        t.parse_prediction({"duplicate": "maybe"})


def test_quality_filtering_parse():
    t = QualityFilteringTask()
    assert t.parse_prediction({"decision": "KEEP"})["decision"] == "keep"


def test_instruction_parse():
    t = InstructionGenerationTask()
    out = t.parse_prediction(
        {"instruction": " Q ", "input": None, "output": " A long enough answer here "}
    )
    assert out["instruction"] == "Q"
    assert out["input"] == ""


def test_mixing_parse_accepts_flat_or_nested():
    t = DatasetMixingTask()
    assert t.parse_prediction({"ratios": {"Math": 50.0, "Code": 50.0}})["ratios"]["Math"] == 50.0
    assert t.parse_prediction({"Math": 40, "Code": 60})["ratios"]["Code"] == 60.0
