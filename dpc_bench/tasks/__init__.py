"""Task registry."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import Task
from .corpus_dedup import CorpusDedupTask
from .corpus_filtering import CorpusFilteringTask
from .curriculum_scheduling import CurriculumSchedulingTask
from .data_utility_prediction import DataUtilityPredictionTask
from .dataset_mixing import DatasetMixingTask
from .diversity_selection import DiversitySelectionTask
from .format_repair import FormatRepairTask
from .hard_sample_generation import HardSampleGenerationTask
from .instruction_generation import InstructionGenerationTask
from .quality_filtering import QualityFilteringTask
from .quality_ranking import QualityRankingTask
from .reasoning_generation import ReasoningGenerationTask
from .semantic_dedup import SemanticDedupTask

TASK_CLASSES: Dict[str, Type[Task]] = {
    SemanticDedupTask.name: SemanticDedupTask,
    QualityFilteringTask.name: QualityFilteringTask,
    CorpusFilteringTask.name: CorpusFilteringTask,
    FormatRepairTask.name: FormatRepairTask,
    QualityRankingTask.name: QualityRankingTask,
    DiversitySelectionTask.name: DiversitySelectionTask,
    InstructionGenerationTask.name: InstructionGenerationTask,
    ReasoningGenerationTask.name: ReasoningGenerationTask,
    HardSampleGenerationTask.name: HardSampleGenerationTask,
    DatasetMixingTask.name: DatasetMixingTask,
    CurriculumSchedulingTask.name: CurriculumSchedulingTask,
    DataUtilityPredictionTask.name: DataUtilityPredictionTask,
    CorpusDedupTask.name: CorpusDedupTask,
}


def list_tasks() -> List[str]:
    return list(TASK_CLASSES.keys())


def get_task(name: str) -> Task:
    key = name.strip().lower()
    if key not in TASK_CLASSES:
        raise KeyError(f"unknown task {name!r}; choose from {list_tasks()}")
    return TASK_CLASSES[key]()
