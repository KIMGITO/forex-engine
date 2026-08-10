"""Research layer — walk-forward validation, chronological splits, and
leakage-safe parameter optimization.

THIS IS A BASELINE ESTABLISHMENT LAYER. It does NOT implement ML/AI, live
execution, or broker integration. Its purpose is to determine whether
strategies have a genuine statistical edge using transparent, causal,
reproducible methodology.
"""

from app.research.config import ResearchConfig
from app.research.dataset import PartitionedResearchRepository, sync_partition
from app.research.optimizer import GridSearchOptimizer, grid_space_to_candidates
from app.research.reports import build_research_report
from app.research.splits import make_time_split, split_frame
from app.research.walk_forward import (
    WalkForwardRunner,
    build_walk_forward_windows,
    run_walk_forward,
)

__all__ = [
    "GridSearchOptimizer",
    "PartitionedResearchRepository",
    "ResearchConfig",
    "WalkForwardRunner",
    "build_research_report",
    "build_walk_forward_windows",
    "grid_space_to_candidates",
    "make_time_split",
    "run_walk_forward",
    "split_frame",
    "sync_partition",
]