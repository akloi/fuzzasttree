"""astshannon: inverse-generative anomaly matching and mutation for javac fuzzing.

Self-contained port of the astsubtree framework onto the ast_shannon
multi-dimensional anomaly pattern library (5028 role-labeled pattern graphs,
18 typed semantic relations, KL/IDF feature weights, DIE guardrails).
"""

from .library import ShannonLibrary
from .matcher import ShannonMatcher
from .models import FusionMutationRequest, FusionPlan, HostRegion, MatchResult, MutationRequest, ShannonPattern
from .mutator import (
    CommandSnippetMutator,
    HistoricalSnippetMutator,
    MutationOutput,
    OpenAICompatibleSnippetMutator,
    SnippetMutationError,
    SnippetMutator,
)
from .parser import JavaSyntaxTree, parse_java, syntax_error_count
from .pipeline import MutationPipeline, PipelineOptions
from .relations import HostAnalysis

__version__ = "1.0.0"

__all__ = [
    "ShannonLibrary",
    "ShannonMatcher",
    "ShannonPattern",
    "HostRegion",
    "MatchResult",
    "MutationRequest",
    "FusionMutationRequest",
    "FusionPlan",
    "MutationPipeline",
    "PipelineOptions",
    "HostAnalysis",
    "JavaSyntaxTree",
    "parse_java",
    "syntax_error_count",
    "SnippetMutator",
    "HistoricalSnippetMutator",
    "CommandSnippetMutator",
    "OpenAICompatibleSnippetMutator",
    "MutationOutput",
    "SnippetMutationError",
]
