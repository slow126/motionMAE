"""
vector_representations package
==============================
Flow vector representation and coverage analysis tools.
"""

from .flow_vector import FlowVector, FlowVectorConfig
from .vector_utils import (
    load_vector_coverage,
    load_multiple_vector_coverage,
    get_vectors_from_json,
    save_vector_coverage,
)
from .vector_coverage import (
    compute_mmd,
    compute_containment_metrics,
    compute_kmeans_comparison,
    compare_vector_coverage,
)

__all__ = [
    'FlowVector',
    'FlowVectorConfig',
    'load_vector_coverage',
    'load_multiple_vector_coverage',
    'get_vectors_from_json',
    'save_vector_coverage',
    'compute_mmd',
    'compute_containment_metrics',
    'compute_kmeans_comparison',
    'compare_vector_coverage',
]

