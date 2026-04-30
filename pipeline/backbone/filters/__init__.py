"""Wrappers de compatibilidade para filtros de backbone legados."""

from pipeline.backbone.filters.disparity_filter import DisparityFilter
from pipeline.backbone.filters.doubly_stochastic_filter import DoublyStochasticFilter
from pipeline.backbone.filters.glanb import GLANBFilter
from pipeline.backbone.filters.h_backbone import HBackboneFilter
from pipeline.backbone.filters.high_salience_skeleton import HighSalienceSkeleton
from pipeline.backbone.filters.marginal_likelihood import MarginalLikelihoodFilter
from pipeline.backbone.filters.noise_corrected import NoiseCorrectedFilter

__all__ = [
    "DisparityFilter",
    "DoublyStochasticFilter",
    "GLANBFilter",
    "HBackboneFilter",
    "HighSalienceSkeleton",
    "MarginalLikelihoodFilter",
    "NoiseCorrectedFilter",
]
