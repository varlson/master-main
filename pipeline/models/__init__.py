"""Wrappers para os modelos ativos do projeto."""

from models.DCRNN import DCRNN
from models.DGCRN import DGCRN
from models.MTGNN import MTGNN
from models.PatchSTG import PatchSTG
from models.STICformer import STICformer
from models.WaveNet import GraphWaveNet

__all__ = [
    "DCRNN",
    "DGCRN",
    "MTGNN",
    "PatchSTG",
    "STICformer",
    "GraphWaveNet",
]
