# ProteinMPNN Core Architecture
# Inverse protein folding: 3D structure → amino acid sequence
# Dauparas et al., Science 2022

from core_architecture.model import ProteinMPNN
from core_architecture.config import Config

__all__ = ["ProteinMPNN", "Config"]
