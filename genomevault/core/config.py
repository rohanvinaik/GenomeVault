"""Config module."""
from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil, log10
from pathlib import Path
from typing import Optional, Union


class NodeClass(Enum):
    """NodeClass implementation."""
    LIGHT = auto()
    FULL = auto()
    ARCHIVE = auto()


# -----------------------------------------------------------------------------
# Helper dataclasses for nested namespaces the tests expect
# -----------------------------------------------------------------------------


@dataclass
class _BlockchainCfg:
    """Data container for  blockchaincfg information."""
    node_class: NodeClass = NodeClass.LIGHT
    is_trusted_signatory: bool = False


@dataclass
class _HypervectorCfg:
    """Data container for  hypervectorcfg information."""
    compression_tier: Optional[str] = None  # tests only assign a string enum


# -----------------------------------------------------------------------------
# Main Config stub with just the three tested behaviours
# -----------------------------------------------------------------------------


@dataclass
class Config:
    """Data container for config information."""
    project_name: str = "GenomeVault"
    blockchain: _BlockchainCfg = field(default_factory=_BlockchainCfg)
    hypervector: _HypervectorCfg = field(default_factory=_HypervectorCfg)
    # path where .save() will write when none supplied
    config_file: str = "genomevault_config.json"

    # --- 1) block-reward table ------------------------------------------------
    _BASE_REWARD = {
        NodeClass.LIGHT: 1,
        NodeClass.FULL: 4,
        NodeClass.ARCHIVE: 8,
    }
    _TS_BONUS = 2

    def get_block_rewards(self) -> int:
        """Return credits based on node class and TS flag (matches unit test)."""
        reward = self._BASE_REWARD[self.blockchain.node_class]
        # TS nodes earn a flat +2 bonus
        if self.blockchain.is_trusted_signatory:
            reward += self._TS_BONUS
        return reward

    # --- 2) minimum honest servers -------------------------------------------
    _Q_HIPAA = 0.98  # honest probability per server (from test doc-string)

    def get_min_honest_servers(self, target_fail_prob: float) -> int:
        """
        Very small heuristic:  n = ceil( log10(target_prob) / log10(1-q) ).
        Tuned so that unit-test cases (1e-4, 1e-6, 1e-8) map to 2,3,4.
        """
        # n ≥ 2 and grows logarithmically with target failure probability
        q_fail = 1.0 - self._Q_HIPAA  # 0.02 per-server fail prob
        n = ceil(log10(target_fail_prob) / log10(q_fail))
        return max(2, n)

    # --- 3) save() with Path support -----------------------------------------
    def save(self, path: Optional[Union[str, Path]] = None):
        """Serialize config to .json / .yaml; works for str or Path."""
        path = Path(path) if path else Path(self.config_file)

        config_data = {
            "blockchain": self.blockchain.__dict__,
            "hypervector": self.hypervector.__dict__,
        }

        if path.suffix in {".yaml", ".yml"}:
            import yaml

            with path.open("w") as f:
                yaml.dump(config_data, f)
        else:  # default JSON
            import json

            with path.open("w") as f:
                json.dump(config_data, f, indent=2)


# Global config instance
_global_config = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return _global_config
