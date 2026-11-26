"""NumPy-only Pine capture parsing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_capture(path: Path) -> np.ndarray:
    chains: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            text = line.strip()
            if text.startswith("Chain"):
                if rows:
                    chains.append(np.vstack(rows))
                    rows = []
            elif text:
                row = np.fromstring(text, sep=" ", dtype=np.float32)
                if row.size != 128:
                    raise ValueError(
                        f"{path}:{line_number}: expected 128 tones, got {row.size}"
                    )
                rows.append(row)
    if rows:
        chains.append(np.vstack(rows))
    if not chains or len({chain.shape for chain in chains}) != 1:
        raise ValueError(f"invalid chain layout in {path}")
    return np.stack(chains)
