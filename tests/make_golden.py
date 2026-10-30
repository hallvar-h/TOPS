"""Write tests/data/v040_k2a.npz. Run once, at tag v0.4.0, never by CI:

    uv run python tests/make_golden.py
"""
from pathlib import Path

import numpy as np

from golden_scenarios import snapshot

if __name__ == '__main__':
    path = Path(__file__).parent / 'data' / 'v040_k2a.npz'
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **snapshot())
    print(f'wrote {path}')
