from __future__ import annotations

import sys
from pathlib import Path


def ensure_perceiver_vendor_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "models" / "perceiver-io"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    return vendor_root

