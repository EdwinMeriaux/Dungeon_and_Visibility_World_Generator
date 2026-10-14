import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

import json
import pytest
from wgv.generation import generate
from wgv.sampling import prepare
from wgv.visibility import compute


@pytest.fixture
def sensed_world(tmp_path):
    """Generate fresh evidence for sensing and corruption checks."""
    config = json.loads((PROJECT / "configs/showcase/2d_polygonal_1.json").read_text())
    folder = tmp_path / "world"
    generate(config, folder)
    prepare(folder, count=64, observers=3)
    compute(folder)
    return folder
