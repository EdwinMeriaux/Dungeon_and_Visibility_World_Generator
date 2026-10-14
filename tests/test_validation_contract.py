"""Validation must detect corrupt sensing artifacts, not only geometry errors."""

import numpy as np
import pytest
from wgv.validation import validate



@pytest.mark.parametrize("artifact", ["visible_target_indices.npy", "targets.npy",
                                    "visibility_continuous.csr"])
def test_saved_visibility_contract_rejects_corruption(sensed_world, artifact):
    folder = sensed_world
    validate(folder)
    path = folder/artifact
    if artifact.endswith(".csr"):
        path.write_bytes(path.read_bytes()[:-8])
    elif artifact == "targets.npy":
        points = np.load(path)
        points[[0, 1]] = points[[1, 0]]
        np.save(path, points)
    else:
        np.save(path, np.array([], dtype=np.int64))
    with pytest.raises((AssertionError, ValueError)):
        validate(folder)
