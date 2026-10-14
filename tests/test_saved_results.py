"""Fresh sensing outputs and CSR integrity checks."""
import json
import numpy as np
import pytest
from scipy.sparse import load_npz
from wgv.visibility import read_csr



def test_saved_visibility_rows_and_binary_replay_agree(sensed_world):
    folder=sensed_world
    matrix=load_npz(folder/"visibility_continuous.npz")
    binary=read_csr(folder/"visibility_continuous.csr")
    assert (matrix!=binary).nnz==0
    report=json.loads((folder/"visibility_continuous.json").read_text())
    prefixes=report["prefix_coverage"]
    assert prefixes==sorted(prefixes)
    union=np.asarray(matrix.sum(axis=0)).ravel()>0
    assert np.isclose(union.mean(),report["coverage_fraction"])
    assert np.array_equal(np.load(folder/"visible_target_indices.npy"),np.flatnonzero(union))
    assert np.array_equal(np.load(folder/"unobserved_target_indices.npy"),np.flatnonzero(~union))


def test_csr_reader_rejects_truncated_output(tmp_path, sensed_world):
    source=sensed_world/"visibility_continuous.csr"
    truncated=tmp_path/"bad.csr"
    truncated.write_bytes(source.read_bytes()[:-8])
    with pytest.raises(ValueError,match="truncated"):
        read_csr(truncated)
    truncated.write_bytes(b"WGVCSR01")
    with pytest.raises(ValueError,match="header"):
        read_csr(truncated)
