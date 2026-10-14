"""Discrete targets must represent free cells, including nonunit/nonzero grids."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import load_npz
from wgv.sampling import grid_centers, center_indices, prepare_discrete
from wgv.visibility import rust_input, compute, paired_representations
from wgv.validation import validate
from wgv.generation import generate
from wgv.sampling import prepare

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dimension", [2, 3])
def test_center_conversion_with_nonzero_origin_and_unequal_spacing(dimension):
    free = np.ones((3,) * dimension, dtype=bool)
    free[(1,) * dimension] = False
    origin = np.arange(dimension) + 11.
    spacing = np.arange(dimension) + .25
    occupancy = {"origin": origin.tolist(), "spacing_xyz": spacing.tolist()}
    points, indices = grid_centers(free, occupancy)
    assert len(points) == free.sum()
    assert np.array_equal(indices, np.argwhere(free))
    assert np.array_equal(points, origin + (indices + .5) * spacing)
    assert np.array_equal(center_indices(points, free, occupancy), indices)
    with pytest.raises(ValueError, match="centers"):
        center_indices(points + .1 * spacing, free, occupancy)
    with pytest.raises(ValueError, match="free cells"):
        center_indices([origin + 1.5 * spacing], free, occupancy)
    with pytest.raises(ValueError, match="outside"):
        center_indices([origin - .5 * spacing], free, occupancy)


@pytest.fixture(params=["2d_polygonal_1", "3d_orthogonal_1"])
def folder(tmp_path, request):
    config = json.loads((PROJECT / "configs/showcase" / (request.param + ".json")).read_text())
    target = tmp_path / "world"
    generate(config, target)
    prepare(target, count=64, observers=3)
    return target


def test_discrete_defaults_enumerate_all_free_centers_without_changing_continuous_samples(folder):
    before = (folder / "targets.npy").read_bytes()
    payload = rust_input(folder, mode="discrete")
    free = np.load(folder / "free.npy")
    occupancy = json.loads((folder / "metadata.json").read_text())["occupancy"]
    expected, indices = grid_centers(free, occupancy)
    assert np.array_equal(payload["targets"], expected)
    assert np.array_equal(np.load(folder / "discrete_target_cell_indices.npy"), indices)
    assert len(payload["targets"]) == int(free.sum())
    center_indices(payload["observers"], free, occupancy)
    assert (folder / "targets.npy").read_bytes() == before
    shifted = expected[:2] + .1 * np.asarray(occupancy["spacing_xyz"])
    with pytest.raises(ValueError, match="centers"):
        rust_input(folder, mode="discrete", observers=payload["observers"], targets=shifted)


def test_center_queries_and_matched_continuous_queries_use_identical_positions(folder):
    report = paired_representations(folder)
    assert report["target_rule"] == "grid_centers"
    assert not (folder / "paired_source_target_indices.npy").exists()
    a = json.loads((folder / "visibility_paired_continuous_input.json").read_text())
    b = json.loads((folder / "visibility_paired_discrete_input.json").read_text())
    assert a["targets"] == b["targets"]
    assert a["observers"] == b["observers"]
    assert b["coordinate_rule"] == "free grid centers"
    grid = np.load(folder / "free.npy")
    assert report["target_count"] == int(grid.sum())
    direct, result = compute(folder, mode="discrete")
    assert (direct != load_npz(folder / "visibility_paired_discrete.npz")).nnz == 0
    assert result["coordinate_rule"] == "free grid centers"
    assert "discrete targets and observers are free grid centers" in validate(folder)["checks"]


def test_legacy_comparison_requires_an_explicit_rule(folder):
    report = paired_representations(folder, target_rule="legacy_common_samples")
    payload = json.loads((folder / "visibility_paired_discrete_input.json").read_text())
    assert report["target_rule"] == "legacy_common_samples"
    assert payload["coordinate_rule"] == "legacy arbitrary points"
    assert np.array_equal(np.load(folder / "paired_targets.npy"),
                          np.load(folder / "targets.npy")[np.load(folder / "paired_source_target_indices.npy")])
