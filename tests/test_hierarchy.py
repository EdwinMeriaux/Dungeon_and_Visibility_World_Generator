import numpy as np
import pytest
from wgv.hierarchy import measure, leaf_blocks


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("value", [False, True])
def test_uniform_root_is_single_leaf(dimension, value):
    report = measure(np.full((8,)*dimension, value, bool))
    assert report["total_nodes"] == 1
    assert report["leaves"] == 1


@pytest.mark.parametrize("dimension", [2, 3])
def test_checkerboard_has_complete_tree(dimension):
    mask = np.indices((8,)*dimension).sum(axis=0) % 2 == 0
    report = measure(mask)
    assert report["total_nodes"] == sum((2**dimension)**k for k in range(4))
    assert report["leaves"] == mask.size


@pytest.mark.parametrize("shape", [(7, 5), (7, 5, 3)])
def test_leaf_partition_reconstructs_every_entry_including_padding(shape):
    mask = np.random.default_rng(19).random(shape) > .5
    report = measure(mask, spacing=[.25]*len(shape))
    recovered = np.zeros(report["root_shape_xyz"], bool)
    visits = np.zeros_like(recovered, dtype=int)
    blocks = list(leaf_blocks(mask))
    for index, edge, value, _ in blocks:
        slices = tuple(slice(i, i+edge) for i in index)
        recovered[slices] = value
        visits[slices] += 1
    expected = np.zeros_like(recovered)
    expected[tuple(slice(0, n) for n in shape)] = mask
    assert np.array_equal(recovered, expected)
    assert np.all(visits == 1)
    assert len(blocks) == report["leaves"]


def test_axis_permutation_preserves_node_counts():
    mask = np.random.default_rng(7).random((8, 8, 8)) > .7
    assert measure(mask)["total_nodes"] == measure(mask.transpose(2, 0, 1))["total_nodes"]


def test_rectangular_physical_blocks_are_identified():
    report = measure(np.ones((4, 4, 4), bool), spacing=[1, 1, 2])
    assert report["geometry_of_blocks"] == "rectangular dyadic"
    assert report["root_extent_xyz"] == [4, 4, 8]


def test_nonboolean_mask_and_bad_spacing_rejected():
    with pytest.raises(ValueError):
        measure(np.ones((4, 4), int))
    with pytest.raises(ValueError):
        measure(np.ones((4, 4), bool), spacing=[1, 0])
