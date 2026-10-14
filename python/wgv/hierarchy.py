"""Exact full-child region quadtrees/octrees of finite Boolean masks.

Trees are measurement artifacts. Visibility still uses the existing engine.
Axis order is x,y[,z]; non-dyadic arrays are padded on their high sides.
"""
import itertools
import numpy as np


def mask_levels(mask):
    mask = np.asarray(mask)
    if mask.ndim not in (2, 3) or any(n < 1 for n in mask.shape):
        raise ValueError("a nonempty 2D or 3D mask is required")
    if mask.dtype != np.bool_:
        raise ValueError("occupancy mask must have Boolean dtype")
    side = 1 << (max(mask.shape)-1).bit_length()
    padded = np.zeros((side,)*mask.ndim, dtype=np.uint8)
    padded[tuple(slice(0, n) for n in mask.shape)] = mask
    levels = [padded]
    while levels[-1].shape[0] > 1:
        child = levels[-1]
        shape = tuple(v for n in child.shape for v in (n//2, 2))
        blocks = child.reshape(shape)
        axes = tuple(range(1, 2*mask.ndim, 2))
        lo, hi = blocks.min(axis=axes), blocks.max(axis=axes)
        levels.append(np.where((lo == hi) & (lo != 2), lo, 2).astype(np.uint8))
    return levels[::-1]


def measure(mask, spacing=None, origin=None):
    """Measure all retained nodes, pruning descendants of uniform blocks."""
    mask = np.asarray(mask)
    levels = mask_levels(mask)
    dimension = mask.ndim
    spacing = np.ones(dimension) if spacing is None else np.asarray(spacing, float)
    origin = np.zeros(dimension) if origin is None else np.asarray(origin, float)
    if spacing.shape != (dimension,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("one finite positive spacing per axis is required")
    if origin.shape != (dimension,) or np.any(~np.isfinite(origin)):
        raise ValueError("one finite origin coordinate per axis is required")
    active = np.ones((1,)*dimension, dtype=bool)
    profile = []
    for depth, level in enumerate(levels):
        free = active & (level == 1)
        blocked = active & (level == 0)
        mixed = active & (level == 2)
        profile.append({"depth": depth, "nodes": int(active.sum()),
                        "free_leaves": int(free.sum()), "blocked_leaves": int(blocked.sum()),
                        "mixed_nodes": int(mixed.sum()),
                        "block_spacing_xyz": (spacing * (2**(len(levels)-1-depth))).tolist()})
        active = mixed
        for axis in range(dimension):
            active = active.repeat(2, axis=axis)
    internal = sum(r["mixed_nodes"] for r in profile)
    free_leaves = sum(r["free_leaves"] for r in profile)
    blocked_leaves = sum(r["blocked_leaves"] for r in profile)
    leaves = free_leaves+blocked_leaves
    total = sum(r["nodes"] for r in profile)
    branching = 2**dimension
    assert leaves == 1+(branching-1)*internal
    assert total == 1+branching*internal
    padded_side = levels[-1].shape[0]
    depth_max = len(levels)-1
    isotropic = bool(np.allclose(spacing, spacing[0], rtol=1e-12, atol=0))
    return {"schema_version": 1, "dimension": dimension,
            "kind": "quadtree" if dimension == 2 else "octree",
            "geometry_of_blocks": "square/cubic" if isotropic else "rectangular dyadic",
            "classification": "exact homogeneity of saved Boolean mask; not continuous-region certification",
            "children_per_internal_node": branching,
            "input_shape_xyz": list(mask.shape), "root_shape_xyz": [padded_side]*dimension,
            "padding": "blocked entries on high sides; origin retained",
            "spacing_xyz": spacing.tolist(), "origin_xyz": origin.tolist(),
            "root_extent_xyz": (spacing*padded_side).tolist(),
            "maximum_allowed_depth": depth_max, "total_nodes": total,
            "internal_nodes": internal, "leaves": leaves,
            "free_leaves": free_leaves, "blocked_leaves": blocked_leaves,
            "dense_entries": int(mask.size), "padded_dense_entries": padded_side**dimension,
            "full_tree_nodes": (branching**(depth_max+1)-1)//(branching-1),
            "node_to_padded_entry_ratio": total/(padded_side**dimension),
            "profile": profile}


def leaf_blocks(mask):
    """Yield (lower index, edge in finest cells, value, depth) for display/replay."""
    levels = mask_levels(mask)
    dimension = np.asarray(mask).ndim
    side = levels[-1].shape[0]
    stack = [(0, (0,)*dimension)]
    while stack:
        depth, index = stack.pop()
        value = int(levels[depth][index])
        edge = side//(2**depth)
        if value != 2:
            yield tuple(i*edge for i in index), edge, bool(value), depth
        else:
            for bits in reversed(list(itertools.product((0, 1), repeat=dimension))):
                stack.append((depth+1, tuple(2*i+b for i, b in zip(index, bits))))
