"""Artifact checks, including independent raster and connectivity checks."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import load_npz
from .generation import load_world
from .sampling import membership, center_indices, grid_centers
from . import solid
from .visibility import read_csr


def validate(folder):
    folder = Path(folder)
    world, metadata = load_world(folder)
    dimension = metadata["dimension"]
    geometry_file = folder/("geometry.geojson" if dimension == 2 else "solid.json")
    assert hashlib.sha256(geometry_file.read_bytes()).hexdigest() == metadata["geometry_sha256"]
    free = np.load(folder/"free.npy")
    assert list(free.shape) == metadata["occupancy"]["shape_xyz"] and free.dtype == bool
    neighbors = metadata["occupancy"]["neighbors"]
    c = {4: 1, 8: 2} if dimension == 2 else {6: 1, 18: 2, 26: 3}
    components = ndimage.label(free, ndimage.generate_binary_structure(dimension, c[neighbors]))[1]
    assert components == metadata["occupancy"]["free_components"]
    spacing = np.asarray(metadata["occupancy"]["spacing_xyz"])
    indices = np.argwhere(free)
    origin = np.asarray(metadata["occupancy"]["origin"])
    centers = origin+(indices+.5)*spacing
    assert np.all(membership(world, dimension, centers))
    if dimension == 2:
        assert world.geom_type == "Polygon" and world.is_valid
        # An independent difference predicate checks every cell, including
        # blocked cells; it does not reuse the generator's covers call.
        axes = [o+(np.arange(n)+.5)*s for n, s, o in zip(free.shape, spacing, origin)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        boxes = shapely.box(points[:,0]-spacing[0]/2, points[:,1]-spacing[1]/2,
                            points[:,0]+spacing[0]/2, points[:,1]+spacing[1]/2)
        reconstructed = shapely.is_empty(shapely.difference(boxes, world)).reshape(free.shape)
        assert np.array_equal(reconstructed, free)
    else:
        scale=float(metadata.get("similarity_scale",1))
        # All eight vertices of every accepted voxel are independently checked.
        # The generation certificate is stronger: a Lipschitz clearance bound
        # certifies the entire circumsphere, including voxel interiors.
        for corner in itertools.product((-1, 1), repeat=3):
            assert np.all(membership(world, 3, centers+spacing*np.asarray(corner)/2))
        certificate = json.loads((folder/"connectivity_certificate.json").read_text())
        assert len(certificate["overlap_witnesses"]) == len(world["positive"])-1
        for witness in certificate["overlap_witnesses"]:
            for key in ("a", "b"):
                assert solid.primitive_clearance(world["positive"][witness[key]], witness["point"])[0] > 0
            assert solid.contains(world, [witness["point"]])[0]
        corridors = [p for p in world["positive"] if p["role"] == "corridor"]
        for obstacle, record in zip(world["negative"], [r for r in metadata["obstacle_records"] if not r.get("population_summary")]):
            assert all(solid.disjoint_primitives(obstacle, corridor, .1*scale) for corridor in corridors)
            host = world["positive"][record["host_room"]]
            low, high = solid.bounds(obstacle)
            corners = np.asarray(list(itertools.product(*zip(low, high))))
            assert np.all(solid.primitive_clearance(host, corners) > .39*scale)
        for i, obstacle in enumerate(world["negative"]):
            assert all(solid.disjoint_primitives(obstacle, other, .2*scale) for other in world["negative"][:i])
    for name in ("targets.npy", "observers.npy"):
        if (folder/name).exists():
            assert np.all(membership(world, dimension, np.load(folder/name)))
    # Check the persistent sensing contract for every saved query, including
    # archived paired queries. This also catches corruption in saved target sets.
    for binary in folder.glob("visibility_*.csr"):
        label = binary.stem.removeprefix("visibility_")
        matrix = load_npz(binary.with_suffix(".npz"))
        payload = json.loads((folder/f"visibility_{label}_input.json").read_text())
        saved = json.loads(binary.with_suffix(".json").read_text())
        assert matrix.shape == (len(payload["observers"]), len(payload["targets"]))
        assert (matrix != read_csr(binary)).nnz == 0
        assert matrix.nnz == saved["nonzero"]
        assert np.array_equal(np.asarray(matrix.sum(axis=1)).ravel(), saved["per_observer_counts"])
        union = np.asarray(matrix.sum(axis=0)).ravel() > 0
        assert int(union.sum()) == saved["union_count"]
        assert np.isclose(union.mean(), saved["coverage_fraction"], rtol=0, atol=1e-14)
        arrays = (("observers.npy", "targets.npy") if label == "continuous" else
                  ("paired_observers.npy", "paired_targets.npy") if label.startswith("paired_") else
                  ("discrete_observers.npy", "discrete_targets.npy") if label == "discrete" else None)
        if arrays:
            for key, name in zip(("observers", "targets"), arrays):
                assert np.array_equal(np.asarray(payload[key]), np.load(folder/name))
        for key, selected in (("visible_union", union), ("unobserved", ~union)):
            filename = saved.get("target_set_files", {}).get(key)
            if filename:
                assert np.array_equal(np.load(folder/filename), np.flatnonzero(selected))
    if (folder/"visibility_continuous.npz").exists():
        matrix = load_npz(folder/"visibility_continuous.npz")
        assert matrix.shape == (len(np.load(folder/"observers.npy")), len(np.load(folder/"targets.npy")))
        coverage = np.asarray(matrix.sum(axis=0)).ravel() > 0
        report = json.loads((folder/"visibility_continuous.json").read_text())
        assert int(coverage.sum()) == report["union_count"]
    if (folder/"discrete_targets.npy").exists():
        expected, expected_indices = grid_centers(free, metadata["occupancy"])
        assert np.array_equal(np.load(folder/"discrete_targets.npy"), expected)
        assert np.array_equal(np.load(folder/"discrete_target_cell_indices.npy"), expected_indices)
        viewpoints = np.load(folder/"discrete_observers.npy")
        center_indices(viewpoints, free, metadata["occupancy"])
        assert len(np.unique(viewpoints, axis=0)) == len(viewpoints)
        for label in ("discrete", "paired_discrete"):
            if not (folder/f"visibility_{label}.npz").exists():
                continue
            saved = json.loads((folder/f"visibility_{label}.json").read_text())
            if saved.get("coordinate_rule") != "free grid centers":
                continue  # Preserve older arbitrary-point comparison evidence.
            payload = json.loads((folder/f"visibility_{label}_input.json").read_text())
            center_indices(payload["targets"], free, metadata["occupancy"])
            center_indices(payload["observers"], free, metadata["occupancy"])
            matrix = load_npz(folder/f"visibility_{label}.npz")
            assert matrix.shape == (len(payload["observers"]), len(payload["targets"]))
            union = np.asarray(matrix.sum(axis=0)).ravel() > 0
            assert int(union.sum()) == saved["union_count"]
    report = {"passed": True, "world": folder.name, "continuous_components": 1,
              "discrete_components": int(components), "free_cells": int(free.sum()),
              "checks": ["geometry hash", "representation dimensions", "connectivity",
                         "conservative cell or voxel consistency", "valid samples", "visibility output consistency"]}
    if (folder/"discrete_targets.npy").exists():
        report["checks"].append("discrete targets and observers are free grid centers")
    (folder/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
