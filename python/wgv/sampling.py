"""Continuous point sampling and discrete free-cell/voxel center targets."""
import json
from pathlib import Path
import numpy as np
import shapely
from scipy.spatial import cKDTree
from . import solid
from .generation import load_world


def grid_centers(free, occupancy):
    """Enumerate every free cell center in C-order, in physical world units."""
    free = np.asarray(free)
    dimension = free.ndim
    if dimension not in (2, 3) or free.dtype != bool:
        raise ValueError("occupancy must be a 2D or 3D Boolean array")
    origin = np.asarray(occupancy["origin"], float)
    spacing = np.asarray(occupancy["spacing_xyz"], float)
    if (origin.shape != (dimension,) or spacing.shape != (dimension,)
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)):
        raise ValueError("grid origin and positive spacing must match the dimension")
    indices = np.argwhere(free)
    return origin + (indices + .5) * spacing, indices


def center_indices(points, free, occupancy):
    """Validate legal grid-center coordinates and return their integer indices."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != free.ndim or not np.all(np.isfinite(points)):
        raise ValueError("discrete positions require finite coordinates matching the grid dimension")
    origin = np.asarray(occupancy["origin"], float)
    spacing = np.asarray(occupancy["spacing_xyz"], float)
    fractional = (points - origin) / spacing - .5
    rounded = np.rint(fractional)
    if not np.all(np.abs(fractional - rounded) <= 1e-8):
        raise ValueError("discrete positions must be cell or voxel centers")
    if not np.all((rounded >= 0) & (rounded < np.asarray(free.shape))):
        raise ValueError("discrete center is outside the grid")
    indices = rounded.astype(int)
    if not np.all(free[tuple(indices.T)]):
        raise ValueError("discrete centers must belong to free cells or voxels")
    return indices


def prepare_discrete(folder, observers=8):
    """Save all free cell/voxel centers and distinct grid-center viewpoints."""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    free = np.load(folder / "free.npy")
    points, indices = grid_centers(free, metadata["occupancy"])
    if not 1 <= observers <= len(points):
        raise ValueError("observer count exceeds the number of free grid centers")
    source_path = folder / "observers.npy"
    if source_path.exists():
        source = np.load(source_path)
        if len(source) != observers:
            raise ValueError("discrete observer count must match the saved continuous observers")
        if source.shape != (observers, free.ndim) or not np.all(np.isfinite(source)):
            raise ValueError("saved continuous observers have invalid coordinates")
        chosen = []
        for observer in source:
            order = np.argsort(np.linalg.norm(points - observer, axis=1), kind="stable")
            chosen.append(next(int(i) for i in order if int(i) not in chosen))
        observer_points = points[chosen]
        selection = "nearest distinct free grid center to each continuous observer"
        displacements = np.linalg.norm(observer_points - source, axis=1).tolist()
    else:
        observer_points = select_observers(points, observers)
        selection = "deterministic Euclidean farthest-point sequence on grid centers"
        displacements = None
    np.save(folder / "discrete_targets.npy", points)
    np.save(folder / "discrete_target_cell_indices.npy", indices)
    np.save(folder / "discrete_observers.npy", observer_points)
    report = {"method": "all_free_grid_centers", "target_count": len(points),
              "observer_count": observers, "observer_selection": selection,
              "observer_displacements": displacements,
              "coordinates": "origin + (cell index + 0.5) * spacing; axes x,y[,z]",
              "target_order": "C-order free cell indices, one target per free cell or voxel",
              "geometry_sha256": metadata["geometry_sha256"]}
    (folder / "discrete_sampling.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def membership(world, dimension, points, clearance=0):
    points = np.asarray(points, float)
    if dimension == 2:
        inside = shapely.contains_xy(world, points[:, 0], points[:, 1])
        if clearance > 0:
            inside &= shapely.distance(shapely.points(points), world.boundary) > clearance
        return inside
    return solid.contains(world, points, margin=clearance)


def sample(world, dimension, size, count, seed, method="uniform", **options):
    rng = np.random.default_rng(seed)
    size = np.asarray(size, float)
    if count < 0 or int(count) != count:
        raise ValueError("sample count must be a nonnegative integer")
    if count == 0:
        return np.empty((0, dimension))
    if method == "explicit":
        points = np.asarray(options["coordinates"], float)
        if points.shape != (count, dimension) or not np.all(membership(world, dimension, points)):
            raise ValueError("explicit samples have invalid shape or are outside free space")
        return points
    if method == "lattice":
        spacing = float(options.get("spacing", 1.))
        axes = [np.arange(spacing/2, bound, spacing) for bound in size]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
        points = points[membership(world, dimension, points)]
        if len(points) < count:
            raise ValueError("lattice contains fewer valid points than requested")
        return points[np.linspace(0, len(points)-1, count, dtype=int)]
    if method == "boundary":
        if dimension != 2:
            raise ValueError("analytic 3D boundary sampling is not implemented")
        rings = [world.exterior, *world.interiors]
        lengths = np.asarray([r.length for r in rings])
        indices = rng.choice(len(rings), count, p=lengths/lengths.sum())
        return np.asarray([rings[i].interpolate(rng.uniform(0, rings[i].length)).coords[0] for i in indices])
    if method not in ("uniform", "poisson_disk", "clearance", "gaussian", "callback"):
        raise ValueError(f"unsupported sample method {method}")
    accepted = []
    minimum_distance = float(options.get("minimum_distance", .6))
    minimum_clearance = float(options.get("clearance", .5 if method == "clearance" else 0))
    for batch in range(int(options.get("max_batches", 2000))):
        if method == "gaussian":
            points = rng.normal(options.get("mean", size/2), options.get("std", size/5), (2048, dimension))
        elif method == "callback":
            points = np.asarray(options["callback"](rng, 2048), float)
            if points.shape != (2048, dimension):
                raise ValueError("sampling callback returned the wrong shape")
        else:
            points = rng.uniform(0, size, (2048, dimension))
        points = points[np.all((points > 0) & (points < size), axis=1)]
        points = points[membership(world, dimension, points, minimum_clearance)]
        if method == "poisson_disk":
            # Rejection dart throwing against accepted points; no linear-time or
            # maximal-packing claim. Counts can fail when separation is too big.
            for point in points:
                if not accepted or np.min(np.linalg.norm(np.asarray(accepted)-point, axis=1)) >= minimum_distance:
                    accepted.append(point)
                    if len(accepted) == count:
                        return np.asarray(accepted)
        else:
            accepted.extend(points[:count-len(accepted)])
            if len(accepted) == count:
                return np.asarray(accepted)
    raise RuntimeError(f"could sample only {len(accepted)}/{count} valid {method} points")


def select_observers(points, count=8):
    if not 1 <= count <= len(points):
        raise ValueError("observer count exceeds sample count")
    # Deliberately simple and deterministic: spatially spread viewpoints,
    # without using visibility or making an optimal-placement claim.
    chosen = [int(np.argmin(points[:, 0] + .1*points[:, 1]))]
    closest = np.full(len(points), np.inf)
    for _ in range(count-1):
        closest = np.minimum(closest, np.linalg.norm(points-points[chosen[-1]], axis=1))
        closest[chosen] = -1
        chosen.append(int(np.argmax(closest)))
    return points[chosen]


def prepare(folder, count=1500, observers=8, seed=7001, method="uniform"):
    folder = Path(folder)
    world, metadata = load_world(folder)
    points = sample(world, metadata["dimension"], metadata["world_size"], count, seed, method)
    observer_points = select_observers(points, observers)
    np.save(folder/"targets.npy", points)
    np.save(folder/"observers.npy", observer_points)
    report = {"method": method, "seed": seed, "target_count": count, "observer_count": observers,
              "observer_selection": "deterministic Euclidean farthest-point sequence",
              "coordinates": "continuous world coordinates; array axes x,y[,z]",
              "all_targets_valid": bool(np.all(membership(world, metadata["dimension"], points)))}
    if metadata["dimension"] == 3:
        # Independent uniform domain samples estimate volume, with binomial
        # standard error. The conservative voxel measure is reported separately.
        random = np.random.default_rng(seed+99).uniform(0, metadata["world_size"], (60000, 3))
        fraction = float(np.mean(membership(world, 3, random)))
        box_volume = float(np.prod(metadata["world_size"]))
        report.update({"volume_estimate": fraction*box_volume,
                       "volume_mc_samples": len(random),
                       "volume_standard_error": box_volume*np.sqrt(fraction*(1-fraction)/len(random))})
    (folder/"sampling.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
