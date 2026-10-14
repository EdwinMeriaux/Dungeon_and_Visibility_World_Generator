"""Archived layout reuse and opt-in distribution-configurable construction."""
import sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
LEGACY = PROJECT / "vendor" / "non_orthogonal_worlds"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
import generate as original_helpers
import generate_arbitrary as original_layout

FAMILIES = original_layout.FAMILIES


def create_layout(config, rng, requested_rooms, width, loop_fraction):
    family = config["family"]
    if family not in FAMILIES:
        raise ValueError(f"unknown construction family {family}")
    size = float(min(config["world_size"][:2]))
    args = {"rooms": requested_rooms, "hallway_width": width, "loop_fraction": loop_fraction}
    sampling = None
    if "layout" in config or "placement" in config.get("rooms", {}):
        from .layout_distributions import construct
        rooms, paths, centers, elevations, sampling = construct(
            config, rng, requested_rooms, width, loop_fraction)
    else:
        world, rooms, paths, _ = original_layout.raw_layout(size, family, rng, args)
        if world.geom_type != "Polygon" or not world.is_valid:
            raise ValueError("legacy layout union is disconnected or invalid")
        _, rooms, paths, _ = original_layout.fit_layout(world, rooms, paths, size, rng)
        centers = np.asarray([r.centroid.coords[0] for r in rooms])
    # Paths have explicit intermediate junctions. A graph on all unique path
    # vertices is valid for every family, unlike inventing room-to-room edges.
    nodes, edges, index = [], set(), {}
    for path in paths:
        ids = []
        for point in path["points"]:
            key = tuple(np.round(point, 8))
            if key not in index:
                index[key] = len(nodes)
                nodes.append(list(point))
            ids.append(index[key])
        for a, b in zip(ids, ids[1:]):
            if a != b:
                edges.add(tuple(sorted((a, b))))
    adjacency = {i: set() for i in range(len(nodes))}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining, components = set(adjacency), 0
    while remaining:
        stack = [remaining.pop()]
        components += 1
        while stack:
            for j in adjacency[stack.pop()]:
                if j in remaining:
                    remaining.remove(j)
                    stack.append(j)
    result = {"family": family, "centers": centers.tolist(),
            "template_rooms": [list(r.exterior.coords) for r in rooms],
            "paths": paths, "graph_nodes": nodes, "graph_edges": sorted(edges),
            "graph_components": components,
            "graph_cycle_rank": len(edges) - len(nodes) + components,
            "room_count_requested": requested_rooms, "room_count_achieved": len(rooms)}
    if sampling is not None:
        result["sampling"] = sampling
        result["room_elevations"] = elevations
    return result


def rectilinear(points):
    """Replace oblique segments by axis-aligned two-segment routes."""
    result = [list(points[0])]
    for endpoint in points[1:]:
        previous = result[-1]
        if previous[0] != endpoint[0] and previous[1] != endpoint[1]:
            result.append([endpoint[0], previous[1]])
        result.append(list(endpoint))
    return result
