"""Two-stage geometric realization with measured, checked constraints."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import unary_union
from scipy import ndimage
from scipy.spatial import ConvexHull
from .parameters import draw, draw_point
from .layout import create_layout, rectilinear, original_layout, LEGACY, PROJECT
from . import solid


def curve_segments(radius, tolerance):
    if radius <= 0 or tolerance <= 0:
        raise ValueError("curve radius and tolerance must be positive")
    # Sagitta bound for a circular arc. Affine ellipse scaling is accounted
    # for by using its largest semiaxis.
    angle = math.acos(max(-1., min(1., 1-tolerance/radius)))
    return max(4, int(math.ceil(math.pi/(4*max(angle, 1e-8)))))


def polygon_shape(kind, center, width, length, rng, vertices=8, tolerance=.025, options=None):
    options = options or {}
    if kind in ("rectangle", "square"):
        shape = box(-width/2, -length/2, width/2, length/2)
    elif kind == "orthogonal":
        shape = unary_union([box(-width/2, -length/2, width/2, length/5),
                             box(-width/2, -length/2, width/5, length/2)])
    elif kind in ("circle", "ellipse"):
        shape = Point(0, 0).buffer(1., quad_segs=curve_segments(max(width, length)/2, tolerance))
        shape = affinity.scale(shape, width/2, length/2, origin=(0, 0))
    elif kind == "rounded":
        radius = min(width, length)*draw(options.get("rounded_radius_fraction", .24), rng,
                                       minimum=.001, maximum=.49)
        shape = box(-width/2+radius, -length/2+radius, width/2-radius, length/2-radius)
        shape = shape.buffer(radius, quad_segs=curve_segments(radius, tolerance))
    elif kind == "smooth":
        n = draw(options["smooth_vertices"], rng, integer=True, minimum=16, maximum=10000) if "smooth_vertices" in options else max(96, int(math.ceil(math.pi*max(width, length)/math.sqrt(tolerance))))
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        phase = draw(options["smooth_phase"], rng) if "smooth_phase" in options else rng.uniform(0, 2*np.pi)
        a = draw(options.get("smooth_amplitude_3", .09), rng, minimum=0, maximum=.4)
        b = draw(options.get("smooth_amplitude_5", .04), rng, minimum=0, maximum=.4)
        radii = 1 + a*np.sin(3*angles+phase) + b*np.cos(5*angles)
        shape = Polygon(np.column_stack([np.cos(angles)*width/2*radii,
                                         np.sin(angles)*length/2*radii]))
    elif kind in ("triangle", "regular_polygon", "convex_polygon", "irregular_polygon"):
        n = 3 if kind == "triangle" else vertices
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = np.ones(n) if kind == "regular_polygon" else (
            np.asarray([draw(options["radial_factor"], rng, minimum=.01) for _ in range(n)])
            if "radial_factor" in options else rng.uniform(.62, 1., n))
        shape = Polygon(np.column_stack([np.cos(angles)*width/2*radii,
                                         np.sin(angles)*length/2*radii]))
        if kind == "convex_polygon":
            shape = shape.convex_hull
    else:
        raise ValueError(f"unsupported 2D shape {kind}")
    return affinity.translate(shape, *center)


def realize_2d(layout, config, rng):
    geometry = config["geometry"]
    tolerance = float(config.get("curve_tolerance", .025))
    room_cfg = config.get("rooms", {})
    rooms, corridors, room_records = [], [], []
    for center, template in zip(layout["centers"], layout["template_rooms"]):
        template = Polygon(template)
        bounds = template.bounds
        width = draw(room_cfg.get("width", max(6., bounds[2]-bounds[0])), rng, minimum=1)
        length = draw(room_cfg.get("length", max(6., bounds[3]-bounds[1])), rng, minimum=1)
        vertices = draw(room_cfg.get("vertices", 8), rng, integer=True, minimum=3, maximum=64)
        if geometry == "orthogonal":
            kind = draw(room_cfg.get("shape", {"distribution": "categorical",
                                             "values": ["rectangle", "orthogonal"]}), rng)
            if kind not in ("rectangle", "square", "orthogonal"):
                raise ValueError("orthogonal rooms require orthogonal shapes")
        elif geometry == "polygonal":
            kind = draw(room_cfg.get("shape", "irregular_polygon"), rng)
        elif geometry == "curvilinear":
            kind = draw(room_cfg.get("shape", {"distribution": "categorical",
                         "values": ["ellipse", "rounded", "smooth"]}), rng)
        else:
            raise ValueError(f"unsupported 2D geometry {geometry}")
        room = polygon_shape(kind, center, width, length, rng, vertices, tolerance, room_cfg)
        orientation = draw(room_cfg.get("orientation", {"distribution": "uniform", "min": -35, "max": 35}), rng)
        if geometry == "orthogonal":
            orientation = 90*round(orientation/90)
        room = affinity.rotate(room, orientation, origin=center)
        rooms.append(room)
        room_records.append({"shape": kind, "width": width, "length": length,
                             "vertices_requested": vertices, "orientation": orientation,
                             "center": list(center)})
    for path in layout["paths"]:
        points = rectilinear(path["points"]) if geometry == "orthogonal" else path["points"]
        corridor_cfg = config.get("corridors", {})
        minimum_width = draw(corridor_cfg.get("minimum_achieved_width", 3.5), rng, minimum=.01)
        requested_width = draw(corridor_cfg.get("path_width", path["width"]), rng, minimum=.01)
        width = max(minimum_width, requested_width)
        if geometry == "curvilinear":
            rounds = draw(corridor_cfg.get("smoothing_rounds", 2), rng, integer=True, minimum=0, maximum=8)
            points = original_layout.smooth_path(points, rounds=rounds).tolist()
        path["realization_2d"] = {"points": points, "width": width,
                                  "width_requested": requested_width, "minimum_width_sampled": minimum_width}
        corridor = LineString(points).buffer(width/2,
                        quad_segs=curve_segments(width/2, tolerance),
                        cap_style=3 if geometry != "curvilinear" else 1,
                        join_style=2 if geometry != "curvilinear" else 1)
        corridors.append(corridor)
    size = config["world_size"]
    domain = box(.6, .6, size[0]-.6, size[1]-.6)
    free = unary_union(rooms+corridors)
    if layout.get("sampling", {}).get("fit", {}).get("mode") == "none" and not domain.covers(free):
        raise ValueError("sampled primitives leave the domain with layout.fit_mode=none")
    free = free.intersection(domain)
    if free.geom_type != "Polygon" or not free.is_valid:
        raise ValueError("realized 2D world is not one valid polygon")
    return free, rooms, corridors, room_records


def obstacles_2d(free, rooms, corridors, config, rng):
    records, obstacles = [], []
    tolerance = config.get("curve_tolerance", .025)
    for population in config.get("obstacles", []):
        count = draw(population.get("count", 0), rng, integer=True, minimum=0, maximum=1000)
        kind = draw(population["type"], rng)
        if config["geometry"] == "orthogonal" and kind not in ("rectangle", "square", "orthogonal"):
            raise ValueError("orthogonal worlds require orthogonal obstacles")
        achieved = 0
        for obstacle_index in range(count):
            width = draw(population.get("width", population.get("radius", 1.)), rng, minimum=.1)
            if "radius" in population and "width" not in population:
                width *= 2
            length = draw(population.get("length", width), rng, minimum=.1)
            vertices = draw(population.get("vertices", 7), rng, integer=True, minimum=3, maximum=64)
            clearance = draw(population.get("clearance", .7), rng, minimum=0)
            accepted = False
            for trial in range(int(config.get("obstacle_trials", 800))):
                center = obstacle_center(population, free.bounds, rooms, corridors, rng, 2)
                hole = polygon_shape(kind, center, width, length, rng, vertices, tolerance, population)
                if not free.contains(hole.buffer(clearance)):
                    continue
                candidate = free.difference(hole)
                if candidate.geom_type != "Polygon" or not candidate.is_valid:
                    continue
                free = candidate
                obstacles.append(hole)
                records.append({"population": len(records), "type": kind, "center": list(center),
                                "width": width, "length": length, "vertices": vertices,
                                "clearance_requested": clearance})
                achieved += 1
                accepted = True
                break
            if not accepted and population.get("exact_count", True):
                raise ValueError(f"could place only {achieved}/{count} {kind} obstacles")
        records.append({"population_summary": True, "type": kind,
                        "requested_count": count, "achieved_count": achieved})
    return free, obstacles, records


def obstacle_center(population, extent, rooms, corridors, rng, dimension):
    mode = population.get("placement", "uniform")
    if isinstance(mode, dict):
        if mode.get("distribution") == "categorical":
            mode = draw(mode, rng)
        else:
            if mode.get("type") == "callback":
                raise ValueError("callbacks are Python-only; pass a callable placement function")
            units = mode.get("coordinates", "world")
            if units not in ("world", "normalized"):
                raise ValueError("obstacle placement coordinates must be world or normalized")
            center = draw_point(mode, rng, dimension)
            if units == "normalized":
                low, high = np.asarray(extent[:dimension]), np.asarray(extent[dimension:])
                center = low+center*(high-low)
            return center
    if callable(mode):
        return np.asarray(mode(rng), float)
    if mode in ("room", "room_specific"):
        index = draw(population["room_index"], rng, integer=True, minimum=0, maximum=len(rooms)-1) if "room_index" in population else int(rng.integers(len(rooms)))
        room = rooms[index]
        if dimension == 2:
            low, high = np.asarray(room.bounds[:2]), np.asarray(room.bounds[2:])
        else:
            low, high = solid.bounds(room)
        return rng.uniform(low, high)
    if mode in ("corridor", "corridor_specific"):
        index = draw(population["corridor_index"], rng, integer=True, minimum=0, maximum=len(corridors)-1) if "corridor_index" in population else int(rng.integers(len(corridors)))
        corridor = corridors[index]
        if dimension == 2:
            low, high = np.asarray(corridor.bounds[:2]), np.asarray(corridor.bounds[2:])
        else:
            low, high = solid.bounds(corridor)
        return rng.uniform(low, high)
    if mode in ("uniform", "clearance", "boundary_biased"):
        low, high = np.asarray(extent[:dimension]), np.asarray(extent[dimension:])
        center = rng.uniform(low, high)
        if mode == "boundary_biased":
            axis = draw(population["boundary_axis"], rng, integer=True, minimum=0, maximum=dimension-1) if "boundary_axis" in population else int(rng.integers(dimension))
            fraction = draw(population["boundary_fraction"], rng, minimum=0, maximum=1) if "boundary_fraction" in population else rng.beta(.5, .5)
            center[axis] = low[axis] + fraction*(high[axis]-low[axis])
        return center
    raise ValueError(f"unknown obstacle placement mode {mode}")


def realize_3d(layout, config, rng):
    geometry = config["geometry"]
    room_cfg = config.get("rooms", {})
    floors = draw(config.get("floors", 3), rng, integer=True, minimum=1, maximum=8)
    height = config["world_size"][2]
    floor_centers = np.linspace(4.5, height-4.5, floors)
    centers2 = np.asarray(layout["centers"])
    elevation_rule = config.get("junction_elevation_rule", "legacy_nearest")
    if elevation_rule not in ("legacy_nearest", "lowest_index"):
        raise ValueError("junction_elevation_rule must be legacy_nearest or lowest_index")
    layout["junction_elevation_rule"] = elevation_rule
    z_rooms = floor_centers[np.arange(len(centers2)) % floors]
    if "floor" in room_cfg:
        z_rooms = np.asarray([floor_centers[draw(room_cfg["floor"], rng, integer=True,
                                               minimum=0, maximum=floors-1)] for _ in centers2])
    if "elevation" in room_cfg:
        z_rooms = np.asarray([draw(room_cfg["elevation"], rng, minimum=0, maximum=height) for _ in centers2])
    for i, value in enumerate(layout.get("room_elevations", [])):
        if value is not None:
            z_rooms[i] = value
    config["floors"] = floors
    layout["floor_schedule"] = {"count_sampled": floors,
                                "reference_elevations_before_fit": floor_centers.tolist(),
                                "room_elevations_before_fit": z_rooms.tolist()}
    rooms, corridors, records = [], [], []
    for i, (center2, template) in enumerate(zip(centers2, layout["template_rooms"])):
        center = [*center2, z_rooms[i]]
        width = draw(room_cfg.get("width", 8.5), rng, minimum=2)
        length = draw(room_cfg.get("length", 8.5), rng, minimum=2)
        room_height = draw(room_cfg.get("height", 7.), rng, minimum=2)
        sampled_width, sampled_length = width, length
        orientation = draw(room_cfg.get("orientation", 0), rng)
        if geometry == "orthogonal":
            orientation = 90*round(orientation/90)
            if round(orientation/90) % 2:
                width, length = length, width
            room = solid.box(center, [width, length, room_height])
        elif geometry == "polyhedral":
            n = draw(room_cfg.get("vertices", 8), rng, integer=True, minimum=3, maximum=24)
            polygon = polygon_shape("convex_polygon", center2, width, length, rng, n, options=room_cfg).convex_hull
            polygon = affinity.rotate(polygon, orientation, origin=tuple(center2))
            room = solid.prism(list(polygon.exterior.coords)[:-1], z_rooms[i], room_height)
        elif geometry == "curvilinear":
            if not np.isclose(orientation/90, round(orientation/90)):
                raise ValueError("axis-aligned 3D ellipsoids require orientation in multiples of 90 degrees")
            if round(orientation/90) % 2:
                width, length = length, width
            room = solid.ellipsoid(center, [width/2, length/2, room_height/2])
        else:
            raise ValueError(f"unsupported 3D geometry {geometry}")
        rooms.append(room)
        records.append({"width": sampled_width, "length": sampled_length, "height": room_height,
                        "floor": int(np.argmin(abs(floor_centers-z_rooms[i])))
                        if np.isclose(np.min(abs(floor_centers-z_rooms[i])), 0) else None,
                        "center_before_solid_fit": list(center), "orientation": orientation})
    # Identical 2D junctions share one elevation, so branches actually connect.
    junction_z = {}
    for path in layout["paths"]:
        for point in path["points"]:
            key = tuple(np.round(point, 8))
            if key not in junction_z:
                distances = np.linalg.norm(centers2-np.asarray(point), axis=1)
                nearest = int(np.argmin(distances))
                if elevation_rule == "lowest_index":
                    # Archived configurations keep their original convention.
                    tolerance = 1e-12 * max(1., float(np.max(np.abs(centers2))),
                                           float(np.max(np.abs(point))))
                    nearest = np.flatnonzero(distances <= distances.min() + tolerance)[0]
                junction_z[key] = float(z_rooms[nearest])
    paths3 = []
    for path in layout["paths"]:
        raw = np.asarray(path["points"])
        endpoints_z = [junction_z[tuple(np.round(p, 8))] for p in raw]
        if geometry == "orthogonal":
            points3 = [[*raw[0], endpoints_z[0]]]
            for p, z in zip(raw[1:], endpoints_z[1:]):
                previous = points3[-1]
                points3.extend([[p[0], previous[1], previous[2]],
                                [p[0], p[1], previous[2]], [p[0], p[1], z]])
        else:
            points3 = [[*p, z] for p, z in zip(raw, endpoints_z)]
        corridor_cfg = config.get("corridors", {})
        minimum_width = draw(corridor_cfg.get("minimum_achieved_width", 4.), rng, minimum=.01)
        requested_width = draw(corridor_cfg.get("path_width", path["width"]), rng, minimum=.01)
        width = max(minimum_width, requested_width)
        for a, b in zip(points3, points3[1:]):
            if np.linalg.norm(np.asarray(a)-b) < 1e-8:
                continue
            if geometry == "curvilinear":
                # A connected union of analytic balls follows a sampled path.
                # This is a piecewise-spherical sweep, not an exact smooth tube.
                count = max(2, int(math.ceil(np.linalg.norm(np.asarray(a)-b)/(width*.33)))+1)
                for center in np.linspace(a, b, count):
                    corridors.append(solid.ellipsoid(center, [width/2]*3, "corridor"))
            elif geometry == "orthogonal":
                delta = np.abs(np.asarray(b)-a)
                lengths = np.where(delta > 1e-9, delta+width*.7, width)
                corridors.append(solid.box((np.asarray(a)+b)/2, lengths, role="corridor"))
            else:
                corridors.append(solid.segment_box(a, b, width))
        paths3.append({"points": points3, "width": width, "width_before_fit": width,
                       "width_requested": requested_width, "minimum_width_sampled": minimum_width})
    # Room centroids and path endpoints can differ in the legacy layouts.
    # Explicit short joining passages ensure every chamber is attached.
    nodes3 = np.asarray([p for path in paths3 for p in path["points"]])
    for room, record in zip(rooms, records):
        center = np.asarray(room["center"])
        nearest = nodes3[np.argmin(np.linalg.norm(nodes3-center, axis=1))]
        width = draw(config.get("corridors", {}).get("minimum_achieved_width", 4.), rng, minimum=.01)
        record["attachment_width_before_fit"] = width
        if geometry == "curvilinear":
            count = max(2, int(np.ceil(np.linalg.norm(center-nearest)/(width*.33)))+1)
            corridors.extend(solid.ellipsoid(p, [width/2]*3, "corridor")
                             for p in np.linspace(center, nearest, count))
        elif geometry == "orthogonal":
            current = center.copy()
            for axis in range(3):
                endpoint = current.copy()
                endpoint[axis] = nearest[axis]
                if np.linalg.norm(current-endpoint) > 1e-8:
                    lengths = np.full(3, width)
                    lengths[axis] += abs(current[axis]-endpoint[axis])
                    corridors.append(solid.box((current+endpoint)/2, lengths, role="corridor"))
                current = endpoint
        else:
            corridors.append(solid.segment_box(center, nearest, width))
    return {"positive": rooms+corridors, "negative": []}, rooms, corridors, records, paths3


def obstacles_3d(world, rooms, corridors, config, rng):
    records = []
    bounds = [0, 0, 0, *config["world_size"]]
    for population in config.get("obstacles", []):
        count = draw(population.get("count", 0), rng, integer=True, minimum=0, maximum=1000)
        kind = draw(population["type"], rng)
        if config["geometry"] == "orthogonal" and kind not in ("cuboid", "column", "orthogonal"):
            raise ValueError("orthogonal 3D worlds require orthogonal obstacles")
        achieved = 0
        for index in range(count):
            radius = draw(population.get("radius", .7), rng, minimum=.1)
            lengths = [draw(population.get(key, 2*radius), rng, minimum=.1)
                       for key in ("width", "length", "height")]
            host_clearance = draw(population.get("clearance", .4), rng, minimum=.4)
            corridor_clearance = draw(population.get("corridor_clearance", .12), rng, minimum=.12)
            obstacle_clearance = draw(population.get("obstacle_clearance", .25), rng, minimum=.25)
            accepted = False
            for trial in range(int(config.get("obstacle_trials", 2000))):
                placement = dict(population)
                placement.setdefault("placement", "room")
                center = obstacle_center(placement, bounds, rooms, corridors, rng, 3)
                if kind in ("cuboid", "column", "orthogonal"):
                    hole = solid.box(center, lengths, role="obstacle")
                elif kind in ("prism", "pyramid", "polyhedron"):
                    if kind == "pyramid":
                        x, y, z = lengths
                        vertices = np.asarray([[-x/2,-y/2,-z/2], [x/2,-y/2,-z/2],
                                               [x/2,y/2,-z/2], [-x/2,y/2,-z/2], [0,0,z/2]])+center
                        hole = solid.polyhedron(vertices, "obstacle")
                    else:
                        n = draw(population.get("vertices", 5), rng, integer=True, minimum=3, maximum=24)
                        polygon = polygon_shape("convex_polygon", center[:2], lengths[0], lengths[1], rng, n, options=population)
                        hole = solid.prism(list(polygon.exterior.coords)[:-1], center[2], lengths[2], "obstacle")
                elif kind in ("sphere", "ellipsoid"):
                    hole = solid.ellipsoid(center, [radius]*3 if kind == "sphere" else np.asarray(lengths)/2, "obstacle")
                else:
                    raise ValueError(f"unsupported 3D obstacle {kind}")
                low, high = solid.bounds(hole)
                corners = np.asarray(list(__import__("itertools").product(*zip(low, high))))
                host = next((i for i, room in enumerate(rooms)
                             if np.all(solid.primitive_clearance(room, corners) > host_clearance)), None)
                if host is None:
                    continue
                # Obstacles stay strictly inside a convex chamber and outside
                # every corridor and previously inserted obstacle. These
                # sufficient conditions protect the connectivity certificate.
                if not all(solid.disjoint_primitives(hole, corridor, corridor_clearance) for corridor in corridors):
                    continue
                if not all(solid.disjoint_primitives(hole, other, obstacle_clearance) for other in world["negative"]):
                    continue
                world["negative"].append(hole)
                records.append({"type": kind, "center": list(center), "lengths": lengths,
                                "radius": radius, "host_room": host, "clearance_requested": host_clearance,
                                "corridor_clearance_requested": corridor_clearance,
                                "obstacle_clearance_requested": obstacle_clearance,
                                "certificate": "strictly interior; disjoint from all corridors and obstacles"})
                achieved += 1
                accepted = True
                break
            if not accepted and population.get("exact_count", True):
                raise ValueError(f"could place only {achieved}/{count} {kind} obstacles")
        records.append({"population_summary": True, "type": kind,
                        "requested_count": count, "achieved_count": achieved})
    return records


def fit_solid(world, paths, size, allow_fitting=True):
    """Fit every primitive within the declared domain; preserve all overlaps."""
    bounds = [solid.bounds(p) for p in world["positive"]]
    low, high = np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)
    size = np.asarray(size, float)
    if not allow_fitting and (np.any(low < .5) or np.any(high > size-.5)):
        raise ValueError("sampled primitives leave the domain with layout.fit_mode=none")
    scale = float(min(1., np.min((size-1)/(high-low))))
    shift = np.maximum(.5-low*scale, 0.)
    shift -= np.maximum(high*scale+shift-(size-.5), 0.)
    for primitive in world["positive"]:
        primitive["center"] = (np.asarray(primitive["center"])*scale+shift).tolist()
        if primitive["kind"] == "polyhedron":
            planes = np.asarray(primitive["planes"])
            planes[:, :3] /= scale
            planes[:, 3] -= planes[:, :3] @ shift
            primitive["planes"] = planes.tolist()
            primitive["vertices"] = (np.asarray(primitive["vertices"])*scale+shift).tolist()
        else:
            primitive["radii"] = (np.asarray(primitive["radii"])*scale).tolist()
    for path in paths:
        path["points"] = (np.asarray(path["points"])*scale+shift).tolist()
        path["width"] *= scale
    return {"uniform_scale": scale, "translation": shift.tolist(),
            "actual_room_dimensions": "sampled dimensions multiplied by uniform_scale"}


def discrete(world, config):
    dimension = config["dimension"]
    shape = np.asarray(config.get("resolution", [100, 100] if dimension == 2 else [70, 70, 42]), int)
    size = np.asarray(config["world_size"], float)
    if len(shape) != dimension or np.any(shape < 4) or np.any(shape > 256):
        raise ValueError("resolution must have one integer in [4,256] per dimension")
    spacing = size/shape
    axes = [(np.arange(n)+.5)*h for n, h in zip(shape, spacing)]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = coordinates.reshape(-1, dimension)
    if dimension == 2:
        x, y = points.T
        cells = shapely.box(x-spacing[0]/2, y-spacing[1]/2,
                           x+spacing[0]/2, y+spacing[1]/2)
        free = shapely.covers(world, cells).reshape(tuple(shape))
        method = "entire square covered by continuous free-space polygon"
    else:
        radius = float(np.linalg.norm(spacing)/2)
        free = solid.contains(world, points, margin=radius+1e-9).reshape(tuple(shape))
        method = "1-Lipschitz CSG clearance lower bound exceeds voxel circumsphere radius"
    neighbors = int(config.get("neighbors", 4 if dimension == 2 else 6))
    connectivity = {2: {4: 1, 8: 2}, 3: {6: 1, 18: 2, 26: 3}}
    if neighbors not in connectivity[dimension]:
        raise ValueError("unsupported movement-neighbor convention")
    labels, components = ndimage.label(free, ndimage.generate_binary_structure(dimension, connectivity[dimension][neighbors]))
    if config.get("constraints", {}).get("connected", True) and components != 1:
        raise ValueError(f"conservative discrete representation has {components} components")
    return free, {"shape_xyz": shape.tolist(), "spacing_xyz": spacing.tolist(),
                  "origin": [0.]*dimension, "array_axes": "x,y" if dimension == 2 else "x,y,z",
                  "centers": "origin + (index + 0.5) * spacing",
                  "neighbors": neighbors, "free_components": int(components), "rule": method}


def generate(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output must be new or empty: {output}")
    requested_config = copy.deepcopy(config)
    config = copy.deepcopy(config)
    dimension = config.get("dimension")
    if dimension not in (2, 3) or len(config.get("world_size", [])) != dimension:
        raise ValueError("dimension/world_size mismatch")
    if config.get("layout_file") and ("layout" in config or "placement" in config.get("rooms", {})):
        raise ValueError("choose a frozen layout_file or generated layout/placement distributions")
    seed = config.get("seed", 0)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    started = time.perf_counter()
    failures = []
    for attempt in range(int(config.get("max_attempts", 24))):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt, 101]))
        try:
            config = copy.deepcopy(requested_config)
            config["world_size"] = [draw(v, rng, minimum=5.000001) if isinstance(v, (dict, list, tuple)) else v
                                    for v in config["world_size"]]
            if not all(np.isfinite(config["world_size"])) or min(config["world_size"]) <= 5:
                raise ValueError("world dimensions must be finite and greater than 5")
            if "resolution" in config:
                config["resolution"] = [draw(v, rng, integer=True, minimum=4, maximum=256)
                                        for v in config["resolution"]]
            if "curve_tolerance" in config:
                config["curve_tolerance"] = draw(config["curve_tolerance"], rng, minimum=1e-6)
            for key in ("family", "geometry"):
                if isinstance(config.get(key), dict):
                    config[key] = draw(config[key], rng)
            rooms_requested = draw(config.get("rooms", {}).get("count", 12), rng, integer=True, minimum=4, maximum=80)
            width = draw(config.get("corridors", {}).get("width", 5.), rng, minimum=1)
            loops = draw(config.get("corridors", {}).get("loop_fraction", .3), rng, minimum=0, maximum=1)
            if config.get("layout_file"):
                layout_path = Path(config["layout_file"])
                if not layout_path.is_absolute():
                    layout_path = PROJECT/layout_path
                layout = json.loads(layout_path.read_text())
            else:
                layout = create_layout(config, rng, rooms_requested, width, loops)
            if dimension == 2:
                world, rooms, corridors, room_records = realize_2d(layout, config, rng)
                world, obstacles, obstacle_records = obstacles_2d(world, rooms, corridors, config, rng)
                geometry = {"type": "Feature", "properties": {"coordinate_system": "world units; y increases upward"},
                            "geometry": mapping(world)}
                continuous_components = 1
                connectivity_certificate = {"method": "one valid Shapely Polygon", "components": 1}
                measure = {"free_area": float(world.area), "boundary_length": float(world.length),
                           "polygon_holes": len(world.interiors),
                           "boundary_vertices": sum(len(r.coords)-1 for r in [world.exterior, *world.interiors])}
            else:
                world, rooms, corridors, room_records, paths3 = realize_3d(layout, config, rng)
                fitting = fit_solid(world, paths3, config["world_size"],
                                    allow_fitting=layout.get("sampling", {}).get("fit", {}).get("mode") != "none")
                for record in room_records:
                    record["center"] = (np.asarray(record["center_before_solid_fit"])*fitting["uniform_scale"]
                                        +fitting["translation"]).tolist()
                    record["dimensions_after_solid_fit"] = [record[key]*fitting["uniform_scale"]
                                                            for key in ("width", "length", "height")]
                obstacle_records = obstacles_3d(world, rooms, corridors, config, rng)
                geometry = {"type": "constructive_solid", "dimension": 3,
                            "positive": world["positive"], "negative": world["negative"]}
                witnesses = solid.connection_certificate(world)
                connectivity_certificate = {"method": "positive-volume primitive overlaps; strictly interior disjoint obstacles; corridor protection",
                                             "components": 1, "overlap_witnesses": witnesses}
                continuous_components = 1
                measure = {"solid_primitives": len(world["positive"]),
                           "convex_obstacles": len(world["negative"]),
                           "minimum_continuous_cavities_certified": len(world["negative"]),
                           "continuous_tunnel_count": None}
            free, occupancy = discrete(world, config)
            break
        except ValueError as error:
            failures.append(str(error))
    else:
        raise RuntimeError(f"generation failed after {len(failures)} attempts: {failures[-3:]}")
    output.mkdir(parents=True, exist_ok=True)
    geometry_name = "geometry.geojson" if dimension == 2 else "solid.json"
    geometry_bytes = (json.dumps(geometry, sort_keys=True, separators=(",", ":"))+"\n").encode()
    (output/geometry_name).write_bytes(geometry_bytes)
    np.save(output/"free.npy", free)
    (output/"configuration.json").write_text(json.dumps(requested_config, indent=2)+"\n")
    (output/"resolved_configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    (output/"layout.json").write_text(json.dumps(layout, indent=2)+"\n")
    (output/"connectivity_certificate.json").write_text(json.dumps(connectivity_certificate, indent=2)+"\n")
    if dimension == 2:
        (output/"primitives.geojson").write_text(json.dumps({"type": "FeatureCollection", "features":
            [{"type": "Feature", "properties": {"role": role}, "geometry": mapping(shape)}
             for role, shapes in [("room", rooms), ("corridor", corridors), ("obstacle", obstacles)]
             for shape in shapes]}, separators=(",", ":"))+"\n")
    metadata = {"schema_version": 1, "dimension": dimension, "geometry": config["geometry"],
                "family": config["family"], "seed": seed, "accepted_attempt": attempt,
                "world_size": config["world_size"], "room_count_requested": rooms_requested,
                "room_count_achieved": len(rooms), "room_parameters": room_records,
                "corridor_count": len(layout["paths"]), "corridor_width_requested": width,
                "loop_fraction_requested": loops, "graph_cycle_rank": layout["graph_cycle_rank"],
                "graph_components": layout["graph_components"], "obstacle_records": obstacle_records,
                "obstacle_count_requested": sum(r["requested_count"] for r in obstacle_records if r.get("population_summary")),
                "obstacle_count_achieved": sum(r["achieved_count"] for r in obstacle_records if r.get("population_summary")),
                "continuous_components": continuous_components, "occupancy": occupancy,
                "free_cells": int(free.sum()), "occupancy_free_fraction": float(free.mean()),
                "conservative_discrete_measure": float(free.sum()*np.prod(occupancy["spacing_xyz"])),
                "curve_tolerance_requested": config.get("curve_tolerance", .025),
                "curve_approximation": ("analytic ellipsoids and piecewise-spherical sweeps" if dimension == 3 and config["geometry"] == "curvilinear"
                                        else "polygonal approximation; circular sagitta controlled, smooth Fourier shapes sampled without a global certified bound"
                                        if config["geometry"] == "curvilinear" else "planar primitives"),
                "geometry_sha256": hashlib.sha256(geometry_bytes).hexdigest(),
                "measurements": measure, "generation_seconds": time.perf_counter()-started,
                "rejected_attempts": failures, "validation_status": "passed"}
    metadata["parameter_sampling"] = {
        "configuration_requested": "configuration.json",
        "configuration_resolved": "resolved_configuration.json",
        "layout_model": layout.get("sampling", {}).get("model", "archived_six_family"),
        "layout_draws": "layout.json: sampling" if "sampling" in layout else None,
        "draw_scopes": "world-level settings once per attempt; room dimensions per room; path dimensions per path; obstacle dimensions per obstacle; placement per proposal",
        "acceptance": "feasibility rejection conditions the requested distributions; no silent clipping of sampled locations"}
    metadata["corridor_parameters"] = paths3 if dimension == 3 else [p["realization_2d"] for p in layout["paths"]]
    sources = [Path(__file__), Path(__file__).with_name("solid.py"),
               Path(__file__).with_name("layout.py"), Path(__file__).with_name("parameters.py"),
               Path(__file__).with_name("layout_distributions.py"),
               LEGACY/"generate.py", LEGACY/"generate_arbitrary.py"]
    metadata["source_hashes"] = {str(p.resolve().relative_to(PROJECT)):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if dimension == 3:
        metadata["paths3"] = paths3
        metadata["solid_fitting"] = fitting
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def load_world(folder):
    folder = Path(folder)
    metadata = json.loads((folder/"metadata.json").read_text())
    if metadata["dimension"] == 2:
        from shapely.geometry import shape
        world = shape(json.loads((folder/"geometry.geojson").read_text())["geometry"])
    else:
        world = json.loads((folder/"solid.json").read_text())
    return world, metadata
