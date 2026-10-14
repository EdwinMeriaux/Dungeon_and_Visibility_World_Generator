"""Configurable six-family layouts; the archived generator remains unchanged.

Every scalar construction choice below uses the common distribution sampler.
Locations can instead be drawn from a joint law in physical or normalized units.
This opt-in model does not change seeds or geometry of the existing showcases.
"""
import math
import numpy as np
from scipy.spatial import Delaunay, QhullError
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from .parameters import draw, draw_point


PARAMETER_GROUPS = {
    "template": ["radius", "vertices", "phase", "angle_jitter", "radial_factor", "aspect", "rotation"],
    "scatter": ["angle", "radius_fraction", "modulation", "phase", "aspect"],
    "clusters": ["count", "angle", "radius_fraction", "spread_fraction", "offset_x", "offset_y"],
    "caverns": ["spread_fraction", "offset_x", "offset_y"],
    "winding": ["nodes", "length_fraction", "frequency", "phase", "amplitude_fraction",
                "backbone_rooms", "backbone_room_index", "branches", "branch_start_index",
                "branch_angle", "branch_length_fraction", "bend_x_fraction", "bend_y_fraction"],
    "branches": ["count", "angle_jitter", "segments", "turn", "length_fraction",
                 "bend_x_fraction", "bend_y_fraction"],
    "loops": ["count", "center_x_fraction", "center_y_fraction", "vertices", "radius_fraction",
              "modulation", "harmonics", "phase", "aspect", "rooms", "room_vertex_index"],
    "graph": ["weight_multiplier"],
    "paths": ["width_multiplier", "bend_fraction", "smooth", "smoothing_rounds"],
    "fit": ["rotation", "span_fraction", "translation_x_fraction", "translation_y_fraction"],
}
PARAMETER_KEYS = {f"{group}.{key}" for group, keys in PARAMETER_GROUPS.items() for key in keys}


class LayoutSampler:
    def __init__(self, config, rng):
        self.rng = rng
        self.specs = config.get("layout", {}).get("parameters", {})
        if not isinstance(self.specs, dict):
            raise ValueError("layout.parameters must be a mapping of named scalar laws")
        unknown = set(self.specs)-PARAMETER_KEYS
        if unknown:
            raise ValueError(f"unknown layout parameters: {sorted(unknown)}")
        self.records = []

    def get(self, key, default, **bounds):
        spec = self.specs.get(key, default)
        value = draw(spec, self.rng, **bounds)
        self.records.append({"parameter": key, "specification": spec, "value": value})
        return value

    def select(self, key, candidates, count):
        """Sample distinct indices; custom discrete laws are conditioned on availability."""
        if key not in self.specs:
            selected = self.rng.choice(candidates, count, replace=False).tolist()
        else:
            selected = []
            for _ in range(10000):
                index = self.get(key, 0, integer=True, minimum=0, maximum=max(candidates))
                if index in candidates and index not in selected:
                    selected.append(index)
                if len(selected) == count:
                    break
            else:
                raise ValueError(f"{key} cannot supply {count} distinct available indices")
        self.records.append({"parameter": key, "selected_indices": selected})
        return selected


def placement_dimension(spec, dimension):
    """A planar law can be shared with 3D while elevation is supplied separately."""
    if dimension == 2:
        return 2
    if not isinstance(spec, dict):
        return len(spec)
    kind = spec.get("type", spec.get("distribution", "independent"))
    if kind == "independent":
        return 3 if "z" in spec else 2
    if kind == "mixture":
        dims = {placement_dimension(s, dimension) for s in spec["components"]}
        if len(dims) != 1:
            raise ValueError("all placement mixture components must have the same dimension")
        return dims.pop()
    return len(spec["value"] if kind == "fixed" else spec["mean"] if kind in ("gaussian", "normal") else spec["min"])


def construct(config, rng, requested_rooms, width, loop_fraction):
    from .layout import original_layout
    family = config["family"]
    size = float(min(config["world_size"][:2]))
    s = LayoutSampler(config, rng)
    g = s.get
    rooms, centers, elevations, paths, room_draws = [], [], [], [], []
    placement = config.get("rooms", {}).get("placement")
    placement_kind = placement.get("type", "independent") if isinstance(placement, dict) else None
    absolute_placement = placement is not None and placement_kind != "family"
    units = placement.get("coordinates", "world") if isinstance(placement, dict) else "world"
    if units not in ("world", "normalized"):
        raise ValueError("room placement coordinates must be world or normalized")

    def path(points, multiplier=(.82, 1.15), smooth=None, role="layout"):
        points = np.asarray(points, float)
        factor = g("paths.width_multiplier", multiplier, minimum=.01)
        smooth = g("paths.smooth", {"distribution": "categorical", "values": [0, 1], "weights": [.55, .45]}
                   if smooth is None else int(smooth), integer=True, minimum=0, maximum=1)
        rounds = g("paths.smoothing_rounds", 2, integer=True, minimum=0, maximum=8)
        if smooth:
            points = original_layout.smooth_path(points, rounds)
        paths.append({"points": points.tolist(), "width": width*factor, "role": role})

    def room(suggested, radius_spec, attach=False):
        center = np.asarray(suggested, float)
        z = None
        if placement is not None:
            law = placement["offset"] if placement_kind == "family" else placement
            dim = placement_dimension(law, config["dimension"])
            if dim not in (2, config["dimension"]) or (placement_kind == "family" and dim != 2):
                raise ValueError("family offsets must be planar; use rooms.elevation for height")
            sampled = draw_point(law, rng, dim)
            if units == "normalized":
                sampled *= np.asarray(config["world_size"][:dim])
            center = center+sampled if placement_kind == "family" else sampled[:2]
            if dim == 3:
                z = float(sampled[2])
            s.records.append({"parameter": "rooms.placement", "value": sampled.tolist(), "coordinates": "world"})
        radius = g("template.radius", radius_spec, minimum=.01)
        n = g("template.vertices", [5, 12], integer=True, minimum=3, maximum=64)
        phase = g("template.phase", [0, 2*math.pi])
        angles = np.linspace(0, 2*math.pi, n, endpoint=False)+phase
        angles += [g("template.angle_jitter", [-.12, .12]) for _ in range(n)]
        radii = radius*np.asarray([g("template.radial_factor", [.65, 1.25], minimum=.01) for _ in range(n)])
        aspect = g("template.aspect", [.55, 1.25], minimum=.01)
        rotation = g("template.rotation", [0, 360])
        shape = Polygon(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii*aspect]))
        shape = affinity.rotate(shape, rotation, origin=(0, 0))
        shape = affinity.translate(shape, *center)
        if not shape.is_valid:
            raise ValueError("sampled template is not a valid polygon")
        rooms.append(shape); centers.append(center); elevations.append(z)
        room_draws.append({"center_before_fit": center.tolist(), "elevation": z,
                           "radius": radius, "vertices": n, "aspect": aspect, "rotation": rotation})
        if attach and np.linalg.norm(center-np.asarray(suggested)) > 1e-8:
            path([suggested, center], multiplier=1, smooth=False, role="placement_attachment")
        return center

    def connect(points):
        points = np.asarray(points)
        try:
            triangles = Delaunay(points).simplices
            candidates = {tuple(sorted((int(t[i]), int(t[(i+1)%3])))) for t in triangles for i in range(3)}
        except QhullError:
            # Collinear point distributions still need a connected graph.
            candidates = {(i, j) for i in range(len(points)) for j in range(i+1, len(points))}
        weights = {e: np.linalg.norm(points[e[0]]-points[e[1]])*
                   g("graph.weight_multiplier", [.8, 1.2], minimum=.01) for e in sorted(candidates)}
        parent = list(range(len(points)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        tree = []
        for a, b in sorted(candidates, key=lambda e: (weights[e], e)):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb; tree.append((a, b))
        unused = sorted(candidates-set(tree), key=lambda e: (weights[e], e))
        for a, b in tree+unused[:int(math.ceil(len(unused)*loop_fraction))]:
            p, q = points[a], points[b]
            delta = q-p
            normal = np.asarray([-delta[1], delta[0]])/max(np.linalg.norm(delta), 1e-9)
            bend = (p+q)/2+normal*np.linalg.norm(delta)*g("paths.bend_fraction", [-.28, .28])
            path([p, bend, q])

    if family == "scattered_rooms":
        phase = g("scatter.phase", [0, 2*math.pi])
        modulation = g("scatter.modulation", .25, minimum=0, maximum=.9)
        aspect = g("scatter.aspect", [.6, 1.25], minimum=.01)
        for _ in range(requested_rooms):
            a = g("scatter.angle", [0, 2*math.pi])
            r = size*g("scatter.radius_fraction", {"distribution": "beta", "alpha": 2, "beta": 1,
                                                  "min": 0, "max": .38}, minimum=0)
            r *= 1+modulation*math.sin(3*a+phase)
            room([r*math.cos(a), r*math.sin(a)*aspect], [size*.042, size*.17])
        connect(centers)
    elif family == "room_clusters":
        count = g("clusters.count", [2, min(5, requested_rooms)], integer=True, minimum=1, maximum=requested_rooms)
        # Unlike the legacy rounding rule, allocate exactly the requested rooms.
        for i in range(count):
            a = g("clusters.angle", [0, 2*math.pi])
            r = size*g("clusters.radius_fraction", [.16, .37], minimum=0)
            anchor = np.asarray([r*math.cos(a), r*math.sin(a)])
            std = size*g("clusters.spread_fraction", [.045, .10], minimum=0)
            for _ in range(requested_rooms//count+int(i < requested_rooms % count)):
                offset = [g(f"clusters.offset_{axis}", {"distribution": "normal", "mean": 0, "std": std}) for axis in ("x", "y")]
                room(anchor+offset, [size*.055, size*.14])
        connect(centers)
    elif family == "caverns":
        std = size*g("caverns.spread_fraction", [.12, .22], minimum=0)
        for _ in range(requested_rooms):
            center = [g(f"caverns.offset_{axis}", {"distribution": "normal", "mean": 0, "std": std}) for axis in ("x", "y")]
            room(center, [size*.12, size*.25])
        connect(centers)
    elif family == "winding_passages":
        n = g("winding.nodes", [5, 9], integer=True, minimum=3, maximum=100)
        t = np.linspace(-1, 1, n)
        length = g("winding.length_fraction", .36, minimum=.01)
        frequency = g("winding.frequency", [.7, 1.6], minimum=.01)
        phase = g("winding.phase", [0, 6])
        amplitude = g("winding.amplitude_fraction", [.10, .24], minimum=0)
        backbone = np.column_stack([t*size*length, np.sin(t*math.pi*frequency+phase)*size*amplitude])
        path(backbone, [1.15, 1.65], True)
        count = g("winding.backbone_rooms", min(4, n), integer=True, minimum=1, maximum=n)
        for i in s.select("winding.backbone_room_index", list(range(n)), count):
            room(backbone[i], [size*.075, size*.15], attach=True)
        count = g("winding.branches", [3, 7], integer=True, minimum=0, maximum=100)
        for _ in range(count):
            i = g("winding.branch_start_index", [1, n-2], integer=True, minimum=1, maximum=n-2)
            start = backbone[i]
            a = g("winding.branch_angle", [0, 2*math.pi])
            length = size*g("winding.branch_length_fraction", [.13, .30], minimum=.01)
            end = start+np.asarray([math.cos(a), math.sin(a)])*length
            offset = size*np.asarray([g(f"winding.bend_{axis}_fraction", {"distribution": "normal", "mean": 0, "std": .03}) for axis in ("x", "y")])
            path([start, (start+end)/2+offset, end], [.8, 1.05], True)
            room(end, [size*.05, size*.105], attach=True)
    elif family == "branches":
        room([0, 0], [size*.12, size*.20], attach=True)
        count = g("branches.count", [3, 6], integer=True, minimum=1, maximum=100)
        for a in np.linspace(0, 2*math.pi, count, endpoint=False):
            a += g("branches.angle_jitter", [-.45, .45])
            previous = np.zeros(2)
            for _ in range(g("branches.segments", [1, 3], integer=True, minimum=1, maximum=100)):
                a += g("branches.turn", [-.7, .7])
                length = size*g("branches.length_fraction", [.11, .21], minimum=.01)
                end = previous+np.asarray([math.cos(a), math.sin(a)])*length
                offset = size*np.asarray([g(f"branches.bend_{axis}_fraction", {"distribution": "normal", "mean": 0, "std": .022}) for axis in ("x", "y")])
                path([previous, (previous+end)/2+offset, end], [.8, 1.12])
                room(end, [size*.06, size*.14], attach=True)
                previous = end
    elif family == "irregular_loops":
        loops = []
        for _ in range(g("loops.count", [2, 4], integer=True, minimum=1, maximum=100)):
            center = size*np.asarray([g(f"loops.center_{axis}_fraction", [-.22, .22]) for axis in ("x", "y")])
            n = g("loops.vertices", [9, 18], integer=True, minimum=4, maximum=100)
            angles = np.linspace(0, 2*math.pi, n, endpoint=False)
            radius = size*g("loops.radius_fraction", [.11, .24], minimum=.01)
            modulation = g("loops.modulation", .23, minimum=0, maximum=.9)
            harmonics = g("loops.harmonics", [2, 4], integer=True, minimum=1, maximum=30)
            phase = g("loops.phase", [0, 6])
            radii = radius*(1+modulation*np.sin(angles*harmonics+phase))
            aspect = g("loops.aspect", [.5, 1.35], minimum=.01)
            vertices = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)*aspect])+center
            loops.append(vertices)
            path(np.vstack([vertices, vertices[0]]), [.8, 1.1])
            count = g("loops.rooms", [1, 3], integer=True, minimum=1, maximum=n)
            for i in s.select("loops.room_vertex_index", list(range(n)), count):
                room(vertices[i], [size*.06, size*.13], attach=True)
        for a, b in zip(loops, loops[1:]):
            distances = np.sum((a[:, None]-b[None])**2, axis=2)
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            path([a[i], b[j]], 1, False)
    else:
        raise ValueError(f"unknown layout family {family}")

    def union():
        return unary_union(rooms+[LineString(p["points"]).buffer(p["width"]/2, quad_segs=4) for p in paths])
    world = union()
    if world.geom_type != "Polygon" or not world.is_valid:
        raise ValueError("sampled layout union is disconnected or invalid")
    mode = config.get("layout", {}).get("fit_mode", "none" if absolute_placement else "fit")
    transform = {"mode": mode, "uniform_scale": 1., "rotation": 0., "translation": [0., 0.]}
    if mode == "fit":
        angle = g("fit.rotation", [0, 360])
        lowx, lowy, highx, highy = affinity.rotate(world, angle, origin=(0, 0)).bounds
        span = g("fit.span_fraction", [.78, .94], minimum=.01, maximum=1)
        scale = min(span*size, size-4.5)/max(highx-lowx, highy-lowy)
        extent = np.asarray([highx-lowx, highy-lowy])*scale
        fractions = np.asarray([g(f"fit.translation_{axis}_fraction", [0, 1], minimum=0, maximum=1) for axis in ("x", "y")])
        shift = 2+fractions*(size-4-extent)-np.asarray([lowx, lowy])*scale
        radians = math.radians(angle)
        rotation = np.asarray([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
        matrix = [*(scale*rotation).ravel(), *shift]
        rooms = [affinity.affine_transform(r, matrix) for r in rooms]
        centers = (np.asarray(centers)@(scale*rotation).T+shift).tolist()
        for p in paths:
            p["points"] = (np.asarray(p["points"])@(scale*rotation).T+shift).tolist(); p["width"] *= scale
        transform.update(uniform_scale=scale, rotation=angle, translation=shift.tolist())
        world = union()
    elif mode != "none":
        raise ValueError("layout.fit_mode must be fit or none")
    return rooms, paths, np.asarray(centers), elevations, {
        "model": "distribution_configurable_v1", "parameter_draws": s.records,
        "room_draws": room_draws, "room_placement_specification": placement,
        "fit": transform, "count_rule": "exact in room-first families; passage-derived otherwise"}
