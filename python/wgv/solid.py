"""Continuous 3D CSG: unions of convex polyhedra/ellipsoids, minus obstacles.

The exported mesh is a display approximation. Membership and visibility use
the analytic primitives, not the marching-cubes mesh or occupancy array.
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull


def polyhedron(vertices, role="room"):
    vertices = np.asarray(vertices, float)
    hull = ConvexHull(vertices)
    return {"kind": "polyhedron", "planes": hull.equations.tolist(),
            "vertices": vertices.tolist(), "faces": hull.simplices.tolist(),
            "center": vertices.mean(axis=0).tolist(), "role": role}


def box(center, lengths, rotation=None, role="room"):
    center, lengths = np.asarray(center, float), np.asarray(lengths, float)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    return polyhedron(center + (signs * lengths / 2) @ rotation.T, role)


def ellipsoid(center, radii, role="room"):
    return {"kind": "ellipsoid", "center": list(map(float, center)),
            "radii": list(map(float, radii)), "role": role}


def prism(vertices2, z, height, role="room"):
    v = np.asarray(vertices2, float)
    return polyhedron(np.vstack([np.column_stack([v, np.full(len(v), z-height/2)]),
                                 np.column_stack([v, np.full(len(v), z+height/2)])]), role)


def segment_box(a, b, width, height=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = b-a
    length = np.linalg.norm(delta)
    if length < 1e-8:
        return box(a, [width]*3, role="corridor")
    axis = delta/length
    reference = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([0., 1., 0.])
    side = np.cross(reference, axis)
    side /= np.linalg.norm(side)
    up = np.cross(axis, side)
    rotation = np.column_stack([axis, side, up])
    return box((a+b)/2, [length+width*.7, width, height or width], rotation, "corridor")


def primitive_clearance(primitive, points):
    """A Lipschitz lower bound on interior distance; negative outside."""
    points = np.atleast_2d(points).astype(float)
    if primitive["kind"] == "polyhedron":
        planes = np.asarray(primitive["planes"])
        return np.min(-(points @ planes[:, :3].T + planes[:, 3]) /
                      np.linalg.norm(planes[:, :3], axis=1), axis=1)
    radii = np.asarray(primitive["radii"])
    normalized = np.linalg.norm((points-np.asarray(primitive["center"]))/radii, axis=1)
    # min(radius)*(1 - normalized radius) is 1-Lipschitz.
    return radii.min()*(1-normalized)


def clearance(solid, points):
    points = np.atleast_2d(points).astype(float)
    result = np.full(len(points), -np.inf)
    for primitive in solid["positive"]:
        result = np.maximum(result, primitive_clearance(primitive, points))
    for obstacle in solid["negative"]:
        result = np.minimum(result, -primitive_clearance(obstacle, points))
    return result


def contains(solid, points, margin=0):
    return clearance(solid, points) > margin


def bounds(primitive):
    if primitive["kind"] == "polyhedron":
        vertices = np.asarray(primitive["vertices"])
        return vertices.min(axis=0), vertices.max(axis=0)
    center, radius = np.asarray(primitive["center"]), np.asarray(primitive["radii"])
    return center-radius, center+radius


def disjoint_boxes(a, b, margin=0):
    al, ah = bounds(a)
    bl, bh = bounds(b)
    return bool(np.any(ah+margin < bl) or np.any(bh+margin < al))


def disjoint_primitives(a, b, margin=0):
    """Sufficient separation certificate; never infer disjointness from misses."""
    if disjoint_boxes(a, b, margin):
        return True
    def radius(primitive):
        if primitive["kind"] == "ellipsoid":
            return max(primitive["radii"])
        return np.max(np.linalg.norm(np.asarray(primitive["vertices"])-primitive["center"], axis=1))
    # Clearance functions are 1-Lipschitz. If their value remains negative
    # throughout the other's enclosing ball, the primitives cannot intersect.
    return bool(primitive_clearance(a, b["center"])[0]+radius(b) < -margin or
                primitive_clearance(b, a["center"])[0]+radius(a) < -margin)


def primitive_interval(primitive, p, q, margin=0):
    """Independent Python line-clipping reference for Rust tests."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    direction = q-p
    lo, hi = 0., 1.
    if primitive["kind"] == "polyhedron":
        for nx, ny, nz, offset in primitive["planes"]:
            normal = np.array([nx, ny, nz])
            value = np.dot(normal, p)+offset+margin*np.linalg.norm(normal)
            slope = np.dot(normal, direction)
            if abs(slope) < 1e-13:
                if value >= 0:
                    return None
            elif slope > 0:
                hi = min(hi, -value/slope)
            else:
                lo = max(lo, -value/slope)
            if hi < lo:
                return None
    else:
        radii = np.asarray(primitive["radii"])-margin
        if np.any(radii <= 0):
            return None
        source = (p-np.asarray(primitive["center"]))/radii
        delta = direction/radii
        a, b, c = np.dot(delta, delta), 2*np.dot(source, delta), np.dot(source, source)-1
        if a < 1e-24:
            return (0., 1.) if c < 0 else None
        discriminant = b*b-4*a*c
        if discriminant < 0:
            return None
        root = np.sqrt(discriminant)
        lo, hi = max(lo, (-b-root)/(2*a)), min(hi, (-b+root)/(2*a))
    return (lo, hi) if hi >= lo else None


def visible(solid, p, q, epsilon=1e-8):
    if not np.all(contains(solid, [p, q])):
        return False
    if np.linalg.norm(np.asarray(p)-q) < epsilon:
        return True
    intervals = sorted(x for primitive in solid["positive"]
                       if (x := primitive_interval(primitive, p, q, epsilon)) is not None)
    reached = 0.
    for lo, hi in intervals:
        if lo > reached + 1e-10:
            return False
        reached = max(reached, hi)
    if reached < 1-1e-10:
        return False
    return not any(primitive_interval(obstacle, p, q, -epsilon) is not None
                   for obstacle in solid["negative"])


def overlap_witness(a, b, epsilon=1e-5):
    candidates = [a["center"], b["center"],
                  ((np.asarray(a["center"])+np.asarray(b["center"]))/2).tolist()]
    for primitive in (a, b):
        if primitive["kind"] == "polyhedron":
            candidates.extend(primitive["vertices"])
    points = np.asarray(candidates)
    valid = (primitive_clearance(a, points) > epsilon) & (primitive_clearance(b, points) > epsilon)
    if np.any(valid):
        return points[np.flatnonzero(valid)[0]].tolist()
    # For off-center thin intersections solve a small constrained feasibility
    # problem. A found witness is checked analytically; failure is conservative.
    from scipy.optimize import minimize
    result = minimize(lambda p: float(np.sum((p-np.asarray(a["center"]))**2)),
                      np.mean([a["center"], b["center"]], axis=0), method="SLSQP",
                      constraints=[{"type": "ineq", "fun": lambda p:
                                    float(primitive_clearance(a, p)[0])-epsilon},
                                   {"type": "ineq", "fun": lambda p:
                                    float(primitive_clearance(b, p)[0])-epsilon}],
                      options={"maxiter": 60, "ftol": 1e-9})
    if primitive_clearance(a, result.x)[0] > epsilon/2 and primitive_clearance(b, result.x)[0] > epsilon/2:
        return result.x.tolist()
    return None


def connection_certificate(solid):
    positive = solid["positive"]
    remaining = set(range(len(positive)))
    seen = {remaining.pop()}
    witnesses = []
    while remaining:
        progress = False
        for j in sorted(remaining):
            for i in sorted(seen):
                if disjoint_boxes(positive[i], positive[j]):
                    continue
                witness = overlap_witness(positive[i], positive[j])
                if witness is not None and contains(solid, [witness], margin=1e-6)[0]:
                    witnesses.append({"a": i, "b": j, "point": witness})
                    remaining.remove(j)
                    seen.add(j)
                    progress = True
                    break
        if not progress:
            raise ValueError(f"no positive-overlap certificate for {len(remaining)} solid primitives")
    return witnesses
