"""Irregular worlds with independent silhouettes, clustered rooms, and varied topology."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union

import generate as base

ROOT = Path(__file__).resolve().parent
FAMILIES = ('scattered_rooms', 'room_clusters', 'winding_passages', 'caverns', 'branches', 'irregular_loops')


def organic_room(center, radius, rng):
    n = int(rng.integers(5, 13))
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)+rng.uniform(0, 2*np.pi)
    angles += rng.uniform(-.12, .12, n)
    radii = rng.uniform(.65, 1.25, n)*radius
    polygon = Polygon(np.column_stack((np.cos(angles)*radii, np.sin(angles)*radii*rng.uniform(.55, 1.25))))
    polygon = affinity.rotate(polygon, rng.uniform(0, 360), origin=(0, 0))
    return affinity.translate(polygon, *center)


def smooth_path(points, rounds=2):
    points = np.asarray(points, dtype=float)
    for _ in range(rounds):
        combined = np.empty((2*(len(points)-1), 2))
        combined[::2] = .75*points[:-1]+.25*points[1:]
        combined[1::2] = .25*points[:-1]+.75*points[1:]
        points = np.vstack((points[0], combined, points[-1]))
    return points


def raw_layout(size, family, rng, config):
    rooms, paths = [], []
    width = config['hallway_width'] or rng.uniform(4.2, 6.0)*(size/50)**.18

    def room(center, radius):
        rooms.append(organic_room(center, radius, rng))

    def path(points, multiplier=1., smooth=False):
        points = smooth_path(points) if smooth else np.asarray(points)
        paths.append({'points': points.tolist(), 'width': width*multiplier})

    def connect(centers, fraction):
        edges, _ = base.graph_edges(np.asarray(centers), rng, fraction)
        for a, b in edges:
            p, q = np.asarray(centers[a]), np.asarray(centers[b])
            delta = q-p
            normal = np.array([-delta[1], delta[0]])/max(np.linalg.norm(delta), 1e-9)
            bend = (p+q)/2+normal*np.linalg.norm(delta)*rng.uniform(-.28, .28)
            path([p, bend, q], rng.uniform(.82, 1.15), rng.random() < .45)

    if family == 'scattered_rooms':
        count = config['rooms'] or int(rng.integers(10, 23 if size == 50 else 38))
        angles = rng.uniform(0, 2*np.pi, count)
        radii = np.sqrt(rng.random(count))*size*.38
        radii *= 1+.25*np.sin(3*angles+rng.uniform(0, 2*np.pi))
        centers = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)*rng.uniform(.6, 1.25)))
        for center in centers:
            radius = np.clip(rng.lognormal(np.log(size*.085), .45), size*.042, size*.17)
            room(center, radius)
        connect(centers, config['loop_fraction'] if config['loop_fraction'] is not None else rng.uniform(.12, .60))
    elif family == 'room_clusters':
        clusters = int(rng.integers(2, 6))
        angles = rng.uniform(0, 2*np.pi, clusters)
        radii = rng.uniform(.16, .37, clusters)*size
        anchors = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
        centers = []
        for anchor in anchors:
            count = max(2, round(config['rooms']/clusters)) if config['rooms'] else int(rng.integers(3, 8))
            cluster = anchor+rng.normal(0, rng.uniform(.045, .10)*size, (count, 2))
            centers.extend(cluster)
            for center in cluster:
                room(center, size*rng.uniform(.055, .14))
        connect(centers, config['loop_fraction'] if config['loop_fraction'] is not None else rng.uniform(.02, .28))
    elif family == 'caverns':
        count = config['rooms'] or int(rng.integers(3, 8))
        centers = rng.normal(0, size*rng.uniform(.12, .22), (count, 2))
        for center in centers:
            room(center, size*rng.uniform(.12, .25))
        connect(centers, config['loop_fraction'] if config['loop_fraction'] is not None else rng.uniform(0, .35))
    elif family == 'winding_passages':
        count = int(rng.integers(5, 10))
        t = np.linspace(-1, 1, count)
        backbone = np.column_stack((t*size*.36, np.sin(t*np.pi*rng.uniform(.7, 1.6)+rng.uniform(0, 6))*size*rng.uniform(.10, .24)))
        path(backbone, rng.uniform(1.15, 1.65), True)
        for i in rng.choice(len(backbone), min(4, len(backbone)), replace=False):
            room(backbone[i], size*rng.uniform(.075, .15))
        for _ in range(int(rng.integers(3, 8))):
            i = int(rng.integers(1, len(backbone)-1))
            start = backbone[i]
            angle = rng.uniform(0, 2*np.pi)
            end = start+np.array([math.cos(angle), math.sin(angle)])*size*rng.uniform(.13, .30)
            path([start, (start+end)/2+rng.normal(0, size*.03, 2), end], rng.uniform(.8, 1.05), True)
            room(end, size*rng.uniform(.05, .105))
    elif family == 'branches':
        room([0, 0], size*rng.uniform(.12, .20))
        count = int(rng.integers(3, 7))
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)+rng.uniform(-.45, .45, count)
        for angle in angles:
            previous = np.zeros(2)
            for j in range(int(rng.integers(1, 4))):
                angle += rng.uniform(-.7, .7)
                endpoint = previous+np.array([math.cos(angle), math.sin(angle)])*size*rng.uniform(.11, .21)
                path([previous, (previous+endpoint)/2+rng.normal(0, size*.022, 2), endpoint], rng.uniform(.8, 1.12), rng.random() < .6)
                room(endpoint, size*rng.uniform(.06, .14))
                previous = endpoint
    else:
        loops = []
        count = int(rng.integers(2, 5 if size == 50 else 7))
        centers = rng.uniform(-.22, .22, (count, 2))*size
        for center in centers:
            n = int(rng.integers(9, 19))
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radius = size*rng.uniform(.11, .24)
            radii = radius*(1+.23*np.sin(angles*rng.integers(2, 5)+rng.uniform(0, 6)))
            vertices = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)*rng.uniform(.5, 1.35)))+center
            loops.append(vertices)
            path(np.vstack((vertices, vertices[0])), rng.uniform(.8, 1.1), rng.random() < .5)
            for vertex in vertices[rng.choice(n, int(rng.integers(1, 4)), replace=False)]:
                room(vertex, size*rng.uniform(.06, .13))
        for i in range(1, len(loops)):
            distances = np.sum((loops[i][:, None, :]-loops[i-1][None, :, :])**2, axis=2)
            a, b = np.unravel_index(distances.argmin(), distances.shape)
            path([loops[i][a], loops[i-1][b]])
    halls = [LineString(p['points']).buffer(p['width']/2, quad_segs=4) for p in paths]
    world = unary_union(rooms+halls)
    return world, rooms, paths, width


def fit_layout(world, rooms, paths, size, rng):
    angle = rng.uniform(0, 360)
    world = affinity.rotate(world, angle, origin=(0, 0))
    minx, miny, maxx, maxy = world.bounds
    scale = min(rng.uniform(.78, .94)*size, size-4.5)/max(maxx-minx, maxy-miny)
    extent = np.array([maxx-minx, maxy-miny])*scale
    shift = rng.uniform(2, size-2-extent)-np.array([minx, miny])*scale
    radians = math.radians(angle)
    matrix = [scale*math.cos(radians), -scale*math.sin(radians),
              scale*math.sin(radians), scale*math.cos(radians), *shift]
    # 'world' above was rotated only to calculate its bounds. Apply the same
    # full affine transform to original room/path primitives and their union.
    rooms = [affinity.affine_transform(room, matrix) for room in rooms]
    fitted_paths = []
    for path in paths:
        line = affinity.affine_transform(LineString(path['points']), matrix)
        fitted_paths.append({'points': list(line.coords), 'width': path['width']*scale})
    geometry = unary_union(rooms+[LineString(p['points']).buffer(p['width']/2, quad_segs=4) for p in fitted_paths])
    return geometry, rooms, fitted_paths, scale


def make_world(size, index, seed, config):
    family = config['family'] if config['family'] != 'varied' else FAMILIES[index % len(FAMILIES)]
    targets = {'scattered_rooms': (6, 15), 'room_clusters': (6, 15), 'winding_passages': (5, 12),
               'caverns': (8, 19), 'branches': (4, 11), 'irregular_loops': (5, 14)}
    layout_rng = np.random.default_rng(np.random.SeedSequence([seed, size, index, 2002]))
    low, high = targets[family]
    minimum_holes = config['min_holes'] if config['min_holes'] is not None else int(layout_rng.integers(low, high)*(size/50)**.85)
    for attempt in range(180):
        rng = np.random.default_rng(np.random.SeedSequence([seed, size, index, attempt, 2]))
        world, rooms, paths, width = raw_layout(size, family, rng, config)
        if world.geom_type != 'Polygon' or not world.is_valid:
            continue
        world, rooms, paths, scale = fit_layout(world, rooms, paths, size, rng)
        world = base.remove_tiny_holes(world, 3.5)
        if world.geom_type != 'Polygon' or not world.is_valid:
            continue
        free = base.rasterize(world, size)
        if base.grid_metrics(free)[0] != 1 or not .14 < free.mean() < .82:
            continue
        hole_components = base.grid_metrics(free)[1]
        if any(area < 3 for area in hole_components):
            continue
        # Holes vary in size and shape, including elongated wall islands.
        for trial in range(300):
            if len(hole_components) >= minimum_holes:
                break
            # Sample roomy regions with subcell jitter. Adapt obstacle size to
            # local clearance so narrow branches don't force dense, broad maps.
            choices = np.argwhere(ndimage.distance_transform_edt(free) >= 2.8)
            if not len(choices):
                break
            y, x = choices[int(rng.integers(len(choices)))]
            center = np.array([x, y])+rng.uniform(-.45, .45, 2)
            clearance = world.boundary.distance(Point(center))
            maximum = min(4.8*(size/50)**.55, (clearance-1.5)/1.35)
            if maximum < 1.05:
                continue
            radius = rng.uniform(1.05, maximum)
            hole = organic_room(center, radius, rng)
            if hole.area < 2.6 or not world.contains(hole.buffer(1.5, quad_segs=3)):
                continue
            candidate = world.difference(hole)
            candidate_free = base.rasterize(candidate, size)
            components, areas = base.grid_metrics(candidate_free)
            if components == 1 and len(areas) > len(hole_components) and all(a >= 3 for a in areas):
                world, free, hole_components = candidate, candidate_free, areas
        components, hole_areas = base.grid_metrics(free)
        if len(hole_areas) < minimum_holes or any(area < 3 for area in hole_areas) or not .12 < free.mean() < .78:
            continue
        starts = base.deployments(free, world, np.random.SeedSequence([seed, size, index, 2999]))
        oblique = total = 0.
        for path in paths:
            for dx, dy in np.diff(np.asarray(path['points']), axis=0):
                length = float(np.hypot(dx, dy))
                total += length
                if min(abs(dx), abs(dy)) > .1763*max(abs(dx), abs(dy)):
                    oblique += length
        metadata = {'version': 2, 'generator_version': 2, 'world': f'dungeon_{index}.png', 'size': size,
                    'world_seed': seed, 'world_index': index, 'accepted_attempt': attempt, 'mode': family,
                    'room_count': len(rooms), 'hallway_count': len(paths), 'hallway_width': width*scale,
                    'hallway_width_min': min(p['width'] for p in paths), 'hallway_width_max': max(p['width'] for p in paths),
                    'minimum_holes': minimum_holes, 'polygon_holes': len(world.interiors), 'grid_holes': len(hole_areas),
                    'grid_hole_areas': sorted(hole_areas), 'free_grid_cells': int(free.sum()),
                    'free_fraction': float(free.mean()), 'free_components': components,
                    'continuous_area': world.area, 'convex_hull_fill': world.area/world.convex_hull.area,
                    'oblique_hallway_length_fraction': oblique/total, 'deployment_points': starts,
                    'hallways': paths, 'room_centers': [list(room.centroid.coords[0]) for room in rooms],
                    'room_polygons': [list(room.exterior.coords) for room in rooms],
                    'coordinates': 'Integer grid centers (x,y); y increases downward; cell edges at half integers.',
                    'raster_rule': 'White iff the entire unit cell square is covered by the free-space polygon.',
                    'geometry_file': f'dungeon_{index}.geojson', 'shapely_version': shapely.__version__}
        return world, free, metadata
    raise RuntimeError(f'No feasible {family} world for {size} index {index}, target holes {minimum_holes}, after 180 attempts.')


def generate_job(job):
    size, index, seed, config, output = job
    world, free, metadata = make_world(size, index, seed, config)
    folder = output/f'{size}x{size}'
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'dungeon_{index}'
    png = folder/(stem+'.png')
    Image.fromarray(free.astype(np.uint8)*255).save(png)
    feature = {'type': 'Feature', 'properties': {'coordinate_system': 'local grid units, y down',
                                              'size': size, 'world': stem, 'family': metadata['mode']}, 'geometry': mapping(world)}
    geojson = folder/(stem+'.geojson')
    geojson.write_text(json.dumps(feature, separators=(',', ':'))+'\n')
    (folder/(stem+'.svg')).write_text(base.svg_world(world, size))
    metadata['png_sha256'] = hashlib.sha256(png.read_bytes()).hexdigest()
    metadata['geometry_sha256'] = hashlib.sha256(geojson.read_bytes()).hexdigest()
    (folder/(stem+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    return {k: metadata[k] for k in ('world', 'size', 'world_index', 'mode', 'room_count', 'hallway_count',
                                    'hallway_width', 'grid_holes', 'polygon_holes', 'free_grid_cells', 'free_fraction',
                                    'oblique_hallway_length_fraction', 'accepted_attempt', 'png_sha256', 'geometry_sha256')}


def gallery(output, entries):
    base.gallery(output, entries)
    destination = output/'index.html'
    page = destination.read_text()
    start = page.index('<label>Layout ')
    end = page.index('</label>', start)+len('</label>')
    options = ''.join(f'<option value="{family}">{family.replace("_", " ").title()}</option>' for family in FAMILIES)
    page = page[:start]+f'<label>Layout <select id="mode"><option value="all">All layouts</option>{options}</select></label>'+page[end:]
    page = page.replace('<h1>Non-orthogonal worlds</h1>', '<h1>Arbitrary non-orthogonal worlds</h1>')
    page = page.replace('Angled rooms, connecting hallways, enclosed obstacles.', 'Scattered rooms, clusters, winding passages, caverns, branches, and irregular loops.')
    destination.write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', choices=[50, 100], default=[50, 100])
    parser.add_argument('--count', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--family', choices=['varied', *FAMILIES], default='varied')
    parser.add_argument('--rooms', type=int, help='For scattered rooms, clusters, and caverns')
    parser.add_argument('--min-holes', type=int, help='Override varied hole-count targets')
    parser.add_argument('--hallway-width', type=float, help='Width before the world is fitted into its grid')
    parser.add_argument('--loop-fraction', type=float, help='Extra connections for scattered rooms, clusters, and caverns')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'arbitrary_v2')
    args = parser.parse_args()
    if min(args.count, args.workers) < 1 or args.seed < 0:
        parser.error('Positive count/workers and nonnegative seed required.')
    if args.rooms is not None and args.rooms < 4:
        parser.error('At least four rooms required.')
    if args.min_holes is not None and args.min_holes < 0:
        parser.error('Minimum holes must be nonnegative.')
    if args.hallway_width is not None and (not math.isfinite(args.hallway_width) or args.hallway_width < 3):
        parser.error('Hallway width must be finite and at least 3.')
    if args.loop_fraction is not None and not 0 <= args.loop_fraction <= 1:
        parser.error('Loop fraction must be in [0,1].')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or (output.exists() and any(output.iterdir())):
        parser.error('Choose a new or empty output subfolder in non_orthogonal_worlds/.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'configuration.json').write_text(json.dumps(vars(args), default=str, indent=2))
    config = {k: getattr(args, k) for k in ('family', 'rooms', 'min_holes', 'hallway_width', 'loop_fraction')}
    jobs = [(size, index, args.seed, config, output) for size in sorted(set(args.sizes)) for index in range(args.count)]
    entries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_job, job) for job in jobs]
        for future in as_completed(futures):
            entry = future.result()
            entries.append(entry)
            print(f'[{len(entries)}/{len(jobs)}] {entry["size"]} {entry["world"]} {entry["mode"]}: '
                  f'{entry["grid_holes"]} holes, {entry["free_fraction"]:.0%} free, attempt {entry["accepted_attempt"]}', flush=True)
    entries.sort(key=lambda e: (e['size'], e['world_index']))
    with (output/'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    gallery(output, entries)
    print(f'Gallery: {output/"index.html"}', flush=True)


if __name__ == '__main__':
    main()
