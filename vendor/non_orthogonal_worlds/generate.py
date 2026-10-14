"""Generate connected, non-orthogonal polygon worlds and conservative grid maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import Delaunay
import shapely
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
MODES = ('mixed', 'many_holes', 'hallways')
CROSS = ndimage.generate_binary_structure(2, 1)
FULL = np.ones((3, 3), dtype=bool)


@lru_cache(maxsize=8)
def grid_cells(size):
    y, x = np.mgrid[:size, :size]
    return shapely.box(x.ravel()-.5, y.ravel()-.5, x.ravel()+.5, y.ravel()+.5)


def rasterize(world, size):
    # A white pixel's WHOLE unit square is free. This also makes the connecting
    # segment between adjacent white centers collision-free in the vector world.
    shapely.prepare(world)
    return shapely.covers(world, grid_cells(size)).reshape(size, size)


def grid_metrics(free):
    _, components = ndimage.label(free, CROSS)
    labels, count = ndimage.label(~free, FULL)
    outer = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    areas = np.bincount(labels.ravel())
    holes = [int(areas[i]) for i in range(1, count+1) if i not in outer]
    return int(components), holes


def polygon_room(center, radius, rng, vertices=None):
    count = vertices or int(rng.integers(5, 10))
    angles = np.linspace(0, 2*math.pi, count, endpoint=False)
    angles += rng.uniform(-.16, .16, count)
    rotation = rng.uniform(0, 2*math.pi)
    radii = radius * rng.uniform(.83, 1.13, count)
    local = np.column_stack((np.cos(angles)*radii,
                             np.sin(angles)*radii*rng.uniform(.67, 1.05)))
    rot = np.array([[math.cos(rotation), -math.sin(rotation)],
                    [math.sin(rotation), math.cos(rotation)]])
    return Polygon(local @ rot.T + center).convex_hull


def graph_edges(centers, rng, loop_fraction):
    triangles = Delaunay(centers).simplices
    edges = {tuple(sorted((int(t[i]), int(t[(i+1)%3])))) for t in triangles for i in range(3)}
    ordered = sorted(edges)
    weights = {e: np.linalg.norm(centers[e[0]]-centers[e[1]])*rng.uniform(.8, 1.2)
               for e in ordered}
    parent = list(range(len(centers)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = []
    for a, b in sorted(ordered, key=lambda e: (weights[e], e)):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tree.append((a, b))
    unused = sorted(edges-set(tree), key=lambda e: (weights[e], e))
    # Additional graph edges create routes around enclosed wall islands.
    count = int(math.ceil(len(unused)*loop_fraction))
    selected = tree + unused[:count]
    return selected, len(selected)-len(centers)+1


def remove_tiny_holes(world, minimum_area):
    if world.geom_type != 'Polygon':
        return world
    rings = [r.coords for r in world.interiors if Polygon(r).area >= minimum_area]
    return Polygon(world.exterior.coords, rings)


def deployments(free, world, seed):
    # Five different starts, spread by distance along the actual walkable grid.
    rng = np.random.default_rng(seed)
    candidates = np.argwhere(free & (ndimage.distance_transform_edt(free) >= 1.4))
    if len(candidates) < 5:
        candidates = np.argwhere(free)
    first = tuple(candidates[int(rng.integers(len(candidates)))])
    chosen = [first]
    best = np.full(free.shape, np.inf)
    for _ in range(4):
        distances = np.full(free.shape, -1, dtype=int)
        distances[chosen[-1]] = 0
        queue = [chosen[-1]]
        for y, x in queue:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < len(free) and 0 <= nx < len(free) and free[ny, nx] and distances[ny, nx] < 0:
                    distances[ny, nx] = distances[y, x]+1
                    queue.append((ny, nx))
        best = np.minimum(best, distances)
        scores = best[candidates[:, 0], candidates[:, 1]]
        options = np.flatnonzero(scores == scores.max())
        chosen.append(tuple(candidates[int(rng.choice(options))]))
    points = [[int(x), int(y)] for y, x in chosen]
    assert len({tuple(p) for p in points}) == 5
    assert all(world.contains(Point(p)) for p in points)
    return points


def make_world(size, index, seed, config):
    mode = config['mode'] if config['mode'] != 'varied' else MODES[index % len(MODES)]
    default_rooms = 16 if size == 50 else 36
    room_count = config['rooms'] or default_rooms
    minimum_holes = config['min_holes'] if config['min_holes'] is not None else (8 if size == 50 else 24)
    for attempt in range(150):
        rng = np.random.default_rng(np.random.SeedSequence([seed, size, index, attempt]))
        columns = math.ceil(math.sqrt(room_count))
        rows = math.ceil(room_count/columns)
        margin = 5.5 if size == 50 else 8.5
        xs, ys = np.linspace(margin, size-1-margin, columns), np.linspace(margin, size-1-margin, rows)
        spacing = min(xs[1]-xs[0], ys[1]-ys[0])
        centers = np.array([(x, y) for y in ys for x in xs])[:room_count]
        centers += rng.uniform(-.20, .20, centers.shape)*spacing
        corridor_width = config['hallway_width'] or ((3.1 if size == 50 else 3.8) * rng.uniform(.96, 1.12))
        fraction = config['loop_fraction']
        if fraction is None:
            fraction = {'mixed': .72, 'many_holes': .96, 'hallways': .70}[mode]
        edges, cycles = graph_edges(centers, rng, fraction)
        radius_factor = {'mixed': .36, 'many_holes': .32, 'hallways': .25}[mode]
        rooms = [polygon_room(c, spacing*radius_factor*rng.uniform(.82, 1.12), rng) for c in centers]
        corridors, paths = [], []
        oblique_length, total_length = 0., 0.
        for a, b in edges:
            start, end = centers[a], centers[b]
            delta = end-start
            normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
            middle = (start+end)/2 + normal*rng.uniform(-.1, .1)*np.linalg.norm(delta)
            points = np.array([start, middle, end])
            line = LineString(points)
            width = corridor_width*rng.uniform(.93, 1.07)
            corridors.append(line.buffer(width/2, quad_segs=3, join_style=2))
            paths.append({'rooms': [a, b], 'points': points.tolist(), 'width': width})
            for p, q in zip(points, points[1:]):
                d = q-p
                length = np.linalg.norm(d)
                angle = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
                total_length += length
                if 10 <= angle <= 80:
                    oblique_length += length
        world = unary_union(rooms+corridors).intersection(shapely.box(1.5, 1.5, size-2.5, size-2.5))
        world = remove_tiny_holes(world, 4.)
        if world.geom_type != 'Polygon' or not world.is_valid:
            continue
        # Add isolated, rotated obstacles when the hallway cycles alone don't
        # meet the requested hole count. Reject cuts that break raster access.
        for _ in range(100):
            if len(world.interiors) >= minimum_holes:
                break
            center = rng.uniform(3, size-4, 2)
            hole = polygon_room(center, rng.uniform(1.15, 1.85)*(size/50)**.3, rng, vertices=6)
            if world.contains(hole.buffer(1.4)):
                candidate = world.difference(hole)
                candidate_free = rasterize(candidate, size)
                if grid_metrics(candidate_free)[0] == 1:
                    world = candidate
        free = rasterize(world, size)
        components, hole_areas = grid_metrics(free)
        fill = float(free.mean())
        if components != 1 or len(hole_areas) < minimum_holes or any(area < 3 for area in hole_areas):
            continue
        if not .20 <= fill <= .75 or oblique_length/total_length < .35:
            continue
        # Sealed exterior ensures every grid hole is genuinely enclosed.
        if np.any(free[0]) or np.any(free[-1]) or np.any(free[:, 0]) or np.any(free[:, -1]):
            continue
        starts = deployments(free, world, np.random.SeedSequence([seed, size, index, 999]))
        meta = {'version': 1, 'world': f'dungeon_{index}.png', 'size': size,
                'world_seed': seed, 'world_index': index, 'accepted_attempt': attempt,
                'mode': mode, 'room_count': room_count, 'hallway_count': len(paths),
                'hallway_width': corridor_width, 'loop_fraction': fraction,
                'graph_cycles': cycles, 'minimum_holes': minimum_holes,
                'polygon_holes': len(world.interiors), 'grid_holes': len(hole_areas),
                'grid_hole_areas': sorted(hole_areas), 'free_grid_cells': int(free.sum()),
                'free_fraction': fill, 'free_components': components,
                'continuous_area': float(world.area),
                'oblique_hallway_length_fraction': float(oblique_length/total_length),
                'deployment_points': starts, 'room_centers': centers.tolist(),
                'hallways': paths, 'room_polygons': [list(p.exterior.coords) for p in rooms],
                'coordinates': 'Integer grid centers (x,y); y increases downward; cell edges at half integers.',
                'raster_rule': 'White iff the entire unit cell square is covered by the free-space polygon.',
                'geometry_file': f'dungeon_{index}.geojson',
                'generator_version': 1, 'shapely_version': shapely.__version__}
        return world, free, meta
    raise RuntimeError(f'Unable to make size={size}, index={index} with these constraints in 150 attempts.')


def svg_world(world, size, starts=None):
    def ring_path(ring):
        coords = list(ring.coords)
        return 'M'+'L'.join(f'{x:.8f},{y:.8f}' for x, y in coords)+'Z'
    path = ''.join(ring_path(r) for r in [world.exterior, *world.interiors])
    dots = ''
    if starts:
        dots = ''.join(f'<circle cx="{x}" cy="{y}" r=".55" fill="#ed7aa7"/>' for x, y in starts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-.5 -.5 {size} {size}">'
            f'<rect x="-.5" y="-.5" width="{size}" height="{size}" fill="#132031"/>'
            f'<path d="{path}" fill="#e7f0f6" fill-rule="evenodd"/>{dots}</svg>')


def vector_image(world, size, pixels):
    image = Image.new('RGB', (pixels, pixels), '#132031')
    draw = ImageDraw.Draw(image)
    def coords(ring):
        return [((x+.5)*pixels/size, (y+.5)*pixels/size) for x, y in ring.coords]
    draw.polygon(coords(world.exterior), fill='#e7f0f6')
    for ring in world.interiors:
        draw.polygon(coords(ring), fill='#132031')
    return image


def generate_job(job):
    size, index, seed, config, output = job
    world, free, metadata = make_world(size, index, seed, config)
    folder = output/f'{size}x{size}'
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'dungeon_{index}'
    png = folder/(stem+'.png')
    Image.fromarray(free.astype(np.uint8)*255, mode='L').save(png)
    feature = {'type': 'Feature', 'properties': {'coordinate_system': 'local grid units, y down',
                                              'size': size, 'world': stem},
               'geometry': mapping(world)}
    geojson = folder/(stem+'.geojson')
    geojson.write_text(json.dumps(feature, separators=(',', ':'))+'\n')
    (folder/(stem+'.svg')).write_text(svg_world(world, size))
    metadata['png_sha256'] = hashlib.sha256(png.read_bytes()).hexdigest()
    metadata['geometry_sha256'] = hashlib.sha256(geojson.read_bytes()).hexdigest()
    (folder/(stem+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    return {k: metadata[k] for k in ('world', 'size', 'world_index', 'mode', 'room_count',
                                    'hallway_count', 'hallway_width', 'grid_holes', 'polygon_holes',
                                    'free_grid_cells', 'free_fraction', 'oblique_hallway_length_fraction',
                                    'accepted_attempt', 'png_sha256', 'geometry_sha256')}


def gallery(output, entries):
    cards = []
    for entry in entries:
        size = entry['size']
        stem = Path(entry['world']).stem
        prefix = f'{size}x{size}/{stem}'
        label = f'{size} × {size} · {stem} · {entry["mode"].replace("_", " ")}'
        cards.append(f'<article data-size="{size}" data-mode="{entry["mode"]}"><h2>{html.escape(label)}</h2>'
                     f'<img loading="lazy" src="{prefix}.svg" data-vector="{prefix}.svg" data-grid="{prefix}.png" alt="{label}">'
                     f'<p>{entry["grid_holes"]} grid holes · {entry["hallway_count"]} hallways · {entry["free_fraction"]:.0%} free cells</p>'
                     f'<a href="{prefix}.png" download>Grid PNG</a> · <a href="{prefix}.svg">Vector SVG</a> · '
                     f'<a href="{prefix}.geojson" download>Geometry</a> · <a href="{prefix}.json">Metadata</a></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Non-orthogonal worlds</title><style>
body{margin:0;background:#0b1320;color:#e7f0f6;font:16px system-ui;padding:28px}main{max-width:1550px;margin:auto}
h1{margin:0 0 8px}p{color:#aebdd0}label{display:inline-block;margin:10px 22px 15px 0}select{font:inherit;padding:7px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}article{background:#192638;padding:16px;border-radius:10px}
article[hidden]{display:none}h2{font-size:17px}img{width:100%;aspect-ratio:1;image-rendering:pixelated;background:#132031}a{color:#68d5ec}
</style><main><h1>Non-orthogonal worlds</h1><p>Angled rooms, connecting hallways, enclosed obstacles. Light = free space; dark = walls.</p>
<label>Size <select id="size"><option value="all">Both sizes</option>SIZE_OPTIONS</select></label>
<label>Layout <select id="mode"><option value="all">All layouts</option><option value="mixed">Mixed</option><option value="many_holes">Many holes</option><option value="hallways">Hallways</option></select></label>
<label>View <select id="view"><option value="vector">Continuous geometry</option><option value="grid">Conservative grid map</option></select></label>
<p id="count"></p><div class="cards">CARDS</div><p>White grid cells fit entirely inside the continuous free-space polygon. Each grid is connected and has five saved deployment positions.</p></main>
<script>const size=document.getElementById('size'),mode=document.getElementById('mode'),view=document.getElementById('view');
function update(){let n=0;document.querySelectorAll('article').forEach(card=>{card.hidden=(size.value!=='all'&&card.dataset.size!==size.value)||(mode.value!=='all'&&card.dataset.mode!==mode.value);if(!card.hidden)n++;const img=card.querySelector('img');img.src=img.dataset[view.value];});document.getElementById('count').textContent=n+' worlds';} [size,mode,view].forEach(el=>el.onchange=update);update();</script></html>'''
    options = ''.join(f'<option value="{s}">{s} × {s}</option>' for s in sorted({e['size'] for e in entries}))
    (output/'index.html').write_text(page.replace('SIZE_OPTIONS', options).replace('CARDS', ''.join(cards)))
    try:
        title_font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 29)
        small_font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    except OSError:
        title_font = small_font = ImageFont.load_default()
    from shapely.geometry import shape
    for size in sorted({e['size'] for e in entries}):
        selected = [e for e in entries if e['size'] == size][:6]
        sheet = Image.new('RGB', (1536, 1140), '#0b1320')
        draw = ImageDraw.Draw(sheet)
        draw.text((22, 18), f'Non-orthogonal worlds · {size} × {size}', fill='#e7f0f6', font=title_font)
        draw.text((22, 62), 'Continuous geometry · light floor, dark walls · conservative grid PNGs also saved', fill='#aebdd0', font=small_font)
        for i, entry in enumerate(selected):
            stem = Path(entry['world']).stem
            world = shape(json.loads((output/f'{size}x{size}'/(stem+'.geojson')).read_text())['geometry'])
            x, y = 16+(i%3)*512, 112+(i//3)*512
            sheet.paste(vector_image(world, size, 480), (x, y))
            draw.text((x, y+483), f'{stem} · {entry["grid_holes"]} holes · {entry["hallway_count"]} hallways', fill='#e7f0f6', font=small_font)
        sheet.save(output/f'preview_{size}x{size}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', choices=[50, 100], default=[50, 100])
    parser.add_argument('--count', type=int, default=60, help='Worlds per size')
    parser.add_argument('--seed', type=int, default=0, help='World-generation seed, independent of deployment seed')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--mode', choices=['varied', *MODES], default='varied')
    parser.add_argument('--rooms', type=int, help='Default: 16 at 50×50, 36 at 100×100')
    parser.add_argument('--min-holes', type=int, help='Minimum grid holes; default 8 or 24, by size')
    parser.add_argument('--hallway-width', type=float, help='Nominal hallway width in cell units')
    parser.add_argument('--loop-fraction', type=float, help='Fraction of additional graph edges, in [0,1]')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'generated')
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or args.seed < 0:
        parser.error('Count/workers must be positive; seed must be nonnegative.')
    if args.rooms is not None and not 9 <= args.rooms <= 80:
        parser.error('Rooms must be between 9 and 80.')
    if args.min_holes is not None and args.min_holes < 0:
        parser.error('Minimum holes must be nonnegative.')
    if args.hallway_width is not None and (not math.isfinite(args.hallway_width) or args.hallway_width < 2.8):
        parser.error('Hallway width must be finite and at least 2.8 cell units.')
    if args.loop_fraction is not None and not 0 <= args.loop_fraction <= 1:
        parser.error('Loop fraction must be in [0,1].')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error('Output must be in a subfolder of non_orthogonal_worlds/.')
    if output.exists() and any(output.iterdir()):
        parser.error('Output is not empty. Choose a new output directory to preserve existing worlds.')
    output.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ('rooms', 'min_holes', 'hallway_width', 'loop_fraction', 'mode')}
    (output/'configuration.json').write_text(json.dumps(vars(args), default=str, indent=2))
    jobs = [(s, i, args.seed, config, output) for s in sorted(set(args.sizes)) for i in range(args.count)]
    entries = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(generate_job, job) for job in jobs]
        for future in as_completed(futures):
            entry = future.result()
            entries.append(entry)
            print(f'[{len(entries)}/{len(jobs)}] {entry["size"]}×{entry["size"]} {entry["world"]}: '
                  f'{entry["grid_holes"]} holes, {entry["hallway_count"]} hallways (attempt {entry["accepted_attempt"]})', flush=True)
    entries.sort(key=lambda e: (e['size'], e['world_index']))
    with (output/'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    gallery(output, entries)
    print(f'Gallery: {output/"index.html"}', flush=True)


if __name__ == '__main__':
    main()
