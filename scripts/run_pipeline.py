#!/usr/bin/env python3
"""Generate dungeon worlds, compute visibility and save experiment outputs."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"python"))
from wgv.generation import generate
from wgv.sampling import prepare
from wgv.visibility import compute, paired_representations, BINARY
from wgv.validation import validate


def showcase_configs():
    families = {
        (2, "orthogonal"): ["scattered_rooms", "branches", "room_clusters"],
        (2, "polygonal"): ["scattered_rooms", "caverns", "irregular_loops"],
        (2, "curvilinear"): ["winding_passages", "branches", "caverns"],
        (3, "orthogonal"): ["room_clusters", "branches", "irregular_loops"],
        (3, "polyhedral"): ["scattered_rooms", "caverns", "winding_passages"],
        (3, "curvilinear"): ["branches", "irregular_loops", "scattered_rooms"],
    }
    for (dimension, geometry), selected in families.items():
        for variant, family in enumerate(selected):
            config = {"dimension": dimension, "geometry": geometry, "family": family,
                "seed": 42+variant*11, "world_size": [50,50] if dimension==2 else [50,50,30],
                "resolution": [100,100] if dimension==2 else [70,70,42],
                "neighbors": 4 if dimension==2 else 6, "floors": 3,
                "curve_tolerance": .025, "max_attempts": 30,
                "rooms": {"count": 10 if family!="caverns" else 5,
                          "vertices": {"distribution":"uniform_int","min":5,"max":11}},
                "corridors": {"width": {"distribution":"uniform","min":5.,"max":6.2},
                              "loop_fraction": .2+variant*.2,
                              "minimum_achieved_width": 3.8 if dimension==2 else 4.4},
                "constraints":{"connected":True}}
            if dimension==2:
                kinds = {"orthogonal":["rectangle","orthogonal"],"polygonal":["convex_polygon","irregular_polygon"],
                         "curvilinear":["ellipse","rounded"]}[geometry]
                config["obstacles"] = [{"type":kind,"count":3+variant,
                    "radius":{"distribution":"uniform","min":.55,"max":1.05},
                    "placement":"room"} for kind in kinds]
            else:
                # Larger chambers leave protected interior space for obstacles.
                config["rooms"].update({"width":10., "length":10., "height":8.})
                kinds = {"orthogonal":["cuboid"],"polyhedral":["pyramid"],"curvilinear":["sphere"]}[geometry]
                config["obstacles"] = [{"type":kind,"count":3+variant,
                    "radius":{"distribution":"uniform","min":.45,"max":.7},
                    "placement":"room"} for kind in kinds]
            yield f"{dimension}d_{geometry}_{variant+1}", config


def collect(dataset, statistics=None):
    rows = []
    for folder in sorted(dataset.iterdir()):
        if not (folder/"metadata.json").exists():
            continue
        metadata = json.loads((folder/"metadata.json").read_text())
        sampling = json.loads((folder/"sampling.json").read_text()) if (folder/"sampling.json").exists() else {}
        visibility = json.loads((folder/"visibility_continuous.json").read_text()) if (folder/"visibility_continuous.json").exists() else {}
        paired = json.loads((folder/"paired_visibility.json").read_text()) if (folder/"paired_visibility.json").exists() else {}
        rows.append({"world":folder.name,"dimension":metadata["dimension"],"geometry":metadata["geometry"],
            "family":metadata["family"],"seed":metadata["seed"],"attempt":metadata["accepted_attempt"],
            "rooms_requested":metadata["room_count_requested"],"rooms_achieved":metadata["room_count_achieved"],
            "obstacles_requested":metadata["obstacle_count_requested"],"obstacles_achieved":metadata["obstacle_count_achieved"],
            "constructed_connections":metadata["corridor_count"],"graph_cycle_rank":metadata["graph_cycle_rank"],
            "continuous_components":metadata["continuous_components"],
            "discrete_components":metadata["occupancy"]["free_components"],
            "free_cells":metadata["free_cells"],"free_fraction":metadata["occupancy_free_fraction"],
            "continuous_measure":metadata["measurements"].get("free_area",sampling.get("volume_estimate")),
            "volume_standard_error":sampling.get("volume_standard_error"),
            "targets":sampling.get("target_count"),"observers":sampling.get("observer_count"),
            "sample_coverage":visibility.get("coverage_fraction"),
            "paired_targets":paired.get("target_count"),
            "paired_continuous_coverage":paired.get("continuous_coverage"),
            "paired_discrete_coverage":paired.get("discrete_coverage"),
            "paired_matrix_disagreements":paired.get("matrix_disagreements"),
            "rust_seconds":visibility.get("computation_seconds"),
            "subprocess_seconds":visibility.get("subprocess_seconds"),
            "generation_seconds":metadata["generation_seconds"],
            "validation": ("passed" if json.loads((folder/"validation.json").read_text()).get("passed")
                           else "failed") if (folder/"validation.json").exists() else "pending"})
    statistics = statistics or dataset/"statistics"
    statistics.mkdir(parents=True,exist_ok=True)
    if rows:
        with (statistics/"showcase.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    versions = {key:importlib.metadata.version(key) for key in ["numpy","scipy","shapely","matplotlib","scikit-image"]}
    environment = {"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
                   "processor":platform.processor(),"versions":versions,
                   "rustc": (subprocess.check_output(["rustc","--version"],text=True).strip()
                             if shutil.which("rustc") else "not installed; packaged binary"),
                   "hardware_note":"single host; visibility timing excludes JSON parsing and CSR writing",
                   "visibility_threads":4}
    (statistics/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--config",type=Path)
    source.add_argument("--showcase",action="store_true")
    parser.add_argument("--output",type=Path, help="World folder, or batch root with --showcase")
    parser.add_argument("--stage",choices=["generate","visibility","validate","render","all"],default="all")
    parser.add_argument("--targets",type=int,default=1500)
    parser.add_argument("--only",help="Restrict showcase world name")
    args=parser.parse_args()
    if args.targets < 1:
        parser.error("--targets must be positive")
    if args.only and not args.showcase:
        parser.error("--only requires --showcase")
    if args.config:
        config=json.loads(args.config.read_text())
        jobs=[(args.output or PROJECT/"outputs/example",config)]
    elif args.showcase:
        dataset=args.output or PROJECT/"outputs/showcase"
        jobs=[]
        for name,default in showcase_configs():
            if args.only and name!=args.only:
                continue
            config_path=PROJECT/"configs/showcase"/f"{name}.json"
            # Saved configuration files are authoritative and are never rewritten.
            config=json.loads(config_path.read_text()) if config_path.exists() else default
            jobs.append((dataset/name,config))
        if not jobs:
            parser.error(f"unknown showcase world: {args.only}")
    else:
        parser.error("use --showcase or --config")
    # Preflight the entire batch before creating any output or rebuilding code.
    for folder,config in jobs:
        if args.stage in ("all","generate") and (folder/"metadata.json").exists():
            saved=json.loads((folder/"configuration.json").read_text())
            if saved!=config:
                raise ValueError(f"existing world config differs: {folder}; choose a fresh output")
        if args.stage in ("all","visibility") and folder.exists():
            saved_queries=(list(folder.glob("visibility_*")) + list(folder.glob("*targets.npy"))
                           + list(folder.glob("*observers.npy")) + list(folder.glob("*sampling.json")))
            if saved_queries:
                raise ValueError(f"existing sensing results preserved: {folder}; choose a fresh --output "
                                 "for a new run, or --stage validate to check saved data")
    if args.stage in ("all","visibility") and not BINARY.exists():
        subprocess.run(["cargo","build","--locked","--release","--manifest-path",
                         str(PROJECT/"rust/visibility_engine/Cargo.toml")],check=True)
    for folder,config in jobs:
        print(f"{args.stage}: {folder.name}",flush=True)
        if args.stage in ("all","generate"):
            if (folder/"metadata.json").exists():
                print("  existing matching world preserved",flush=True)
            else:
                metadata=generate(config,folder)
                print(f"  {metadata['room_count_achieved']} rooms, {metadata['obstacle_count_achieved']} obstacles; "
                      f"attempt {metadata['accepted_attempt']}; {metadata['generation_seconds']:.2f} s",flush=True)
        if args.stage in ("all","visibility"):
            prepare(folder,count=args.targets)
            matrix,report=compute(folder)
            print(f"  union sample coverage {report['coverage_fraction']:.3f}; Rust {report['computation_seconds']:.4f} s",flush=True)
            _,grid_report=compute(folder,mode="discrete")
            print(f"  grid-center coverage {grid_report['coverage_fraction']:.3f}",flush=True)
            paired=paired_representations(folder)
            print(f"  matched continuous/discrete queries: {paired['target_count']} common targets",flush=True)
        if args.stage in ("all","validate"):
            validate(folder)
        if args.stage in ("all","render"):
            from wgv.visualization import render_world
            figure_stem=folder/"figures/world"
            render_world(folder,figure_stem)
    if args.showcase:
        collect(dataset)


if __name__=="__main__":
    main()
