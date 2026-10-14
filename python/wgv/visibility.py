"""Standalone Rust subprocess interface and compact CSR outputs."""
import json
import struct
import subprocess
import time
import platform
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from .generation import load_world
from .sampling import center_indices, prepare_discrete

PROJECT = Path(__file__).resolve().parents[2]
BINARY = PROJECT/"rust/visibility_engine/target/release/visibility_engine"
_PACKAGED_BINARY = PROJECT/"bin/macos-arm64/visibility_engine"
if not BINARY.exists() and platform.system()=="Darwin" and platform.machine()=="arm64" and _PACKAGED_BINARY.exists():
    BINARY = _PACKAGED_BINARY


def rust_input(folder, mode="continuous", observers=None, targets=None, sensor=None,
               allow_off_center=False):
    folder = Path(folder)
    world, metadata = load_world(folder)
    grid_mode = mode in ("discrete", "bresenham")
    if grid_mode and (observers is None or targets is None):
        source = folder / "observers.npy"
        count = len(np.load(source)) if source.exists() else 8
        prepare_discrete(folder, observers=count)
    if observers is None:
        observers = np.load(folder/("discrete_observers.npy" if grid_mode else "observers.npy"))
    if targets is None:
        targets = np.load(folder/("discrete_targets.npy" if grid_mode else "targets.npy"))
    payload = {"schema_version": 1, "dimension": metadata["dimension"], "mode": mode,
               "observers": np.asarray(observers).tolist(), "targets": np.asarray(targets).tolist(),
               "epsilon": 1e-8, "sensor": sensor or {}}
    if mode == "continuous":
        if metadata["dimension"] == 2:
            payload["polygons"] = [list(r.coords) for r in [world.exterior, *world.interiors]]
        else:
            payload["solid"] = world
    elif mode in ("discrete", "bresenham"):
        occupancy = metadata["occupancy"]
        free = np.load(folder/"free.npy")
        if not allow_off_center:
            center_indices(targets, free, occupancy)
            center_indices(observers, free, occupancy)
        payload["coordinate_rule"] = "legacy arbitrary points" if allow_off_center else "free grid centers"
        payload["grid"] = {"shape": occupancy["shape_xyz"], "spacing": occupancy["spacing_xyz"],
                           "origin": occupancy["origin"],
                           "free": free.ravel(order="C").tolist()}
    else:
        raise ValueError(f"unknown visibility mode {mode}")
    return payload


def read_csr(path):
    raw = Path(path).read_bytes()
    if raw[:8] != b"WGVCSR01":
        raise ValueError("invalid visibility CSR magic")
    if len(raw)<32:
        raise ValueError("truncated visibility CSR header")
    rows, columns, nonzero = struct.unpack_from("<QQQ", raw, 8)
    expected = 32+8*(rows+1+nonzero)
    if len(raw) != expected:
        raise ValueError("truncated visibility CSR")
    offsets = np.frombuffer(raw, dtype="<u8", count=rows+1, offset=32)
    indices = np.frombuffer(raw, dtype="<u8", count=nonzero, offset=32+8*(rows+1))
    if offsets[0] != 0 or offsets[-1] != nonzero or np.any(indices >= columns):
        raise ValueError("invalid CSR offsets/indices")
    if np.any(offsets[1:]<offsets[:-1]) or max(rows,columns,nonzero)>np.iinfo(np.int64).max:
        raise ValueError("invalid CSR sizes/offsets")
    offsets=offsets.astype(np.int64)
    indices=indices.astype(np.int64)
    return csr_matrix((np.ones(nonzero, bool), indices, offsets), shape=(rows, columns))


def compute(folder, mode="continuous", observers=None, targets=None, label=None, sensor=None, threads=4,
            allow_off_center=False):
    folder = Path(folder)
    label = label or mode
    payload = rust_input(folder, mode, observers, targets, sensor, allow_off_center=allow_off_center)
    input_path = folder/f"visibility_{label}_input.json"
    prefix = folder/f"visibility_{label}"
    input_path.write_text(json.dumps(payload, separators=(",", ":"))+"\n")
    started = time.perf_counter()
    completed = subprocess.run([str(BINARY), "--input", str(input_path), "--output", str(prefix),
                                "--threads", str(threads)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Rust visibility failed: {completed.stderr}")
    matrix = read_csr(prefix.with_suffix(".csr"))
    save_npz(folder/f"visibility_{label}.npz", matrix)
    union = np.asarray(matrix.sum(axis=0)).ravel() > 0
    stem="" if label=="continuous" else f"{label}_"
    visible_file=f"{stem}visible_target_indices.npy"
    unobserved_file=f"{stem}unobserved_target_indices.npy"
    np.save(folder/visible_file,np.flatnonzero(union))
    np.save(folder/unobserved_file,np.flatnonzero(~union))
    report = json.loads(prefix.with_suffix(".json").read_text())
    report.update({"subprocess_seconds": time.perf_counter()-started,
                   "union_count": int(union.sum()), "coverage_fraction": float(union.mean()),
                   "coverage_kind": "fraction of the specified finite target set",
                   "coordinate_rule": payload.get("coordinate_rule", "continuous points"),
                   "target_set_files":{"visible_union":visible_file,"unobserved":unobserved_file},
                   "prefix_coverage": [float(np.mean(np.asarray(matrix[:n].sum(axis=0)).ravel() > 0))
                                       for n in (1, 2, 4, 8) if n <= matrix.shape[0]]})
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    return matrix, report


def paired_representations(folder, target_rule="grid_centers"):
    """Compare both representations at grid centers; retain explicit legacy replay."""
    if target_rule not in ("grid_centers", "legacy_common_samples"):
        raise ValueError("unknown paired target rule")
    folder=Path(folder)
    if target_rule == "grid_centers":
        source_observers=np.load(folder/"observers.npy")
        sampling=prepare_discrete(folder, observers=len(source_observers))
        common=np.load(folder/"discrete_targets.npy")
        observers=np.load(folder/"discrete_observers.npy")
        np.save(folder/"paired_targets.npy",common)
        np.save(folder/"paired_observers.npy",observers)
        np.save(folder/"paired_target_cell_indices.npy",np.load(folder/"discrete_target_cell_indices.npy"))
        # These indices referred to the old continuous sample pool, not grid cells.
        (folder/"paired_source_target_indices.npy").unlink(missing_ok=True)
        continuous,c_report=compute(folder,mode="continuous",observers=observers,targets=common,label="paired_continuous")
        grid,g_report=compute(folder,mode="discrete",observers=observers,targets=common,label="paired_discrete")
        report={"target_rule":target_rule,"target_count":len(common),"observer_count":len(observers),
                "target_cell_indices":"paired_target_cell_indices.npy",
                "observer_adjustment":sampling["observer_selection"],
                "observer_displacements":sampling["observer_displacements"],
                "continuous_coverage":c_report["coverage_fraction"],"discrete_coverage":g_report["coverage_fraction"],
                "matrix_disagreements":int((continuous!=grid).nnz),
                "discrete_visible_continuous_blocked":int(grid.nnz-grid.multiply(continuous).nnz),
                "coordinate_arrays":["paired_observers.npy","paired_targets.npy"],
                "comparison_scope":"identical free grid-center coordinates in continuous geometry and conservative grid"}
        (folder/"paired_visibility.json").write_text(json.dumps(report,indent=2)+"\n")
        return report
    _,metadata=load_world(folder)
    points=np.load(folder/"targets.npy")
    source_observers=np.load(folder/"observers.npy")
    occupancy=metadata["occupancy"]
    free=np.load(folder/"free.npy")
    index=np.floor((points-np.asarray(occupancy["origin"]))/occupancy["spacing_xyz"]).astype(int)
    valid=np.all((index>=0)&(index<np.asarray(free.shape)),axis=1)
    legal=np.zeros(len(points),bool)
    legal[valid]=free[tuple(index[valid].T)]
    source_indices=np.flatnonzero(legal)
    common=points[legal]
    if len(common)<len(source_observers):
        raise ValueError("insufficient common legal targets for paired visibility")
    chosen=[]
    for observer in source_observers:
        order=np.argsort(np.linalg.norm(common-observer,axis=1),kind="stable")
        chosen.append(next(int(i) for i in order if int(i) not in chosen))
    observers=common[chosen]
    np.save(folder/"paired_targets.npy",common)
    np.save(folder/"paired_observers.npy",observers)
    np.save(folder/"paired_source_target_indices.npy",source_indices)
    (folder/"paired_target_cell_indices.npy").unlink(missing_ok=True)
    continuous,c_report=compute(folder,mode="continuous",observers=observers,targets=common,label="paired_continuous")
    grid,g_report=compute(folder,mode="discrete",observers=observers,targets=common,label="paired_discrete",allow_off_center=True)
    report={"target_rule":target_rule,"target_count":len(common),"observer_count":len(observers),
            "source_target_count":len(points),"source_target_indices":"paired_source_target_indices.npy",
            "observer_adjustment":"nearest distinct common legal target to each original observer",
            "observer_displacements":np.linalg.norm(observers-source_observers,axis=1).tolist(),
            "continuous_coverage":c_report["coverage_fraction"],"discrete_coverage":g_report["coverage_fraction"],
            "matrix_disagreements":int((continuous!=grid).nnz),
            "discrete_visible_continuous_blocked":int(grid.nnz-grid.multiply(continuous).nnz),
            "coordinate_arrays":["paired_observers.npy","paired_targets.npy"],
            "comparison_scope":"identical continuous coordinates; conservative grid and analytic geometry have separately recorded wall conventions"}
    (folder/"paired_visibility.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
