import json
from pathlib import Path
import numpy as np
import pytest
from wgv.generation import generate,load_world
from wgv.sampling import sample,membership
from wgv.validation import validate
from wgv import solid

PROJECT=Path(__file__).resolve().parents[1]


def test_midpoint_elevation_ties_are_stable_when_requested():
    from shapely.geometry import box
    from wgv.generation import realize_3d
    for perturbation in (-1e-14, 1e-14):
        layout = {"centers": [[15, 25], [35, 25]], "room_elevations": [10, 20],
                  "template_rooms": [list(box(13, 23, 17, 27).exterior.coords),
                                     list(box(33, 23, 37, 27).exterior.coords)],
                  "paths": [{"points": [[15, 25], [25+perturbation, 26], [35, 25]], "width": 5}]}
        config = {"geometry": "polyhedral", "world_size": [50, 50, 30],
                  "rooms": {"width": 6, "length": 6, "height": 5},
                  "junction_elevation_rule": "lowest_index"}
        _, _, _, _, paths = realize_3d(layout, config, np.random.default_rng(42))
        assert paths[0]["points"][1][2] == 10


def test_deterministic_geometry_and_conservative_grid(tmp_path):
    config=json.loads((PROJECT/"configs/showcase/2d_polygonal_1.json").read_text())
    a=generate(config,tmp_path/"a");b=generate(config,tmp_path/"b")
    assert a["geometry_sha256"]==b["geometry_sha256"]
    assert np.array_equal(np.load(tmp_path/"a/free.npy"),np.load(tmp_path/"b/free.npy"))
    assert validate(tmp_path/"a")["passed"]


def test_output_protection(tmp_path):
    folder=tmp_path/"existing";folder.mkdir();(folder/"data.txt").write_text("preserve")
    with pytest.raises(ValueError):
        generate({"dimension":2},folder)
    assert (folder/"data.txt").read_text()=="preserve"


def test_source_provenance_survives_import_through_a_symlink(tmp_path, monkeypatch):
    import hashlib
    import wgv.generation as module
    alias = tmp_path/"project_alias"
    alias.symlink_to(PROJECT, target_is_directory=True)
    monkeypatch.setattr(module, "__file__", str(alias/"python/wgv/generation.py"))
    config = json.loads((PROJECT/"configs/showcase/2d_polygonal_1.json").read_text())
    metadata = module.generate(config, tmp_path/"world")
    assert metadata["source_hashes"]["python/wgv/generation.py"] == hashlib.sha256(
        (PROJECT/"python/wgv/generation.py").read_bytes()).hexdigest()
    assert validate(tmp_path/"world")["passed"]


@pytest.mark.parametrize("method",["uniform","lattice","poisson_disk","clearance","gaussian"])
def test_sampling_validity_and_reproducibility(method):
    from shapely.geometry import box
    world=box(0,0,10,10)
    a=sample(world,2,[10,10],30,41,method)
    b=sample(world,2,[10,10],30,41,method)
    assert np.array_equal(a,b) and np.all(membership(world,2,a))
    if method=="poisson_disk":
        from scipy.spatial.distance import pdist
        assert np.min(pdist(a))>=.6


@pytest.mark.parametrize("dimension,geometry",[
    (2,"orthogonal"),(2,"polygonal"),(2,"curvilinear"),
    (3,"orthogonal"),(3,"polyhedral"),(3,"curvilinear")])
def test_showcase_class_artifacts(tmp_path,dimension,geometry):
    config=json.loads((PROJECT/f"configs/showcase/{dimension}d_{geometry}_1.json").read_text())
    folder=tmp_path/"world"
    generate(config,folder)
    report=validate(folder)
    assert report["passed"] and report["discrete_components"]==1
    world,m=load_world(folder)
    assert m["obstacle_count_achieved"]==m["obstacle_count_requested"]
    if dimension==3:
        extent=np.asarray(m["world_size"])
        for p in world["positive"]:
            low,high=solid.bounds(p)
            assert np.all(low>=0) and np.all(high<=extent)
        # Multiple elevations must be joined in actual geometry, and
        # the conservative six-connected volume spans those elevations.
        free=np.load(folder/"free.npy")
        occupied=np.argwhere(free)
        z=(occupied[:,2]+.5)*m["occupancy"]["spacing_xyz"][2]
        assert np.ptp(z)>15
