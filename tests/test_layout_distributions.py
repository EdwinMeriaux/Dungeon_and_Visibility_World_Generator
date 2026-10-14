import json
from pathlib import Path
import numpy as np
import pytest
from wgv.generation import generate
from wgv.layout import create_layout, FAMILIES
from wgv.validation import validate

PROJECT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family",FAMILIES)
def test_every_layout_family_accepts_named_parameter_laws(family):
    config={"dimension":2,"world_size":[50,50],"family":family,
            "layout":{"parameters":{"paths.width_multiplier":[1.1,1.1],"paths.smooth":0}},
            "rooms":{}}
    a=create_layout(config,np.random.default_rng(12),10,5,.3)
    b=create_layout(config,np.random.default_rng(12),10,5,.3)
    assert a==b
    scale=a["sampling"]["fit"]["uniform_scale"]
    assert all(np.isclose(p["width"],5*1.1*scale) for p in a["paths"])
    assert a["graph_components"]==1
    if family in ("scattered_rooms","room_clusters","caverns"):
        assert a["room_count_achieved"]==10


@pytest.mark.parametrize("name",["2d_independent_rooms","2d_mixture_rooms","3d_independent_rooms"])
def test_distribution_examples_repeat_and_keep_sampled_sizes(tmp_path,name):
    config=json.loads((PROJECT/f"configs/distributions/{name}.json").read_text())
    metadata=generate(config,tmp_path/name)
    repeated=generate(config,tmp_path/(name+"_repeat"))
    assert repeated["geometry_sha256"] == metadata["geometry_sha256"]
    assert np.array_equal(np.load(tmp_path/name/"free.npy"),
                          np.load(tmp_path/(name+"_repeat")/"free.npy"))
    assert validate(tmp_path/name)["passed"]
    assert metadata["parameter_sampling"]["layout_model"]=="distribution_configurable_v1"
    for r in metadata["room_parameters"]:
        for axis in ("width","length"):
            assert config["rooms"][axis][0]<=r[axis]<=config["rooms"][axis][1]
        if metadata["dimension"]==3:
            assert 8<=r["center"][2]<=22
            assert np.array_equal(r["center"],r["center_before_solid_fit"])
            assert metadata["solid_fitting"]["uniform_scale"]==1
    if name=="2d_independent_rooms":
        assert all(12<=r["center"][0]<=38 and 12<=r["center"][1]<=38 for r in metadata["room_parameters"])
    assert (tmp_path/name/"configuration.json").read_text()==json.dumps(config,indent=2)+"\n"


@pytest.mark.parametrize("dimension", [2,3])
def test_gaussian_room_and_vertex_counts_generate_valid_worlds(tmp_path,dimension):
    name="2d_independent_rooms" if dimension==2 else "3d_independent_rooms"
    config=json.loads((PROJECT/f"configs/distributions/{name}.json").read_text())
    config["rooms"]["count"]={"distribution":"normal","mean":8.5,"std":1,"min":7,"max":10}
    config["rooms"]["vertices"]={"distribution":"gaussian","mean":7,"std":1,"min":5,"max":9}
    folder=tmp_path/"gaussian_counts"
    metadata=generate(config,folder)
    assert 7<=metadata["room_count_requested"]<=10
    assert metadata["room_count_achieved"]==metadata["room_count_requested"]
    if dimension==2:
        assert all(5<=r["vertices_requested"]<=9 for r in metadata["room_parameters"])
    assert validate(folder)["passed"]


def test_normalized_locations_and_collinear_room_connections(tmp_path):
    config=json.loads((PROJECT/"configs/distributions/2d_independent_rooms.json").read_text())
    config["rooms"]["placement"]={"x":[.3,.7],"y":.5,"coordinates":"normalized"}
    config["obstacles"]=[]
    metadata=generate(config,tmp_path/"normalized")
    assert all(15<=r["center"][0]<=35 and r["center"][1]==25 for r in metadata["room_parameters"])
    assert validate(tmp_path/"normalized")["passed"]


def test_unknown_parameter_names_do_not_disappear_silently():
    config={"dimension":2,"world_size":[50,50],"family":"scattered_rooms",
            "layout":{"parameters":{"scatter.raduis_fraction":[0,1]}}}
    with pytest.raises(ValueError,match="unknown layout parameters"):
        create_layout(config,np.random.default_rng(12),10,5,.3)


def test_radial_fraction_has_declared_world_scale():
    config={"dimension":2,"world_size":[50,50],"family":"scattered_rooms",
            "layout":{"fit_mode":"none","parameters":{"scatter.angle":0,
                "scatter.modulation":0,"scatter.radius_fraction":.2}}}
    layout=create_layout(config,np.random.default_rng(12),4,5,.3)
    assert np.allclose(layout["centers"],[[10,0]]*4)


def test_no_fit_rejects_out_of_domain_rooms_instead_of_moving_them(tmp_path):
    config=json.loads((PROJECT/"configs/distributions/2d_independent_rooms.json").read_text())
    config["rooms"]["placement"]={"x":1,"y":[20,30]}
    config["obstacles"]=[];config["max_attempts"]=1
    with pytest.raises(RuntimeError,match="leave the domain"):
        generate(config,tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()


def test_world_extent_resolution_and_elevation_accept_ranges(tmp_path):
    config=json.loads((PROJECT/"configs/distributions/3d_independent_rooms.json").read_text())
    config["world_size"]=[[50,52],[50,52],[30,32]]
    config["resolution"]=[[45,50],[45,50],[28,30]]
    config["floors"]=[2,4]
    config["rooms"]["placement"].pop("z")
    config["rooms"]["elevation"]=[10,20];config["obstacles"]=[]
    metadata=generate(config,tmp_path/"ranges")
    assert all(a<=x<=b for x,(a,b) in zip(metadata["world_size"],config["world_size"]))
    assert all(a<=x<=b for x,(a,b) in zip(metadata["occupancy"]["shape_xyz"],config["resolution"]))
    assert all(10<=r["center"][2]<=20 for r in metadata["room_parameters"])
    effective=json.loads((tmp_path/"ranges/resolved_configuration.json").read_text())
    layout=json.loads((tmp_path/"ranges/layout.json").read_text())
    assert 2<=effective["floors"]<=4
    assert layout["floor_schedule"]["count_sampled"]==effective["floors"]
    assert all(r["floor"] is None for r in metadata["room_parameters"])
    assert validate(tmp_path/"ranges")["passed"]
