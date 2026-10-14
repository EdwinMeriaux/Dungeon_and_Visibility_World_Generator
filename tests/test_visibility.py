import itertools
import json
import subprocess
import numpy as np
import pytest
from shapely.geometry import Polygon,LineString
from wgv.visibility import BINARY,read_csr
from wgv import solid


def execute(tmp_path,payload):
    payload={"schema_version":1,"epsilon":1e-8,**payload}
    path=tmp_path/"input.json";prefix=tmp_path/"out"
    path.write_text(json.dumps(payload))
    result=subprocess.run([str(BINARY),"--input",str(path),"--output",str(prefix),"--threads","2"],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return read_csr(prefix.with_suffix(".csr")).toarray()


@pytest.mark.parametrize("polygon",[
    Polygon([(0,0),(8,0),(8,8),(0,8)]),
    Polygon([(0,0),(8,0),(8,8),(0,8)],holes=[[(3,3),(5,3),(5,5),(3,5)]]),
    Polygon([(0,0),(8,1),(7,8),(0,6)],holes=[[(3,2),(5,3),(3,5)]]),
])
def test_polygon_rust_against_independent_geos(tmp_path,polygon):
    rng=np.random.default_rng(81)
    points=[]
    while len(points)<25:
        candidate=rng.uniform(.1,7.5,2)
        if polygon.contains(__import__("shapely").Point(candidate)):
            points.append(candidate)
    rings=[list(r.coords) for r in [polygon.exterior,*polygon.interiors]]
    matrix=execute(tmp_path,{"dimension":2,"mode":"continuous","polygons":rings,
                            "observers":np.asarray(points).tolist(),"targets":np.asarray(points).tolist()})
    expected=np.zeros(matrix.shape,bool)
    for i,a in enumerate(points):
        for j,b in enumerate(points):
            if i==j:
                expected[i,j]=True
            else:
                line=LineString([a,b])
                expected[i,j]=polygon.covers(line) and line.intersection(polygon.boundary).is_empty
    assert np.array_equal(matrix,expected)
    assert np.array_equal(matrix,matrix.T)


def test_2d_wall_grazing_doorway_and_boundary_target(tmp_path):
    world=Polygon([(0,0),(10,0),(10,10),(0,10)],holes=[[(4,3),(6,3),(6,7),(4,7)]])
    matrix=execute(tmp_path,{"dimension":2,"mode":"continuous",
        "polygons":[list(r.coords) for r in [world.exterior,*world.interiors]],
        "observers":[[2,3],[2,5]],
        "targets":[[8,3],[8,2],[4,3]],
        "sensor":{"boundary_targets":True}})
    assert np.array_equal(matrix,[[False,True,True],[False,False,True]])


def ray_touches_box(p,q,low,high):
    lo,hi=0.,1.
    for a,b,l,h in zip(p,q,low,high):
        direction=b-a
        if abs(direction)<1e-13:
            if not l<=a<=h:
                return False
        else:
            t0,t1=(l-a)/direction,(h-a)/direction
            lo=max(lo,min(t0,t1));hi=min(hi,max(t0,t1))
    return lo<=hi and hi>1e-12 and lo<1-1e-12


@pytest.mark.parametrize("dimension",[2,3])
def test_supercover_against_exhaustive_blocked_cell_boxes(tmp_path,dimension):
    rng=np.random.default_rng(10)
    shape=(4,)*dimension
    free=rng.random(shape)>.23
    indices=np.argwhere(free)
    points=indices+.5
    payload={"dimension":dimension,"mode":"discrete","grid":{
        "shape":list(shape),"spacing":[1.]*dimension,"origin":[0.]*dimension,
        "free":free.ravel().tolist()},"observers":points.tolist(),"targets":points.tolist()}
    matrix=execute(tmp_path,payload)
    blocked=np.argwhere(~free)
    expected=np.asarray([[not any(ray_touches_box(p,q,cell,cell+1) for cell in blocked)
                          for q in points] for p in points])
    assert np.array_equal(matrix,expected)
    assert np.array_equal(matrix,matrix.T)


@pytest.mark.parametrize("geometry",["polyhedral","curvilinear","mixed"])
def test_3d_exact_solid_and_symmetry(tmp_path,geometry):
    outer=solid.box([0,0,0],[10,10,10])
    obstacle=solid.box([0,0,0],[2,2,2])
    if geometry in ("curvilinear","mixed"):
        obstacle=solid.ellipsoid([0,0,0],[1,1.5,1])
    if geometry=="curvilinear":
        outer=solid.ellipsoid([0,0,0],[5,5,5])
    world={"positive":[outer],"negative":[obstacle]}
    points=[[-3,0,0],[3,0,0],[-3,2,0],[3,2,0],[0,0,3],[0,0,-3]]
    matrix=execute(tmp_path,{"dimension":3,"mode":"continuous","solid":world,
                            "observers":points,"targets":points})
    expected=np.asarray([[solid.visible(world,p,q) for q in points] for p in points])
    assert np.array_equal(matrix,expected)
    assert np.array_equal(matrix,matrix.T)
    assert not matrix[0,1] and matrix[2,3] and not matrix[4,5]


def test_vertical_shaft_and_disconnected_volume(tmp_path):
    lower=solid.box([0,0,0],[6,6,3]);upper=solid.box([0,0,8],[6,6,3])
    shaft=solid.box([0,0,4],[2,2,10])
    p,q=[0,0,0],[0,0,8]
    world={"positive":[lower,upper,shaft],"negative":[]}
    assert execute(tmp_path,{"dimension":3,"mode":"continuous","solid":world,
                            "observers":[p],"targets":[q]})[0,0]
    world["positive"].pop()
    assert not execute(tmp_path,{"dimension":3,"mode":"continuous","solid":world,
                                "observers":[p],"targets":[q]})[0,0]


def test_range_and_field_of_view(tmp_path):
    matrix=execute(tmp_path,{"dimension":2,"mode":"continuous",
        "polygons":[[[0,0],[10,0],[10,10],[0,10]]],"observers":[[5,5]],
        "targets":[[6,5],[4,5],[8,5],[5,6]],
        "sensor":{"range":2,"axis":[1,0],"half_angle_degrees":45}})
    assert np.array_equal(matrix,[[True,False,False,False]])


def test_invalid_coordinates_produce_readable_error(tmp_path):
    path=tmp_path/"bad.json"
    path.write_text(json.dumps({"schema_version":1,"dimension":2,"mode":"continuous",
        "polygons":[[[0,0],[2,0],[2,2],[0,2]]],"observers":[[4,4]],"targets":[[1,1]]}))
    result=subprocess.run([str(BINARY),"--input",str(path),"--output",str(tmp_path/"bad")],
                          capture_output=True,text=True)
    assert result.returncode==2 and "observer 0 is not in free space" in result.stderr
