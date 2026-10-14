import numpy as np
import pytest
from wgv.parameters import draw, draw_point

@pytest.mark.parametrize("spec",[
    4,{"distribution":"fixed","value":4},
    {"distribution":"uniform_int","min":2,"max":9},
    {"distribution":"uniform","min":.2,"max":.8},
    {"distribution":"normal","mean":4,"std":.2},
    {"distribution":"truncated_normal","mean":4,"std":1,"min":2,"max":5},
    {"distribution":"lognormal","mean":0,"sigma":.3},
    {"distribution":"exponential","scale":1},
    {"distribution":"poisson","lambda":8},
    [2, 9], {"min":2,"max":9},
    {"distribution":"beta","alpha":2,"beta":3,"min":2,"max":9},
    {"distribution":"categorical","values":[3,7],"weights":[1,2]},
    {"distribution":"mixture","components":[2,{"distribution":"uniform","min":3,"max":5}],"weights":[1,3]},
])
def test_distribution_reproducibility(spec):
    a,b=np.random.default_rng(4),np.random.default_rng(4)
    assert [draw(spec,a) for _ in range(30)]==[draw(spec,b) for _ in range(30)]

@pytest.mark.parametrize("spec",[
    2.5,
    {"distribution":"uniform","min":2,"max":5}])
def test_integer_parameters_do_not_silently_round(spec):
    with pytest.raises(ValueError):
        draw(spec,np.random.default_rng(2),integer=True)

@pytest.mark.parametrize("spec",[
    {"distribution":"normal","mean":0,"std":-1},
    {"distribution":"uniform","min":5,"max":1},
    {"distribution":"poisson","lambda":-1},
    {"distribution":"categorical","values":[1,2],"weights":[-1,2]},
    float("nan"), True])
def test_invalid_parameters(spec):
    with pytest.raises(ValueError):
        draw(spec,np.random.default_rng(2))


def test_uniform_range_respects_integer_contract():
    rng=np.random.default_rng(17)
    values=[draw([2,4],rng,integer=True) for _ in range(100)]
    assert set(values)=={2,3,4}
    with pytest.raises(ValueError):
        draw([2.5,4],rng,integer=True)


@pytest.mark.parametrize("spec",[
    {"x":[1,3],"y":{"distribution":"normal","mean":4,"std":.2}},
    {"type":"uniform","min":[1,2],"max":[3,4]},
    {"type":"gaussian","mean":[1,2],"covariance":[[2,.5],[.5,1]]},
    {"type":"mixture","components":[{"type":"fixed","value":[1,2]},
        {"type":"gaussian","mean":[4,5],"std":.1}],"weights":[1,3]},
])
def test_point_distributions_reproduce_whole_locations(spec):
    a,b=np.random.default_rng(8),np.random.default_rng(8)
    assert np.array_equal([draw_point(spec,a,2) for _ in range(40)],
                          [draw_point(spec,b,2) for _ in range(40)])


@pytest.mark.parametrize("spec",[
    {"type":"gaussian","mean":[1,2],"covariance":[[1,2],[2,1]]},
    {"type":"gaussian","mean":[1,2],"covariance":[[1,0],[1,1]]},
    {"type":"gaussian","mean":[1,2],"std":-1},
    {"type":"uniform","min":[1,2],"max":[3]},
    {"type":"independent","x":[1,3]},
    {"type":"mixture","components":[],"weights":[]},
])
def test_invalid_point_distributions_are_rejected(spec):
    with pytest.raises(ValueError):
        draw_point(spec,np.random.default_rng(8),2)


def test_correlated_placement_retains_its_joint_law():
    # Singular covariance forces y=x, which independent axis draws cannot reproduce.
    points=np.asarray([draw_point({"type":"gaussian","mean":[0,0],
        "covariance":[[1,1],[1,1]]},np.random.default_rng(i),2) for i in range(20)])
    assert np.allclose(points[:,0],points[:,1],atol=1e-7)


@pytest.mark.parametrize("name", ["normal", "gaussian", "truncated_normal"])
def test_gaussian_count_matches_rounded_continuous_draws(name):
    spec={"distribution":name,"mean":8.5,"std":1}
    if name=="truncated_normal":
        spec.update(min=7,max=10)
    a,b=np.random.default_rng(12),np.random.default_rng(12)
    floats=[draw(spec,a) for _ in range(40)]
    counts=[draw(spec,b,integer=True) for _ in range(40)]
    assert counts==[int(np.floor(x+.5)) for x in floats]
    assert all(isinstance(n,int) for n in counts)


@pytest.mark.parametrize("value,expected", [(8.4,8),(8.5,9),(8.6,9),(-1.5,-2)])
def test_gaussian_rounding_thresholds_and_half_ties(value,expected):
    spec={"distribution":"normal","mean":value,"std":0}
    assert draw(spec,np.random.default_rng(1),integer=True)==expected
    assert draw(spec,np.random.default_rng(1))==value


def test_bounded_gaussian_rejects_instead_of_clamping_and_reproduces():
    class Proposals:
        def __init__(self):
            self.values=iter([6.4,10.8,8.6])
        def normal(self,mean,std):
            return next(self.values)
    spec={"distribution":"normal","mean":8.5,"std":1,"min":7,"max":10}
    assert draw(spec,Proposals(),integer=True)==9
    a,b=np.random.default_rng(8),np.random.default_rng(8)
    first=[draw(spec,a,integer=True) for _ in range(100)]
    assert first==[draw(spec,b,integer=True) for _ in range(100)]
    assert set(first)=={7,8,9,10}


def test_gaussian_rounding_keeps_global_constraints_and_mixture_support():
    law={"distribution":"normal","mean":2.6,"std":0}
    assert draw({"distribution":"mixture","components":[law]},np.random.default_rng(2),integer=True)==3
    with pytest.raises(ValueError,match="below minimum"):
        draw(law,np.random.default_rng(2),integer=True,minimum=4)


@pytest.mark.parametrize("spec", [
    {"distribution":"normal","mean":8,"std":1,"min":10,"max":7},
    {"distribution":"normal","mean":8,"std":1,"min":7.2},
    {"distribution":"normal","mean":float("nan"),"std":1},
    {"distribution":"normal","mean":8,"std":float("inf")},
    {"distribution":"normal","mean":8,"std":1,"max":float("inf")},
])
def test_invalid_gaussian_count_settings(spec):
    with pytest.raises(ValueError):
        draw(spec,np.random.default_rng(1),integer=True)


def test_impossible_gaussian_bounds_raise_an_error():
    with pytest.raises(ValueError,match="failed to produce"):
        draw({"distribution":"normal","mean":2,"std":0,"min":7,"max":10},
             np.random.default_rng(1),integer=True)
