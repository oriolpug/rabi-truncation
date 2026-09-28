import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.grid import momentum_modes, resolve_grid, grid_summary
from src.experiment import estimate_config, output_times
from src.xp_config import ExperimentConfig


def config(**kwargs):
    return ExperimentConfig({'k_0': 1., 'sigma_k': .3, 'x_0': 0.},
                            {'L': 2*np.pi, 'omega_0': 1., 'D': .2, 'x_tls': 0., 'coupling': 'sqrt'},
                            {'T': .3, 'dt': .1}, {'ir_cutoff': 0., 'uv_cutoff': 2.}, **kwargs)


@pytest.mark.parametrize('M', [1, 3, 5, 9])
def test_explicit_grid_is_symmetric_and_keeps_exact_M(M):
    grid = momentum_modes({'L': 2*np.pi}, CTRL_M_EXPLICIT=True, M=M)
    assert len(grid) == M and 0 in grid
    assert_allclose(grid, -grid[::-1])
    assert_allclose(grid, np.arange(-(M//2), M//2+1))


@pytest.mark.parametrize('M', [0, 2, 4, -3, 3.0, True])
def test_invalid_M_is_rejected(M):
    with pytest.raises(ValueError):
        momentum_modes({'L': 2*np.pi}, CTRL_M_EXPLICIT=True, M=M)


@pytest.mark.parametrize('ir,uv,expected', [(0, 2, [-2,-1,0,1,2]), (.2, 2.8, [-2,-1,1,2]),
                                          (1, 1, [-1,1]), (0,0,[0]), (1e-15,1,[-1,1])])
def test_cutoff_control_and_zero_policy(ir, uv, expected):
    assert_allclose(momentum_modes({'L': 2*np.pi}, {'ir_cutoff': ir, 'uv_cutoff': uv}), expected)


@pytest.mark.parametrize('ir,uv', [(-1,2),(2,1),(np.nan,2),(.1,.2)])
def test_invalid_or_empty_bands(ir, uv):
    with pytest.raises(ValueError):
        momentum_modes({'L': 2*np.pi}, {'ir_cutoff': ir, 'uv_cutoff': uv})


def test_modes_and_cutoffs_produce_identical_grid_and_resources():
    a, b = config(CTRL_M_EXPLICIT=True, M=5), config()
    assert_allclose(resolve_grid(a)[1], resolve_grid(b)[1])
    for item in (a, b):
        estimate = estimate_config(item, print_report=False)
        assert estimate['n_modes'] == 5
        assert estimate['dimension'] == 2*56
        assert estimate['grid']['selected']['equivalent_M'] == 5
        assert estimate['grid']['selected']['zero_present']
        assert estimate['n_times'] == 4


def test_selection_reports_when_two_parameters_cannot_encode_the_grid():
    item = config(mode_selection=True, photon_window=0, atom_window=0)
    base, selected = resolve_grid(item)
    assert_allclose(selected, [-1,1])
    info = grid_summary(base, selected, 2*np.pi)
    assert info['selected']['equivalent_M'] is None
    assert info['selected']['radial_cutoffs_exact']
    assert not info['selected']['zero_present']
    info = grid_summary(base, np.array([-1.,1.,2.]), 2*np.pi)
    assert not info['selected']['radial_cutoffs_exact']


@pytest.mark.parametrize('T,dt,expected', [(.3,.1,[0,.1,.2,.3]), (.35,.1,[0,.1,.2,.3,.35]),(.05,.1,[0,.05]),(1e-15,1,[0,1e-15])])
def test_output_times_keep_zero_and_exact_final_time(T, dt, expected):
    assert_allclose(output_times({'T': T, 'dt': dt}), expected)
