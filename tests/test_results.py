from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from src.experiment import Experiment
from src.xp_config import ExperimentConfig


def config(**kwargs):
    return ExperimentConfig({'k_0':1.,'sigma_k':.4,'x_0':0.,'n':0},
                            {'L':2*np.pi,'omega_0':1.,'D':.4,'x_tls':.3,'coupling':'sqrt','initial_state':'e'},
                            {'T':1.,'dt':.1},n_max=2,CTRL_M_EXPLICIT=True,M=3,**kwargs)


@pytest.mark.parametrize('scheme',['full','full+totalcap','truncated'])
def test_resonant_RWA_vacuum_decay_matches_analytic_collective_rabi_frequency(scheme):
    c=config(truncation=scheme,RWA=True)
    experiment=Experiment(c); experiment.propagate_state()
    expected=np.cos(np.sqrt(2)*c.param_atom['D']/np.sqrt(c.param_atom['L'])*experiment.times)**2
    assert_allclose(experiment.compute_excited_probability(),expected,atol=2e-8)


@pytest.mark.parametrize('rwa',[True,False])
def test_solver_matches_matrix_exponential_and_final_only_storage(rwa):
    c=config(RWA=rwa)
    a,b=Experiment(c),Experiment(replace(c,store_state=False))
    a.propagate_state(); b.propagate_state()
    expected=expm(-1j*a.H.full()*a.times[-1])@a.state0.full()[:,0]
    assert_allclose(a.result.final_state.full()[:,0],expected,atol=2e-8)
    assert_allclose(b.result.final_state.full()[:,0],expected,atol=2e-8)
    assert_allclose(b.compute_excited_probability(-1),a.compute_excited_probability(-1))
    with pytest.raises(ValueError):
        b.compute_energy(.2)


def test_zero_mode_kept_but_decoupled_for_sqrt_profile():
    c=replace(config(),M=1)
    experiment=Experiment(c); experiment.propagate_state()
    assert_allclose(experiment.k_tab,[0])
    assert_allclose(experiment.compute_excited_probability(),1.,atol=2e-8)


def test_interaction_coordinate_matches_the_imposed_spatial_phase():
    c=config()
    c=replace(c,param_atom={**c.param_atom,'coupling':'flat','initial_state':'g'},
              param_photon={**c.param_photon,'n':1})
    experiment=Experiment(c); experiment.propagate_state()
    coordinate=-c.param_atom['x_tls']
    phi=experiment.one_photon_wavefunction(0,[coordinate])[0]
    vacuum_excited=2*experiment.basis.index[(0,0,0)]+1
    amplitude=(experiment.H.full()@experiment.state0.full()[:,0])[vacuum_excited]
    assert_allclose(amplitude,1j*c.param_atom['D']*phi,atol=1e-14)
