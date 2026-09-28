from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.experiment import Experiment
from src.xp_config import ExperimentConfig


@pytest.mark.parametrize('scheme',['full','full+totalcap','truncated'])
@pytest.mark.parametrize('profile',['sqrt','flat'])
@pytest.mark.parametrize('rwa',[False,True])
def test_both_energy_partitions_sum_to_the_exact_expectation(scheme,profile,rwa):
    config=ExperimentConfig({'k_0':0.,'sigma_k':.7,'x_0':-.3,'state':'coherent','alpha':.4+.2j},
                            {'L':2*np.pi,'omega_0':1.,'D':.4,'x_tls':.2,'coupling':profile,'initial_state':'+'},
                            {'T':.3,'dt':.1},n_max=2,truncation=scheme,RWA=rwa,
                            CTRL_M_EXPLICIT=True,M=3)
    experiment=Experiment(config); experiment.propagate_state()
    k,modes,_,atom=experiment.compute_energy_profile_modes()
    numbers,excitation=experiment.compute_energy_profile_excitations()
    total=experiment.compute_energy()
    assert_allclose(modes.sum(axis=1)+atom,total,atol=1e-13)
    assert_allclose(excitation.sum(axis=1),total,atol=1e-13)
    assert_allclose(total,total[0],atol=2e-8)
    obs=experiment.compute_observables()
    assert_allclose(obs['norm'],1.,atol=2e-8)
    assert_allclose(sum(v for key,v in obs.items() if key.endswith('_photon')),obs['norm'],atol=1e-13)
    assert k[1]==0
