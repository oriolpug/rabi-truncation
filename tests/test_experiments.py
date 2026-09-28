from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.xp_config import ExperimentConfig
from experiment.atom_evolution import run_atom_evolution
from experiment.energy_profile import run_energy_profile
from experiment.full_cumulative_fidelity import run_cap_convergence
from experiment.sweep_D_fidelity import run_coupling_sweep
from experiment.compare_mode_selection import run_mode_selection
from experiment.scattering import run_scattering


def config():
    return ExperimentConfig({'k_0':1.,'sigma_k':.5,'x_0':-.3,'n':1},
                            {'L':2*np.pi,'omega_0':1.,'D':.2,'x_tls':.2,'coupling':'sqrt'},
                            {'T':.2,'dt':.1},n_max=2,CTRL_M_EXPLICIT=True,M=3)


def test_all_experiment_functions_and_no_configuration_mutation():
    c=config()
    runs=run_atom_evolution(c)
    assert set(runs)=={'full+totalcap','truncated'}
    energy=run_energy_profile(c)
    assert_allclose(energy.energy_profiles['E_excitation'].sum(axis=1),energy.energy_profiles['E_total'])
    convergence=run_cap_convergence(c,[1,2])
    assert_allclose(convergence['F_state_initial'],1.)
    sweep=run_coupling_sweep(c,[0.,.2])
    assert sweep['F_state'].shape==(1,2)
    assert_allclose(sweep['F_state_initial'],1.)
    modes=run_mode_selection(c,[.2],photon_windows=[0.,1.],atom_windows=[0.,1.])
    assert modes['F_atom_heatmap'].shape==(2,2,2)
    assert_allclose(modes['F_state'][1,0],1.)
    scattering=run_scattering(c.param_photon,c.param_atom,c.param_time_evol,n_max=2,
                               CTRL_M_EXPLICIT=True,M=3)
    assert scattering.observables['p_zero_1g'][0]>0
    assert c.param_atom['D']==.2 and c.M==3 and c.n_max==2


@pytest.mark.parametrize('name',['atom_evolution','energy_profile','full_cumulative_fidelity',
                                'sweep_g_fidelity','compare_mode_selection'])
def test_historical_script_paths_can_execute_with_D(name,tmp_path):
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,'-B',str(root/'experiments_Uri'/f'{name}.py'),
             '--D','.2','--ctrl-m-explicit','--M','3','--n-max','2','--T','.2','--dt','.1',
             '--out',str(tmp_path/f'{name}.png')]
    if name=='full_cumulative_fidelity': command+=['--caps','1,2']
    if name in ('sweep_g_fidelity','compare_mode_selection'): command+=['--D-values','0.1,0.2']
    if name=='compare_mode_selection': command+=['--photon-windows','0,1','--atom-windows','0,1']
    import os
    env={**os.environ,'MPLBACKEND':'Agg','MPLCONFIGDIR':str(tmp_path/'mpl')}
    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/f'{name}.png').is_file()
    archive=np.load(tmp_path/f'{name}.npz')
    assert 'configuration_json' in archive


def test_notebook_structure_and_user_conventions():
    root=Path(__file__).resolve().parents[1]
    notebooks=list((root/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==6
    for path in notebooks:
        data=json.loads(path.read_text())
        source=''.join(''.join(cell['source']) for cell in data['cells'])
        assert "'coupling': 'sqrt'" in source
        assert 'CTRL_M_EXPLICIT' in source and 'estimate_config' in source
        for cell in data['cells']:
            if cell['cell_type']=='code': compile(''.join(cell['source']),str(path),'exec')
