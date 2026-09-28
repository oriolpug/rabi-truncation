"""Small script entry point; the same function is usable from a notebook."""

import csv
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.xp_config import ExperimentConfig
from src.experiment import Experiment


def run_scattering(param_photon, param_atom, param_time_evol, cutoffs,
                   n_max=3,
                   truncation="full+totalcap", 
                   RWA=False,
                   store_state=True, 
                   store_results=False, 
                   progress=False):
    
    """Propagate one chosen truncation and return the Experiment object."""

    config = ExperimentConfig(param_photon=param_photon,
                              param_atom=param_atom,
                              param_time_evol=param_time_evol,
                              cutoffs=cutoffs, 
                              n_max=n_max,
                              truncation=truncation, 
                              RWA=RWA,
                              store_state=store_state)
    
    experiment = Experiment(config)
    experiment.propagate_state(progress=progress)
    observables = experiment.compute_observables()

    if store_results:
        out = project_root / "results" / "csv_files" / "scattering.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(observables)
            writer.writerows(zip(*observables.values()))
    return experiment


#if __name__ == "__main__":
#
#    param_photon = {"k_0": 1.0, "sigma_k": 0.10, "x_0": -12.0}
#
#    param_atom = {"omega_0": 1.0, 
#                  "D": 0.45, 
#                  "x_tls": 0.0,
#                  "L": 20 * np.pi, 
#                  "coupling": "sqrt"}
#    param_time_evol = {"T": 28.0, "dt": 0.1}
#    cutoffs = {"ir_cutoff": 0.8, "uv_cutoff": 1.2}
#    experiment = run_scattering(param_photon, param_atom, param_time_evol, cutoffs)
#    for name, values in experiment.observables.items():
#        print(name, values[-1])
