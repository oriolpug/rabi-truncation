"""Shared experiment execution, CLI inputs and exports."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.experiment import Experiment, estimate_config
from src.xp_config import ExperimentConfig


def run(config, progress=False):
    experiment = Experiment(config)
    experiment.propagate_state(progress=progress)
    experiment.compute_observables()
    return experiment


def save_arrays(path, config, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(asdict(config), default=lambda value: str(value))
    np.savez(path, configuration_json=metadata, **arrays)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--D', type=float, default=0.2)
    p.add_argument('--L', type=float, default=2 * np.pi)
    p.add_argument('--omega-0', type=float, default=1.0)
    p.add_argument('--x-tls', type=float, default=0.0)
    p.add_argument('--coupling', choices=('sqrt', 'flat'), default='sqrt')
    p.add_argument('--ctrl-m-explicit', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--M', type=int, default=5)
    p.add_argument('--ir-cutoff', type=float, default=0.0)
    p.add_argument('--uv-cutoff', type=float, default=2.0)
    p.add_argument('--k-0', type=float, default=1.0)
    p.add_argument('--sigma-k', type=float, default=0.4)
    p.add_argument('--x-0', type=float, default=-2.0)
    p.add_argument('--photon-state', choices=('number', 'coherent'), default='number')
    p.add_argument('--n', type=int, default=1)
    p.add_argument('--alpha', type=complex, default=1.0)
    p.add_argument('--atom-state', choices=('g', 'e', '+', '-'), default='g')
    p.add_argument('--n-max', type=int, default=2)
    p.add_argument('--truncation', choices=('full', 'full+totalcap', 'truncated'), default='full+totalcap')
    p.add_argument('--T', type=float, default=4.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--RWA', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--progress', action='store_true')
    return p


def config_from_args(args):
    return ExperimentConfig(
        param_photon={'k_0': args.k_0, 'sigma_k': args.sigma_k, 'x_0': args.x_0,
                      'state': args.photon_state, 'n': args.n, 'alpha': args.alpha},
        param_atom={'omega_0': args.omega_0, 'D': args.D, 'L': args.L,
                    'x_tls': args.x_tls, 'coupling': args.coupling, 'initial_state': args.atom_state},
        param_time_evol={'T': args.T, 'dt': args.dt},
        cutoffs={'ir_cutoff': args.ir_cutoff, 'uv_cutoff': args.uv_cutoff},
        n_max=args.n_max, truncation=args.truncation, RWA=args.RWA,
        CTRL_M_EXPLICIT=args.ctrl_m_explicit, M=args.M)


def finish(fig, args, config, **arrays):
    import matplotlib.pyplot as plt
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        save_arrays(args.out.with_suffix('.npz'), config, **arrays)
        plt.close(fig)
        print(f'Saved {args.out}')
    else:
        plt.show()
