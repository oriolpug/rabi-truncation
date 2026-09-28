"""Final-state convergence between neighbouring total photon caps."""

from dataclasses import replace

import numpy as np

from src.fidelities import compare_states
from ._common import config_from_args, finish, parser, run


def run_cap_convergence(config, caps=(1, 2, 3, 4), progress=False):
    caps = list(caps)
    if len(caps) < 2 or any(b <= a for a, b in zip(caps, caps[1:])):
        raise ValueError('At least two strictly increasing caps are required')
    runs = [run(replace(config, n_max=cap, truncation='full+totalcap', store_state=False), progress)
            for cap in caps]
    final, initial = [], []
    for a, b in zip(runs, runs[1:]):
        final.append(compare_states(a.result.final_state, a.basis, a.k_tab,
                                    b.result.final_state, b.basis, b.k_tab))
        initial.append(compare_states(a.state0, a.basis, a.k_tab, b.state0, b.basis, b.k_tab))
    return {'caps': np.array(caps), 'runs': runs,
            **{name: np.array([row[name] for row in final]) for name in ('F_state', 'F_atom')},
            **{f'{name}_initial': np.array([row[name] for row in initial])
               for name in ('F_state', 'F_atom')}}


def plot_cap_convergence(results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ('F_state', 'F_atom')):
        ax.plot(results['caps'][:-1], results[name], 'o-', label='final')
        ax.plot(results['caps'][:-1], results[f'{name}_initial'], 'o--', label='initial')
        ax.set(xlabel='lower cap of compared pair', ylabel=name, ylim=(0, 1.05))
        ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None):
    p = parser(__doc__)
    p.set_defaults(photon_state='coherent')
    p.add_argument('--caps', default='1,2,3,4')
    args = p.parse_args(argv)
    config = config_from_args(args)
    caps = [int(value) for value in args.caps.split(',')]
    from src.experiment import estimate_config
    estimate_config(replace(config, n_max=max(caps), truncation='full+totalcap', store_state=False))
    results = run_cap_convergence(config, caps, args.progress)
    finish(plot_cap_convergence(results), args, config,
           **{key: value for key, value in results.items() if key != 'runs'})


if __name__ == '__main__':
    main()
