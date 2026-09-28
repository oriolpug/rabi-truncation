"""Energy heatmaps and animation in mode and photon-number coordinates."""

import numpy as np

from ._common import config_from_args, finish, parser, run


def run_energy_profile(config, progress=False):
    if not config.store_state:
        raise ValueError('Energy time profiles require store_state=True')
    experiment = run(config, progress)
    k, modes, _, atom = experiment.compute_energy_profile_modes()
    numbers, excitation = experiment.compute_energy_profile_excitations()
    experiment.energy_profiles = {'time': experiment.times, 'k': k, 'E_modes': modes,
                                  'E_atom': atom, 'photon_numbers': numbers,
                                  'E_excitation': excitation, 'E_total': experiment.compute_energy()}
    return experiment


def plot_energy_profile(experiment):
    import matplotlib.pyplot as plt
    data = experiment.energy_profiles
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, x, values, xlabel in ((axes[0], data['k'], data['E_modes'], 'k'),
                                 (axes[1], data['photon_numbers'], data['E_excitation'], 'photon number')):
        image = ax.pcolormesh(x, data['time'], values, shading='nearest')
        fig.colorbar(image, ax=ax)
        ax.set(xlabel=xlabel, ylabel='t')
    axes[2].plot(data['time'], data['E_total'], label='total')
    axes[2].plot(data['time'], data['E_atom'], label='TLS')
    axes[2].set(xlabel='t', ylabel='energy')
    axes[2].legend()
    fig.tight_layout()
    return fig


def animate_energy_profile(experiment, path=None, fps=15, stride=1):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    data = experiment.energy_profiles
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    field, = axes[0].plot(data['k'], data['E_modes'][0], 'o-', label='field')
    atom, = axes[0].plot([0], [data['E_atom'][0]], 's', label='TLS energy')
    bars = axes[1].bar(data['photon_numbers'], data['E_excitation'][0])
    for ax, values in ((axes[0], np.append(data['E_modes'], data['E_atom'])),
                       (axes[1], data['E_excitation'])):
        lower, upper = values.min(), values.max()
        pad = max((upper - lower) * 0.1, 1e-6)
        ax.set_ylim(lower - pad, upper + pad)
        ax.set_ylabel('energy')
    axes[0].set_xlabel('k')
    axes[0].legend()
    axes[1].set_xlabel('photon number')
    def update(i):
        field.set_ydata(data['E_modes'][i])
        atom.set_ydata([data['E_atom'][i]])
        for bar, value in zip(bars, data['E_excitation'][i]):
            bar.set_height(value)
        fig.suptitle(f't={data["time"][i]:.3f}')
        return field, atom, *bars
    animation = FuncAnimation(fig, update, frames=range(0, len(data['time']), stride),
                              interval=1000 / fps, blit=False)
    if path is not None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
    return animation


def main(argv=None):
    p = parser(__doc__)
    p.add_argument('--gif', type=str)
    args = p.parse_args(argv)
    config = config_from_args(args)
    from src.experiment import estimate_config
    estimate_config(config)
    experiment = run_energy_profile(config, args.progress)
    if args.gif:
        animate_energy_profile(experiment, args.gif)
    finish(plot_energy_profile(experiment), args, config, **experiment.energy_profiles)


if __name__ == '__main__':
    main()
