"""
Initializes full (untruncated) states, and evolves them according to the QR hamiltonian.
Computes the fidelity of the output states, as a function of the excitation cap N
Plot: <psi(t=T,N_exc=n+1)|psi(t=T,N_exc=n)>(n)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt

from utilities import Config, CoherentState
from simulation import Simulation
from states import StateTotalCap
from fidelities import fidelity_statevector


def compute_and_evolve(N_max: int = 10, g: float = 0.01, alpha: complex = 1 + 0j, RWA: bool = False):
    """
    Run full-basis simulations for N = 1 .. N_max and return consecutive fidelities.

    Returns
    -------
    N_values  : list[int]   — excitation caps 1 .. N_max-1 (the 'lower' N of each pair)
    fidelities: list[float] — F(N) = |<psi(T, N+1)|psi(T, N)>|^2
    """
    final_states = []

    for N in range(1, N_max + 1):
        try:
            config = Config(
                modes=64,
                length=20,
                excitation_cap=N,
                g=g,
                RWA=RWA,
                state=CoherentState(alpha),
                truncation="full+totalcap",
            )

            sim = Simulation(config)
            sim.time_evolve()

            final_vec = sim.result.states[-1].full()[:, 0]
            final_state = StateTotalCap.from_vector(config, final_vec)
            final_states.append(final_state)

            print(f"  N={N:2d}  dim={sim.H.shape[0]}  done")
        except Exception as e:
            print(f"  N={N:2d}  ERROR: {e} — stopping loop")
            break

    # F(N) = |<psi(N+1)|psi(N)>|^2 — pass larger-N state first so that
    # common_basis returns type(state2) = the smaller basis as the reference
    n_ran = len(final_states)
    N_values = list(range(1, n_ran))
    fidelities = [
        fidelity_statevector(final_states[i + 1], final_states[i])
        for i in range(n_ran - 1)
    ]

    return N_values, fidelities


def main(**kwargs):
    N_max = int(kwargs.get('N', 10))

    alpha = kwargs.get('alpha', 1.0)
    if not isinstance(alpha, list):
        alpha = [complex(alpha)]
    else:
        alpha = [complex(a) for a in alpha]

    g = kwargs.get('g', 0.01)
    if not isinstance(g, list):
        g = [float(g)]
    else:
        g = [float(gi) for gi in g]

    rwa_raw = kwargs.get('RWA', 'False')
    RWA = rwa_raw if isinstance(rwa_raw, bool) else rwa_raw.lower() in ('true', '1', 'yes')

    fig, ax = plt.subplots()

    for gi in g:
        for ai in alpha:
            print(f"Running g={gi}, alpha={ai}, RWA={RWA}")
            N_values, fidelities = compute_and_evolve(N_max=N_max, g=gi, alpha=ai, RWA=RWA)
            ax.plot(N_values, fidelities, marker='o', label=f'g={gi}, α={ai}')

    ax.set_xlabel('N')
    ax.set_ylabel(r'$|\langle\psi(T,\,N{+}1)|\psi(T,\,N)\rangle|^2$')
    ax.set_title('Cumulative fidelity vs excitation cap (full basis)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    save = kwargs.get('save', None)
    if save:
        fig.savefig(save, dpi=150)
        print(f"Plot saved to {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
