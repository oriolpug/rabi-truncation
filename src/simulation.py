"""
Orchestrates the simulation flow:
1. pass config parameters
2. build hamiltonian
3. initialize state
4. time-evolve
5. compute expectation values (optional)
6. store result state for later fidelity computation with other simulations
"""
from states import *
from utilities import *
from hamiltonians import *
from energy_profile import EnergyProfile
import numpy as np

class Simulation:

    def __init__(self, config):
        self.config = config

        # Initialize hamiltonian
        self.H = hamiltonian(config)

        # Initialize state
        self.state0 = state(config)

    def time_evolve(self, method: str ='bdf', observables = None):
        # Run time evolution with a given discretization scheme
        self.times = np.linspace(0, self.config.t, int(self.config.t/self.config.dt))

        self.result = qutip.sesolve(qutip.Qobj(self.H), qutip.Qobj(self.state0), self.times, e_ops=observables,
                               options={"store_states": True, "normalize_output": True, "progress_bar": "tqdm",
                                        "method": method})

        # self.states = [self.result.states[i].full()[:, 0] for i in range(len(self.times))]

    def get_expectation_value(self, t: float = None, index: int = 0):
        # t == None -> get full O(t)
        if t is None:
            return self.result.expect[index]
        else:
            return self.result.expect[index][t]

    def compute_atom_density_matrix(self, t: float = None):
        cls = {"full": StateFull, "truncated": StateTruncated, "truncated+atom": StateAtom, "full+totalcap": StateTotalCap}[self.config.truncation]
        if t is None:  # get full time series
            states = [cls.from_vector(self.config, state.full()[:,0]) for state in self.result.states]
            return [state.atom_density_matrix() for state in states]
        elif t == -1:  # get last value
            state = cls.from_vector(self.config, self.result.states[-1].full()[:,0])
            return state.atom_density_matrix()
        else:  # get value at index closest to t
            idx = int(t / self.config.dt)
            state = cls.from_vector(self.config, self.result.states[idx].full()[:, 0])
            return state.atom_density_matrix()

    def compute_excited_probability(self, t: float = None):
        dms = self.compute_atom_density_matrix(t)
        if isinstance(dms, list):
            prob = [dm[1,1] for dm in dms]
        else:
            prob = dms[1,1]
        return prob

    def compute_entropy(self, t: float = None):
        dms = self.compute_atom_density_matrix(t)
        if isinstance(dms, list):
            S = [entropy(dm) for dm in dms]
        else:
            S = entropy(dms)
        return S

    def compute_energy(self, t: float = None):
        if t is None:  # get full time series
            return [qutip.expect(self.H, state) for state in self.result.states]
        elif t == -1:  # get last value
            return qutip.expect(self.H, self.result.states[-1])
        else:  # get value at index closest to t
            idx = int(t / self.config.dt)
            return qutip.expect(self.H, self.result.states[idx])

    def _energy_profile(self) -> EnergyProfile:
        if not hasattr(self, "_energy_profile_obj"):
            self._energy_profile_obj = EnergyProfile(self.config, self.H)
        return self._energy_profile_obj

    def _states_for(self, t: float):
        """Resolve the `t` convention to a list of state vectors (see compute_energy)."""
        if t is None:
            return [state.full()[:, 0] for state in self.result.states]
        elif t == -1:
            return [self.result.states[-1].full()[:, 0]]
        else:
            idx = int(t / self.config.dt)
            return [self.result.states[idx].full()[:, 0]]

    def compute_energy_profile_modes(self, t: float = None):
        """Energy resolved by field mode.

        Returns ``(mode_axis, E_modes, atom_axis_x, E_atom)`` where ``mode_axis`` is the
        wave-vector grid of the field modes and the bare-atom energy sits at k=0. For a
        full time series (``t is None``) ``E_modes`` has shape ``(n_times, n_modes)`` and
        ``E_atom`` has shape ``(n_times,)``; otherwise both are for the single requested time.
        """
        ep = self._energy_profile()
        results = [ep.energy_modes_vec(psi) for psi in self._states_for(t)]
        E_modes = np.array([r[0] for r in results])
        E_atom = np.array([r[1] for r in results])
        if t is not None:
            E_modes, E_atom = E_modes[0], float(E_atom[0])
        return ep.mode_axis, E_modes, 0.0, E_atom

    def compute_energy_profile_excitations(self, t: float = None):
        """Energy resolved by photon excitation number nu.

        Returns ``(excitation_axis, E_nu)``. For a full time series ``E_nu`` has shape
        ``(n_times, nu_max + 1)``; otherwise shape ``(nu_max + 1,)`` for the requested time.
        """
        ep = self._energy_profile()
        profile = np.array([ep.energy_excitations_vec(psi) for psi in self._states_for(t)])
        if t is not None:
            profile = profile[0]
        return ep.excitation_axis, profile
