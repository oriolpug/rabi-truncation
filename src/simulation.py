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
        cls = {"full": StateFull, "truncated": StateTruncated, "truncated+atom": StateAtom}[self.config.truncation]
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
            return [qutip.expect(self.H, state.full()[:, 0]) for state in self.result.states]
        elif t == -1:  # get last value
            return qutip.expect(self.H, self.result.states[-1].full()[:, 0])
        else:  # get value at index closest to t
            idx = int(t / self.config.dt)
            return qutip.expect(self.H, self.result.states[idx].full()[:, 0])
