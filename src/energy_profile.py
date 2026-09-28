"""Exact energy partitions by field mode and total photon occupation."""

import numpy as np


class EnergyProfile:
    def __init__(self, experiment):
        self.experiment = experiment
        self.occupations = np.asarray(experiment.basis.states)
        self.excitation_axis = np.arange(experiment.photon_numbers.max() + 1)

    def energy_modes_vec(self, vector):
        exp = self.experiment
        probability = np.abs(vector.reshape(-1, 2)) ** 2
        field = np.abs(exp.k_tab) * (self.occupations.T @ probability.sum(axis=1))
        h = exp.hamiltonian
        contributions = np.real(vector[h.transition_rows].conj() *
                                exp.param_atom["D"] * h.transition_values *
                                vector[h.transition_cols])
        interaction = np.bincount(h.transition_modes, weights=contributions,
                                  minlength=exp.n_modes)
        atom = exp.param_atom["omega_0"] * probability[:, 1].sum()
        return field + interaction, float(atom)

    def energy_excitations_vec(self, vector):
        exp = self.experiment
        energy = np.real(vector.conj() * (exp.hamiltonian.H0 + exp.param_atom["D"] *
                                         exp.hamiltonian.V).dot(vector))
        return np.bincount(exp.photon_numbers, weights=energy.reshape(-1, 2).sum(axis=1),
                           minlength=len(self.excitation_axis))
