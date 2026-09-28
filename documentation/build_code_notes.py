"""Regenerate English source commentary with current, verified line ranges."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each ordered marker begins an operation block. The next marker closes it.
# Comments belong here, not in the generated LaTeX or copied source listings.
BLOCKS = {
    'src/xp_config.py': [
        ('"""', r'This module defines configuration data only. Imports do not allocate a grid or launch propagation.'),
        ('@dataclass', r'The dataclass preserves named inputs and supports \code{dataclasses.replace}; validation occurs in the consuming numerical routines.'),
        ('param_photon:', r'Packet inputs $k_0,\sigma_k,x_0$, preparation type, and either integer $n$ or complex $\alpha$; see \eqref{eq:packet}--\eqref{eq:coherent}.'),
        ('param_atom:', r'$\omega_0,D,L,x_{\mathrm{tls}}$, coupling profile, and optional initial TLS label. The box length is stored here but also controls the photon grid.'),
        ('param_time_evol:', r'Final time $T$, output spacing \code{dt}, and optional integration method/tolerances. Output spacing does not fix internal solver steps.'),
        ('cutoffs:', r'The radial band is required only under cutoff control. Explicit-count control ignores this dictionary.'),
        ('n_max:', r'$N$ and the basis select the retained occupation set, \eqref{eq:bases}. RWA selects Hamiltonian terms; storage selects retained outputs, not a different Hilbert space.'),
        ('CTRL_M_EXPLICIT:', r'True selects exact odd $M$ with zero, \eqref{eq:explicit}; false selects the radial band, \eqref{eq:cutoffs}. No hidden decrement of $M$ occurs.'),
        ('mode_selection:', r'Additional mode windows, applied after resolving the base grid. Their radii are factors times $\sigma_k$, \eqref{eq:selection}.'),
        ('def with_D', r'Return a replaced configuration with a new atomic dictionary. Other inputs are reused; the supplied dictionary is not overwritten during a sweep.')],
    'src/grid.py': [
        ('"""', r'\code{integer} accepts integral types above its minimum, including NumPy integers. It rejects booleans and all floats rather than silently coercing them.'),
        ('def momentum_modes', r'One grid generator for both evolution and estimation; its inputs describe the base grid, before any optional windows.'),
        ('L = float', r'Convert and require finite $L>0$. Require an actual boolean control so a truthy string cannot silently select the wrong branch.'),
        ('spacing =', r'Compute the physical spacing $\Delta k=2\pi/L$, \eqref{eq:grid}; it is never inferred from a chosen number of modes.'),
        ('if CTRL_M_EXPLICIT:', r'Require positive odd integer $M$ and construct indices $-J,\ldots,J$, $J=(M-1)/2$. Even counts raise an error before basis allocation.'),
        ('if cutoffs is None:', r'Cutoff control requires finite ordered radial bounds $0\leq\mathrm{IR}\leq\mathrm{UV}$. These validation steps are skipped when $M$ controls the grid.'),
        ('lower =', r'Ceiling/floor of cutoff ratios with dimensionless $10^{-12}$ boundary tolerances. The positive lower index is subsequently restricted to at least one; compare \eqref{eq:count}.'),
        ('positive =', r'Enumerate positive indices, reflect them in reversed order for the negative side, and add zero only when input IR is exactly zero. A tiny positive IR is not rounded to zero.'),
        ('modes =', r'Multiply integer indices by $\Delta k$ and reject an empty band. Sorted signed momenta, not frequencies, are returned.'),
        ('def select_modes', r'Validate positive finite $\sigma_k$ and nonnegative finite factors. This is a requested subset operation; it is not a right-moving preparation filter.'),
        ('centres =', r'Window centres $k_0,+\omega_0,-\omega_0$ with radii $w_p\sigma_k,w_a\sigma_k$, as in \eqref{eq:selection}.'),
        ('mask =', r'For each window use an absolute momentum tolerance, add the nearest mode if empty, and union its mask. Ties use the first sorted entry. The returned mask preserves base-grid ordering.'),
        ('def resolve_grid', r'Resolve the same base grid for estimator and engine. With selection return its indexed subset; otherwise return a distinct array copy with the same values.'),
        ('def grid_summary', r'Recover integer indices from the physical spacing and compute radial extrema. This representation report is independent of the photon cap and basis type.'),
        ('expected =', r'Test equality with the entire symmetric radial band, \eqref{eq:radialexact}; minima/maxima alone would miss holes or unpaired endpoints. Compute the smallest nonzero $|k|$ separately.'),
        ('return {"n_modes"', r'Report count, signed extrema, effective cutoffs, zero, integer indices and exact representability. An equivalent explicit $M$ exists only for a radial-exact set containing zero.'),
        ('return {"delta_k"', r'Return separate base and selected descriptions with their common physical spacing. A selected count may differ from the explicit base-grid input.')],
    'src/states.py': [
        ('"""', r'Combinations with replacement enumerate total-cap occupations; a Cartesian product enumerates full. These algorithms avoid testing every candidate product against a total cap.'),
        ('class FockBasis', r'Validate at least one retained mode and nonnegative integer cap. Store the representation name; $N=0$ is a valid vacuum-only photon space.'),
        ('vacuum =', r'One zero tuple of length $M$ represents the photon vacuum. TLS components are attached later through the alternating index rule \eqref{eq:index}.'),
        ('if truncation == "truncated"', r'Append $n\bm e_m$ in mode-first, occupation-second order after vacuum. Multimode configurations are absent at every cap; dimension $2(1+MN)$.'),
        ('elif truncation == "full+totalcap"', r'For each total from zero to $N$, enumerate multisets of occupied mode indices. \code{bincount} produces the tuple without duplicates; dimension $2\binom{M+N}{N}$.'),
        ('elif truncation == "full"', r'Cartesian product of $0,\ldots,N$ on each mode. The final occupation changes fastest. Unknown names, including the removed extra-oscillator scheme, raise an error.'),
        ('self.states =', r'Store the exact tuples, their dictionary indices and photon totals. Ket dimension is twice the configuration count because the TLS remains a separate binary factor.'),
        ('def initial_state', r'Read finite signed $k_0$, finite $x_0$ and positive finite $\sigma_k$. All values in the supplied selected momentum array enter preparation.'),
        ('exponent =', r'Gaussian amplitude and phase from \eqref{eq:packet}. Subtract the largest real exponent, then normalize the whole packet; the common stabilizing factor cancels.'),
        ('atom =', r'Define the normalized $g,e,+,-$ vectors in TLS order $(g,e)$ and reject unknown initial labels. Ground is the default.'),
        ('vector =', r'Allocate the joint complex ket and select number or coherent preparation. The default is number, with $n=1$ below.'),
        ('if kind == "number"', r'Require integer $0\leq n\leq N$. For $n=0$ write the TLS vector onto vacuum exactly once, independent of how many modes exist.'),
        ('for m, amplitude', r'For $n>0$, place $c_m\ket{a_0}$ at occupation $n\bm e_m$ in alternating components. This is \eqref{eq:number}, not the collective-mode Fock state \eqref{eq:collective}.'),
        ('elif kind == "coherent"', r'Read and validate complex $\alpha$. The algorithm projects a product of coherent states, not a single coherent state restricted to one ray.'),
        ('for i, occupation', r'Multiply $(\alpha c_m)^{n_m}/\sqrt{n_m!}$ over all modes for each allowed tuple, then multiply both TLS components. The common coherent exponential is omitted because it cancels in \eqref{eq:coherent}.'),
        ('norm =', r'Normalize the complete projected ket once, rejecting zero or nonfinite norm. The retained coherent mass and its basis dependence are derived in \eqref{eq:weights}; no intersection normalization is involved.')],
    'src/hamiltonians.py': [
        ('"""', r'Use COO triplets and CSR storage rather than a dense $d\times d$ allocation. QuTiP wraps the final sparse operator for evolution.'),
        ('class Hamiltonian', r'Store basis and physical parameters; require finite $L>0$, $\omega_0\geq0$ and finite TLS position. Build the free term and interaction separately.'),
        ('def free', r'For each occupation compute $\sum_m|k_m|n_m$, then place it on ground and add $\omega_0$ on excited TLS. The ground-state energy origin is explicit, \eqref{eq:entries}.'),
        ('def interaction', r'Select $f_m=\sqrt{|k_m|}$ or $1$; reject other labels. The signed momentum survives in the spatial phase even though the profile uses $|k|$.'),
        ('u =', r'Compute $u_m=\ii f_m e^{-\ii k_mx_{\mathrm{tls}}}/\sqrt L$, \eqref{eq:H}. $D$ is not multiplied yet, allowing the same interaction matrix to be scaled.'),
        ('rows, cols', r'Allocate row/column/value lists and a mode tag per transition. Tags support exact interaction-energy attribution in \eqref{eq:energytriplets}.'),
        ('for i, occupation', r'Iterate occupations, TLS labels and changed modes. The matrix column is the source $2i+s$; the target row flips the TLS to $1-s$.'),
        ('if occupation[m]', r'Annihilation requires positive occupation. Under RWA it also requires source TLS $g$. All retained bases are downward closed, so dictionary lookup of the decremented tuple is valid.'),
        ('values.append(u[m] *', r'Record $u_m\sqrt{n_m}$ and the changed-mode tag. The factor is the bosonic matrix element, with the imposed complex annihilation phase.'),
        ('if not self.RWA or atom == 1:', r'Creation is allowed for either TLS without RWA and only source $e$ under RWA. The incremented tuple is tested, not assumed present.'),
        ('if j is not None:', r'Omit targets outside the chosen space, exactly implementing projection. Retained creation receives $u_m^*\sqrt{n_m+1}$, conjugate to its reverse annihilation.'),
        ('self.transition_rows', r'Keep numerical triplet/tag arrays for diagnostics and convert COO to CSR. Zero-valued triplets can remain recorded, notably for sqrt at $k=0$.'),
        ('def build_hamiltonian', r'Require finite real $D$ and return $H_0+DV$ as a QuTiP operator. Hermiticity follows from paired transitions and real scaling, not solver output normalization.')],
    'src/fidelities.py': [
        ('"""', r'This module implements exactly two squared metrics; ambient full-space vectors and a ratio field proxy are not constructed.'),
        ('def _vector', r'Accept a QuTiP ket or one-dimensional array of the basis dimension. Normalize the entire nonzero finite ket for metric evaluation, retaining raw norm diagnostics elsewhere.'),
        ('def atom_density_matrix', r'Reshape alternating components to $C$ with shape $(B,2)$, then compute $C^{\mathsf T}C^*$ as in \eqref{eq:rho}. For a raw ket its trace is the squared norm.'),
        ('def _mode_maps', r'Start the union with the first physical momentum list and its identity map. Internal union order need not be sorted because occupation keys are canonicalized.'),
        ('for k in k_b', r'Match each second-grid mode with absolute tolerance $10^{-12}$ and zero relative tolerance. Reuse its union index, append if absent, or reject an ambiguous match.'),
        ('def _components', r'For each tuple, discard zero occupations and sort positive $(\text{union index},n)$ pairs. Store the two TLS amplitudes; missing modes are vacuum implicitly and vacuum has an empty key.'),
        ('def compare_states', r'Normalize both kets, align physical modes, and build occupation dictionaries. This low-level API cannot check box length or Hamiltonian conventions, which remain caller responsibilities.'),
        ('overlap =', r'Sum the conjugate first-state times second-state TLS dot product over every shared physical key, \eqref{eq:fstate}. Do not normalize on that intersection; example \eqref{eq:intersectionexample} would otherwise be wrong.'),
        ('rho_a, rho_b', r'Reduce the individually normalized kets, square QuTiP root fidelity, and return the two metrics clipped for numerical overshoot. Vacuum embedding leaves these atomic reductions unchanged.'),
        ('def fidelities_over_time', r'Require both histories, exactly matching output arrays and equal $L$ to absolute $10^{-12}$. These checks precede pairing, preventing silent comparison of different time samples or boxes.'),
        ('pairs =', r'Compare matching stored states and return two one-dimensional arrays of length $N_t$. The simulated comparator is an independent input, not the ambient embedding choice.')],
    'src/energy_profile.py': [
        ('"""', r'Energy partitions consume the already-built Hamiltonian and propagated vector; there is no second interaction implementation.'),
        ('class EnergyProfile', r'Cache the occupation matrix $n_{i,m}$ and photon-total axis $0,\ldots,\nu_{\max}$. The function $\nu(q)$ and projectors $Q_\nu$ are defined in \eqref{eq:energy-sector-projectors}; full can reach $MN$.'),
        ('def energy_modes_vec', r'Reshape the raw vector to $C_{i,s}$ of shape $(B,2)$ and form $|C_{i,s}|^2$. Calculations use raw quadratic expectations; see \eqref{eq:energy-raw-normalized} for normalization.'),
        ('field =', r'The transpose occupation matrix times summed TLS probabilities gives $\langle N_m\rangle=\sum_{i,s}n_{i,m}|C_{i,s}|^2$. Multiply by $\omega_m=|k_m|$ to obtain the bare field term $F_m$ in \eqref{eq:energy-contributions}.'),
        ('contributions =', r'For each record $\ell$, index source $c_\ell$, target $r_\ell$ and complex value $z_\ell$, then compute $\Re[v_{r_\ell}^*Dz_\ell v_{c_\ell}]$. The record and derivation are \eqref{eq:energy-record-definition}--\eqref{eq:energytriplets}; values contain $u_m$ but not $D$.'),
        ('interaction =', r'Weighted \code{bincount} by tag $m_\ell$ sums directed terms into $W_m=\langle DV_m\rangle$. Both reverse orientations are already present; do not multiply this accumulated result by two, \eqref{eq:energy-reverse-pair}.'),
        ('atom =', r'Return $E_m=F_m+W_m$ and the separate bare TLS term $E_a=\omega_0\sum_i|C_{i,1}|^2$. Interaction is assigned to the mode columns by convention, not included in the TLS marker, \eqref{eq:energy-allocation}.'),
        ('def energy_excitations_vec', r'Use the same experiment and Hamiltonian; the result is indexed by photon total, keeping both atomic components. It is independent of the mode-tag partition.'),
        ('energy =', r'Form the sparse matrix action and its row terms $\Re[v_q^*(Hv)_q]$, which define \eqref{eq:energy-sector-row-definition}. These terms include cross-sector coherences rather than projecting the Hamiltonian to its diagonal blocks.'),
        ('return np.bincount', r'First sum the two TLS rows of each photon configuration, then group by total photon number. This gives $E_\nu=\langle(Q_\nu H+H Q_\nu)/2\rangle$, \eqref{eq:energysectors}; it is not conditional energy \eqref{eq:energy-conditional}.')],
    'src/experiment.py': [
        ('"""', r'One common engine imports grid, basis, Hamiltonian and diagnostics. \code{momentum_modes} remains importable from this module for API convenience.'),
        ('def output_times', r'Validate finite positive $T,\code{dt}$. Generate nominal output spacing with a roundoff tolerance, always including zero and exact $T$; if $\code{dt}>T$, return both endpoints.'),
        ('def resource_estimation', r'Build configuration from the dictionary API and call the same grid resolver as propagation. Optional selection requires packet information; dimension is evaluated without enumerating a basis.'),
        ('if truncation ==', r'Use exact dimensions \eqref{eq:dimensions} with the selected count. Reject unknown basis names; no dense allocation is needed to estimate exponential full-space growth.'),
        ('times = output_times', r'Count outputs and estimate $16d$ bytes per ket, retained history depending on storage, two retained vector copies and the sparse-entry bound, \eqref{eq:memory}.'),
        ('summary =', r'Return base/selected representations and heuristic feasibility. Printed output exposes other-control implications and explicitly names excluded memory costs; it does not enforce a solver allocation limit.'),
        ('def estimate_config', r'Forward all grid, cap, selection and storage inputs from the dataclass to the dictionary estimator without a separate estimate formula.'),
        ('class Experiment', r'Copy physical dictionaries, resolve base/selected grids, generate output times, build basis and projected Hamiltonian, then prepare the ket. Coefficient arrays are absent until propagation.'),
        ('def propagate_state', r'Run \code{sesolve} with BDF/default tolerances, raw output normalization and final-state storage. Build a contiguous vector array; alternating ground/excited slices are views, shape $(N_t,B)$ or $(B,)$ for final-only.'),
        ('def _one_photon_indices', r'Find tuples with total exactly one, then identify their occupied mode. This works across all three basis orderings and gives an empty list at $N=0$.'),
        ('def compute_observables', r'Require propagation, promote final-only arrays internally, and compute raw norm, TLS excitation, signed $1g$ populations and all photon totals. Sum identities are \eqref{eq:populationchecks}.'),
        ('def one_photon_wavefunction', r'The first argument is an output index, not a physical time. Extract only $1g$ amplitudes and multiply by positive Fourier phases and $L^{-1/2}$, \eqref{eq:wavefunction}.'),
        ('def _vectors_for', r'Resolve diagnostic requests: all retained states for None, final for $-1$, or nearest stored output for an in-range time. No interpolation occurs; final-only runs reject earlier requests.'),
        ('def compute_atom_density_matrix', r'Reduce each requested raw ket. None returns a list, an explicit time one matrix; final-only with None returns a one-element list.'),
        ('def compute_excited_probability', r'Return the real excited diagonal of the raw TLS reduction. None returns an array, an explicit time a scalar; norm drift remains visible.'),
        ('def compute_entropy', r'Normalize the atomic reduction by the full squared ket norm before natural-log von Neumann entropy. This is \eqref{eq:entropy}, bounded by $\log2$ for normalized valid states.'),
        ('def compute_energy(self', r'Apply the same sparse $H_0+DV$ and take the real raw expectation. Return a time array or scalar without introducing a second energy-origin convention.'),
        ('def compute_energy_profile_modes', r'For requested kets, return $(k,E_m,0.0,E_a)$ with histories or one profile. The zero is a retained tuple-interface placeholder, not a supplemental atomic mode.'),
        ('def compute_energy_profile_excitations', r'Return the photon-total axis and row-grouped energy partition. The time selector follows the same rules as other diagnostics; summed energy agrees with direct expectation.')],
}

FUNCTION_NOTES = {
    'run': r'Construct one \code{Experiment}, propagate, then compute observables in that order. Return the numerical object so notebook analyses can reuse its kets and Hamiltonian.',
    'save_arrays': r'Create the output directory and write NPZ arrays plus \code{configuration_json} from the base dataclass. Non-JSON values stringify; ordinary numerical arrays can be loaded without pickle.',
    'parser': r'Shared CLI names use $D,L,M,k_0,\sigma_k,x_0,N,T$. Default profile is sqrt. Explicit $M$ requires its boolean flag; default CLI grid control uses cutoffs.',
    'config_from_args': r'Construct the three dictionaries and dataclass with the current conventions. Both count and cutoff inputs may be stored, but only the selected control is active; historical values are not automatically converted.',
    'finish': r'With an output path, save the figure and same-stem NPZ, then close it. Otherwise display the figure. Backend selection remains the environment or notebook responsibility.',
    'run_scattering': r'Preserve the user dictionary API and append grid-control inputs. Run shared dynamics; optional CSV contains observable columns only and overwrites its fixed repository results path.',
    'plot_scattering': r'Plot initial/middle/final $1g$ Fourier densities and directional/zero/TLS/norm populations. The interaction marker is $-x_{\mathrm{tls}}$, derived in \eqref{eq:position}.',
    'run_atom_evolution': r'Replace the configuration by each scheme and force history storage. Return a dictionary of experiments, so excitation, entropy and notebook fidelity derive from the same stored trajectories.',
    'plot_atom_evolution': r'Plot raw $P_e$ and normalized TLS entropy against output times for every returned scheme. This plot does not itself propagate or compute cross-state fidelity.',
    'run_cap_convergence': r'Require at least two strictly increasing caps; run total-cap final-only trajectories at each. Compare adjacent list entries initially and finally using physical embedding; retain all run objects.',
    'plot_cap_convergence': r'Two metric panels distinguish initial and final fidelities. The horizontal value is the lower cap in each adjacent pair, not necessarily a comparison of $N$ with $N+1$.',
    'run_coupling_sweep': r'Validate a finite nonempty one-dimensional $D$ array and force histories. At each $D$, propagate the chosen comparator and each candidate, reusing it for an identical scheme. Return $(S,Q)$ sample means and initial metrics, \eqref{eq:mean}.',
    'plot_coupling_sweep': r'Plot sample means versus physical $D$ and dashed initial values, separately for both metrics. The unsuffixed result is neither final fidelity nor a time-integral quadrature.',
    'run_energy_profile': r'Require history, propagate once, and attach mode/TLS, photon-number and total energies to the experiment. Their shapes and sum identities are documented above.',
    'plot_energy_profile': r'Plot time--momentum and time--photon-total heatmaps plus total/TLS energy. Negative partition values are energies, not invalid probabilities.',
    'animate_energy_profile': r'Reuse existing arrays for animation and optional GIF output, with stride and frame rate affecting display only. The marker at momentum zero represents separate TLS energy, not another oscillator.',
    'run_mode_selection': r'Use an unselected total-cap comparator at each coupling. Sweep schemes and both selection flags; return $(S,2,Q)$ means/initial metrics. At one fixed $D$, scan windows to return $(S,P,A)$ mean-only heatmaps.',
    'plot_mode_selection': r'Plot selection on/off sweeps and window maps for both metrics. Transpose the stored $(P,A)$ slice for plotting so the horizontal axis remains photon window and vertical axis atom window.',
    'main': r'Parse current CLI inputs, report relevant estimates before simulation, call the module numerical routine, then plot/export. The main guard preserves importability from notebooks.',
}


def generate():
    sections = []
    for filename, blocks in BLOCKS.items():
        lines = (ROOT / filename).read_text().splitlines()
        matches = []
        for marker, comment in blocks:
            after = matches[-1][0] if matches else 0
            found = [i + 1 for i, line in enumerate(lines) if i + 1 > after and marker in line]
            if not found:
                raise ValueError(f'Missing marker {marker!r} in {filename}')
            matches.append((found[0], comment))
        sections.append(r'\paragraph{\code{' + filename + '}}\n')
        sections.append(r'\lstinputlisting{../' + filename + '}\n')
        sections.append(r'\begin{longtable}{p{.12\textwidth}p{.79\textwidth}}' + '\n')
        sections.append(r'\toprule Lines & Mathematical and implementation commentary\\\midrule\endhead' + '\n')
        for index, (first, comment) in enumerate(matches):
            last = matches[index + 1][0] - 1 if index + 1 < len(matches) else len(lines)
            sections.append(f'{first}--{last} & {comment}' + r'\\' + '\n')
        sections.append(r'\bottomrule\end{longtable}' + '\n')
    for path in sorted((ROOT / 'experiment').glob('*.py')):
        if path.name == '__init__.py':
            continue
        filename = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text())
        sections.append(r'\paragraph{\code{' + filename + '}}\n')
        sections.append(r'\lstinputlisting{../' + filename + '}\n')
        sections.append(r'\begin{longtable}{p{.12\textwidth}p{.79\textwidth}}' + '\n')
        sections.append(r'\toprule Lines & Mathematical and implementation commentary\\\midrule\endhead' + '\n')
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                comment = FUNCTION_NOTES[node.name]
                sections.append(f'{node.lineno}--{node.end_lineno} & ' + r'\code{' + node.name + '}: ' + comment + r'\\' + '\n')
        sections.append(r'\bottomrule\end{longtable}' + '\n')
    target = ROOT / 'documentation' / 'code_notes.tex'
    target.write_text('% Generated by build_code_notes.py; edit the generator.\n' + ''.join(sections))
    print(f'Generated {target.relative_to(ROOT)} from actual line numbers.')


if __name__ == '__main__':
    generate()
