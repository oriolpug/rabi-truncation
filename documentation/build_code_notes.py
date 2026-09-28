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
        ('def momentum_modes', r'Construct the physical base grid used by evolution, before any optional windows.'),
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
        ('def resolve_grid', r'Resolve the base grid for the engine. With selection return its indexed subset; otherwise return a distinct array copy with the same values.'),
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
        ('rows, cols', r'Temporary row/column/value lists hold the sparse interaction entries. A column labels the joint source ket and a row its target; coefficients contain $u_m$ but not $D$.'),
        ('for i, occupation', r'Iterate occupations, TLS labels and changed modes. The matrix column is the source $2i+s$; the target row flips the TLS to $1-s$.'),
        ('if occupation[m]', r'Annihilation requires positive occupation. Under RWA it also requires source TLS $g$. All retained bases are downward closed, so dictionary lookup of the decremented tuple is valid.'),
        ('values.append(u[m] *', r'Record $u_m\sqrt{n_m}$. The factor is the bosonic matrix element, with the imposed complex annihilation phase.'),
        ('if not self.RWA or atom == 1:', r'Creation is allowed for either TLS without RWA and only source $e$ under RWA. The incremented tuple is tested, not assumed present.'),
        ('if j is not None:', r'Omit targets outside the chosen space, exactly implementing projection. Retained creation receives $u_m^*\sqrt{n_m+1}$, conjugate to its reverse annihilation.'),
        ('return coo_matrix', r'Convert temporary COO row/column/value records to the CSR interaction matrix. The assembly buffers are not kept as separate diagnostic arrays; zero coefficients can remain as stored sparse entries.'),
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
    'src/experiment.py': [
        ('"""', r'One common engine imports grid, basis, Hamiltonian and diagnostics. \code{momentum_modes} remains importable from this module for API convenience.'),
        ('def output_times', r'Validate finite positive $T,\code{dt}$. Generate nominal output spacing with a roundoff tolerance, always including zero and exact $T$; if $\code{dt}>T$, return both endpoints.'),
        ('def resource_estimation', r'Compute one informative DataFrame directly from dictionary inputs: cavity, retained mode count, physical bounds, output settings and vector memory. No project helper or feasibility threshold is involved.'),
        ('if truncation ==', r'Use exact dimensions \eqref{eq:dimensions} with the selected count. Reject unknown basis names; no dense allocation is needed to estimate exponential full-space growth.'),
        ('T, dt =', r'Count endpoint-inclusive outputs directly from $T$ and $\code{dt}$. Estimate $16d$ bytes per ket and two retained vector copies, depending on history storage; see \eqref{eq:memory}.'),
        ('rows =', r'Build the single group/parameter/value DataFrame. Retained $M$, signed bounds and effective radial extrema expose the chosen representation. The caller decides whether to display the table and run.'),
        ('def estimate_config', r'Forward all grid, cap, selection and storage inputs from the dataclass to the dictionary estimator without a separate estimate formula.'),
        ('class Experiment', r'Copy physical dictionaries, resolve base/selected grids, generate output times, build basis and projected Hamiltonian, then prepare the ket. Coefficient arrays are absent until propagation.'),
        ('def propagate_state', r'Run \code{sesolve} with BDF/default tolerances, raw output normalization and final-state storage. Build a contiguous vector array; alternating ground/excited slices are views, shape $(N_t,B)$ or $(B,)$ for final-only.'),
        ('def _one_photon_indices', r'Find tuples with total exactly one, then identify their occupied mode. This works across all three basis orderings and gives an empty list at $N=0$.'),
        ('def compute_observables', r'Require propagation, promote final-only arrays internally, and compute raw norm, TLS excitation, signed $1g$ populations and all photon totals. Sum identities are \eqref{eq:populationchecks}.'),
        ('def observables_dataframe', r'Return raw observable histories indexed by time or an endpoint table. Final-only storage yields a final column only. Formatting neither rounds stored data nor renormalizes probabilities.'),
        ('def one_photon_wavefunction', r'The first argument is an output index, not a physical time. Extract only $1g$ amplitudes and multiply by positive Fourier phases and $L^{-1/2}$, \eqref{eq:wavefunction}.'),
        ('def _vectors_for', r'Resolve diagnostic requests: all retained states for None, final for $-1$, or nearest stored output for an in-range time. No interpolation occurs; final-only runs reject earlier requests.'),
        ('def compute_atom_density_matrix', r'Reduce each requested raw ket. None returns a list, an explicit time one matrix; final-only with None returns a one-element list.'),
        ('def compute_excited_probability', r'Return the real excited diagonal of the raw TLS reduction. None returns an array, an explicit time a scalar; norm drift remains visible.'),
        ('def compute_energy(self', r'Apply the same sparse $H_0+DV$ and take the real raw expectation. Return a time array or scalar without introducing a second energy-origin convention.'),
    ],
}

FUNCTION_NOTES = {
    'run_scattering': r'Assemble ExperimentConfig from named dictionary inputs; construct \code{Experiment}, call \code{propagate_state} and \code{compute_observables} directly. Return the propagated object. Optional CSV stores raw observable columns at its fixed repository results path.',
    'run_atom_evolution': r'For each scheme, replace the configuration and force history storage, construct Experiment, propagate and compute observables explicitly. Return the experiments for notebook excitation, entropy and fidelity analyses.',
    'run_cap_convergence': r'Require at least two strictly increasing caps; run total-cap final-only trajectories at each. Compare adjacent list entries initially and finally using physical embedding; retain all run objects.',
    'run_coupling_sweep': r'Validate a finite nonempty one-dimensional $D$ array and force histories. At each $D$, propagate the chosen comparator and each candidate, reusing it for an identical scheme. Return $(S,Q)$ sample means and initial metrics, \eqref{eq:mean}.',
    'run_mode_selection': r'Use an unselected total-cap comparator at each coupling. Sweep schemes and both selection flags; return $(S,2,Q)$ means/initial metrics. At one fixed $D$, scan windows to return $(S,P,A)$ mean-only heatmaps.',
}


def generate():
    """Regenerate English LaTeX source listings and verified line-range comments.

    Parameters
    ----------
    None
        Uses ROOT, ordered BLOCKS markers and FUNCTION_NOTES in this module.

    Returns
    -------
    None
        Writes documentation/code_notes.tex with listings of current sources
        and commentary tied to their actual line numbers. Missing operation
        markers raise ValueError; compiling main.tex is a separate step.
    """
    sections = []
    for filename, blocks in BLOCKS.items():
        lines = (ROOT / filename).read_text().splitlines()
        tree = ast.parse('\n'.join(lines))
        doc_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and ast.get_docstring(node) is not None:
                first = node.body[0]
                doc_lines.update(range(first.lineno, first.end_lineno + 1))
        matches = []
        for marker, comment in blocks:
            after = matches[-1][0] if matches else 0
            found = [i + 1 for i, line in enumerate(lines)
                     if i + 1 > after and marker in line
                     and (i + 1 not in doc_lines or (marker == '"""' and i == 0))]
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
