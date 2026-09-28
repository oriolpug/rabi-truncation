"""Tabular presentation of grid, memory and population diagnostics."""

import pandas as pd


def resource_tables(estimate, config):
    """Describe the chosen grid and its cost as two reusable DataFrames.

    Parameters
    ----------
    estimate : dict
        Numerical resource estimate, including ``grid`` with base/selected
        momentum descriptions. Dimensions count both TLS states.
    config : ExperimentConfig
        Inputs used for the estimate; determines which grid control is active.

    Returns
    -------
    tables : dict[str, pandas.DataFrame]
        ``grid_table`` has one row per base/selected grid, physical bounds,
        count, zero-mode policy and exact equivalence to the other control.
        ``resource_table`` has one row of basis, storage and memory estimates.
        GiB means bytes / 2**30; one complex128 ket requires 16*d bytes.

    Notes
    -----
    Table attributes record requested inputs and excluded memory costs.
    Effective IR/UV are extrema of retained |k|, not the requested bounds.
    A missing equivalent M means an explicit symmetric zero-containing grid
    cannot reproduce that subset. No basis or Hamiltonian is allocated here.
    """
    grid = estimate['grid']
    grid_table = pd.DataFrame.from_dict(
        {name: {key: value for key, value in grid[name].items()
                if key != 'integer_indices'} for name in ('base', 'selected')},
        orient='index')
    grid_table.index.name = 'grid'
    grid_table['equivalent_M'] = grid_table['equivalent_M'].astype('Int64')
    grid_table.insert(0, 'L', config.param_atom['L'])
    grid_table.insert(1, 'delta_k', grid['delta_k'])
    grid_table.insert(2, 'control', 'explicit M' if config.CTRL_M_EXPLICIT else 'IR/UV')
    grid_table.insert(3, 'requested_M', config.M if config.CTRL_M_EXPLICIT else pd.NA)
    cutoffs = config.cutoffs if not config.CTRL_M_EXPLICIT else None
    grid_table.insert(4, 'requested_IR', cutoffs['ir_cutoff'] if cutoffs else pd.NA)
    grid_table.insert(5, 'requested_UV', cutoffs['uv_cutoff'] if cutoffs else pd.NA)
    grid_table.attrs['integer_indices'] = {
        name: grid[name]['integer_indices'] for name in ('base', 'selected')}
    resource_table = pd.DataFrame([{
        'basis': config.truncation, 'n_max': config.n_max,
        'store_state': config.store_state,
        **{key: estimate[key] for key in (
            'n_modes', 'dimension', 'n_times', 'ket_gib', 'history_gib',
            'retained_vectors_gib', 'hamiltonian_nnz_bound', 'feasible')}}])
    resource_table.attrs['memory_excludes'] = (
        'Sparse matrices, basis/Python objects, solver workspace and plots.')
    resource_table.attrs['feasibility_rule'] = (
        'dimension <= 100000 and retained_vectors_gib <= 1; heuristic only.')
    return {'grid_table': grid_table, 'resource_table': resource_table}


def display_resource_tables(tables):
    """Display resource DataFrames in a notebook or as terminal tables.

    Parameters
    ----------
    tables : dict[str, pandas.DataFrame]
        Mapping containing ``grid_table`` and ``resource_table``; an estimate
        returned by resource_estimation also satisfies this contract.

    Returns
    -------
    None
        Displays two HTML-capable tables in an active IPython shell, or prints
        their tabular text in a terminal. The input frames are not modified.
    """
    try:
        from IPython import get_ipython
        from IPython.display import display
        interactive = get_ipython() is not None
    except ImportError:
        interactive = False
    for name in ('grid_table', 'resource_table'):
        if interactive:
            display(tables[name])
        else:
            print(tables[name].to_string())


def observables_table(observables, summary=False, initial_available=True):
    """Convert raw time-series diagnostics to a history or endpoint table.

    Parameters
    ----------
    observables : dict[str, numpy.ndarray]
        Equal-length real arrays from Experiment.compute_observables, with
        ``time``, squared ket norm and raw sector populations.
    summary : bool, optional
        False returns all samples; True returns one row per observable and
        columns ``initial``/``final`` when the initial state was stored.
    initial_available : bool, optional
        False identifies final-only storage, so a summary has only ``final``.

    Returns
    -------
    frame : pandas.DataFrame
        History: shape (K, J), indexed by time, with J observable columns.
        Summary: shape (J+1, 2), or (J+1, 1) for final-only runs; includes time
        as an observable row. Values retain solver norm drift and precision.
    """
    frame = pd.DataFrame(observables)
    if not summary:
        return frame.set_index('time')
    if initial_available:
        result = frame.iloc[[0, -1]].T
        result.columns = ['initial', 'final']
    else:
        result = frame.iloc[[-1]].T
        result.columns = ['final']
    result.index.name = 'observable'
    return result
