'''Check that the refactored evaluation reproduces the paper results.

fixtures/baseline_cell4.pkl was produced by running cell 4 of notebooks/Explain embeddings.ipynb
with the original overlap_utils (commit 4b170b1). The setup below mirrors that cell.
Slow (~2.5 min) and needs the data folder: run with `uv run pytest -m slow`.
'''
import os
import pickle

import numpy as np
import pytest

import data_utils as du
import overlap_utils as ou

REPO = os.path.join(os.path.dirname(__file__), '..')
GFM_MODS = ['alphaearth', 'tessera', 'geoclip', 'satclip']


def run_cell4(n_splits=20):
    r2_per_split = {}
    r2_cols, mse_cols = [], []
    for sample_type in ['cropharvest', 'biomass', 'random_sample', 'lc_stratified_sample']:
        extra = ['bioclim', 'human_footprint'] if sample_type in ['lc_stratified_sample', 'random_sample'] else []
        df_all, col_names = du.merge_modalities(parent_folder=os.path.join(REPO, 'data/'), sample_type=sample_type,
                                                modalities=GFM_MODS + extra, zscore_embeddings=False)
        regressor_list = list(GFM_MODS)
        for im1, m1 in enumerate(GFM_MODS):
            for m2 in GFM_MODS[im1 + 1:]:
                col_names[f'{m1}_{m2}'] = col_names[m1] + col_names[m2]
                regressor_list.append(f'{m1}_{m2}')
        regressor_list.append('all_gfm')

        method = 'regression'
        kwargs = {'n_splits': n_splits, 'regression_method': 'ridge', 'zscore_embeddings': True}
        if sample_type == 'cropharvest':
            target_list = list(col_names['cropharvest'])
            method, kwargs = 'classification', {'n_splits': n_splits, 'zscore_embeddings': True}
        elif sample_type == 'biomass':
            target_list = list(col_names['biomass'])
        elif sample_type == 'lc_stratified_sample':
            col_names['pop_density'] = ['pop_density']
            target_list = ['dynamicworld', 'pop_density'] + col_names['dynamicworld']
        else:
            col_names['meandist_road'] = ['meandist_road']
            target_list = ['bioclim', 'meandist_road']
        for t in target_list:
            col_names.setdefault(t, [t])

        r2, mse, _, per_split = ou.get_overlap_matrix(df_all=df_all, col_names=col_names, method=method,
                                                      regressor_list=regressor_list, target_list=target_list,
                                                      kwargs_for_method=kwargs)
        r2_cols.append(r2)
        mse_cols.append(mse)
        r2_per_split.update(per_split)
    return np.concatenate(r2_cols, axis=1), np.concatenate(mse_cols, axis=1), r2_per_split


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, 'data', 'downstream_tasks')), reason='data not available')
def test_reproduces_paper_results(monkeypatch):
    monkeypatch.chdir(os.path.join(REPO, 'notebooks'))  # merge_modalities loads '../content/...'
    with open(os.path.join(REPO, 'tests', 'fixtures', 'baseline_cell4.pkl'), 'rb') as f:
        baseline = pickle.load(f)
    r2, mse, r2_per_split = run_cell4()

    # The land-cover targets of lc_stratified_sample are float32 (content/centre_pixel_lc_values.npy).
    # The original code fitted those in float32, the new code in float64: differences up to ~1e-7.
    atol = 1e-6
    np.testing.assert_allclose(r2, baseline['r2_matrix'], rtol=0, atol=atol)
    np.testing.assert_allclose(mse, baseline['mse_matrix'], rtol=0, atol=atol)
    assert r2_per_split.keys() == baseline['r2_per_split'].keys()
    for k, v in baseline['r2_per_split'].items():
        np.testing.assert_allclose(r2_per_split[k], v, rtol=0, atol=atol, err_msg=str(k))
