import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import evaluation as ev
from fusion import ConcatFusion


def make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 4)) * 10 + 3  # unscaled on purpose
    b = rng.normal(size=(n, 6))
    y = np.column_stack([a[:, 0] + b[:, 0], 100 * b[:, 1]]) + 0.1 * rng.normal(size=(n, 2))
    df = pd.DataFrame(np.hstack([a, b, y]), columns=[f'a{i}' for i in range(4)] + [f'b{i}' for i in range(6)] + ['y0', 'y1'])
    df['label'] = np.where(a[:, 1] + b[:, 2] > 3, 'high', 'low')
    col_names = {'a': [f'a{i}' for i in range(4)], 'b': [f'b{i}' for i in range(6)], 'y': ['y0', 'y1']}
    return df, col_names


def test_concat_fusion_is_identity_and_clonable():
    X = np.arange(12.0).reshape(3, 4)
    fusion = ConcatFusion({'a': 1, 'b': 3})
    assert np.array_equal(fusion.fit_transform(X), X)
    assert clone(fusion).block_dims == {'a': 1, 'b': 3}


@pytest.mark.parametrize('scaling', ['fold', 'global', None])
def test_regression_uses_both_blocks(scaling):
    df, col_names = make_data()
    both = ev.cross_validate(df, col_names, ['a', 'b'], 'y', scaling=scaling)
    only_a = ev.cross_validate(df, col_names, ['a'], 'y', scaling=scaling)
    assert both.score > 0.99
    assert only_a.score < 0.6
    assert len(both.score_per_split) == 4
    assert both.predictions.shape == both.target.shape == (len(df), 2)


@pytest.mark.parametrize('scaling', ['fold', 'global', None])
def test_classification(scaling):
    df, col_names = make_data()
    res = ev.cross_validate(df, col_names, ['a', 'b'], 'label', task_type='classification', scaling=scaling)
    assert res.score > 0.95
    assert np.all(np.isnan(res.mse_per_point))


def test_splits_are_shared_across_inputs():
    '''Paired comparisons need identical folds, whatever the inputs.'''
    df, col_names = make_data()
    r1 = ev.cross_validate(df, col_names, ['a'], 'y', n_splits=5)
    r2 = ev.cross_validate(df, col_names, ['a', 'b'], 'y', n_splits=5)
    assert np.array_equal(r1.target, r2.target)
    assert len(r1.score_per_split) == len(r2.score_per_split) == 5
