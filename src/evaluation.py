"""Cross-validated evaluation of a (fusion, head) pipeline that predicts a target from embeddings."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import zscore
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusion import ConcatFusion


@dataclass
class CVResult:
    score: float  # regression: R2 over all out-of-fold predictions; classification: mean accuracy over splits
    score_per_split: np.ndarray
    mse_per_point: np.ndarray  # regression: squared error normalised by target variance; classification: NaN
    target: np.ndarray
    predictions: np.ndarray


def cross_validate(df, col_names, inputs, target, task_type='regression', fusion=None, head=None,
                   n_splits=4, scaling='fold', random_state=42):
    '''Predict `target` from the embeddings in `inputs` (list of keys in col_names) with k-fold CV.

    scaling:
        'fold':   z-score inputs (and regression targets) with training-fold statistics.
                  Regression R2 is the uniform average over outputs.
        'global': z-score inputs and regression targets on all data before splitting. This
                  reproduces the original paper results; R2 is variance-weighted over outputs.
        None:     no scaling; R2 is variance-weighted over outputs.
    The folds depend only on n_splits, random_state and len(df), so different inputs, fusions and
    heads are evaluated on the same splits (needed for paired tests and complementarity).
    '''
    assert n_splits > 1
    assert task_type in ['regression', 'classification'], f'Task type {task_type} not supported.'
    assert scaling in ['fold', 'global', None], f'Scaling {scaling} not supported.'
    block_dims = {m: len(col_names[m]) for m in inputs}
    X = df[[c for m in inputs for c in col_names[m]]].values.astype(float)
    if scaling == 'global':
        X = zscore(X, axis=0)

    if fusion is None:
        fusion = ConcatFusion(block_dims)
    if head is None:
        head = Ridge(alpha=1.0) if task_type == 'regression' else LogisticRegression(max_iter=1000)
    steps = [StandardScaler()] if scaling == 'fold' else []
    pipeline = make_pipeline(*steps, fusion, head)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if task_type == 'regression':
        return _cross_validate_regression(X, df[col_names.get(target, [target])].values.astype(float),
                                          pipeline, kf, scaling)
    else:
        return _cross_validate_classification(X, df[target], pipeline, kf)


def _cross_validate_regression(X, Y, pipeline, kf, scaling):
    if scaling == 'global':
        Y = zscore(Y, axis=0)
    if scaling == 'fold':
        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        multioutput = 'uniform_average'
    else:
        model = pipeline
        multioutput = 'variance_weighted'
        var_target = np.var(Y, axis=0)

    Y_pred = np.zeros_like(Y)
    mse_per_point = np.zeros(len(Y))
    score_per_split = []
    for train_index, test_index in kf.split(X):
        fitted = clone(model).fit(X[train_index], Y[train_index])
        pred = fitted.predict(X[test_index]).reshape(len(test_index), -1)
        Y_pred[test_index] = pred
        var = np.var(Y[train_index], axis=0) if scaling == 'fold' else var_target
        mse_per_point[test_index] = np.mean((Y[test_index] - pred) ** 2 / var, axis=1)
        score_per_split.append(r2_score(Y[test_index], pred, multioutput=multioutput))

    return CVResult(score=r2_score(Y, Y_pred, multioutput=multioutput), score_per_split=np.array(score_per_split),
                    mse_per_point=mse_per_point, target=Y, predictions=Y_pred)


def _cross_validate_classification(X, labels, pipeline, kf):
    class_to_int = {cls: i for i, cls in enumerate(labels.unique())}
    y = labels.map(class_to_int).values

    y_pred = np.zeros(len(y), dtype=int)
    score_per_split = []
    for train_index, test_index in kf.split(X):
        fitted = clone(pipeline).fit(X[train_index], y[train_index])
        y_pred[test_index] = fitted.predict(X[test_index])
        score_per_split.append(np.mean(y_pred[test_index] == y[test_index]))

    return CVResult(score=np.mean(score_per_split), score_per_split=np.array(score_per_split),
                    mse_per_point=np.full(len(y), np.nan), target=y, predictions=y_pred)
