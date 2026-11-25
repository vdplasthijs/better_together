"""Fusion methods: combine several embedding blocks into one feature matrix.

A fusion method receives X with the columns of each embedding side by side, in the
order of `block_dims` (e.g. {'alphaearth': 64, 'tessera': 128}). It is fitted on the
training fold only, so methods that learn something (PCA, CCA, ...) do not leak.
"""
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class Fusion(TransformerMixin, BaseEstimator, ABC):
    def __init__(self, block_dims):
        self.block_dims = block_dims

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X):
        ...


class ConcatFusion(Fusion):
    """Concatenate all blocks (X is already concatenated, so this is the identity)."""

    def transform(self, X):
        return X
