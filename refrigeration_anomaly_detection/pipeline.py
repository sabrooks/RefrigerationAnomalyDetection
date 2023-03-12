from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

import numpy as np
from numpy.typing import NDArray

import pywt

from typing import Tuple, Optional, List
import datetime as dt

Window = Tuple[int, int]


class Wavelet(TransformerMixin, BaseEstimator):
    """
    Class for calculating the wavelet of the a time series of refrigerator temperatures
    """
    widths: NDArray

    def __init__(self, widths=List[int]) -> None:
        super().__init__()
        self.widths = widths

    def fit(self, X: NDArray, y=None) -> 'Wavelet':
        return self

    def transform(self, X: NDArray, y=None) -> NDArray:
        cwtmatr = pywt.cwt(X, self.widths, "mexh")
        coef, _ = cwtmatr
        return coef


class CompressionDistortion(TransformerMixin, BaseEstimator):
    """
    Class for calculating distortion introduced by compressing the wavelet.
    """
    pca: PCA
    n_components: int
    normal_window: slice

    def __init__(self, n_components: int = 2, normal_window: Optional[slice] = None) -> None:
        super().__init__()
        self.n_components = n_components
        self.normal_window = normal_window

    def fit(self, X: NDArray, y=None) -> 'CompressionDistortion':
        """
        Fits the PCA based on the training data

        Parameters:
        ----------
        X: 2D NDArray Wavelet

        Returns:
        CompressionDistortion
        """
        pca = PCA()
        pca.set_params({"n_components": self.n_components})
        X_train = X if self.normal_window else X[slice]
        pca.fit(X_train)
        self.pca = pca
        return self

    def transform(self, X: NDArray, y=None) -> NDArray:
        """
        Compresses and reconstructs wavelet
        Parameters:
        ----------
        X: 2D NDArray Wavelet

        Returns:
        -----------
        Reconstructed Wavelet, 2D NDArray
        """
        compressed = self.pca.transform(X)
        reconstructed = self.pca.inverse_transform(compressed)
        return np.expand_dims(reconstructed, axis=-1)


class Passthrough(TransformerMixin, BaseEstimator):
    def fit(self, X: NDArray, y=None) -> 'Passthrough':
        return self

    def transform(self, X: NDArray, y=None) -> NDArray:
        return np.expand_dims(X, axis=-1)


class MSE(TransformerMixin, BaseEstimator):
    """
    Class to calculate the mean squared error between the orginal wavelet and compressed wavelet
    """

    def fit(self, X: NDArray, y=None) -> 'MSE':
        return self

    def transform(self, X: NDArray, y=None) -> NDArray:
        """
        Compresses and reconstructs wavelet
        Parameters:
        ----------
        X: 3D NDArray [Reconstructed Wavelet, Original Wavelet]

        Returns:
        -----------
        """
        y_pred = X[:, :, 0]
        y_true = X[:, :, 1]
        return mean_squared_error(y_true, y_pred)


pca_orginal = FeatureUnion([
    ("pca", CompressionDistortion(normal_window=slice(0, 100))),
    ("passthrough", Passthrough())])

anomaly_pipeline = Pipeline([
    ("wavelet", Wavelet()),
    ("pca_original", pca_orginal),
    ("mse", MSE()),
    ("threshold", SGDClassifier(loss="log"))])
