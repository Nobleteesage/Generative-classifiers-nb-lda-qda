"""Generative Gaussian classifiers: Naive Bayes, LDA and QDA.

Reusable pieces of the project:

* Gaussian Naive Bayes with a single feature Z and with a duplicated
  feature (Z, Z), and the resulting log posterior odds.
* LDA and QDA fitted before and after adding one extreme observation,
  with validation log loss, fitted variances and decision boundaries.

Author: Goriola-Obafemi Babatunde Sukanmi
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Experiment 1: Naive Bayes training data.
NB_Z = np.array([-2, -1, 0, 0, 1, 2], dtype=float)
NB_Y = np.array([0, 0, 0, 1, 1, 1])

#: Experiment 2: LDA/QDA training data, validation data and the extra point.
X_CLASS0 = np.array([-2, -1, 0, 1, 2], dtype=float)
X_CLASS1 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
X_VAL = np.array([-2.5, -0.5, 0.5, 2.5, 3.5, 5.0])
Y_VAL = np.array([0, 0, 0, 1, 1, 1])
EXTRA_POINT = (9.5, 1)
PRIORS = [0.5, 0.5]


# ---------------------------------------------------------------------------
# Experiment 1: Gaussian Naive Bayes with a duplicated feature
# ---------------------------------------------------------------------------

def fit_naive_bayes(copies: int = 1, z: np.ndarray = NB_Z,
                    y: np.ndarray = NB_Y) -> GaussianNB:
    """Fit Gaussian NB on `copies` identical copies of the feature z."""
    X = np.column_stack([z] * copies)
    return GaussianNB().fit(X, y)


def nb_proba(model: GaussianNB, z: np.ndarray) -> np.ndarray:
    """p(1 | z) for a model fitted on k copies of z."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    X = np.column_stack([z] * model.n_features_in_)
    return model.predict_proba(X)[:, 1]


def nb_log_odds(model: GaussianNB, z: np.ndarray) -> np.ndarray:
    """log p(1 | z) - log p(0 | z) for a model fitted on k copies of z."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    X = np.column_stack([z] * model.n_features_in_)
    lp = model.predict_log_proba(X)
    return lp[:, 1] - lp[:, 0]


# ---------------------------------------------------------------------------
# Experiment 2: LDA and QDA with one additional observation
# ---------------------------------------------------------------------------

def make_training_data(add_extra: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Initial training data, optionally with the extra point (9.5, 1)."""
    X = np.r_[X_CLASS0, X_CLASS1]
    y = np.r_[np.zeros(len(X_CLASS0), int), np.ones(len(X_CLASS1), int)]
    if add_extra:
        X = np.r_[X, EXTRA_POINT[0]]
        y = np.r_[y, EXTRA_POINT[1]]
    return X.reshape(-1, 1), y


def fit_lda_qda(add_extra: bool = False):
    """Fit LDA and QDA with fixed equal priors."""
    X, y = make_training_data(add_extra)
    lda = LinearDiscriminantAnalysis(priors=PRIORS).fit(X, y)
    qda = QuadraticDiscriminantAnalysis(priors=PRIORS, store_covariance=True).fit(X, y)
    return lda, qda


def proba(model, x: np.ndarray) -> np.ndarray:
    """p(1 | x) for a one-dimensional input."""
    return model.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]


def validation_log_loss(model) -> float:
    """Validation log loss on X_VAL, Y_VAL."""
    return float(log_loss(Y_VAL, proba(model, X_VAL), labels=[0, 1]))


def fitted_variances(model) -> tuple[float, float]:
    """Class variances actually used for prediction.

    For LDA the pooled variance is recovered from the slope of the log odds,
    sigma^2 = (mu_1 - mu_0) / coef, which works for every scikit-learn
    version. For QDA the class variances are read from `covariance_`.
    """
    if isinstance(model, LinearDiscriminantAnalysis):
        mu0, mu1 = model.means_.ravel()
        pooled = float((mu1 - mu0) / model.coef_[0, 0])
        return pooled, pooled
    v0, v1 = (float(c.ravel()[0]) for c in model.covariance_)
    return v0, v1


def decision_boundaries(model, lo: float = -20.0, hi: float = 20.0,
                        num: int = 40001) -> list[float]:
    """Points in [lo, hi] where p(1 | x) crosses 0.5."""
    grid = np.linspace(lo, hi, num)
    s = proba(model, grid) - 0.5
    idx = np.where((s[:-1] < 0) != (s[1:] < 0))[0]
    # linear interpolation between the two grid points around each crossing
    return [float(grid[i] - s[i] * (grid[i + 1] - grid[i]) / (s[i + 1] - s[i])) for i in idx]
