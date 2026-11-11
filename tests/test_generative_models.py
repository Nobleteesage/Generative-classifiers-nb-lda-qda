"""Checks against values derived by hand in the report.

The LDA/QDA tests only check properties that hold for every scikit-learn
version (older versions divide variances by n - 1, newer ones by n).
"""

import numpy as np
import pytest

from src.generative_models import (
    decision_boundaries, fit_lda_qda, fit_naive_bayes, fitted_variances,
    nb_log_odds, nb_proba, validation_log_loss,
)


# ---------------- Naive Bayes ----------------

def test_nb_fitted_parameters():
    m = fit_naive_bayes(copies=1)
    assert m.theta_.ravel() == pytest.approx([-1.0, 1.0])
    assert m.var_.ravel() == pytest.approx([2 / 3, 2 / 3], rel=1e-6)
    assert m.class_prior_ == pytest.approx([0.5, 0.5])


def test_nb_probabilities_at_04():
    assert nb_proba(fit_naive_bayes(1), 0.4)[0] == pytest.approx(0.7685, abs=1e-4)
    assert nb_proba(fit_naive_bayes(2), 0.4)[0] == pytest.approx(0.9168, abs=1e-4)


def test_nb_log_odds_formulas():
    z = np.array([-1.5, -0.3, 0.4, 1.7])
    assert nb_log_odds(fit_naive_bayes(1), z) == pytest.approx(3 * z, rel=1e-6)
    assert nb_log_odds(fit_naive_bayes(2), z) == pytest.approx(6 * z, rel=1e-6)


def test_duplicated_feature_doubles_log_odds():
    z = np.linspace(-2, 2, 41)
    z = z[np.abs(z) > 1e-9]
    ratio = nb_log_odds(fit_naive_bayes(2), z) / nb_log_odds(fit_naive_bayes(1), z)
    assert ratio == pytest.approx(np.full_like(z, 2.0), rel=1e-9)


# ---------------- LDA and QDA ----------------

def test_qda_equals_lda_on_initial_data():
    lda, qda = fit_lda_qda(add_extra=False)
    v0, v1 = fitted_variances(qda)
    assert v0 == pytest.approx(v1)
    assert validation_log_loss(lda) == pytest.approx(validation_log_loss(qda))
    assert decision_boundaries(lda) == pytest.approx([1.75], abs=1e-3)


def test_outlier_increases_both_log_losses_but_lda_more():
    lda_b, qda_b = fit_lda_qda(add_extra=False)
    lda_a, qda_a = fit_lda_qda(add_extra=True)
    assert validation_log_loss(lda_a) > validation_log_loss(lda_b)
    assert validation_log_loss(qda_a) > validation_log_loss(qda_b)
    assert validation_log_loss(qda_a) < validation_log_loss(lda_a)


def test_lda_after_boundary_moves_to_new_midpoint():
    lda, _ = fit_lda_qda(add_extra=True)
    assert lda.means_.ravel() == pytest.approx([0.0, 4.5])
    assert decision_boundaries(lda) == pytest.approx([2.25], abs=1e-3)


def test_qda_after_only_class1_variance_grows_and_second_boundary_appears():
    _, qda_b = fit_lda_qda(add_extra=False)
    _, qda_a = fit_lda_qda(add_extra=True)
    v0_b, v1_b = fitted_variances(qda_b)
    v0_a, v1_a = fitted_variances(qda_a)
    assert v0_a == pytest.approx(v0_b)
    assert v1_a > 3 * v1_b
    bounds = decision_boundaries(qda_a)
    assert len(bounds) == 2
    assert -7 < bounds[0] < -5.5
    assert 2 < bounds[1] < 2.3
