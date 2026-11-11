"""Reproduce every number and figure reported in the README.

Usage:
    python run_experiments.py

Outputs:
    results/naive_bayes.csv
    results/lda_qda_summary.csv
    results/lda_qda_validation_predictions.csv
    figures/nb_probabilities.png
    figures/nb_log_odds.png
    figures/lda_qda_curves.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from src.generative_models import (
    EXTRA_POINT, NB_Y, NB_Z, X_CLASS0, X_CLASS1, X_VAL, Y_VAL,
    decision_boundaries, fit_lda_qda, fit_naive_bayes, fitted_variances,
    nb_log_odds, nb_proba, proba, validation_log_loss,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

BLUE, RED, ORANGE = "#1f77b4", "#d62728", "#ff7f0e"
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def naive_bayes_experiment() -> None:
    model_a = fit_naive_bayes(copies=1)
    model_b = fit_naive_bayes(copies=2)

    z0 = 0.4
    df = pd.DataFrame({
        "model": ["Model A: Z", "Model B: (Z, Z)"],
        "mu0": [model_a.theta_[0, 0], model_b.theta_[0, 0]],
        "mu1": [model_a.theta_[1, 0], model_b.theta_[1, 0]],
        "var": [model_a.var_[0, 0], model_b.var_[0, 0]],
        "log_odds_at_0.4": [nb_log_odds(model_a, z0)[0], nb_log_odds(model_b, z0)[0]],
        "p1_at_0.4": [nb_proba(model_a, z0)[0], nb_proba(model_b, z0)[0]],
    })
    df.to_csv(RES / "naive_bayes.csv", index=False, float_format="%.4f")
    print("\nGaussian Naive Bayes\n", df.round(4).to_string(index=False))

    z = np.linspace(-2, 2, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z, nb_proba(model_a, z), lw=2, color=BLUE, label="Model A: feature $Z$")
    ax.plot(z, nb_proba(model_b, z), lw=2, color=ORANGE, label="Model B: features $(Z, Z)$")
    ax.scatter(NB_Z, NB_Y, c="k", zorder=3, label="training data")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(z0, color="gray", ls="--", lw=1)
    ax.set_xlabel("$z$"); ax.set_ylabel(r"$\hat p(1\mid z)$")
    ax.set_title("Gaussian Naive Bayes: a duplicated feature makes the model overconfident")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "nb_probabilities.png", bbox_inches="tight"); plt.close(fig)

    zc = z[np.abs(z) > 1e-9]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(zc, nb_log_odds(model_a, zc), lw=2, color=BLUE, label="Model A: $3z$")
    ax.plot(zc, nb_log_odds(model_b, zc), lw=2, color=ORANGE, label="Model B: $6z$")
    ax.set_xlabel("$z$"); ax.set_ylabel("log posterior odds")
    ax.set_title("The duplicated feature doubles the log odds")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "nb_log_odds.png", bbox_inches="tight"); plt.close(fig)


def lda_qda_experiment() -> None:
    lda_b, qda_b = fit_lda_qda(add_extra=False)
    lda_a, qda_a = fit_lda_qda(add_extra=True)
    models = {("LDA", "before"): lda_b, ("QDA", "before"): qda_b,
              ("LDA", "after"): lda_a, ("QDA", "after"): qda_a}

    rows, preds = [], {"x": X_VAL, "y": Y_VAL}
    for (name, when), m in models.items():
        v0, v1 = fitted_variances(m)
        mu0, mu1 = m.means_.ravel()
        rows.append({"model": name, "data": when, "mu0": mu0, "mu1": mu1,
                     "var0": v0, "var1": v1,
                     "boundaries": "; ".join(f"{b:.2f}" for b in decision_boundaries(m)),
                     "val_log_loss": validation_log_loss(m)})
        preds[f"{name} {when}"] = proba(m, X_VAL)
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / "lda_qda_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(preds).to_csv(RES / "lda_qda_validation_predictions.csv",
                               index=False, float_format="%.4f")
    print(f"\nLDA and QDA (scikit-learn {sklearn.__version__})\n",
          summary.round(4).to_string(index=False))

    styles = {("LDA", "before"): dict(color=BLUE, ls="-", lw=6, alpha=0.4),
              ("LDA", "after"): dict(color=BLUE, ls="--", lw=2),
              ("QDA", "before"): dict(color=RED, ls="-", lw=2),
              ("QDA", "after"): dict(color=RED, ls="--", lw=2)}
    grid = np.linspace(-6, 10, 3201)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, m in models.items():
        ax.plot(grid, proba(m, grid), label=f"{key[0]} {key[1]}", **styles[key])
    ax.scatter(X_CLASS0, np.zeros_like(X_CLASS0), c="k", marker="o", zorder=3, label="train class 0")
    ax.scatter(X_CLASS1, np.ones_like(X_CLASS1), c="k", marker="s", zorder=3, label="train class 1")
    ax.scatter([EXTRA_POINT[0]], [1], c="orange", marker="*", s=250, zorder=4,
               label=f"added point {EXTRA_POINT}")
    ax.scatter(X_VAL, Y_VAL, facecolors="none", edgecolors="green", s=110, lw=2, zorder=3,
               label="validation")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("$x$"); ax.set_ylabel(r"$\hat p(1\mid x)$")
    ax.set_title("LDA and QDA before and after adding one extreme point")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "lda_qda_curves.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    naive_bayes_experiment()
    lda_qda_experiment()
    print(f"\nFigures saved to {FIG}\nResults saved to {RES}")
