"""
ols.py

Core Ordinary Least Squares (OLS) regression utilities implemented
using explicit matrix operations with NumPy.

This module provides minimal helper functions to:
- estimate regression coefficients using the closed-form OLS solution,
- generate predictions,
- compute residuals,
- evaluate model fit using the coefficient of determination (R²).

No machine learning libraries are used; all computations rely directly
on linear algebra.
"""

import numpy as np


def fit_ols(X, y):
    """
    Estimate OLS regression coefficients using the normal equation.

    The OLS estimator is computed as:
        β = (XᵀX)⁻¹ Xᵀy

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n_samples, n_features).
    y : array-like
        Response variable of length n_samples.

    Returns
    -------
    numpy.ndarray
        Estimated coefficient vector β of shape (n_features, 1).
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta


def predict(X, beta):
    """
    Generate predictions from a fitted OLS model.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n_samples, n_features).
    beta : array-like
        Estimated coefficient vector.

    Returns
    -------
    numpy.ndarray
        Predicted values ŷ of shape (n_samples, 1).
    """
    X = np.asarray(X)
    beta = np.asarray(beta)
    return X @ beta


def r2_score(y, y_hat):
    """
    Compute the coefficient of determination (R²).

    R² is defined as:
        R² = 1 − (SS_res / SS_tot)

    where SS_res is the residual sum of squares and SS_tot
    is the total sum of squares.

    Parameters
    ----------
    y : array-like
        Observed response values.
    y_hat : array-like
        Predicted response values.

    Returns
    -------
    float
        R² value measuring goodness of fit.
    """
    y = np.asarray(y).reshape(-1, 1)
    y_hat = np.asarray(y_hat).reshape(-1, 1)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def residuals(y, y_hat):
    """
    Compute regression residuals.

    Residuals are defined as:
        e = y − ŷ

    Parameters
    ----------
    y : array-like
        Observed response values.
    y_hat : array-like
        Predicted response values.

    Returns
    -------
    numpy.ndarray
        Residual vector of shape (n_samples, 1).
    """
    y = np.asarray(y).reshape(-1, 1)
    y_hat = np.asarray(y_hat).reshape(-1, 1)
    return y - y_hat
