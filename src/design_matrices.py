"""
design_matrices.py

Design matrix construction utilities for linear regression models
implemented using matrix operations (NumPy only).

This module provides simple helper functions to construct design matrices
with:
- an intercept term,
- optional dummy (indicator) variables,
- optional interaction terms.

The functions are intentionally minimal and closely follow the
mathematical formulation of regression models used in the coursework.
"""

import numpy as np


def X_simple(x):
    """
    Construct the design matrix for simple linear regression.

    Model form:
        y = β₀ + β₁ x

    Design matrix:
        X = [1, x]

    Parameters
    ----------
    x : array-like
        Predictor variable values.

    Returns
    -------
    numpy.ndarray
        Design matrix of shape (n_samples, 2), where the first column
        is the intercept term and the second column is x.
    """
    x = np.asarray(x).reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x])


def X_dummy(x, g):
    """
    Construct the design matrix for linear regression with a dummy variable.

    Model form:
        y = β₀ + β₁ x + β₂ g

    where g is a binary indicator variable (e.g., group membership).

    Design matrix:
        X = [1, x, g]

    Parameters
    ----------
    x : array-like
        Predictor variable values.
    g : array-like
        Dummy variable (typically 0 or 1).

    Returns
    -------
    numpy.ndarray
        Design matrix of shape (n_samples, 3), containing the intercept,
        predictor x, and dummy variable g.
    """
    x = np.asarray(x).reshape(-1, 1)
    g = np.asarray(g).reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x, g])


def X_interaction(x, g):
    """
    Construct the design matrix for linear regression with a dummy variable
    and an interaction term.

    Model form:
        y = β₀ + β₁ x + β₂ g + β₃ (x · g)

    Design matrix:
        X = [1, x, g, x*g]

    Parameters
    ----------
    x : array-like
        Predictor variable values.
    g : array-like
        Dummy variable (typically 0 or 1).

    Returns
    -------
    numpy.ndarray
        Design matrix of shape (n_samples, 4), containing the intercept,
        predictor x, dummy variable g, and interaction term x*g.
    """
    x = np.asarray(x).reshape(-1, 1)
    g = np.asarray(g).reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x, g, x * g])