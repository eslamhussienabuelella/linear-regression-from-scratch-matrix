# Linear Regression from Scratch (Matrix-Based)

This project implements **Ordinary Least Squares (OLS) linear regression from scratch using matrix operations**, without relying on machine-learning libraries such as `scikit-learn`.

The aim is to demonstrate a **clear understanding of statistical modelling**, linear algebra, and how **dummy variables and interaction terms** affect regression interpretation.

---

## 📌 Project Objectives

- Implement the **closed-form OLS solution** using NumPy
- Construct **design matrices manually** (intercept, dummy variables, interactions)
- Demonstrate how:
  - dummy variables shift intercepts
  - interaction terms change slopes
- Apply models to both **toy data** and a **real dataset**
- Produce **interpretable visualisations** (fitted lines and residual diagnostics)

---

## 📂 Project Structure

```text
linear-regression-from-scratch-matrix/
├── src/
│   ├── ols.py                 # Core OLS matrix implementation
│   └── design_matrices.py     # Design matrix builders
│
├── scripts/
│   ├── model_1_simple.py
│   ├── model_2_dummy.py
│   ├── model_3_interaction.py
│   ├── model_4_dogs.py
│   ├── model_5_dogs.py
│   └── model_6_dogs.py
│
├── data/
│   └── dogs.csv               # Real dataset
│
├── figures/                   # Generated plots
│
├── requirements.txt
├── LICENSE
└── README.md
```

## 🧠 Core Implementation

This project implements **Ordinary Least Squares (OLS) regression using explicit matrix algebra**, rather than high-level machine learning libraries.

The goal is to make the mathematical foundations of regression fully transparent.

---


## Ordinary Least Squares via Matrix Algebra
Given a design matrix $X \in \mathbb{R}^{n \times p}$ and a response vector $y \in \mathbb{R}^{n}$, the OLS estimator is computed using: $\beta = (X^T X)^{-1} X^T y$

`β = (Xᵀ X)⁻¹ Xᵀ y`



This closed-form solution is implemented directly in `src/ols.py` using NumPy:

- `fit_ols(X, y)`  
  Computes regression coefficients using the normal equation.

- `predict(X, beta)`  
  Generates predictions via matrix multiplication.

- `r2_score(y, y_hat)`  
  Computes the coefficient of determination.

- `residuals(y, y_hat)`  
  Returns model residuals for diagnostic analysis.

No optimisation routines or ML abstractions are used—only linear algebra.

---

## Design Matrix Construction

Design matrices are constructed explicitly in `src/design_matrices.py` to reflect the underlying statistical model:

- **Simple regression**:  
  `y = β₀ + β₁ x`

- **Dummy-variable regression**:  
  `y = β₀ + β₁ x + β₂ g`

- **Interaction regression**:  
  `y = β₀ + β₁ x + β₂ g + β₃ (x · g)`

  
By building the design matrices manually, the effect of:
- intercept shifts (dummy variables)
- slope changes (interaction terms)

becomes directly interpretable through the estimated coefficients.

---

### Separation of Concerns

The project is structured to mirror professional analytical workflows:

- **`src/`**  
  Contains reusable, model-agnostic linear algebra utilities.

- **`scripts/`**  
  Contains experimental scripts that define specific statistical models, generate figures, and evaluate results.


## License and Usage
This repository is **view-only**. Reuse, modification, redistribution, or
commercial use is **not permitted**. Academic citation is required if referenced.







