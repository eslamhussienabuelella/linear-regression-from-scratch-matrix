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
