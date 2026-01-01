import numpy as np
import matplotlib.pyplot as plt

from src.design_matrices import X_simple
from src.ols import fit_ols, predict, r2_score, residuals


def main():
    # Six points: (0,1), (3,3), (6,2), (1,2), (4,6), (7,8)
    x = np.array([0, 3, 6, 1, 4, 7])
    y = np.array([1, 3, 2, 2, 6, 8])

    # Build design matrix and fit OLS
    X = X_simple(x)
    beta = fit_ols(X, y)
    y_hat = predict(X, beta)

    # Metrics
    r2 = r2_score(y, y_hat)
    res = residuals(y, y_hat)

    print("Model 1 (Simple OLS)")
    print("Beta (intercept, slope):\n", beta)
    print("R²:", r2)

    # Plot fitted line (extended range)
    x_ext = np.arange(-1, 9)
    X_ext = X_simple(x_ext)
    y_ext = predict(X_ext, beta)

    plt.figure()
    plt.plot(x_ext, y_ext, "k-")
    plt.scatter(x, y, s=50)
    plt.xlim(-1, 8)
    plt.ylim(0, 9)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model 1: Simple Linear Regression (Matrix OLS)")
    
    plt.grid()
    plt.savefig("figures/model1_fit.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


    # Residuals vs fitted
    plt.figure()
    plt.plot(y_hat.ravel(), res.ravel(), "b.")
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.grid()
    plt.savefig("figures/model1_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
