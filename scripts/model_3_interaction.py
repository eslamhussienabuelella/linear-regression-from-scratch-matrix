import numpy as np
import matplotlib.pyplot as plt

from src.design_matrices import X_interaction
from src.ols import fit_ols, predict, r2_score, residuals


def main():
    # Six points: (0,1), (3,3), (6,2), (1,2), (4,6), (7,8)
    x = np.array([0, 3, 6, 1, 4, 7])
    y = np.array([1, 3, 2, 2, 6, 8])

    # Group indicator:
    # First 3 points = group 1 (g=1)
    # Last 3 points  = group 0 (g=0)
    g = np.array([1, 1, 1, 0, 0, 0])

    # Build design matrix with interaction and fit OLS
    X = X_interaction(x, g)
    beta = fit_ols(X, y)
    y_hat = predict(X, beta)

    # Metrics
    r2 = r2_score(y, y_hat)
    res = residuals(y, y_hat)

    print("Model 3 (Interaction: Different Slopes)")
    print("Beta (intercept, slope, dummy, interaction):\n", beta)
    print("R²:", r2)

    # Extended x-range for plotting fitted lines
    x_ext = np.arange(-1, 9)

    # Predict for group 1 and group 0
    X_ext_g1 = X_interaction(x_ext, np.ones_like(x_ext))
    X_ext_g0 = X_interaction(x_ext, np.zeros_like(x_ext))

    y_ext_g1 = predict(X_ext_g1, beta)
    y_ext_g0 = predict(X_ext_g0, beta)

    # --- Plot fitted lines + data ---
    plt.figure()
    plt.plot(x_ext, y_ext_g1, "g--", label="Group 1 (g=1)")
    plt.plot(x_ext, y_ext_g0, "r--", label="Group 0 (g=0)")

    plt.plot(x[g == 1], y[g == 1], "g.", markersize=10)
    plt.plot(x[g == 0], y[g == 0], "r.", markersize=10)

    plt.xlim(-1, 8)
    plt.ylim(0, 9)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model 3: Dummy + Interaction (Different Slopes)")
    plt.grid()
    plt.legend()

    plt.savefig("figures/model3_fit_interaction_slopes.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # --- Residuals vs fitted ---
    plt.figure()
    plt.plot(y_hat.ravel(), res.ravel(), "b.")
    plt.axhline(0, linestyle="--")
    plt.title("Model 3: Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.grid()

    plt.savefig("figures/model3_residuals_vs_fitted.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
