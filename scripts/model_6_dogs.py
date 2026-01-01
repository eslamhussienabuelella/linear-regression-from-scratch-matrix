import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ols import fit_ols, predict, r2_score, residuals


def main():
    # Load data (change to "data/dogs.csv" if that’s your structure)
    dogs_raw = pd.read_csv("data/dogs.csv").drop(columns=["Unnamed: 0"], errors="ignore")

    # ------------------------------------------------------------
    # Model #6: loss ~ breed + weight*supplement
    # Create dummies with drop_first=True then create WB = weight * supplement_B
    # ------------------------------------------------------------
    dogs = pd.get_dummies(dogs_raw, columns=["breed", "supplement"], drop_first=True).astype(float)
    dogs["WB"] = dogs["weight"] * dogs.get("supplement_B", 0.0)

    # Build X and y
    y = dogs["loss"].to_numpy().reshape(-1, 1)
    X_no_intercept = dogs.drop(columns=["loss"]).to_numpy()
    X = np.hstack([np.ones((len(dogs), 1)), X_no_intercept])

    # Fit OLS (matrix solution from your src module)
    beta = fit_ols(X, y)
    y_hat = predict(X, beta)

    # Metrics
    r2 = r2_score(y, y_hat)
    res = residuals(y, y_hat)

    print("Model 6 (Dogs): loss ~ breed + weight*supplement")
    print("R²:", r2)

    # ----- coefficient lookup (robust to column ordering) -----
    cols = dogs.drop(columns=["loss"]).columns.tolist()

    def idx(name):
        return cols.index(name) if name in cols else None

    def b(i):
        return beta[1 + i, 0] if i is not None else 0.0

    b0 = beta[0, 0]
    bw = b(idx("weight"))

    # breed dummies (base breed is the dropped category)
    b_lab = b(idx("breed_Labrador"))
    b_span = b(idx("breed_Spaniel"))

    # supplement dummy (base supplement is the dropped category, usually A)
    b_supB = b(idx("supplement_B"))

    # interaction term WB = weight * supplement_B
    b_WB = b(idx("WB"))

    # ------------------------------------------------------------
    # Correct fitted lines (use a sorted weight grid)
    #
    # Supplement A (supplement_B = 0) => WB = 0
    #   CA: b0 + bw*w
    #   LA: b0 + bw*w + b_lab
    #   SA: b0 + bw*w + b_span
    #
    # Supplement B (supplement_B = 1) => WB = w
    #   CB: b0 + b_supB + (bw + b_WB)*w
    #   LB: b0 + b_supB + b_lab  + (bw + b_WB)*w
    #   SB: b0 + b_supB + b_span + (bw + b_WB)*w
    # ------------------------------------------------------------
    w_min, w_max = dogs_raw["weight"].min(), dogs_raw["weight"].max()
    w_grid = np.linspace(w_min, w_max, 200)

    # A lines
    y_CA = b0 + bw * w_grid
    y_LA = b0 + bw * w_grid + b_lab
    y_SA = b0 + bw * w_grid + b_span

    # B lines (slope changes)
    slope_B = (bw + b_WB)
    y_CB = b0 + b_supB + slope_B * w_grid
    y_LB = b0 + b_supB + b_lab + slope_B * w_grid
    y_SB = b0 + b_supB + b_span + slope_B * w_grid

    # ----- Plot: fitted lines + observed points -----
    plt.figure()
    plt.plot(w_grid, y_CA, "r-", label="CA predicted")
    plt.plot(w_grid, y_LA, "b-", label="LA predicted")
    plt.plot(w_grid, y_SA, "g-", label="SA predicted")
    plt.plot(w_grid, y_CB, "k-", label="CB predicted")
    plt.plot(w_grid, y_LB, "m-", label="LB predicted")
    plt.plot(w_grid, y_SB, "y-", label="SB predicted")

    # observed points
    #plt.scatter(dogs_raw["weight"], dogs_raw["loss"], s=30)

    # point labels (CA/LA/SA/CB/LB/SB)
    for i in range(len(dogs_raw)):
        tag = dogs_raw.loc[i, "breed"][0] + dogs_raw.loc[i, "supplement"]
        plt.text(dogs_raw.loc[i, "weight"] + 0.1,
                 dogs_raw.loc[i, "loss"] + 0.1,
                 tag)
        if tag == "CA":
            plt.plot(dogs_raw.loc[i, "weight"], dogs_raw.loc[i, "loss"], "rs")
        elif tag == "SA":
           plt.plot(dogs_raw.loc[i, "weight"], dogs_raw.loc[i, "loss"], "gs")
        elif tag == "LA":
           plt.plot(dogs_raw.loc[i, "weight"], dogs_raw.loc[i, "loss"], "bs")
        elif tag == "CB":
            plt.plot(dogs_raw.loc[i, "weight"], dogs_raw.loc[i, "loss"], "ks")
        elif tag == "LB":
            plt.plot(dogs_raw.loc[i, "weight"], dogs_raw.loc[i, "loss"], "ms")
        else:
            plt.plot(dogs_raw.loc[i, "weight"], dogs_raw.loc[i, "loss"], "ys")

    plt.grid()
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.title("Model 6: fitted model for weight vs loss (weight*supplement)")
    plt.legend(loc="lower right", bbox_to_anchor=(1.35, 0.1))
    plt.savefig("figures/model6_weight_vs_loss_lines.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # ----- Plot: residuals vs fitted -----
    plt.figure()
    plt.plot(y_hat.ravel(), res.ravel(), "b.")
    plt.axhline(0, color="g", linestyle="--")
    plt.title("Model 6: Residuals vs fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.grid()
    plt.savefig("figures/model6_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
