import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ols import fit_ols, predict, r2_score, residuals


def main():
    # Load data (adjust path if you use data/dogs.csv)
    dogs_raw = pd.read_csv("data/dogs.csv").drop(columns=["Unnamed: 0"], errors="ignore")

    # ------------------------------------------------------------
    # Model #5: loss ~ weight + breed*supplement
    # Dummy coding with drop_first=True:
    # base breed = Collie, base supplement = A (typical alphabetical base)
    # Add interaction dummies:
    #   LB = breed_Labrador * supplement_B
    #   SB = breed_Spaniel  * supplement_B
    # ------------------------------------------------------------
    dogs = pd.get_dummies(dogs_raw, columns=["breed", "supplement"], drop_first=True).astype(float)

    # Create interaction dummy variables (as in your original)
    dogs["LB"] = dogs.get("breed_Labrador", 0.0) * dogs.get("supplement_B", 0.0)
    dogs["SB"] = dogs.get("breed_Spaniel", 0.0) * dogs.get("supplement_B", 0.0)

    # Build X and y
    y = dogs["loss"].to_numpy().reshape(-1, 1)

    X_no_intercept = dogs.drop(columns=["loss"]).to_numpy()
    X = np.hstack([np.ones((len(dogs), 1)), X_no_intercept])

    # Fit OLS using your core function
    beta = fit_ols(X, y)
    y_hat = predict(X, beta)

    # Metrics
    r2 = r2_score(y, y_hat)
    res = residuals(y, y_hat)

    print("Model 5 (Dogs): loss ~ weight + breed*supplement")
    print("R²:", r2)

    # ------------------------------------------------------------
    # Recreate the 6 fitted lines plot: CA, LA, SA, CB, LB, SB
    # Beta order (after intercept):
    #   weight, breed_Labrador, breed_Spaniel, supplement_B, LB, SB
    # This matches your original matrix interpretation. :contentReference[oaicite:1]{index=1}
    # ------------------------------------------------------------
    cols = dogs.drop(columns=["loss"]).columns.tolist()
    idx_weight = cols.index("weight")

    idx_breed_lab = cols.index("breed_Labrador") if "breed_Labrador" in cols else None
    idx_breed_span = cols.index("breed_Spaniel") if "breed_Spaniel" in cols else None
    idx_sup_B = cols.index("supplement_B") if "supplement_B" in cols else None
    idx_LB = cols.index("LB") if "LB" in cols else None
    idx_SB = cols.index("SB") if "SB" in cols else None

    def b(idx):
        return beta[1 + idx, 0] if idx is not None else 0.0

    b0 = beta[0, 0]
    b_w = b(idx_weight)
    b_lab = b(idx_breed_lab)
    b_span = b(idx_breed_span)
    b_supB = b(idx_sup_B)
    b_LB = b(idx_LB)
    b_SB = b(idx_SB)

    w = dogs_raw["weight"].to_numpy()
    w_grid = np.linspace(w.min(), w.max(), 100)

    # Supplement A
    y_CA = b0 + b_w * w_grid
    y_LA = b0 + b_w * w_grid + b_lab
    y_SA = b0 + b_w * w_grid + b_span

    # Supplement B (includes breed*supplement interaction for L and S)
    y_CB = b0 + b_w * w_grid + b_supB
    y_LB = b0 + b_w * w_grid + b_lab + b_supB + b_LB
    y_SB = b0 + b_w * w_grid + b_span + b_supB + b_SB

    plt.figure()
    plt.plot(w_grid, y_CA, "r-", label="CA predicted")
    plt.plot(w_grid, y_LA, "b-", label="LA predicted")
    plt.plot(w_grid, y_SA, "g-", label="SA predicted")
    plt.plot(w_grid, y_CB, "k-", label="CB predicted")
    plt.plot(w_grid, y_LB, "m-", label="LB predicted")
    plt.plot(w_grid, y_SB, "y-", label="SB predicted")

    # Observed points + labels
    for i in range(len(dogs_raw)):
        breed = dogs_raw.loc[i, "breed"]
        sup = dogs_raw.loc[i, "supplement"]
        wi = dogs_raw.loc[i, "weight"]
        li = dogs_raw.loc[i, "loss"]

        tag = breed[0] + sup  # CA, LA, SA, CB, LB, SB
        if sup == "A":
            c = {"C": "r", "L": "b", "S": "g"}.get(breed[0], "k")
        else:
            c = {"C": "k", "L": "m", "S": "y"}.get(breed[0], "k")

        plt.plot(wi, li, marker=".", color=c, linestyle="None")
        plt.text(wi + 0.1, li + 0.1, tag)

    plt.grid()
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.title("Model 5: fitted model for weight vs loss (breed*supplement)")
    plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))
    plt.savefig("figures/model5_weight_vs_loss_lines.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Residuals vs fitted
    plt.figure()
    plt.plot(y_hat.ravel(), res.ravel(), "b.")
    plt.title("Model 5: Residuals vs fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.axhline(0, color="g", linestyle="--")
    plt.grid()
    plt.savefig("figures/model5_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
