import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ols import fit_ols, predict, r2_score, residuals


def main():
    # Load data
    dogs_raw = pd.read_csv("data/dogs.csv").drop(columns=["Unnamed: 0"], errors="ignore")

    # Dummy variables (drop_first avoids dummy variable trap)
    dogs = pd.get_dummies(dogs_raw, columns=["breed", "supplement"], drop_first=True).astype(float)

    # Build X and y
    y = dogs["loss"].to_numpy().reshape(-1, 1)
    X_no_intercept = dogs.drop(columns=["loss"]).to_numpy()
    X = np.hstack([np.ones((len(dogs), 1)), X_no_intercept])

    # Fit OLS
    beta = fit_ols(X, y)
    y_hat = predict(X, beta)

    # Metrics
    r2 = r2_score(y, y_hat)
    res = residuals(y, y_hat)

    print("Model 4 (Dogs): loss ~ weight + breed + supplement")
    print("Beta:\n", beta)
    print("R²:", r2)

    # ------------------------------------------------------------
    # Recreate your original 6 fitted lines plot (Weight vs Loss)
    # Model interpretation with drop_first=True:
    # Base categories are the alphabetically-first ones in each column.
    # In your original plot you treated:
    # - Breed base = Collie (C)
    # - Supplement base = A
    # That matches typical dummy coding for (Collie, Labrador, Spaniel) and (A, B).
    # ------------------------------------------------------------

    # Identify which dummy columns exist and their positions
    # Expected order after get_dummies:
    # columns of X_no_intercept correspond to: weight, breed_..., breed_..., supplement_...
    cols = dogs.drop(columns=["loss"]).columns.tolist()

    # Find indices (robust to exact dummy column names)
    idx_weight = cols.index("weight")

    # breed dummies might be: 'breed_Labrador', 'breed_Spaniel' (base Collie)
    idx_breed_lab = cols.index("breed_Labrador") if "breed_Labrador" in cols else None
    idx_breed_span = cols.index("breed_Spaniel") if "breed_Spaniel" in cols else None

    # supplement dummy likely: 'supplement_B' (base A)
    idx_sup_B = cols.index("supplement_B") if "supplement_B" in cols else None

    # Weight values for drawing lines (sorted)
    w = dogs_raw["weight"].to_numpy()
    w_grid = np.linspace(w.min(), w.max(), 100)

    # Helper: compute line y = β0 + βw*w + βbreed + βsup
    b0 = beta[0, 0]
    b_w = beta[1 + idx_weight, 0]  # +1 because intercept is beta[0]

    def coef_at(idx):
        return beta[1 + idx, 0] if idx is not None else 0.0

    b_lab = coef_at(idx_breed_lab)
    b_span = coef_at(idx_breed_span)
    b_supB = coef_at(idx_sup_B)

    # Lines (same equations as your original approach) :contentReference[oaicite:2]{index=2}
    # Supplement A
    y_CA = b0 + b_w * w_grid
    y_LA = b0 + b_w * w_grid + b_lab
    y_SA = b0 + b_w * w_grid + b_span

    # Supplement B
    y_CB = b0 + b_w * w_grid + b_supB
    y_LB = b0 + b_w * w_grid + b_lab + b_supB
    y_SB = b0 + b_w * w_grid + b_span + b_supB

    # Plot fitted lines + observed points (Weight vs Loss)
    plt.figure()
    plt.plot(w_grid, y_CA, "r-", label="CA predicted")
    plt.plot(w_grid, y_LA, "b-", label="LA predicted")
    plt.plot(w_grid, y_SA, "g-", label="SA predicted")
    plt.plot(w_grid, y_CB, "k-", label="CB predicted")
    plt.plot(w_grid, y_LB, "m-", label="LB predicted")
    plt.plot(w_grid, y_SB, "y-", label="SB predicted")

    # Plot observed points with labels like your original
    # Determine observed category from original (not dummy) columns
    for i in range(len(dogs_raw)):
        breed = dogs_raw.loc[i, "breed"]
        sup = dogs_raw.loc[i, "supplement"]
        wi = dogs_raw.loc[i, "weight"]
        li = dogs_raw.loc[i, "loss"]

        tag = breed[0] + sup  # e.g., "LA", "SB"
        # color per group (roughly matching your original)
        if sup == "A":
            c = {"C": "r", "L": "b", "S": "g"}.get(breed[0], "k")
        else:
            c = {"C": "k", "L": "m", "S": "y"}.get(breed[0], "k")

        plt.plot(wi, li, marker=".", color=c, linestyle="None")
        plt.text(wi + 0.1, li + 0.1, tag)

    plt.grid()
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.title("Model 4: fitted model for weight vs loss")
    plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))
    plt.savefig("figures/model4_weight_vs_loss_lines.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Residuals vs fitted
    plt.figure()
    plt.plot(y_hat.ravel(), res.ravel(), "b.")
    plt.title("Model 4: Residuals vs fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.axhline(0, color="g", linestyle="--")
    plt.grid()
    plt.savefig("figures/model4_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
