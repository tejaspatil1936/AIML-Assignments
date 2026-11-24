"""Assignment 3 - Regression models (Diabetes progression dataset) + Gradient Descent from scratch."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
data = load_diabetes(as_frame=True)
Xd, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.2, random_state=42)
print(f"{B}{C}Dataset: Diabetes progression  samples={len(Xd)} features={Xd.shape[1]} target=disease progression after 1 yr{X}")

# ---------------- Part 1: model comparison ----------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (a=1)": Ridge(alpha=1.0),
    "Lasso (a=0.1)": Lasso(alpha=0.1),
    "Decision Tree (d=4)": DecisionTreeRegressor(max_depth=4, random_state=0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0),
}
print(f"\n{B}{C}=== Part 1: Regression models, 80/20 split + 5-fold CV ==={X}")
print(f"{'Model':22s}{'MAE':>8s}{'MSE':>10s}{'RMSE':>8s}{'R2':>8s}{'CV R2':>8s}")
rows, preds = {}, {}
for n, m in models.items():
    m = make_pipeline(StandardScaler(), m); m.fit(Xtr, ytr); p = m.predict(Xte)
    mae, mse, r2 = mean_absolute_error(yte, p), mean_squared_error(yte, p), r2_score(yte, p)
    cv = cross_val_score(make_pipeline(StandardScaler(), models[n]), Xd, y, cv=5, scoring="r2").mean()
    rows[n] = r2; preds[n] = p
    print(f"{n:22s}{mae:8.2f}{mse:10.1f}{np.sqrt(mse):8.2f}{r2:8.3f}{cv:8.3f}")
best = max(rows, key=rows.get)
print(f"{G}Best test R2: {best} ({rows[best]:.3f}){X}")

# ---------------- Part 2: Gradient descent from scratch ----------------
sc = StandardScaler().fit(Xtr)
A = np.c_[np.ones(len(Xtr)), sc.transform(Xtr)]; At = np.c_[np.ones(len(Xte)), sc.transform(Xte)]
yt = ytr.values

def gd(A, y, lr, epochs, batch=None, seed=0):
    rng = np.random.default_rng(seed); w = np.zeros(A.shape[1]); hist = []
    n = len(y)
    for _ in range(epochs):
        if batch is None:
            w -= lr * (2 / n) * A.T @ (A @ w - y)
        else:
            idx = rng.permutation(n)
            for i in range(0, n, batch):
                b = idx[i:i + batch]; w -= lr * (2 / len(b)) * A[b].T @ (A[b] @ w - y[b])
        hist.append(np.mean((A @ w - y) ** 2))
    return w, hist

w_cf = np.linalg.lstsq(A, yt, rcond=None)[0]
mse_cf = np.mean((A @ w_cf - yt) ** 2)
print(f"\n{B}{C}=== Part 2: Gradient Descent for Linear Regression (standardised features) ==={X}")
print(f"Closed-form (normal eq.) train MSE = {mse_cf:.2f}, test R2 = {r2_score(yte, At @ w_cf):.4f}")
print(f"\n{'Variant':26s}{'lr':>7s}{'epochs':>8s}{'final MSE':>11s}{'|w-w*|':>9s}{'test R2':>9s}")
curves = {}
for name, lr, ep, bs in [("Batch GD", 0.001, 300, None), ("Batch GD", 0.01, 300, None), ("Batch GD", 0.1, 300, None),
                         ("Batch GD", 0.5, 50, None), ("Batch GD", 1.05, 30, None),
                         ("Mini-batch GD (32)", 0.01, 300, 32), ("Stochastic GD (1)", 0.001, 100, 1)]:
    with np.errstate(all="ignore"):
        w, h = gd(A, yt, lr, ep, bs)
    ok = np.isfinite(h[-1]) and h[-1] < 1e7
    r2 = r2_score(yte, At @ w) if ok else float("nan")
    tag = f"{name}"
    if ok:
        print(f"{tag:26s}{lr:7.3f}{ep:8d}{G}{h[-1]:11.1f}{X}{np.linalg.norm(w - w_cf):9.2f}{r2:9.4f}")
    else:
        print(f"{tag:26s}{lr:7.3f}{ep:8d}{R}{'>1e7':>11s}{'inf':>9s}{'nan':>9s}  DIVERGED{X}")
    curves[f"{name} lr={lr}"] = h
print(f"{Y}Observation:{X} small lr converges slowly, lr 0.01-0.1 reaches the closed-form MSE, lr>=0.5 diverges (step too large).")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for k, h in curves.items():
    if np.isfinite(h[-1]) and h[-1] < 1e7: ax[0].plot(h, label=k)
ax[0].axhline(mse_cf, color="k", ls="--", lw=1, label="closed form"); ax[0].set_yscale("log")
ax[0].set_title("GD convergence (train MSE)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=6)
ax[1].barh(list(rows), list(rows.values()), color="#555555"); ax[1].set_title("Test R2 by model")
p = preds[best]; ax[2].scatter(yte, p, s=14, color="#333333"); ax[2].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "k--")
ax[2].set_title(f"Actual vs predicted ({best})"); ax[2].set_xlabel("actual"); ax[2].set_ylabel("predicted")
plt.tight_layout(); plt.savefig("Assignment_3/regression_results.png", dpi=140)
print(f"{G}Saved plot: Assignment_3/regression_results.png{X}")
