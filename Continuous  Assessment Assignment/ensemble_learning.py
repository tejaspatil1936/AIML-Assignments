"""Assignment 5 - Ensemble learning: Stacking, AdaBoost, Voting vs base classifiers."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
d = load_breast_cancer(); Xd, y = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.25, stratify=y, random_state=7)
print(f"{B}{C}Dataset: Breast Cancer Wisconsin  {Xd.shape}, train={len(Xtr)}, test={len(Xte)}{X}")


def base():
    return [("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
            ("dt", DecisionTreeClassifier(max_depth=4, random_state=0)),
            ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
            ("nb", GaussianNB())]


models = {f"Base: {n.upper()}": m for n, m in base()}
models["Voting (hard, majority)"] = VotingClassifier(base(), voting="hard")
models["Voting (soft)"] = VotingClassifier(base(), voting="soft")
models["Stacking (meta=LogReg)"] = StackingClassifier(base(), final_estimator=LogisticRegression(max_iter=1000), cv=5)
models["AdaBoost (stumps)"] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0)
models["AdaBoost (depth-3 trees)"] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=0)

cv = StratifiedKFold(5, shuffle=True, random_state=1)
print(f"\n{'Model':26s}{'Acc':>7s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}{'AUC':>7s}{'CV-Acc':>8s}{'CV-std':>8s}")
res = {}
for n, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) if hasattr(m, "predict_proba") and n != "Voting (hard, majority)" else float("nan")
    s = cross_val_score(m, Xd, y, cv=cv)
    res[n] = (accuracy_score(yte, p), precision_score(yte, p), recall_score(yte, p), f1_score(yte, p), auc, s.mean(), s.std())
    col = G if n.startswith(("Voting", "Stack", "Ada")) else Y
    print(f"{col}{n:26s}{X}" + "".join(f"{v:7.3f}" for v in res[n][:5]) + f"{res[n][5]:8.3f}{res[n][6]:8.3f}")

bb = max((k for k in res if k.startswith("Base")), key=lambda k: res[k][5])
be = max((k for k in res if not k.startswith("Base")), key=lambda k: res[k][5])
print(f"\n{B}Best base (CV): {bb} {res[bb][5]:.3f}   Best ensemble (CV): {be} {res[be][5]:.3f}{X}")
print(f"{G if res[be][5] > res[bb][5] else R}Ensemble gain over best base model = {res[be][5]-res[bb][5]:+.3f}{X}")

ada = models["AdaBoost (stumps)"]
stage = [accuracy_score(yte, p) for p in ada.staged_predict(Xte)]
print(f"\nAdaBoost stumps accuracy after 1/10/50/100 rounds: " + ", ".join(f"{stage[i-1]:.3f}" for i in (1, 10, 50, 100)))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
names = list(res); cols = ["#aaaaaa" if n.startswith("Base") else "#333333" for n in names]
ax[0].barh(names, [res[n][5] for n in names], xerr=[res[n][6] for n in names], color=cols)
ax[0].set_xlim(0.85, 1.0); ax[0].set_title("5-fold CV accuracy (light=base, dark=ensemble)"); ax[0].tick_params(labelsize=7)
ax[1].plot(range(1, 101), stage, color="#333333"); ax[1].set_title("AdaBoost (stumps): test accuracy vs rounds"); ax[1].set_xlabel("boosting rounds")
plt.tight_layout(); plt.savefig("Assignment_5/ensemble_results.png", dpi=140)
print(f"{G}Saved plot: Assignment_5/ensemble_results.png{X}")
