"""Assignment 4 - Classification: SVM, Decision Tree, KNN, Naive Bayes on 2 datasets."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"


def build():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, probability=True, random_state=0)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Naive Bayes": GaussianNB(),
    }


all_res, cms = {}, {}
for dname, loader in [("Breast Cancer (binary)", load_breast_cancer), ("Wine (3-class)", load_wine)]:
    d = loader(); Xd, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.25, stratify=y, random_state=42)
    print(f"\n{B}{C}=== {dname}: {Xd.shape[0]} samples, {Xd.shape[1]} features, classes={len(set(y))} ==={X}")
    print(f"{'Model':16s}{'Acc':>7s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}{'LogLoss':>9s}{'CV-Acc':>8s}{'Train-Acc':>10s}")
    for n, m in build().items():
        m.fit(Xtr, ytr); p = m.predict(Xte); pr = m.predict_proba(Xte)
        a = accuracy_score(yte, p); avg = "binary" if len(set(y)) == 2 else "macro"
        P, Rc, F = (f(yte, p, average=avg) for f in (precision_score, recall_score, f1_score))
        ll = log_loss(yte, np.clip(pr, 1e-15, 1)); cv = cross_val_score(build()[n], Xd, y, cv=5).mean()
        tr = accuracy_score(ytr, m.predict(Xtr))
        print(f"{n:16s}{a:7.3f}{P:7.3f}{Rc:7.3f}{F:7.3f}{ll:9.3f}{cv:8.3f}{tr:10.3f}")
        all_res[(dname, n)] = (a, P, Rc, F, ll); cms[(dname, n)] = confusion_matrix(yte, p)
    bestm = max((k for k in all_res if k[0] == dname), key=lambda k: all_res[k][3])
    print(f"{G}Best F1: {bestm[1]} ({all_res[bestm][3]:.3f}){X}")

# Hyper-parameter tuning demo on breast cancer
d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, stratify=d.target, random_state=42)
print(f"\n{B}{C}=== Hyper-parameter tuning (GridSearchCV, 5-fold) on Breast Cancer ==={X}")
grids = {"SVM": (make_pipeline(StandardScaler(), SVC()), {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]}),
         "Decision Tree": (DecisionTreeClassifier(random_state=0), {"max_depth": [2, 3, 4, 6, None]}),
         "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 11]})}
for n, (m, g) in grids.items():
    gs = GridSearchCV(m, g, cv=5).fit(Xtr, ytr)
    print(f"{n:14s} best={Y}{gs.best_params_}{X}  cv={gs.best_score_:.3f}  test acc={G}{gs.score(Xte, yte):.3f}{X}")

# plots
models = list(build())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for i, dn in enumerate(["Breast Cancer (binary)", "Wine (3-class)"]):
    f1 = [all_res[(dn, m)][3] for m in models]; ll = [all_res[(dn, m)][4] for m in models]
    x = np.arange(4); ax[i].bar(x - .2, f1, .4, color="#333333", label="F1"); ax[i].bar(x + .2, ll, .4, color="#aaaaaa", label="Log-loss")
    ax[i].set_xticks(x); ax[i].set_xticklabels(models, fontsize=7); ax[i].set_title(dn); ax[i].legend()
cm = cms[("Breast Cancer (binary)", "SVM (RBF)")]; ax[2].imshow(cm, cmap="Greys"); ax[2].set_title("SVM confusion matrix (Breast Cancer)")
for (i, j), v in np.ndenumerate(cm): ax[2].text(j, i, v, ha="center", va="center", color="red")
plt.tight_layout(); plt.savefig("Assignment_4/classification_results.png", dpi=140)
print(f"{G}Saved plot: Assignment_4/classification_results.png{X}")
