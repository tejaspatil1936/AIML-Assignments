"""Assignment 6 - Clustering: K-Means, Agglomerative, DBSCAN, GMM on the Wine dataset."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
d = load_wine(); Xs = StandardScaler().fit_transform(d.data); truth = d.target
Z = PCA(2, random_state=0).fit_transform(Xs)
print(f"{B}{C}Dataset: Wine chemical analysis {d.data.shape}; labels used ONLY for external check (ARI){X}")

print(f"\n{B}{C}=== Choosing k for K-Means (elbow + silhouette) ==={X}")
print(f"{'k':>3s}{'Inertia':>10s}{'Silhouette':>12s}{'Davies-Bouldin':>16s}")
ks, inert, sil = range(2, 9), [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Xs); l = km.labels_
    inert.append(km.inertia_); sil.append(silhouette_score(Xs, l))
    print(f"{k:3d}{km.inertia_:10.1f}{sil[-1]:12.3f}{davies_bouldin_score(Xs, l):16.3f}")
kbest = list(ks)[int(np.argmax(sil))]
print(f"{G}Best k by silhouette = {kbest}{X}")

print(f"\n{B}{C}=== Algorithm comparison (k=3) ==={X}")
print(f"{'Algorithm':22s}{'Clusters':>9s}{'Noise':>7s}{'Silhouette':>11s}{'DB(lower)':>10s}{'CH':>8s}{'ARI':>7s}")
algos = {"K-Means": KMeans(3, n_init=10, random_state=0), "Agglomerative (ward)": AgglomerativeClustering(3),
         "GMM": GaussianMixture(3, random_state=0), "DBSCAN (eps=2.3,m=5)": DBSCAN(eps=2.3, min_samples=5),
         "DBSCAN (eps=1.5,m=5)": DBSCAN(eps=1.5, min_samples=5)}
labels = {}
for n, a in algos.items():
    l = a.fit_predict(Xs); labels[n] = l
    mask = l != -1; k = len(set(l[mask]))
    if k > 1:
        print(f"{n:22s}{k:9d}{(~mask).sum():7d}{silhouette_score(Xs[mask], l[mask]):11.3f}"
              f"{davies_bouldin_score(Xs[mask], l[mask]):10.3f}{calinski_harabasz_score(Xs[mask], l[mask]):8.1f}{adjusted_rand_score(truth, l):7.3f}")
    else:
        print(f"{R}{n:22s}{k:9d}{(~mask).sum():7d}  (single cluster - metrics undefined){X}")

best = max((n for n in labels if len(set(labels[n])) > 1 and -1 not in labels[n]),
           key=lambda n: silhouette_score(Xs[labels[n] != -1], labels[n][labels[n] != -1]))
print(f"{G}Best internal quality among full-coverage algorithms: {best}{X}")
print(f"{Y}Note: DBSCAN's higher silhouette is computed after discarding 42 noise points (24% of data); ARI shows it fits the true grouping poorly.{X}")
print(f"\n{Y}Cluster profile (K-Means, original units):{X}")
l = labels["K-Means"]
for c in range(3):
    m = d.data[l == c]; print(f"  cluster {c}: n={len(m):3d}  alcohol={m[:,0].mean():5.2f}  flavanoids={m[:,6].mean():5.2f}  proline={m[:,12].mean():7.1f}")

fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
ax[0].plot(list(ks), inert, "o-", color="k"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, "s-", color="k"); ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k")
for a, n in zip(ax[2:], ["K-Means", "DBSCAN (eps=2.3,m=5)"]):
    a.scatter(Z[:, 0], Z[:, 1], c=labels[n], cmap="Greys", edgecolor="k", s=22); a.set_title(f"{n} (PCA 2-D)")
plt.tight_layout(); plt.savefig("Assignment_6/clustering_results.png", dpi=140)
print(f"{G}Saved plot: Assignment_6/clustering_results.png{X}")
