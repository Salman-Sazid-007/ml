import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage

#  DATA SOURCE
# self made
# from sklearn.datasets import make_blobs
# X0, y0 = make_blobs(n_samples=600, centers=[[0,0],[5,5],[0,6]], cluster_std=[.4,1.2,.6], random_state=42)
# df = pd.DataFrame(X0, columns=["feature_1","feature_2"]); df["target"] = y0; LABEL = "target"

# sklearn built-in
from sklearn.datasets import load_wine
df = load_wine(as_frame=True).frame
LABEL = "target"          # others: load_iris, load_breast_cancer

# user-provided
# df = pd.read_csv("penguins.csv", na_values=["?","N/A","NA","na",""," ","missing","-","null","None"])
# LABEL = "species"      # None if the data has no label column (needed for ORIGINAL plot + ARI)

K = 3            # number of clusters. Check the elbow plot and change if needed
SAMPLE = 0       # set 5000 for big data. 0 = use all
EPS = 2.0        # DBSCAN radius. NO PCA = many dimensions = distances are BIG, so eps must be big.
                 # 2D data -> try 0.3 / 0.5 | 13D data -> try 2.0 / 2.5
USE = "all"      # or a list e.g. ["KMeans", "GMM"]

# DISPLAY THE DATA
pd.set_option("display.width", 200, "display.max_columns", 50)
print("\nDATA BEFORE CLEANING")
print("Shape:", df.shape, " (rows =", df.shape[0], ", columns =", df.shape[1], ")")
print("\n--- FIRST 5 ROWS ---\n", df.head())

# CLEAN
df = df.drop_duplicates().reset_index(drop=True)

def junk(c):                                       # constant / ID / date column?
    num = pd.api.types.is_numeric_dtype(df[c])
    return (df[c].nunique() <= 1 or not num and (df[c].nunique() == len(df) or
            pd.to_datetime(df[c], errors="coerce", format="mixed").notna().mean() > .8))

drop_list = [c for c in df.columns if c != LABEL and junk(c)]
print("\nuseless columns:", drop_list)
df = df.drop(columns=drop_list)

for c in df.columns:                               # text -> number
    if not pd.api.types.is_numeric_dtype(df[c]):
        cl = df[c].astype(str).str.replace(",", "").str.replace("$", "").str.strip()
        n = pd.to_numeric(cl, errors="coerce")
        df[c] = n if n.notna().mean() > .8 else LabelEncoder().fit_transform(cl.str.lower())

df = df.fillna(df.median()).dropna()

if SAMPLE and len(df) > SAMPLE:
    df = df.sample(SAMPLE, random_state=42).reset_index(drop=True)
    print("Sampled to:", df.shape)

#  REMOVE THE LABEL (clustering must not see it)
y = None
if LABEL is not None and LABEL in df.columns:
    y = df[LABEL]; df = df.drop(columns=LABEL)
print("Feature shape:", df.shape)

# EDA PLOTS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df.corr(), ax=ax1, cmap="coolwarm", annot=(df.shape[1] <= 8))
ax1.set_title("Correlation Heatmap")
df.var().sort_values().tail(15).plot(kind="barh", ax=ax2, color="seagreen")
ax2.set_title("Feature Variance")
plt.suptitle("EDA: overview"); plt.tight_layout(); plt.show()

# SCALE  (no PCA - clustering runs on all the features)
X = StandardScaler().fit_transform(df)
COLS = list(df.columns)                            # feature names, used for the plot labels
print("\n--- FIRST 5 ROWS (scaled) ---\n", pd.DataFrame(X, columns=COLS).head().round(3))
print("Clustering will run on", X.shape[1], "features")

SS = 2000 if len(X) > 2000 else None               # sample silhouette on big data

# ELBOW METHOD
ks, wcss, sil_k = list(range(1, 11)), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    wcss.append(km.inertia_)
    if k >= 2:
        sil_k.append(silhouette_score(X, km.labels_, sample_size=SS, random_state=42))

# elbow = the K furthest from the straight line joining the first and last point
x1, y1, x2, y2 = ks[0], wcss[0], ks[-1], wcss[-1]
line = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** .5
dist = [abs((y2 - y1) * k - (x2 - x1) * w + x2 * y1 - y2 * x1) / line for k, w in zip(ks, wcss)]
best_k = ks[dist.index(max(dist))]
print("Best K by Elbow:", best_k, "| Best K by Silhouette:", range(2, 11)[sil_k.index(max(sil_k))],
      "| Using K =", K)

# MODELS
models = [("KMeans", KMeans(n_clusters=K, n_init=10, random_state=42)),
          ("Hierarchical", AgglomerativeClustering(n_clusters=K, linkage="ward")),
          ("GMM", GaussianMixture(n_components=K, random_state=42)),
          ("DBSCAN", DBSCAN(eps=EPS, min_samples=5))]
if USE != "all": models = [m for m in models if m[0] in USE]

# RUN ALL MODELS
def run(data):
    print(f"\nCLUSTERING  ({data.shape[1]} features)\n")
    all_labels, scores = [], []
    for name, model in models:
        labels = model.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)      # -1 = DBSCAN noise
        s = {"Clusters": n_clusters, "Noise": list(labels).count(-1)}
        if n_clusters > 1:
            s["Silhouette"] = silhouette_score(data, labels, sample_size=SS, random_state=42)
        else:                                      # 1 cluster -> silhouette is undefined
            s["Silhouette"] = 0
        if y is not None: s["ARI"] = adjusted_rand_score(y, labels)
        print("\n", name)
        for metric, value in s.items():
            print(f"{metric:<18}: {value: .4f}")
        all_labels.append(labels); scores.append(s)
    return all_labels, scores

labs, res = run(X)

names = [m[0] for m in models]
MAIN = "ARI" if y is not None else "Silhouette"
KEYS = [k for k in res[0] if k not in ("Clusters", "Noise")]

# RESULT TABLE
print("\nFINAL RESULT TABLE\n")
print(f"{'Model':<16}{'Clusters':<11}{'Noise':<8}" + "".join(f"{k:<14}" for k in KEYS))
for name, s in zip(names, res):
    print(f"{name:<16}{s['Clusters']:<11}{s['Noise']:<8}" + "".join(f"{s[k]:<14.4f}" for k in KEYS))

# PLOT EVERY METRIC
fig, axes = plt.subplots(1, len(KEYS), figsize=(4.8 * len(KEYS), 4.5), squeeze=False)
pos = np.arange(len(names))                        # one slot per model
for ax, k in zip(axes[0], KEYS):
    ax.bar(pos, [r[k] for r in res], .5, color="steelblue")
    ax.set_xticks(pos); ax.set_xticklabels(names, rotation=40, fontsize=7)
    ax.set_title(k + " (higher better)")
plt.suptitle("All clustering metrics"); plt.tight_layout(); plt.show()

# CLUSTER PLOTS + ELBOW + DENDROGRAM
NCOL = len(models) + 1                             # +1 for the ORIGINAL column
fig, axes = plt.subplots(2, NCOL, figsize=(5.2 * NCOL, 9), squeeze=False)
ax_elbow, ax_sil, ax_dend = axes[0][:3]
for ax in axes[0][3:]: ax.axis("off")              # unused top-row slots

ax_elbow.plot(ks, wcss, marker="o"); ax_elbow.axvline(best_k, color="red", ls="--")
ax_elbow.set(title=f"Elbow Method (best K = {best_k})", xlabel="K", ylabel="WCSS")

ax_sil.plot(range(2, 11), sil_k, marker="o", color="green")
ax_sil.set(title="Silhouette vs K", xlabel="K")

n_dendro = min(300, len(X))                        # a full dendrogram is unreadable
picked = np.random.default_rng(42).choice(len(X), n_dendro, replace=False)
dendrogram(linkage(X[picked], method="ward"), ax=ax_dend, no_labels=True)
ax_dend.set_title(f"Dendrogram ({n_dendro} samples)")

# no PCA, so the scatter shows the first 2 features of the scaled data
xx, yy = X[:, 0], X[:, 1] if X.shape[1] > 1 else X[:, 0]

ax0 = axes[1, 0]                                   # ORIGINAL clusters (ground truth)
if y is not None:
    ax0.scatter(xx, yy, c=y, cmap="viridis", s=12)
    ax0.set_title(f"ORIGINAL\ntrue '{LABEL}' ({y.nunique()} groups)")
else:                                              # no label column -> nothing to colour by
    ax0.scatter(xx, yy, c="grey", s=12)
    ax0.set_title("ORIGINAL DATA\n(no label column)")
ax0.set(xlabel=COLS[0], ylabel=COLS[1] if len(COLS) > 1 else COLS[0])

for i, name in enumerate(names):
    ax = axes[1, i + 1]
    ax.scatter(xx, yy, c=labs[i], cmap="viridis", s=12)
    ax.set_title(f"{name}\n{MAIN} = {round(res[i][MAIN], 3)}")
    ax.set(xlabel=COLS[0], ylabel=COLS[1] if len(COLS) > 1 else COLS[0])

plt.suptitle("ORIGINAL clusters vs every model", fontsize=14)
plt.tight_layout(rect=(0, 0, 1, .96)); plt.show()

# FINAL VERDICT
ranking = sorted(zip([r[MAIN] for r in res], names, res), key=lambda t: t[0], reverse=True)
print("\n" + "=" * 60, "\nFINAL VERDICT\n" + "=" * 60)
print("\nRANKING by " + MAIN + ":")
for rank, (main_score, name, s) in enumerate(ranking, 1):
    print(f"  {rank}. {name:<14} " + "  ".join(f"{k}={s[k]:.4f}" for k in KEYS))

print("\n>>> BEST CLUSTERING :", ranking[0][1], "|", MAIN, "=", round(ranking[0][0], 4))
