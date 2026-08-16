import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage

##################### 1. Load data #####################
df = pd.read_csv("Dry_Bean.csv")
print(df.head())
print("Shape:", df.shape)

##################### 2. Find null and duplicate values #####################
print("Total null:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

##################### 3. Remove null and duplicate values #####################
df = df.dropna()
df = df.drop_duplicates()
print("After cleaning:", df.shape)

##################### 4. Label Encoder #####################
le = LabelEncoder()
for col in df.select_dtypes(exclude="number").columns:
    df[col] = le.fit_transform(df[col].astype(str))

##################### 5. Separate features and target #####################
X = df.drop(columns="Class")
y = df["Class"]        # sudhu ARI milanor jonno, training e use hobe na

##################### 6. EDA - IMPORTANT FEATURE PLOTTING (model er age) #####################
corr = X.corr()
var = X.var().sort_values(ascending=False)
print("\nTop variance features:\n", var.head())

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].bar(df["Class"].value_counts().index.astype(str), df["Class"].value_counts().values, color="steelblue")
ax[0].set_title("Class Distribution")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Number of Samples")

sns.heatmap(corr, ax=ax[1], cmap="coolwarm")
ax[1].set_title("Correlation Heatmap (PCA keno dorkar)")

ax[2].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap="viridis", s=5)
ax[2].set_xlabel(X.columns[0])
ax[2].set_ylabel(X.columns[1])
ax[2].set_title("Raw feature scatter")

plt.tight_layout()
plt.show()

##################### 7. Feature scaling #####################
X = StandardScaler().fit_transform(X)

##################### 8. PCA - reduce dimension #####################
p95 = PCA(n_components=0.95).fit(X)
print("PCA(0.95) needs", p95.n_components_, "components")

pca = PCA(n_components=2)     # 2D te namale DBSCAN o kaj kore, r plot o kora jay
X = pca.fit_transform(X)
print("After PCA:", X.shape)
print("Variance kept:", pca.explained_variance_ratio_.sum())

##################### 9. Find best K by Elbow Method #####################
ks = list(range(1, 11))
wcss = []
for k in ks:
    wcss.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_)

# elbow = prothom o shesh point er line theke sobcheye dure thaka point
x1, y1 = ks[0], wcss[0]
x2, y2 = ks[-1], wcss[-1]
dist = []
for i in range(len(ks)):
    d = abs((y2 - y1) * ks[i] - (x2 - x1) * wcss[i] + x2 * y1 - y2 * x1)
    d = d / (((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5)
    dist.append(d)
best_k = ks[dist.index(max(dist))]
print("Best K from Elbow Method:", best_k)

##################### 10. Cross check by Silhouette Score #####################
sil = []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    sil.append(silhouette_score(X, labels, sample_size=2000, random_state=42))
print("Best K from Silhouette:", range(2, 11)[sil.index(max(sil))])

##################### 11. Find best eps for DBSCAN #####################
min_s = 50
eps_auto = 0.25
best_sil = -1
print("eps    clusters   silhouette")
for e in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    lb = DBSCAN(eps=e, min_samples=min_s).fit_predict(X)
    nc = len(set(lb)) - (1 if -1 in lb else 0)
    if nc >= 2:
        s = silhouette_score(X, lb, sample_size=2000, random_state=42)
        print(f"{e:<7}{nc:<11}{s: .3f}")
        if s > best_sil:
            best_sil = s
            eps_auto = e
    else:
        print(f"{e:<7}{nc:<11}   -")
print("Best eps selected:", eps_auto)

##################### 12. Model list #####################
# ekhane sudhu ei list ta bodlabe - niche r kono code bodlate hobe na
models = [
    ("KMeans", KMeans(n_clusters=best_k, n_init=10, random_state=42)),
    ("Hierarchical", AgglomerativeClustering(n_clusters=best_k, linkage="ward")),
    ("GMM", GaussianMixture(n_components=best_k, random_state=42)),
    ("DBSCAN", DBSCAN(eps=eps_auto, min_samples=min_s)),
]

##################### 13. Apply all models #####################
all_labels = []
all_name = []
all_ari = []

for name, model in models:
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ari = adjusted_rand_score(y, labels)
    if n_clusters > 1:
        sil_value = silhouette_score(X, labels, sample_size=2000, random_state=42)
    else:
        sil_value = 0

    print("==============================", name, "==============================")
    print("Clusters found:", n_clusters)
    print(f"ARI : {ari: .3f}")
    print(f"Silhouette : {sil_value: .3f}")

    all_labels.append(labels)
    all_name.append(name)
    all_ari.append(ari)

##################### 14. Draw ALL plots in one figure #####################
fig, ax = plt.subplots(2, 4, figsize=(22, 10))

# --- row 1 : elbow, silhouette, true class, dendrogram ---
ax[0, 0].plot(ks, wcss, marker="o")
ax[0, 0].axvline(best_k, color="red", linestyle="--")
ax[0, 0].set_title("Elbow Method (best K = " + str(best_k) + ")")
ax[0, 0].set_xlabel("K")
ax[0, 0].set_ylabel("WCSS")

ax[0, 1].plot(range(2, 11), sil, marker="o", color="green")
ax[0, 1].set_title("Silhouette Score")
ax[0, 1].set_xlabel("K")
ax[0, 1].set_ylabel("Silhouette")

ax[0, 2].scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=4)
ax[0, 2].set_title("True Class (for reference)")
ax[0, 2].set_xlabel("PC1")
ax[0, 2].set_ylabel("PC2")

# dendrogram - 13000 point er dendrogram pora jay na, tai 300 ta sample neya hoy
sample = X[np.random.default_rng(42).choice(len(X), 300, replace=False)]
link = linkage(sample, method="ward")
dendrogram(link, ax=ax[0, 3], no_labels=True)
ax[0, 3].set_title("Dendrogram (300 sample)")
ax[0, 3].set_xlabel("Samples")
ax[0, 3].set_ylabel("Distance")

# --- row 2 : 4 ta clustering result ---
for i in range(len(models)):
    ax[1, i].scatter(X[:, 0], X[:, 1], c=all_labels[i], cmap="viridis", s=4)
    ax[1, i].set_title(all_name[i] + "\nARI = " + str(round(all_ari[i], 3)))
    ax[1, i].set_xlabel("PC1")
    ax[1, i].set_ylabel("PC2")

plt.tight_layout()
plt.show()

print("BEST CLUSTERING:", all_name[all_ari.index(max(all_ari))], "=", round(max(all_ari), 4))
