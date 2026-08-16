import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

##################### 1. Load data #####################
df = pd.read_csv("Dry_Bean.csv")
print(df.head())
print("Shape:", df.shape)
print(df.info())
print(df.describe())

##################### 2. Find null and duplicate values #####################
print("Null values:\n", df.isnull().sum())
print("Total null:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

##################### 3. Remove null and duplicate values #####################
df = df.dropna()
df = df.drop_duplicates()
print("After cleaning:", df.shape)

##################### 4. Label Encoder #####################
le = LabelEncoder()
for col in df.select_dtypes(exclude="number").columns:   # sob pandas version e kaj kore
    df[col] = le.fit_transform(df[col].astype(str))
print(df.head())

##################### 5. Separate features and target #####################
X = df.drop(columns="Class")
y = df["Class"]

##################### 6. EDA - IMPORTANT FEATURE PLOTTING (model er age) #####################
imp_model = RandomForestClassifier(n_estimators=100, random_state=42)
imp_model.fit(X, y)
importance = pd.Series(imp_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)
top4 = importance.index[:4]        # sobcheye important 4 ta feature

fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# 6a. class distribution - data balanced kina dekhe
df["Class"].value_counts().plot(kind="bar", ax=ax[0, 0], color="steelblue")
ax[0, 0].set_title("Class Distribution")
ax[0, 0].set_xlabel("Class")
ax[0, 0].set_ylabel("Number of Samples")

# 6b. feature importance - kon feature beshi kaje lage
importance.plot(kind="barh", ax=ax[0, 1], color="seagreen")
ax[0, 1].set_title("Feature Importance (Random Forest)")
ax[0, 1].invert_yaxis()

# 6c. correlation heatmap - kon feature gulo eki jinis mapche (PCA keno lagbe)
sns.heatmap(X.corr(), ax=ax[0, 2], cmap="coolwarm", cbar=True)
ax[0, 2].set_title("Correlation Heatmap")

# 6d. top feature er histogram
ax[1, 0].hist(X[top4[0]], bins=40, color="darkorange")
ax[1, 0].set_title("Histogram: " + top4[0])
ax[1, 0].set_xlabel(top4[0])

# 6e. top feature class onujayi boxplot - class alada kore kina
sns.boxplot(x=y, y=X[top4[1]], ax=ax[1, 1])
ax[1, 1].set_title("Boxplot: " + top4[1] + " by Class")

# 6f. top 2 feature er scatter
sc1 = ax[1, 2].scatter(X[top4[0]], X[top4[1]], c=y, cmap="viridis", s=5)
ax[1, 2].set_xlabel(top4[0])
ax[1, 2].set_ylabel(top4[1])
ax[1, 2].set_title("Scatter: top 2 features")

plt.tight_layout()
plt.show()

##################### 7. Train test split #####################
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

##################### 8. Feature scaling #####################
scaling = StandardScaler()
X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(X_test)   # split er por scaling, na hole data leakage hobe

##################### 9. PCA - reduce dimension #####################
print("Before PCA:", X_train.shape)
pca = PCA(n_components=0.95)      # 95% variance rakhbe, koyta component lagbe PCA nije thik korbe
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("After PCA:", X_train_pca.shape)
print("Variance kept:", pca.explained_variance_ratio_.sum())

##################### 10. Model list #####################
# ekhane sudhu ei list ta bodlabe - niche r kono code bodlate hobe na
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("SVM", SVC(kernel="rbf")),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Naive Bayes", GaussianNB()),
]

##################### 11. BEFORE PCA - train and predict #####################
names = []
acc_before = []
all_cf = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cf = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred, average="weighted")
    recall_value = recall_score(y_test, y_pred, average="weighted")
    precision_value = precision_score(y_test, y_pred, average="weighted")

    print("=============== BEFORE PCA :", name, "===============")
    print(f"Accuracy of the model is : {accuracy: .2f}")
    print(f"F1-Score : {f1_value: .2f}")
    print(f"Recall: {recall_value: .2f}")
    print(f"Precision: {precision_value: .2f}")
    print(f"Confusion_Matrix:\n{cf}")
    print(f"Classification Report :\n{cr}")

    names.append(name)
    acc_before.append(accuracy)
    all_cf.append(cf)

##################### 12. AFTER PCA - same models again #####################
acc_after = []
for name, model in models:
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"=== AFTER PCA : {name} === Accuracy: {accuracy: .2f}")
    acc_after.append(accuracy)

##################### 13. Comparison table #####################
print("\n--------- COMPARISON TABLE ---------")
print("Model                  Before PCA   After PCA")
for i in range(len(names)):
    print(f"{names[i]:<22} {acc_before[i]:.4f}      {acc_after[i]:.4f}")

##################### 14. Draw all confusion matrix #####################
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
for i in range(len(models)):
    r = i // 3
    c = i % 3
    ConfusionMatrixDisplay(all_cf[i]).plot(ax=ax[r, c], cmap="Blues", colorbar=False)
    ax[r, c].set_title(names[i] + "\nAccuracy = " + str(round(acc_before[i], 3)))
plt.tight_layout()
plt.show()

##################### 15. Draw accuracy comparison and PCA #####################
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

pos = np.arange(len(names))
ax[0].bar(pos - 0.2, acc_before, 0.4, label="Before PCA")
ax[0].bar(pos + 0.2, acc_after, 0.4, label="After PCA")
ax[0].set_xticks(pos)
ax[0].set_xticklabels(names, rotation=30, fontsize=8)
ax[0].set_ylim(0, 1)
ax[0].set_title("Accuracy: Before vs After PCA")
ax[0].legend()

ax[1].plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), marker="o")
ax[1].axhline(0.95, color="red", linestyle="--")
ax[1].set_title("PCA Scree Plot")
ax[1].set_xlabel("Components")
ax[1].set_ylabel("Cumulative Variance")

ax[2].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="viridis", s=5)
ax[2].set_title("Data in 2D PCA space")
ax[2].set_xlabel("PC1")
ax[2].set_ylabel("PC2")

plt.tight_layout()
plt.show()

print("BEST MODEL:", names[acc_before.index(max(acc_before))], "=", round(max(acc_before), 4))
