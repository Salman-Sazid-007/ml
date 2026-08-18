import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay, r2_score,
                             mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)

# ================= 1. DATA SOURCE - keep one, comment the other =================
# from sklearn.datasets import load_wine    # or load_iris / load_breast_cancer / load_digits / load_diabetes
# df = load_wine(as_frame=True).frame; TARGET = "target"; DROP = []

df = pd.read_csv("titanic.csv", na_values=["?","N/A","NA","na",""," ","missing","-","null","None"])
TARGET = "survived"   # None = use the last column
DROP = ["alive","class","who","adult_male"]   # titanic LEAK cols | mpg -> ["name"] | others -> []

TEST_SIZE = 0.3   # exam wants 70/30
SAMPLE = 0        # set 5000 for big data (diamonds). 0 = use all
USE = "all"       # or a list e.g. ["Random Forest", "SVM"]

# ================= 2. DISPLAY THE DATA =================
pd.set_option("display.width", 200, "display.max_columns", 50)
print("DATA BEFORE CLEANING")
print("Shape:", df.shape, " (rows =", df.shape[0], ", columns =", df.shape[1], ")")
print("\n--- RANDOM 5 ROWS ---\n", df.sample(min(5, len(df)), random_state=1))

# ================= 3. CLEAN =================
if TARGET is None:
    TARGET = df.columns[-1]
    print("\n WARNING", TARGET)

df = df.drop(columns=[c for c in DROP if c in df.columns])   # drop leak / ID columns
df = df.drop_duplicates().reset_index(drop=True)

for c in df.columns:                               # text -> number
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = LabelEncoder().fit_transform(df[c].astype(str).str.strip().str.lower())

df = df.fillna(df.median())                        # missing -> median

if SAMPLE and len(df) > SAMPLE:
    df = df.sample(SAMPLE, random_state=42).reset_index(drop=True)
    print("Sampled to:", df.shape)

X, y = df.drop(columns=TARGET), df[TARGET]
TASK = "classification" if y.nunique() <= 20 else "regression"

# ================= 4. MODELS + METRICS BY TASK =================
if TASK == "classification":
    models = [("Logistic Regression", LogisticRegression(max_iter=1000)),
              ("KNN", KNeighborsClassifier()),
              ("SVM", SVC(kernel="rbf")),
              ("Decision Tree", DecisionTreeClassifier(random_state=42)),
              ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
              ("Naive Bayes", GaussianNB())]
    def score(t, p):
        return {"Accuracy":  accuracy_score(t, p),
                "F1":        f1_score(t, p, average="weighted"),
                "Precision": precision_score(t, p, average="weighted", zero_division=0),
                "Recall":    recall_score(t, p, average="weighted")}
    MAIN = "Accuracy"
else:
    models = [("Linear Regression", LinearRegression()),
              ("KNN", KNeighborsRegressor()),
              ("SVR", SVR()),
              ("Decision Tree", DecisionTreeRegressor(random_state=42)),
              ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42))]
    def score(t, p):
        return {"R2":   r2_score(t, p),
                "MSE":  mean_squared_error(t, p),
                "RMSE": mean_squared_error(t, p) ** 0.5,
                "MAE":  mean_absolute_error(t, p),
                "MAPE": mean_absolute_percentage_error(t, p)}
    MAIN = "R2"

if USE != "all": models = [m for m in models if m[0] in USE]
names = [m[0] for m in models]

# ================= 5. EDA PLOTS =================
RF = RandomForestClassifier if TASK == "classification" else RandomForestRegressor
imp = pd.Series(RF(random_state=42).fit(X, y).feature_importances_, X.columns).sort_values()
print("\nFeature importance:\n", imp.iloc[::-1].head(10))      # iloc[::-1] = biggest first

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
imp.tail(15).plot.barh(ax=ax1, color="seagreen", title="Feature Importance")
sns.heatmap(X.corr(), ax=ax2, cmap="coolwarm"); ax2.set_title("Correlation Heatmap")
plt.tight_layout(); plt.show()

# ================= 6. SPLIT + SCALE =================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
sc = StandardScaler()
Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)   # fit on train only = no data leakage
print("\nTrain:", Xtr.shape, "| Test:", Xte.shape)
print("\n--- 5 SCALED TRAIN ROWS ---\n", pd.DataFrame(Xtr, columns=X.columns).head())

# ================= 7. RUN ALL MODELS =================
def run(train_X, test_X):
    scores, cms = [], []
    for name, model in models:
        pred = model.fit(train_X, ytr).predict(test_X)
        s = score(yte, pred); scores.append(s)
        print("\n", name)
        for metric, value in s.items():
            print(f"{metric:<10}: {value: .4f}")
        if TASK == "classification":
            cms.append(confusion_matrix(yte, pred))
    return scores, cms

res, cms = run(Xtr, Xte)
KEYS = list(res[0].keys())         # metric names

# ================= 8. RESULT TABLE =================
print("\nFINAL RESULT TABLE\n")
print(f"{'Model':<22}" + "".join(f"{k:<12}" for k in KEYS))
for name, s in zip(names, res):
    print(f"{name:<22}" + "".join(f"{s[k]:<12.4f}" for k in KEYS))

# ================= 9. PLOT EVERY METRIC =================
fig, axes = plt.subplots(1, len(KEYS), figsize=(5 * len(KEYS), 4.5), squeeze=False)
pos = np.arange(len(names))                        # one slot per model
for ax, k in zip(axes[0], KEYS):
    ax.bar(pos, [r[k] for r in res], .5, color="steelblue")
    ax.set_xticks(pos); ax.set_xticklabels(names, rotation=40, fontsize=7)
    ax.set_title(k)
plt.suptitle("All metrics"); plt.tight_layout(); plt.show()

# ================= 10. CONFUSION MATRICES (3 per row) =================
if TASK == "classification":
    n_rows = int(np.ceil(len(models) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows))
    axes = np.array(axes).flatten()                # grid -> flat list
    for i, (cm, name) in enumerate(zip(cms, names)):
        ConfusionMatrixDisplay(cm).plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(f"{name}\n{MAIN} = {round(res[i][MAIN], 3)}")
    for empty in axes[len(models):]: empty.axis("off")
    plt.suptitle("Confusion Matrix"); plt.tight_layout(); plt.show()

# ================= 11. FINAL VERDICT =================
ranking = sorted(zip([r[MAIN] for r in res], names, res), key=lambda t: t[0], reverse=True)
print("\nFINAL VERDICT  (TASK = " + TASK + ")\n")
print("\nRANKING by " + MAIN + ":")
for rank, (main_score, name, s) in enumerate(ranking, 1):
    print(f"  {rank}. {name:<22} " + "  ".join(f"{k}={s[k]:.4f}" for k in KEYS))

print("\nBEST MODEL :", ranking[0][1], "|", MAIN, "=", round(ranking[0][0], 4))
