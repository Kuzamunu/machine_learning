import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)


def plot_confusion(cm, classes, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)


df = pd.read_csv("../forestCover.csv", na_values="?")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X = X.drop(columns=["Observation_ID", "Water_Level", "Facet", "Inclination"])

cardinality_threshold = 20
feature_cardinality = X.nunique()
categorical_cols = feature_cardinality[
    feature_cardinality <= cardinality_threshold
].index.tolist()
numerical_cols = feature_cardinality[
    feature_cardinality > cardinality_threshold
].index.tolist()

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=X.index)

sample_size = 31012
splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)
for _, sample_indices in splitter.split(X, y_encoded):
    X_sample = X.iloc[sample_indices]
    y_sample = y_encoded.iloc[sample_indices]
classes = np.unique(y_sample)


knn_num = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
)
knn_cat = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
knn_pre = ColumnTransformer(
    [("num", knn_num, numerical_cols), ("cat", knn_cat, categorical_cols)]
)
knn_pipeline = Pipeline(
    [
        ("preprocessor", knn_pre),
        ("smote", "passthrough"),
        ("classifier", KNeighborsClassifier()),
    ]
)

dt_num = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
dt_cat = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
dt_pre = ColumnTransformer(
    [("num", dt_num, numerical_cols), ("cat", dt_cat, categorical_cols)]
)
dt_pipeline = Pipeline(
    [
        ("preprocessor", dt_pre),
        ("smote", "passthrough"),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        ),
    ]
)


knn_param_tune = {
    "smote": ["passthrough", SMOTE(k_neighbors=5)],
    "classifier__n_neighbors": randint(2, 8),
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}
dt_param_tune = {
    "smote": ["passthrough", SMOTE(k_neighbors=5)],
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__min_samples_split": randint(2, 10),
    "classifier__min_samples_leaf": randint(1, 5),
    "classifier__criterion": ["gini", "entropy"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_model(pipeline, param_dist, X, y, model_name, classes):
    print(f"\n{model_name} evaluation")
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=15,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    best_params = search.best_params_
    print(f"Best params: {best_params}")
    print(f"Best CV score (F1-macro): {search.best_score_:.4f}")

    all_metrics = []
    all_true = []
    all_preds = []
    fold = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_model = pipeline.set_params(**best_params)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_train = fold_model.predict(X.iloc[train_idx])
        y_pred_test = fold_model.predict(X.iloc[test_idx])
        all_true.extend(y.iloc[test_idx])
        all_preds.extend(y_pred_test)

        acc_tr = accuracy_score(y.iloc[train_idx], y_pred_train)
        f1_tr = f1_score(y.iloc[train_idx], y_pred_train, average="macro")
        acc_te = accuracy_score(y.iloc[test_idx], y_pred_test)
        f1_te = f1_score(y.iloc[test_idx], y_pred_test, average="macro")
        kappa = cohen_kappa_score(y.iloc[test_idx], y_pred_test)
        mcc = matthews_corrcoef(y.iloc[test_idx], y_pred_test)
        all_metrics.append(
            {
                "model": model_name,
                "fold": fold,
                "accuracy_train": acc_tr,
                "f1_macro_train": f1_tr,
                "accuracy_test": acc_te,
                "f1_macro_test": f1_te,
                "kappa": kappa,
                "mcc": mcc,
                "train_test_gap": acc_tr - acc_te,
            }
        )
        fold += 1

    cm_cv = confusion_matrix(all_true, all_preds, labels=classes)
    plot_confusion(
        cm_cv,
        classes,
        f"{model_name} --- CV Performance (All Test Folds)",
        os.path.join(OUTDIR, f"confusion_{model_name.lower()}_cv.svg"),
    )
    final_model = pipeline.set_params(**best_params)
    final_model.fit(X, y)
    cm_final = confusion_matrix(y, final_model.predict(X), labels=classes)
    plot_confusion(
        cm_final,
        classes,
        f"{model_name} --- Final Model (Trained on Data)",
        os.path.join(OUTDIR, f"confusion_{model_name.lower()}_final.svg"),
    )
    return all_metrics, best_params


print("starting")
knn_metrics, knn_params = evaluate_model(
    knn_pipeline, knn_param_tune, X_sample, y_sample, "kNN", classes
)
dt_metrics, dt_params = evaluate_model(
    dt_pipeline, dt_param_tune, X_sample, y_sample, "DecisionTree", classes
)
all_results = knn_metrics + dt_metrics
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(OUTDIR, "fold_metrics.csv"), index=False)
summary_table = df_results.groupby("model")[
    [
        "accuracy_train",
        "f1_macro_train",
        "accuracy_test",
        "f1_macro_test",
        "kappa",
        "mcc",
        "train_test_gap",
    ]
].agg(["mean", "std", "min", "max"])
summary_table.to_csv(os.path.join(OUTDIR, "full_summary.csv"))
print("\ncompleted!")
