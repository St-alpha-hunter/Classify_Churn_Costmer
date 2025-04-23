# config_model_params.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

model_params = {
    "Logistic": (
        LogisticRegression(max_iter=1000),
        {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        }
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }
    ),
    "GBDT": (
        GradientBoostingClassifier(),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    ),
    "SVM": (
        SVC(),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    )
}
