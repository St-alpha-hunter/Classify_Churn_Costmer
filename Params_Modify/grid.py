from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC
import pandas as pd


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from imblearn.combine import SMOTEENN
import pandas as pd
import inspect

def grid_search_models_f1(X, y, model_params_dict, cv=5, average="weighted", verbose=True):
    results = []
    scorer = make_scorer(f1_score, average=average)

    for name, (model, params) in model_params_dict.items():
        grid = GridSearchCV(model, params, scoring=scorer, cv=cv, n_jobs=-1)
        grid.fit(X, y)
        best_score = grid.best_score_
        best_params = grid.best_params_
        results.append({
            "Model": name,
            "Best F1 Score": best_score,
            "Best Params": best_params
        })
        if verbose:
            print(f"✅ {name} 完成：F1 = {best_score:.4f}")
            print("最佳参数：", best_params)
            print("-" * 40)

    return pd.DataFrame(results).sort_values("Best F1 Score", ascending=False)
