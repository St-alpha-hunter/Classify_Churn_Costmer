from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import pandas as pd

def compare_models_f1(X, y, models, cv=5, average="binary", verbose=True):
    scorer = make_scorer(f1_score, average=average)
    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        results.append({
            "Model": name,
            "F1 Score": scores.mean(),
            "Std": scores.std()
        })

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    
    if verbose:
        print(results_df)
    
    return results_df
