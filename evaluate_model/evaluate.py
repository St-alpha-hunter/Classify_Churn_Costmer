from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from model.train_model import train_model
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import pandas as pd
import numpy as np


class evaluate:
    def __init__(self, model,X_test, y_test,target_col = "Churn"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.target_col = target_col
    


    def evaluate_regress_model(self):
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    def evaluate_classify_model(self, average="binary"):
        y_pred = self.model.predict(self.X_test)

        # 如果模型支持 predict_proba，计算 AUC
        try:
            y_proba = self.model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_proba)
        except:
            auc = None

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=average)
        precision = precision_score(self.y_test, y_pred, average=average)
        recall = recall_score(self.y_test, y_pred, average=average)

        print(f"✅ Accuracy : {acc:.4f}")
        print(f"✅ F1 Score : {f1:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall   : {recall:.4f}")
        if auc is not None:
            print(f"✅ AUC      : {auc:.4f}")
        else:
            print("⚠️  当前模型不支持 predict_proba，无法计算 AUC")

        return {
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc
        }

    def enhanced_cross_validate(self, model, features=None, cv=5, verbose=True, return_df=False):
        """
        支持 MAE、MSE、R² 的交叉验证函数
        """
        scoring = {
            'MAE': 'neg_mean_absolute_error',
            'MSE': 'neg_mean_squared_error',
            'R2': 'r2'
        }
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(self.model, features, self.target_col, cv=kfold, scoring=scoring, return_train_score=False)
        # 转为 DataFrame 方便后处理
        df_scores = pd.DataFrame(scores)
        if verbose:
            print(f"📊 {cv}-折交叉验证结果:")
            for metric in ['MAE', 'MSE', 'R2']:
                test_col = f'test_{metric}'
                mean = df_scores[test_col].mean()
                std = df_scores[test_col].std()
                if metric != 'R2':  # MAE, MSE 是负的
                    mean, std = -mean, std
                print(f"🔹 {metric:<4}: {mean:.4f} ± {std:.4f}")
        if return_df:
            return df_scores
        else:
            return {metric: df_scores[f'test_{metric}'] for metric in scoring}
    ##指定评估标准（回归模型最常见如 MAE、RMSE、R²）