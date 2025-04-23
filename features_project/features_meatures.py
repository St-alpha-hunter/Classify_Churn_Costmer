import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier, plot_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor




class FeatureAnalysis:
    def __init__(self,df,features=None,model_cls=None,target_col="Churn"):
        self.df = df
        self.features = features
        self.model_cls = model_cls
        self.target_col = target_col

    ##分析特征值重要性
    def plot_feature_importance(self):
        if self.model_cls == XGBRegressor:
           model = XGBRegressor(n_estimators=100, random_state=42)
           model.fit(self.df[self.features], self.df[self.target_col])
           plot_importance(model, max_num_features=15)
           plt.title("Top 15 Feature Importance (XGB)")
           plt.show()

        elif self.model_cls == LGBMRegressor:
            model = LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.feature_importances_
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (LGBM)")
            plt.show()

        elif self.model_cls == CatBoostRegressor:
            model = CatBoostRegressor(verbose=0, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.get_feature_importance()
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (CatBoost)")
            plt.show()

        elif self.model_cls == RandomForestRegressor:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.feature_importances_
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (Random Forest)")
            plt.show()

        elif self.model_cls == XGBClassifier:
            model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            plot_importance(model, max_num_features=15)
            plt.title("Top 15 Feature Importance (XGBClassifier)")
            plt.show()

        elif self.model_cls == LGBMClassifier:
            model = LGBMClassifier(n_estimators=100, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.feature_importances_
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (LGBMClassifier)")
            plt.show()

        elif self.model_cls == CatBoostClassifier:
            model = CatBoostClassifier(verbose=0, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.get_feature_importance()
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (CatBoostClassifier)")
            plt.show()

        elif self.model_cls == RandomForestClassifier:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.feature_importances_
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (Random Forest Classifier)")
            plt.show()

        elif self.model_cls == LogisticRegression:
            model = LogisticRegression(max_iter=1000)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.coef_[0]
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.abs().nlargest(15).plot(kind='barh')
            plt.title("Top 15 Coefficients (Logistic Regression)")
            plt.show()

        elif self.model_cls == SVC:
            model = SVC(kernel='linear')
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.coef_[0]
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.abs().nlargest(15).plot(kind='barh')
            plt.title("Top 15 Coefficients (SVM)")
            plt.show()

        elif self.model_cls == MLPClassifier:
            print("MLPClassifier 不支持特征重要性图。")

    #一键画图
    def plot_feature_distribution(self):
        for col in self.features:
            plt.figure(figsize=(6, 3))
            sns.set_style("whitegrid")
            sns.histplot(self.df[col], kde=True)
            plt.title(f"{col} (skew = {self.df[col].skew():.2f})")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
            
    #和目标的相关性
    def plot_feature_vs_target(self):
        for col in self.features:
            plt.figure(figsize=(6, 3))
            sns.set_style("whitegrid")
            sns.scatterplot(data=self.df, x=col, y=self.target_col, alpha=0.3)
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title(f"{col} vs {self.target_col}")
            plt.tight_layout()
            plt.show()


    ##特征值相关性分析
    def plot_feature_correlation(self, add_features = None, threshold=0.85,verbose=True):
        """
        显示特征间的相关性热力图，并标记高相关对。
        """
        if add_features is None:
           add_features = self.df.select_dtypes(include='number').columns
        corr_matrix = self.df[add_features].corr()
        # 热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Feature Correlation Heatmap")
        plt.show()
        # 输出高度相关的特征对
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    score = corr_matrix.iloc[i, j]
                    high_corr.append((col1, col2, round(score, 3)))
        if high_corr:
            print(" 高度相关的特征对（|相关性| > {})：".format(threshold))
            for col1, col2, score in high_corr:
                print(f"{col1} & {col2} → 相关系数: {score}")
        else:
            print("没有检测到高度相关的特征对")
        #自动处理高相关特征值对
        #取的是 相关矩阵的上三角（即只看每对组合一次，跳过对称部分）,靠右的列优先被删，靠左的列优先保留
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > abs(threshold))]
        if verbose:
            print(f"🔍 检测到 {len(to_drop)} 个高相关特征将被删除（阈值：{threshold}）：")
            print(to_drop)
        advanced_feature = [col for col in self.features if col not in to_drop]
        return advanced_feature


    ### VIF检验
def check_multicollinearity(df, features=None, threshold=10, verbose= True):
    missing = [feat for feat in features if feat not in df.columns]
    

    """
    检查 DataFrame 中数值型变量的多重共线性（使用 VIF 指标）

    参数:
    df : pd.DataFrame —— 要检查的特征数据
    threshold : float —— 判断共线性是否严重的阈值（通常设为 5 或 10)

    返回:
    vif_df : pd.DataFrame —— 每个变量对应的 VIF 值
    """
    if features:
        df = df[features]  # 只保留指定的列

    # 只保留数值型变量
    print("传入的特征数:", len(features) if features else "None")
    print("原始 df.shape:", df.shape)

    numeric_df = df.select_dtypes(include=["number"]).dropna()


    print("最终进入VIF计算的列数:", numeric_df.shape[1])
    print("进入VIF计算的特征列:", numeric_df.columns.tolist())
    # 加常数项用于 statsmodels
    X = sm.add_constant(numeric_df)

    vif_data = {
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }

    vif_df = pd.DataFrame(vif_data)

    print("🧪 方差膨胀因子(VIF)检测结果:")
    print(vif_df[vif_df["VIF"] > threshold])

    #删除特征值逻辑，每轮删除一个最大的VIF
    while True:
        X = sm.add_constant(numeric_df)
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        vif = vif.drop("const")

        max_vif = vif.max()
        if max_vif > threshold:
            drop_feat = vif.idxmax()
            if verbose:
                print(f"⚠️  删除高VIF特征: {drop_feat} (VIF={max_vif:.2f})")
            numeric_df = numeric_df.drop(columns=[drop_feat])
           
        else:
            break
    print("去掉缺失值后的形状:", numeric_df.shape)
    VIF_name = vif.index.tolist()
    advanced_features_ultimate = [col for col in VIF_name ]

    return advanced_features_ultimate

    



def feature_selection_by_k(features, target_col, max_k=20, rank_features=10, model_cls=RandomForestRegressor):

    results = []

    # Step 1: 用过滤法选 top max_k 个初始特征
    filter_selector = SelectKBest(score_func=f_regression, k=max_k)
    X_filtered = filter_selector.fit_transform(features, target_col)
    filtered_cols = features.columns[filter_selector.get_support()]
    X_filtered_df = features[filtered_cols]

    # Step 2: 遍历 k，嵌入法评估每组前 k 特征的贡献
    for k in range(10, max_k + 1):
        X_k = X_filtered_df.iloc[:, :k]
        model = model_cls(n_estimators=50, random_state=42)
        model.fit(X_k, target_col)

        importances = model.feature_importances_
        num_features_to_rank = min(rank_features, k)
        top_k_idx = np.argsort(importances)[::-1][:num_features_to_rank]
        top_k_features = X_k.columns[top_k_idx]
        X_top_k = X_k[top_k_features]  # optional

        results.append((k, top_k_features.tolist()))

    results_df = pd.DataFrame(results, columns=["num_features", "top_features"])
    return results_df


def select_final_top_features(features, target_col, max_k=20, top_k=15, model_cls=RandomForestRegressor):
    from sklearn.feature_selection import SelectKBest, f_regression
    import numpy as np

    # Step 1: 过滤法初筛
    filter_selector = SelectKBest(score_func=f_regression, k=max_k)
    X_filtered = filter_selector.fit_transform(features, target_col)
    filtered_cols = features.columns[filter_selector.get_support()]
    X_filtered_df = features[filtered_cols]

    # Step 2: 嵌入法评估特征重要性
    model = model_cls(n_estimators=100, random_state=42)
    model.fit(X_filtered_df, target_col)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_features = X_filtered_df.columns[top_idx]
    top_features = top_features.tolist()

    return top_features
