#路径管理/全局变量
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns       # sns 是 seaborn 的常用别名
import matplotlib.pyplot as plt  # plt 是 matplotlib.pyplot 的常用别名

# 设置为项目根目录（包含 data, pipline 等文件夹的目录）
project_root = os.path.abspath("..")
os.chdir(project_root)
sys.path.append(project_root)
from utils.path_helper import get_data_path

#基础模块
from PreAnalysis.Preprocess import preprocess
from pipeline.pipeline import processing_data
from features_project.features import add_selected_features
from features_project.features_meatures import FeatureAnalysis
from features_project.features_meatures import feature_selection_by_k
from features_project.features_meatures import select_final_top_features
from features_project.features_meatures import check_multicollinearity

#模型导入
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

##调参
from Params_Modify.grid import grid_search_models_f1
from Params_Modify.config_model_params import model_params
from utils.config import best_model_configs
##训练
from train_model.train import train_model_classify

##评估
from evaluate_model.evaluate import evaluate


#读取数据
df = pd.read_csv(get_data_path("TrainingSet.csv"))
print(df.shape)
print("✅step1- 读取数据完成")


##实例化后方便初步分析
process_data = preprocess(df)
#填充前看图 raw pictures
#process_data. picture_analysis()  ##if you want to analyse through the pictures, canceling "##"
##查验并填充
process_data.missingdata_count()
df_withoutfalse = process_data.detect_extreme()
print(df_withoutfalse.shape)
#完成填充缺失值
df_withoutfalse = processing_data(df_withoutfalse)
df_withoutfalse.processing_number_data()
df_cleaned = df_withoutfalse.processing_amount_data()
print(df_cleaned.shape)
#填充后看图 new pictures   ##if you want to analyse through the pictures, canceling "##"
process_data_after = preprocess(df_cleaned)
#process_data_after.picture_analysis()  ##if you want to analyse through the pictures, canceling "##"
process_data_after.missingdata_count()
print(df_cleaned.shape)
print("✅step2- 填充缺失值完成")


#录入特征值
My_features = ["normal_age", "Data_Usage_log", "Monthly_Charges_log", "hot_gender","Payment_Method_hot", "Customer_Service_Calls_hot","Avg_Tenure_Month", "Contract_Type"]
#加入特征值
df_cleaned_features,added_features = add_selected_features(df_cleaned, features_to_use=My_features)
for col in df_cleaned_features.columns:
    print(col)
#print(df_cleaned_features[added_features].dtypes)
#转bool为数值
df_cleaned_features[added_features] = df_cleaned_features[added_features].astype("int")
df_cleaned_features_1 = df_cleaned_features.copy()
print("✅step3- 特征工程录入完成")


#相关性分析
df_cleaned_features = FeatureAnalysis(df_cleaned_features, features=added_features, model_cls=RandomForestClassifier, target_col="Churn")
#df_cleaned_features.plot_feature_importance()  ##if you want to analyse through the pictures, canceling "##"
#df_cleaned_features.plot_feature_distribution()  ##if you want to analyse through the pictures, canceling "##"
#df_cleaned_features.plot_feature_vs_target()    ##if you want to analyse through the pictures, canceling "##"
advanced_feature = df_cleaned_features.plot_feature_correlation(add_features=added_features)
###########
df_cleaned_features_1[advanced_feature] = df_cleaned_features_1[advanced_feature].astype("int")
###########
print("✅step4- 特征工程相关性分析完成")


##VIF分析
for feature in advanced_feature:
    print(feature)
advanced_features_ultimate = check_multicollinearity(df_cleaned_features_1, features = advanced_feature, threshold=10, verbose= True)
print(len(advanced_features_ultimate))
print("✅step5- 特征工程VIF完成")


##分析每个特征数下的最优组合
df_result = feature_selection_by_k(df_cleaned_features_1[advanced_features_ultimate], target_col = df_cleaned_features_1["Churn"], max_k=20, rank_features=10, model_cls = RandomForestClassifier)
print(df_result)


##直接选出最终最重要的10个特征
final_features = select_final_top_features(df_cleaned_features_1[advanced_features_ultimate], target_col = df_cleaned_features_1["Churn"], max_k=20, top_k=15, model_cls = RandomForestClassifier)
for feature in final_features:
    print(feature)
print("✅step5- 特征工程筛选完成")


##网格化调参
X = df_cleaned_features_1[final_features]
y = df_cleaned_features_1["Churn"]
grid_results = grid_search_models_f1(X, y, model_params)
print(grid_results)
##记入config
print("✅step6- 网格化调参完成")

#训练模型
after_trained_model,X_train, X_test, y_train, y_test = train_model_classify(df_cleaned_features = df_cleaned_features_1, model_config = best_model_configs["RandomForest"], final_features = final_features)
print("✅step7- 模型训练完成")

#评估模型
ev = evaluate(model = after_trained_model, X_test = X_test, y_test = y_test)
ev.evaluate_classify_model(average="weighted") 
print("✅step8- 模型评估完成")