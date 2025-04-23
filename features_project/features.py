import os
import sys
import random
import pandas as pd
import numpy as np
from pipeline.pipeline import processing_data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor



# 1. 特征函数注册字典
FEATURE_FUNCTIONS = {}

def register_feature(name):
    def decorator(func):
        FEATURE_FUNCTIONS[name] = func
        return func
    return decorator

#0 
@register_feature("age") 
def age(df):
    df["age"] = df["Age"]
    return df

#1
@register_feature("standardize_age") 
def normal_age(df):
    df["standardize_age"] = (df["Age"] - df["Age"].mean())/df["Age"].std()
    return df

#2
@register_feature("normal_age") 
def normal_age(df):
    df["normal_age"] = (df["Age"] - df["Age"].min())/(df["Age"].max() - df["Age"].min())
    return df

#3
@register_feature("hot_gender") 
def hot_gender(df):
    print("处理前 hot_gender 的 df.columns:", df.columns.tolist())
    df = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
    return df
#['Gender_Female', 'Gender_Male', 'Gender_Non-binary']
#Gender_Female
#Gender_Male
#Gender_Non-binary

#4
@register_feature("Tenure_Months")
def Tenure_Months(df):
    df["Tenure_Months"] = df["Tenure_Months"]
    return df 


#5
@register_feature("Avg_Tenure_Month")
def Avg_Tenure_Month(df):
    df["Avg_Tenure_Month"] = (df["Tenure_Months"] - df["Tenure_Months"].min())/df["Tenure_Months"].std()
    return df 


#6
@register_feature("Contract_Type")
def Contract_Type(df):
    df = pd.get_dummies(df, columns=['Contract_Type'], prefix='Contract_Type')
    return df
#["Contract_Type_Month-to-month","Contract_Type_One year","Contract_Type_Two year"]
#Contract_Type_Month-to-month
#Contract_Type_One year
#Contract_Type_Two year


#7
@register_feature("Data_Usage_log")
def Data_Usage_log(df):
    df['Data_Usage_log'] = np.log1p(df['Data_Usage_GB'])  # log(1 + x) 防止 log(0)
    #scaler = StandardScaler()
    #df['Data_Usage_scaled'] = scaler.fit_transform(df[['Data_Usage_log']])
    return df

#8
@register_feature("Monthly_Charges_log")
def Monthly_Charges_log(df):
    df['Monthly_Charges_log'] = np.log1p(df['Monthly_Charges'])  # log(1 + x) 防止 log(0)
    #scaler = StandardScaler()
    #df['Monthly_Charges_scaled'] = scaler.fit_transform(df[['Monthly_Charges_log']])
    return df

#9
@register_feature("Payment_Method_hot")
def Payment_Method_hot(df):
    df = pd.get_dummies(df, columns=['Payment_Method'], prefix='Payment_Method')
    return df
#["Payment_Method_Electronic check","Payment_Method_Bank transfer (automatic)","Payment_Method_Credit card (automatic)","Payment_Method_Mailed check","Payment_Method_Mailed Unknown"]
#Payment_Method_Electronic check
#Payment_Method_Bank transfer (automatic)
#Payment_Method_Credit card (automatic)
#Payment_Method_Mailed check
#Payment_Method_Mailed Unknown


#10
@register_feature("Customer_Service_Calls_hot")
def Customer_Service_Calls_hot(df):
    df = pd.get_dummies(df, columns=['Customer_Service_Calls'], prefix='Customer_Service_Calls')
    return df
#[
#  "Customer_Service_Calls_0.0",
#  "Customer_Service_Calls_1.0",
#    "Customer_Service_Calls_2.0",
#    "Customer_Service_Calls_3.0",
#    "Customer_Service_Calls_4.0",
#    "Customer_Service_Calls_5.0",
#    "Customer_Service_Calls_15.0",
#    "Customer_Service_Calls_16.0"
#] 

#Customer_Service_Calls_0
#Customer_Service_Calls_1
#Customer_Service_Calls_2
#Customer_Service_Calls_3
#Customer_Service_Calls_4
#Customer_Service_Calls_5

def add_selected_features(df, features_to_use = None):
    if features_to_use is None:
        features_to_use = list(FEATURE_FUNCTIONS.keys())  # 默认全加，用的时候一定要手动录入！！
    
    

    df_cleaned_featured = df.copy()
    for feat in features_to_use:
        if feat in FEATURE_FUNCTIONS:
            df_cleaned_featured = FEATURE_FUNCTIONS[feat](df_cleaned_featured)
        else:
            print(f"[警告] 未注册的特征函数: {feat}")
    print("可用特征函数：", list(FEATURE_FUNCTIONS.keys()))
    added_features = [col for col in df_cleaned_featured if col not in df.columns]
    return df_cleaned_featured,added_features

def list_registered_features():
    return list(FEATURE_FUNCTIONS.keys())
