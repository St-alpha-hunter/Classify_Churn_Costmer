import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns       # sns 是 seaborn 的常用别名
import matplotlib.pyplot as plt  # plt 是 matplotlib.pyplot 的常用别名
from utils.path_helper import get_project_root


class preprocess:
    def __init__(self,df):
            self.df = df

    def picture_analysis(self,                       
                            col_age = "Age",
                            col_gender = "Gender",
                            col_tenure_months = "Tenure_Months",
                            col_contract_type = "Contract_Type",
                            col_monthly_charges = "Monthly_Charges",
                            col_data_usage_gb = "Data_Usage_GB",
                            col_payment_method = "Payment_Method",
                            col_customer_service_calls = "Customer_Service_Calls",):
        
        #age
        sns.kdeplot(self.df['Age'],fill = True)
        plt.show()
        #Gender
        sns.histplot(data=self.df, x='Gender',multiple='dodge')
        plt.show()

        ##Tenure_Months
        sns.kdeplot(self.df['Tenure_Months'],fill = True)
        plt.show()
        
        ##Contract_Type
        sns.histplot(data=self.df, x='Contract_Type',multiple='dodge')
        plt.show()

        #Monthly_Charges
        sns.kdeplot(self.df['Monthly_Charges'],fill = True)
        plt.show()

        #Data_Usage_GB
        sns.kdeplot(self.df['Data_Usage_GB'],fill = True)
        plt.show()

        #Payment_Method
        sns.histplot(data=self.df, x='Payment_Method', multiple='dodge')
        # 设置 x 轴标签旋转 45 度
        plt.xticks(rotation=45)
        plt.show()

        #Calls
        sns.histplot(data=self.df, x='Customer_Service_Calls', multiple='layer')
        # 设置 x 轴标签旋转 45 度
        plt.xticks(rotation=45)
        plt.show()


    def missingdata_count(self):
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            print(f"{column}: 缺失值数量 = {missing_count}")


    def detect_extreme(self):
         self.df = self.df[self.df['Customer_Service_Calls'] != -2]
         self.df = self.df[self.df['Age'] > 0]
         self.df = self.df[self.df['Tenure_Months'] > 0]
         self.df = self.df[self.df['Monthly_Charges'] > 0]
         self.df = self.df[self.df['Monthly_Charges'] > 0]
         df = self.df[self.df['Data_Usage_GB'] > 0]
         return df
