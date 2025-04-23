import os
import sys
import numpy as np
import pandas as pd
from utils.path_helper import get_project_root, get_data_path
from sklearn.ensemble import RandomForestRegressor




class processing_data:
        def __init__(self,df):
                self.df = df.copy()

        def processing_number_data(self, 
                                col_gender = "Gender",
                                col_monthly_charges = "Monthly_Charges",
                                col_contract_type =  "Contract_Type",                                
                                col_data_usage_gb = "Data_Usage_GB"):
            
            self.df['Contract_Type'] = self.df['Contract_Type'].fillna("Unknown")
            self.df['Data_Usage_GB'] = self.df.groupby('Contract_Type')['Data_Usage_GB'].transform(lambda x: x.fillna(x.median()))
            df = self.df
            return df
    
        def processing_amount_data(self,                                    
                                    col_payment_method = "Payment_Method",
                                    col_customer_service_calls = "Customer_Service_Calls"):
 
            self.df['Customer_Service_Calls'] = self.df['Customer_Service_Calls'].fillna("Unknown")
            self.df['Payment_Method'] = self.df['Payment_Method'].fillna("Unknown")
            df = self.df
            return df

        def processing_Median_fill(self,
                                          col_monthly_charges = "Monthly_Charges",
                                          col_tenure_months = "Tenure_Months",
                                          col_age = "Age"):
            self.df['Age'] = self.df.groupby('Contract_Type')['Age'].transform(lambda x: x.fillna(x.median()))
            self.df['Tenure_Months'] = self.df.groupby('Contract_Type')['Tenure_Months'].transform(lambda x: x.fillna(x.median()))
            self.df['Monthly_Charges'] = self.df.groupby('Contract_Type')['Monthly_Charges'].transform(lambda x: x.fillna(x.median()))
            df = self.df
            return df


    ##保留原对象和赋值选一个就好
    #Data_Usage_GB
    #Contract_Type
    #Payment_Method
    #Customer_Service_Calls