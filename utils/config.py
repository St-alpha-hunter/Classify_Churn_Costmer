from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



###############
#规定列名#
###############

col_costomer = "CustomerID",
col_age = "Age",
col_gender = "Gender",
col_tenure_months = "Tenure_Months",
col_contract_type = "Contract_Type",
col_monthly_charges = "Monthly_Charges",
col_data_usage_gb = "Data_Usage_GB",
col_payment_method = "Payment_Method",
col_customer_service_calls = "Customer_Service_Calls",
col_churn = "Churn"



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

best_model_configs = {
    "Logistic": {
        "model_cls": LogisticRegression,
        "params": {
            "C": 10,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000
        }
    },
    "RandomForest": {
        "model_cls": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "class_weight": "balanced"
        }
    },
    "XGBoost": {
        "model_cls": XGBClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
    },
    "GBDT": {
        "model_cls": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05
        }
    },
    "SVM": {
        "model_cls": SVC,
        "params": {
            "C": 0.1,
            "kernel": "linear",
            "probability": True  # 可选项，若你需要 predict_proba
        }
    }
}
