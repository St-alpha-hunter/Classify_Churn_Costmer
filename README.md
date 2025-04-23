##### PROJECT NAME #####
------------Churn Prediction
This project aims to predict customer churn through machine learning models, 
support multi-model comparison, automatic feature screening, model parameter 
adjustment and evaluation, and integrate commonly used models with automatic saving mechanisms.

###### PROJECT STRUCTURE ######
```text
PROJECT STRUCTURE
.
├── data/                    # store raw data
│   ├── TestSet.csv
│   └── TrainfSet.csv
├── utils/                  
│   ├── config.py            # global variables
│   └── path_helper.py       # path management
├── PreAnalysis/
│   └── Preprocess.py        # process and visualize before feature selection
├── pipeline/
│   └── pipeline.py          # data cleaning
├── train_model/
│   └── train.py             # train and save model
├── features_project/
│   ├── features_meatures.py
│   └── features.py          # define features
├── evaluate_model/         # evaluation scripts
├── notebooks/              # Jupyter notebooks
├── models_saved/           # auto-saved models
├── main.py                 # entry point
└── README.md               # project instructions
```


###### INSTALLMEMT --- Environment ######
1.Clone the Repository
  git clone https://github.com/St-alpha-hunter/Classify_Churn_Costmer
  cd Classify_Churn_Costmer

2.Create Virtual Environment (Recommended)
  #Using venv
  python -m venv venv
  source venv/bin/activate   # On Windows use: venv\Scripts\activate

3.pip install -r requirements.txt

4.python main.py

###### 📦 Project Structure Overview ######

# 💠 Data Preprocessing + Feature Engineering
df_cleaned = preprocess_data(raw_df)

# 💠 Feature Selection (Correlation + Multicollinearity filtering)
selected_features = FeatureAnalysis(
    df_cleaned, 
    features=added_features, 
    target_col="Churn"
).run()

# 💠 Model Training + Auto Param Loading + Model Saving
after_trained_model,X_train, X_test, y_train, y_test = train_model_classify(df_cleaned_features = df_cleaned_features_1, model_config = best_model_configs["RandomForest"], final_features = final_features)

# 💠 Model Loading + Prediction + Evaluation
model = joblib.load("models_saved/rf_model.pkl")
ev = evaluate(model, X_test, y_test)
ev.evaluate_classify_model(average="weighted")

# 💠 bouns: model Comparison (with optional SMOTEENN oversampling + class_weight balancing)
compare_models_f1(
    X=df_cleaned[final_features], 
    y=df_cleaned["Churn"], 
    use_smoteenn=True, 
    average="binary"  # Options: "binary", "macro", "weighted"
)

###### Improvement/TODO ######
1. do some front-end development (using steamlit)
2. publish this project on some Cloud-platform
3. develop logger module to record the experiments 
4. others

