# import important labraries
# %% 
import pandas as pd
import numpy as np

# %%
import joblib

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as skpipeline
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


# %%
df = pd.read_csv("European_Bank.csv")
df = df.drop(columns=["Surname", "Year","CustomerId"])
# %%
num_cols = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary'
]
cat_cols = [
    'Geography',
    'Gender',
    'HasCrCard',
    'IsActiveMember'
]
#Data Preprocessing
# %%
num_transformer = skpipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = skpipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

#train_test_split
# %%
X = df.drop(columns=["Exited"])
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 65)
# modelling
# %%
pipeline = imbpipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state = 42)),
    ('model', XGBClassifier(
        gamma=2,
        n_estimators=266,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Model training
# %%
pipeline.fit(X_train, y_train)
#Prediction and evaluation
# %%
y_pred_train = pipeline.predict(X_train)
y_pred_test= pipeline.predict(X_test)

print("train_accuracy:",accuracy_score(y_train,y_pred_train))
print("test_accuracy:",accuracy_score(y_test,y_pred_test))
print("cv:",cross_val_score(pipeline,X_train,y_train,cv=5,scoring="accuracy").mean())

y_prob = pipeline.predict_proba(X_test)[:,1]
auc_score = roc_auc_score(y_test, y_prob)
print("AUC-ROC Score:", auc_score)

f1 = f1_score(y_test,y_pred_test)
print("F1 Score:", f1)

precision = precision_score(y_test, y_pred_test)
print("Precision:", precision)

recall = recall_score(y_test, y_pred_test)
print("Recall",recall)

# save the model
# %%
joblib.dump(pipeline, 'xgb_pipeline.pkl')
# %%
