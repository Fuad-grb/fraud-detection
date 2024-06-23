import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

mlflow.set_tracking_uri("http://localhost:5000")

logged_model = 'runs:/bc4daa3be3d044c39df2607b9a376b58/model'
model = mlflow.sklearn.load_model(logged_model)

preprocessor = joblib.load('preprocessor.pkl')

test_data = pd.read_csv(r'C:\Users\hp\OneDrive\Рабочий стол\unitask2\fraud_detection\fraudTest.csv')

test_data = test_data.drop(columns=['trans_num'])
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time']).astype('int64') / 10**9

X_test_preprocessed = preprocessor.transform(test_data.drop('is_fraud', axis=1))

predictions = model.predict(X_test_preprocessed)
print("Test predictions:")
print(predictions)

output = pd.DataFrame({'prediction': predictions})
output.to_csv(r'C:\Users\hp\OneDrive\Рабочий стол\unitask2\fraud_detection\predictions.csv', index=False)
