import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn

train_data = pd.read_csv(r'C:\Users\hp\OneDrive\Рабочий стол\unitask2\fraud_detection\fraudTrain.csv')
test_data = pd.read_csv(r'C:\Users\hp\OneDrive\Рабочий стол\unitask2\fraud_detection\fraudTest.csv')

train_data = train_data.drop(columns=['trans_num'])
test_data = test_data.drop(columns=['trans_num'])
train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time']).astype('int64') / 10**9
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time']).astype('int64') / 10**9
train_data.drop_duplicates(inplace=True)

train_data_sample = train_data.sample(frac=0.05, random_state=42)

X_train = train_data_sample.drop('is_fraud', axis=1)
y_train = train_data_sample['is_fraud'].fillna(train_data_sample['is_fraud'].median())
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud'].fillna(train_data['is_fraud'].median())

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=20, cv=3, scoring='f1', n_jobs=-1, random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud_detection_experiment")

try:
    with mlflow.start_run():
        random_search.fit(X_train_resampled, y_train_resampled)
        best_model = random_search.best_estimator_
        
        predictions = best_model.predict(X_test_preprocessed)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_param("n_estimators", best_model.get_params()['n_estimators'])
        mlflow.log_param("max_depth", best_model.get_params()['max_depth'])
        mlflow.log_param("learning_rate", best_model.get_params()['learning_rate'])
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(best_model, "model")

        print(f'Accuracy: {accuracy}')
        report = classification_report(y_test, predictions)
        print("Classification Report:")
        print(report)

        joblib.dump(best_model, 'credit_card_fraud_model.pkl')

        joblib.dump(preprocessor, 'preprocessor.pkl')

except Exception as e:
    print(f"RandomizedSearchCV failed: {e}")
