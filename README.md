## Credit Card Fraud Detection API
A machine learning system that detects fraudulent credit card transactions and serves predictions through a REST API built with Django, containerized with Docker.
### What it does
The model takes a raw transaction record and returns a fraud/legitimate classification. The full pipeline — preprocessing, resampling, training, and serving — is implemented end to end. Experiment tracking is handled via MLflow.
### The problem and approach
Fraud detection is a heavily imbalanced classification problem. In this dataset, fraudulent transactions make up roughly 0.35% of all records, which means a model that predicts "legitimate" for everything achieves 99.7% accuracy while being completely useless. The meaningful metric is F1-score on the fraud class.
To address the imbalance, I used ADASYN (Adaptive Synthetic Sampling), which generates synthetic minority samples adaptively — focusing on harder-to-classify fraud cases rather than oversampling uniformly like SMOTE does. The classifier is XGBoost, tuned via RandomizedSearchCV over 20 iterations with 3-fold cross-validation, optimizing for F1.
### Dataset
Kaggle — Credit Card Fraud Detection (Sparkov)
Simulated transactions from 2019–2020 across 800+ customers and 693 merchants. Each record contains transaction amount, timestamp, merchant category, cardholder demographics, and geolocation of both the cardholder and the merchant.
### Model performance
Overall accuracy is 99.73%, but that number is dominated by the majority class and means little. What matters:

Fraud class precision: 0.69
Fraud class recall: 0.43
Fraud class F1: 0.53

Recall being lower than precision means the model misses more fraudulent transactions than it falsely flags — a tradeoff that could be shifted by lowering the decision threshold at the cost of more false positives.
### Quickstart
With Docker:
#### bash 
git clone https://github.com/Fuad-grb/fraud-detection.git
cd fraud-detection
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
### Locally:
#### bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
## API
Send a POST request to /predict/ with a features array of 21 values in the original column order (transaction timestamp as unix, card number, merchant, category, amount, cardholder name/gender/DOB/address/job, zip, lat/long, city population, merchant lat/long).
bashcurl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 1577836818, 2703186000000000, "fraud_Rippin, Kub and Mann",
       "misc_net", 4.97, "Jennifer", "Banks", "F", "561 Perry Cove",
       "Moravian Falls", "NC", 28654, 36.0788, -81.1781, 3495,
       "Psychologist, counselling", "1988-03-09", 1325376018, 36.011293, -82.048315]}'
Response: {"prediction": [0]} — 0 is legitimate, 1 is fraud.
Retraining
Requires fraudTrain.csv and fraudTest.csv from Kaggle. Start an MLflow server on port 5000, then run python train.py. The script logs parameters and metrics to MLflow and saves the model and preprocessor artifacts locally.
