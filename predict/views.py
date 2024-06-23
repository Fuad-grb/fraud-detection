from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionSerializer
import joblib
import numpy as np
import pandas as pd

model = joblib.load('credit_card_fraud_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            features = serializer.validated_data['features']
            features_df = pd.DataFrame([features], columns=preprocessor.feature_names_in_)
            features_preprocessed = preprocessor.transform(features_df)
            prediction = model.predict(features_preprocessed)
            return Response({'prediction': prediction.tolist()}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
