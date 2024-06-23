import requests

url = 'http://localhost:8000/predict/'

data = {
    'features': [0, 1577836818, 1234567890123456, 1, 2, 107.23, 3, 4, 1, 
                 5, 6, 7, 12345, 40.7128, -74.0060, 8000000, 8, 292192800,
                 1577836818, 40.7128, -74.0060]  
}

response = requests.post(url, json=data)

print('Status Code:', response.status_code)

print('Response Text:', response.text)

try:
    prediction = response.json()
    print('Prediction:', prediction)
except requests.exceptions.JSONDecodeError as e:
    print('JSON Decode Error:', e)
