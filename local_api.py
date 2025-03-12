import requests
import json

# URL for the API (local)
base_url = "http://localhost:8000"

# Test the GET endpoint
response = requests.get(f"{base_url}/")
print(f"Status Code: {response.status_code}")
print(f"Result: {response.json().get('greeting')}")

# Sample data for prediction
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Test the POST endpoint
response = requests.post(f"{base_url}/data/", json=data)
print(f"Status Code: {response.status_code}")
print(f"Result: {response.json().get('result')}")