import requests

BASE_URL = "https://mlp-inference-api.azurewebsites.net"

def test_train():
    print("➡️  Test /train")
    payload = {"architecture": [2, 4, 1], "alpha": 0.1, "iterations": 5000}
    response = requests.post(f"{BASE_URL}/train", json=payload)
    print("Status:", response.status_code)
    print("Response:", response.json())
    print()

def test_predict():
    print("➡️  Test /predict")
    payload = {"inputs": [1.0, 0.0]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Status:", response.status_code)
    print("Response:", response.json())
    print()

if __name__ == "__main__":
    test_train()
    test_predict()
