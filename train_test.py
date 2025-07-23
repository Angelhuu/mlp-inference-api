# train_test.py

import requests

url = "http://127.0.0.1:8000/train"

data = {
    "architecture": [2, 3, 1],
    "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "y": [[0], [1], [1], [0]],
    "alpha": 0.1,
    "iterations": 10000
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
