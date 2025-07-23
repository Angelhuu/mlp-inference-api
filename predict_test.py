# predict_test.py

import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "inputs": [1, 0]  # modifie ce vecteur pour tester d'autres entrées
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Prediction:", response.json())

