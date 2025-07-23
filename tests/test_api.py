import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

class TestAPI(unittest.TestCase):

    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_train_and_predict(self):
        train_payload = {
            "architecture": [4, 5, 3],
            "X": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "y": [[0, 1, 0], [1, 0, 0]],
            "alpha": 0.01,
            "iterations": 50
        }

        train_response = client.post("/train", json=train_payload)
        self.assertEqual(train_response.status_code, 200)
        self.assertEqual(train_response.json()["status"], "trained")

        predict_payload = {"inputs": [0.1, 0.2, 0.3, 0.4]}
        predict_response = client.post("/predict", json=predict_payload)
        self.assertEqual(predict_response.status_code, 200)
        outputs = predict_response.json()["outputs"]
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(isinstance(val, float) for val in outputs))

if __name__ == '__main__':
    unittest.main()
