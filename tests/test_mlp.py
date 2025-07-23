import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from mlp import MLP

class TestML(unittest.TestCase):

    def setUp(self):
        self.mlp = MLP([784, 64, 32, 10])

    def test_output_shape(self):
        input_sample = np.random.uniform(-1, 1, 784)
        output = self.mlp.predict(input_sample)
        self.assertEqual(output.shape[0], 10, "La sortie doit avoir 10 neurones (10 classes).")

    def test_output_sum_softmax(self):
        input_sample = np.random.uniform(-1, 1, 784)
        output = self.mlp.predict(input_sample)
        self.assertAlmostEqual(sum(output), 1.0, places=3,
                               msg="Softmax devrait donner une somme proche de 1.0")

if __name__ == '__main__':
    unittest.main()
