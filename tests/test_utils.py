import pickle
import requests
import unittest

from inspaipy import utils 
from utils import categorical_to_numpy, one_hot_encoding, logits_to_one_hot_encoding

# import your module here if the functions are in a separate module
# from your_module import categorical_to_numpy, one_hot_encoding, logits_to_one_hot_encoding

class TestFunctions(unittest.TestCase):

    def test_categorical_to_numpy(self):
        # Test for categorical_to_numpy function
        labels_in = ['dog', 'cat']
        expected_output = np.array([[1, 0], [0, 1]])
        result = categorical_to_numpy(labels_in)
        np.testing.assert_array_equal(result, expected_output)

    def test_one_hot_encoding(self):
        # Test for one_hot_encoding function
        input_array = np.array([0, 1, 2, 1])
        expected_output = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        result = one_hot_encoding(input_array)
        np.testing.assert_array_equal(result, expected_output)

    def test_logits_to_one_hot_encoding(self):
        # Test for logits_to_one_hot_encoding function
        logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        expected_output = np.array([[0, 1], [1, 0], [0, 1]])
        result = logits_to_one_hot_encoding(logits)
        np.testing.assert_array_equal(result, expected_output)

    # The plot_one_image function is a bit tricky to test as it outputs a plot.
    # Generally, visual outputs like plots are manually inspected rather than tested via unit tests.

if __name__ == '__main__':
    unittest.main()


