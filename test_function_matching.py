import unittest

import numpy as np
import pandas as pd

from function_matching import (
    FunctionAnalyzer,
    calculate_max_deviation,
    calculate_sse,
    calculate_threshold,
    is_valid_match,
)


class TestFunctionMatching(unittest.TestCase):
    """Check numerical calculations and test-point matching."""

    def test_squared_error(self):
        self.assertEqual(calculate_sse(np.array([2, 4, 6]), np.array([1, 3, 5])), 3)

    def test_max_deviation(self):
        self.assertEqual(calculate_max_deviation(np.array([2, 5, 9]), np.array([1, 3, 6])), 3)

    def test_threshold(self):
        self.assertAlmostEqual(calculate_threshold(2), 2 * np.sqrt(2))

    def test_valid_match(self):
        self.assertTrue(is_valid_match(5.5, 5, 1))

    def test_invalid_match(self):
        self.assertFalse(is_valid_match(7, 5, 1))

    def test_match_at_threshold(self):
        self.assertTrue(is_valid_match(6, 5, 1))

    def test_analyzer_selects_closest_function(self):
        training = pd.DataFrame({"x": [0, 1], **{f"y{i}": [i, i + 1] for i in range(1, 5)}})
        ideal = pd.DataFrame({"x": [0, 1], **{f"y{i}": [i, i + 1] for i in range(1, 51)}})
        analyzer = FunctionAnalyzer(training, ideal)
        self.assertEqual(analyzer.select_functions(), {1: 1, 2: 2, 3: 3, 4: 4})

    def test_analyzer_matches_test_point(self):
        training = pd.DataFrame({"x": [0, 1], **{f"y{i}": [i, i + 1] for i in range(1, 5)}})
        ideal = pd.DataFrame({"x": [0, 1], **{f"y{i}": [i, i + 1] for i in range(1, 51)}})
        analyzer = FunctionAnalyzer(training, ideal)
        analyzer.select_functions()
        matches = analyzer.match_test_points(pd.DataFrame({"x": [0], "y": [1]}))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["ideal_function"], 1)
        self.assertEqual(matches[0]["deviation"], 0)


if __name__ == "__main__":
    unittest.main()
