import numpy as np
import pandas as pd


def calculate_sse(training_values, ideal_values):
    """Return the sum of squared differences."""
    return np.sum((training_values - ideal_values) ** 2)


def calculate_max_deviation(training_values, ideal_values):
    """Return the largest absolute difference."""
    return np.max(np.abs(training_values - ideal_values))


def calculate_threshold(max_deviation):
    """Return the permitted deviation for test points."""
    return np.sqrt(2) * max_deviation


def is_valid_match(test_y, ideal_y, threshold):
    """Check whether a test point is within the threshold."""
    return abs(test_y - ideal_y) <= threshold


class FunctionAnalyzer:
    """Select ideal functions and assign matching test points."""

    def __init__(self, training_data, ideal_data):
        self.training_data = training_data
        self.ideal_data = ideal_data
        self.best_functions = {}
        self.thresholds = {}

    def select_functions(self):
        """Select the ideal function with the smallest SSE for each training series."""
        combined = pd.merge(self.training_data, self.ideal_data, on="x", validate="one_to_one")
        if len(combined) != len(self.training_data):
            raise ValueError("Training and ideal datasets have different x values.")

        for number in range(1, 5):
            training = combined[f"y{number}_x"]
            errors = {}
            for candidate in range(1, 51):
                column = f"y{candidate}_y" if candidate <= 4 else f"y{candidate}"
                errors[candidate] = calculate_sse(training, combined[column])

            selected = min(errors, key=errors.get)
            column = f"y{selected}_y" if selected <= 4 else f"y{selected}"
            max_deviation = calculate_max_deviation(training, combined[column])
            self.best_functions[number] = selected
            self.thresholds[number] = calculate_threshold(max_deviation)

            print(f"Training y{number}: Best ideal function = {selected}, SSE = {errors[selected]:.2f}")
            print(f"Maximum deviation = {max_deviation:.4f}, Threshold = {self.thresholds[number]:.4f}")

        return self.best_functions

    def match_test_points(self, test_data):
        """Return all valid test-point assignments with their deviations."""
        if not self.best_functions:
            raise ValueError("Select ideal functions before matching test points.")

        matches = []
        for _, point in test_data.iterrows():
            ideal_rows = self.ideal_data.loc[np.isclose(self.ideal_data["x"], point["x"])]
            if ideal_rows.empty:
                print(f"No ideal value found for x = {point['x']}")
                continue

            ideal_row = ideal_rows.iloc[0]
            for training_number, ideal_number in self.best_functions.items():
                ideal_y = ideal_row[f"y{ideal_number}"]
                deviation = abs(point["y"] - ideal_y)
                if is_valid_match(point["y"], ideal_y, self.thresholds[training_number]):
                    matches.append({
                        "x": point["x"],
                        "y": point["y"],
                        "ideal_function": ideal_number,
                        "deviation": deviation,
                    })
        return matches
