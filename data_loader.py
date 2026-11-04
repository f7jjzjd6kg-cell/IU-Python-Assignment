import os

import pandas as pd


class DataValidationError(Exception):
    """Report invalid input data."""


class DataLoader:
    """Read CSV files."""

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Return the CSV contents as a DataFrame."""
        return pd.read_csv(self.file_path)


class ValidatedDataLoader(DataLoader):
    """Read CSV files and check their basic structure."""

    def load_data(self):
        """Reject empty data and incomplete training files."""
        data = super().load_data()
        if data.empty:
            raise DataValidationError(f"The CSV file {self.file_path} is empty.")

        if os.path.basename(self.file_path) == "train.csv":
            required = {"x", "y1", "y2", "y3", "y4"}
            missing = required - set(data.columns)
            if missing:
                raise DataValidationError(f"Missing required column: {sorted(missing)[0]}")

        return data
