import os
import tempfile
import unittest

from data_loader import DataLoader, DataValidationError, ValidatedDataLoader


class TestDataLoader(unittest.TestCase):
    """Check CSV loading and validation errors."""

    def test_load_training_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write("x,y1,y2,y3,y4\n1,2,3,4,5\n")
            data = ValidatedDataLoader(path).load_data()
            self.assertEqual(data.shape, (1, 5))

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                DataLoader(os.path.join(folder, "missing.csv")).load_data()

    def test_empty_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "empty.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write("x,y1,y2,y3,y4\n")
            with self.assertRaises(DataValidationError):
                ValidatedDataLoader(path).load_data()

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write("x,y1,y2,y3\n1,2,3,4\n")
            with self.assertRaises(DataValidationError):
                ValidatedDataLoader(path).load_data()


if __name__ == "__main__":
    unittest.main()
