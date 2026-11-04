# IU Programming with Python – Written Assignment

This repository contains the Python implementation of the IU Programming with Python written assignment.

## Project Overview

The application loads training, ideal, and test datasets from CSV files. It selects four ideal functions using the least-squares criterion, assigns test points according to the specified deviation threshold, stores the data and results in SQLite, and generates interactive visualizations using Bokeh.

## Main Components

- `data_loader.py`: CSV loading and data validation
- `function_matching.py`: Function selection and test-point matching
- `main.py`: Program execution, SQLite persistence, and visualization
- `test_data_loader.py`: Unit tests for data loading
- `test_function_matching.py`: Unit tests for function matching

## Running the Tests

Install the required dependencies and run the tests from the project directory:

```bash
python -m unittest discover -v
```

## Dataset

The project uses the CSV datasets supplied for the IU course assignment. Place `train.csv`, `ideal.csv`, and `test.csv` in a folder named `data` before running the main program.
