# IU Programming with Python Assignment

This project implements a function matching application using Python.

## Functionality

The application compares four training functions with 50 ideal functions using the least squares method. Test points are assigned to selected ideal functions based on a deviation threshold.

## Technologies

- Python
- Pandas
- NumPy
- SQLAlchemy
- SQLite
- Bokeh
- unittest

## Project Structure

- `main.py` – Main application, database storage and visualization
- `data_loader.py` – CSV loading and validation
- `function_matching.py` – Function matching and deviation calculations
- `test_data_loader.py` – Data loading tests
- `test_function_matching.py` – Function matching tests

## Running the Application

Install the required Python packages:

```bash
pip install pandas numpy sqlalchemy bokeh
```

Place the provided CSV datasets in a `data` directory and run:

```bash
python main.py
```

Run the unit tests:

```bash
python -m unittest discover -v
```

The application stores the results in a SQLite database and generates interactive Bokeh visualizations.