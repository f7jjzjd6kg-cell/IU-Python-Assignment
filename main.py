from collections import Counter

import pandas as pd
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from sqlalchemy import create_engine, text

from data_loader import DataLoader, DataValidationError, ValidatedDataLoader
from function_matching import FunctionAnalyzer


def save_results(engine, training_data, ideal_data, test_data, matches):
    """Store input datasets and matched test points in SQLite."""
    tables = {
        "training_data": training_data,
        "ideal_data": ideal_data,
        "test_data": test_data,
        "test_matches": pd.DataFrame(
            matches, columns=["x", "y", "ideal_function", "deviation"]
        ),
    }
    for name, data in tables.items():
        data.to_sql(name, engine, if_exists="replace", index=False)

    with engine.connect() as connection:
        for name in tables:
            count = connection.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            print(f"{name}: {count} records")


def create_plot(training_data, ideal_data, matches, training_number, ideal_number, width, height):
    """Draw one training function, its ideal function and assigned test points."""
    plot = figure(
        title=f"Training y{training_number} vs Ideal {ideal_number}",
        x_axis_label="x", y_axis_label="y", width=width, height=height,
    )
    plot.line(training_data["x"], training_data[f"y{training_number}"],
              color="blue", line_width=2, legend_label="Training")
    plot.line(ideal_data["x"], ideal_data[f"y{ideal_number}"],
              color="red", line_width=2, legend_label="Ideal")
    selected = [item for item in matches if item["ideal_function"] == ideal_number]
    plot.scatter(x=[item["x"] for item in selected],
                 y=[item["y"] for item in selected],
                 color="green", size=7, marker="circle", legend_label="Matched test points")
    return plot


def main():
    """Run the analysis, save its results and create Bokeh plots."""
    try:
        training_data = ValidatedDataLoader("data/train.csv").load_data()
        ideal_data = DataLoader("data/ideal.csv").load_data()
        test_data = DataLoader("data/test.csv").load_data()
    except (DataValidationError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error loading data: {error}") from error

    analyzer = FunctionAnalyzer(training_data, ideal_data)
    best_functions = analyzer.select_functions()
    matches = analyzer.match_test_points(test_data)

    counts = Counter((item["x"], item["y"]) for item in matches)
    print("Selected ideal functions:", best_functions)
    print("Total test points:", len(test_data))
    print("Valid matches:", len(matches))
    print("Unique matched test points:", len(counts))
    print("Unmatched test points:", len(test_data) - len(counts))
    print("Test points with multiple matches:", sum(count > 1 for count in counts.values()))

    engine = create_engine("sqlite:///iu_assignment.db")
    save_results(engine, training_data, ideal_data, test_data, matches)
    print("Matches stored in database:", len(matches))

    first_ideal = best_functions[1]
    output_file("function_comparison.html")
    show(create_plot(training_data, ideal_data, matches, 1, first_ideal, 900, 500))

    plots = [create_plot(training_data, ideal_data, matches, number, ideal, 600, 350)
             for number, ideal in best_functions.items()]
    output_file("all_function_comparisons.html")
    show(gridplot(plots, ncols=2))


if __name__ == "__main__":
    main()
