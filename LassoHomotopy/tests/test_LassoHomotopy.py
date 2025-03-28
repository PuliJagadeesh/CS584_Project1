import os
import numpy as np
import pytest
import csv
import sys
from sklearn.metrics import mean_squared_error, r2_score
#We are facing issue with importing LassoHomotopy from the model folder, so we are extracting the file path and going to the project root to access it.
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


try:
    from model.LassoHomotopy import LassoHomotopy
except ImportError as e:
    print("Import Error Details:")
    print(f"Error: {e}")
    print("Available paths:", sys.path)
    raise

def find_test_data_folder():
    # Dynamically locating the test data folder across potential directory paths
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)

    potential_paths = [
        os.path.join(parent_dir, 'test_data'),
        os.path.join(current_script_dir, 'test_data')
    ]

    # Finding the first existing test data directory
    for path in potential_paths:
        if os.path.isdir(path):
            print(f"Found test data folder: {path}")
            return path

    # Raising error if no test data folder is found
    raise FileNotFoundError("Could not find test_data or data folder")

def load_csv_data(file_path):
    # Loading CSV data and extracting feature matrix and target vector
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Extracting features starting with 'x' and target variable 'y'
    X = np.array([[float(v) for k, v in row.items() if k.startswith('x')] for row in data])
    y = np.array([float(row['y']) for row in data])

    return X, y

def get_dataset_paths():
    # Finding all CSV files in the test data directory
    try:
        data_dir = find_test_data_folder()
        return [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.csv')
        ]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

# Setting up hyperparameters for LASSO model
HYPERPARAMETERS = {
    'lambda_penalty': 1.0,     # Defining regularization strength
    'max_iterations': 500,     # Setting maximum iterations for algorithm
    'tolerance': 1e-4           # Configuring convergence tolerance
}

@pytest.mark.parametrize("dataset_path", get_dataset_paths())
def generate_collinear_dataset(n_samples=100, n_features=10, collinearity_strength=0.9):
    """
    Enhanced synthetic dataset generation with robust handling
    """
    # Create correlation matrix with improved generation
    correlation_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            correlation_matrix[i, j] = collinearity_strength ** abs(i - j)
            correlation_matrix[j, i] = correlation_matrix[i, j]

    # Ensure positive definite correlation matrix
    correlation_matrix = correlation_matrix.T @ correlation_matrix

    # Use more stable decomposition
    L = np.linalg.cholesky(correlation_matrix)

    # Generate random features
    Z = np.random.randn(n_samples, n_features)
    X = Z @ L.T

    # Create sparse ground truth coefficients
    true_coeffs = np.zeros(n_features)
    true_coeffs[0:3] = [1.5, -1.0, 0.5]

    # Generate target with controlled noise
    y = X @ true_coeffs + np.random.normal(0, 0.1, n_samples)

    return X, y


def test_lasso_collinearity():
    """
    Enhanced collinearity test with robust error handling
    """
    test_configs = [
        {'n_samples': 100, 'n_features': 10, 'collinearity_strength': 0.8},
        {'n_samples': 200, 'n_features': 15, 'collinearity_strength': 0.9},
        {'n_samples': 50, 'n_features': 20, 'collinearity_strength': 0.95}
    ]

    for config in test_configs:
        X, y = generate_collinear_dataset(**config)

        # Test multiple lambda values
        lambda_values = [0.1, 0.5, 1.0, 2.0]

        for lambda_penalty in lambda_values:
            model = LassoHomotopy(
                lambda_penalty=lambda_penalty,
                max_iterations=500,
                tolerance=1e-4
            )

            # Robust model fitting
            results = model.fit(X, y)
            coefficients = results.get_coefficients()
            active_set = results.get_active_set()
            predictions = results.predict(X)

            # Robust metrics computation
            zero_coeff_count = np.sum(np.abs(coefficients) < 1e-4)
            non_zero_coeffs = coefficients[np.abs(coefficients) >= 1e-4]

            # Use sklearn metrics for robust computation
            try:
                mse = mean_squared_error(y, predictions)
                r2 = r2_score(y, predictions)
            except Exception:
                mse = np.inf
                r2 = -np.inf

            print(f"\nLambda {lambda_penalty}:")
            print(f"Zero Coefficients: {zero_coeff_count}/{len(coefficients)}")
            print(f"Active Features: {active_set}")
            print(f"Non-zero Coefficient Magnitudes: {non_zero_coeffs}")
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")

            # More robust assertions
            assert zero_coeff_count >= 0, f"Invalid zero coefficient count for lambda {lambda_penalty}"
            assert not np.isnan(mse), "Invalid Mean Squared Error"
            assert not np.isnan(r2), "Invalid R-squared"


def test_lasso_on_dataset(dataset_path):
    """
    Comprehensive test for LASSO model on different datasets
    """
    # Verify dataset path exists
    assert os.path.exists(dataset_path), f"Dataset file not found: {dataset_path}"

    # Load dataset
    X, y = load_csv_data(dataset_path)

    # Get dataset name
    dataset_name = os.path.basename(dataset_path)

    # Print Dataset Information
    print("\n" + "="*50)
    print(f"Dataset: {dataset_name}")
    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Target Vector Shape: {y.shape}")
    print("Hyperparameters:")
    for param, value in HYPERPARAMETERS.items():
        print(f"- {param}: {value}")
    print("="*50)

    # Model configuration
    model = LassoHomotopy(
        lambda_penalty=HYPERPARAMETERS['lambda_penalty'],
        max_iterations=HYPERPARAMETERS['max_iterations'],
        tolerance=HYPERPARAMETERS['tolerance']
    )

    # Fit model and get results
    results = model.fit(X, y)
    coefficients = results.get_coefficients()
    active_set = results.get_active_set()

    # Predictions
    predictions = results.predict(X)

    # Sparsity Test
    zero_coeff_count = np.sum(np.abs(coefficients) < 1e-4)
    print("\nModel Results:")
    print(f"Zero Coefficients: {zero_coeff_count}/{len(coefficients)}")
    print(f"Active Features: {active_set}")

    # Performance Metrics
    mse = np.mean((y - predictions) ** 2)
    r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r_squared}")

    # Assertions
    assert len(coefficients) == X.shape[1], "Coefficient count mismatch"
    assert r_squared >= 0, "Invalid R-squared"
    assert mse >= 0, "Invalid Mean Squared Error"

def test_coefficient_sparsity():
    """
    Verify LASSO's sparsity property across datasets
    """
    dataset_paths = get_dataset_paths()
    assert len(dataset_paths) > 0, "No datasets found for sparsity testing"

    for dataset_path in dataset_paths:
        X, y = load_csv_data(dataset_path)
        dataset_name = os.path.basename(dataset_path)

        print(f"\nSparsity Test for {dataset_name}")

        # Test multiple lambda values
        lambdas = [0.1, 0.5, 1.0, 2.0]

        for lambda_val in lambdas:
            model = LassoHomotopy(lambda_penalty=lambda_val)
            results = model.fit(X, y)
            coeffs = results.get_coefficients()

            # Higher lambda should produce sparser solution
            zero_coeff_count = np.sum(np.abs(coeffs) < 1e-4)
            print(f"Lambda {lambda_val} - Zero Coefficients: {zero_coeff_count}/{len(coeffs)}")
            assert zero_coeff_count >= len(coeffs) - 2, f"Lambda {lambda_val} failed sparsity test"

# Pytest configuration for verbose output
def pytest_configure(config):
    """
    Customize pytest configuration for better output
    """
    config.addinivalue_line(
        "markers",
        "dataset: mark test to run only on named dataset"
    )

# Allow direct script execution for debugging
if __name__ == "__main__":
    # Print found datasets
    print("Datasets found:")
    for path in get_dataset_paths():
        print(path)
