import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
from tqdm import tqdm # we realized its taking lot of time for our lasso model training on the 6datasets we have generated, so used tqdm to check the time progress.

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Debugging import
try:
    from model.LassoHomotopy import LassoHomotopy
except ImportError as e:
    print("Import Error Details:")
    print(f"Error: {e}")
    print("Available paths:", sys.path)
    raise


def load_csv_data(file_path):
    """
    Load data from CSV file
    """
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Extract features and target
    X = np.array([[float(v) for k, v in row.items() if k.startswith('x')] for row in data])
    y = np.array([float(row['y']) for row in data])

    return X, y


def visualize_coefficient_transition(X, y, dataset_name, output_dir):
    """
    Optimized visualization of coefficient magnitude transitions
    """
    # Computational Complexity Analysis
    print(f"Dataset: {dataset_name}")
    print(f"Dataset shape: {X.shape}")
    print(f"Total samples: {len(X)}")

    # Ensure we have enough samples for splitting
    if len(X) < 10:
        print(f"Warning: Dataset {dataset_name} is too small for meaningful analysis.")
        return

    # Split data
    split_point = max(1, int(len(X) * 0.8))  # Ensure at least one sample in initial set
    X_initial = X[:split_point]
    y_initial = y[:split_point]
    X_online = X[split_point:]
    y_online = y[split_point:]

    def compute_magnitude_changes(initial_coeffs, samples):

        # Preallocate array for magnitude changes
        magnitude_changes = np.zeros((len(samples), len(initial_coeffs)))

        # Cumulative data preparation
        X_cumulative = np.vstack([X_initial, samples])
        y_cumulative = np.concatenate([y_initial, samples[:, -1]])

        # Single model fitting with all samples
        try:
            incremental_model = LassoHomotopy(lambda_penalty=0.8)
            incremental_results = incremental_model.fit(X_cumulative, y_cumulative)
            incremental_coeffs = incremental_results.get_coefficients()

            # Compute magnitude changes
            magnitude_changes = np.abs(incremental_coeffs - initial_coeffs)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return np.zeros_like(magnitude_changes)

        return magnitude_changes

    # Initial model training
    try:
        initial_model = LassoHomotopy(lambda_penalty=0.7)
        initial_results = initial_model.fit(X_initial, y_initial)
        initial_coeffs = initial_results.get_coefficients()
    except Exception as e:
        print(f"Error in initial model training for {dataset_name}: {e}")
        return

    # Batch processing of online samples
    batch_size = max(1, min(100, len(X_online)))  # Ensure at least one sample per batch
    batched_online_samples = [
        X_online[i:i + batch_size]
        for i in range(0, len(X_online), batch_size)
    ]

    # Efficient magnitude change computation
    all_magnitude_changes = []
    for batch in tqdm(batched_online_samples, desc=f"Processing {dataset_name}"):
        batch_changes = compute_magnitude_changes(initial_coeffs, batch)
        all_magnitude_changes.append(batch_changes)

    # Combine results
    if not all_magnitude_changes:
        print(f"No magnitude changes computed for {dataset_name}")
        return

    magnitude_trajectories = np.vstack(all_magnitude_changes)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Visualization
    plt.figure(figsize=(15, 10))

    # Bar plot of average magnitude changes
    plt.clf()  # Clear previous figure
    avg_magnitude_changes = np.mean(magnitude_trajectories, axis=0)
    plt.bar(
        range(len(avg_magnitude_changes)),
        avg_magnitude_changes,
        align='center',
        alpha=0.7
    )

    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Average Coefficient Magnitude Change', fontsize=12)
    plt.title(f'Coefficient Magnitude Changes - {dataset_name}', fontsize=14)
    plt.xticks(range(len(avg_magnitude_changes)),
               [f'Feature {i + 1}' for i in range(len(avg_magnitude_changes))])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save bar plot
    bar_plot_path = os.path.join(output_dir, f'{dataset_name}_coeff_bar.png')
    plt.savefig(bar_plot_path, bbox_inches='tight')
    plt.close()

    # Heatmap visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(
        magnitude_trajectories.T,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    plt.colorbar(label='Magnitude Change')
    plt.xlabel('Online Sample Batch Index', fontsize=12)
    plt.ylabel('Feature Index', fontsize=12)
    plt.title(f'Coefficient Magnitude Changes - {dataset_name}', fontsize=14)
    plt.tight_layout()

    # Save heatmap
    heatmap_path = os.path.join(output_dir, f'{dataset_name}_coeff_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    # Print some statistics
    print("\nMagnitude Change Statistics:")
    print("Mean magnitude changes:", np.mean(magnitude_trajectories, axis=0))
    print("Max magnitude changes:", np.max(magnitude_trajectories, axis=0))


def process_all_datasets():

    #Process and visualize all datasets in the test_data directory

    # Find data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset
    processed_datasets = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            try:
                file_path = os.path.join(data_dir, filename)

                # Load data
                X, y = load_csv_data(file_path)

                # Visualize dataset
                visualize_coefficient_transition(X, y, filename[:-4], output_dir)

                processed_datasets += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nProcessed {processed_datasets} datasets.")


# Main execution
if __name__ == "__main__":
    process_all_datasets()
