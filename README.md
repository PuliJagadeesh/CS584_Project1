# Project 1: LASSO Homotopy Regression Implementation

## Project Overview
This project implements the **LASSO (Least Absolute Shrinkage and Selection Operator) Homotopy algorithm**, a powerful regularization technique for linear regression that performs **feature selection** and prevents **overfitting**. You can read about this method in [this paper](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) and the references therein.

---

## File Structure
```
LassoHomotopy/
│
├── model/
│   └── LassoHomotopy.py        # Main LASSO Homotopy implementation
│
├── tests/
│   ├── __init__.py             # Enables Python package testing
│   ├── test_LassoHomotopy.py   # Comprehensive test suite
│
├── test_data/                  # Directory for test datasets
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── ... (other CSV files)
│
├── visualizations/
│   ├── visualizations.py       # Script for plotting results
│
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
```

---

## Features
- **Sparse Feature Selection** – Automatically eliminates irrelevant features.
- **Regularized Linear Regression** – Prevents overfitting.
- **Handles High-Dimensional Datasets** – Works well even with many features.
- **Supports Custom Regularization Strength** – Allows fine-tuning via hyperparameters.

---

## Prerequisites
- Python 3.7+
- NumPy
- Scikit-learn (optional, for metrics)
- Pytest (for testing)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PuliJagadeesh/CS584_Project1/LassoHomotopy.git
cd LassoHomotopy
```

### 2. Create a Virtual Environment (Recommended)
#### Using `venv`
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using `conda`
```bash
conda create -n lassohomotopy python=3.8
conda activate lassohomotopy
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Basic Usage

### Importing the Model
```python
from model.LassoHomotopy import LassoHomotopy

# Create a LASSO Homotopy model
model = LassoHomotopy(
    lambda_penalty=1.0,     # Regularization strength
    max_iterations=500,     # Maximum algorithm iterations
    tolerance=1e-4          # Convergence tolerance
)
```

### Training the Model
```python
# Assuming X (features) and y (target) are numpy arrays
results = model.fit(X, y)

# Get coefficients
coefficients = results.get_coefficients()

# Get active features
active_features = results.get_active_set()

# Make predictions
predictions = results.predict(X_test)
```

---

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run a Specific Test
```bash
pytest tests/test_LassoHomotopy.py
```

---

## Model Details

### What does the model do and when should it be used?
LASSO stands for **Least Absolute Shrinkage and Selection Operator**:
- **Shrinkage** – Reduces high coefficient values, mitigating overfitting.
- **Feature Selection** – Drives irrelevant features' coefficients to zero, effectively performing feature selection.

#### **Model Purpose**
- Performs **regularized linear regression (LASSO)**.
- Handles **high-dimensional datasets**.
- Selects important features by eliminating irrelevant ones.

#### **Use Cases**
- **Predictive modeling** with many potential predictors.
- **Feature selection** in machine learning pipelines.
- **Overfitting prevention** in complex datasets.
- **Handling multicollinearity** in regression problems.
- **Sparse signal recovery** in high-dimensional data.

#### **Ideal Scenarios**
- **Genomics research** (gene selection from large datasets).
- **Financial modeling** (feature selection for stock price prediction).
- **Image processing** (identifying key features in images).
- **Neuroscience data analysis** (selecting significant EEG signals).
- **Econometric studies** (choosing influential economic indicators).

#### **When to Use**
- **Limited training samples**.
- **Many potentially irrelevant features**.
- **Need for interpretable models**.
- **Desire to reduce model complexity**.

---

## Testing Strategy

### **How did you test your model to determine if it is working correctly?**

#### **1. Synthetic Dataset Generation**
- Created **controlled collinear datasets**.
- Varied **sample sizes, feature counts, and correlation strengths**.

#### **2. Key Testing Approaches**
- **Collinearity Test**
  - Verify feature selection in correlated data.
  - Check sparsity across different regularization strengths.
  - Validate prediction accuracy.

- **Sparsity Verification**
  - Ensure coefficients reduce with increased regularization.
  - Confirm elimination of less important features.
  - Track the count of zero coefficients.

- **Performance Metrics**
  - Mean Squared Error (MSE).
  - R-squared calculation.
  - Active feature set analysis.

---

## Parameter Tuning
### **Exposed Hyperparameters**
| Parameter        | Default Value | Description |
|-----------------|--------------|-------------|
| `lambda_penalty` | 1.0 | Regularization strength |
| `max_iterations` | 500 | Maximum number of iterations |
| `tolerance`     | 1e-4 | Convergence tolerance |

---

## Limitations & Future Improvements

### **Are there specific inputs your implementation struggles with?**
- **Choosing an optimal `lambda_penalty`**
  - Model performance is highly dependent on this parameter.
  - **Future Improvement**: Implement **cross-validation** to automatically find the best `lambda_penalty`.

- **Training Time on Large Datasets**
  - Large-scale data processing takes time.
  - **Future Improvement**: Use **GPU acceleration (Kaggle/Colab)** for faster training and better visualization.

> **Dataset Details:** Refer to `Dataset_details.docs` for explanations on dataset selection and evaluation rationale.

---



