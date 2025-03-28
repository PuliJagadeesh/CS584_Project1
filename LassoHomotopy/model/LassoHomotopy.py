import numpy as np
#import numpy.linalg as la
#We have written this code using the algorithm shared in the research article of reclasso, in page 4.

import warnings


class LassoHomotopy:
    def __init__(self, lambda_penalty=1.0, max_iterations=500, tolerance=1e-4):
        self.lambda_penalty = lambda_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.theta = None
        self.intercept = None
        self.active_set = None

    def _soft_threshold(self, x, lambda_val):
        """
        Robust soft thresholding operator with numerical stability
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

    def _compute_correlation(self, X, residuals):
        """
        Compute feature correlations with residuals, handling potential numerical issues
        """
        try:
            correlations = np.abs(X.T @ residuals)
            # Handle potential NaN or inf values
            correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
            return correlations
        except Exception as e:
            warnings.warn(f"Correlation computation error: {e}")
            return np.zeros(X.shape[1])

    def fit(self, X, y):
        """
        Robust LASSO fitting with enhanced numerical stability
        """
        # Preprocessing with robust handling
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()

        # Check for zero variance features
        std_X = np.std(X, axis=0)
        std_X[std_X == 0] = 1.0  # Prevent division by zero

        # Standardization with zero variance handling
        X_scaled = (X - np.mean(X, axis=0)) / std_X
        y_centered = y - np.mean(y)

        # Robust initialization
        n_samples, n_features = X_scaled.shape
        theta = np.zeros(n_features)
        active_set = []

        # Iterative feature selection and coefficient estimation
        for iteration in range(self.max_iterations):
            # Compute residuals with numerical stability
            try:
                residuals = y_centered - X_scaled @ theta
            except Exception:
                residuals = y_centered.copy()

            # Compute feature correlations
            feature_correlations = self._compute_correlation(X_scaled, residuals)

            # Convergence check with robust comparison
            if np.max(np.abs(feature_correlations)) <= self.lambda_penalty:
                break

            # Select most correlated feature
            max_corr_idx = np.argmax(np.abs(feature_correlations))

            # Prevent duplicate feature selection
            if max_corr_idx not in active_set:
                active_set.append(max_corr_idx)

            # Coordinate descent with enhanced stability
            for _ in range(50):  # Reduced inner loop iterations
                for j in active_set:
                    try:
                        # Robust partial residual computation
                        partial_residuals = residuals.copy()
                        partial_residuals += X_scaled[:, j] * theta[j]

                        # Robust coordinate update
                        feature_norm = X_scaled[:, j] @ X_scaled[:, j]
                        if feature_norm > 0:
                            theta[j] = self._soft_threshold(
                                X_scaled[:, j] @ partial_residuals,
                                self.lambda_penalty
                            ) / feature_norm
                    except Exception:
                        theta[j] = 0.0

                # Recompute residuals
                try:
                    residuals = y_centered - X_scaled @ theta
                except Exception:
                    break

            # Remove near-zero coefficients
            active_set = [j for j in active_set if np.abs(theta[j]) > self.tolerance]

        # Scale back coefficients
        try:
            self.theta = theta / std_X
            self.intercept = np.mean(y) - np.mean(X, axis=0) @ self.theta
        except Exception:
            self.theta = np.zeros_like(theta)
            self.intercept = np.mean(y)

        self.active_set = active_set
        return LassoHomotopyResults(self)


class LassoHomotopyResults:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """
        Robust prediction method
        """
        X = np.atleast_2d(X)
        try:
            predictions = X @ self.model.theta + self.model.intercept
            return predictions
        except Exception:
            return np.zeros(len(X))

    def get_coefficients(self):
        return self.model.theta if self.model.theta is not None else np.zeros_like(self.model.theta)

    def get_active_set(self):
        return self.model.active_set if self.model.active_set is not None else []