import numpy as np
#import numpy.linalg as la
#We have written this code using the algorithm shared in the research article of reclasso, in page 4.

class LassoHomotopy:
    """
    Recursive LASSO Homotopy Method Implementation
    """

    def __init__(self, lambda_penalty=1.0, max_iterations=500, tolerance=1e-4):
        """
        Initializing the LASSO Homotopy Model
        """
        self.lambda_penalty = lambda_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Tracking the model state
        self.theta = None
        self.intercept = None
        self.active_set = None

    def _soft_threshold(self, x, lambda_val):
        """
        Applying the soft thresholding operator for LASSO
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

    def _compute_correlation(self, X, residuals):
        """
        Computing feature correlations with residuals
        """
        return np.abs(X.T @ residuals)

    def fit(self, X, y):
        """
        Fitting the LASSO model using the Homotopy Method
        """

        # Preprocessing: Standardizing the features and centering the target variable
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()
        n_samples, n_features = X.shape

        # Standardizing features and centering target
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
        y_centered = y - y.mean()

        # Initializing variables
        theta = np.zeros(n_features)
        active_set = []

        # Step 1: Computing the initial correlation
        correlations = self._compute_correlation(X_scaled, y_centered)

        # Step 2: Starting the homotopy iterations
        for iteration in range(self.max_iterations):
            # Computing residuals
            residuals = y_centered - X_scaled @ theta

            # Step 3: Computing feature correlations
            feature_correlations = self._compute_correlation(X_scaled, residuals)

            # Step 4: Checking for convergence
            if np.max(feature_correlations) <= self.lambda_penalty:
                break

            # Step 5: Selecting the most correlated feature
            max_corr_idx = np.argmax(feature_correlations)
            active_set.append(max_corr_idx)

            # Step 6: Updating the coefficients using coordinate descent
            for _ in range(100):  # Inner loop for coordinate descent
                for j in active_set:
                    # Computing partial residuals
                    partial_residuals = residuals + X_scaled[:, j] * theta[j]

                    # Computing coordinate-wise update
                    theta[j] = self._soft_threshold(
                        X_scaled[:, j] @ partial_residuals,
                        self.lambda_penalty
                    ) / (X_scaled[:, j] @ X_scaled[:, j])

                # Recomputing residuals
                residuals = y_centered - X_scaled @ theta

            # Step 7: Removing near-zero coefficients from the active set
            active_set = [j for j in active_set if np.abs(theta[j]) > self.tolerance]

        # Step 8: Restoring the original scale
        self.theta = theta / X.std(axis=0)
        self.intercept = y.mean() - X.mean(axis=0) @ self.theta
        self.active_set = active_set

        return LassoHomotopyResults(self)


class LassoHomotopyResults:
    """
    Storing and providing methods for LASSO model results
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """
        Making predictions using the learned coefficients
        """
        X = np.atleast_2d(X)
        return X @ self.model.theta + self.model.intercept

    def get_coefficients(self):
        """
        Returning the learned coefficients
        """
        return self.model.theta

    def get_active_set(self):
        """
        Returning the indices of active features
        """
        return self.model.active_set
