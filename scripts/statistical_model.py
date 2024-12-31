# Import necessary libraries
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class StatisticalModel:
    def __init__(self, data):
        self.data = data

    def fit_model_per_zipcode(self, data):
        """
        Fits a linear regression model for each zipcode to predict total claims.
        Parameters:
        - data: pd.DataFrame with features and target ('TotalClaims').
        Returns:
        - models: Dictionary with zipcodes as keys and trained models as values.
        - predictions: Dictionary with zipcodes as keys and predictions as values.
        """
        # Store models and predictions
        models = {}
        predictions = {}
        # Group by Zipcode
        grouped = data.groupby('PostalCode')
        
        for zipcode, group in grouped:
            # Define features and target
            X = group.drop(columns=['TotalClaims', 'PostalCode'])
            y = group['TotalClaims']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize and fit the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store the model and predictions
            models[zipcode] = model
            predictions[zipcode] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'mse': mean_squared_error(y_test, y_pred)
            }
            
            # Print the performance for each zipcode
            print(f"Zipcode: {zipcode}")
            print(f"Mean Squared Error: {predictions[zipcode]['mse']:.4f}")
        
        return models, predictions

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.
        
        Parameters:
        - X: Features data
        - y: Target data
        - test_size: Proportion of the dataset to include in the test split
        - random_state: Seed for the random number generator

        Returns:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training target
        - y_test: Testing target
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_models(self, X_train, y_train):
        """
        Trains multiple regression models on the provided training data.
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        
        Returns:
        - lr_model: Trained Linear Regression model
        - rfr_model: Trained Random Forest Regressor model
        - xgb_model: Trained XGBoost Regressor model
        """
        # Initialize models
        lr_model = LinearRegression()
        rfr_model = RandomForestRegressor(random_state=42)
        xgb_model = xgb.XGBRegressor(random_state=42)
        
        # Train models
        lr_model.fit(X_train, y_train)
        rfr_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        
        return lr_model, rfr_model, xgb_model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the performance of a model using test data.
        
        Parameters:
        - model: The trained model to evaluate
        - X_test: Testing features
        - y_test: Testing target
        
        Returns:
        - mae: Mean Absolute Error of the model
        - mse: Mean Squared Error of the model
        - r2: R-squared Score of the model
        - y_pred: Predictions made by the model
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mae, mse, r2, y_pred

    def plot_metrics(self, models, mae_scores, mse_scores, r2_scores):
        """
        Plots comparison metrics for different models.
        
        Parameters:
        - models: List of model names
        - mae_scores: List of Mean Absolute Error scores for each model
        - mse_scores: List of Mean Squared Error scores for each model
        - r2_scores: List of R-squared scores for each model
        """
        # Plot Mean Absolute Error (MAE) scores
        plt.figure(figsize=(10, 6))
        plt.bar(models, mae_scores, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Comparison of MAE Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot Mean Squared Error (MSE) scores
        plt.figure(figsize=(10, 6))
        plt.bar(models, mse_scores, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Comparison of MSE Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot R-squared scores
        plt.figure(figsize=(10, 6))
        plt.bar(models, r2_scores, color='salmon')
        plt.xlabel('Models')
        plt.ylabel('R-squared Score')
        plt.title('Comparison of R-squared Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def analyze_feature_importance_shap(model, X_train):
        """
        Analyze feature importance using SHAP.

        Parameters:
        - model: The trained model to explain
        - X_train: Training features used to fit the model
        
        Returns:
        - shap_values: SHAP values for the features
        """
        # Create a SHAP explainer
        explainer = shap.Explainer(model, X_train)
        
        # Calculate SHAP values
        shap_values = explainer(X_train)
        
        # Summary plot
        shap.summary_plot(shap_values, X_train)
        
        return shap_values