import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

class DataPreprocessing:
    def __init__(self, data):
        self.data = data

    def feature_engineering(self, data):
        """
        Creates new features based on existing columns in the data.

        Features created:
        - 'VehicleAge': Age of the vehicle based on 'VehicleIntroDate'.
        - 'IsNewVehicle': Binary indicator for new vehicles.
        - 'ClaimsToPremiumRatio': Ratio of 'TotalClaims' to 'TotalPremium', with a small constant to avoid division by zero.

        Parameters:
        ----------
        data : pd.DataFrame
            Input data with required columns.

        Returns:
        -------
        pd.DataFrame
            Data with added features.
        """
        # Convert date columns to datetime if they are not already
        data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], errors='coerce')

       # Create Vehicle Age Feature
        data['VehicleAge'] = (pd.Timestamp.now() - data['VehicleIntroDate']).dt.days // 365
        
        # Create feature for whether the vehicle is new
        data['IsNewVehicle'] = data['NewVehicle'].apply(lambda x: 1 if x else 0)
    
         # Calculate Total Claims to Premium Ratio: Ratio of total claims to total premium
        data['ClaimsToPremiumRatio'] = data['TotalClaims'] / (data['TotalPremium'] + 1e-6)  # Adding a small value to avoid division by zero

        return data

    def handle_missing_values(self,data):
        """
            Handles missing values in the input data by filling them with appropriate values based on data type.

        For numeric columns, missing values are replaced with the mean of the column.
        For categorical columns, missing values are replaced with the mode (most frequent value) of the column.

        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame containing the data with missing values.
        Returns:
        -------
        pd.DataFrame
            The DataFrame with missing values handled according to the data type.

        Notes:
        -----
        - This function assumes that the input DataFrame contains numeric and categorical columns.
        - It modifies the DataFrame in place by filling missing values.
        """
        # Handle missing values based on data type
        for column in data.columns:
            if data[column].dtype in ['float64', 'int64']:  # Numeric columns
                # Replace missing values with the mean
                data[column].fillna(data[column].mean(), inplace=True)
            elif data[column].dtype == 'object':  # Categorical columns
                # Replace missing values with the mode
                data[column].fillna(data[column].mode()[0], inplace=True)
        return data

    def encode_categorical_data(self, data, method, columns):
        """
        Encode categorical data based on the specified method.
        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data to encode.
        - method (str): The encoding method to use ('onehot' or 'label').
        - columns (list): List of columns to encode.
        Returns:
        - pd.DataFrame: The DataFrame with encoded categorical features.
        """
        # Check if the specified columns are present in the DataFrame
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame")
        if method == 'onehot':
            # One-Hot Encoding
            encoded_data = pd.get_dummies(data, columns=columns, drop_first=True)
        elif method == 'label':
            # Label Encoding
            label_encoders = {}
            encoded_data = data.copy()  # Copy the original data
            for col in columns:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col])
                label_encoders[col] = le
            self.label_encoders = label_encoders  # Store label encoders for future use
        else:
            raise ValueError("Method should be 'onehot' or 'label'")
        # Ensure 1s and 0s instead of booleans by converting to int
        encoded_data = encoded_data.astype(int)
        return encoded_data

    def scale_data(self, data,method='standard'):
        """
        Scales the numerical features of the input data using StandardScaler.
        Parameters:
        ----------
        data : pd.DataFrame or np.ndarray
            The data to be scaled, containing numerical features only.

        Returns:
        -------
        pd.DataFrame
            The scaled data as a DataFrame, with the original column names preserved.
        Raises:
        ------
        ValueError
            If an unsupported method is provided.
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'any':
            # Default to StandardScaler if 'any' is specified
            scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaling method. Choose 'standard', 'minmax'")
        
        # Fit and transform the data
        scaled_data = scaler.fit_transform(data)
        
        # Return a DataFrame with original column names
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        return scaled_data
