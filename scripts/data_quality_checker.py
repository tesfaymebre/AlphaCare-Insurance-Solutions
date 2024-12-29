import pandas as pd
import numpy as np

class DataQualityChecker:
    """
    A class to perform data quality checks on a pandas DataFrame.
    """
    def __init__(self, data):
        """
        Initialize with a pandas DataFrame.
        """
        self.data = data

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame and provide a detailed report as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Column', 'Missing Count', 'Missing Percentage'
        """
        missing_count = self.data.isnull().sum()
        missing_percentage = (self.data.isnull().mean() * 100)
        
        # Create a DataFrame for the report
        report_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percentage.values
        })
        
        return report_df

    def check_data_types(self, expected_types):
        """
        Check if the columns in the DataFrame have the expected data types.
        Args:
            expected_types (dict): A dictionary with column names as keys and expected data types as values.
        Returns:
            pd.DataFrame: A DataFrame with columns 'Column', 'Actual Data Type', 'Expected Data Type', and 'Match'.
        """
        results = []
        for col, dtype in expected_types.items():
            actual_dtype = self.data[col].dtype
            dtype=self.data[col].dtype
            match = pd.api.types.is_dtype_equal(actual_dtype, dtype)
            results.append({
                'Column': col,
                'Actual Data Type': actual_dtype,
                'Expected Data Type': dtype,
                'Match': match
            })
        return pd.DataFrame(results)


    def check_duplicates(self):
        """
        Check for duplicate rows in the DataFrame and provide a summary.

        Returns:
            str: A message indicating whether duplicates are present or not.
        """
        has_duplicates = self.data.duplicated().any()
        if has_duplicates:
            return "Duplicate rows found in the DataFrame."
        else:
            return "No duplicate rows found in the DataFrame."
