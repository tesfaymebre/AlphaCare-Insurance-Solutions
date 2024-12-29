import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceEDA:
    def __init__(self, data):
        self.data = data
    
    # Data Summarization
    def summarize_data(self, df):
        """
        Provides descriptive statistics and data types.
        """
        print("Data Summary:\n", df.describe())
        print("\nData Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isnull().sum())
        
    # Univariate Analysis: Plotting histograms and bar charts
    def plot_bar_charts(self,df,categorical_cols):
        """
        Plots bar charts for categorical columns.
        """
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=df)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=90, ha='right')
            plt.show()
    def plot_histograms(self,df, numerical_cols):
        """
        Plots histograms for numerical columns.
        """
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=90, ha='right')
            plt.show()
    
        
        # Bivariate Analysis: Scatter plots and correlation matrices
    def plot_bivariate_analysis(self, df, col1, col2, group_col):
        """
        Plots scatter plots and correlation matrices.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col1, y=col2, hue=group_col, data=df)
        plt.title(f'Scatter Plot of {col1} vs {col2} by {group_col}')
        plt.show()

        corr_matrix = df[[col1, col2]].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='YlGn')
        plt.title('Correlation Matrix')
        plt.show()
        # Scatter plot functions to visualize relationships
    def scatter_plot(self, df, col_x, col_y):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col_x, y=col_y, data=df)
        plt.title(f'Scatter Plot of {col_x} vs. {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()
    # Data Comparison: Trends over geography
    def plot_trends_over_geography(self, df, trend_col, geography_col):
        """
        Compares trends over geographical locations.
        """
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=geography_col, y=trend_col, data=df)
        plt.title(f'Trend of {trend_col} over {geography_col}')
        plt.xticks(rotation=45)
        plt.show()
    # Outlier Detection: Box plots
    def plot_outlier_detection(self, df, numerical_cols):
        """
        Detects outliers using box plots for numerical columns and displays them in 3 columns.
        """
        # Number of rows needed
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  # Calculate number of rows needed
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust figure size
        # Flatten axes array in case of multiple rows and columns
        axes = axes.flatten()
        # Loop through the numerical columns and plot each boxplot
        for i, col in enumerate(numerical_cols):
            sns.boxplot(x=df[col].dropna(), ax=axes[i])
            axes[i].set_title(f'Box Plot for {col}')
        # Remove empty subplots if any
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()
            # Creative Visualizations: Example plots
    def create_creative_plots(self, df):
        """
        Creates creative plots for insights.
        """
        # Example: Scatter plot with regression line
        plt.figure(figsize=(10, 8))
        sns.regplot(x='SumInsured', y='CalculatedPremiumPerTerm', data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.title('SumInsured vs CalculatedPremiumPerTerm')
        plt.show()
        # Total claims, vehicle type, & province 
        plt.figure(figsize=(10, 5))
        sns.barplot(x='VehicleType', y='TotalClaims', hue='Province', data=df)
        plt.title('TotalClaims by VehicleType and Province')
        plt.xticks(rotation=45)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='TotalClaims', y='TotalPremium', data=df)
        plt.title('Distribution of TotalClaims by TotalPremium')
        plt.show()
   