import pandas as pd
import scipy.stats as stats

class ABHypothesisTester:
    def __init__(self, data):
        self.data = data
    # Define control and test groups
    def create_groups(self, df, feature, group_A_values, group_B_values):
        group_A = df[df[feature].isin(group_A_values)]
        group_B = df[df[feature].isin(group_B_values)]
        return group_A, group_B

    # Hypothesis Testing for Gender
    def create_gender_groups(self, df):
        group_A = df[df['Gender'] == 'Male']
        group_B = df[df['Gender'] == 'Female']
        return group_A, group_B
    
    # Perform Hypothesis Testing for ZipCode
    def create_zipcode_groups(self, df):
        # Define a threshold or specific zip codes for grouping
        zipcodes = df['PostalCode'].unique()
        mid_index = len(zipcodes) // 2
        group_A = df[df['PostalCode'].isin(zipcodes[:mid_index])]
        group_B = df[df['PostalCode'].isin(zipcodes[mid_index:])]
        return group_A, group_B
    # Hypothesis Testing Function
    def hypothesis_test(self, group_A, group_B, column, test_type='t'):
        if test_type == 't':
            # T-test for numerical data
            t_stat, p_value = stats.ttest_ind(group_A[column].dropna(), group_B[column].dropna())
        elif test_type == 'chi2':
            # Chi-Squared test for categorical data
            contingency_table = pd.crosstab(group_A[column].dropna(), group_B[column].dropna())
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        else:
            raise ValueError("Unsupported test type. Use 't' for t-test or 'chi2' for chi-squared test.")
        
        return p_value
    # Reporting Results
    def report_results(self, p_value, test_name):
        if p_value < 0.05:
            result = "Reject the null hypothesis"
        else:
            result = "Fail to reject the null hypothesis"
        
        print(f"{test_name}: p-value = {p_value:.4f} -> {result}")