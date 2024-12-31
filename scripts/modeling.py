import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

class InsuranceDataModeling:
    def __init__(self, data):
        self.data = data

    def handle_missing_data(self):
        """Handle missing data by imputing with median for numerical and most frequent for categorical columns."""
        # Impute numerical columns with median
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='median')
        self.data[numerical_cols] = imputer.fit_transform(self.data[numerical_cols])

        # Impute categorical columns with most frequent value
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        imputer = SimpleImputer(strategy='most_frequent')
        self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
        
        return self.data

    def feature_engineering(self):
        """Feature engineering: Create new features."""
        # Example: Create a Premium-to-Claim ratio
        self.data['Premium_to_Claim_Ratio'] = self.data['Premium'] / (self.data['Total_Claim'] + 1)  # Avoid division by zero
        return self.data

    def encode_categorical_data(self):
        """Encode categorical data using one-hot encoding or label encoding."""
        # Ensure the columns exist in the data
        categorical_cols = ['Province', 'Zipcode', 'Gender']
        
        # Check if the columns exist and perform encoding
        for col in categorical_cols:
            if col in self.data.columns:
                if col == 'Gender':  # Label encoding for binary columns
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    self.data['Gender'] = label_encoder.fit_transform(self.data['Gender'])
                else:  # One-hot encoding for other categorical columns
                    self.data = pd.get_dummies(self.data, columns=[col], drop_first=True)
        
        return self.data



    def train_test_split_data(self):
        """Split the data into training and test sets."""
        # List of columns to drop
        columns_to_drop = ['Total_Claim']  # Drop 'Total_Claim' as it's not part of the target
        self.data = self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns])
        
        X = self.data.drop(columns=['Premium'])  # Use 'Premium' as the target variable
        y = self.data['Premium']  # Set 'Premium' as the target variable
        
        # Check if 'Premium' exists
        if 'Premium' not in self.data.columns:
            raise ValueError("Column 'Premium' is missing from the data.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train a linear regression model and evaluate."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train a decision tree model and evaluate."""
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train a random forest model and evaluate."""
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Return the trained model as well
        return model, mse, r2


    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train an XGBoost model and evaluate."""
        model = xgb.XGBRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def feature_importance_analysis(self, model, X_train):
        """Analyze feature importance using RandomForestRegressor."""
        importances = model.feature_importances_
        indices = X_train.columns
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance (Random Forest)')
        plt.barh(indices, importances)
        plt.xlabel('Feature Importance')
        plt.show()

    def shap_analysis(self, model, X_train, X_test):
        """Perform SHAP analysis for model interpretation."""
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test)

    def compare_models(self, results):
        """Compare the performance of different models."""
        comparison_df = pd.DataFrame(results).T
        return comparison_df