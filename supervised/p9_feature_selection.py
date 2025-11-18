'''
Simple Feature Selection Example: 
Backward Elimination, Forward Selection, and LASSO on Student Pass/Fail Dataset
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Sample dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Define features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']


# Add constant to the model
X = sm.add_constant(X)

# Fit the model using Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# Display the model summary
print(model.summary())

# Remove feature with highest p-value if greater than 0.05
if model.pvalues['StudyHours'] > 0.05:
    X_REDUCED = pd.DataFrame(X).drop(columns='StudyHours')
    model = sm.OLS(y, X_REDUCED).fit()

# Final model after backward elimination
print(model.summary())

def forward_selection(X_data, y_data):
    """Forward Selection Function"""
    remaining_features = set(X_data.columns)
    selected_features = []
    current_score = 0.0

    while remaining_features:
        scores_with_candidates = []

        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_data[features_to_test], y_data, test_size=0.2, random_state=42)

            # Train the model
            fs_model = LinearRegression()
            fs_model.fit(X_tr, y_tr)
            fs_pred = fs_model.predict(X_te)
            score = r2_score(y_te, fs_pred)

            scores_with_candidates.append((score, feature))

        # Select the feature with the highest score
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates[0]

        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
        else:
            break

    return selected_features


best_features = forward_selection(pd.DataFrame(X), y)
print(f"Selected features using Forward Selection: {best_features}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the LASSO model with alpha (regularization parameter)
lasso_model = Lasso(alpha=0.1)

# Train the LASSO model
lasso_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = lasso_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')

# Display the coefficients of the features
print(f'LASSO Coefficients: {lasso_model.coef_}')
