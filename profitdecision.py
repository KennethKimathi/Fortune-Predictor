# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Loading the dataset
df = pd.read_csv(r"AC\Ornate\fortunepredictor\fortune1000_2024 (1).csv")

# Displaying the first few rows of the dataset to understand its structure
print(df.head())

# Converting categorical string values to binary values (1 and 0)
binary_columns = {
    'Worlds_Most_Admired_Companies': {'yes': 1, 'no': 0},
    'Growth_in_Jobs': {'yes': 1, 'no': 0}
}

for column, mapping in binary_columns.items():
    df[column] = df[column].map(mapping)

# Handling missing values
# By removing rows where any of the features or target are missing
df = df.dropna(subset=['Profits_M', 'Growth_in_Jobs', 'Revenues_M', 'Assets_M', 'MarketCap_Updated_M', 'Change_in_Rank', 'Number_of_employees', 'RevenuePercentChange', 'ProfitsPercentChange', 'MarketCap_March28_M', 'Worlds_Most_Admired_Companies'])


# Defining features (X) and target variable (y)
X = df[['Growth_in_Jobs', 'Revenues_M', 'Assets_M', 'MarketCap_Updated_M', 'Change_in_Rank', 'Number_of_employees', 'RevenuePercentChange', 'ProfitsPercentChange', 'MarketCap_March28_M', 'Worlds_Most_Admired_Companies']]
y = df['Profits_M']

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Training the model on the training data
dt_model.fit(X_train, y_train)

# Predicting on the test data
y_pred = dt_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Calculating and printing the baseline MSE
# Predict the mean of the target variable
mean_prediction = y_train.mean()
baseline_predictions = [mean_prediction] * len(y_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
print(f'Baseline Mean Squared Error (MSE): {baseline_mse}')

#Visualizing feature importance
feature_importances = dt_model.feature_importances_
feature_names = X.columns

# Print feature importances
for name, importance in zip(feature_names, feature_importances):
    print(f'Feature: {name}, Importance: {importance:.4f}')

# Visualizing the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree Regressor')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()

# Select 10 random samples from the test set for prediction
sample_size = 10
X_sample = X_test.sample(n=sample_size, random_state=42)
y_sample_actual = y_test[X_sample.index]
y_sample_pred = dt_model.predict(X_sample)

# Print actual and predicted profits
print("Decision Tree Actual vs Predicted Profits for 10 Companies:")
comparison_df = pd.DataFrame({'Actual Profits': y_sample_actual, 'Predicted Profits': y_sample_pred})
print(comparison_df)

# Plot graph of actual vs predicted profits
plt.figure(figsize=(10, 6))
plt.plot(comparison_df.index, comparison_df['Actual Profits'], marker='o', linestyle='-', color='b', label='Actual Profits')
plt.plot(comparison_df.index, comparison_df['Predicted Profits'], marker='x', linestyle='--', color='r', label='Predicted Profits')
plt.title('Actual vs Predicted Profits for 10 Companies (Decision tree)')
plt.xlabel('Company Index')
plt.ylabel('Profits (in million $)')
plt.legend()
plt.show()
