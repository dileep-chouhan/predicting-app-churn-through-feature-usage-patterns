import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_users = 500
data = {
    'daily_logins': np.random.randint(0, 5, size=num_users),
    'avg_session_duration': np.random.randint(1, 60, size=num_users),
    'features_used': np.random.randint(1, 10, size=num_users),
    'churned': np.random.choice([0, 1], size=num_users, p=[0.8, 0.2]) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Data Analysis and Preparation ---
# No significant cleaning needed for synthetic data
# Split data into features (X) and target (y)
X = df[['daily_logins', 'avg_session_duration', 'features_used']]
y = df['churned']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training and Evaluation ---
# Train a logistic regression model (simple model for demonstration)
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
# Feature Importance (Illustrative -  Logistic Regression coefficients)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
#Churn Rate by Feature (Illustrative)
plt.figure(figsize=(10,6))
sns.countplot(x='features_used', hue='churned', data=df)
plt.title('Churn Rate by Number of Features Used')
plt.xlabel('Number of Features Used')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_by_features.png')
print("Plot saved to churn_by_features.png")