# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('Churn_Modelling.csv')

# Data Preprocessing
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Feature Engineering: Interaction terms (Example: multiplying "Age" and "Tenure" to capture interaction)
df['Age_Tenure'] = df['Age'] * df['Tenure']

# Feature Engineering: Tenure Grouping
df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[0, 5, 10, 20, 30, 40], labels=['Short', 'Medium', 'Long', 'Very Long', 'Very Very Long'])

# Convert the "Tenure_Group" into dummy variables
df = pd.get_dummies(df, columns=['Tenure_Group'], drop_first=True)

# Features and label
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle Class Imbalance (SMOTE for oversampling)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define models
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, bootstrap=True, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

# Hybrid Model: Voting Classifier
voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
voting_clf.fit(X_train_smote, y_train_smote)

# Prediction
y_pred = voting_clf.predict(X_test)

# Evaluation
print(f"\n🔍 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hybrid Model (Voting Classifier)")
plt.show()

# Feature Importance (from Random Forest inside Voting Classifier)
rf_fitted = voting_clf.named_estimators_['rf']
importances = rf_fitted.feature_importances_

feature_names = df.drop('Exited', axis=1).columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df.sort_values(by='Importance', ascending=False).set_index('Feature').plot(kind='bar', figsize=(12,6))
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()
