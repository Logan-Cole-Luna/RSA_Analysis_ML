import os
import pandas as pd
import numpy as np
import time
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.number import GCD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
rsa_df = pd.read_csv("rsa_key_dataset.csv")
X_train = pd.read_csv("rsa_key_X_train.csv")
X_test = pd.read_csv("rsa_key_X_test.csv")
y_train = pd.read_csv("rsa_key_y_train.csv").squeeze()  # Converting to Series
y_test = pd.read_csv("rsa_key_y_test.csv").squeeze()    # Converting to Series

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=strength_mapping.keys(),
            yticklabels=strength_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Define quantum resilience based on key size
def quantum_resilience(key_size):
    required_qubits = key_size * 2
    return required_qubits

rsa_df['quantum_resilience_qubits'] = rsa_df['key_size'].apply(quantum_resilience)

# Display quantum resilience
print(rsa_df[['key_size', 'quantum_resilience_qubits']].head())

# Define target for quantum resilience
y_quantum = rsa_df['quantum_resilience_qubits']

# Split the data for quantum resilience prediction
X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_train, y_quantum, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train_q, y_train_q)

# Predictions
y_pred_q = rf_regressor.predict(X_test_q)

# Evaluation Metrics
mae = mean_absolute_error(y_test_q, y_pred_q)
r2 = r2_score(y_test_q, y_pred_q)

print(f"Mean Absolute Error: {mae:.2f} qubits")
print(f"R-squared: {r2:.2f}")

# Feature Importance for Regression
importances_q = rf_regressor.feature_importances_
feature_importance_q = pd.Series(importances_q, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_q, y=feature_importance_q.index)
plt.title('Feature Importance for Quantum Resilience Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Example: Perturb key size and observe prediction changes
def adversarial_example(original_key_size, perturbation=256):
    perturbed_key_size = original_key_size + perturbation
    example = {
        'key_size': perturbed_key_size,
        'public_exponent': 65537,
        'modulus_size': perturbed_key_size,
        'p_size': perturbed_key_size // 2,
        'q_size': perturbed_key_size // 2,
        'generation_time': 0.5
    }
    df_example = pd.DataFrame([example])
    predicted_strength = rf_model.predict(df_example)
    predicted_resilience = rf_regressor.predict(df_example)
    return predicted_strength, predicted_resilience

# Original key size
original_key_size = 2048
pred_strength, pred_resilience = adversarial_example(original_key_size)
print(f"Original Key Size: {original_key_size} bits")
print(f"Predicted Security Strength: {pred_strength[0]}")
print(f"Predicted Quantum Resilience: {pred_resilience[0]} qubits")

# Perturbed key size
perturbed_strength, perturbed_resilience = adversarial_example(original_key_size, perturbation=256)
print(f"\nPerturbed Key Size: {original_key_size + 256} bits")
print(f"Predicted Security Strength: {perturbed_strength[0]}")
print(f"Predicted Quantum Resilience: {perturbed_resilience[0]} qubits")

# Summary of Model Performance
print("=== Security Strength Prediction ===")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(class_report)

print("\n=== Quantum Resilience Prediction ===")
print(f"Mean Absolute Error: {mae:.2f} qubits")
print(f"R-squared: {r2:.2f}")

# Feature Importance Summary
print("\n=== Feature Importance for Security Strength ===")
print(feature_importance)

print("\n=== Feature Importance for Quantum Resilience ===")
print(feature_importance_q)
