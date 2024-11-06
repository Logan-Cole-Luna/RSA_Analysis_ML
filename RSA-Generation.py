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

# Function to generate RSA keys with error handling
def generate_rsa_key(key_size=2048, e=65537):
    try:
        key = RSA.generate(key_size, e=e)
        return key
    except ValueError as ve:
        print(f"ValueError for key_size={key_size}, e={e}: {ve}")
        return None
    except Exception as ex:
        print(f"Unexpected error for key_size={key_size}, e={e}: {ex}")
        return None

# Function to extract features from RSA key
def extract_features(key):
    try:
        key_size = key.size_in_bits()
        e = key.e
        n = key.n
        p = key.p
        q = key.q
        phi = (p - 1) * (q - 1)
        d = key.d
        return {
            'key_size': key_size,
            'public_exponent': e,
            'modulus_size': key.size_in_bytes() * 8,
            'p_size': p.bit_length(),
            'q_size': q.bit_length(),
            'generation_time': getattr(key, 'generation_time', np.nan)
        }
    except AttributeError as ae:
        print(f"AttributeError during feature extraction: {ae}")
        return None
    except Exception as ex:
        print(f"Unexpected error during feature extraction: {ex}")
        return None

# Generate RSA keys and collect data
def generate_rsa_dataset(num_keys=100, key_sizes=[1024, 2048, 3072, 4096], public_exponents=[65537]):
    data = []
    attempts = 0
    generated = 0
    max_attempts = num_keys * 10

    while generated < num_keys and attempts < max_attempts:
        key_size = np.random.choice(key_sizes)
        e = np.random.choice(public_exponents)
        key = generate_rsa_key(key_size=2048, e=65537)
        generation_time = time.time() - time.time()

        if key is not None:
            features = extract_features(key)
            if features is not None:
                features['generation_time'] = generation_time
                data.append(features)
                generated += 1
                if generated % 100 == 0:
                    print(f"Generated {generated} keys so far...")
        attempts += 1

    if generated < num_keys:
        print(f"Generated only {generated} keys out of {num_keys} requested after {attempts} attempts.")
    else:
        print(f"Successfully generated {generated} keys.")

    df = pd.DataFrame(data)
    return df

# Generate the dataset
num_keys = 100
rsa_df = generate_rsa_dataset(num_keys=num_keys)

# Export the full dataset
rsa_df.to_csv("rsa_key_dataset.csv", index=False)
print("Exported rsa_key_dataset.csv")

# Handle missing values if any
rsa_df = rsa_df.dropna()

# Define security strength based on key size
def assign_security_strength(key_size):
    if key_size < 2048:
        return 'Weak'
    elif key_size < 3072:
        return 'Moderate'
    elif key_size < 4096:
        return 'Strong'
    else:
        return 'Very Strong'

rsa_df['security_strength'] = rsa_df['key_size'].apply(assign_security_strength)
strength_mapping = {'Weak': 1, 'Moderate': 2, 'Strong': 3, 'Very Strong': 4}
rsa_df['security_strength_label'] = rsa_df['security_strength'].map(strength_mapping)

# Features and target
features = ['key_size', 'public_exponent', 'modulus_size', 'p_size', 'q_size', 'generation_time']
X = rsa_df[features]
y = rsa_df['security_strength_label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Export the train and test sets
X_train.to_csv("rsa_key_X_train.csv", index=False)
X_test.to_csv("rsa_key_X_test.csv", index=False)
y_train.to_csv("rsa_key_y_train.csv", index=False)
y_test.to_csv("rsa_key_y_test.csv", index=False)
print("Exported train and test sets.")
