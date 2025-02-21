# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:35:41 2025

@author: AD14407
"""
import pandas as pd
import numpy as np
import tldextract
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef

filenm = "..\\Datasets\\PhiUSIIL_Phishing_URL_Dataset.csv"

# Load dataset
df = pd.read_csv(filenm) 

# Feature Extraction Function
def extract_features(url):
    url_length = len(url)
    
    # Extract domain and subdomains
    domain_info = tldextract.extract(url)
    domain_length = len(domain_info.domain)
    subdomain_length = len(domain_info.subdomain)
    
    # Count special characters
    special_chars = len(re.findall(r'[@_-]', url))
    
    # Count number of dots (subdomains)
    num_dots = url.count(".")
    
    # Check if HTTPS is present
    is_https = 1 if url.startswith("https") else 0
    
    return [url_length, domain_length, subdomain_length, special_chars, num_dots, is_https]

# Apply feature extraction to all URLs
df_features = df["URL"].apply(extract_features)
df_features = pd.DataFrame(df_features.tolist(), columns=["URL_Length", "Domain_Length", "Subdomain_Length", "Special_Chars", "Num_Dots", "Is_HTTPS"])

# Add label column back to the dataset
df_features["label"] = df["label"]

# Split data into training and testing sets (80% train, 20% test)
X = df_features.drop(columns=["label"])
y = df_features["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare (including Gradient Boosting models)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Adaboost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Confusion matrix for FPR & FNR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # Store results
    results[name] = {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2),
        "MCC": round(mcc, 2),
        "FPR": round(fpr, 2),
        "FNR": round(fnr, 2),
    }

    # Save the best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Convert results dictionary to a DataFrame
results_df = pd.DataFrame(results).T

# Display results
print("\nModel Performance Comparison:\n")
print(results_df)

# Save the best model as a .pkl file for use in app.py
model_filename = "phishing_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(best_model, model_file)

print(f"\nBest model ({best_model.__class__.__name__}) saved as {model_filename}")

# Plot Accuracy Comparison
plt.figure(figsize=(10, 5))
results_df["Accuracy"].sort_values().plot(kind="barh", color="skyblue")
plt.xlabel("Accuracy (%)")
plt.title("ML Model Accuracy Comparison for Phishing Detection")
plt.show()
