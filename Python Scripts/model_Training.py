import pandas as pd
import numpy as np
import tldextract
import re
import pickle
import logging
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
# List of known trusted domains (Whitelist)
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# List of known trusted domains (Whitelist)
def load_trusted_domains(filename):
    try:
        with open(filename, 'r') as file:
            trusted_domains = {line.strip() for line in file if line.strip()}
        return trusted_domains
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return set()

# Load the trusted domains
TRUSTED_DOMAINS = load_trusted_domains('Trusted_Domains.txt')
filenm = "PhiUSIIL_Phishing_URL_Dataset.csv"
 
# Feature extraction function
def extract_features(url):
    """Extracts improved features from a URL."""
    url_length = len(url)
    
    # Extract domain and subdomains
    domain_info = tldextract.extract(url)
    domain = domain_info.domain + "." + domain_info.suffix
    domain_length = len(domain_info.domain)
    subdomain_length = len(domain_info.subdomain)
    
    # Count special characters
    special_chars = len(re.findall(r'[@_-]', url))
    
    # Count number of dots (subdomains)
    num_dots = url.count(".")
    
    # Check if HTTPS is present
    is_https = 1 if url.startswith("https") else 0
    
    # Check if domain is in trusted list
    is_trusted = 1 if domain in TRUSTED_DOMAINS else 0 # Manually increase importance of trusted domains
    
    return [url_length, domain_length, subdomain_length, special_chars, num_dots, is_https, is_trusted]

# Load dataset
df = pd.read_csv(filenm)  

# Define feature names
FEATURE_NAMES = ["URL_Length", "Domain_Length", "Subdomain_Length", "Special_Chars", "Num_Dots", "Is_HTTPS", "Is_Trusted"]

# Apply feature extraction
df_features = df["URL"].apply(extract_features)
df_features = pd.DataFrame(df_features.tolist(), columns=FEATURE_NAMES)

# Add label column
df_features["label"] = df["label"]

# **Oversample Trusted Sites to Force Their Influence**
df_legit_trusted = df_features[(df_features["Is_Trusted"] > 0) & (df_features["label"] == 0)]
df_balanced = pd.concat([df_features, df_legit_trusted] * 5, ignore_index=True)  

# Split data
# Split data
X = df_balanced.drop(columns=["label"])
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with modified weighting
model = RandomForestClassifier(n_estimators=100, random_state=42,)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#add
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
# Confusion matrix for FPR & FNR
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"MCC: {mcc:.2f}")
print(f"False Positive Rate: {fpr:.2f}")
print(f"False Negative Rate: {fnr:.2f}")
# Save the trained model as a .pkl file
# Save the trained model
with open("phishing_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully.")

# **Check Feature Importance**
importances = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": model.feature_importances_})
print(importances.sort_values(by="Importance", ascending=False))


