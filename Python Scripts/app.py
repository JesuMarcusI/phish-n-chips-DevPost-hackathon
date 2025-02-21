from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import tldextract
import re
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Load the trained phishing detection model safely
MODEL_PATH = "phishing_model.pkl"
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

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

# Initialize Flask app
app = Flask(__name__)

FEATURE_NAMES = ["URL_Length", "Domain_Length", "Subdomain_Length", "Special_Chars", "Num_Dots", "Is_HTTPS", "Is_Trusted"]
# List of known trusted domains
TRUSTED_DOMAINS = {"facebook.com", "wikipedia.org", "youtube.com", "google.com"}
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
    is_trusted = 1 if domain in TRUSTED_DOMAINS else 0
    return pd.DataFrame([[url_length, domain_length, subdomain_length, special_chars, num_dots, is_https, is_trusted]], columns=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check logs for details."})

    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "No URL provided"})

        # Extract features with correct column names
        features = extract_features(url)

        # Get phishing probability
        phishing_prob = model.predict_proba(features)[:, 1][0]

        # **Manually adjust the prediction for trusted domains**
        if features["Is_Trusted"][0] == 1:
            phishing_prob -= 0.5 # Reduce phishing probability significantly
            

        # Apply threshold
        prediction = 1 if phishing_prob > 0.5 else 0 # 1 = Phishing, 0 = Legitimate
        result = "Phishing" if prediction == 1 else "Legitimate"

        # Log request details
        logging.info(f"Received URL: {url}")
        logging.info(f"Extracted Features: {features.to_numpy()}")
        logging.info(f"Raw Phishing Probability: {phishing_prob:.2f}")
        logging.info(f"Prediction: {result}")

        return jsonify({"url": url, "prediction": result})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Phishing detection service is running"})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
