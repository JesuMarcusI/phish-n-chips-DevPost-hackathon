# phish-n-chips-DevPost-hackathon
DevPostHacakathon - Phishing Detector -_Phish-n-Chips
#  Phish & Chips - AI-Powered Phishing Detection
 Overview
Phish & Chips is a machine learning-based phishing detection system that:
- Trains a Random Forest model to identify phishing websites.
- Deploys as a Flask API for real-time detection.
- Integrates with a Firefox browser extension to alert users.
---
 Installation & Setup
 1️⃣ Setting Up the Training Environment
 Requirements
Ensure you have Python 3.8+ installed, then install dependencies:
```sh
pip install -r requirements.txt
```
 Training the Model
```sh
python train_model.py
```
This will generate a `phishing_model.pkl` file for the API.
---
 2️⃣ Running the Flask API
 Start the API
```sh
python app.py
```
This launches the API at `http://localhost:5000`, where you can send URLs for phishing analysis.
 Test the API
```sh
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"url": "https://example.com"}'
```
---
 3️⃣ Installing the Browser Extension (Firefox)
 Steps:
1. Open Firefox and go to `about:debugging/runtime/this-firefox`.
2. Click "Load Temporary Add-on."
3. Select the `manifest.json` file from the `browser_extension/` folder.
4. The extension icon should appear in your browser toolbar.
 Using the Extension
- Green ✅ = Legitimate site
- Red ⚠️ = Phishing detected
- Warning Page 🚨 = High-risk phishing site
---
 Usage
- Train the model (`train_model.py`).
- Run the API (`app.py`).
- Install and test the browser extension.
- Monitor logs for debugging and improvements.
---
