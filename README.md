# 🦷 Oral Disease Classifier

An end-to-end Deep Learning project that classifies oral diseases from images using a Convolutional Neural Network (CNN) and deploys them via a Streamlit web app.

## 📊 The Dataset
The model was trained on the **Oral Image Dataset** containing 6 classes:
- Calculus
- Hypodontia
- Mouth Ulcer
- Normal
- Caries
- Tooth Discoloration

## 🧠 The Model
- **Architecture:** Custom CNN with 3 Convolutional blocks, MaxPooling, Dropout (to prevent overfitting), and Dense layers.
- **Input Size:** 128x128 pixels
- **Framework:** TensorFlow / Keras
- **Results:** Achieved a validation accuracy of ~82% over 20 epochs. 

## 🚀 The App
The web app is built with **Streamlit**. It allows users to upload an image and instantly see the model's prediction along with the confidence score for all 6 classes. 

*Note: The trained model file (`.keras`) is hosted on Google Drive due to size constraints and is automatically downloaded by the app upon startup using `gdown`.*

## 💻 How to Run Locally
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
