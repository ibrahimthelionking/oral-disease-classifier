import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 1. Page Configuration
st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷", layout="centered")

# 2. Title and Description
st.title("🦷 Oral Disease Classifier")
st.write("Upload an oral image to detect potential diseases. This app uses a Convolutional Neural Network trained on dental images.")

# 3. The exact class names from your training folder (alphabetical order!)
CLASS_NAMES = ['Calculus', 'Hypodontia', 'Mouth ulcer', 'Normal', 'caries', 'tooth discoloration']

# 4. Download from Google Drive 
MODEL_PATH = "oral_disease_model.keras"
FILE_ID = "1vQGowhJBKH3na34UkY6v9u-Yk4Er-Aaa"  

# 5. Load the trained model (cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
  if not os.path.exists(MODEL_PATH):
      # This downloads the file if it doesn't exist
      url = f'https://drive.google.com/uc?id={1vQGowhJBKH3na34UkY6v9u-Yk4Er-Aaa}'
      gdown.download(url, MODEL_PATH, quiet=False)
  # Load the saved model
  model = tf.keras.models.load_model('oral_disease_model.keras')
  return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 6. Build the User Interface
uploaded_file = st.file_uploader("Choose an oral image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for the model
    img_resized = image.resize((128, 128)) # Must match IMG_SIZE from training
    img_array = np.array(img_resized) / 255.0 # Rescale exactly like the model expects
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 128, 128, 3)
    
    # Make prediction
    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array)[0] # Get the array of 6 probabilities
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions) * 100
    
    # Display Results
    st.divider()
    st.subheader("Detection Results")
    
    # Show all predictions in a neat bar chart format
    st.write("**Confidence Scores for all classes:**")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = predictions[i] * 100
        st.write(f"{class_name}: {prob:.2f}%")
        st.progress(int(prob)) # Adds a visual progress bar
        
    st.divider()
    
    # Big highlight for the final prediction
    if confidence > 50:
        st.success(f"**Predicted Diagnosis:** {predicted_class_name} \n\n**Confidence:** {confidence:.2f}%")
    else:
        st.warning(f"**Predicted Diagnosis:** {predicted_class_name} \n\n**Confidence:** {confidence:.2f}% (Low Confidence - Please consult a dentist)")
