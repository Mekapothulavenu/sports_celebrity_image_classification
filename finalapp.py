import streamlit as st
from PIL import Image
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf

# Load your trained model
@st.cache_resource
def load_saved_model():
    return load_model("app_model.h5")

# Streamlit page configuration
st.set_page_config(page_title="Sports Celebrity Image Classifier", layout="wide", page_icon="🏀")

# Load the model
model = load_saved_model()

# Page Title
st.title("🎾 Sports Celebrity Image Classifier")
st.write("Welcome! Upload an image, and the classifier will predict the type of sport and identify the player. Let's get started!")

# Add spacing
st.divider()

# Dropdown for Sports Persons
st.header("Select a Sports Person")
sports_persons = st.selectbox(
    "Who are you uploading an image of?",
    ["Select a Sports Person", "Kane Williamson", "Kobe Bryant", "Lionel Messi", "Maria Sharapova", "MS Dhoni", "Neeraj Chopra"]
)
if sports_persons == "Select a Sports Person":
    st.warning("⚠️ Please select a sports person from the dropdown.")
else:
    st.write(f"*Selected Sports Person:* {sports_persons}")

# Add spacing
st.divider()

# Main Section for Image Upload
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=500)

    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((224, 224))  # Resize to match model input size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    preprocessed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(preprocessed_image)
    confidence = np.max(prediction) * 100

    # Define sport categories and class names
    categories = ["Cricket", "Basketball", "Football", "Tennis", "Cricket", "Javelin Throw"]
    class_names = ["Kane Williamson", "Kobe Bryant", "Lionel Messi", "Maria Sharapova", "MS Dhoni", "Neeraj Chopra"]

    # Get predicted category and class
    predicted_category = categories[np.argmax(prediction)]
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_person = class_names[predicted_class[0]]
    confidence_scores = tf.nn.softmax(predictions[0]).numpy()

    # Add spacing
    st.divider()

    # Check if the selected person matches the prediction
    if sports_persons != predicted_person:
        st.error(f"⚠️ The uploaded image does not match the selected person ({sports_persons}). The model predicted: {predicted_person}.")
    else:
        # Display Results
        st.header("Prediction Results")
        st.success(f"🏅 *Sport*: {predicted_category}")
        st.info(f"*Predicted Player*: {predicted_person}")

    st.subheader("Data Visualization for Confidence Scores")
    data = pd.DataFrame({
        "Class": class_names,
        "Confidence": confidence_scores
    })

    # Display the bar chart
    st.bar_chart(data.set_index("Class"))

    st.subheader("Confidence Scores")
    for i, score in enumerate(confidence_scores):
        st.write(f"**{class_names[i]}**: {score:.2%}")

    # Provide some feedback based on confidence
    if confidence > 80:
        st.balloons()
        st.write("🎉 Great! The model is very confident in its prediction!")
    else:
        st.warning("The model's confidence is below 80%. You might want to try another image.")
else:
    st.info("Please upload an image to see the prediction.")

# Add final spacing
st.markdown("---")
st.markdown(
    "*Note*: This model is trained on a specific dataset and may not generalize to all sports images. "
    "Ensure the uploaded image is clear and focuses on the player for best results."
)
