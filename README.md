# Sports Celebrity Image Classification Web App

This project is a deep learning-based application for classifying images of sports celebrities. It employs pre-trained models like MobileNetV2 for accurate predictions and features a user-friendly interface built with Streamlit.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [How to Run the App](#how-to-run-the-app)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Overview

The **Sports Celebrity Image Classification** app classifies images into one of six predefined categories:

- Cricket: Kane Williamson, MS Dhoni
- Basketball: Kobe Bryant
- Football: Lionel Messi
- Tennis: Maria Sharapova
- Javelin Throw: Neeraj Chopra

The application provides real-time predictions, confidence scores, and visualizations of the results.

## Features

- **Image Upload:** Users can upload images in JPG, JPEG, or PNG formats.
- **Dropdown Selection:** Pre-select the sports celebrity from a list.
- **Real-Time Prediction:** Displays the predicted category, person, and confidence scores.
- **Data Visualization:** Interactive bar charts to visualize confidence scores.

## Dataset

The dataset includes images of six sports celebrities from various sources. Data augmentation techniques were applied to increase dataset variability and reduce overfitting.

## Technologies Used

- **Python**
- **TensorFlow and Keras:** For building and training deep learning models.
- **Streamlit:** For creating the web application.
- **Pandas, NumPy, and Matplotlib:** For data manipulation and visualization.
- **Pre-Trained Models:** MobileNetV2 for transfer learning.

## Model Architecture

### Custom CNN
A custom convolutional neural network with the following structure:

- Convolutional and MaxPooling layers
- Dropout for regularization
- Dense layers for classification

### Transfer Learning Models
- **MobileNetV2:** Lightweight and efficient for deployment.
- **ResNet50:** High accuracy but computationally intensive.

The best-performing model was **MobileNetV2** with an accuracy of **86.1%**.

## How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sports-celebrity-classifier.git
   cd sports-celebrity-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model:
   Place the `app_model.h5` file in the project directory.

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Open the application in your browser at `http://localhost:8501`.

## Results

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Custom CNN    | 54.4%    | 0.54      | 0.55   | 0.55     |
| MobileNetV2   | 83%    | 0.86      | 0.86   | 0.86     |
| ResNet50      | ~86%     | 0.88      | 0.88   | 0.88     |

- MobileNetV2 was chosen for deployment due to its balance of performance and efficiency.

## Future Work

- Expand the dataset to include more sports and celebrities.
- Integrate advanced architectures like Vision Transformers.
- Add features such as live video analysis in the Streamlit app.

## Acknowledgments

1. TensorFlow and Keras Documentation
2. MobileNetV2 Research Paper
3. Online tutorials and community forums
4. Python libraries: Matplotlib, Pandas, NumPy, OpenCV



# Sports Celebrity Image Classifier

This repository contains a Streamlit web application that classifies images of sports celebrities into predefined categories. The application uses a trained deep learning model and provides interactive visualizations for predictions.

## Features

- Upload an image of a sports celebrity.
- Real-time predictions of the type of sport and player identity.
- Confidence scores displayed as a bar chart.
- Dropdown for selecting the expected sports celebrity.
- User-friendly interface.

---

## Demo

### Home Page
![Home Page](https://github.com/Mekapothulavenu/sports_celebrity_image_classification/blob/fb22faff20ff02f97a794bec6dde8c756252dd7e/code%20_and_app_images/code%20%26%20app%20images/app1.png "Streamlit App Home Page")

### Prediction Result
![Prediction Result](https://github.com/Mekapothulavenu/sports_celebrity_image_classification/blob/fb22faff20ff02f97a794bec6dde8c756252dd7e/code%20_and_app_images/code%20%26%20app%20images/app5.png "Prediction with Confidence Scores")

### Bar Chart Visualization
![Bar Chart](https://github.com/Mekapothulavenu/sports_celebrity_image_classification/blob/fb22faff20ff02f97a794bec6dde8c756252dd7e/code%20_and_app_images/code%20%26%20app%20images/app6.png "Confidence Scores Bar Chart")

---

## How to Run the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sports-celebrity-classifier.git
   cd sports-celebrity-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your browser to interact with the app.

---

## Model Overview

The application uses a fine-tuned MobileNetV2 model trained on a dataset of sports celebrity images. The model achieves high accuracy and efficiency, making it suitable for lightweight deployment.

---

## File Structure

- `app.py`: Streamlit application code.
- `app_model.h5`: Trained MobileNetV2 model.
- `requirements.txt`: Python dependencies.
- `images/`: Screenshots of the app.

---

## Code Snippet

### Main Streamlit Code
```python
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained model
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

# Image Upload Section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=500)

    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    preprocessed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(preprocessed_image)
    confidence_scores = tf.nn.softmax(prediction[0]).numpy()

    # Define sport categories and class names
    categories = ["Cricket", "Basketball", "Football", "Tennis", "Cricket", "Javelin Throw"]
    class_names = ["Kane Williamson", "Kobe Bryant", "Lionel Messi", "Maria Sharapova", "MS Dhoni", "Neeraj Chopra"]

    # Get predicted category and class
    predicted_class = np.argmax(prediction, axis=1)
    predicted_person = class_names[predicted_class[0]]

    # Display Results
    st.header("Prediction Results")
    st.success(f"🏅 *Sport*: {categories[predicted_class[0]]}")
    st.info(f"*Predicted Player*: {predicted_person}")

    # Visualization
    st.subheader("Confidence Scores")
    data = pd.DataFrame({"Class": class_names, "Confidence": confidence_scores})
    st.bar_chart(data.set_index("Class"))
else:
    st.info("Please upload an image to see the prediction.")
```

---

## References

1. TensorFlow and Keras Documentation.
2. MobileNetV2 Research Paper.
3. Online tutorials and resources for transfer learning.
4. Python Libraries: Matplotlib, Pandas, NumPy, OpenCV.

---

## License

This project is licensed under the MIT License.

