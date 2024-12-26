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
