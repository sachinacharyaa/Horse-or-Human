
## Introduction

This project aims to create a machine learning model capable of distinguishing between humans and horses in images.
The model is trained on a dataset containing labeled images of both classes and is implemented using Google Colab 
for ease of use and accessibility.

## Features

- Binary image classification (Human vs. Horse)
- Utilizes TensorFlow and Keras for model development
- Google Colab for easy setup and execution
- Example scripts for training, evaluation, and inference

## Requirements

- Google Colab account
- Python 3.x
- TensorFlow
- Keras
- OpenCV (for image processing)
- NumPy
- Matplotlib (for visualization)

## Usage

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/sachinacharyaa/human-horse-detector.git
    ```
2. Open the `human_horse_detector.ipynb` notebook in Google Colab.
3. Follow the instructions in the notebook to set up the environment, download the dataset, and train the model.

## Model Training

The model training process includes the following steps:

1. **Data Preparation**: Load and preprocess the dataset. The dataset should be split into training and validation sets.
2. **Model Architecture**: Define the Convolutional Neural Network (CNN) architecture using Keras.
3. **Training**: Compile and train the model using the training set.
4. **Validation**: Evaluate the model's performance on the validation set.

## Evaluation

After training the model, evaluate its performance using the test set. The notebook includes code to generate accuracy and loss plots, confusion matrices, and other relevant metrics.

## Inference

To use the trained model for inference on new images, follow these steps:

1. Load the trained model:
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('path/to/your/saved_model.h5')
    ```
2. Preprocess the input image:
    ```python
    import cv2
    import numpy as np
    
    img = cv2.imread('path/to/image.jpg')
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = np.expand_dims(img, axis=0)
    ```
3. Predict the class:
    ```python
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print("Horse detected!")
    else:
        print("Human detected!")
    ```



