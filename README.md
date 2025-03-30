# -Leukemia-detection-from-blood-images-using-CNN-model-
"Leukemia detection from blood images using MobileNet" aims to develop an AI-powered system for automatic leukemia diagnosis. By leveraging MobileNet's convolutional neural network architecture, the system analyzes blood images to detect abnormal white blood cells, enabling early diagnosis and treatment.
Blood Cell Cancer Classification Model
This repository contains a deep learning model for classifying blood cell cancer images. The model is built using TensorFlow and Keras, and utilizes the MobileNet architecture.

Key Features:
1. Data Preprocessing: The code includes data preprocessing steps, such as image resizing and data augmentation.
2. Model Training: The model is trained using a dataset of blood cell cancer images, with a custom top-3 accuracy metric.
3. Model Evaluation: The model is evaluated on a validation dataset, with metrics including accuracy, top-2 accuracy, and top-3 accuracy.
4. Model Conversion: The trained model is converted to a TensorFlow Lite (TFLite) model for deployment on mobile devices.

Files:
1. bloodwisepnormalmodel.h5: The trained Keras model.
2. BloodWise.tflite: The converted TFLite model.
3. code.py: The Python code for training and evaluating the model.

Requirements:
1. TensorFlow: Version 2.x
2. Keras: Version 2.x
3. NumPy: Version 1.x
4. Pandas: Version 1.x
