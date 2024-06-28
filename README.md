# Convolutional Neural Network for Image Classification

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow and Keras. The model is built, trained, and tested to achieve high accuracy in predicting whether a given image is of a cat or a dog.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Preprocessing the Training Set](#preprocessing-the-training-set)
- [Preprocessing the Test Set](#preprocessing-the-test-set)
- [Building the CNN](#building-the-cnn)
- [Training the CNN](#training-the-cnn)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
This project aims to build a Convolutional Neural Network (CNN) for binary image classification. The dataset consists of images of cats and dogs. The CNN model will be trained to distinguish between these two classes and then used to make predictions on new images.

## Requirements
To run this project, you need to have the following libraries installed:
- TensorFlow
- Keras
- NumPy

## Data Preprocessing
Data preprocessing is a crucial step in training a CNN. It involves scaling the pixel values, augmenting the images to prevent overfitting, and organizing the dataset for the model.

## Preprocessing the Training Set
The training set is augmented using the ImageDataGenerator class from Keras. Augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied to increase the diversity of the training data.
## Preprocessing the Test Set
The test set is only rescaled to match the scaling applied to the training set.
## Building the CNN
The Convolutional Neural Network is built using the Sequential API from Keras. It consists of convolutional layers, pooling layers, a flattening layer, and fully connected (dense) layers.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

## Training the CNN
The model is compiled and trained using the training set. The binary crossentropy loss function is used, and the accuracy is monitored during training. The model is trained for 25 epochs.
## Making Predictions
After training, the model can be used to make predictions on new images. Here, an example image is loaded, preprocessed, and passed through the model to predict whether it is a cat or a dog.
## Results
The CNN model achieves high accuracy on the training set and performs well on the validation set. After training for 25 epochs, the model can accurately classify new images as either cats or dogs.

## Conclusion
This project demonstrates the implementation of a Convolutional Neural Network for image classification. The model is built and trained to achieve high accuracy, and it can make accurate predictions on new images. This project provides a solid foundation for understanding and applying CNNs in image classification tasks.

Feel free to explore, modify, and experiment with the code to improve the model's performance or to apply it to other image classification tasks.


