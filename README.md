# ImageDenoising

This project applies a Convolutional Neural Network (CNN) architecture to perform image denoising by reducing salt-and-pepper noise in images. The model leverages multiwavelet transformations for enhanced feature extraction, achieving high accuracy in restoring clean images from noisy input.

Project Overview
The purpose of this project is to create an AI-based solution for image denoising using MultiWavelet CNNs. By training the model on images with salt-and-pepper noise, it learns to reconstruct clean images effectively, achieving a denoising accuracy of 96.98% with CNN and 95.35% with an RNN model. This project also implements K-fold cross-validation to further optimize accuracy.

Features
Noise Reduction: Effectively removes salt-and-pepper noise from images.
MultiWavelet CNN Architecture: Uses a multi-layer CNN for in-depth feature extraction.
Accuracy Optimization: K-fold cross-validation applied for enhanced model accuracy.
Training on Diverse Image Data: Trained on a large dataset of noisy images to improve generalization.
Technologies Used
Python: Programming language used for the project
Keras & TensorFlow: Libraries used to build and train the CNN model
NumPy & OpenCV: Libraries for image processing and data handling
Matplotlib: Visualization of training results and model performance
Model Architecture
Input Layer: Takes in 180x180x3 images as input.
10 Convolutional Layers: Deep convolutional layers with Batch Normalization and ReLU activation for feature extraction.
Output Layer: Uses tanh activation for output, enabling a range that preserves fine image details.

Results
The MultiWavelet CNN achieved an accuracy of 96.98% on the test dataset, effectively removing noise while preserving key image details. The model consistently performed well across validation sets due to K-fold cross-validation.

