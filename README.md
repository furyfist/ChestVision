# Lung Cancer CT Scan Classifier

This project is a command-line application built with Python and PyTorch to classify lung CT scan images into one of seven categories, including normal and various types of cancerous conditions.

## Overview

The core of this project is a deep learning model that leverages transfer learning to achieve accurate classification on a specialized medical imaging dataset. The entire pipeline, from data loading to training and final prediction, is handled through a set of Python scripts.

## Features

* **Multi-Class Classification**: Distinguishes between 7 different lung tissue classifications.
* **Transfer Learning**: Utilizes a pre-trained ResNet-50 model, fine-tuned for this specific task.
* **Data Handling**: Uses the `dorsar/lung-cancer` dataset from the Hugging Face Hub, with a custom `Dataset` class for image processing.
* **End-to-End Scripts**: Includes separate scripts for training the model from scratch and for running predictions on new images using the saved model.

## Technology Stack

* Python 3
* PyTorch
* Torchvision
* Hugging Face Datasets
* Pillow (PIL)

## Project Structure

* `src/dataset.py`: Defines the custom PyTorch `Dataset` for loading and transforming images.
* `src/train.py`: Handles the complete model training and validation loop, saving the final weights.
* `src/predict.py`: Loads the saved model weights and runs a prediction on a single image file.
* `requirements.txt`: Lists all Python dependencies.

## Setup and Usage

1.  **Clone the repository.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Train the Model:**
    Run the training script. This will train the model and create the `lung_cancer_classifier.pth` file.
    ```bash
    python src/train.py
    ```
5.  **Run a Prediction:**
    Update the `image_path` variable inside `src/predict.py` with the path to your image, then run the script.
    ```bash
    python src/predict.py
    ```

## Current Status

The model achieves a validation accuracy of approximately 77% after 16 epochs of training on the provided dataset. The trained model weights are saved and can be used for inference.