# Flower Classification with TensorFlow and TensorFlow Hub

This project involves building a deep learning model to classify flower species using TensorFlow and TensorFlow Hub. The model is trained on a flower dataset and can predict the species of a given flower image.

## Table of Contents

- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

## Project Description

In this project, a deep neural network is built and trained to classify images of flowers into various species. The model uses a pre-trained feature extractor from TensorFlow Hub, which provides an efficient way to transfer knowledge from a model trained on a large dataset (like ImageNet) to a smaller task-specific dataset.

### Key Features:
- Use of TensorFlow and TensorFlow Hub for model building and training.
- A custom Keras layer wrapper to integrate TensorFlow Hub models.
- Image preprocessing for resizing, normalization, and model input formatting.
- Command-line interface (CLI) to interact with the model and predict flower species from a given image.

## Technologies Used

- **TensorFlow**: Open-source machine learning framework for building and training neural networks.
- **TensorFlow Hub**: A library for reusable machine learning modules.
- **Python**: Programming language used for the implementation.
- **NumPy**: Library for numerical computations.
- **Matplotlib**: Library for plotting and visualizing results.
- **Pillow (PIL)**: Library for image processing.
- **argparse**: Python library for parsing command-line arguments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AmroEid77/Image-Classifier-to-Identify-Flowers
    ```
2. Create a virtual environment
  ```bash
  conda create -n tf_flower python=3.10.0
   ```
3. Activate the env
  ```bash
  conda activate tf_flower
   ```
4. install the requirements file
   ```bash
   pip install -r requirements.txt
    ```

## Usage

### Predicting Flower Class from an Image

Run the `predict.py` script from the command line with the path to an image and the saved model file.

#### Basic Usage

  ```bash
  python predict.py ./test_images/wild_pansy.jpg flower.h5
   ```

  ```bash
  python predict.py ./test_images/wild_pansy.jpg flower.h5 --top_k 3
   ```

  ```bash
  python predict.py ./test_images/wild_pansy.jpg flower.h5 --category_names label_map.json
   ```
