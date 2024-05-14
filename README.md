# Sentiment Classification with Neural Networks

<p align="center">
  <img src="https://github.com/kunaldudhavat/sentiment-classification/blob/main/images/sentiment-classification-graphic.png" alt="Topic classification" title="Topic classification">
</p>

## Introduction

This repository contains my project on training a neural network for binary sentiment classification. The goal of this project is to classify sentences as either positive or negative based on their sentiment. The implementation is done in Python using Jupyter Notebook.

## Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Approach](#approach)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

### Neural Network Model

The neural network model in this project defines the probability of a sentence `x` being positive as:

<p align="center">
  <img src="https://github.com/kunaldudhavat/sentiment-classification/blob/main/images/sentiment-classifier.png" alt="Topic classification" title="Topic classification">
</p>

where `enc_theta` is a differentiable function that encodes a sentence into a `d`-dimensional vector, and `w` and `b` are additional parameters to learn.

### Training

The training objective is to minimize the empirical cross-entropy loss between the true labels and the predicted labels. This is achieved using gradient descent.

## Dataset

The dataset used in this project is SST-2, which is a version of Stanford Sentiment Treebank prepared by the GLUE Benchmark for sentence-level binary sentiment classification of movie reviews (either positive or negative). It consists of sentences labeled with positive or negative sentiment. The data is divided into training and testing sets to evaluate the performance of the model. The dataset is stored in the `data/SST-2` directory.

## Approach

The approach taken in this project includes the following steps:

1. **Data Preprocessing**: Loading and preprocessing the dataset, including text cleaning and vectorization.
2. **Model Implementation**: Implementing the neural network model with parameters `w` and `b`.
3. **Loss Function**: Implementing the binary cross-entropy loss function.
4. **Encoders**: Implementing different encoders such as CNN and BiLSTM to encode sentences into vectors.
5. **Hyperparameter Tuning**: Tuning hyperparameters such as learning rate and batch size to optimize model performance.
6. **Model Evaluation**: Evaluating the trained model on the test set and visualizing the results.

## Requirements

- Python 3.8+
- Jupyter Notebook
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- TensorFlow or PyTorch

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sentiment-classification.git
   cd sentiment-classification

## Usage

1. Running the notebook:
   ```sh
   jupyter notebook
  
2. Open the sentiment-classificaiton.ipynb file and run all the cells


## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
