# Automatic Image Captioning using CNN-RNN on Flickr8k Dataset

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements an automatic image captioning system that generates natural language descriptions for images. Leveraging the power of deep learning, the model combines a Convolutional Neural Network (CNN) for extracting visual features and a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units for generating the corresponding captions. The system is trained and evaluated on the popular Flickr8k dataset.

## Key Features

* **CNN-RNN Architecture:** Employs a robust architecture combining ResNet50 for image feature extraction and an LSTM-based decoder for caption generation.
* **Image Feature Extraction:** Utilizes a pre-trained ResNet50 model to efficiently compute rich feature vectors from input images.
* **LSTM-based Caption Generation:** An LSTM network decodes the image features into sequential word embeddings, ultimately generating a natural language caption.
* **Teacher Forcing:** The model is trained using teacher forcing to accelerate learning and improve caption quality.
* **Flickr8k Dataset:** Trained and evaluated on the Flickr8k dataset, which contains 8,091 images, each with five human-written captions.
* **Clear Data Preprocessing:** Includes steps for caption cleaning, tokenization, vocabulary construction, and sequence preparation.
* **Organized Code:** The codebase is structured logically with inline comments for easy understanding.

## Model Architecture

The core of the image captioning system consists of two main components:

1.  **Encoder (CNN):**
    * Utilizes a pre-trained **ResNet50** model to extract a 2048-dimensional feature vector representing the input image.
    * These features are then projected using dense layers (`init_h` and `init_c`) to initialize the hidden and cell states of the LSTM decoder.

2.  **Decoder (LSTM):**
    * Takes a sequence of word indices as input (during training, these are the ground truth captions).
    * An **Embedding Layer** transforms these word indices into dense word embeddings of `embedding_dim` size.
    * An **LSTM Layer** with `lstm_units` processes the embedded sequence, initialized with the image features from the encoder. The `return_sequences=True` argument ensures that the LSTM outputs a sequence of hidden states for each word.
    * A final **Dense Layer** with a `softmax` activation predicts the probability distribution over the vocabulary for each word in the output sequence.

  ## Sample Outputs

Here are a couple of examples showcasing the generated captions from the model:
## Sample Outputs

Here are a couple of examples showcasing the generated captions from the model:
## Sample Outputs

Here are a couple of examples showcasing the generated captions from the model:

**Image 1:**

<img src="Screenshot From 2025-05-12 02-01-50.png" alt="A group of dogs running in a grassy field" width="400" style="margin-bottom: 10px;">

**Generated Caption:**
<start> a group of dogs are running in a grassy field </end>

**Image 2:**

<img src="Screenshot From 2025-05-12 02-02-38.png" alt="A man is walking on a dock near a lake" width="400" style="margin-bottom: 10px;">

**Generated Caption:**
<start> a man is walking on a dock near a lake </end>
