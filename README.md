
# Deep Learning Case Study 2 - Group A

## Overview

This repository showcases two projects focused on image classification, image generation, and sentiment prediction using advanced deep learning techniques. These projects utilize the Yelp datasets to solve real-world challenges in image and text data.

---

### **App 1: Image Classification and Generation**

- **URL**: [Image Classification and Generation](https://huggingface.co/spaces/shubhambhavsar/DL_Case_Study2)  
- **Description**: This project contains four models built on the Yelp Photos Dataset to perform image classification and generation tasks.  
  - **Models**:
    1. **VGG Transfer Learnt Model**: A pre-trained VGG model fine-tuned to classify images.
    2. **Custom CNN Model**: A Convolutional Neural Network built from scratch for image classification.
    3. **WGAN**: A Wasserstein Generative Adversarial Network for realistic image generation.
    4. **DCGAN**: A Deep Convolutional Generative Adversarial Network for generating detailed images.  

- **Dataset**: Yelp Photos Dataset.  
  - **Task**: Classify images into 5 categories: 
    - `drink`, `food`, `outside`, `inside`, `menu`.  

---

### **App 2: Sentiment Prediction Using Transformers**

- **URL**: [Sentiment Prediction Using Transformers](https://huggingface.co/spaces/shubhambhavsar3/DL_Transformers)  
- **Description**: Implements transformer-based models for sentiment analysis on Yelp restaurant reviews.  
  - **Models**:
    1. **Causal Transformer with Positional Encodings**: Predicts sentiment using an autoregressive approach.
    2. **Non-Causal Transformer with Positional Encodings**: Predicts sentiment by analyzing the entire text sequence.  

- **Dataset**: Yelp Restaurant Dataset.  
  - **Task**: Predict sentiment polarity (e.g., positive, negative, neutral) of user reviews.  

---

## Features

### **Image Classification and Generation**
- **Image Classification**:
  - Classifies images into 5 categories using VGG and custom CNN models.
- **Image Generation**:
  - Generates realistic images with WGAN and DCGAN.

### **Sentiment Prediction Using Transformers**
- **Sentiment Analysis**:
  - Predicts sentiment polarity of text reviews using causal and non-causal transformers.

---

## Deployment Details

- Both applications are deployed on Hugging Face Spaces, offering interactive interfaces for seamless predictions.
- Built with **Streamlit**, the applications are designed for user-friendly interaction.

---

## Results Interpretation

- **Image Classification and Generation**:  
  - Displays classification predictions and generated images with intuitive visualization.  
- **Sentiment Prediction Using Transformers**:  
  - Outputs sentiment predictions with confidence scores for better interpretability.  

---

## Access the Applications

**Image Classification and Generation**  
[https://huggingface.co/spaces/shubhambhavsar/DL_Case_Study2](https://huggingface.co/spaces/shubhambhavsar/DL_Case_Study2)  

**Sentiment Prediction Using Transformers**  
[https://huggingface.co/spaces/shubhambhavsar3/DL_Transformers](https://huggingface.co/spaces/shubhambhavsar3/DL_Transformers)  

---