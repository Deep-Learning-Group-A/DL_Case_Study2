import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Classification App", layout="wide")
st.title("Deep Learning Case Study 2 - Group A")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Model", ["VGG Model", "CNN Model", "WGAN", "DCGAN"])

class_labels = ['drink', 'food', 'outside', 'inside', 'menu']

def predict_vgg(model, image):
    image = image.resize((224, 224)) 
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) 
    image_array /= 255.0
    return model.predict(image_array)

def predict_cnn(model, image):
    image = image.resize((128, 128)) 
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0
    return model.predict(image_array)

# To display predictions with confidence percentages
def display_predictions(predictions, class_labels):
    
    # Convert predictions to percentages
    predictions_percentage = [round(score * 100, 2) for score in predictions[0]]

    # Find the predicted class
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_index]
    confidence = predictions_percentage[predicted_index]

    # Display results
    st.write("### Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class.capitalize()} **{confidence:.2f}%**")
    for label, score in zip(class_labels, predictions_percentage):
        st.write(f"{label.capitalize()}: **{score:.2f}%**")

# Load models separately with caching
@st.cache_resource
def load_vgg_model():
    return load_model("transfer_learned_model_vgg.h5")

@st.cache_resource
def load_cnn_model():
    return load_model("CNN_Model.h5")

# WGAN Generator model
class WGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(WGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Load the WGAN
@st.cache_resource
def load_wgan_generator():
    model_path = "generator_epoch_100.pth"
    latent_dim = 100
    generator = WGANGenerator(latent_dim)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator

# Generate images using the WGAN
def generate_wgan_images(generator, num_images=5, latent_dim=100):
    z = torch.randn(num_images, latent_dim, 1, 1)
    with torch.no_grad():
        generated_images = generator(z).cpu().numpy()
    return generated_images

# DCGAN Generator model
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(DCGANGenerator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.gen(x)

@st.cache_resource
def load_dcgan_generator():
    z_dim = 100
    channels_img = 3
    features_g = 64
    generator = DCGANGenerator(z_dim, channels_img, features_g)
    generator.load_state_dict(torch.load("generator_full.pth", map_location=torch.device('cpu')))
    generator.eval()
    return generator

# Generate images using the DCGAN
def generate_dcgan_images(generator, num_images=5, z_dim=100):
    z = torch.randn(num_images, z_dim, 1, 1)
    with torch.no_grad():
        generated_images = generator(z).cpu().numpy()
    return generated_images

# Logic for VGG Model Page
if page == "VGG Model":
    st.subheader("VGG Transfer Learning Model")
    st.write("Upload an image to classify it using the VGG transfer learning model.")
    
    model = load_vgg_model()
    uploaded_file = st.file_uploader("Choose an image for VGG Model...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")
        predictions = predict_vgg(model, image)
        display_predictions(predictions, class_labels)

# Logic for CNN Model Page
elif page == "CNN Model":
    st.subheader("CNN Model")
    st.write("Upload an image to classify it using the custom CNN model.")
    
    model = load_cnn_model()
    uploaded_file = st.file_uploader("Choose an image for CNN Model...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")
        predictions = predict_cnn(model, image)
        display_predictions(predictions, class_labels)

# Logic for WGAN Page
elif page == "WGAN":
    st.subheader("WGAN Image Generation")
    st.write("Generate images using the WGAN Generator.")
    
    generator = load_wgan_generator()
    num_images = st.slider("Number of Images to Generate", min_value=1, max_value=10, value=5)
    
    if st.button("Generate Images"):
        st.write("Generating images...")
        generated_images = generate_wgan_images(generator, num_images=num_images, latent_dim=100)
        fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
        for i, img in enumerate(generated_images):
            img = (img + 1) / 2
            axs[i].imshow(np.transpose(img, (1, 2, 0)))
            axs[i].axis('off')
        st.pyplot(fig)

# Logic for DCGAN Page
elif page == "DCGAN":
    st.subheader("DCGAN Image Generation")
    st.write("Generate images using the DCGAN Generator.")
    
    generator = load_dcgan_generator()
    num_images = st.slider("Number of Images to Generate", min_value=1, max_value=10, value=5)
    
    if st.button("Generate Images"):
        st.write("Generating images...")
        generated_images = generate_dcgan_images(generator, num_images=num_images, z_dim=100)
        fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
        for i, img in enumerate(generated_images):
            img = (img + 1) / 2
            axs[i].imshow(np.transpose(img, (1, 2, 0)))
            axs[i].axis('off')
        st.pyplot(fig)