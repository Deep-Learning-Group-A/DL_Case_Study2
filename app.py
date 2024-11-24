import streamlit as st
import torch
import torch.nn as nn
import tensorflow as tf
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import gdown
import os

def download_vgg_model():
    url = "https://drive.google.com/uc?id=1XZv3yXTwMx6H5uQ2AshkK1cyerYQvoph"
    output = "models/transfer_learned_model_vgg.keras"
    if not os.path.exists(output):
        st.write("Downloading VGG model...")
        gdown.download(url, output, quiet=False)
    return output


# ------------------ GAN Generator Class ------------------ #
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.gen(x)

# ------------------ Utility Functions ------------------ #

@st.cache_resource
def load_generator_model():
    """Load DCGAN generator model."""
    generator_path = "models/generator_full.pth"  # GAN model path
    generator = torch.load(generator_path, map_location=torch.device("cpu"))
    generator.eval()
    return generator

@st.cache_resource
def load_vgg_model():
    """Load VGG model from Google Drive if not already downloaded."""
    model_path = download_vgg_model()
    model = tf.keras.models.load_model(model_path)
    return model

def generate_gan_images(generator, latent_dim, num_images):
    """Generate images using DCGAN."""
    noise = torch.randn(num_images, latent_dim, 1, 1)
    with torch.no_grad():
        images = generator(noise).cpu()
    return images

def predict_vgg(model, image):
    """Predict using VGG model."""
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    preds = model.predict(image)
    return preds

# ------------------ Pages for Different Models ------------------ #

def home_page():
    st.title("Model Hub")
    st.write("Welcome! Select a model from the sidebar to interact with it.")

def dcgan_page():
    st.title("DCGAN Image Generator")
    generator = load_generator_model()
    latent_dim = 100  # Latent vector size
    num_images = st.slider("Number of images to generate", 1, 16, 4)

    if st.button("Generate Images"):
        images = generate_gan_images(generator, latent_dim, num_images)
        images = (images + 1) / 2  # Rescale images to [0, 1]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")
        ax.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
        st.pyplot(fig)

def vgg_page():
    st.title("VGG Transfer Learning Model")
    vgg_model = load_vgg_model()
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Class label mappings
    class_labels = {0: 'drink', 1: 'food', 2: 'inside', 3: 'menu', 4: 'outside'}

    if uploaded_file is not None:
        # Load image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated here

        # Predict using VGG
        if st.button("Classify Image"):
            preds = predict_vgg(vgg_model, image_array)[0]  # Get the first batch's predictions
            predicted_class = np.argmax(preds)  # Get the index of the highest probability
            predicted_label = class_labels[predicted_class]
            confidence = preds[predicted_class] * 100  # Convert to percentage

            # Display the predicted class
            st.write(f"Prediction: **{predicted_label}** with confidence **{confidence:.2f}%**")

            # Convert all predictions to percentages and display with class labels
            st.write("Full predictions:")
            for i, prob in enumerate(preds):
                st.write(f"{class_labels[i]}: **{prob * 100:.2f}%**")



# ------------------ App Structure ------------------ #

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "DCGAN Generator", "VGG Model"])

# Render selected page
if page == "Home":
    home_page()
elif page == "DCGAN Generator":
    dcgan_page()
elif page == "VGG Model":
    vgg_page()
