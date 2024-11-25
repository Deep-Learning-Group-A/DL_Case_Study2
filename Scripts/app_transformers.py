import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import tensorflow as tf
import ast

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Sentiment Analysis")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Model", ["Causal Transformer", "Non-Causal Transformer"])

transformer_class_labels = ['Negative', 'Neutral', 'Positive']

# Positional Embedding layer
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="pos_emb",
            shape=(self.max_len, self.embed_dim),
            initializer="random_normal"
        )
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positional_embeddings = self.position_embeddings[:seq_len, :]
        return inputs + positional_embeddings

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self.embed_dim
        })
        return config

@st.cache_resource
def load_causal_tokenizer():
    tokenizer_path = "tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    if isinstance(tokenizer.word_index, str):
        tokenizer.word_index = ast.literal_eval(tokenizer.word_index)

    if not isinstance(tokenizer.word_index, dict):
        raise ValueError("Tokenizer word_index is not a dictionary.")
    if not isinstance(tokenizer.oov_token, str):
        raise ValueError("Tokenizer oov_token is not a string.")
   
    return tokenizer

@st.cache_resource
def load_noncausal_tokenizer():
    tokenizer_path = "tokenizer_noncausal.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load causal transformer model
@st.cache_resource
def load_causal_transformer_model():
    model_path = "causal_position_transformer"
    model = tf.saved_model.load(model_path)
    return model

# Load non-causal transformer model
@st.cache_resource
def load_ncausal_transformer_model():
    return tf.keras.models.load_model(
        "transformer_model.h5",
        custom_objects={"PositionalEmbedding": PositionalEmbedding}
    )

# Predict sentiment for causal transformer
def predict_sentiment(model, text, tokenizer):
    try:
        # Tokenize and pad input text
        sequences = tokenizer.texts_to_sequences([str(text)])
        padded_sequences = pad_sequences(sequences, maxlen=100, padding="post")

        # Get predictions from the model
        outputs = model(padded_sequences)
        probabilities = tf.nn.softmax(outputs).numpy()
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        return predicted_class, confidence, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Predict sentiment for non-causal transformer
def predict_transformer(model, text, tokenizer):
    try:
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
        predictions = model.predict(padded_sequences)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def display_predictions(predicted_class, confidence, probabilities, class_labels):
    # Convert predictions to percentages
    predictions_percentage = [round(score * 100, 2) for score in probabilities[0]]

    # Find the predicted class
    predicted_label = class_labels[predicted_class]

    # Display results
    st.write("### Prediction Results")
    st.write(f"**Predicted Class:** {predicted_label.capitalize()} **{confidence * 100:.2f}%**")
    for label, score in zip(class_labels, predictions_percentage):
        st.write(f"{label.capitalize()}: **{score:.2f}%**")

# Logic for Causal Transformer Page
if page == "Causal Transformer":
    st.subheader("Causal Transformer Model")

    # Load tokenizer and causal model
    tokenizer = load_causal_tokenizer()
    causal_transformer_model = load_causal_transformer_model()

    input_text = st.text_area("Enter text for sentiment analysis...")

    if st.button("Analyze Sentiment"):
        if input_text.strip():
            st.write("Classifying sentiment...")
            predicted_class, confidence, probabilities = predict_sentiment(causal_transformer_model, input_text, tokenizer)
            
            if probabilities is not None:
                # Display predictions
                display_predictions(predicted_class, confidence, probabilities, transformer_class_labels)
        else:
            st.warning("Please enter text to analyze.")

# Logic for Non-Causal Transformer Page
elif page == "Non-Causal Transformer":
    st.subheader("Non-Causal Transformer Model")
    
    # Load tokenizer and non-causal model
    tokenizer = load_noncausal_tokenizer()
    transformer_model = load_ncausal_transformer_model()

    input_text = st.text_area("Enter text for sentiment analysis...")

    if st.button("Analyze Sentiment"):
        if input_text.strip():
            st.write("Classifying sentiment...")
            predictions = predict_transformer(transformer_model, input_text, tokenizer)
            
            if predictions is not None:
                # Display predictions
                display_predictions(np.argmax(predictions), np.max(predictions), predictions, transformer_class_labels)
        else:
            st.warning("Please enter text to analyze.")