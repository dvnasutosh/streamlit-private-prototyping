import streamlit as st
import tempfile
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
torch.classes.__path__ = []

# Ensure the custom directories exist
model_dir = "./static/ml_models/embedding_models"
tmp_dir = "./tmp"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# Cache the model so it loads only once per session,
# and download/save the model to the specified directory.
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_dir)

st.title("Tempfile & Vector Embedding Demo")

model = load_model()

# User text input
text_input = st.text_area("Enter your text to generate an embedding:")

if st.button("Generate Embedding"):
    if text_input:
        # Generate embedding for the input text
        embedding = model.encode(text_input)
        # Convert embedding to a list and display its length
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        st.write("Generated Embedding as list:")
        st.write(embedding_list)
        st.write("Length of embedding:", len(embedding_list))
        
        # Write the embedding to a temporary file in the ./tmp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy", dir=tmp_dir) as temp_file:
            np.save(temp_file, embedding)
            temp_filename = temp_file.name

        st.write("Embedding saved to temporary file:")
        st.code(temp_filename)

        # Read back the embedding from the temporary file to confirm it was saved correctly
        loaded_embedding = np.load(temp_filename)
        loaded_embedding_list = loaded_embedding.tolist() if isinstance(loaded_embedding, np.ndarray) else list(loaded_embedding)
        st.write("Loaded Embedding from file (as list):")
        st.write(loaded_embedding_list)
        st.write("Length of loaded embedding:", len(loaded_embedding_list))

        # Optionally, clean up the temporary file
        os.remove(temp_filename)
    else:
        st.error("Please enter some text.")
