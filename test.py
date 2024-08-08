import streamlit as st
from PIL import Image
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastembed import TextEmbedding
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import time

# Initialize embedding model
embedding_model = TextEmbedding()

# Initialize BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Your Qdrant API key and URL
api_key = "iARSA_Jprb0k6Nun2pL_qFkejK_5_jE8YD8n8qVSwuLGMZ4O4G_-Ag"
qdrant_url = 'https://0d23ce49-803b-4c5e-b670-c5d80a86e6c1.us-east4-0.gcp.cloud.qdrant.io:6333'

# Initialize Qdrant client with API key
client = QdrantClient(url=qdrant_url, api_key=api_key)

# Function to process image and text, and upsert to Qdrant
def process_image_and_text(image, text):
    img_px = list(image.getdata())
    img_size = image.size
    embed = embedding_model.embed([text])
    for i in embed:
        embed = i
    point = PointStruct(id=str(uuid.uuid4()), vector=embed, payload={"pixel_lst": img_px, "img_size": img_size, "image_text": text})
    
    # Retry logic for upsert operation
    retries = 3
    for attempt in range(retries):
        try:
            client.upsert(collection_name="clip_embedding", points=[point], wait=True)
            st.success("Image and text have been processed and upserted to Qdrant!")
            break
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Retrying due to error: {e}. Attempt {attempt + 1}/{retries}")
                time.sleep(2)  # Wait before retrying
            else:
                st.error(f"Failed after {retries} attempts: {e}")

# Function to convert pixel list and image size to an Image object
def create_image_from_pixels(pixel_lst, img_size):
    # Ensure pixel values are tuples of three integers
    pixel_lst = [tuple(pixel) if isinstance(pixel, list) else pixel for pixel in pixel_lst]
    # Create a blank image with the given size
    img = Image.new('RGB', img_size)
    # Put data (pixel values) into the image
    img.putdata(pixel_lst)
    return img

# Function to generate a description for the image using BLIP
def generate_image_description(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# About the Developer
def about_me():
    col1, col2 = st.columns(2)
    # Header
    with col1:
        st.title('Ankit Jangid')
        st.write('Jaipur, Rajasthan')
        st.write('+91 9461962044')
        st.write('[LinkedIn Profile](https://www.linkedin.com/in/ankit-jangid)')
        st.write('ajladaniya425@gmail.com')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Process", "Search", "About"])

# Home page
if page == "Home":
    st.title("Image and Text Processing with Qdrant")
    st.markdown("""
    ### Welcome to the Image and Text Processing App!

    This application allows you to upload images along with their descriptions, and then store them in a Qdrant database. You can also search for images based on a query.

    **Instructions:**
    1. **Upload & Process:**
        - Upload multiple images and provide corresponding descriptions.
        - The images and descriptions will be processed and stored in Qdrant.
    2. **Search:**
        - Enter a query to search for a similar image in the database.
        - The most similar image will be retrieved and displayed.

    **Technologies Used:**
    - **Streamlit:** For building the web application interface.
    - **Qdrant:** For storing and searching image embeddings.
    - **FastEmbed:** For generating text embeddings.

    **How It Works:**
    - When you upload an image and description, the description is converted to an embedding vector using FastEmbed.
    - The image pixels and the embedding are stored in Qdrant.
    - When you search for an image, the query is also converted to an embedding vector, and Qdrant is used to find the most similar image based on the stored embeddings.

    **Get Started:**
    Use the sidebar to navigate to the "Upload & Process" or "Search" page.
    """)

# Upload & Process page
elif page == "Upload & Process":
    st.title("Upload and Process Images and Texts")

    # Multiple image upload
    uploaded_images = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    auto_caption = st.checkbox("Automatically generate image captions")
    text_inputs = st.text_area("Enter descriptions for the images (one per line, sequence will be ↑)")

    # Process button for multiple images and descriptions
    if st.button("Process Images and Texts"):
        if uploaded_images:
            texts = text_inputs.split('\n') if not auto_caption else [generate_image_description(Image.open(img)) for img in uploaded_images]
            if len(uploaded_images) != len(texts):
                st.error("The number of images and descriptions must match.")
            else:
                for image_file, text in zip(uploaded_images, texts):
                    image = Image.open(image_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    process_image_and_text(image, text)
        else:
            st.error("Please upload images and enter descriptions.")

# Search page
elif page == "Search":
    st.title("Search for Similar Images")

    # Search query
    query = st.text_input("Enter a query to search for a similar image")
    if st.button("Search"):
        if query:
            try:
                embed = embedding_model.embed([query])
                for i in embed:
                    embed = i
                search_result = client.search(collection_name="clip_embedding", query_vector=embed, limit=1)

                if search_result:
                    payload = search_result[0].payload
                    pixel_lst = payload["pixel_lst"]
                    img_size = payload["img_size"]
                    img = create_image_from_pixels(pixel_lst, tuple(img_size))
                    st.image(img, caption='Retrieved Image', use_column_width=True)
                else:
                    st.error("No matching image found.")
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")
        else:
            st.error("Please enter a query.")

elif page == "About":
    about_me()
