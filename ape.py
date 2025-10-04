import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PIL import Image

# Load environment variables gggyy
load_dotenv()

# Get API key from environment variables
api_key = "AIzaSyD_FcN1LN0n0KtF-FAWvJGrLJFDtFKQ0GQ"
if not api_key:
    st.error("API key is missing. Please set the GOOGLE_API_KEY in the environment.")
else:
    genai.configure(api_key=api_key)

# Function to get Google Gemini Pro Vision API response
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to process the uploaded image and convert it into a required format
def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        # Prepare the image data in the required format
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Health App")
st.header("Gemini Health App")

# Input field for prompt
input = st.text_input("Input Prompt: ", key="input")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""  # Placeholder for image

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Button to trigger API call
submit = st.button("Tell me the total calories")

# Input prompt for the image recognition model
input_prompt = """
You are an expert nutritionist. You need to analyze the food items in the image
and calculate the total calories. Please provide the details of each food item with its calorie intake
in the following format:

1. Item 1 - calories
2. Item 2 - calories
...
"""

# When the submit button is clicked
if submit:
    try:
        # Process the uploaded image
        image_data = input_image_setup(uploaded_file)
        
        # Get the response from the API
        response = get_gemini_response(input, image_data, input_prompt)
        
        # Display the response
        st.subheader("The Response is")
        st.write(response)
    except FileNotFoundError as e:
        st.error(str(e))
