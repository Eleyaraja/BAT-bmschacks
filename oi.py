import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PIL import Image
import time

# Load environment variables
load_dotenv()

# Get API key from environment variables or set manually
api_key = "AIzaSyD_FcN1LN0n0KtF-FAWvJGrLJFDtFKQ0GQ"
if not api_key:
    st.error("API key is missing. Please set the GOOGLE_API_KEY in the environment.")
else:
    genai.configure(api_key=api_key)

# Function to get Google Gemini response for image and prompt
def get_gemini_response(input, image_data, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content([input, image_data[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return None

# Function to process the uploaded image and prepare it for API call
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # Read image file into bytes for API processing
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No image file uploaded.")

# Initialize Streamlit app
st.set_page_config(page_title="BAT Health App")
st.header("Gemini Health App")

# Interactive Sidebar for additional user inputs
st.sidebar.header("User Inputs")
input_prompt = st.sidebar.text_area(
    "Enter Custom Prompt:",
    """You are an expert nutritionist. You need to analyze the food items in the image
    and calculate the total calories. Please provide the details of each food item with its calorie intake
    in the following format:

    1. Item 1 - calories
    2. Item 2 - calories
    ...
    """,
    help="Provide a prompt for analyzing the image (e.g., 'Calculate the calories in the food items')."
)

# Input field for custom user input (text)
input = st.text_input("Input Custom Query (optional):", key="input", help="Provide additional instructions for image analysis.")

# File uploader for image input (accepting jpg, jpeg, png)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""  # Placeholder for image

# Display the uploaded image with instructions
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.markdown("**Instructions**: Upload an image with food items to analyze.")

# Button to trigger API call
submit = st.button("Analyze Image and Calculate Calories")

# Display a progress spinner when the button is clicked
if submit:
    if uploaded_file is None:
        st.error("Please upload an image before submitting.")
    else:
        with st.spinner("Processing your image..."):
            # Simulate some processing time (optional)
            time.sleep(2)  # Simulate API processing time

            try:
                # Process the uploaded image and prepare it for the API
                image_data = input_image_setup(uploaded_file)

                # Get the response from the Gemini API
                response = get_gemini_response(input, image_data, input_prompt)

                if response:
                    # Display the response
                    st.subheader("The Response is")
                    st.write(response)
                else:
                    st.warning("No response received from Gemini.")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Interactive additional options (dropdown, multiple questions)
st.sidebar.header("Additional Options")
ask_more_questions = st.sidebar.checkbox("Ask another question based on the image?", help="Check to ask multiple questions.")
if ask_more_questions:
    additional_question = st.sidebar.text_input("Enter additional question:")
    if additional_question:
        st.sidebar.button("Submit Additional Question")
        st.write(f"**Your additional question**: {additional_question}")

# Show footer message with info or credits
st.markdown("### Powered by Google Gemini AI")
st.markdown("This app uses Google Gemini to analyze food images and calculate calories. "
            "It helps you track your daily nutrition intake. "
            "All images are processed through the AI model to extract relevant details.")

