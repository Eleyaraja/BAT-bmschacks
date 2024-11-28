import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import SQLite
import google.generativeai as genai

# Directly load API keys (without loading from .env)
api_key = "YOUR_GOOGLE_API_KEY"  # Insert your Google Gemini API key here

# Configure Google Gemini API
genai.configure(api_key=api_key)

# SQLite database to store vectors
DATABASE = "doctor_ai_vectors.db"

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                st.warning("Warning: No text found in one of the pages.")
    return text

# Function to split the text into chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the vector store in SQLite
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use Gemini embeddings for text
    
    # Initialize SQLite vector store
    vector_store = SQLite(embedding_model=embeddings, database=DATABASE)
    
    # Insert text chunks into the database
    for chunk in text_chunks:
        vector_store.add_texts([chunk])

    st.success("Text processing and vector store created in SQLite!")

# Function to create the conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    You are a doctor AI. Answer the following question with the most accurate medical information based on the provided context. 
    If the answer is not available, say: "Answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    model = genai.ChatModel("gemini-1.5-flash")  # Use Gemini 1.5 model for chat-based answers
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to process user input, perform similarity search, and generate a response
def user_input(user_question):
    try:
        # Use Gemini embeddings for question embedding
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Initialize SQLite vector store
        vector_store = SQLite(embedding_model=embeddings, database=DATABASE)

        # Get the vector embedding for the question
        query_embedding = embeddings.embed(user_question)

        # Perform similarity search in the SQLite database
        results = vector_store.similarity_search(query_embedding, top_k=5)
        
        # Get the documents from the search results
        docs = [result['text'] for result in results]
        
        # Get the conversational chain for answering
        chain = get_conversational_chain()

        # Generate and display the response
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Streamlit UI and logic
def main():
    st.set_page_config(page_title="Chat with Doctor AI💁")
    st.header("Chat with Doctor AI💁")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar for uploading files and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        # Trigger file processing and vector store creation
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload PDF files before processing.")
                return

            with st.spinner("Processing..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                if raw_text.strip() == "":
                    st.warning("No text found in the PDFs.")
                    return

                # Split the text into chunks for vector storage
                text_chunks = get_text_chunks(raw_text)

                # Create and save the vector store in SQLite
                get_vector_store(text_chunks)
                st.success("Text extraction and SQLite index creation complete!")

# Run the app
if __name__ == "__main__":
    main()

