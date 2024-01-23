import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Replace OpenAI imports with Gemini related ones
from google.oauth2 import service_account
from google_cloud import aiplatform

# Sidebar contents
with st.sidebar:
    st.title("**Yapay Zeka Chat Botu**")
    st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Google Gemini API](https://cloud.google.com/vertex-ai/docs/predictions/vision-text-language)
        ''')
    add_vertical_space(5)
    #st.write('Made by [Faruk Alam](https://youtube.com/@engineerprompt)')

load_dotenv()

def main():
    st.header("Chat with pdf")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is None:
        return

    # Progress indicator and error handling
    with st.spinner("Processing PDF..."):
        try:
            pdf_reader = PdfReader(pdf)
        except Exception as e:
            st.error("Error reading PDF:", e)
            return

    # Extract text from the PDF
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Text splitting and caching
    store_name = pdf.name[:-4]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text=text)

    # Embeddings (with caching)
    try:
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    except FileNotFoundError:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    # User query and input validation
    query = st.text_input("Ask questions about your PDF file:")
    if not query:
        st.warning("Please enter a query.")
        return

    # Search for relevant documents using embeddings
    docs = VectorStore.similarity_search(query=query, k=3)

    # Connect to Gemini API
    credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    endpoint = aiplatform.gapic.EndpointServiceClient(client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"})

    # Generate response using Gemini API
    try:
        response = endpoint.generate_text(name=endpoint.common_location_path(project="your-project-id", location="us-central1"), contents=[{"text": query}, *docs])
    except Exception as e:
        st.error("Error generating response:", e)
        return

    # Show response
    st.success("Response generated successfully!")
    st.write(response.text)

if __name__ == '__main__':
    main()
