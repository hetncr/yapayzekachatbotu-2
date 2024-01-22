import streamlit as st
import os
import google.generativeai as genai
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

st.set_page_config(page_title="Yapay Zeka Chat Botu",page_icon="📄")
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color:grey;'>Getting Started 🚀</h1>", unsafe_allow_html=True)
    #st.markdown("""<span ><font size=2>1. Get Started: Begin by adding your Gemini API key.</font></span>""",unsafe_allow_html=True)
    st.markdown("""<span ><font size=2>2.Explore Documents: Upload a document and fire away with your questions about the content</font></span>""",unsafe_allow_html=True)
    google_api_key = st.text_input("Google API Key", key="chatbot_api_key", type="password")
    "[Get an Google API key](https://makersuite.google.com/app/apikey)"
    uploaded_file = st.file_uploader("Choose a PDF file 📄", accept_multiple_files=False,type="pdf")
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        
    # st.divider()
    #st.markdown("<h1 color:black;'>Lets Connect! 🤝</h1>", unsafe_allow_html=True)
    #"[Linkedin](https://www.linkedin.com/in/muvva-thriveni/)" "  \t\t\t\t"  "[GitHub](https://github.com/MuvvaThriveni)"
    
home_title="PDF ile Etkileşime Gir🦜📄"
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=4>Beta</font></span>""",unsafe_allow_html=True)
st.write("(Pdf yüklemek için sol üstteki ok işaretine tıklayın!)")

st.caption(" A streamlit chatbot powered by Gemini to talk with PDF 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if uploaded_file is not None:
        if prompt := st.chat_input():
            if not google_api_key:
                st.info("Please add your Google API key to continue.")
                st.stop()
            genai.configure(api_key=google_api_key)
            with open("temp_pdf_file.pdf", "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            loader=PyPDFLoader(file_path="temp_pdf_file.pdf")
            pages=loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
            content = "\n\n".join(str(page.page_content) for page in pages)
            splits = text_splitter.split_text(content)
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=google_api_key)
            model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,google_api_key=google_api_key)
            prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """
            prompts = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompts)
            vector_index = Chroma.from_texts(splits, embeddings).as_retriever()
            docs = vector_index.get_relevant_documents(prompt)
            stuff_answer = stuff_chain(
                             {"input_documents": docs, "question": prompt}, return_only_outputs=True
                            )
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            #response=qa({"question": prompt})
            msg=stuff_answer['output_text']
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
else:
    
    if prompt := st.chat_input():
            if not google_api_key:
                st.info("Please add your Google API key to continue.")
                st.stop()
            else:
                st.info("Please Upload your document 📄 to continue.")
                st.stop()       
