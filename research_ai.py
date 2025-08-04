import streamlit as st
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os

# Streamlit UI
st.title("Arnav AI — Research Assistant (PDF + Web Search)")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
web_url = st.text_input("Enter Website URL to Scrape Content")
query = st.text_input("Ask your Question")

texts = []

# Process PDFs
if uploaded_files:
    for file in uploaded_files:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)

# Process Website Content
if web_url:
    try:
        response = requests.get(web_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        web_text = soup.get_text()
        texts.append(web_text)
    except Exception as e:
        st.error(f"Failed to fetch website: {e}")

# QA with LangChain
if texts and query:
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_texts(texts, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api_key),
        retriever=vectordb.as_retriever()
    )
    result = qa_chain.run(query)
    st.write("Answer:", result)
