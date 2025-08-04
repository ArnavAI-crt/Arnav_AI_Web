import streamlit as st
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import os
import pyttsx3
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import langchain_openai

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

st.title("Arnav AI — Research Assistant (Voice + Web + PDF)")

# Microphone Voice Input Component
class AudioProcessor(AudioProcessorBase):
    def recv_audio(self, frame):
        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(frame.tobytes(), frame.sample_rate, 2)
        try:
            text = recognizer.recognize_google(audio_data)
            st.session_state["voice_input"] = text
        except sr.UnknownValueError:
            pass
        return frame

webrtc_streamer(key="speech-to-text", audio_processor_factory=AudioProcessor)

# File Uploads & URL Input
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
web_url = st.text_input("Enter Website URL to Scrape Content")
query = st.text_input("Ask your Question (or use Microphone)")

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

# Use Voice Input if no Text Query
final_query = query if query else st.session_state.get("voice_input", "")

# QA with LangChain
if texts and final_query:
    embeddings = langchain_openai.OpenAIEmbeddings()
    vectordb = Chroma.from_texts(texts, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vectordb.as_retriever()
    )
    result = qa_chain.run(final_query)
    st.write("Answer:", result)

    # AI Speaks Answer
    engine.say(result)
    engine.runAndWait()
