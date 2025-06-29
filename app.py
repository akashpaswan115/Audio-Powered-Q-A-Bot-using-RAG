import streamlit as st
import json
import os
import time
import sys
from dotenv import load_dotenv
import requests
import yt_dlp
from pathlib import Path
from langchain.document_loaders import TextLoader
from euriai.langchain_llm import EuriaiLangChainLLM
from langchain.chains import RetrievalQA, LLMChain
from euriai.langchain_embed import EuriaiEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_token = os.getenv('ASSEMBLY_AI_KEY')
euron_api_key = os.getenv('EURON_API_TOKEN')

base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# yt-dlp function for YouTube video
def save_audio(url):
    try:
        os.makedirs('temp', exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'temp/%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_filename = ydl.prepare_filename(info).replace('.webm', '.mp3')
        logger.info(f"Downloaded audio: {audio_filename}")
        return Path(audio_filename).name
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        st.error(f"Error downloading audio: {str(e)}")
        return None

def assemblyai_stt(audio_filename):
    try:
        audio_path = os.path.join('temp', audio_filename)
        with open(audio_path, "rb") as f:
            upload_res = requests.post(base_url + "/upload", headers=headers, data=f)
        upload_res.raise_for_status()
        upload_url = upload_res.json()["upload_url"]

        data = {"audio_url": upload_url}
        res = requests.post(base_url + "/transcript", json=data, headers=headers)
        res.raise_for_status()

        transcript_id = res.json()['id']
        polling_url = f"{base_url}/transcript/{transcript_id}"
        while True:
            result = requests.get(polling_url, headers=headers).json()
            if result['status'] == 'completed':
                break
            elif result['status'] == 'error':
                raise RuntimeError(f"Transcription error: {result['error']}")
            time.sleep(3)

        transcription_text = result['text']
        word_timestamps = result['words']

        os.makedirs('docs', exist_ok=True)
        with open('docs/transcription.txt', 'w') as f:
            f.write(transcription_text)
        with open('docs/word_timestamps.json', 'w') as f:
            json.dump(word_timestamps, f)

        return transcription_text, word_timestamps
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        st.error(f"Transcription error: {str(e)}")
        return None, None

@st.cache_resource
def setup_qa_chain():
    try:
        loader = TextLoader('docs/transcription.txt')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = EuriaiEmbeddings(api_key=euron_api_key)
        vectorstore = FAISS.from_documents(texts, embeddings)

        retriever = vectorstore.as_retriever()

        chat = EuriaiLangChainLLM(api_key=euron_api_key, model="gpt-4.1-nano")

        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        with open('docs/word_timestamps.json', 'r') as file:
            word_timestamps = json.load(file)

        return qa_chain, word_timestamps
    except Exception as e:
        st.error(f"Error setting up QA system: {str(e)}")
        return None, None

def find_relevant_timestamps(answer, word_timestamps):
    relevant = []
    answer_words = answer.lower().split()
    for word in word_timestamps:
        if word['text'].lower() in answer_words:
            relevant.append(word['start'])
    return relevant

def generate_summary(transcription):
    chat = EuriaiLangChainLLM(api_key=euron_api_key, model="gpt-4.1-nano")
    summary_prompt = PromptTemplate(
        input_variables=["transcription"],
        template="Summarize the following transcription in 3-5 sentences:\n\n{transcription}"
    )
    summary_chain = LLMChain(llm=chat, prompt=summary_prompt)
    return summary_chain.run(transcription)

st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")

st.title("Chat with Your Audio/Video")

input_source = st.text_input("Enter the YouTube video URL")

if input_source:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Your uploaded video")
        st.video(input_source)
        audio_filename = save_audio(input_source)
        if audio_filename:
            transcription, word_timestamps = assemblyai_stt(audio_filename)
            if transcription:
                st.info("Transcription completed. You can now ask questions.")
                st.text_area("Transcription", transcription, height=300)
                qa_chain, word_timestamps = setup_qa_chain()

                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                        st.subheader("Summary")
                        st.write(summary)

    with col2:
        st.info("Chat Below")
        query = st.text_input("Ask your question here...")
        if query:
            if qa_chain:
                with st.spinner("Generating answer..."):
                    result = qa_chain({"query": query})
                    answer = result['result']
                    st.success(answer)

                    timestamps = find_relevant_timestamps(answer, word_timestamps)
                    if timestamps:
                        st.subheader("Relevant Timestamps")
                        for ts in timestamps[:5]:
                            st.write(f"{ts // 60}:{ts % 60:02d}")
            else:
                st.error("QA system is not ready. Please transcribe first.")

def cleanup_temp_files():
    for f in os.listdir('temp'):
        os.remove(os.path.join('temp', f))
