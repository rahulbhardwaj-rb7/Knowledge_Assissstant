import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os
import time
import asyncio

# Load environment variables
load_dotenv()

# Extract text from uploaded files
def extract_text_from_files(files):
    text = ""
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif ext == ".csv":
            df = pd.read_csv(file)
            text += df.to_string(index=False)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
            text += df.to_string(index=False)
        elif ext == ".txt":
            text += file.read().decode("utf-8")
    return text

# Split text into smaller chunks
def chunk_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create Chroma vector store
async def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    return Chroma.from_texts(chunks, embedding=embeddings, persist_directory="vector_db")

# Create conversational retrieval chain
async def create_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.3)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Generate charts using Plotly
def generate_chart(df, chart_type, x_column, y_column, color_column=None):
    chart_map = {
        "bar": lambda: px.bar(df, x=x_column, y=y_column, color=color_column),
        "line": lambda: px.line(df, x=x_column, y=y_column, color=color_column),
        "scatter": lambda: px.scatter(df, x=x_column, y=y_column, color=color_column),
        "pie": lambda: px.pie(values=df[x_column].value_counts(), names=df[x_column].value_counts().index),
        "histogram": lambda: px.histogram(df, x=x_column),
    }
    return chart_map.get(chart_type, lambda: None)()

# Typewriter effect for displaying text
def typewriter_effect(text, speed=0.01):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(speed)

# Main application
def main():
    st.set_page_config(page_title="Knowledge Assistant", page_icon="🤖")
    st.title("🤖 Knowledge Assistant")

    if st.button("🧹 Clear Chat History"):
        st.session_state.pop("conversation", None)
        st.session_state.pop("vector_store", None)
        st.success("Chat history cleared!")

    question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader("Upload files", type=["pdf", "csv", "xlsx", "xls", "txt"], accept_multiple_files=True)

        if st.button("Process Files") and uploaded_files:
            with st.spinner("Processing..."):
                raw_text = extract_text_from_files(uploaded_files)
                chunks = chunk_text(raw_text)
                vector_store = asyncio.run(create_vector_store(chunks))
                st.session_state.vector_store = vector_store
                st.session_state.conversation = asyncio.run(create_conversation_chain(vector_store))
            st.success("Files processed! You can now ask questions.")

        st.header("📊 Generate Chart")
        chart_type = st.selectbox("Select chart type", ["bar", "line", "scatter", "pie", "histogram"])
        x_column = st.text_input("X-axis column")
        y_column = st.text_input("Y-axis column")
        color_column = st.text_input("Color column (optional)")
        if st.button("Generate Chart") and uploaded_files:
            df = pd.read_csv(uploaded_files[0]) if uploaded_files[0].name.endswith(".csv") else pd.read_excel(uploaded_files[0])
            chart = generate_chart(df, chart_type, x_column, y_column, color_column)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.error("Invalid chart configuration.")

    if question and "conversation" in st.session_state:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.run(question)
        typewriter_effect(response)

if __name__ == "__main__":
    main()