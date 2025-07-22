from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader
import streamlit as st
import os
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.3,
    convert_system_message_to_human=True
)

def correct_spelling(text):
    if not text or not text.strip():
        return text

    PROMPT = """Fix spelling errors only. Keep proper names, structure, and meaning intact.
    Text: "{text}"
    Corrected:"""

    response = llm.invoke(PROMPT.format(text=text))
    corrected = response.content.strip().strip('"\'')
    
    return corrected

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

def get_chunked_text(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    return Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="vector_store"
    )

def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

def main():
    st.title("Knowledge Assistant")
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )

    if st.button("Process Files") and uploaded_files:
        with st.spinner("Processing files..."):
            text = extract_text_from_files(uploaded_files)
            text_chunks = get_chunked_text(text)
            vector_store = get_vector_store(text_chunks)
            
            st.session_state.vector_store = vector_store
            st.session_state.conversation_chain = get_conversation_chain(vector_store)
            st.success("Files processed and vector store created!")
        
    question = st.text_input("Ask a question:")
    if question and "conversation_chain" in st.session_state:
        corrected_question = correct_spelling(question)
        with st.spinner("Getting answer..."):
            st.session_state.chat_history = st.session_state.conversation_chain.chat_history
            response = st.session_state.conversation_chain.invoke(
                {"question": corrected_question, "chat_history": st.session_state.chat_history}
            )

if __name__ == "__main__":
    main()