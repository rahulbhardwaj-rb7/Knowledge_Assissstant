import streamlit as st
import nest_asyncio
from utils.vector_database import VectorDatabase
from utils.spellcheck import correct_spelling
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_core.documents import Document
from pathlib import Path

try:
    nest_asyncio.apply()
except ValueError:
    # uvloop or other incompatible event loop is already running
    pass

class QuestionAnsweringSystem:
    def __init__(self, google_api_key=None):
        if google_api_key is None:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        self.vector_db = VectorDatabase(api_key=google_api_key)
        self.llm = None  # Will be initialized when needed

    async def initialize_llm(self, google_api_key=None):
        if google_api_key is None:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not self.llm:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
        return self.llm

    @staticmethod
    def create_instance(google_api_key=None):
        if google_api_key is None:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        try:
            return QuestionAnsweringSystem(google_api_key)
        except Exception as e:
            print(f"Failed to create QA system: {e}")
            return None

    def _load_pdf(self, file_path):
        self.vector_db.chart_generator.extract_tables_from_pdf(file_path)
        return PyPDFLoader(file_path).load()

    def _load_txt(self, file_path):
        return TextLoader(file_path, encoding='utf-8').load()

    def _load_csv(self, file_path):
        self.vector_db.chart_generator.extract_csv_tables(file_path)
        return CSVLoader(file_path).load()

    def _load_excel(self, file_path):
        self.vector_db.chart_generator.extract_excel_tables(file_path)
        excel_file = pd.ExcelFile(file_path)
        documents = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if df.empty:
                continue
            content_lines = [f"Sheet: {sheet_name}", "=" * 50]
            content_lines.append("Columns: " + ", ".join(df.columns.astype(str)))
            content_lines.append("")
            max_rows = min(100, len(df))
            for idx, row in df.head(max_rows).iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if row_text.strip():
                    content_lines.append(f"Row {idx + 1}: {row_text}")
            if len(df) > max_rows:
                content_lines.append(f"\n... and {len(df) - max_rows} more rows")
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                content_lines.append("\nSummary Statistics:")
                for col in numeric_columns:
                    stats = df[col].describe()
                    content_lines.append(f"{col}: Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
            content = "\n".join(content_lines)
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "sheet_name": sheet_name,
                    "file_type": "excel",
                    "rows": len(df),
                    "columns": len(df.columns)
                }
            )
            documents.append(doc)
        return documents

    def add_documents_to_db(self, file_paths):
        if not file_paths:
            return "❌ No files provided"
        loader_map = {
            '.pdf': self._load_pdf,
            '.txt': self._load_txt,
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
        }
        documents = []
        for file_path in file_paths:
            file_ext = Path(file_path).suffix.lower()
            loader = loader_map.get(file_ext)
            if not loader:
                continue  # Unsupported file type
            try:
                docs = loader(file_path)
                valid_docs = [doc for doc in docs if hasattr(doc, 'page_content') and doc.page_content.strip()]
                documents.extend(valid_docs)
            except Exception:
                continue
        if not documents:
            return "❌ No documents could be loaded. Please check your files."
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        doc_chunks = text_splitter.split_documents(documents)
        text_chunks = [chunk.page_content.strip() for chunk in doc_chunks if hasattr(chunk, 'page_content') and chunk.page_content.strip()]
        if not text_chunks:
            return "❌ No valid text chunks could be created from documents"
        self.vector_db.add_documents(text_chunks)
        tables = self.vector_db.get_available_tables()
        table_summary = f" and {len(tables)} tables" if tables else ""
        return f"✅ Successfully processed {len(file_paths)} files into {len(text_chunks)} chunks{table_summary}"

    def answer_question_with_context(self, question, conversation_context=None, current_topic=""):
        stats = self.vector_db.get_stats()
        if stats.get("total_documents", 0) == 0:
            return "❌ No documents available. Please upload some documents first.", [], ""
        
        # Apply spellcheck for better search
        corrected_question = correct_spelling(question)
        
        answer, related_questions, detected_topic = self.vector_db.query_with_context(
            question=corrected_question,
            conversation_context=conversation_context,
            current_topic=current_topic,
            top_k=3
        )
        return answer, related_questions, detected_topic

    def answer_question(self, question):
        return self.answer_question_with_context(question, [], "")

    def get_database_stats(self):
        try:
            return self.vector_db.get_stats()
        except Exception as e:
            return {"error": f"Error getting stats: {str(e)}"}

    def clear_database(self):
        try:
            return self.vector_db.clear_database()
        except Exception as e:
            return f"❌ Error clearing database: {str(e)}"

    def get_available_tables(self):
        try:
            return self.vector_db.get_available_tables()
        except Exception:
            return []

    def get_table_info(self, table_id):
        try:
            return self.vector_db.get_table_info(table_id)
        except Exception:
            return None

    def suggest_charts_for_table(self, table_id):
        try:
            return self.vector_db.suggest_charts_for_table(table_id)
        except Exception:
            return []

    def generate_chart(self, table_id, chart_type, x_column=None, y_column=None, color_column=None):
        try:
            return self.vector_db.generate_chart(table_id, chart_type, x_column, y_column, color_column)
        except Exception as e:
            return None, f"Error generating chart: {str(e)}"