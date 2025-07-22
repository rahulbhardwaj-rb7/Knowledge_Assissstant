import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from utils.content_chunking import ChartGenerator
import pandas as pd

load_dotenv()

class VectorDatabase:
    def __init__(self, api_key):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        self.chart_generator = ChartGenerator()
        self.vector_store = None
        self.documents = []

    def _load_pdf(self, file_path):
        self.chart_generator.extract_tables_from_pdf(file_path)
        return PyPDFLoader(file_path).load()

    def _load_txt(self, file_path):
        return TextLoader(file_path).load()

    def _load_csv(self, file_path):
        self.chart_generator.extract_csv_tables(file_path)
        return CSVLoader(file_path).load()

    def _load_excel(self, file_path):
        self.chart_generator.extract_excel_tables(file_path)
        excel_file = pd.ExcelFile(file_path)
        docs = []
        for sheet in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            if df.empty:
                continue
            content = df.to_string()
            doc = Document(page_content=content, metadata={"source": file_path, "sheet_name": sheet})
            docs.append(doc)
        return docs

    def _get_loader(self, ext):
        return {
            '.pdf': self._load_pdf,
            '.txt': self._load_txt,
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
        }.get(ext)

    def add_documents(self, chunks):
        if not chunks:
            return
        valid = [c.strip() for c in chunks if c and isinstance(c, str) and len(c.strip()) > 10]
        if not valid:
            return
        self.documents.extend(valid)
        try:
            docs = [Document(page_content=c) for c in valid]
            if self.vector_store:
                self.vector_store.add_documents(docs)
                self.vector_store.persist()
            else:
                db_path = "vector_db/knowledge_base"
                os.makedirs("vector_db", exist_ok=True)
                self.vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=db_path
                )
                self.vector_store.persist()
        except Exception:
            pass

    def create_vector_db(self, file_paths, vector_db_dir, db_name):
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        docs = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            loader = self._get_loader(ext)
            if not loader:
                continue
            try:
                loaded = loader(file_path)
                docs.extend([d for d in loaded if hasattr(d, 'page_content') and d.page_content.strip()])
            except Exception:
                continue
        if not docs:
            return "No documents were successfully loaded"
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=vector_db_dir
        )
        return f"Successfully processed {len(file_paths)} files with {len(chunks)} chunks and {len(self.chart_generator.get_all_tables())} tables"

    def load_vector_db(self, db_path):
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        try:
            all_docs = self.vector_store.get()
            self.documents = all_docs['documents'] if all_docs['documents'] else []
        except:
            self.documents = []

    def query(self, question, top_k=3):
        if len(self.documents) == 0 or not question or len(question.strip()) == 0:
            return []
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search_with_score(question, k=top_k)
                formatted = [(doc.page_content, float(score)) for doc, score in results]
                if formatted:
                    return formatted
                else:
                    return self._keyword_fallback_search(question, top_k)
            except Exception:
                return self._keyword_fallback_search(question, top_k)
        else:
            return self._keyword_fallback_search(question, top_k)

    def _keyword_fallback_search(self, question, top_k=3):
        qwords = set(w.lower() for w in question.split() if len(w) > 2)
        results = []
        for doc in self.documents:
            dwords = set(w.lower() for w in doc.split() if len(w) > 2)
            inter = qwords.intersection(dwords)
            if inter:
                score = len(inter) / len(qwords) if qwords else 0
                results.append((doc, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def query_with_context(self, question, conversation_context=None, current_topic="", top_k=3):
        if len(self.documents) == 0 or not question or len(question.strip()) == 0:
            return "❌ No documents available", [], ""
        context_summary = self._build_context_summary(conversation_context, current_topic)
        enhanced_question = self._enhance_question_with_context(question, context_summary)
        relevant_docs = self.query(enhanced_question, top_k)
        if not relevant_docs:
            return "❌ I couldn't find relevant information to answer your question.", [], current_topic
        document_context = "\n\n".join([doc[0] for doc in relevant_docs])
        prompt = self._create_context_aware_prompt(question, document_context, context_summary, current_topic)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        detected_topic = self._detect_topic(question, answer, current_topic)
        related_questions = self._generate_context_aware_questions(question, answer, context_summary, detected_topic)
        return answer, related_questions, detected_topic

    def _build_context_summary(self, conversation_context, current_topic):
        if not conversation_context:
            return ""
        recent = conversation_context[-3:] if len(conversation_context) > 3 else conversation_context
        parts = []
        if current_topic:
            parts.append(f"Current topic: {current_topic}")
        if recent:
            parts.append("Recent conversation:")
            for i, ex in enumerate(recent, 1):
                parts.append(f"Q{i}: {ex['question']}")
                parts.append(f"A{i}: {ex['answer'][:200]}...")
        return "\n".join(parts)

    def _enhance_question_with_context(self, question, context_summary):
        if not context_summary:
            return question
        prompt = f"""
        Given this conversation context, reformulate the current question to include relevant context for better document search.
        Keep the enhanced question concise and focused.
        Context:
        {context_summary}
        Current Question: {question}
        Enhanced Question (for document search):"""
        try:
            response = self.llm.invoke(prompt)
            enhanced = response.content if hasattr(response, 'content') else str(response)
            return enhanced.strip().strip('"\'')
        except Exception:
            return question

    def _create_context_aware_prompt(self, question, document_context, conversation_context, current_topic):
        return f"""You are a knowledgeable assistant with access to documents and conversation history.
CONVERSATION CONTEXT:
{conversation_context if conversation_context else "No previous conversation"}
RELEVANT DOCUMENTS:
{document_context}
CURRENT QUESTION: {question}
INSTRUCTIONS:
1. Use the conversation context to understand what the user is referring to
2. If the question references previous topics or uses pronouns (he, she, it, they, this, that), use the context to resolve them
3. Provide a clear, specific answer based on the documents
4. If continuing a topic, acknowledge the connection to previous discussion
5. If the information isn't in the documents, say so clearly
ANSWER:"""

    def _detect_topic(self, question, answer, current_topic):
        prompt = f"""
        Based on this question and answer, identify the main topic being discussed.
        Return only the topic (2-4 words max), nothing else.
        Question: {question}
        Answer: {answer[:300]}...
        Current Topic: {current_topic if current_topic else "None"}
        Main Topic:"""
        try:
            response = self.llm.invoke(prompt)
            topic = response.content if hasattr(response, 'content') else str(response)
            topic = topic.strip().strip('"\'').strip('.')
            if topic and len(topic.split()) <= 4 and topic.lower() != current_topic.lower():
                return topic
            return current_topic
        except Exception:
            return current_topic

    def _generate_context_aware_questions(self, question, answer, context_summary, topic):
        prompt = f"""
        Based on the conversation context and current topic, suggest 3 follow-up questions.
        Make them natural continuations of the conversation.
        Topic: {topic}
        Current Q&A:
        Q: {question}
        A: {answer[:200]}...
        Context: {context_summary[:300] if context_summary else "No previous context"}
        Suggest 3 natural follow-up questions:"""
        try:
            response = self.llm.invoke(prompt)
            related_text = response.content if hasattr(response, 'content') else str(response)
            questions = []
            for line in related_text.split('\n'):
                line = line.strip()
                if line and '?' in line:
                    clean_line = line.lstrip('123456789.- ').strip()
                    if clean_line and len(clean_line) > 10:
                        questions.append(clean_line)
            return questions[:3] if questions else []
        except Exception:
            return []

    def get_available_tables(self):
        return self.chart_generator.get_all_tables()

    def get_table_info(self, table_id):
        return self.chart_generator.get_table_info(table_id)

    def suggest_charts_for_table(self, table_id):
        return self.chart_generator.suggest_charts(table_id)

    def generate_chart(self, table_id, chart_type, x_column=None, y_column=None, color_column=None):
        return self.chart_generator.generate_chart(table_id, chart_type, x_column, y_column, color_column)

    def get_stats(self):
        if not self.vector_store:
            return {
                "total_documents": len(self.documents),
                "database_type": "Keyword Search Only",
                "embedding_model": "Google Generative AI (embedding-001)",
                "status": "No vector database"
            }
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "total_documents": count,
                "database_type": "Chroma",
                "embedding_model": "Google Generative AI (embedding-001)",
                "status": "Active"
            }
        except Exception as e:
            return {
                "total_documents": len(self.documents),
                "database_type": "Chroma (Error getting stats)",
                "embedding_model": "Google Generative AI (embedding-001)",
                "status": f"Error: {e}"
            }

    def clear_database(self):
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
                self.vector_store = None
            self.documents = []
            self.chart_generator.clear_tables()
            import shutil
            if os.path.exists("./knowledge_base"):
                shutil.rmtree("./knowledge_base")
            return "✅ Database and tables cleared successfully"
        except Exception as e:
            return f"❌ Error clearing database: {e}"
