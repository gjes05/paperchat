
import os
import tempfile
from typing import List, Optional
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec

load_dotenv()


EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL      = "gemini-2.5-flash"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 5

class PaperChatRAG:
    def __init__(self):
        self.google_api_key   = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name       = os.getenv("PINECONE_INDEX_NAME", "PaperChat")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=self.google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=self.google_api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self._ensure_index()

        self.vector_store: Optional[PineconeVectorStore] = None
        self.retriever = None
        self.chat_history: List[tuple] = []  # list of (human, ai) string tuples

    # Pinecone Setup 

    def _ensure_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

    # Document Processing 

    def load_pdf(self, uploaded_file) -> List[Document]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)

        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)

    def ingest_documents(self, uploaded_files) -> dict:
        all_chunks = []
        file_stats = {}

        for uploaded_file in uploaded_files:
            docs   = self.load_pdf(uploaded_file)
            chunks = self.split_documents(docs)
            all_chunks.extend(chunks)
            file_stats[uploaded_file.name] = {
                "pages":  len(docs),
                "chunks": len(chunks)
            }

        self.vector_store = PineconeVectorStore.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
        self.chat_history = []

        return {"total_chunks": len(all_chunks), "files": file_stats}

    # Query

    def query(self, question: str) -> dict:
        if not self.retriever:
            raise RuntimeError("No documents ingested. Please upload PDFs first.")

        # Step 1: condense follow-uP questions using history
        standalone_question = self._condense_question(question)

        # Step 2: retrieve relevant chunks
        docs = self.retriever.invoke(standalone_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Step 3: build history  string
        history_text = ""
        for human, ai in self.chat_history[-5:]:
            history_text += f"Human: {human}\nAssistant: {ai}\n\n"

        # Step 4:generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are PaperChat, an expert AI assistant. Answer questions based ONLY on "
             "the provided document context. If the answer is not in the context, say "
             "'I couldn't find that in the uploaded documents.' Be concise and cite the "
             "document where possible.\n\n"
             "CONVERSATION HISTORY:\n{history}\n\n"
             "DOCUMENT CONTEXT:\n{context}"),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "history":  history_text,
            "context":  context,
            "question": question,
        })

        # Step 5: save to history
        self.chat_history.append((question, answer))

        # Step 6: deduplicate sources
        sources = []
        seen = set()
        for doc in docs:
            key = (doc.metadata.get("source_file"), doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file":    doc.metadata.get("source_file", "Unknown"),
                    "page":    doc.metadata.get("page", "?"),
                    "snippet": doc.page_content[:200].strip()
                })

        return {"answer": answer, "sources": sources}

    def _condense_question(self, question: str) -> str:
        """Rephrase a follow-up into a standalone question using recent history."""
        if not self.chat_history:
            return question

        history_text = "\n".join(
            f"Human: {h}\nAssistant: {a}" for h, a in self.chat_history[-3:]
        )

        condense_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the conversation history and a follow-up question, rephrase the "
             "follow-up into a single standalone question with full context. "
             "Return ONLY the rephrased question.\n\nConversation history:\n{history}"),
            ("human", "Follow-up: {question}"),
        ])

        chain = condense_prompt | self.llm | StrOutputParser()
        return chain.invoke({"history": history_text, "question": question})

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []