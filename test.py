import logging
from flask import Flask, request, jsonify
from functools import wraps
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import os
from pydantic import BaseModel, PrivateAttr
from typing import List

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

class ChromaDBRetriever(BaseRetriever, BaseModel):
    """Enhanced retriever for educational content"""
    _collection: any = PrivateAttr()
    _embedding_function: any = PrivateAttr()
    top_k: int = 5
    min_similarity: float = 0.65

    def __init__(self, **data):
        super().__init__(**data)
        chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        client = chromadb.HttpClient(host=chroma_host, port=8000)
        self._collection = client.get_collection(
            name="acca_doc6",
            embedding_function=embedding_function
        )
        self._embedding_function = embedding_function

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query = self._preprocess_query(query)
        logger.debug(f"Query after preprocessing: {query}")

        results = self._collection.query(
            query_texts=[query],
            n_results=self.top_k,
            include=['documents', 'distances', 'metadatas']
        )

        documents = []
        if results['documents'] and results['documents'][0]:
            for doc, distance, metadata in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                similarity = 1 / (1 + distance)
                if similarity >= self.min_similarity:
                    documents.append(
                        Document(
                            page_content=doc,
                            metadata={**metadata, "similarity_score": round(similarity, 3)}
                        )
                    )
        documents = sorted(documents, key=lambda x: x.metadata["similarity_score"], reverse=True)
        logger.debug(f"Retrieved documents: {[doc.metadata for doc in documents]}")
        return documents

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        accounting_terms = {
            "define": "definition",
            "what is": "definition",
            "example": "example",
            "calculation": "example",
            "required": "requirement",
            "disclose": "disclosure"
        }

        query_lower = query.lower()
        for term, category in accounting_terms.items():
            if term in query_lower:
                return f"{query} {category}"
        return query


def create_rag_chain():
    template = """You are an expert accounting professor teaching students about accounting standards. \
    Use the following pieces of context to answer the question. Pay special attention to definitions, examples, and requirements.\n\n    Context:\n    {context}\n\n    Question: {question}\n\n    Instructions:\n    1. Use ONLY the provided context for your answer.\n    2. If you find multiple relevant pieces, organize them logically.\n    3. If the answer isn't in the context, state clearly that it cannot be answered.\n\n    Answer: """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0,
        model="gpt-4-0125-preview",
        max_tokens=1000
    )
    retriever = ChromaDBRetriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        }
    )

    return qa_chain

def rag_response(question: str) -> dict:
    qa_chain = create_rag_chain()

    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get('result', '').strip()
        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result.get('source_documents', [])
        ]

        if not answer:
            logger.warning("No answer generated.")

        return {
            "text_answer": answer,
            "sources": sources,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error during chain execution: {e}")
        return {
            "text_answer": "",
            "sources": [],
            "success": False,
            "error": str(e)
        }


# Flask App Setup
app = Flask(__name__)
AUTH_TOKEN = 'Ab@123'

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Authorization header is missing"}), 401

        if token.startswith('Bearer '):
            token = token[7:]

        if token != AUTH_TOKEN:
            return jsonify({"error": "Invalid authentication token"}), 401

        return f(*args, **kwargs)
    return decorated

@app.route('/query', methods=['POST'])
@require_auth
def query():
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body", "success": False}), 400

        question = data['question']
        response = rag_response(question)
        return jsonify({"answer": response["text_answer"], "sources": response["sources"]})

    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
