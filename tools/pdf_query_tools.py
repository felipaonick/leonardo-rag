import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from pathlib import Path
from streamlit import session_state as ss


def embed_pdf_in_qdrant(pdf_path: Path, collection_name: str):
    """
    Loads a single PDF, splits it, and writes chunks into the given Qdrant collection.
    """
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://host.docker.internal:11434"
    )

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    texts_to_split = [doc.page_content for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents(texts_to_split)

    pdf_name = pdf_path.name

    for idx, chunk in enumerate(chunks):
        chunk.metadata["source"] = pdf_name
        chunk.metadata["chunk_id"] = str(uuid4())
        chunk.metadata["chunk_index"] = idx + 1

    # Ingest in the correct collection
    qdrant = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url="http://qdrant:6333",
        collection_name=collection_name
    )


def get_llm_model(use_ollama: bool = True, model: str = "llama3.2:3b"):
    """
    Get the appropriate LLM model based on configuration
    """

    if use_ollama:
        base_url = "http://host.docker.internal:11434"
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1
        )
    else:
        return ChatGroq(model="llama3-8b-8192")


def check_ollama_available():
    """
    Check if OLLAMA should be used based on environment or availability
    """

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        try:
            import requests
            # Support Docker OLLAMA or local OLLAMA
            base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
            response = requests.get(f"{base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    return False


@tool
def pdf_query(query: str) -> str:
    """
    Searches for the given query in the specified Qdrant collection.
    """
     # Recupera collection_name scelto dall'utente
    collection_name = ss.get("collection_name", "default_docs")

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://host.docker.internal:11434"
    )

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=collection_name,
        url="http://qdrant:6333"
    )

    found_docs = qdrant.similarity_search(query, k=4)
    if not found_docs:
        return "No relevant chunks found."
    else:
        return found_docs


# miglioria con QA chain
@tool
def pdf_query_with_qa(query: str) -> str:
    """Runs semantic search in the specified collection and QA chain"""

    use_ollama = check_ollama_available()

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    llm = get_llm_model(use_ollama, ollama_model)

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://host.docker.internal:11434"
    )

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=collection_name,
        url="http://qdrant:6333"
    )

    found_docs = qdrant.similarity_search(query, k=4)

    # Usiamo QA chain per dare risposte migliori

    if not found_docs:
        return "No relevant chunks found."

    # 3️⃣ QA chain
    try:
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        result = qa_chain.run(input_documents=found_docs, question=query)
        return result
    except Exception as e:
        print(f"QA chain failed: {e}")
        return str(found_docs)
