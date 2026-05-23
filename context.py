import gdown
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def download_context_data(
    pdfs: Iterable[dict[str, str]],
    path: str = "./context_data",
) -> None:
    """
    Downloads PDFs and stores them in local storage.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        gdown.download(pdf["url"], f"{path}/{pdf['filename']}", quiet=True)


def load_context_data(path: str = "./context_data") -> list[Document]:
    """
    Loads multiple PDFs into LangChain Document objects.
    """
    loader = PyPDFDirectoryLoader(path)
    return loader.load()


def chunk_context_data(context_data: list[Document]) -> list[Document]:
    """
    Splits context data into overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    return text_splitter.split_documents(context_data)


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Gets an embedding model for vectorizing context data.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(
    chunks: list[Document],
    embedding_model: Embeddings | None = None,
    path: str = "./chromadb",
) -> Chroma:
    """
    Creates a persistent vector store from chunked documents.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=path,
    )


def get_vector_store(
    embedding_model: Embeddings | None = None,
    path: str = "./chromadb",
) -> Chroma:
    """
    Gets an existing persistent vector store.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=path,
        embedding_function=embedding_model,
    )


if __name__ == "__main__":
    pdfs = (
        {
            "url": "https://quanticedu.github.io/praxa/Longest Running Shows on Broadway 2025.pdf",
            "filename": "Longest Running Shows on Broadway.pdf",
        },
        {
            "url": "https://quanticedu.github.io/praxa/Every play and musical coming to the West End in 2025.pdf",
            "filename": "Every play and musical coming to the West End in 2025.pdf",
        },
    )

    download_context_data(pdfs)

    context_data = load_context_data()
    chunks = chunk_context_data(context_data)
    embedding_model = get_embedding_model()
    vector_store = create_vector_store(chunks, embedding_model)

    for num, chunk in enumerate(chunks[:5]):
        print("-----")
        print(f"Chunk {num}:")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content: {chunk.page_content[:500]}")

    retrieved_chunks = vector_store.similarity_search(
        "A play written by Ryan Calais Cameron."
    )

    print(f"\nQuery retrieved {len(retrieved_chunks)} chunks.")

    for chunk in retrieved_chunks:
        print("-----")
        print(f"Chunk content: {chunk.page_content[:500]}")
        print(f"Chunk metadata: {chunk.metadata}")
