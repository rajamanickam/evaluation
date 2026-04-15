import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdfs(pdf_dir="pdfs", index_path="faiss_index"):
    """
    Load PDFs, chunk them, create embeddings, and store in FAISS vector database.
    
    Args:
        pdf_dir: Directory containing PDF files
        index_path: Path where FAISS index will be saved
    """
    print(f"Loading PDFs from {pdf_dir}...")
    
    # Step 1: Load PDF Files from a Directory
    all_docs = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"  Loading {filename}...")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    
    print(f"Loaded {len(all_docs)} document pages.")
    
    # Step 2: Convert Documents into Chunks
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")
    
    # Step 3: Create Embeddings and Store in Vector Database
    print("Creating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Step 4: Save the Vector Database
    print(f"Saving FAISS index to {index_path}...")
    vector_db.save_local(index_path)
    
    print("Ingestion complete!")
    return vector_db

if __name__ == "__main__":
    # Run ingestion
    ingest_pdfs(pdf_dir="pdfs", index_path="faiss_index")