import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

def setup_chatbot(index_path="faiss_index"):
    """
    Load the FAISS index and set up the RAG chatbot.
    
    Args:
        index_path: Path where FAISS index is saved
        
    Returns:
        rag_chain: The configured RAG chain
    """
    print("Loading FAISS index...")
    
    # Step 1: Load Embeddings (must use same model as ingestion)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Step 2: Load the Vector Database
    vector_db = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Step 3: Set Up the Chat Model
    print("Setting up LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Step 4: Create Retriever
    retriever = vector_db.as_retriever()
    
    # Step 5: Create RAG Chain with Prompt
    prompt = PromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the question concisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    )
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)
    
    print("Chatbot ready!\n")
    return rag_chain

def run_chatbot(rag_chain):
    """
    Run the interactive chatbot loop.
    
    Args:
        rag_chain: The configured RAG chain
    """
    print("You can now ask questions about the PDF documents. Type 'exit' to quit.\n")
    
    while True:
        query = input("Enter your question: ")
        if query.strip().lower() in ("exit", "quit"):
            print("Exiting...")
            break
        
        # Query embedding happens automatically in retriever
        # Vector search and retrieval happens in rag_chain
        response = rag_chain.invoke({"input": query})
        print("\nAnswer:", response["answer"])
        print("-" * 50)

if __name__ == "__main__":
    # Set up and run the chatbot
    rag_chain = setup_chatbot(index_path="faiss_index")
    run_chatbot(rag_chain)