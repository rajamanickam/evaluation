"""
Simple RAG System Evaluation using RAGAS with Google Gemini

Install required packages:
pip install ragas 
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_huggingface import  HuggingFaceEmbeddings


# Set your Google Gemini API key
 
os.environ["GOOGLE_API_KEY"] =os.getenv("GEMINI_API_KEY")


# Initialize Gemini LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    temperature=0.1,
   # google_api_key=os.getenv("GEMINI_API_KEY"),
    convert_system_message_to_human=True
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample data from your RAG system
# Each entry should contain:
# - question: user's question
# - answer: RAG system's generated answer
# - contexts: list of retrieved context chunks
# - ground_truth: the correct/expected answer

data = {
    "question": [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is photosynthesis?"
    ],
    "answer": [
        "The capital of France is Paris, which is located in the north-central part of the country.",
        "Alexander Graham Bell is credited with inventing the telephone in 1876.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy."
    ],
    "contexts": [
        ["Paris is the capital and most populous city of France. It is located in north-central France."],
        ["Alexander Graham Bell was a scientist and inventor. He is credited with inventing the first practical telephone in 1876."],
        ["Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose."]
    ],
    "ground_truth": [
        "Paris is the capital of France.",
        "Alexander Graham Bell invented the telephone.",
        "Photosynthesis is the process plants use to convert light into chemical energy."
    ]
}

# Convert to Dataset format
dataset = Dataset.from_dict(data)

# Define metrics to evaluate
metrics = [
    faithfulness,        # Measures factual consistency of answer with context
    answer_relevancy,    # Measures how relevant the answer is to the question
    context_recall,      # Measures if retrieved context contains ground truth
    context_precision,   # Measures if relevant contexts are ranked higher
]

# Evaluate the RAG system with Gemini
print("Evaluating RAG system with Google Gemini...")
print("This may take a few moments...\n")

try:
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    
    # Display results
    print("\n" + "="*50)
    print("RAGAS Evaluation Results (Using Gemini)")
    print("="*50)
    print(f"\nFaithfulness Score: {result['faithfulness']:.4f}")
    print(f"Answer Relevancy Score: {result['answer_relevancy']:.4f}")
    print(f"Context Recall Score: {result['context_recall']:.4f}")
    print(f"Context Precision Score: {result['context_precision']:.4f}")
    print("\n" + "="*50)
    
    # Convert to pandas DataFrame for detailed view
    df = result.to_pandas()
    print("\nDetailed Results:")
    print(df)
    
    # Save results
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\nResults saved to 'rag_evaluation_results.csv'")
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you've set a valid GOOGLE_API_KEY")
    print("2. Get your API key from: https://makersuite.google.com/app/apikey")
    print("3. Check your internet connection")
    print("4. Verify the API key has proper permissions")