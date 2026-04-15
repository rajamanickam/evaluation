from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import NLTKTextSplitter

text = """
LangChain is a framework\n\n for\n\n developing applications powered by language models.

It enables applications that are context-aware and reasoned.
"""
chunking_strategy="TokenTextSplitter"

if chunking_strategy == "CharacterTextSplitter":

    #Fixed-Size Chunking (Character-based)
    text_splitter = CharacterTextSplitter(
        chunk_size=5,
        chunk_overlap=2
    )
elif chunking_strategy == "RecursiveCharacterTextSplitter":
    #Recursive Character Chunking (Most Recommended)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=5
    )

elif chunking_strategy == "TokenTextSplitter":
#Token-Based Chunking 

    text_splitter = TokenTextSplitter(
    chunk_size=5,
    chunk_overlap=2
    )
elif chunking_strategy == "NLTKTextSplitter":
    #Sentence-Based Chunking (NLTK)
    text_splitter = NLTKTextSplitter(
    chunk_size=2,  # number of sentences
    chunk_overlap=1
    )

chunks = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")



