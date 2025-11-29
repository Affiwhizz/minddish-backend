"""
Build ChromaDB from transcripts during Render deployment
"""

import os
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print("Building ChromaDB from transcripts...")

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment!")
    exit(1)

# Load curated index
with open("data/curated_index.json", "r") as f:
    curated_index = json.load(f)

print(f"Loaded {len(curated_index)} video entries")

# Create documents from transcripts
documents = []
transcripts_dir = Path("data/transcripts")

for entry in curated_index:
    video_id = entry['video_id']
    transcript_file = transcripts_dir / f"{video_id}.txt"
    
    if transcript_file.exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create document with metadata
        doc = Document(
            page_content=text,
            metadata={
                'video_id': video_id,
                'title': entry.get('title', 'Unknown'),
                'url': entry.get('url', ''),
                'collection': entry.get('cuisine', 'Unknown'),
                'method': entry.get('method', 'curated')
            }
        )
        documents.append(doc)

print(f"Created {len(documents)} documents")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(api_key=api_key)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"ChromaDB created with {len(chunks)} chunks at ./chroma_db")
print("Build complete!")