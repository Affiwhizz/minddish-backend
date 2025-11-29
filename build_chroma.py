"""
Build ChromaDB from transcripts during Render deployment
"""

import os
import json
import re
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


def clean_transcript_text(text: str) -> str:
    """Clean transcript text by removing timestamps, music markers, and noise."""
    if not text:
        return ""
    
    # Remove timestamp patterns
    text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
    
    # Remove music/sound markers
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# Create documents from transcripts
documents = []
transcripts_dir = Path("data/transcripts")

for entry in curated_index:
    video_id = entry['video_id']
    transcript_file = transcripts_dir / f"{video_id}.txt"
    
    if transcript_file.exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Clean the transcript
        cleaned_text = clean_transcript_text(raw_text)
        
        if not cleaned_text:
            print(f"Warning: Empty transcript for {video_id}")
            continue
        
        # Create document with metadata (using 'collection' from curated_index.json)
        doc = Document(
            page_content=cleaned_text,
            metadata={
                'video_id': video_id,
                'title': entry.get('title', 'Unknown'),
                'url': entry.get('url', ''),
                'collection': entry.get('collection', 'Unknown'),  # Fixed: was 'cuisine'
                'method': entry.get('transcript_source', 'local')
            }
        )
        documents.append(doc)
        print(f"  Loaded: {entry.get('title', video_id)}")
    else:
        print(f"  Warning: Transcript not found for {video_id}")

print(f"\nCreated {len(documents)} documents")

# Split into chunks (matching notebook settings)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

# Remove existing chroma_db if it exists (clean rebuild)
chroma_path = Path("./chroma_db")
if chroma_path.exists():
    import shutil
    shutil.rmtree(chroma_path)
    print("Removed existing chroma_db")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="minddish_curated"
)

print(f"\nChromaDB created with {len(chunks)} chunks at ./chroma_db")
print(f"Collection: minddish_curated")
print("Build complete!")