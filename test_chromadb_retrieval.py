"""
Test ChromaDB Retrieval Directly
Run this from minddish-backend directory
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print(" ERROR: OPENAI_API_KEY not found in .env file!")
    exit(1)

print(f" API Key loaded: {api_key[:10]}...")

# Load ChromaDB
print("\n📂 Loading ChromaDB from ./chroma_db...")
embeddings = OpenAIEmbeddings()

try:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    print(" ChromaDB loaded successfully")
except Exception as e:
    print(f" Failed to load ChromaDB: {e}")
    exit(1)

# Test queries
test_queries = [
    "ingredients for Eforiro soup",
    "Nigerian cooking",
    "palm oil vegetables",
    "how to make soup"
]

print("\n" + "="*60)
print(" TESTING CHROMADB RETRIEVAL")
print("="*60)

for query in test_queries:
    print(f"\n Query: '{query}'")
    print("-" * 60)
    
    try:
        results = vectorstore.similarity_search(query, k=3)
        
        if results:
            print(f" Found {len(results)} results:\n")
            
            for i, doc in enumerate(results):
                source = doc.metadata.get('source', 'Unknown')
                video_id = doc.metadata.get('video_id', 'Unknown')
                chunk_id = doc.metadata.get('chunk_id', '?')
                
                print(f"Result {i+1}:")
                print(f"  Source: {source}")
                print(f"  Video ID: {video_id}")
                print(f"  Chunk: {chunk_id}")
                print(f"  Content: {doc.page_content[:150]}...")
                print()
        else:
            print("❌ NO RESULTS FOUND for this query")
    
    except Exception as e:
        print(f"❌ Error during search: {e}")

# Check collection stats
print("\n" + "="*60)
print("📊 CHROMADB STATISTICS")
print("="*60)

try:
    collection = vectorstore._collection
    count = collection.count()
    print(f"Total documents in ChromaDB: {count}")
    
    if count > 0:
        # Get all metadata to see what's actually stored
        all_data = collection.get()
        
        # Extract unique video IDs
        video_ids = set()
        for metadata in all_data['metadatas']:
            if metadata and 'video_id' in metadata:
                video_ids.add(metadata['video_id'])
        
        print(f"Unique videos: {len(video_ids)}")
        for vid_id in video_ids:
            print(f"  - {vid_id}")
    else:
        print("⚠️  ChromaDB is EMPTY! No documents indexed.")

except Exception as e:
    print(f" Error getting stats: {e}")

print("\n" + "="*60)
print(" TEST COMPLETE")
print("="*60)