import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# After line: st.session_state.service = MindDishService(openai_api_key=api_key)
# Add these debug lines:

#import os
#st.write(f"Current directory: {os.getcwd()}")
#st.write(f"ChromaDB path: {os.path.abspath('./chroma_db')}")
#st.write(f"ChromaDB exists: {os.path.exists('./chroma_db')}")
#st.write(f"Files in chroma_db: {os.listdir('./chroma_db') if os.path.exists('./chroma_db') else 'N/A'}")

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.minddish_service import MindDishService

st.set_page_config(page_title="MindDish.ai Demo", page_icon="🍳", layout="wide")

# Initialize service with API key
if 'service' not in st.session_state:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in .env file!")
        st.stop()
    st.session_state.service = MindDishService(openai_api_key=api_key)

# Sidebar stats
with st.sidebar:
    st.header("📊 System Stats")
    try:
        stats = st.session_state.service.get_stats()
        st.metric("Total Chunks", stats['total_chunks'])
        st.metric("Total Videos", stats['total_videos'])
        st.metric("Available Tools", stats['tools_available'])
        
        st.header("🎥 Indexed Videos")
        videos = st.session_state.service.list_videos()
        st.write(f"{len(videos['videos'])} cooking videos")
        
        # Show video list
        if videos['videos']:
            for video in videos['videos'][:5]:  # Show first 5
                st.text(f"• {video['title']}")
    except Exception as e:
        st.error(f"Error loading stats: {e}")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about a recipe... (e.g., 'How to make Nigerian Efo Riro?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.service.chat(prompt)
                st.markdown(response['response'])
                
                # Show sources if available
                if response.get('sources'):
                    with st.expander("📚 Sources"):
                        for source in response['sources']:
                            st.write(f"- {source}")
                
                st.session_state.messages.append({"role": "assistant", "content": response['response']})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.service.clear_memory()
    st.rerun()