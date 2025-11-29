"""
MindDish.ai Service - Production Backend
Multilingual RAG-based cooking assistant Uses pre-built ChromaDB and local transcripts

"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Service instances cache (session-based)
_service_instances = {}

class MindDishService:
    """Complete MindDish.ai RAG service with web search"""
    
    def __init__(self, openai_api_key: str, tavily_api_key: Optional[str] = None, session_id: str = "default"):
        self.session_id = session_id
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        # Load existing ChromaDB
        self.vectorstore = self._load_vectorstore()
        
        # Initialize chat history
        self.chat_history = []
        self.awaiting_permission = False
        self.pending_question = None
        
        # Create tools (includes web search!)
        self.tools = self._create_tools()
        
        # Create permission-aware prompt manager
        self.prompt_manager = self._create_prompt_manager()
    
    def _load_vectorstore(self):
        """Load pre-built ChromaDB vectorstore"""
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        
        # Path to your ChromaDB
        chroma_path = Path("./chroma_db")
        
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {chroma_path}. "
                "Please copy your data/chroma_minddish folder to ./chroma_db"
            )
        
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings
        )
        
        return vectorstore
    
    def _create_tools(self) -> List[StructuredTool]:
        """Create all MindDish tools including WEB SEARCH"""
        
        tools = []
        
        # Tool 1: Video QA (primary RAG)
        def video_qa(query: str) -> str:
            """Search indexed cooking videos and answer questions"""
            docs = self.vectorstore.similarity_search(query, k=5)
            if not docs:
                return "No relevant information found in indexed videos."
            
            context = "\n\n".join([d.page_content for d in docs[:3]])
            
            prompt = f"""Based on these cooking video transcripts:

{context}

Question: {query}

Answer the question using ONLY the information above. If the answer isn't in the transcripts, say so."""
            
            response = self.llm.invoke(prompt)
            
            # Add source info
            sources = []
            for doc in docs[:3]:
                title = doc.metadata.get('title', 'Unknown')
                if title not in sources:
                    sources.append(title)
            
            return f"{response.content}\n\nSources: {', '.join(sources)}"
        
        tools.append(StructuredTool.from_function(
            func=video_qa,
            name="video_qa_tool",
            description="Search and answer questions from indexed cooking videos. Use this FIRST for any cooking question."
        ))
        
        # Tool 2: WEB SEARCH - THE WOW FACTOR!
        def web_search_cooking(query: str) -> str:
            """Search the web for cooking information, recipes, substitutions, nutrition"""
            try:
                # Try Tavily first (if API key available)
                if self.tavily_api_key:
                    from tavily import TavilyClient
                    client = TavilyClient(api_key=self.tavily_api_key)
                    results = client.search(query, max_results=3)
                    
                    response = f"Web search results for '{query}':\n\n"
                    for i, result in enumerate(results.get('results', []), 1):
                        response += f"{i}. {result['title']}\n"
                        response += f"   {result['content'][:200]}...\n"
                        response += f"   Source: {result['url']}\n\n"
                    
                    return response
                else:
                    # Fallback to DuckDuckGo (free, no API key)
                    from duckduckgo_search import DDGS
                    results = DDGS().text(query, max_results=3)
                    
                    response = f"Web search results for '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        response += f"{i}. {result['title']}\n"
                        response += f"   {result['body'][:200]}...\n"
                        response += f"   Source: {result['href']}\n\n"
                    
                    return response
                    
            except Exception as e:
                return f"Web search failed: {str(e)}"
        
        tools.append(StructuredTool.from_function(
            func=web_search_cooking,
            name="web_search_cooking_tool",
            description="Search the web for cooking tips, recipes, nutrition info, substitutions. Use AFTER checking indexed videos when user grants permission."
        ))
        
        # Tool 3: List videos
        def list_videos(query: str = "") -> str:
            """List all indexed videos"""
            collection = self.vectorstore._collection
            all_data = collection.get()
            
            video_titles = set()
            for metadata in all_data['metadatas']:
                if metadata and 'title' in metadata:
                    video_titles.add(metadata['title'])
            
            return f"Indexed videos ({len(video_titles)}):\n" + "\n".join(f"- {title}" for title in sorted(video_titles))
        
        tools.append(StructuredTool.from_function(
            func=list_videos,
            name="list_videos_tool",
            description="List all indexed cooking videos"
        ))
        
        # Tool 4: Extract ingredients
        def extract_ingredients(video_title: str) -> str:
            """Extract ingredients from a specific video"""
            docs = self.vectorstore.similarity_search(f"ingredients {video_title}", k=3)
            
            if not docs:
                return f"No ingredients found for '{video_title}'"
            
            context = "\n\n".join([d.page_content for d in docs])
            
            prompt = f"""Extract all ingredients mentioned in this cooking video transcript:

{context}

List only the ingredients, one per line."""
            
            response = self.llm.invoke(prompt)
            return response.content
        
        tools.append(StructuredTool.from_function(
            func=extract_ingredients,
            name="extract_ingredients_tool",
            description="Extract ingredients list from a specific video"
        ))
        
        # Tool 5: Cooking time
        def cooking_time(video_title: str) -> str:
            """Get cooking time from video"""
            docs = self.vectorstore.similarity_search(f"cooking time {video_title}", k=3)
            
            if not docs:
                return f"No cooking time found for '{video_title}'"
            
            context = "\n\n".join([d.page_content for d in docs])
            
            prompt = f"""From this cooking video, extract timing information:

{context}

List prep time, cook time, and total time if mentioned."""
            
            response = self.llm.invoke(prompt)
            return response.content
        
        tools.append(StructuredTool.from_function(
            func=cooking_time,
            name="cooking_time_tool",
            description="Get cooking and prep times from a video"
        ))
        
        return tools
    
    def _create_prompt_manager(self):
        """Create permission-aware prompt manager"""
        
        class PromptManager:
            def __init__(self, llm, vectorstore):
                self.llm = llm
                self.vectorstore = vectorstore
            
            def get_smart_response(self, question: str, permission_granted: bool = False):
                """Check transcripts first, ask permission if not found"""
                
                # Search videos
                docs = self.vectorstore.similarity_search(question, k=5)
                
                if not docs or len(docs) == 0:
                    return {
                        "answer": "I don't have this information in my indexed videos. Would you like me to search the web for this? (Reply 'yes' to proceed)",
                        "needs_permission": True,
                        "sources": [],
                        "source_type": "none"
                    }
                
                # Found in videos
                context = "\n\n".join([d.page_content for d in docs[:3]])
                
                prompt = f"""You are MindDish.ai. Answer based on these video transcripts:

{context}

Question: {question}

If the transcripts contain the answer, provide a helpful response with specific details.
If the transcripts DON'T contain the answer, say: "I don't have this information in my indexed videos."
"""
                
                response = self.llm.invoke(prompt)
                answer = response.content
                
                # Check if LLM says it doesn't have the info
                if "don't have this information" in answer.lower():
                    return {
                        "answer": "I don't have this information in my indexed videos. Would you like me to search the web for this? (Reply 'yes' to proceed)",
                        "needs_permission": True,
                        "sources": [],
                        "source_type": "none"
                    }
                
                sources = []
                for doc in docs[:3]:
                    sources.append({
                        'title': doc.metadata.get('title', 'Unknown'),
                        'url': doc.metadata.get('url', ''),
                        'cuisine': doc.metadata.get('collection', 'Unknown')
                    })
                
                return {
                    "answer": answer,
                    "needs_permission": False,
                    "sources": sources,
                    "source_type": "transcripts"
                }
        
        return PromptManager(self.llm, self.vectorstore)
    
    def chat(self, message: str, enhance_with_web: bool = True) -> Dict:
        """Main chat method with permission handling and WEB SEARCH"""
        
        # Check if waiting for permission
        if self.awaiting_permission:
            if message.lower() in ['yes', 'y', 'ok', 'sure', 'go ahead', 'search', 'search web']:
                # User granted permission - USE WEB SEARCH!
                
                # Find the web search tool
                web_search_tool = None
                for tool in self.tools:
                    if tool.name == "web_search_cooking_tool":
                        web_search_tool = tool
                        break
                
                if web_search_tool and enhance_with_web:
                    # Execute web search with the pending question
                    web_result = web_search_tool.func(self.pending_question)
                    
                    # Format response
                    answer = f"Using web search (not from indexed videos)\n\n{web_result}"
                    source_type = "web_search"
                else:
                    # Fallback to general knowledge
                    prompt = f"""The user asked: {self.pending_question}

This information is not in the indexed cooking videos. Provide a helpful answer using your general cooking knowledge.

Note: This is general cooking knowledge, not from the video database."""
                    
                    response = self.llm.invoke(prompt)
                    answer = f"Using general cooking knowledge (not from indexed videos)\n\n{response.content}"
                    source_type = "general_knowledge"
                
                self.awaiting_permission = False
                self.pending_question = None
                
                self.chat_history.append(HumanMessage(content=message))
                self.chat_history.append(AIMessage(content=answer))
                
                return {
                    "response": answer,
                    "status": "success_with_permission",
                    "source_type": source_type
                }
            else:
                # User declined
                self.awaiting_permission = False
                self.pending_question = None
                
                response = "No problem! I'll stick to my indexed videos. What else would you like to know about cooking?"
                
                self.chat_history.append(HumanMessage(content=message))
                self.chat_history.append(AIMessage(content=response))
                
                return {
                    "response": response,
                    "status": "permission_declined"
                }
        
        # Normal query processing
        result = self.prompt_manager.get_smart_response(message)
        
        if result['needs_permission']:
            self.awaiting_permission = True
            self.pending_question = message
            
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=result['answer']))
            
            return {
                "response": result['answer'],
                "status": "awaiting_permission"
            }
        
        # Answer found in videos
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=result['answer']))
        
        return {
            "response": result['answer'],
            "status": "success",
            "sources": result.get('sources', []),
            "source_type": result.get('source_type')
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        collection = self.vectorstore._collection
        count = collection.count()
        
        # Get unique videos
        all_data = collection.get()
        video_titles = set()
        for metadata in all_data['metadatas']:
            if metadata and 'title' in metadata:
                video_titles.add(metadata['title'])
        
        return {
            "total_chunks": count,
            "total_videos": len(video_titles),
            "tools_available": len(self.tools),
            "messages_in_memory": len(self.chat_history),
            "awaiting_permission": self.awaiting_permission,
            "web_search_enabled": self.tavily_api_key is not None or True
        }
    
    def list_videos(self) -> Dict:
        """List all indexed videos"""
        collection = self.vectorstore._collection
        all_data = collection.get()
        
        videos = []
        seen_titles = set()
        
        for metadata in all_data['metadatas']:
            if metadata and 'title' in metadata:
                title = metadata['title']
                if title not in seen_titles:
                    videos.append({
                        'title': title,
                        'url': metadata.get('url', ''),
                        'cuisine': metadata.get('collection', 'Unknown'),
                        'video_id': metadata.get('video_id', '')
                    })
                    seen_titles.add(title)
        
        return {"videos": videos}
    
    def clear_memory(self):
        """Clear conversation history"""
        self.chat_history = []
        self.awaiting_permission = False
        self.pending_question = None
    
    def index_new_video(self, youtube_url: str, custom_name: Optional[str] = None) -> Dict:
        """Index a new video (using local transcripts)"""
        return {
            "status": "error",
            "message": "Dynamic video indexing not yet implemented. Use pre-indexed videos."
        }


def get_minddish_service(openai_api_key: str, tavily_api_key: Optional[str] = None, session_id: str = "default") -> MindDishService:
    """Get or create service instance for session"""
    
    if session_id not in _service_instances:
        _service_instances[session_id] = MindDishService(
            openai_api_key=openai_api_key,
            tavily_api_key=tavily_api_key,
            session_id=session_id
        )
    
    return _service_instances[session_id]