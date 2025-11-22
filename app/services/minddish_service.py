"""
MindDish.ai Service - Production Backend
Multilingual RAG-based cooking assistant with session management

"""

import os
import re
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool


import tempfile
import yt_dlp
from openai import OpenAI


# Web search wrapper using Tavily

def web_search(query: str, max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
    """
    Tavily web search wrapper for MindDish.ai
    
    Args:
        query: Search query
        max_results: Number of results (default: 5)
        search_depth: "basic" (cheaper) or "advanced" (deeper)
    
    Returns:
        dict with 'status', 'results', 'answer'
    """
    try:
        from tavily import TavilyClient
        
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not tavily_api_key:
            return {
                "status": "error",
                "results": [],
                "message": "TAVILY_API_KEY not found"
            }
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Search with Tavily
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=[],
            exclude_domains=[]
        )
        
        # Format results
        formatted_results = []
        for result in response.get('results', []):
            formatted_results.append({
                'title': result.get('title', 'No title'),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0)
            })
        
        return {
            "status": "success",
            "results": formatted_results,
            "query": query,
            "answer": response.get('answer', '')
        }
    
    except ImportError:
        return {
            "status": "error",
            "results": [],
            "message": "tavily-python not installed"
        }
    except Exception as e:
        return {
            "status": "error",
            "results": [],
            "message": f"Web search failed: {str(e)}"
        }


# MindDish Tools

def create_minddish_tools(vectorstore, llm, indexed_videos):
    """
    Create ALL 17 MindDish tools matching notebook names exactly
    """
    
    # Core RAG tools (3)
    
    @tool
    def video_qa_tool(question: str) -> str:
        """Answer cooking questions using RAG across all indexed videos"""
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(question)
            
            if not docs:
                return "I don't have that information in the indexed videos."
            
            context = "\n\n".join([doc.page_content for doc in docs])
            source = docs[0].metadata.get('source', 'Unknown video')
            
            prompt = f"""Based on this cooking video content, answer the question.
            
Context from {source}:
{context}

Question: {question}

Answer:"""
            
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def transcript_search_tool(keyword: str) -> str:
        """Search for specific keyword mentions across all cooking videos"""
        results = vectorstore.similarity_search(keyword, k=5)
        
        video_counts = {}
        for doc in results:
            video_title = doc.metadata.get('source', 'Unknown')
            video_counts[video_title] = video_counts.get(video_title, 0) + 1
        
        if not video_counts:
            return f"'{keyword}' not found in any videos"
        
        response = f"Found '{keyword}' in:\n"
        for video, count in sorted(video_counts.items(), key=lambda x: x[1], reverse=True):
            response += f"  - {video}: {count} mention(s)\n"
        
        return response
    
    @tool
    def list_videos_tool(query: str = "") -> str:
        """List all indexed cooking videos with details"""
        if not indexed_videos:
            return "No videos indexed yet."
        
        response = f"Indexed Cooking Videos ({len(indexed_videos)}):\n\n"
        for vid_id, info in indexed_videos.items():
            response += f"- {info.get('title', 'Unknown')}\n"
            response += f"  ID: {vid_id}\n"
            response += f"  Chunks: {info.get('chunks', 0)}\n\n"
        return response
    
    # Video analysis tools (3)
    
    @tool
    def video_summary_tool(video_title_or_id: str) -> str:
        """Generate a comprehensive summary of a specific cooking video"""
        video_id = None
        for vid, info in indexed_videos.items():
            if video_title_or_id.lower() in str(info.get('title', '')).lower() or video_title_or_id == vid:
                video_id = vid
                break
        
        if not video_id:
            return f"Video '{video_title_or_id}' not found. Use list_videos_tool to see available videos."
        
        try:
            results = vectorstore.similarity_search("recipe ingredients steps instructions", k=10)
            
            if not results:
                return "No content found for video"
            
            all_text = " ".join([doc.page_content for doc in results[:8]])
            summary_prompt = f"""Summarize this cooking video including:
- Dish being made
- Main ingredients
- Key cooking steps
- Cooking time and techniques

Content: {all_text[:2000]}

Summary:"""
            
            summary = llm.invoke(summary_prompt)
            return summary.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def compare_videos_tool(topic: str) -> str:
        """Compare how different cooking videos discuss a specific topic (ingredient, technique, dish)"""
        results = vectorstore.similarity_search(topic, k=8)
        
        video_content = {}
        for doc in results:
            video_title = doc.metadata.get('source', 'Unknown')
            if video_title not in video_content:
                video_content[video_title] = []
            video_content[video_title].append(doc.page_content)
        
        if not video_content:
            return f"No videos discuss '{topic}'"
        
        comparison = f"Comparison: '{topic}'\n\n"
        for video, chunks in video_content.items():
            excerpt = ' '.join(chunks[:2])
            comparison += f"{video}:\n   {excerpt}...\n\n"
        
        return comparison
    
    @tool
    def find_related_videos_tool(topic: str) -> str:
        """Find which cooking videos are most relevant to a topic"""
        results = vectorstore.similarity_search(topic, k=10)
        
        video_scores = {}
        for doc in results:
            video_title = doc.metadata.get('source', 'Unknown')
            video_scores[video_title] = video_scores.get(video_title, 0) + 1
        
        if not video_scores:
            return f"No videos found about '{topic}'"
        
        response = f"Videos related to '{topic}':\n"
        for video, score in sorted(video_scores.items(), key=lambda x: x[1], reverse=True):
            response += f"  - {video} ({score} relevant chunks)\n"
        
        return response
    
    # Recipe detail tools (4)
    @tool
    def extract_ingredients_tool(recipe_name: str) -> str:
        """Extract all ingredients mentioned in a specific cooking video"""
        results = vectorstore.similarity_search(f"{recipe_name} ingredients", k=5)
        
        if not results:
            return f"No ingredients found for '{recipe_name}'"
        
        context = "\n".join([doc.page_content for doc in results[:3]])
        prompt = f"""Extract ONLY the ingredients list from this recipe content:

{context}

List the ingredients clearly:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    @tool
    def cooking_time_tool(recipe_name: str) -> str:
        """Extract cooking times, prep times, and total time from a video"""
        results = vectorstore.similarity_search(f"{recipe_name} time minutes hours temperature", k=4)
        
        if not results:
            return f"No timing information found for '{recipe_name}'"
        
        context = "\n".join([doc.page_content for doc in results[:3]])
        prompt = f"""Extract timing information from this content:

{context}

Provide: prep time, cooking time, total time, temperatures:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    @tool
    def equipment_checker_tool(recipe_name: str) -> str:
        """List all cooking equipment and tools needed for a recipe"""
        results = vectorstore.similarity_search(f"{recipe_name} equipment tools pot pan", k=4)
        
        if not results:
            return f"No equipment information found for '{recipe_name}'"
        
        context = "\n".join([doc.page_content for doc in results[:3]])
        prompt = f"""List the cooking equipment and tools needed:

{context}

Equipment needed:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    @tool
    def cultural_context_tool(recipe_name: str) -> str:
        """Explain cultural context, traditional techniques, or regional variations. Combines video content + latest web knowledge."""
        results = vectorstore.similarity_search(f"{recipe_name} culture traditional origin history", k=4)
        
        cultural_info = ""
        if results:
            context = "\n".join([doc.page_content for doc in results[:2]])
            prompt = f"""Describe the cultural context of this dish:

{context}

Explain: origin, cultural significance, traditional occasions:"""
            response = llm.invoke(prompt)
            cultural_info = f"From videos:\n{response.content}\n\n"
        
        # Enhance with web search
        search_result = web_search(f"{recipe_name} cultural history traditional", max_results=2)
        if search_result["status"] == "success" and search_result["results"]:
            web_info = "\n".join([
                f"- {r['content']}"
                for r in search_result["results"][:2]
            ])
            cultural_info += f"Additional context:\n{web_info}"
        
        return cultural_info if cultural_info else f"No cultural information found for '{recipe_name}'"
    
    # Advanced tools (5)
    
    @tool
    def smart_substitution_tool(ingredient: str) -> str:
        """Suggest safe ingredient substitutions with 3-layer safety system: Layer 1: Dangerous pairs blacklist, Layer 2: Check indexed videos, Layer 3: Web search with safety prompts"""
        
        # Layer 1: Dangerous pairs blacklist
        dangerous_substitutions = {
            "baking soda": ["baking powder", "yeast"],
            "baking powder": ["baking soda", "yeast"],
            "salt": ["sugar"],
            "sugar": ["salt"]
        }
        
        if ingredient.lower() in dangerous_substitutions:
            return f"WARNING: {ingredient} substitutions can be dangerous. Common mistakes: {', '.join(dangerous_substitutions[ingredient.lower()])}. Check a cookbook for safe alternatives."
        
        # Layer 2: Check indexed videos
        video_results = vectorstore.similarity_search(f"{ingredient} substitute alternative replace", k=3)
        
        if video_results and len(video_results) > 0:
            context = "\n".join([doc.page_content for doc in video_results[:2]])
            if ingredient.lower() in context.lower():
                video_source = video_results[0].metadata.get('source', 'a cooking video')
                prompt = f"""Based on this cooking video, suggest substitutions for {ingredient}:

{context}

What can substitute {ingredient}?"""
                response = llm.invoke(prompt)
                return f"From {video_source}:\n{response.content}"
        
        # Layer 3: Web search with safety focus
        search_result = web_search(f"safe cooking substitute for {ingredient}", max_results=3)
        
        if search_result["status"] == "success" and search_result["results"]:
            web_info = "\n".join([
                f"- {r['title']}: {r['content']}" 
                for r in search_result["results"][:2]
            ])
            
            return f"Web sources suggest:\n{web_info}\n\nAlways consider allergies and dietary restrictions!"
        
        # Layer 4: Honest fallback
        return f"I don't have reliable substitution information for '{ingredient}'. Please consult a cooking resource or use your best judgment based on similar ingredients."
    
    @tool
    def recipe_fact_check_tool(claim: str) -> str:
        """Verify if a cooking claim or technique is mentioned in the videos"""
        results = vectorstore.similarity_search(claim, k=5)
        
        if not results:
            return f"Cannot verify: '{claim}' - not mentioned in indexed videos"
        
        context = "\n".join([doc.page_content for doc in results[:3]])
        prompt = f"""Is this claim mentioned or supported in the content?

Claim: {claim}

Content: {context}

Verification:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    @tool
    def suggest_questions_tool(video_or_topic: str) -> str:
        """Suggest interesting questions to ask about the cooking videos"""
        results = vectorstore.similarity_search(video_or_topic, k=5)
        
        if not results:
            return "No videos found to suggest questions about"
        
        context = "\n".join([doc.page_content for doc in results[:3]])
        prompt = f"""Based on this cooking content, suggest 5 interesting questions someone might ask:

{context}

Questions:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    @tool
    def web_search_cooking_tool(query: str) -> str:
        """Search the web for cooking information not available in indexed videos. Use this for nutrition, latest trends, restaurant info, or anything not in videos."""
        search_result = web_search(query, max_results=5)
        
        if search_result["status"] == "success":
            if search_result.get("answer"):
                return f"Web search results for '{query}':\n{search_result['answer']}"
            
            if search_result["results"]:
                results = "\n\n".join([
                    f"{r['title']}:\n{r['content']}"
                    for r in search_result["results"][:3]
                ])
                return f"Web search results:\n{results}"
        
        return f"Could not find web information for '{query}'"
    
    @tool
    def cooking_expert_analysis_tool(question: str) -> str:
        """Get detailed culinary analysis combining video content + latest web knowledge. Use for complex questions, comparisons, or deep dives."""
        
        # Get video context
        video_results = vectorstore.similarity_search(question, k=5)
        video_context = ""
        if video_results:
            video_context = "\n".join([doc.page_content for doc in video_results[:3]])
        
        # Get web context
        search_result = web_search(question, max_results=3, search_depth="advanced")
        web_context = ""
        if search_result["status"] == "success" and search_result["results"]:
            web_context = "\n".join([
                r['content']
                for r in search_result["results"][:2]
            ])
        
        # Combine both
        combined_prompt = f"""Provide detailed culinary analysis:

Question: {question}

Video content:
{video_context}

Web research:
{web_context}

Expert analysis:"""
        
        response = llm.invoke(combined_prompt)
        return response.content
    
    # Language support tool (1)
    
    @tool
    def translate_recipe_tool(recipe_name: str, target_language: str) -> str:
        """Translate cooking instructions or recipes to another language"""
        results = vectorstore.similarity_search(recipe_name, k=3)
        
        if results:
            context = results[0].page_content
            prompt = f"""Translate this recipe content to {target_language}:

{context}

Translation:"""
            response = llm.invoke(prompt)
            return response.content
        
        return f"Recipe '{recipe_name}' not found in indexed videos"
    
    # Nutrition calculator tool (1)
    
    @tool
    def nutrition_calculator_tool(recipe_or_ingredients: str) -> str:
        """Calculate approximate nutritional information for a recipe from a video. Uses video ingredients + web nutrition data."""
        
        # Try to find recipe in videos first
        video_results = vectorstore.similarity_search(f"{recipe_or_ingredients} ingredients", k=3)
        
        ingredient_context = ""
        if video_results:
            ingredient_context = "\n".join([doc.page_content for doc in video_results[:2]])
        
        # Use web search for accurate nutrition data
        search_result = web_search(
            f"calories protein carbs fat {recipe_or_ingredients}", 
            max_results=4
        )
        
        nutrition_data = ""
        if search_result["status"] == "success":
            if search_result.get("answer"):
                nutrition_data = search_result["answer"]
            elif search_result["results"]:
                nutrition_data = "\n".join([
                    f"- {r['title']}: {r['content']}"
                    for r in search_result["results"][:3]
                ])
        
        if not nutrition_data and not ingredient_context:
            return f"Could not calculate nutrition for '{recipe_or_ingredients}'. Try being more specific about ingredients."
        
        # Generate comprehensive response
        response_parts = []
        
        if ingredient_context:
            response_parts.append(f"Recipe ingredients found:\n{ingredient_context}\n")
        
        if nutrition_data:
            response_parts.append(f"Nutritional information:\n{nutrition_data}")
        else:
            response_parts.append("Note: Exact nutritional values vary based on portions and preparation methods.")
        
        return "\n".join(response_parts)
    
    # Return all 17 tools
    return [
        video_qa_tool,
        transcript_search_tool,
        list_videos_tool,
        video_summary_tool,
        compare_videos_tool,
        find_related_videos_tool,
        extract_ingredients_tool,
        cooking_time_tool,
        equipment_checker_tool,
        smart_substitution_tool,
        recipe_fact_check_tool,
        suggest_questions_tool,
        web_search_cooking_tool,
        cultural_context_tool,
        cooking_expert_analysis_tool,
        translate_recipe_tool,
        nutrition_calculator_tool,
    ]


# Session Management

class SessionManager:
    """Simple session management without collecting personal data"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        """Create a new anonymous session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "indexed_videos": {},
            "conversation_count": 0,
            "chat_history": []
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Main Service Class

class MindDishService:
    """
    MindDish.ai Production Service
    Multimodal RAG-based multilingual cooking assistant
    """
    
    def __init__(
        self, 
        openai_api_key: str, 
        chroma_path: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize MindDish service
        
        Args:
            openai_api_key: OpenAI API key
            chroma_path: Path to ChromaDB storage (default: ./chroma_db)
            tavily_api_key: Tavily API key for web search
            session_id: Optional session ID for multi-user support
        """
        print(f"Starting MindDishService init for session: {session_id or 'default'}")
        
        self.session_id = session_id or "default"
        
        # Set API keys
        print("Setting API keys...")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        # Initialize components
        print("Setting chroma path...")
        # FIXED: Use shared ChromaDB for all sessions so indexed videos are accessible everywhere
        self.chroma_path = chroma_path or "./chroma_db"
        
        print("Initializing embeddings...")  
        self.embeddings = OpenAIEmbeddings()
        
        print("Initializing LLM...")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at {self.chroma_path}...")
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        
        # Track indexed videos per session
        self.indexed_videos: Dict[str, Dict] = {}
        
        # Create tools
        print("Creating 17 tools...")
        self.tools = create_minddish_tools(
            self.vectorstore, 
            self.llm, 
            self.indexed_videos
        )
        
        # Conversation memory
        self.chat_history: List = []
        self.conversation_count = 0
        
        # Setup agent
        print("Setting up agent...")
        self.agent = None
        self._setup_agent()
        
        print(f"MindDish.ai initialized for session: {self.session_id}")
        print(f"- {len(self.tools)} tools available")
        print(f"- ChromaDB: {self.chroma_path}")
        print(f"- Web search: {'enabled' if os.getenv('TAVILY_API_KEY') else 'disabled'}")
    
    def _setup_agent(self):
        """Initialize LangChain agent with multilingual system prompt"""
        system_prompt = """You are MindDish.ai, a helpful and knowledgeable multilingual cooking assistant.

LANGUAGE SUPPORT:
You can understand and respond in ANY language including:
- English
- Portuguese (português europeu and português brasileiro)
- Spanish (Español)
- French (Français)
- All Nigerian languages (Pidgin, Yoruba, Igbo, Ibibio concepts)
- And many more!

CRITICAL LANGUAGE HANDLING RULES:
1. ALWAYS detect the language from the CURRENT user question ONLY
2. IGNORE the language of previous messages in the conversation
3. Each question can be in a different language
4. Always respond in the SAME language as the CURRENT question

HOW TO HANDLE MULTILINGUAL QUERIES:
1. Read the CURRENT user question
2. Identify what language THIS question is written in
3. Use your tools to search the knowledge base (tools work internally in English)
4. Format your response in the language of THIS CURRENT question

You analyze YouTube cooking videos and help users with:
- Recipe instructions and techniques
- Ingredient information and substitutions
- Cooking times and equipment needed
- Cultural context and culinary insights

You have 17 specialized tools available. USE THEM! When a user asks about:
- Ingredients → Use extract_ingredients_tool or video_qa_tool
- Cooking time → Use cooking_time_tool
- Equipment → Use equipment_checker_tool
- Substitutions → Use smart_substitution_tool
- Nutrition → Use nutrition_calculator_tool
- Recipes → Use video_qa_tool to search videos first
- Web info → Use web_search_cooking_tool

IMPORTANT: Always try to use your tools to get accurate information from videos or web!

CORE PRINCIPLES:
1. USE YOUR TOOLS to answer questions accurately
2. Always cite your sources (video title or "web search")
3. Prefer video content when available (it's your trusted source)
4. Use web search for: nutrition, substitutions, latest trends, pairings
5. Be warm, helpful, and encouraging
6. Give clear, step-by-step instructions
7. Be culturally respectful

SAFETY FOR SUBSTITUTIONS:
- NEVER suggest dangerous substitutions
- Always warn if substitution significantly changes the dish
- Check videos first, then web, then admit if uncertain

You're helping people learn to cook. Be patient, enthusiastic, and always respect the cultural origins of dishes!"""
        
        try:
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt
            )
            print("Agent created successfully")
        except Exception as e:
            print(f"Agent creation issue: {e}")
            self.agent = None
    
    def chat(self, message: str, enhance_with_web: bool = True) -> Dict:
        """
        Process a chat message
        
        Args:
            message: User query
            enhance_with_web: Whether to allow web search (default: True)
        """
        self.conversation_count += 1
        
        try:
            inputs = {
                "messages": self.chat_history + [HumanMessage(content=message)]
            }
            
            response_text = ""
            
            if self.agent:
                try:
                    for chunk in self.agent.stream(inputs, stream_mode="updates"):
                        if "agent" in chunk and "messages" in chunk["agent"]:
                            for msg in chunk["agent"]["messages"]:
                                if hasattr(msg, 'content'):
                                    response_text += msg.content
                    
                    if not response_text:
                        final_state = self.agent.invoke(inputs)
                        if "messages" in final_state:
                            last_message = final_state["messages"][-1]
                            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                except Exception as agent_error:
                    response_text = self._fallback_direct_tool_call(message)
            else:
                response_text = self._fallback_direct_tool_call(message)
            
            # Update conversation history
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=response_text))
            
            return {
                "response": response_text,
                "status": "success",
                "turn": self.conversation_count,
                "session_id": self.session_id,
                "sources": {
                    "videos": True,
                    "web": enhance_with_web
                }
            }
        
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try rephrasing."
            return {
                "response": error_msg,
                "status": "error",
                "turn": self.conversation_count,
                "session_id": self.session_id
            }
    
    def _fallback_direct_tool_call(self, message: str) -> str:
        """Fallback for direct tool usage when agent fails"""
        message_lower = message.lower()
        
        if 'videos' in message_lower or 'list' in message_lower:
            return self.tools[2].func()
        elif 'ingredient' in message_lower:
            return self.tools[6].func("first video")
        else:
            return self.tools[0].func(message)
    
    def index_new_video(self, youtube_url: str, custom_name: Optional[str] = None) -> Dict:
        """
        Index a new YouTube cooking video using yt-dlp + Whisper API
        
        Args:
            youtube_url: YouTube video URL
            custom_name: Optional custom name for the video
        """
        
        try:
            print(f"\n Indexing new video: {youtube_url}")
            
            # Extract video ID
            video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
            if not video_id_match:
                return {"status": "error", "message": "Invalid YouTube URL"}
            
            video_id = video_id_match.group(1)
            print(f"  Video ID: {video_id}")
            
            # Get video title with yt-dlp
            video_title = custom_name
            if not video_title:
                try:
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                        video_title = info.get('title', f'Video {video_id}')
                except:
                    video_title = f"Video {video_id}"
            
            print(f"  Title: {video_title}")
            
            # Use Whisper API for transcription
            print("  Using Whisper API for transcription...")
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download audio with yt-dlp
                    print("  Downloading audio...")
                    audio_path = os.path.join(temp_dir, f'{video_id}.mp3')
                    
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }],
                        'outtmpl': os.path.join(temp_dir, f'{video_id}.%(ext)s'),
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                    
                    print(f"  Audio downloaded")
                    
                    # Transcribe with Whisper
                    print("  Transcribing (this takes 1-2 minutes)...")
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    with open(audio_path, "rb") as audio:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio,
                            response_format="text"
                        )
                    
                    transcript = transcription
                    method = "whisper-api"
                    print(f"  Transcription complete: {len(transcript)} characters")
            
            except Exception as whisper_error:
                print(f"  Transcription failed: {whisper_error}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "error", 
                    "message": f"Transcription failed: {str(whisper_error)}"
                }
            
            if not transcript:
                return {"status": "error", "message": "Could not extract transcript from video"}
            
            # Chunk the transcript
            print("   Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(transcript)
            print(f"  Created {len(chunks)} chunks")
            
            # Add to vector store with proper metadata
            print("  Adding to ChromaDB...")
            for i, chunk in enumerate(chunks):
                self.vectorstore.add_texts(
                    texts=[chunk],
                    metadatas=[{
                        "source": video_title,
                        "video_id": video_id,
                        "chunk_id": i,
                        "url": youtube_url,
                        "indexed_at": str(datetime.now()),
                        "method": method,
                        "session_id": self.session_id
                    }]
                )
            
            # Track indexed video
            self.indexed_videos[video_id] = {
                "title": video_title,
                "video_id": video_id,
                "url": youtube_url,
                "chunks": len(chunks),
                "indexed_at": str(datetime.now()),
                "method": method
            }
            
            print(f"  Successfully indexed '{video_title}'!")
            
            return {
                "status": "success",
                "message": f"Successfully indexed '{video_title}' using Whisper API",
                "video_info": self.indexed_videos[video_id]
            }
            
        except Exception as e:
            print(f"  Error indexing video: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Failed: {str(e)}"}
    
    def get_stats(self) -> Dict:
        """Get statistics about the current session"""
        return {
            "indexed_videos": len(self.indexed_videos),
            "total_chunks": sum(v.get('chunks', 0) for v in self.indexed_videos.values()),
            "memory_turns": len(self.chat_history) // 2  # Pairs of human/AI messages
        }
    
    def get_indexed_videos(self) -> Dict:
        """Get list of all indexed videos"""
        return {
            "videos": list(self.indexed_videos.values())
        }
    
    def list_videos(self) -> Dict:
        """Alias for get_indexed_videos() for compatibility"""
        return self.get_indexed_videos()
    
    def clear_memory(self) -> Dict:
        """Clear conversation memory for this session"""
        self.chat_history.clear()
        self.conversation_count = 0
        return {
            "status": "success",
            "message": "Conversation memory cleared"
        }
    
    def delete_session(self) -> Dict:
        """Delete this session (clears memory and marks for cleanup)"""
        self.chat_history.clear()
        self.conversation_count = 0
        return {
            "status": "success",
            "message": f"Session {self.session_id} cleared",
            "session_id": self.session_id
        }


# Service management with session support

_minddish_sessions: Dict[str, MindDishService] = {}
_session_manager = SessionManager()

def get_minddish_service(
    openai_api_key: str, 
    chroma_path: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    session_id: Optional[str] = None
) -> MindDishService:
    """
    Get or create MindDish service instance for a session
    
    Args:
        openai_api_key: OpenAI API key
        chroma_path: Path to ChromaDB storage
        tavily_api_key: Tavily API key
        session_id: Session ID (creates new if None)
    
    Returns:
        MindDishService instance
    """
    global _minddish_sessions
    
    # Create new session if needed
    if not session_id:
        session_id = _session_manager.create_session()
    
    # Return existing or create new service
    if session_id not in _minddish_sessions:
        _minddish_sessions[session_id] = MindDishService(
            openai_api_key, 
            chroma_path, 
            tavily_api_key,
            session_id
        )
    
    return _minddish_sessions[session_id]

def get_session_manager() -> SessionManager:
    """Get the global session manager"""
    return _session_manager