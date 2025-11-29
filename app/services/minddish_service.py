"""
MindDish.ai Service - Production Backend
Multilingual RAG-based cooking assistant with 17 specialized tools
Uses create_agent from LangChain 1.1.0+
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent

# Service instances cache (session-based)
_service_instances = {}


class MindDishService:
    """Complete MindDish.ai RAG service with 17 tools"""
    
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
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Load existing ChromaDB
        self.vectorstore = self._load_vectorstore()
        
        # Build indexed videos dictionary
        self.indexed_videos = self._build_indexed_videos()
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
        
        # Initialize chat history
        self.chat_history = []
        self.awaiting_permission = False
        self.pending_question = None
        
        # Create all 17 tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _load_vectorstore(self):
        """Load pre-built ChromaDB vectorstore"""
        # Use relative path that works on Render
        chroma_path = Path("./chroma_db")
        
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {chroma_path}. "
                "Run build_chroma.py first or ensure chroma_db folder exists."
            )
        
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=self.embeddings
        )
        
        return vectorstore
    
    def _build_indexed_videos(self) -> Dict:
        """Build indexed videos dictionary from vectorstore metadata"""
        indexed_videos = {}
        
        try:
            collection = self.vectorstore._collection
            all_data = collection.get()
            
            for metadata in all_data['metadatas']:
                if metadata and 'video_id' in metadata:
                    video_id = metadata['video_id']
                    if video_id not in indexed_videos:
                        indexed_videos[video_id] = {
                            'title': metadata.get('title', 'Unknown'),
                            'collection': metadata.get('collection', 'unknown'),
                            'url': metadata.get('url', f'https://youtu.be/{video_id}'),
                            'chunks': 0,
                            'method': metadata.get('method', 'youtube_transcript_api')
                        }
                    indexed_videos[video_id]['chunks'] += 1
        except Exception as e:
            print(f"Warning: Could not build indexed videos: {e}")
        
        return indexed_videos
    
    def _create_rag_chain(self):
        """Create RAG chain for video QA"""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = f"[{doc.metadata.get('collection', 'unknown')}] {doc.metadata.get('title', 'Unknown')}"
                formatted.append(f"Source: {source}\nContent: {doc.page_content}")
            return "\n\n".join(formatted)
        
        qa_template = """You are MindDish.ai, a cooking assistant that answers questions based ONLY on the provided video transcripts.

Context from cooking videos:
{context}

Question: {question}

Instructions:
- Answer ONLY based on the context provided
- If the information is not in the context, say "I don't have this information in the indexed videos"
- Always cite which video the information comes from
- Be helpful and provide step-by-step instructions when relevant

Answer:"""
        
        qa_prompt = ChatPromptTemplate.from_template(qa_template)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _web_search(self, query: str, max_results: int = 3) -> dict:
        """Search the web using Tavily API"""
        try:
            if not self.tavily_api_key:
                return {"status": "error", "results": [], "message": "TAVILY_API_KEY not configured"}
            
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_api_key)
            
            response = client.search(query=query, max_results=max_results)
            
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
            return {"status": "error", "results": [], "message": "tavily-python not installed"}
        except Exception as e:
            return {"status": "error", "results": [], "message": f"Web search failed: {str(e)[:200]}"}
    
    def _create_tools(self) -> List:
        """Create all 17 MindDish tools"""
        
        # Store references for use in tools
        vectorstore = self.vectorstore
        llm = self.llm
        indexed_videos = self.indexed_videos
        rag_chain = self.rag_chain
        web_search = self._web_search
        
        # TOOL 1: Video QA
        @tool
        def video_qa_tool(question: str) -> str:
            """Answer cooking questions using RAG across all indexed videos."""
            return rag_chain.invoke(question)
        
        # TOOL 2: Transcript Search
        @tool
        def transcript_search_tool(keyword: str) -> str:
            """Search for specific keyword mentions across all cooking videos."""
            results = vectorstore.similarity_search(keyword, k=5)
            
            video_counts = {}
            for doc in results:
                video_title = doc.metadata.get('title', 'Unknown')
                video_counts[video_title] = video_counts.get(video_title, 0) + 1
            
            if not video_counts:
                return f"'{keyword}' not found in any videos"
            
            response = f"Found '{keyword}' in:\n"
            for video, count in sorted(video_counts.items(), key=lambda x: x[1], reverse=True):
                response += f"  - {video}: {count} mention(s)\n"
            
            return response
        
        # TOOL 3: List Videos
        @tool
        def list_videos_tool(query: str = "") -> str:
            """List all indexed cooking videos with details."""
            response = f"Indexed Cooking Videos ({len(indexed_videos)}):\n\n"
            for vid_id, info in indexed_videos.items():
                response += f"- {info['title']}\n"
                response += f"  ID: {vid_id} | Collection: {info['collection']}\n\n"
            return response
        
        # TOOL 4: Video Summary
        @tool
        def video_summary_tool(video_title_or_id: str) -> str:
            """Generate a comprehensive summary of a specific cooking video."""
            video_id = None
            for vid, info in indexed_videos.items():
                if video_title_or_id.lower() in info['title'].lower() or video_title_or_id == vid:
                    video_id = vid
                    break
            
            if not video_id:
                return f"Video '{video_title_or_id}' not found. Use list_videos_tool to see available videos."
            
            results = vectorstore.similarity_search("recipe ingredients steps instructions", k=10)
            
            if not results:
                return f"No content found for video {video_id}"
            
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
        
        # TOOL 5: Compare Videos
        @tool
        def compare_videos_tool(topic: str) -> str:
            """Compare how different cooking videos discuss a specific topic."""
            results = vectorstore.similarity_search(topic, k=8)
            
            video_content = {}
            for doc in results:
                video_title = doc.metadata.get('title', 'Unknown')
                if video_title not in video_content:
                    video_content[video_title] = []
                video_content[video_title].append(doc.page_content)
            
            if not video_content:
                return f"No videos discuss '{topic}'"
            
            comparison = f"Comparison: '{topic}'\n\n"
            for video, chunks in video_content.items():
                excerpt = ' '.join(chunks[:2])[:200]
                comparison += f"{video}:\n   {excerpt}...\n\n"
            
            return comparison
        
        # TOOL 6: Find Related Videos
        @tool
        def find_related_videos_tool(topic: str) -> str:
            """Find which cooking videos are most relevant to a topic."""
            results = vectorstore.similarity_search(topic, k=10)
            
            video_scores = {}
            for doc in results:
                video_title = doc.metadata.get('title', 'Unknown')
                video_scores[video_title] = video_scores.get(video_title, 0) + 1
            
            sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
            
            response = f"Most relevant videos for '{topic}':\n\n"
            for i, (video, score) in enumerate(sorted_videos[:3], 1):
                response += f"{i}. {video} ({score} relevant segments)\n"
            
            return response
        
        # TOOL 7: Extract Ingredients
        @tool
        def extract_ingredients_tool(video_title_or_id: str) -> str:
            """Extract all ingredients mentioned in a specific cooking video."""
            video_id = None
            for vid, info in indexed_videos.items():
                if video_title_or_id.lower() in info['title'].lower() or video_title_or_id == vid:
                    video_id = vid
                    break
            
            if not video_id:
                return f"Video '{video_title_or_id}' not found"
            
            results = vectorstore.similarity_search("ingredients what you need shopping list", k=8)
            
            if not results:
                return "No ingredients found"
            
            all_text = " ".join([doc.page_content for doc in results])
            prompt = f"""Extract all ingredients mentioned in this cooking video:

{all_text[:2000]}

Ingredients:"""
            
            response = llm.invoke(prompt)
            return response.content
        
        # TOOL 8: Cooking Time
        @tool
        def cooking_time_tool(video_title_or_id: str) -> str:
            """Extract cooking times, prep times, and total time from a video."""
            video_id = None
            for vid, info in indexed_videos.items():
                if video_title_or_id.lower() in info['title'].lower() or video_title_or_id == vid:
                    video_id = vid
                    break
            
            if not video_id:
                return f"Video '{video_title_or_id}' not found"
            
            results = vectorstore.similarity_search("minutes hours time cook bake prep", k=6)
            
            if not results:
                return "No timing information found"
            
            all_text = " ".join([doc.page_content for doc in results])
            prompt = f"""Extract all timing information from this cooking video:
- Prep time
- Cook time
- Total time

Content: {all_text[:1500]}

Time breakdown:"""
            
            response = llm.invoke(prompt)
            return response.content
        
        # TOOL 9: Equipment Checker
        @tool
        def equipment_checker_tool(video_title_or_id: str) -> str:
            """List all cooking equipment and tools needed for a recipe."""
            video_id = None
            for vid, info in indexed_videos.items():
                if video_title_or_id.lower() in info['title'].lower() or video_title_or_id == vid:
                    video_id = vid
                    break
            
            if not video_id:
                return f"Video '{video_title_or_id}' not found"
            
            results = vectorstore.similarity_search("pan pot skillet bowl oven stove equipment tools", k=6)
            
            if not results:
                return "No equipment information found"
            
            all_text = " ".join([doc.page_content for doc in results])
            prompt = f"""List all cooking equipment and tools mentioned in this video:

Content: {all_text[:1500]}

Equipment needed:"""
            
            response = llm.invoke(prompt)
            return response.content
        
        # TOOL 10: Smart Substitution (with 3-layer safety)
        @tool
        def smart_substitution_tool(original_ingredient: str, substitute: str, dish: str = "") -> str:
            """Suggest safe ingredient substitutions with 3-layer safety system."""
            # Layer 1: Dangerous substitutions blacklist
            dangerous_pairs = [
                ('baking soda', 'baking powder'),
                ('baking powder', 'baking soda'),
                ('salt', 'sugar'),
                ('sugar', 'salt'),
            ]
            
            ingredient_lower = original_ingredient.lower()
            substitute_lower = substitute.lower()
            
            for pair in dangerous_pairs:
                if any(item in ingredient_lower for item in pair) and any(item in substitute_lower for item in pair):
                    return f"DANGER: Substituting {original_ingredient} with {substitute} is UNSAFE!"
            
            # Layer 2: Check indexed videos
            search_query = f"{original_ingredient} substitute {substitute} alternative"
            results = vectorstore.similarity_search(search_query, k=5)
            
            video_context = ""
            for doc in results:
                content = doc.page_content.lower()
                if ingredient_lower in content:
                    video_context += doc.page_content + "\n"
            
            if video_context:
                prompt = f"""Can I use {substitute} instead of {original_ingredient} in {dish or 'this recipe'}?

Video context: {video_context[:800]}

Provide: YES/NO/DEPENDS, ratio, and how it affects the dish."""
                
                response = llm.invoke(prompt)
                return f"From indexed videos:\n{response.content}"
            
            # Layer 3: Web search
            web_result = web_search(f"substitute {original_ingredient} with {substitute} cooking", max_results=2)
            
            if web_result['status'] == 'success' and web_result['results']:
                web_context = "\n".join([r['content'][:200] for r in web_result['results']])
                prompt = f"""Can I substitute {original_ingredient} with {substitute}?

Web research: {web_context}

Provide clear YES/NO, ratio, and safety notes."""
                
                response = llm.invoke(prompt)
                return f"From web search:\n{response.content}"
            
            return f"No reliable information found for substituting {original_ingredient} with {substitute}."
        
        # TOOL 11: Recipe Fact Check
        @tool
        def recipe_fact_check_tool(claim: str) -> str:
            """Verify cooking claims against indexed videos and web sources."""
            results = vectorstore.similarity_search(claim, k=4)
            
            if results:
                video_evidence = " ".join([doc.page_content for doc in results])
                prompt = f"""Fact-check this cooking claim:
Claim: {claim}

Video evidence: {video_evidence[:1000]}

Is this TRUE, FALSE, or PARTIALLY TRUE? Explain."""
                
                response = llm.invoke(prompt)
                return response.content
            
            return f"No evidence found in indexed videos for: {claim}"
        
        # TOOL 12: Suggest Questions
        @tool
        def suggest_questions_tool(topic: str) -> str:
            """Suggest follow-up questions based on a cooking topic."""
            results = vectorstore.similarity_search(topic, k=3)
            
            context = " ".join([doc.page_content for doc in results]) if results else ""
            
            prompt = f"""Based on this cooking topic: {topic}

Context: {context[:500]}

Suggest 5 follow-up questions the user might want to ask:"""
            
            response = llm.invoke(prompt)
            return response.content
        
        # TOOL 13: Web Search Cooking
        @tool
        def web_search_cooking_tool(query: str) -> str:
            """Search the web for cooking information not in indexed videos."""
            result = web_search(query + " cooking recipe", max_results=3)
            
            if result['status'] == 'success' and result['results']:
                response = f"Web search results for '{query}':\n\n"
                for i, r in enumerate(result['results'], 1):
                    response += f"{i}. {r['title']}\n   {r['content'][:150]}...\n   URL: {r['url']}\n\n"
                return response
            
            return f"No web results found for: {query}"
        
        # TOOL 14: Cultural Context
        @tool
        def cultural_context_tool(term: str) -> str:
            """Explain cultural context of cooking terms, dishes, or ingredients."""
            results = vectorstore.similarity_search(term, k=4)
            
            video_context = " ".join([doc.page_content for doc in results]) if results else ""
            
            if video_context:
                prompt = f"""Explain the cultural context of '{term}' based on this cooking video:

Video Context: {video_context[:500]}

Cultural Context:"""
                
                response = llm.invoke(prompt)
                video_answer = f"From videos:\n{response.content}\n\n"
            else:
                video_answer = f"'{term}' not found in indexed videos.\n\n"
            
            # Enhance with web search
            web_result = web_search(f"{term} traditional cooking cultural significance", max_results=2)
            
            if web_result['status'] == 'success' and web_result['results']:
                web_context = "\n".join([f"- {r['content'][:150]}" for r in web_result['results']])
                return video_answer + f"Additional context from web:\n{web_context}"
            
            return video_answer
        
        # TOOL 15: Cooking Expert Analysis
        @tool
        def cooking_expert_analysis_tool(question: str) -> str:
            """Get detailed culinary analysis combining video content + web knowledge."""
            results = vectorstore.similarity_search(question, k=4)
            video_context = " ".join([doc.page_content for doc in results]) if results else "No relevant video content."
            
            web_result = web_search(question + " cooking science technique", max_results=2)
            web_context = "\n".join([r['content'][:200] for r in web_result.get('results', [])[:2]]) if web_result['status'] == 'success' else "Web unavailable."
            
            prompt = f"""As a culinary expert, analyze this question:

Question: {question}

Video Content: {video_context[:800]}

Web Research: {web_context[:400]}

Provide expert analysis covering technique, science, common mistakes, and pro tips."""
            
            response = llm.invoke(prompt)
            return response.content
        
        # TOOL 16: Translate Recipe
        @tool
        def translate_recipe_tool(text: str, target_language: str = "Portuguese") -> str:
            """Translate cooking instructions or recipes to another language."""
            prompt = f"""Translate this cooking content to {target_language}. Maintain culinary terminology accuracy.

Text to translate:
{text[:1500]}

Translation in {target_language}:"""
            
            response = llm.invoke(prompt)
            return response.content
        
        # TOOL 17: Nutrition Calculator
        @tool
        def nutrition_calculator_tool(video_title_or_recipe: str, servings: int = 1) -> str:
            """Calculate approximate nutritional information for a recipe."""
            try:
                video_id = None
                for vid, info in indexed_videos.items():
                    if video_title_or_recipe.lower() in info['title'].lower():
                        video_id = vid
                        break
                
                if not video_id:
                    return f"Video '{video_title_or_recipe}' not found."
                
                # Get ingredients
                ingredients_text = extract_ingredients_tool.func(video_title_or_recipe)
                
                web_result = web_search(f"nutrition facts calories {video_title_or_recipe}", max_results=2)
                web_context = "\n".join([r['content'][:200] for r in web_result.get('results', [])[:2]]) if web_result['status'] == 'success' else "Web unavailable"
                
                prompt = f"""Calculate approximate nutritional information:

Recipe: {video_title_or_recipe}
Servings: {servings}

Ingredients: {ingredients_text[:1000]}

Web data: {web_context[:500]}

Provide: calories, protein, carbs, fat per serving. Note these are estimates."""
                
                response = llm.invoke(prompt)
                return f"Nutritional Information: {video_title_or_recipe}\nServings: {servings}\n\n{response.content}"
            
            except Exception as e:
                return f"Could not calculate nutrition: {str(e)[:200]}"
        
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
    
    def _create_agent(self):
        """Create the agent using create_agent"""
        system_prompt = """You are MindDish.ai, an expert cooking assistant with access to 28 curated cooking videos across 7 global cuisines (African, French, Portuguese, Jamaican, Syrian, Italian, Indian).

You have 17 specialized tools at your disposal. Use the appropriate tool for each task:
- For recipe questions: use video_qa_tool
- To find specific mentions: use transcript_search_tool
- To list available videos: use list_videos_tool
- For video summaries: use video_summary_tool
- To compare approaches: use compare_videos_tool
- For ingredient lists: use extract_ingredients_tool
- For cooking times: use cooking_time_tool
- For equipment needed: use equipment_checker_tool
- For substitutions: use smart_substitution_tool (has safety checks)
- To verify claims: use recipe_fact_check_tool
- For web searches: use web_search_cooking_tool
- For cultural context: use cultural_context_tool
- For expert analysis: use cooking_expert_analysis_tool
- To translate: use translate_recipe_tool
- For nutrition info: use nutrition_calculator_tool

Guidelines:
- Always cite which video your information comes from
- If information is not in indexed videos, say so and offer to search the web
- Be helpful, accurate, and provide step-by-step guidance when needed"""
        
        agent_graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        
        return agent_graph
    
    def chat(self, message: str, enhance_with_web: bool = True) -> Dict:
        """Main chat method using the agent"""
        try:
            # Invoke agent with message format
            response = self.agent.invoke({"messages": [("user", message)]})
            
            # Extract output from response
            output = response["messages"][-1].content if response.get("messages") else "No response"
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=output))
            
            return {
                "response": output,
                "status": "success",
                "source_type": "agent"
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "status": "error"
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "total_chunks": count,
            "total_videos": len(self.indexed_videos),
            "tools_available": len(self.tools),
            "messages_in_memory": len(self.chat_history),
            "web_search_enabled": self.tavily_api_key is not None
        }
    
    def list_videos(self) -> Dict:
        """List all indexed videos"""
        videos = []
        for video_id, info in self.indexed_videos.items():
            videos.append({
                'title': info['title'],
                'url': info['url'],
                'cuisine': info['collection'],
                'video_id': video_id
            })
        
        return {"videos": videos}
    
    def clear_memory(self):
        """Clear conversation history"""
        self.chat_history = []
        self.awaiting_permission = False
        self.pending_question = None
    
    def index_new_video(self, youtube_url: str, custom_name: Optional[str] = None) -> Dict:
        """Index a new video (placeholder)"""
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