"""
MindDish.ai API Routes

"""

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional
import os
import uuid

# Import your service
from app.services.minddish_service import get_minddish_service

router = APIRouter()

@router.post("/test-minimal")
def test_minimal():
    """Minimal test route - no service, no logic"""
    return {"status": "success", "message": "Minimal route works!"}

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    enhance_with_web: bool = True

class IndexVideoRequest(BaseModel):
    youtube_url: str
    custom_name: Optional[str] = None

# Routes

@router.post("/chat")
def chat(request: ChatRequest, req: Request, response: Response):
    """Chat with MindDish.ai with session management"""
    try:
        # Get or create session from cookie
        session_id = req.cookies.get('minddish_session')
        
        if not session_id:
            session_id = str(uuid.uuid4())
            response.set_cookie(
                key='minddish_session',
                value=session_id,
                max_age=2592000,  # 30 days
                httponly=True,
                samesite='lax'
            )
        
        # Get service instance for this user's session
        service = get_minddish_service(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            session_id=session_id
        )
        
        # Process the chat message
        result = service.chat(request.message, request.enhance_with_web)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-video")
def index_video(request: IndexVideoRequest, req: Request, response: Response):
    """Index a new YouTube cooking video"""
    try:
        session_id = req.cookies.get('minddish_session')
        
        if not session_id:
            session_id = str(uuid.uuid4())
            response.set_cookie(
                key='minddish_session',
                value=session_id,
                max_age=2592000,
                httponly=True,
                samesite='lax'
            )
        
        service = get_minddish_service(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            session_id=session_id
        )
        
        result = service.index_new_video(
            youtube_url=request.youtube_url,
            custom_name=request.custom_name
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def get_stats(req: Request):
    """Get system statistics"""
    try:
        session_id = req.cookies.get('minddish_session', 'default')
        
        service = get_minddish_service(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            session_id=session_id
        )
        
        return service.get_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos")
def list_videos(req: Request):
    """List indexed videos"""
    try:
        session_id = req.cookies.get('minddish_session', 'default')
        
        service = get_minddish_service(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            session_id=session_id
        )
        
        result = service.list_videos()  # Returns {"videos": [...]}
        return result  # Return it directly
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-memory")
def clear_memory(req: Request):
    """Clear conversation memory"""
    try:
        session_id = req.cookies.get('minddish_session', 'default')
        
        service = get_minddish_service(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            session_id=session_id
        )
        
        service.clear_memory()
        return {"status": "success", "message": "Memory cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))