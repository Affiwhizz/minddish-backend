from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router 
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MindDish.ai API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your routes
app.include_router(router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "MindDish.ai API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "tools": 17}