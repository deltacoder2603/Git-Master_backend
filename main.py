import os
import shutil
import uuid
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Load API keys from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file.")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Session management
class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.base_repo_dir = "repos"
        
        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_repo_dir):
            os.makedirs(self.base_repo_dir)
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        repo_dir = os.path.join(self.base_repo_dir, f"repo_{session_id}")
        
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "repo_dir": repo_dir,
            "repo_analyzed": False
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session info and update last_accessed time"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session["last_accessed"] = datetime.now()
        return session
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions and their directories"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if current_time - session_data["last_accessed"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._remove_session(session_id)
    
    def _remove_session(self, session_id: str):
        """Remove session and cleanup its directory"""
        if session_id in self.sessions:
            repo_dir = self.sessions[session_id]["repo_dir"]
            if os.path.exists(repo_dir):
                try:
                    shutil.rmtree(repo_dir)
                except Exception as e:
                    print(f"Error removing directory {repo_dir}: {e}")
            
            del self.sessions[session_id]
            print(f"Session {session_id} cleaned up")
    
    def force_cleanup_session(self, session_id: str):
        """Manually cleanup a specific session"""
        self._remove_session(session_id)

# Global session manager
session_manager = SessionManager()

# Background task to cleanup expired sessions
async def cleanup_sessions_periodically():
    """Background task to periodically clean up expired sessions"""
    while True:
        try:
            session_manager.cleanup_expired_sessions()
            # Run cleanup every 5 minutes
            await asyncio.sleep(300)
        except Exception as e:
            print(f"Error in session cleanup: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    yield
    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# FastAPI app with lifespan
app = FastAPI(
    title="Repository Analyzer API",
    description="Analyze GitHub repositories with AI-powered Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RepoRequest(BaseModel):
    github_url: str

class QARequest(BaseModel):
    question: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

class AnalyzeResponse(BaseModel):
    status: str
    session_id: str

class QAResponse(BaseModel):
    answer: str
    session_id: str

# Dependency to get session
def get_session_dependency(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session

def clone_repository(github_url: str, repo_dir: str):
    """Clone repository using git clone command with authentication"""
    try:
        # Parse the GitHub URL to insert the token
        if github_url.startswith("https://github.com/"):
            # Replace https://github.com/ with https://token@github.com/
            authenticated_url = github_url.replace("https://github.com/", f"https://{GITHUB_TOKEN}@github.com/")
        else:
            # If it's already an authenticated URL or different format, use as is
            authenticated_url = github_url
        
        # Use subprocess to run git clone
        result = subprocess.run(
            ["git", "clone", authenticated_url, repo_dir],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")
        
        return True
        
    except subprocess.TimeoutExpired:
        raise Exception("Repository cloning timed out (5 minutes)")
    except FileNotFoundError:
        raise Exception("Git command not found. Please ensure Git is installed.")
    except Exception as e:
        raise Exception(f"Error cloning repository: {str(e)}")

# API Endpoints
@app.post("/create-session", response_model=SessionResponse)
def create_session():
    """Create a new session for the user"""
    session_id = session_manager.create_session()
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully"
    )

@app.post("/analyze/{session_id}", response_model=AnalyzeResponse)
def analyze_repo(
    session_id: str,
    req: RepoRequest,
    background_tasks: BackgroundTasks,
    session: dict = Depends(get_session_dependency)
):
    """Download and analyze a GitHub repository for a specific session"""
    repo_dir = session["repo_dir"]
    
    # Remove previous repo if exists
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    
    try:
        # Clone repository using git clone command
        clone_repository(req.github_url, repo_dir)
        session["repo_analyzed"] = True
        
        return AnalyzeResponse(
            status="Repository downloaded and ready for Q&A",
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Repo clone failed: {str(e)}")

def read_all_code_files(directory: str, max_chars: int = 50000) -> str:
    """Read all code files from directory with improved file handling"""
    code_parts = []
    total_chars = 0
    
    # Common code file extensions
    code_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rb', '.php', 
        '.cpp', '.c', '.cs', '.h', '.hpp', '.rs', '.kt', '.swift', '.scala',
        '.html', '.css', '.vue', '.svelte', '.dart', '.r', '.m', '.sh'
    }
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in {
            'node_modules', '.git', '__pycache__', '.venv', 'venv', 
            'build', 'dist', 'target', '.idea', '.vscode'
        }]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in code_extensions:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        
                    # Add file header and content
                    file_content = f"\n\n# File: {os.path.relpath(file_path, directory)}\n{content}"
                    
                    # Check if adding this file would exceed limit
                    if total_chars + len(file_content) > max_chars:
                        remaining_chars = max_chars - total_chars
                        if remaining_chars > 100:  # Only add if we have meaningful space left
                            file_content = file_content[:remaining_chars] + "\n... [truncated]"
                            code_parts.append(file_content)
                        break
                    
                    code_parts.append(file_content)
                    total_chars += len(file_content)
                    
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
    
    return "\n".join(code_parts)

@app.post("/ask/{session_id}", response_model=QAResponse)
def ask_question(
    session_id: str,
    req: QARequest,
    session: dict = Depends(get_session_dependency)
):
    """Ask a question about the analyzed repository"""
    if not session.get("repo_analyzed", False):
        raise HTTPException(
            status_code=400, 
            detail="No repository analyzed for this session. Use /analyze endpoint first."
        )
    
    repo_dir = session["repo_dir"]
    if not os.path.exists(repo_dir):
        raise HTTPException(
            status_code=400, 
            detail="Repository directory not found. Please re-analyze the repository."
        )
    
    try:
        # Read code files
        code_context = read_all_code_files(repo_dir)
        
        if not code_context.strip():
            raise HTTPException(
                status_code=400,
                detail="No code files found in the repository or all files are empty."
            )
        
        # Create prompt for Gemini
        prompt = f"""You are a code analysis expert. Analyze the following codebase and answer the user's question accurately and comprehensively.

CODEBASE:
{code_context}

USER QUESTION: {req.question}

Please provide a detailed and helpful answer based on the codebase provided."""
        
        # Generate response using Gemini
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        
        return QAResponse(
            answer=response.text,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/session-status/{session_id}")
def get_session_status(
    session_id: str,
    session: dict = Depends(get_session_dependency)
):
    """Get the current status of a session"""
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "last_accessed": session["last_accessed"].isoformat(),
        "repo_analyzed": session.get("repo_analyzed", False),
        "repo_dir": session["repo_dir"]
    }

@app.delete("/session/{session_id}")
def cleanup_session(session_id: str):
    """Manually cleanup a specific session"""
    session_manager.force_cleanup_session(session_id)
    return {"message": f"Session {session_id} cleaned up successfully"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Repository Analyzer API",
        "version": "1.0.0",
        "endpoints": [
            "POST /create-session - Create a new session",
            "POST /analyze/{session_id} - Analyze a GitHub repository",
            "POST /ask/{session_id} - Ask questions about the repository",
            "GET /session-status/{session_id} - Get session status",
            "DELETE /session/{session_id} - Cleanup session",
            "GET /health - Health check"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
