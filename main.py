import os
import uuid
import asyncio
import requests
import base64
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or "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"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file.")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env file.")

genai.configure(api_key=GEMINI_API_KEY)

class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "repo_analyzed": False,
            "repo_content": {}
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        session["last_accessed"] = datetime.now()
        return session

    def cleanup_expired_sessions(self):
        current_time = datetime.now()
        expired_sessions = [sid for sid, data in self.sessions.items()
                             if current_time - data["last_accessed"] > self.session_timeout]
        for sid in expired_sessions:
            del self.sessions[sid]
            print(f"Session {sid} cleaned up")

    def force_cleanup_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()

async def cleanup_sessions_periodically():
    while True:
        try:
            session_manager.cleanup_expired_sessions()
            await asyncio.sleep(300)
        except Exception as e:
            print(f"Error in session cleanup: {e}")
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Repository Analyzer API",
              description="Analyze GitHub repositories with AI-powered Q&A",
              version="1.0.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def get_session_dependency(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session

def parse_github_url(github_url: str) -> tuple:
    if github_url.endswith('.git'):
        github_url = github_url[:-4]
    if 'github.com/' in github_url:
        parts = github_url.split('github.com/')[-1].split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
    raise ValueError("Invalid GitHub URL format")

def get_default_branch(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("default_branch", "main")
    print(f"[ERROR] Failed to get default branch: {response.status_code} - {response.text}")
    return "main"

def get_repo_contents_via_api(owner: str, repo: str, path: str = "") -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    print(f"[DEBUG] Fetching: {url} - Status: {response.status_code}")
    if response.status_code != 200:
        print(f"[DEBUG] GitHub API Error: {response.text}")
        raise Exception(f"GitHub API error: {response.status_code} - {response.text}")
    return response.json()

def get_file_content(owner: str, repo: str, path: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return ""
    file_data = response.json()
    if file_data.get('encoding') == 'base64':
        return base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')
    return ""

def fetch_all_files(owner: str, repo: str, max_chars: int = 50000) -> str:
    file_parts = []
    total_chars = 0
    
    # Binary file extensions to skip
    binary_exts = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp',
        '.pdf', '.exe', '.dll', '.so', '.dylib', '.zip', '.tar', '.gz', '.7z', '.rar',
        '.mp3', '.wav', '.ogg', '.flac', '.mp4', '.mkv', '.avi', '.mov'
    }
    
    skip_dirs = {
        'node_modules', '.git', '__pycache__', '.venv', 'venv',
        'build', 'dist', 'target', '.idea', '.vscode', '.next', 'out'
    }

    def process_directory(path: str = ""):
        nonlocal total_chars
        try:
            contents = get_repo_contents_via_api(owner, repo, path)
            for item in contents:
                if total_chars >= max_chars:
                    break
                if item['type'] == 'dir':
                    if item['name'] not in skip_dirs:
                        process_directory(item['path'])
                elif item['type'] == 'file':
                    file_ext = os.path.splitext(item['name'])[1].lower()
                    if file_ext in binary_exts:
                        continue  # Skip binary files
                    
                    content = get_file_content(owner, repo, item['path'])
                    if content:
                        file_content = f"\n\n# File: {item['path']}\n{content}"
                        if total_chars + len(file_content) > max_chars:
                            remaining = max_chars - total_chars
                            if remaining > 100:
                                file_content = file_content[:remaining] + "\n... [truncated]"
                                file_parts.append(file_content)
                            break
                        file_parts.append(file_content)
                        total_chars += len(file_content)
        except Exception as e:
            print(f"[ERROR] Directory {path}: {e}")

    default_branch = get_default_branch(owner, repo)
    print(f"[INFO] Default branch: {default_branch}")
    process_directory()
    return "\n".join(file_parts)

@app.post("/create-session", response_model=SessionResponse)
def create_session():
    session_id = session_manager.create_session()
    return SessionResponse(session_id=session_id, message="Session created successfully")

@app.post("/analyze/{session_id}", response_model=AnalyzeResponse)
def analyze_repo(session_id: str, req: RepoRequest, background_tasks: BackgroundTasks, session: dict = Depends(get_session_dependency)):
    try:
        owner, repo = parse_github_url(req.github_url)
        repo_content = fetch_all_files(owner, repo)
        if not repo_content.strip():
            raise HTTPException(status_code=400, detail="No code files found in the repository or repository is empty.")
        session["repo_content"] = repo_content
        session["repo_analyzed"] = True
        session["owner"] = owner
        session["repo_name"] = repo
        return AnalyzeResponse(status="Repository analyzed and ready for Q&A", session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid GitHub URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Repository analysis failed: {str(e)}")

@app.post("/ask/{session_id}", response_model=QAResponse)
def ask_question(session_id: str, req: QARequest, session: dict = Depends(get_session_dependency)):
    if not session.get("repo_analyzed", False):
        raise HTTPException(status_code=400, detail="No repository analyzed for this session. Use /analyze endpoint first.")
    repo_content = session.get("repo_content", "")
    if not repo_content.strip():
        raise HTTPException(status_code=400, detail="No repository content found. Please re-analyze the repository.")
    try:
        prompt = f"""You are a code analysis expert. Analyze the following codebase and answer the user's question accurately and comprehensively.

REPOSITORY: {session.get('owner', 'Unknown')}/{session.get('repo_name', 'Unknown')}

CODEBASE:
{repo_content}

USER QUESTION: {req.question}

Please provide a detailed and helpful answer based on the codebase provided."""
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return QAResponse(answer=response.text, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/session-status/{session_id}")
def get_session_status(session_id: str, session: dict = Depends(get_session_dependency)):
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "last_accessed": session["last_accessed"].isoformat(),
        "repo_analyzed": session.get("repo_analyzed", False),
        "owner": session.get("owner"),
        "repo_name": session.get("repo_name")
    }

@app.delete("/session/{session_id}")
def cleanup_session(session_id: str):
    session_manager.force_cleanup_session(session_id)
    return {"message": f"Session {session_id} cleaned up successfully"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def root():
    return {
        "message": "Repository Analyzer API",
        "version": "1.0.0",
        "endpoints": [
            "POST /create-session",
            "POST /analyze/{session_id}",
            "POST /ask/{session_id}",
            "GET /session-status/{session_id}",
            "DELETE /session/{session_id}",
            "GET /health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
