from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Akashdeep Portfolio API", version="1.0.0")

# CORS — allows the browser to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2:3b"

CONTEXT_FILE = os.path.join(os.path.dirname(__file__), "context.txt")
with open(CONTEXT_FILE, "r") as f:
    CONTEXT = f.read()

SYSTEM_PROMPT = f"""You are the personal AI assistant on Akashdeep Gangatkar's portfolio website.
Your job is to answer visitors' questions about Akashdeep in a helpful, professional, and friendly tone.
Only answer based on the information provided below. If something is not covered, say you don't have that detail and suggest they email Akashdeep directly at akashgangatkaroo@gmail.com.
Keep answers concise and to the point. Do not make up or assume any information.

AKASHDEEP'S PROFILE:
{CONTEXT}"""

# ── Request / Response models (this is what FastAPI does that Flask doesn't) ─
# Pydantic automatically validates that the request body has a "question" field
# and that it's a string. If it's missing or wrong type, FastAPI returns a 422
# error automatically — no manual validation needed.

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str
    ollama: str

# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Receive a question, send it to Ollama, return the answer."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nVisitor's question: {body.question.strip()}\n\nAnswer:",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        answer = resp.json().get("response", "Sorry, I could not generate a response.")
        return ChatResponse(answer=answer.strip())

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Please start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The model took too long to respond. Try again."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if Ollama is reachable."""
    try:
        requests.get("http://localhost:11434/", timeout=3)
        return HealthResponse(status="ok", ollama="running")
    except Exception:
        return HealthResponse(status="ok", ollama="not reachable")


# ── Serve static files (HTML, CSS, resume.pdf) ───────────────────────────────
# Mount static files AFTER API routes so /chat and /health take priority
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.get("/{filename}")
async def serve_file(filename: str):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n  FastAPI portfolio server running at http://localhost:8000")
    print("  Interactive API docs at  http://localhost:8000/docs\n")
    uvicorn.run("FastAPI_server:app", host="0.0.0.0", port=8000, reload=True)
