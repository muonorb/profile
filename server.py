from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Load profile context
CONTEXT_FILE = os.path.join(os.path.dirname(__file__), "context.txt")
with open(CONTEXT_FILE, "r") as f:
    CONTEXT = f.read()

SYSTEM_PROMPT = f"""You are the personal AI assistant on Akashdeep Gangatkar's portfolio website.
Your job is to answer visitors' questions about Akashdeep in a helpful, professional, and friendly tone.
Only answer based on the information provided below. If something is not covered, say you don't have that detail and suggest they email Akashdeep directly at akashgangatkaroo@gmail.com.
Keep answers concise and to the point. Do not make up or assume any information.

IMPORTANT: Akashdeep's resume is available for download directly on this website. If anyone asks for his resume, CV, or to download his resume, tell them they can download it using the "Download Resume" button on this page, or directly at: /resume.pdf

AKASHDEEP'S PROFILE:
{CONTEXT}"""

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided"}), 400

    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is not set on the server."}), 500

    question = data["question"].strip()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"answer": answer.strip()})
    except requests.exceptions.Timeout:
        return jsonify({"error": "The model took too long to respond. Try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "backend": "groq"})

if __name__ == "__main__":
    print("\n  Portfolio server running at http://localhost:5001\n")
    app.run(debug=False, host="0.0.0.0", port=5001)
