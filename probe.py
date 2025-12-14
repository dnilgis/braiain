import time
import os
import json
import requests
from datetime import datetime

# --- IMPORTS ---
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
import cohere

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
HF_KEY = os.environ.get("HF_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
COHERE_KEY = os.environ.get("COHERE_API_KEY")

# 30 Days * 6 scans per day = 180 points
MAX_HISTORY = 180 

PROMPT = "Write a Python function to calculate the Fibonacci sequence. Write the full code."

def get_current_history():
    if os.path.exists("status.json"):
        with open("status.json", "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"history": {}}
    return {"history": {}}

def test_model(name, func):
    try:
        start = time.time()
        result = func()
        latency = round(time.time() - start, 2)
        
        if not result: raise Exception("Empty response")
        
        # Laziness Check: Did they write the code?
        is_lazy = len(result) < 50 or "def" not in result
        
        return {
            "latency": latency,
            "lazy": is_lazy,
            "status": "Online",
            "timestamp": datetime.now().strftime("%b-%d %H:%M") 
        }
    except Exception as e:
        status = "No Key" if "Key" in str(e) or "api_key" in str(e) else "Error"
        return {"latency": 0, "lazy": True, "status": status, "timestamp": datetime.now().strftime("%b-%d %H:%M")}

# --- WRAPPER FUNCTIONS ---
def run_gemini():
    if not GEMINI_KEY: raise Exception("No Key")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model.generate_content(PROMPT).text

def run_openai():
    if not OPENAI_KEY: raise Exception("No Key")
    client = OpenAI(api_key=OPENAI_KEY)
    return client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":PROMPT}]).choices[0].message.content

def run_anthropic():
    if not ANTHROPIC_KEY: raise Exception("No Key")
    client = Anthropic(api_key=ANTHROPIC_KEY)
    return client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=1000, messages=[{"role":"user","content":PROMPT}]).content[0].text

def run_groq():
    if not GROQ_KEY: raise Exception("No Key")
    client = Groq(api_key=GROQ_KEY)
    return client.chat.completions.create(model="llama3-70b-8192", messages=[{"role":"user","content":PROMPT}]).choices[0].message.content

def run_cohere():
    if not COHERE_KEY: raise Exception("No Key")
    co = cohere.Client(COHERE_KEY)
    return co.chat(message=PROMPT, model="command-r-plus").text

def run_hf_mistral():
    if not HF_KEY: raise Exception("No Key")
    headers = {"Authorization": f"Bearer {HF_KEY}"}
    resp = requests.post("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3", headers=headers, json={"inputs": PROMPT})
    return resp.json()[0]['generated_text']

def run_hf_phi3():
    if not HF_KEY: raise Exception("No Key")
    headers = {"Authorization": f"Bearer {HF_KEY}"}
    resp = requests.post("https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct", headers=headers, json={"inputs": PROMPT})
    return resp.json()[0]['generated_text']

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading history...")
    db = get_current_history()
    
    # Ensure structure exists
    if "history" not in db: db["history"] = {}

    models = {
        "Gemini 1.5 Flash": run_gemini,
        "GPT-4o": run_openai,
        "Claude 3.5 Sonnet": run_anthropic,
        "Llama 3 70B (Groq)": run_groq,
        "Command R+ (Cohere)": run_cohere,
        "Mistral 7B (HF)": run_hf_mistral,
        "Phi-3 Mini (HF)": run_hf_phi3
    }

    for name, func in models.items():
        print(f"Testing {name}...")
        result = test_model(name, func)
        
        if name not in db["history"]: db["history"][name] = []
            
        db["history"][name].append(result)
        
        # TRUNCATE to Last 30 Days (180 points)
        if len(db["history"][name]) > MAX_HISTORY:
            db["history"][name] = db["history"][name][-MAX_HISTORY:]

    db["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    with open("status.json", "w") as f:
        json.dump(db, f, indent=4)
        
    print("Scan Complete.")
