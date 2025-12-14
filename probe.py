import time
import openai
import anthropic
from datetime import datetime
import json

# SETUP: Put your keys here (or use environment variables for safety)
OPENAI_KEY = "sk-..."
ANTHROPIC_KEY = "sk-ant-..."

# 1. DEFINE THE TEST
def test_gpt4():
    client = openai.OpenAI(api_key=OPENAI_KEY)
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Write a complete Python script for a functional Snake game. Do not use placeholders."}]
        )
        end_time = time.time()
        content = response.choices[0].message.content
        
        # Grading
        latency = round((end_time - start_time), 2)
        is_lazy = "..." in content or len(content) < 500 # Simple laziness check
        return {"model": "GPT-4", "latency": latency, "lazy": is_lazy, "status": "Online"}
        
    except Exception as e:
        return {"model": "GPT-4", "latency": 0, "lazy": True, "status": "Error"}

def test_claude():
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    start_time = time.time()
    
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Write a complete Python script for a functional Snake game. Do not use placeholders."}]
        )
        end_time = time.time()
        content = message.content[0].text
        
        # Grading
        latency = round((end_time - start_time), 2)
        is_lazy = "..." in content or len(content) < 500
        return {"model": "Claude 3", "latency": latency, "lazy": is_lazy, "status": "Online"}
        
    except Exception as e:
        return {"model": "Claude 3", "latency": 0, "lazy": True, "status": "Error"}

# 2. RUN & SAVE
results = [test_gpt4(), test_claude()]
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data = {
    "last_updated": timestamp,
    "data": results
}

# Save to a file that the website can read
with open("status.json", "w") as f:
    json.dump(data, f)

print(f"Scan complete at {timestamp}")
