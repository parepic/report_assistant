import requests

prompt = "Explain RAG in simple terms."

payload = {
    "model": "llama3",
    "prompt": prompt,
}

res = requests.post("http://localhost:11434/api/generate", json=payload)
data = res.json()

print(data)
print(data["response"])