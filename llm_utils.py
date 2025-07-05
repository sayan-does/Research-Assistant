import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"

# Generic, industry-standard system prompt for RAG
SYSTEM_PROMPT = (
    "You are an advanced AI assistant designed to help users understand and analyze information. "
    "You have access to a vector database containing relevant passages and excerpts from documents provided by the user. "
    "For every user query, you will be given a set of context passages retrieved from this vector database. "
    "Your job is to answer the user's question as accurately and concisely as possible, using only the information present in the provided context.\n"
    "\n"
    "Guidelines:\n"
    "- Always ground your answers in the context provided from the vector database.\n"
    "- If the context does not contain enough information to answer the question, respond with: 'I could not find relevant information in the provided documents.'\n"
    "- Do not make up information, speculate, or use knowledge not present in the context.\n"
    "- If the user asks for a summary, provide a concise summary based only on the context.\n"
    "- If the user asks for key concepts, extract and list them from the context.\n"
    "- If the user asks for citations, refer to the context passages.\n"
    "- If the context contains multiple perspectives or findings, mention them clearly.\n"
    "- Maintain a professional, neutral, and helpful tone.\n"
    "\n"
    "You are not allowed to use any external knowledge or data except what is provided in the context from the vector database."
    "IMPORTANT:\n"
    "- Give the answers to the point and concise.\n" \
    "- Ask the user if they satisfied with the answer or need more information. ask -'Would you like to know more about some specific topics?"
)

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            return []
    def chat_stream(self, model: str, messages, temperature: float = 0.7):
        import json
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
            "max_tokens": 1024
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True
            )
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get('message', {}).get('content'):
                                yield chunk['message']['content']
                            if chunk.get('done', False):
                                break
                        except Exception as e:
                            yield f"[Stream error: {str(e)}]"
            else:
                yield f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            yield f"Connection error: {str(e)}"
    def is_ollama_running(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

def build_rag_prompt(user_query, context=None):
    if context:
        return f"{SYSTEM_PROMPT}\n\nContext from papers:\n{context}\n\nUser Query: {user_query}"
    else:
        return f"{SYSTEM_PROMPT}\n\nUser Query: {user_query}"
