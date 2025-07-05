import time
import json
from rag_utils import EmbeddingModel, VectorDB, process_pdf_to_faiss
from llm_utils import OllamaClient, build_rag_prompt

# Models to evaluate
MODELS = [
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    "gemma3:1b",
    "tinyllama:1.1b"
]

# Dummy questions for evaluation
DUMMY_QUESTIONS = [
    "What is the main contribution of this paper?",
    "Summarize the key findings.",
    "List the key concepts discussed.",
    "What are the limitations or future work mentioned?"
]

# Evaluation metrics
RESULTS = {}

def evaluate_model_on_pdf(model_name, pdf_path, embed_model, vectordb):
    # Prepare context
    chunks = process_pdf_to_faiss(pdf_path, embed_model, vectordb)
    results = []
    ollama = OllamaClient()
    for question in DUMMY_QUESTIONS:
        # RAG: retrieve top 5 chunks
        query_emb = embed_model.encode([question])[0]
        top_chunks = vectordb.search(query_emb, top_k=5)
        context = "\n".join(top_chunks)
        prompt = build_rag_prompt(question, context)
        # Inference timing
        start = time.time()
        response = ""
        for chunk in ollama.chat_stream(model=model_name, messages=[{"role": "user", "content": prompt}]):
            response += chunk
        end = time.time()
        elapsed = end - start
        # Simple hallucination/context check: does response mention a key phrase from context?
        context_hit = any(word.lower() in response.lower() for word in question.split() if len(word) > 4)
        hallucination = not context_hit
        results.append({
            "question": question,
            "response": response,
            "inference_time": elapsed,
            "context_hit": context_hit,
            "hallucination": hallucination
        })
    return results

def main():
    # Hardcoded PDF path for evaluation
    pdf_path = r"/home/sayan/Desktop/Working_Dir/st_chat/data/doc2.pdf"  # Change this path as needed
    embed_model = EmbeddingModel()
    vectordb = VectorDB(dim=384)
    all_results = {}
    for model in MODELS:
        print(f"\nEvaluating model: {model}")
        model_results = evaluate_model_on_pdf(model, pdf_path, embed_model, vectordb)
        all_results[model] = model_results
        for res in model_results:
            print(f"Q: {res['question']}\nA: {res['response'][:200]}...\nTime: {res['inference_time']:.2f}s | Context hit: {res['context_hit']} | Hallucination: {res['hallucination']}\n")
    # Save results
    with open("model_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nEvaluation complete. Results saved to model_eval_results.json")

if __name__ == "__main__":
    main()
