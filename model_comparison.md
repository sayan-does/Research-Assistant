# Model Comparison Report: RAG Research Assistant

This report summarizes the evaluation of four small language models (SLMs) on a research paper using the RAG pipeline. The models tested were:
- **qwen2.5:0.5b**
- **qwen3:0.6b**
- **gemma3:1b**
- **tinyllama:1.1b**

## Evaluation Metrics
- **Context Hit:** Did the answer use relevant context from the retrieved chunks?
- **Hallucination:** Did the answer invent information not present in the context?
- **Inference Time:** How long did the model take to answer?
- **Answer Quality:** (Qualitative, based on sample outputs)

## Findings

### qwen2.5:0.5b
- **Strengths:** Good at extracting main contributions and key concepts. Most answers grounded in context.
- **Weaknesses:** Sometimes misses context for summary questions (hallucination detected). Inference time moderate (~60-70s per answer).

### qwen3:0.6b
- **Strengths:** Very cautious; explicitly states when context is insufficient. Good at listing key concepts and summarizing findings.
- **Weaknesses:** Sometimes over-cautious, but rarely hallucinates. Inference time high (~90-130s per answer).

### gemma3:1b
- **Strengths:** Can extract key concepts and list them clearly.
- **Weaknesses:** Prone to hallucination, especially for summary and limitation questions. Sometimes fails to ground answers in context. Inference time very high (up to 5100s for some queries).

### tinyllama:1.1b
- **Strengths:** None observed; responses are mostly incoherent or nonsensical, indicating the model is not suitable for RAG tasks on research papers.
- **Weaknesses:** Severe hallucination, no context grounding, extremely poor answer quality. Inference time very high (186-450s per answer).

## Quantitative Summary
| Model           | Context Hit | Hallucination | Avg. Inference Time (s) |
|-----------------|-------------|---------------|-------------------------|
| qwen2.5:0.5b    | 3/4         | 1/4           | ~69                     |
| qwen3:0.6b      | 4/4         | 0/4           | ~110                    |
| gemma3:1b       | 1/4         | 3/4           | ~140-5100               |
| tinyllama:1.1b  | 1/4         | 3/4           | ~310                    |

## Recommendations
- **Best for RAG:** `qwen3:0.6b` (most reliable, lowest hallucination, but slow)
- **Best speed/quality tradeoff:** `qwen2.5:0.5b`
- **Not recommended:** `gemma3:1b` and `tinyllama:1.1b` (hallucination, incoherence, or excessive latency)

## Notes
- All models struggle with complex queries and long context due to their small size.
- For production, use larger models or further optimize chunking and prompt engineering.
