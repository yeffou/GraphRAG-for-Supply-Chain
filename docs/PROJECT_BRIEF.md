# Project Brief

Course: Generative AI final project

Chosen topic:
GraphRAG domain specific LLMs

Chosen domain:
Supply-chain risk and resilience

Target system:
A domain-specific question answering assistant over real supply-chain reports/papers.

Core comparison:
1. Baseline vector RAG
2. GraphRAG using extracted entities and relations

Requirements:
- real data only
- no dummy or hardcoded examples
- simple but serious academic project
- evaluation must be real
- fine-tuning is optional, not part of MVP

Presentation goal:
Show that graph-structured retrieval can improve QA over supply-chain reports compared with standard RAG.

Preferred stack:
- Python
- PDF parsing
- sentence-transformers
- FAISS or Chroma
- NetworkX
- OpenRouter-compatible LLM API
- Streamlit

Constraints:
- keep architecture simple
- prioritize correctness and explainability
- do not overengineer