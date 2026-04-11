# AGENTS.md

Project: GraphRAG for Supply-Chain Risk and Resilience Question Answering

You are working on a university final project.

## Core objective
Build an end-to-end domain-specific QA system over real supply-chain reports.
Compare:
1. baseline vector RAG
2. GraphRAG using extracted entities and relations

## Non-negotiable rules
- Never use dummy, toy, or fabricated data unless explicitly requested for a unit test.
- Never hardcode answers to make a demo appear functional.
- Never fabricate metrics, evaluation results, or claims of success.
- Never pretend a pipeline works without providing a way to run and verify it.
- If a dataset schema is unknown, inspect it first instead of assuming.
- If a step is uncertain, state the uncertainty clearly.
- Keep the implementation minimal, modular, and defensible.
- Prefer simple libraries and transparent code.
- Separate baseline RAG code from GraphRAG code clearly.
- Every experiment must save outputs under results/.
- Every evaluation must log dataset, method, parameters, and outputs.
- Fine-tuning is optional and not part of the MVP unless explicitly requested later.

## Working style
- Before coding, always state the plan briefly.
- List files to create or modify before making changes.
- Keep changes scoped to the current task.
- After coding, explain:
  1. what was implemented
  2. how to run it
  3. how to verify it
  4. current limitations

## Project scope
Preferred stack:
- Python
- PDF extraction
- sentence-transformers
- FAISS or Chroma
- NetworkX
- OpenRouter-compatible LLM API
- Streamlit

## Architecture target
- ingestion
- preprocessing
- baseline_rag
- graph construction
- graph retrieval
- generation
- evaluation
- demo app