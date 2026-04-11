# GraphRAG for Supply-Chain Risk and Resilience QA

University final project for a Generative AI course.

## Objective

Build a domain-specific question answering system over real supply-chain reports and compare two retrieval approaches:

1. Baseline vector RAG over document chunks.
2. GraphRAG using extracted entities and relations.

The project is intentionally minimal and transparent. It should be possible to inspect the corpus, the retrieved evidence, the generated answers, and the evaluation outputs.

## MVP Scope

- Use real authoritative public reports only.
- Prefer reports from OECD, WTO, World Bank, UNCTAD, government or agency sources, and directly relevant WEF material.
- Use 6-12 substantial reports, prioritizing rich repeated entities and relations over raw document count.
- Keep company annual or sustainability reports out of the primary MVP corpus.
- Do not use fine-tuning for the MVP.
- Do not fabricate data, metrics, answers, or pipeline outputs.

## Planned Architecture

```text
authoritative public PDFs
  -> corpus manifest
  -> PDF ingestion
  -> page-level text extraction
  -> chunking with metadata
  -> baseline vector index
  -> baseline RAG answering
  -> entity/relation extraction
  -> NetworkX graph construction
  -> graph-based retrieval
  -> GraphRAG answering
  -> evaluation + saved run logs
  -> optional final Streamlit demo
```

## Corpus Manifest

The corpus will be tracked through `data/manifest.jsonl` once real reports are selected. Each line should describe one real document:

```json
{
  "doc_id": "short_stable_id",
  "title": "Report title",
  "source_url": "https://...",
  "publisher": "OECD/WTO/World Bank/UNCTAD/etc.",
  "year": 2024,
  "accessed_at": "2026-04-10",
  "local_path": "data/raw/example.pdf",
  "domain_tags": ["supply_chain", "resilience", "trade"],
  "included_reason": "Explains supply-chain disruptions and policy resilience measures."
}
```

Do not add fake manifest entries. The manifest should only contain real selected reports.

## Repository Layout

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env.example
├── docs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── scripts/
├── src/
│   ├── config.py
│   ├── ingestion/
│   ├── preprocessing/
│   ├── baseline_rag/
│   ├── graph_rag/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
├── tests/
└── results/
    ├── indexes/
    ├── graphs/
    ├── runs/
    └── evaluations/
```

## Results Logging Requirements

Every future run or evaluation output must save enough information to reproduce and audit it:

- method
- LLM model
- embedding model
- retrieval parameters
- prompt template version
- timestamp
- dataset or manifest path/version
- question
- retrieved evidence
- final answer
- output path
- errors or warnings, if any

## Evaluation Plan

The evaluation set should contain 20-30 real questions derived from the selected reports and organized by:

- factual
- relation-based
- multi-hop
- mitigation/comparison

Baseline RAG and GraphRAG must answer the same questions. Scores and retrieved evidence must be saved under `results/evaluations/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` only with real local credentials when API-based generation or extraction is implemented.

## Current Status

This repository currently contains the initial skeleton only. Ingestion, retrieval, graph construction, generation, evaluation, and the demo app are not implemented yet.

