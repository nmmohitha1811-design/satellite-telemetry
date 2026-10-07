# Satellite Telemetry Anomaly Copilot

An end-to-end AI system that detects anomalies in real NASA spacecraft telemetry, explains them using a retrieval-grounded language model, and orchestrates the whole process through a multi-agent pipeline with a visual dashboard.

Built as a 4-layer project combining classical deep learning, Retrieval-Augmented Generation (RAG), agentic orchestration (LangGraph), and a local LLM (Ollama) — using **real NASA JPL data and real JPL/NASA technical documents**, not synthetic examples.

**Author:** NM Mohitha
**Email:** nmmohitha1811@gmail.com
**LinkedIn:** [linkedin.com/in/nm-mohitha](https://www.linkedin.com/in/nm-mohitha)
**GitHub:** [github.com/nmmohitha1811-design](https://github.com/nmmohitha1811-design)

---

## What this project actually does

Ground station operators monitor thousands of telemetry channels per satellite pass. Catching anomalies manually is slow, and when something is flagged, historical context on *why* it might be happening is often scattered across old technical reports nobody has time to search during an active incident.

This project builds a small, honest version of a system that addresses both problems:

1. **Detects** anomalies in real spacecraft telemetry using an LSTM, reproducing a published NASA JPL research benchmark.
2. **Retrieves** relevant passages from real, verified NASA/JPL technical documents using a RAG pipeline — not invented sources.
3. **Explains** a detected anomaly in plain language via a local LLM, citing exactly which document and passage it used, and refusing to answer when it doesn't have grounded support.
4. **Visualizes** the whole pipeline in an interactive dashboard, including a free-form question-answering mode.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LAYER 4: Dashboard                          │
│                    (Streamlit — dashboard.py)                       │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
┌────────────────────────────────┴───────────────────────────────────┐
│                  LAYER 3: Multi-Agent Pipeline (LangGraph)          │
│                                                                      │
│   ┌──────────┐    ┌───────────┐    ┌──────────────┐   ┌──────────┐│
│   │ Detector │ ─▶ │ Retrieval │ ─▶ │ Report Writer │─▶│ Guardrail ││
│   │  Agent   │    │   Agent   │    │     Agent     │   │  Agent   ││
│   └────┬─────┘    └─────┬─────┘    └───────┬──────┘   └────┬─────┘│
└────────┼────────────────┼──────────────────┼────────────────┼─────┘
         │                │                  │                │
         ▼                ▼                  ▼                ▼
┌─────────────────┐  ┌──────────────────────────────┐  ┌─────────────┐
│    LAYER 1       │  │          LAYER 2              │  │  Ollama     │
│ Anomaly Detection│  │   RAG over real NASA/JPL docs  │  │ (local LLM) │
│  (LSTM, Keras)   │  │ (sentence-transformers + numpy)│  │ llama3.2:3b │
│                  │  │                                │  │             │
│ NASA SMAP/MSL    │  │ 3 verified full-text documents │  └─────────────┘
│ telemetry data   │  │ chunked, embedded, searchable  │
└─────────────────┘  └──────────────────────────────┘
```

---

## Layer 1 — Anomaly Detection (`telemanom/`)

Built on NASA JPL's own open-source **Telemanom** codebase (Hundman et al., KDD 2018), using the real public SMAP (satellite) and MSL/Curiosity (rover) telemetry dataset released alongside that paper.

**How it works:** An LSTM is trained per-channel on normal telemetry only (never shown labeled anomalies). At inference time, it predicts the next values in the stream; large, sustained prediction errors — found via a nonparametric dynamic thresholding method, not a simple fixed cutoff — are flagged as anomalies.

**What I did:**
- Trained an LSTM from scratch on a single channel (T-1) to understand the full pipeline: data windowing, the 2-layer 80-unit architecture, dropout, early stopping, and how prediction error becomes an anomaly flag.
- Reproduced the full 82-channel benchmark using the codebase's thresholding pipeline.

**Results (82 channels, 105 real labeled anomalies):**

| Metric | This run | Original paper |
|---|---|---|
| Precision | 87% | 87.5% |
| Recall | 83% | 80.0% |
| True Positives | 87 | — |
| False Positives | 13 | — |
| False Negatives | 18 | — |

**A finding from my own analysis:** Several SMAP channels (e.g. E-1, E-10, E-11, E-12, E-13) share near-identical labeled anomaly windows and false-alarm windows — strong evidence that a handful of physical events are being counted multiple times across channels in the per-channel scoring methodology. This means standard precision/recall figures here somewhat overweight events that happen to appear in many channels at once.

Full results: [`telemanom/results/full_benchmark_82_channels.csv`](telemanom/results/full_benchmark_82_channels.csv)

---

## Layer 2 — RAG over real NASA/JPL documents (`rag-copilot/`)

A Retrieval-Augmented Generation pipeline grounded in **3 verified, full-text, publicly available technical documents** — not scraped abstracts, not synthetic text:

1. **Hundman et al., 2018** — the original Telemanom paper itself (the same methodology used in Layer 1).
2. **Rankin et al. (JPL)** — "Mars Curiosity Rover Mobility Trends During the First Seven Years," a detailed engineering report on real fault-protection events (wheel slip, tilt, actuator faults) with specific sol numbers and thresholds.
3. **Sherwood et al. (JPL)** — "Lessons Learned During Implementation and Early Operations of the DS1 Beacon Monitor Experiment," on general spacecraft fault-response/anomaly-summarization practice.

**Pipeline:** PDF → text extraction (`pypdf`) → chunking (200 words, 40-word overlap, 216 total chunks) → embeddings (`sentence-transformers`, `all-MiniLM-L6-v2`, 384-dim) → cosine-similarity retrieval → grounded generation via a local LLM (`llama3.2:3b` via Ollama) → citation-enforced output.

**The guardrail threshold (0.35) was set empirically, not guessed:** real, relevant test questions scored 0.58–0.75; a deliberately irrelevant control question ("What is the capital of France?") scored 0.06–0.14. The threshold sits in the gap between these measured ranges.

---

## Layer 3 — Multi-Agent Orchestration (`agent-copilot/`, LangGraph)

Four agents, each with one job, connected via a LangGraph state machine with a genuine conditional branch:

- **Detector Agent** — reads Layer 1's real benchmark CSV and extracts a specific anomaly's facts (channel, spacecraft, class, true/false positive counts).
- **Retrieval Agent** — auto-generates a natural-language query from those facts and searches Layer 2's document index.
- **Report Writer Agent** — asks the local LLM to draft a short, cited incident explanation using only the retrieved passages.
- **Guardrail Agent** — checks the retrieval confidence score against the measured 0.35 threshold. Below it, the pipeline **skips the LLM call entirely** and routes to an automatic "insufficient grounding, escalate to human" response, rather than generating and then hiding a result.

---

## Layer 4 — Dashboard (`agent-copilot/dashboard.py`, Streamlit)

A local web UI (`streamlit run dashboard.py`) with two modes:
- **Channel analysis** — pick any channel with a real caught anomaly, run the full 4-agent pipeline, see the detection summary, retrieval confidence, source passages, and final cited report.
- **Ask a Question** — free-form Q&A against the same document index, showing retrieval scores and sources live, so the grounding (or refusal) is always inspectable, not a black box.

---

## Key findings (discovered through testing, not assumed)

This project surfaced two genuine limitations worth being upfront about:

**1. Small local models can misstate simple facts even when given correct input.**
On channel D-8, the structured input correctly stated *"the model correctly caught 0, missed 1."* The LLM's generated report nonetheless said the anomaly *"was correctly caught by the model."* The grounding documents weren't involved in this specific claim at all — this was the model misreading a plain factual sentence during summarization. A production system would need an automated fact-check step (e.g., verifying numeric/boolean claims in the output against the structured input) rather than trusting generation alone for factual fields.

**2. A high similarity score does not guarantee the retrieved text contains the answer.**
Two separate test questions ("what is the satellites' mission" and "causes of anomalies in the DS1 beacon monitor") retrieved genuinely on-topic, substantive passages with strong similarity scores (0.42–0.60, well above the 0.35 threshold) — but neither set of passages actually answered the specific question asked. In both cases, it was the LLM's own instruction-following, not the retrieval threshold, that correctly produced a refusal instead of a fabricated answer. Relying on retrieval score alone as a guardrail would not have caught either case.

---

## Setup & Reproduction

This project uses **three separate Python virtual environments** — one per layer — to avoid dependency conflicts (notably: TensorFlow in Layer 1 requires an older NumPy than the embedding/LLM libraries in Layers 2–4).

### Prerequisites
- Python 3.11+ (3.13 used in development)
- Git
- [Ollama](https://ollama.com) installed, with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`)
- A free [Kaggle](https://kaggle.com) account + API token (for the Layer 1 dataset)

### Layer 1 — Anomaly Detection

```bash
git clone https://github.com/khundman/telemanom.git
cd telemanom
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt    # edit version pins if using Python 3.11+

pip install kaggle
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
# unzip, and place train/ and test/ folders under telemanom/data/

# config.yaml: set train: True, predict: True to train from scratch,
# or train: False, predict: False to reuse saved predictions
python example.py -l labeled_anomalies.csv
```

### Layer 2 — RAG Pipeline

```bash
cd ../rag-copilot
python -m venv venv
venv\Scripts\activate
pip install sentence-transformers pypdf

python download_docs.py      # fetches the 3 verified source PDFs
python extract_text.py       # PDF -> plain text
python chunk_documents.py    # text -> 216 chunks
python embed_chunks.py       # chunks -> embeddings

python rag_answer.py         # interactive Q&A in the terminal
```

### Layer 3 — Multi-Agent Pipeline

```bash
cd ../agent-copilot
# Reuses rag-copilot's venv (no conflicting dependencies)
..\rag-copilot\venv\Scripts\activate
pip install pandas langgraph ollama

python graph_pipeline.py     # runs the full 4-agent pipeline on one anomaly
```

### Layer 4 — Dashboard

```bash
# Still inside rag-copilot's venv
pip install streamlit
streamlit run dashboard.py
```

---

## Repository Structure

```
space-copilot/
├── telemanom/              # Layer 1: LSTM anomaly detection (NASA JPL codebase + my training/analysis)
│   ├── telemanom/          # Core model/detector code
│   ├── results/            # Real benchmark results (87%/83%)
│   ├── labeled_anomalies.csv
│   └── visualize_channel.py, plot_predictions.py   # my own analysis scripts
├── rag-copilot/            # Layer 2: RAG pipeline
│   ├── download_docs.py, extract_text.py, chunk_documents.py, embed_chunks.py
│   ├── search.py, rag_answer.py
│   └── chunks.json, chunk_embeddings.npy
├── agent-copilot/          # Layer 3 + 4: Agents and dashboard
│   ├── detector_agent.py, retrieval_agent.py, report_agent.py, guardrail_agent.py
│   ├── graph_pipeline.py   # LangGraph orchestration
│   └── dashboard.py        # Streamlit UI
└── README.md
```

---

## Tech Stack

**ML/DL:** TensorFlow/Keras (LSTM), scikit-learn
**RAG:** sentence-transformers, NumPy (cosine similarity), pypdf
**Agents:** LangGraph
**LLM:** Ollama (Llama 3.2 3B, local inference)
**Dashboard:** Streamlit
**Data:** NASA JPL SMAP/MSL telemetry (Hundman et al. 2018), real JPL technical publications

---

## Future Work

- Expand the document corpus with SMAP-specific fault documentation (current corpus is MSL-heavy, which measurably weakens retrieval quality for SMAP-specific queries).
- Build a hand-labeled retrieval evaluation set (~15–20 question/answer pairs) to report precision@k rather than anecdotal score examples.
- Add an automated fact-check guardrail that verifies structured numeric/boolean claims (e.g. caught vs. missed) against the report before display.
- Event-level (rather than per-channel) scoring for Layer 1, to correct for the repeated-event double-counting found during analysis.

---

## Acknowledgments

Layer 1 builds directly on NASA JPL's open-source [Telemanom](https://github.com/khundman/telemanom) repository (Apache 2.0 license) and the accompanying paper:

> Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. *KDD '18*.

Layer 2's document corpus includes real JPL technical publications, used here for research/educational purposes with full citation.
