# 🎯 Resume Rank — AI-Powered Resume Evaluator

A production-grade resume ranking system built with **Streamlit** (UI) and **LangChain** (reasoning layer). The LLM produces a fully-typed structured JSON output which drives every element of the UI — no post-processing guesswork.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                         │
│  ─ Input form: company, job title, JD text, resume upload       │
│  ─ Results dashboard: verdict, dial, bars, domain, findings     │
└────────────────────────┬────────────────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────────────────┐
│  ResumeRankingPipeline  (chains/evaluator.py)                   │
│                                                                 │
│  Stage 1 — DomainInferenceChain                                 │
│    ChatPromptTemplate → ChatAnthropic → PydanticOutputParser    │
│    Input : company name + job title                             │
│    Output: DomainInference (JSON)                               │
│      • company_domain, industry_sector, sub_domain              │
│      • key_domain_skills, regulatory_environment                │
│                                                                 │
│  Stage 2 — ResumeEvaluationChain                                │
│    ChatPromptTemplate → ChatAnthropic → PydanticOutputParser    │
│    Input : resume text + JD + DomainInference context           │
│    Output: ResumeEvaluation (JSON)                              │
│      • 6-dimension scoring rubric with weights                  │
│      • strengths, gaps, standout signals, recommendations       │
│      • domain_analysis, summary, hiring_recommendation          │
└────────────────────────┬────────────────────────────────────────┘
                         │ parses into
┌────────────────────────▼────────────────────────────────────────┐
│  Pydantic v2 Schemas  (schemas/evaluation.py)                   │
│  ─ DomainInference                                              │
│  ─ DimensionScore (× 6)                                         │
│  ─ ResumeEvaluation (complete structured output)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scoring Rubric

| # | Dimension            | Weight | What is assessed                                         |
|---|----------------------|--------|----------------------------------------------------------|
| 1 | Technical Skills     |  30%   | Languages, frameworks, tools vs JD requirements         |
| 2 | Domain Experience    |  25%   | Industry/domain alignment with inferred company domain  |
| 3 | Years of Experience  |  15%   | Required YoE in JD vs candidate's career span           |
| 4 | Leadership & Scope   |  15%   | Team size, project ownership, cross-functional impact   |
| 5 | Education & Certs    |  10%   | Degree relevance, professional certifications           |
| 6 | Role Fit (Inferred)  |   5%   | Culture, trajectory, communication style fit            |

**Verdict mapping:**
- 8.0–10.0 → Strong match  
- 6.5–7.9  → Good fit  
- 4.5–6.4  → Partial match  
- 2.5–4.4  → Weak match  
- 1.0–2.4  → Not a fit  

---

## Project Structure

```
resume_ranker/
├── app.py                   # Streamlit application (UI layer)
├── requirements.txt
├── .env.example
│
├── chains/
│   ├── __init__.py
│   └── evaluator.py         # LangChain LCEL chains + pipeline orchestrator
│
├── schemas/
│   ├── __init__.py
│   └── evaluation.py        # Pydantic v2 models (JSON contract)
│
└── utils/
    ├── __init__.py
    └── file_parser.py       # PDF + DOCX text extraction
```

---

## Setup & Run

### 1. Clone / download

```bash
git clone <repo-url>
cd resume_ranker
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How to Use

1. Enter the **company name** — this drives the domain inference chain.
2. Enter the **job title** (optional but improves accuracy).
3. Paste the full **job description**.
4. Upload a **PDF or DOCX** resume.
5. Click **Analyze resume**.

The pipeline runs two LangChain chains in sequence (~15–25 seconds), then renders the full dashboard.

---

## LangSmith Tracing (Optional)

To inspect chain traces in LangSmith:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=resume-ranker
```

---

## Extending the System

### Add a new scoring dimension
1. Update the dimension table in the `_EVAL_SYSTEM` prompt in `chains/evaluator.py`.
2. Adjust weights so they still sum to 1.0.
3. Update the `DimensionScore` min/max length in `schemas/evaluation.py`.

### Swap the LLM
Replace `ChatAnthropic` in `chains/evaluator.py` with any LangChain-compatible chat model:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
```

### Add multi-resume batch mode
Call `pipeline.run(...)` in a loop and aggregate scores into a `st.dataframe`.
