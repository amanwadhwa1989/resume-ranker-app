"""
chains/evaluator.py
────────────────────
LangChain LCEL chains for the three reasoning stages:

  Stage 1 — DomainInferenceChain
      Company name + job title  →  DomainInference (JSON)

  Stage 2 — ResumeEvaluationChain
      Resume text + JD + DomainInference  →  ResumeEvaluation (JSON)

  Stage 3 — ResumeRankingPipeline (orchestrator)
      Runs both chains in sequence, returns the final ResumeEvaluation.

All chains use:
  • ChatAnthropic (claude-sonnet-4-5) as the LLM backbone
  • ChatPromptTemplate for structured prompts
  • PydanticOutputParser to enforce typed JSON output
  • LCEL ( prompt | llm | parser ) composition
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from schemas import DomainInference, ResumeEvaluation

logger = logging.getLogger(__name__)


# ── LLM factory ──────────────────────────────────────────────────────────────

def _make_llm(temperature: float = 0.1) -> ChatAnthropic:
    """
    Returns a ChatAnthropic instance.
    Temperature is kept low for deterministic, analytical outputs.
    """
    return ChatAnthropic(
        model="claude-sonnet-4-5",
        temperature=temperature,
        max_tokens=4096,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — DOMAIN INFERENCE CHAIN
# ═════════════════════════════════════════════════════════════════════════════

_DOMAIN_SYSTEM = """You are a senior talent intelligence analyst with encyclopaedic knowledge \
of industries, company profiles, and technology ecosystems worldwide.

Your task is to infer the employment domain context from a company name and optional job title. \
Think about:
  • What industry/sector does this company operate in?
  • What sub-domain or specialisation is relevant to this role?
  • What skills and technologies are typically expected of engineers/professionals here?
  • What regulatory or compliance environment does this company operate in?

Return ONLY a valid JSON object matching the schema below. No markdown fences. No preamble.

{format_instructions}"""

_DOMAIN_HUMAN = """Company: {company_name}
Job title: {job_title}

Infer the domain context for this company."""


def build_domain_inference_chain() -> Any:
    """
    Returns an LCEL chain:
      { company_name, job_title }  →  DomainInference
    """
    parser = PydanticOutputParser(pydantic_object=DomainInference)
    llm    = _make_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(_DOMAIN_SYSTEM),
        HumanMessagePromptTemplate.from_template(_DOMAIN_HUMAN),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — RESUME EVALUATION CHAIN
# ═════════════════════════════════════════════════════════════════════════════

_EVAL_SYSTEM = """You are a world-class technical recruiter and hiring specialist with 20+ years \
of experience evaluating candidates across software engineering, fintech, data science, \
product management, and operations roles.

## Your Task
Evaluate the provided resume against the job description. Use the pre-inferred domain context \
to calibrate your assessment of domain fit.

## Scoring Rubric — apply STRICTLY

Each dimension is scored 1–10 with these anchor points:
  9–10  Exceptional — clearly exceeds the requirement, strong evidence
  7–8   Strong — meets the requirement well, specific evidence present
  5–6   Moderate — partially meets, some gaps or limited evidence  
  3–4   Weak — significant gaps, requirements mostly not met
  1–2   Poor — major mismatch, requirement not met at all

### The 6 Mandatory Dimensions (weights must sum to 1.0):

| # | Dimension              | Weight | What to assess |
|---|------------------------|--------|----------------|
| 1 | Technical Skills       |  0.30  | Programming languages, frameworks, tools, cloud platforms mentioned in JD vs. resume |
| 2 | Domain Experience      |  0.25  | Industry/domain alignment: company domain vs. candidate's background |
| 3 | Years of Experience    |  0.15  | Required YoE in JD vs. total career span and relevant YoE in resume |
| 4 | Leadership & Scope     |  0.15  | Team size managed, project ownership, cross-functional influence, budget |
| 5 | Education & Certs      |  0.10  | Degree relevance, certifications, continuing education |
| 6 | Role Fit (Inferred)    |  0.05  | Culture, communication style, career trajectory fit |

**Overall score** = sum of (dimension_score × weight). Round to one decimal.

## Verdict Mapping
  8.0–10.0  →  "Strong match"
  6.5–7.9   →  "Good fit"
  4.5–6.4   →  "Partial match"
  2.5–4.4   →  "Weak match"
  1.0–2.4   →  "Not a fit"

## Domain Inference Context
{domain_context}

## Output Format
Return ONLY a valid JSON object matching the schema below. No markdown fences, no preamble.

{format_instructions}"""

_EVAL_HUMAN = """## Job Description
Company: {company_name}
Role: {job_title}

{job_description}

---

## Resume Text
{resume_text}

---

Evaluate this resume against the job description using the rubric above."""


def build_resume_evaluation_chain() -> Any:
    """
    Returns an LCEL chain:
      { resume_text, job_description, company_name, job_title, domain_context }
      →  ResumeEvaluation
    """
    parser = PydanticOutputParser(pydantic_object=ResumeEvaluation)
    llm    = _make_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(_EVAL_SYSTEM),
        HumanMessagePromptTemplate.from_template(_EVAL_HUMAN),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class ResumeRankingPipeline:
    """
    Orchestrates the two-stage LangChain evaluation pipeline.

    Usage:
        pipeline = ResumeRankingPipeline()
        result: ResumeEvaluation = pipeline.run(
            resume_text="...",
            job_description="...",
            company_name="InComm Payments",
            job_title="Staff Software Engineer",
        )

    Attributes:
        domain_chain:     DomainInferenceChain (Stage 1)
        evaluation_chain: ResumeEvaluationChain (Stage 2)
    """

    def __init__(self) -> None:
        self.domain_chain     = build_domain_inference_chain()
        self.evaluation_chain = build_resume_evaluation_chain()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        resume_text: str,
        job_description: str,
        company_name: str = "",
        job_title: str    = "",
    ) -> tuple[DomainInference, ResumeEvaluation]:
        """
        Run the full pipeline.

        Returns:
            (domain_inference, resume_evaluation) — both as Pydantic models.
        """
        logger.info("Stage 1: Inferring domain for '%s'", company_name or "unknown company")
        domain: DomainInference = self.domain_chain.invoke({
            "company_name": company_name or "Unknown Company",
            "job_title":    job_title or "Software Engineer",
        })

        logger.info("Stage 2: Evaluating resume against JD")
        domain_context = _format_domain_context(domain)
        evaluation: ResumeEvaluation = self.evaluation_chain.invoke({
            "resume_text":     resume_text,
            "job_description": job_description,
            "company_name":    company_name or "the company",
            "job_title":       job_title or "this role",
            "domain_context":  domain_context,
        })

        logger.info(
            "Evaluation complete — %s scored %.1f (%s)",
            evaluation.candidate_name,
            evaluation.overall_score,
            evaluation.verdict,
        )
        return domain, evaluation

    # ── Streaming variant ─────────────────────────────────────────────────────

    def stream_evaluation(
        self,
        resume_text: str,
        job_description: str,
        company_name: str = "",
        job_title: str    = "",
    ):
        """
        Generator that yields status updates during processing.
        Useful for Streamlit status/spinner messages.
        """
        yield "status", "Inferring company domain..."
        domain: DomainInference = self.domain_chain.invoke({
            "company_name": company_name or "Unknown Company",
            "job_title":    job_title or "Software Engineer",
        })
        yield "domain", domain

        yield "status", "Evaluating resume against job description..."
        domain_context = _format_domain_context(domain)
        evaluation: ResumeEvaluation = self.evaluation_chain.invoke({
            "resume_text":     resume_text,
            "job_description": job_description,
            "company_name":    company_name or "the company",
            "job_title":       job_title or "this role",
            "domain_context":  domain_context,
        })
        yield "status", "Scoring dimensions and compiling report..."
        yield "evaluation", evaluation


# ── Private helpers ───────────────────────────────────────────────────────────

def _format_domain_context(domain: DomainInference) -> str:
    """
    Serialize DomainInference into a compact readable context block
    that gets injected into the evaluation prompt.
    """
    skills_str = ", ".join(domain.key_domain_skills)
    return (
        f"Company Domain: {domain.company_domain}\n"
        f"Industry Sector: {domain.industry_sector}\n"
        f"Sub-domain: {domain.sub_domain}\n"
        f"Expected Skills in this Domain: {skills_str}\n"
        f"Regulatory Context: {domain.regulatory_environment}\n"
        f"Context: {domain.domain_context}"
    )
