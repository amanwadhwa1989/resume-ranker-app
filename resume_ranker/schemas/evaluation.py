"""
schemas/evaluation.py
─────────────────────
Pydantic v2 models that define the structured JSON contract between
the LangChain reasoning layer and the Streamlit UI.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


# ── Verdict metadata ──────────────────────────────────────────────────────────

VERDICT_LEVELS: dict[str, dict] = {
    "Strong match": {
        "level": "success",
        "icon":  "✅",
        "color": "#0F6E56",
        "bg":    "#E1F5EE",
        "border":"#9FE1CB",
    },
    "Good fit": {
        "level": "info",
        "icon":  "🔵",
        "color": "#185FA5",
        "bg":    "#E6F1FB",
        "border":"#B5D4F4",
    },
    "Partial match": {
        "level": "warning",
        "icon":  "⚠️",
        "color": "#854F0B",
        "bg":    "#FAEEDA",
        "border":"#FAC775",
    },
    "Weak match": {
        "level": "danger",
        "icon":  "❌",
        "color": "#A32D2D",
        "bg":    "#FCEBEB",
        "border":"#F7C1C1",
    },
    "Not a fit": {
        "level": "danger",
        "icon":  "🚫",
        "color": "#A32D2D",
        "bg":    "#FCEBEB",
        "border":"#F7C1C1",
    },
}


# ── Domain inference ──────────────────────────────────────────────────────────

class DomainInference(BaseModel):
    """
    Output of the DomainInferenceChain.
    Infers the employer's industry, sub-domain, and key technology expectations
    purely from the company name (and optionally the job title).
    """

    company_domain: str = Field(
        description="Primary industry/domain of the company, e.g. 'Fintech / Payments'"
    )
    industry_sector: str = Field(
        description="Broader industry sector, e.g. 'Financial Services'"
    )
    sub_domain: str = Field(
        description="More specific sub-domain if applicable, e.g. 'Gift Card Processing / Fraud Detection'"
    )
    key_domain_skills: List[str] = Field(
        description="5–8 skills/technologies typically expected in this domain",
        min_length=3,
    )
    domain_context: str = Field(
        description=(
            "2–3 sentence narrative about what this company's domain means for a "
            "candidate: culture, typical stack, key challenges."
        )
    )
    regulatory_environment: str = Field(
        description="Any notable compliance/regulatory context (PCI-DSS, HIPAA, SOX, etc.) or 'Not applicable'",
        default="Not applicable",
    )


# ── Per-dimension score ───────────────────────────────────────────────────────

class DimensionScore(BaseModel):
    """A single evaluated dimension from the scoring rubric."""

    name: str = Field(description="Dimension name, e.g. 'Technical Skills'")
    score: float = Field(ge=1.0, le=10.0, description="Score from 1–10 (one decimal)")
    rationale: str = Field(
        description="One specific sentence explaining the score with evidence from the resume"
    )
    evidence: str = Field(
        description="Direct evidence or quote from the resume supporting this score"
    )
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Relative weight used in overall score calculation (all weights sum to 1.0)",
    )

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 1)

    @property
    def weighted_score(self) -> float:
        return round(self.score * self.weight, 2)

    @property
    def bar_pct(self) -> float:
        """Percentage width for progress bars."""
        return self.score / 10.0 * 100


# ── Full resume evaluation ────────────────────────────────────────────────────

class ResumeEvaluation(BaseModel):
    """
    Complete structured output of the ResumeEvaluationChain.
    This is the single JSON object consumed by the Streamlit UI.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    candidate_name: str = Field(
        description="Full name extracted from the resume, or 'Candidate' if not found"
    )
    contact_email: str = Field(
        default="",
        description="Email address if present in the resume",
    )
    years_of_experience: float = Field(
        ge=0,
        description="Total years of professional experience inferred from resume",
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    overall_score: float = Field(
        ge=1.0,
        le=10.0,
        description="Weighted overall score (1–10, one decimal)",
    )
    verdict: Literal["Strong match", "Good fit", "Partial match", "Weak match", "Not a fit"] = Field(
        description="Final match verdict"
    )

    # ── Scoring rubric dimensions ─────────────────────────────────────────────
    dimensions: List[DimensionScore] = Field(
        description=(
            "Exactly 6 scored dimensions: Technical Skills, Domain Experience, "
            "Years of Experience, Leadership & Scope, Education & Certifications, Role Fit (Inferred)"
        ),
        min_length=6,
        max_length=6,
    )

    # ── Domain fit ────────────────────────────────────────────────────────────
    domain_match_score: float = Field(
        ge=1.0,
        le=10.0,
        description="Standalone domain alignment score (1–10)",
    )
    candidate_domain_exp: str = Field(
        description="Primary domain/industry of the candidate's background"
    )
    domain_analysis: str = Field(
        description=(
            "2–3 sentence narrative comparing the company's domain expectations "
            "against the candidate's domain background"
        )
    )

    # ── Qualitative findings ──────────────────────────────────────────────────
    strengths: List[str] = Field(
        description="3–5 specific strengths relevant to this JD",
        min_length=3,
        max_length=5,
    )
    gaps: List[str] = Field(
        description="3–5 specific gaps or missing requirements",
        min_length=3,
        max_length=5,
    )
    standout_signals: List[str] = Field(
        description="2–3 notable differentiators or red flags",
        min_length=2,
        max_length=3,
    )
    recommendations: List[str] = Field(
        description="2–3 actionable suggestions for the candidate",
        min_length=2,
        max_length=3,
    )

    # ── Narrative ─────────────────────────────────────────────────────────────
    summary: str = Field(
        description=(
            "3–4 sentence narrative evaluation tying everything together. "
            "Reference specific resume content."
        )
    )
    hiring_recommendation: str = Field(
        description=(
            "One-sentence hiring recommendation for the recruiter, e.g. "
            "'Recommend for technical interview — strong fintech background aligns well with role requirements.'"
        )
    )

    @field_validator("overall_score", "domain_match_score")
    @classmethod
    def round_scores(cls, v: float) -> float:
        return round(v, 1)

    @property
    def verdict_meta(self) -> dict:
        return VERDICT_LEVELS.get(self.verdict, VERDICT_LEVELS["Partial match"])

    @property
    def score_color(self) -> str:
        if self.overall_score >= 8:
            return "#1D9E75"
        if self.overall_score >= 6:
            return "#378ADD"
        if self.overall_score >= 4:
            return "#BA7517"
        return "#E24B4A"

    @property
    def score_label(self) -> str:
        if self.overall_score >= 8:
            return "Excellent"
        if self.overall_score >= 6:
            return "Good"
        if self.overall_score >= 4:
            return "Fair"
        return "Poor"
