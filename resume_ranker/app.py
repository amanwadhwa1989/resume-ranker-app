"""
app.py — Resume Ranking System
══════════════════════════════
Streamlit UI that consumes structured JSON from the LangChain reasoning layer
and renders a full evaluation dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import math
import time
import logging
from pathlib import Path

import streamlit as st

# ── Path setup so relative imports resolve ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from chains   import ResumeRankingPipeline
from schemas  import DomainInference, ResumeEvaluation, VERDICT_LEVELS
from utils    import extract_resume_text, word_count

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Resume Rank — AI Evaluator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Typography ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
code, pre { font-family: 'DM Mono', monospace !important; }

/* ── Hide Streamlit chrome ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; }

/* ── Score dial ──────────────────────────────────────────────────────── */
.dial-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.score-circle {
  width: 140px; height: 140px;
  border-radius: 50%;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  font-weight: 600;
}

/* ── Metric cards ────────────────────────────────────────────────────── */
.metric-card {
  background: #f8f7f3;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 12px;
}
.metric-card h4 {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .06em; color: #888; margin-bottom: 4px;
}
.metric-card p { margin: 0; }

/* ── Verdict banner ──────────────────────────────────────────────────── */
.verdict-banner {
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 20px;
  border: 1px solid;
}

/* ── Progress bar overrides ──────────────────────────────────────────── */
div.stProgress > div > div > div > div { border-radius: 6px; }

/* ── Tag pills ───────────────────────────────────────────────────────── */
.tag-pill {
  display: inline-block;
  font-size: 12px; font-weight: 500;
  padding: 3px 10px;
  border-radius: 100px;
  margin: 2px 3px;
  border: 1px solid;
}

/* ── Section divider ─────────────────────────────────────────────────── */
.section-divider {
  height: 1px;
  background: rgba(0,0,0,.08);
  margin: 24px 0;
}

/* ── Highlight box ───────────────────────────────────────────────────── */
.highlight-box {
  border-left: 3px solid;
  border-radius: 0 8px 8px 0;
  padding: 14px 18px;
  margin: 8px 0;
  background: #f8f7f3;
}

/* ── Domain comparison ───────────────────────────────────────────────── */
.domain-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 12px;
  align-items: center;
  text-align: center;
}
.domain-box {
  background: #f0efe9;
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 13px;
  border: 1px solid rgba(0,0,0,.07);
}
.domain-arrow { font-size: 22px; color: #888; }

/* ── Numbered finding ────────────────────────────────────────────────── */
.finding-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 8px 0;
  border-bottom: 1px solid rgba(0,0,0,.06);
  font-size: 14px;
  line-height: 1.6;
}
.finding-item:last-child { border-bottom: none; }
.finding-num {
  flex-shrink: 0;
  width: 22px; height: 22px;
  border-radius: 50%;
  font-size: 11px; font-weight: 600;
  display: flex; align-items: center; justify-content: center;
  margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "domain":     None,      # DomainInference | None
        "evaluation": None,      # ResumeEvaluation | None
        "error":      None,      # str | None
        "ran":        False,     # bool — has an analysis been completed?
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset() -> None:
    for k in ("domain", "evaluation", "error", "ran"):
        st.session_state[k] = None
    st.session_state["ran"] = False


# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _score_color(score: float) -> str:
    if score >= 8:  return "#1D9E75"
    if score >= 6:  return "#378ADD"
    if score >= 4:  return "#BA7517"
    return "#E24B4A"


def _score_bg(score: float) -> str:
    if score >= 8:  return "#E1F5EE"
    if score >= 6:  return "#E6F1FB"
    if score >= 4:  return "#FAEEDA"
    return "#FCEBEB"


def _render_score_dial(score: float, label: str = "") -> None:
    """SVG-based circular score dial rendered as HTML."""
    circumference = 2 * math.pi * 54  # r=54
    offset = circumference * (1 - score / 10)
    color  = _score_color(score)
    bg_col = _score_bg(score)

    dial_html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
      <div style="position:relative;width:130px;height:130px">
        <svg width="130" height="130" viewBox="0 0 130 130" style="transform:rotate(-90deg)">
          <circle cx="65" cy="65" r="54"
            fill="none" stroke="#e5e5e0" stroke-width="11"/>
          <circle cx="65" cy="65" r="54"
            fill="none"
            stroke="{color}"
            stroke-width="11"
            stroke-linecap="round"
            stroke-dasharray="{circumference:.2f}"
            stroke-dashoffset="{offset:.2f}"/>
        </svg>
        <div style="
          position:absolute;inset:0;
          display:flex;flex-direction:column;
          align-items:center;justify-content:center;
        ">
          <span style="font-size:32px;font-weight:700;color:{color};line-height:1">{score}</span>
          <span style="font-size:12px;color:#999;margin-top:2px">/ 10</span>
        </div>
      </div>
      {f'<span style="font-size:13px;color:#666;font-weight:500">{label}</span>' if label else ''}
    </div>
    """
    st.html(dial_html)


def _render_progress_row(label: str, score: float, rationale: str, evidence: str) -> None:
    """Single dimension row with bar, score, rationale, and evidence."""
    color = _score_color(score)
    pct   = int(score / 10 * 100)

    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
          <span style="font-size:13px;font-weight:600;color:#444">{label}</span>
          <span style="font-size:15px;font-weight:700;color:{color}">{score:.1f}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.progress(pct / 100, text=None)
    st.markdown(
        f'<p style="font-size:12px;color:#777;margin:-6px 0 4px;line-height:1.5">{rationale}</p>',
        unsafe_allow_html=True,
    )
    if evidence:
        st.markdown(
            f'<p style="font-size:11px;color:#aaa;font-style:italic;margin:-4px 0 12px;line-height:1.5">'
            f'Evidence: {evidence}</p>',
            unsafe_allow_html=True,
        )


def _render_finding_list(items: list[str], dot_color: str = "#888") -> None:
    for item in items:
        st.markdown(
            f"""<div class="finding-item">
              <div class="finding-num" style="background:{dot_color}22;color:{dot_color}">•</div>
              <span>{item}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_tag(text: str, style: str = "blue") -> str:
    styles = {
        "blue":  ("background:#E6F1FB;color:#185FA5;border-color:#B5D4F4"),
        "green": ("background:#E1F5EE;color:#0F6E56;border-color:#9FE1CB"),
        "amber": ("background:#FAEEDA;color:#854F0B;border-color:#FAC775"),
        "gray":  ("background:#f0efe9;color:#555;border-color:#ddd"),
    }
    css = styles.get(style, styles["gray"])
    return f'<span class="tag-pill" style="{css}">{text}</span>'


# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS RENDERER
# ═════════════════════════════════════════════════════════════════════════════

def render_results(
    domain: DomainInference,
    ev: ResumeEvaluation,
    company: str,
    job_title: str,
) -> None:
    """
    Render the full evaluation dashboard from the structured JSON output.
    All data comes from the Pydantic models produced by LangChain chains.
    """
    vm = ev.verdict_meta

    # ── 1. Verdict banner ─────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="verdict-banner"
          style="background:{vm['bg']};border-color:{vm['border']}">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
            <div>
              <div style="font-size:22px;font-weight:700;color:{vm['color']};margin-bottom:2px">
                {vm['icon']} {ev.verdict}
              </div>
              <div style="font-size:14px;color:{vm['color']}88">
                {ev.hiring_recommendation}
              </div>
            </div>
            <div style="text-align:right">
              <div style="font-size:11px;color:{vm['color']}88;text-transform:uppercase;letter-spacing:.06em">Overall score</div>
              <div style="font-size:36px;font-weight:800;color:{vm['color']};line-height:1">{ev.overall_score}</div>
              <div style="font-size:12px;color:{vm['color']}88">out of 10 · {ev.score_label}</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── 2. Top metrics row ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Candidate", ev.candidate_name)
    with m2:
        st.metric("Years of experience", f"{ev.years_of_experience:.0f} yrs")
    with m3:
        st.metric("Domain match", f"{ev.domain_match_score}/10")
    with m4:
        email_display = ev.contact_email if ev.contact_email else "—"
        st.metric("Contact", email_display)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 3. Two-column layout: Score Breakdown + Domain Analysis ───────────────
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # LEFT — Score breakdown
    with col_left:
        st.markdown("#### 📊 Score breakdown")
        for dim in ev.dimensions:
            _render_progress_row(
                label=dim.name,
                score=dim.score,
                rationale=dim.rationale,
                evidence=dim.evidence,
            )

    # RIGHT — Domain analysis
    with col_right:
        st.markdown("#### 🏢 Domain analysis")

        # Domain comparison card
        st.markdown(
            f"""<div style="
              background:#f8f7f3;border-radius:12px;
              padding:16px;border:1px solid rgba(0,0,0,.07);
              margin-bottom:14px">
              <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:center;text-align:center;margin-bottom:14px">
                <div style="background:#E6F1FB;border-radius:10px;padding:10px 12px;border:1px solid #B5D4F4">
                  <div style="font-size:10px;font-weight:600;color:#185FA5;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Company domain</div>
                  <div style="font-size:13px;font-weight:600;color:#185FA5">{domain.company_domain}</div>
                  <div style="font-size:11px;color:#185FA588;margin-top:2px">{domain.industry_sector}</div>
                </div>
                <div style="font-size:20px;color:#ccc">⇄</div>
                <div style="background:#E1F5EE;border-radius:10px;padding:10px 12px;border:1px solid #9FE1CB">
                  <div style="font-size:10px;font-weight:600;color:#0F6E56;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Candidate background</div>
                  <div style="font-size:13px;font-weight:600;color:#0F6E56">{ev.candidate_domain_exp}</div>
                </div>
              </div>
              <p style="font-size:13px;color:#555;line-height:1.7;margin:0">{ev.domain_analysis}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Domain match score bar
        dc = _score_color(ev.domain_match_score)
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
              <span style="font-size:12px;color:#888;white-space:nowrap">Domain match score</span>
              <div style="flex:1;height:6px;background:#e5e5e0;border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{ev.domain_match_score*10:.0f}%;background:{dc};border-radius:3px"></div>
              </div>
              <span style="font-size:14px;font-weight:700;color:{dc};min-width:30px">{ev.domain_match_score}/10</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Expected domain skills
        st.markdown(
            '<p style="font-size:12px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Expected domain skills</p>',
            unsafe_allow_html=True,
        )
        tags_html = " ".join(_render_tag(s, "blue") for s in domain.key_domain_skills)
        st.markdown(tags_html, unsafe_allow_html=True)

        if domain.regulatory_environment and domain.regulatory_environment.lower() != "not applicable":
            st.markdown(
                f'<div style="margin-top:10px">'
                + _render_tag(f"⚖️ {domain.regulatory_environment}", "amber")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 4. Qualitative findings ───────────────────────────────────────────────
    st.markdown("#### 🔍 Detailed findings")
    f1, f2 = st.columns(2, gap="medium")

    with f1:
        with st.container(border=True):
            st.markdown("**✅ Strengths**")
            _render_finding_list(ev.strengths, dot_color="#1D9E75")
        st.markdown("")
        with st.container(border=True):
            st.markdown("**💡 Recommendations**")
            _render_finding_list(ev.recommendations, dot_color="#378ADD")

    with f2:
        with st.container(border=True):
            st.markdown("**⚠️ Gaps & missing requirements**")
            _render_finding_list(ev.gaps, dot_color="#BA7517")
        st.markdown("")
        with st.container(border=True):
            st.markdown("**⭐ Standout signals**")
            _render_finding_list(ev.standout_signals, dot_color="#7F77DD")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 5. Overall summary ────────────────────────────────────────────────────
    st.markdown("#### 📝 Overall assessment")
    st.markdown(
        f"""<div class="highlight-box" style="border-color:{vm['color']};">
          <p style="font-size:14px;color:#444;line-height:1.8;margin:0">{ev.summary}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 6. Raw JSON expander ──────────────────────────────────────────────────
    with st.expander("🔧 Raw JSON output (LLM structured response)", expanded=False):
        tab_eval, tab_domain = st.tabs(["Resume evaluation", "Domain inference"])
        with tab_eval:
            st.json(ev.model_dump())
        with tab_domain:
            st.json(domain.model_dump())

    # ── 7. Actions ────────────────────────────────────────────────────────────
    st.markdown("")
    col_dl, col_reset, _ = st.columns([1, 1, 3])
    with col_dl:
        import json
        export = {"domain": domain.model_dump(), "evaluation": ev.model_dump()}
        st.download_button(
            label="⬇️ Download JSON report",
            data=json.dumps(export, indent=2),
            file_name=f"resume_rank_{ev.candidate_name.replace(' ','_').lower()}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_reset:
        if st.button("🔄 New analysis", use_container_width=True):
            _reset()
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ═════════════════════════════════════════════════════════════════════════════

def render_input_form() -> None:
    st.markdown(
        """<div style="margin-bottom:2rem">
          <h1 style="font-size:28px;font-weight:700;letter-spacing:-.02em;margin-bottom:4px">
            🎯 Resume rank
          </h1>
          <p style="color:#888;font-size:15px">
            AI-powered resume evaluation · LangChain reasoning · Structured JSON output
          </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Error from previous run
    if st.session_state.get("error"):
        st.error(f"**Error:** {st.session_state['error']}")
        st.session_state["error"] = None

    with st.form("input_form", clear_on_submit=False):
        # Row 1: Company + Title
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input(
                "Company name",
                placeholder="e.g. Stripe, InComm Payments, Google…",
                help="Used to infer the industry domain and calibrate domain fit scoring.",
            )
        with c2:
            job_title = st.text_input(
                "Job title",
                placeholder="e.g. Staff Software Engineer, Data Scientist…",
            )

        # Job description
        jd = st.text_area(
            "Job description",
            placeholder=(
                "Paste the complete job description here — responsibilities, "
                "required skills, qualifications, preferred experience…"
            ),
            height=250,
            help="The more complete the JD, the more accurate the evaluation.",
        )

        # Resume upload
        uploaded = st.file_uploader(
            "Resume",
            type=["pdf", "docx", "doc"],
            help="Upload a PDF or DOCX resume. Text is extracted locally before analysis.",
        )

        st.markdown(
            '<p style="font-size:12px;color:#aaa;margin-top:-6px">Supported: PDF, DOCX, DOC</p>',
            unsafe_allow_html=True,
        )

        # API key (sidebar or form)
        api_key_env = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key_env:
            api_key = st.text_input(
                "Anthropic API key",
                type="password",
                placeholder="sk-ant-api03-…",
                help="Required if ANTHROPIC_API_KEY env var is not set.",
            )
        else:
            api_key = api_key_env
            st.markdown(
                '<p style="font-size:12px;color:#1D9E75">✅ API key loaded from environment</p>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "🚀 Analyze resume",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        _handle_submit(
            company=company,
            job_title=job_title,
            jd=jd,
            uploaded=uploaded,
            api_key=api_key,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SUBMISSION HANDLER
# ═════════════════════════════════════════════════════════════════════════════

def _handle_submit(
    company: str,
    job_title: str,
    jd: str,
    uploaded,
    api_key: str,
) -> None:
    # Validation
    errors = []
    if not jd.strip():
        errors.append("Job description is required.")
    if uploaded is None:
        errors.append("Please upload a resume file.")
    if not api_key:
        errors.append("Anthropic API key is required (set ANTHROPIC_API_KEY or enter above).")

    if errors:
        for e in errors:
            st.error(e)
        return

    # Set API key for this session
    os.environ["ANTHROPIC_API_KEY"] = api_key

    # Extract resume text
    try:
        with st.spinner("Extracting resume text…"):
            resume_text = extract_resume_text(uploaded)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return

    # Run pipeline with streaming status updates
    pipeline = ResumeRankingPipeline()
    domain_result   = None
    eval_result     = None

    status_placeholder = st.empty()
    progress_bar       = st.progress(0, text="Starting analysis…")

    step_map = {
        "Inferring company domain…":                        0.25,
        "Evaluating resume against job description…":       0.65,
        "Scoring dimensions and compiling report…":         0.90,
    }

    try:
        for event_type, payload in pipeline.stream_evaluation(
            resume_text=resume_text,
            job_description=jd,
            company_name=company,
            job_title=job_title,
        ):
            if event_type == "status":
                pct = step_map.get(payload, 0.5)
                status_placeholder.info(f"⏳ {payload}")
                progress_bar.progress(pct, text=payload)

            elif event_type == "domain":
                domain_result = payload

            elif event_type == "evaluation":
                eval_result = payload

        progress_bar.progress(1.0, text="Done!")
        time.sleep(0.3)
        status_placeholder.empty()
        progress_bar.empty()

    except Exception as exc:
        status_placeholder.empty()
        progress_bar.empty()
        logging.exception("Pipeline error")
        st.error(f"**Analysis failed:** {exc}")
        return

    # Store in session state
    st.session_state["domain"]     = domain_result
    st.session_state["evaluation"] = eval_result
    st.session_state["ran"]        = True
    st.session_state["company"]    = company
    st.session_state["job_title"]  = job_title

    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
**Resume Rank** uses a two-stage LangChain pipeline:

1. **Domain Inference Chain**
   - Input: Company name + job title
   - Output: Industry domain, expected skills, regulatory context

2. **Evaluation Chain**
   - Input: Resume text + JD + domain context
   - Applies a 6-dimension scoring rubric
   - Output: Structured `ResumeEvaluation` JSON

**Models**
- LLM: `claude-sonnet-4-5`
- Parsers: `PydanticOutputParser`
- Framework: LangChain LCEL
        """)

        st.markdown("---")
        st.markdown("### 📐 Scoring rubric")
        rubric = [
            ("Technical Skills",       "30%"),
            ("Domain Experience",      "25%"),
            ("Years of Experience",    "15%"),
            ("Leadership & Scope",     "15%"),
            ("Education & Certs",      "10%"),
            ("Role Fit (Inferred)",     "5%"),
        ]
        for dim, wt in rubric:
            st.markdown(f"- **{dim}** — {wt}")

        st.markdown("---")
        st.markdown("### 🎯 Score legend")
        for score_range, label, color in [
            ("9–10", "Strong match",  "#1D9E75"),
            ("7–8",  "Good fit",      "#378ADD"),
            ("5–6",  "Partial match", "#BA7517"),
            ("3–4",  "Weak match",    "#E24B4A"),
            ("1–2",  "Not a fit",     "#A32D2D"),
        ]:
            st.markdown(
                f'<span style="color:{color};font-weight:600">{score_range}</span> — {label}',
                unsafe_allow_html=True,
            )

        if st.session_state.get("ran"):
            st.markdown("---")
            if st.button("🔄 New analysis", use_container_width=True):
                _reset()
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()
    render_sidebar()

    if st.session_state["ran"] and st.session_state["evaluation"]:
        render_results(
            domain    = st.session_state["domain"],
            ev        = st.session_state["evaluation"],
            company   = st.session_state.get("company", ""),
            job_title = st.session_state.get("job_title", ""),
        )
    else:
        render_input_form()


if __name__ == "__main__":
    main()
