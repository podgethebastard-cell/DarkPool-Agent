# app.py
# Personal AI Assistant Dashboard — GPT-5.2 Council (single-file)
#
# Requirements:
#   pip install streamlit requests pypdf
#
# Run:
#   streamlit run app.py
#
# Env (optional):
#   OPENAI_API_KEY="..."
#   OPENAI_BASE_URL="https://api.openai.com/v1"
#   OPENAI_MODEL="gpt-5.2"
#   OPENAI_API_MODE="auto"            # auto|responses|chat
#   OPENAI_REASONING_EFFORT="none"    # none|low|medium|high|xhigh
#   OPENAI_MAX_OUTPUT_TOKENS="2400"
#   OPENAI_TEMPERATURE="0.3"
#   OPENAI_TIMEOUT_S="60"

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from pypdf import PdfReader


APP_TITLE = "Personal AI Assistant Dashboard - GPT-5.2 Council"


# =========================
# Analyst Prompts (4)
# =========================

ANALYST_1_PROMPT = """You are an elite SMC Quant Pine Script Developer and TradingView Community Advocate. You possess deep knowledge of advanced mathematics, signal processing, and the strict TradingView House Rules/Moderation Guidelines.

Core Objective: Generate Pine Script code that is high-quality, mathematically complex, and fully compliant with open-source community standards.

CRITICAL OPERATIONAL CONSTRAINTS (Zero Tolerance):

NEVER OMIT ANYTHING: You must implement every single detail, parameter, and instruction provided by the user. Do not simplify, summarize, or skip any part of the user's request.

NEVER ASSUME ANYTHING: Do not fill in gaps with "standard" defaults unless explicitly instructed. If the user's logic is incomplete, you must strictly implement what is given or explicitly flag the missing variable, rather than guessing the user's intent.

COMPLEXITY & SYNTHESIS: To meet the "Originality" requirement of TradingView Moderators, you must avoid simple "mashups" (e.g., just stacking RSI and MACD). Instead, blend and merge mathematical concepts. Use novel formulas, normalization techniques, or adaptive logic to synthesize multiple inputs into a unique, non-trivial signal.

Community & Moderation Standards:

Open Source Only: Never output obfuscated or minified code. The logic must be transparent.

Credit & Attribution: If the logic adapts known concepts or other authors' work, you must generate a comment block acknowledging them.

Honesty: In the script description/comments, strictly avoid "hype" (e.g., "100% win rate", "No Repaint"). Focus on the mathematical mechanics.

Educational Value: Code must be heavily commented, explaining the why behind the logic, not just the how.

Technical Requirements:

Always use the //@version=5 (or latest) directive.

Use explicit_plot_zorder for visual clarity.

Use tooltip= in input functions to explain parameters to the user.

Output Format: Provide the full Pine Script code block first, followed by a brief "Moderator-Ready" description draft that explains the complex math used in the script. GPL-3.0 with © DarkPoolCrypto credit.

When user says "full script"- immediately provide
"""

ANALYST_2_PROMPT = """You are an elite Chief Strategy Officer (CSO) and Project Manager with a strong operational team. Your mission is to operationalize the user's Business Plan. The uploaded Business Plan is your single source of truth — always cite specific page headings or sections when giving recommendations. If information is missing, unclear, or contradictory, stop and ask for clarification — never assume or invent.

All responses must be formatted in clean, hierarchical Markdown optimized for Obsidian (H2/H3 headers, concise bullet points). Any new content intended for insertion into the plan should appear in fenced code blocks or clearly labeled sections.

You operate in seven selectable modes:

Critique Mode (The Skeptic) — Challenge assumptions, stress-test projections, and expose risks. Always provide a fix for each problem identified. Maintain a direct, analytical tone.

Build Mode (The Co-Founder) — Expand on existing ideas from the plan. Propose aligned features, partnerships, and revenue streams. Draft additions in the same tone and voice as the plan.

Execution Mode (The Operator) — Translate strategy into executable steps. Produce prioritized backlogs, RACI matrices, OKRs, and checklists. Emphasize low-effort, high-impact actions.

Investor Readiness Mode — Prepare the business for investor or board engagement. Generate concise investor briefs, validate financials, and produce due diligence checklists. Communicate with confidence, precision, and data-backed logic.

Ops Dashboard Mode — Deliver operational visibility and progress tracking. Create KPI/OKR dashboards, progress summaries, and department-level snapshots. Focus on accountability and clarity.

Strategic Sprint Mode — Break plans into 2-week execution cycles with measurable deliverables. Define sprint goals, task sequencing, dependencies, and success metrics. Maintain a tactical and agile tone.

Scaling & Systems Mode — Optimize for growth and repeatability. Develop SOPs, automation maps, and org structure recommendations. Emphasize sustainable systems that scale.

Tone: professional, concise, honest, and action-oriented. Avoid corporate fluff, filler, and jargon. Always drive progress grounded in the source plan.
"""

ANALYST_3_PROMPT = """This GPT acts as a visionary Python Systems Architect and Creative Technologist. It behaves like an inspired designer — expressive, imaginative, and focused on bringing creativity to code. It specializes in crafting interactive and visually engaging Python applications, focusing on creativity, user experience, and design innovation.

When presented with a concept, it identifies the core functionality and the creative hook, then selects the most fitting Python stack—such as Streamlit, PyGame, Kivy, Gradio, or FastUI—to bring the idea to life. It writes elegant, modular, and fully functional MVP code with an emphasis on interactivity, smooth flow, and delightful UX details. It proposes one standout 'X-Factor' feature per project to enhance originality and engagement.

It communicates with clarity and inspiration, avoiding overly technical jargon and explaining its reasoning in accessible, creative terms. It offers brief rationales for design and library choices, and ensures instructions for setup are simple and user-friendly. If details are missing, it uses its design sensibility to infer suitable creative directions.
"""

ANALYST_4_PROMPT = """<System_Prompt>

  <Thinking_Sandbox>
    You reason step-by-step internally before responding.
    You prioritize business viability, speed to revenue, and solo-founder constraints.
    You evaluate ideas using pain severity, willingness to pay, automation feasibility, and legal risk.
    You avoid novelty bias and default to boring, proven patterns.
  </Thinking_Sandbox>

  <Persona>
    You are a Solo AI Business Architect.
    You specialize in helping individuals build profitable one-person AI businesses where AI performs the work and the user makes decisions.
    Your expertise spans micro-SaaS, AI agents, automations, internal tools, dashboards, research engines, and reusable templates.
    You are skeptical of labor-heavy services, custom consulting, marketplaces, and complex multi-tenant SaaS.
    You optimize for leverage, clarity, and early cash flow.
  </Persona>

  <Action_Steps>
    1. Analyze:
       - Identify the user's target market or idea.
       - Evaluate whether the problem is a painkiller (urgent, measurable, budgeted).
       - Reject ideas that require high-touch human labor or unclear buyers.

    2. Decide:
       - Select or refine a single narrow niche with paying customers.
       - Define one core AI engine with:
         • Clear inputs (what the user provides)
         • Clear outputs (what is delivered)
         • Clear actions (what the AI does autonomously)

    3. Design:
       - Propose a simple productized solution (not a service).
       - Recommend a lean architecture (APIs, agents, automations).
       - Specify prompts, workflows, and data sources.
       - Include guardrails, disclaimers, and risk-reduction steps.

    4. Monetize:
       - Recommend simple pricing (flat fee, usage-based, or tiered).
       - Push for early monetization (pre-sell, waitlist with payment, paid pilot).
       - Avoid freemium unless clearly justified.

    5. Launch:
       - Outline a fast launch plan (≤30 days).
       - Recommend distribution channels aligned with the niche.
       - Define success metrics and kill criteria.

    6. Self-Critique:
       - Identify complexity, legal risk, or hidden labor.
       - Simplify the product further if possible.

    7. Refine:
       - Output the final recommendation with maximum clarity and minimum fluff.
  </Action_Steps>

  <Constraints>
    - Favor automation over human effort in all repeatable tasks.
    - Avoid custom work, open-ended consulting, or “done-for-you” services.
    - Prefer single-purpose tools over platforms.
    - Keep the tech stack lean and understandable by a solo founder.
    - Minimize legal and compliance risk; include clear disclaimers where needed.
    - Ask clarifying questions only if they unblock a concrete next step.
    - Bias toward decisive recommendations over brainstorming.
    - Use plain language; avoid jargon unless it improves precision.
  </Constraints>

  <Output_Standard>
    Default format: Structured Markdown with clear sections.
    When appropriate, include:
      - A one-sentence business definition
      - Target customer and pain
      - Core AI engine description
      - Architecture diagram (described in text)
      - Example prompts or agent logic
      - Pricing model
      - 30-day launch plan
      - Key risks and simplifications
    Do not include internal reasoning or analysis.
  </Output_Standard>

</System_Prompt>
"""


# =========================
# Data Models / State
# =========================

@dataclass(frozen=True)
class Analyst:
    key: str
    name: str
    system_prompt: str


@dataclass
class PlanDoc:
    filename: str
    text: str
    headings: List[str]


@dataclass
class AppState:
    plan: Optional[PlanDoc] = None
    last_user_prompt: str = ""
    last_council: Dict[str, str] = field(default_factory=dict)
    last_decision_card: str = ""

    # chat logs keyed by analyst key; message is (role, content)
    chat_logs: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    # per-analyst memory notes (session only)
    memory_notes: Dict[str, str] = field(default_factory=dict)

    # CSO mode selection
    cso_mode: str = "Critique Mode (The Skeptic)"


def get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState(
            chat_logs={"a1": [], "a2": [], "a3": [], "a4": []},
            memory_notes={"a1": "", "a2": "", "a3": "", "a4": ""},
        )
    return st.session_state["app_state"]


# =========================
# Plan Parsing (PDF/MD/TXT)
# =========================

_HEADING_PATTERNS = [
    r"(?m)^(#{1,6})\s+(.+?)\s*$",                    # markdown headings
    r"(?m)^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})\s*$",  # numbered headings
    r"(?m)^\s*([A-Z][A-Z0-9 \-/]{4,60})\s*$",        # ALL CAPS headings
]


def extract_headings(text: str) -> List[str]:
    headings: List[str] = []
    for pat in _HEADING_PATTERNS:
        for m in re.finditer(pat, text):
            groups = [g.strip() for g in m.groups() if g and g.strip()]
            if not groups:
                continue
            h = groups[-1] if len(groups) == 1 else " ".join(groups[-2:])
            h = re.sub(r"\s+", " ", h).strip()
            if h and h not in headings:
                headings.append(h)
    return headings


def read_pdf(upload) -> str:
    reader = PdfReader(upload)
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def parse_plan_upload(upload) -> PlanDoc:
    name = getattr(upload, "name", "uploaded_plan")
    suffix = name.lower().split(".")[-1] if "." in name else ""

    if suffix == "pdf":
        text = read_pdf(upload)
    else:
        raw = upload.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("utf-8", errors="replace")
        text = text.strip()

    headings = extract_headings(text) if text else []
    return PlanDoc(filename=name, text=text, headings=headings)


# =========================
# OpenAI-Compatible Client (GPT-5.2 optimized)
# =========================

def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.3
    max_output_tokens: int = 2400
    reasoning_effort: str = "none"   # none|low|medium|high|xhigh
    api_mode: str = "auto"           # auto|responses|chat
    timeout_s: int = 60


def config_from_env() -> LLMConfig:
    return LLMConfig(
        api_key=env("OPENAI_API_KEY", ""),
        base_url=env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=env("OPENAI_MODEL", "gpt-5.2"),
        temperature=float(env("OPENAI_TEMPERATURE", "0.3")),
        max_output_tokens=int(env("OPENAI_MAX_OUTPUT_TOKENS", env("OPENAI_MAX_TOKENS", "2400"))),
        reasoning_effort=env("OPENAI_REASONING_EFFORT", "none"),
        api_mode=env("OPENAI_API_MODE", "auto"),
        timeout_s=int(env("OPENAI_TIMEOUT_S", "60")),
    )


class LLMError(RuntimeError):
    pass


class LLMClient:
    def headers(self, cfg: LLMConfig) -> Dict[str, str]:
        if not cfg.api_key:
            raise LLMError("Missing API key. Set OPENAI_API_KEY.")
        return {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}

    def should_use_responses(self, cfg: LLMConfig) -> bool:
        mode = (cfg.api_mode or "auto").lower().strip()
        if mode == "responses":
            return True
        if mode == "chat":
            return False
        m = (cfg.model or "").lower()
        return m.startswith("gpt-5") or m.startswith("o")

    def complete(self, packet: Dict[str, Any], *, cfg: LLMConfig) -> str:
        return self.complete_responses(packet, cfg=cfg) if self.should_use_responses(cfg) else self.complete_chat(packet, cfg=cfg)

    # ---- Responses API ----
    def complete_responses(self, packet: Dict[str, Any], *, cfg: LLMConfig) -> str:
        url = cfg.base_url.rstrip("/") + "/responses"

        messages = packet.get("messages", [])
        # Simple & robust: flatten messages into a single input string
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role.upper()}]\n{content}\n")
        input_text = "\n".join(parts).strip()

        body: Dict[str, Any] = {
            "model": cfg.model,
            "input": input_text,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
        }
        if cfg.reasoning_effort:
            body["reasoning"] = {"effort": cfg.reasoning_effort}

        try:
            r = requests.post(url, headers=self.headers(cfg), json=body, timeout=cfg.timeout_s)
        except requests.RequestException as e:
            raise LLMError(f"Network error calling Responses API: {e}") from e

        if r.status_code >= 400:
            raise LLMError(f"Responses API error {r.status_code}: {r.text}")

        data = r.json()
        return self.extract_responses_text(data).strip()

    def extract_responses_text(self, data: Dict[str, Any]) -> str:
        if isinstance(data, dict):
            if isinstance(data.get("output_text"), str):
                return data["output_text"]

            output = data.get("output")
            if isinstance(output, list):
                out_parts = []
                for item in output:
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                if isinstance(c.get("text"), str):
                                    out_parts.append(c["text"])
                                elif c.get("type") == "output_text" and isinstance(c.get("text"), str):
                                    out_parts.append(c["text"])
                    if isinstance(item.get("text"), str):
                        out_parts.append(item["text"])
                if out_parts:
                    return "\n".join(out_parts)

        return json.dumps(data, ensure_ascii=False, indent=2)

    # ---- Chat Completions API ----
    def complete_chat(self, packet: Dict[str, Any], *, cfg: LLMConfig) -> str:
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": cfg.model,
            "messages": packet.get("messages", []),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_output_tokens,
        }

        try:
            r = requests.post(url, headers=self.headers(cfg), json=body, timeout=cfg.timeout_s)
        except requests.RequestException as e:
            raise LLMError(f"Network error calling Chat Completions API: {e}") from e

        if r.status_code >= 400:
            raise LLMError(f"Chat Completions API error {r.status_code}: {r.text}")

        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)


# =========================
# Prompt Assembly
# =========================

def build_single_packet(
    *,
    analyst_name: str,
    system_prompt: str,
    user_prompt: str,
    memory_notes: str = "",
    extra_mode: Optional[str] = None,
    chat_history: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    chat_history = chat_history or []

    memory_block = f"\n\n[Memory Notes]\n{memory_notes}\n" if memory_notes.strip() else ""
    mode_block = f"\n\n[Selected Mode]\n{extra_mode}\n" if extra_mode else ""

    sys = f"[Analyst]\n{analyst_name}\n\n[System Prompt]\n{system_prompt}{memory_block}{mode_block}".strip()
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]

    # Keep last 16 messages to avoid token blow-up
    for role, content in (chat_history[-16:] if chat_history else []):
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_prompt})
    return {"messages": messages}


def build_council_synthesis_packet(user_prompt: str, responses: Dict[str, str], analysts: Dict[str, Analyst]) -> Dict[str, Any]:
    blocks = []
    for key, txt in responses.items():
        blocks.append(f"## {analysts[key].name}\n{txt}")

    synthesis = (
        "You are the Council Chair. Produce a compact Decision Card.\n"
        "Rules:\n"
        "- Be specific and actionable.\n"
        "- Highlight consensus vs disagreement.\n"
        "- Provide a single best next step.\n"
        "- Keep it under ~250 lines.\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Analyst outputs:\n\n" + "\n\n".join(blocks)
    )

    return {
        "messages": [
            {"role": "system", "content": "You synthesize multiple expert outputs into a Decision Card."},
            {"role": "user", "content": synthesis},
        ]
    }


def render_markdown_export(
    *,
    user_prompt: str,
    decision_card: str,
    responses: Dict[str, str],
    analysts: Dict[str, Analyst],
) -> str:
    parts: List[str] = []
    parts.append("# Council Run Export\n")
    parts.append("## User Request\n")
    parts.append(user_prompt.strip() + "\n")

    if decision_card.strip():
        parts.append("## Council Decision Card\n")
        parts.append(decision_card.strip() + "\n")

    parts.append("## Analyst Responses\n")
    for key, txt in responses.items():
        parts.append(f"### {analysts[key].name}\n")
        parts.append(txt.strip() + "\n")

    return "\n".join(parts).strip() + "\n"


# =========================
# Analyst Registry
# =========================

def build_analysts_registry() -> Dict[str, Analyst]:
    # Names can be tweaked here or via the UI (simple override in sidebar)
    return {
        "a1": Analyst(key="a1", name="SMC Quant PineScript Advocate", system_prompt=ANALYST_1_PROMPT),
        "a2": Analyst(key="a2", name="CSO / Project Manager (7-Mode)", system_prompt=ANALYST_2_PROMPT),
        "a3": Analyst(key="a3", name="Python Systems Architect (Creative)", system_prompt=ANALYST_3_PROMPT),
        "a4": Analyst(key="a4", name="Solo AI Business Architect", system_prompt=ANALYST_4_PROMPT),
    }


def get_analyst_system_prompt(
    analyst: Analyst,
    *,
    plan_text: str = "",
    plan_headings: Optional[List[str]] = None,
) -> str:
    if analyst.key != "a2":
        return analyst.system_prompt

    plan_headings = plan_headings or []
    plan_block = ""
    if plan_text.strip():
        headings_preview = "\n".join([f"- {h}" for h in plan_headings[:50]]) if plan_headings else "(No headings detected)"
        plan_block = (
            "\n\n"
            "=== BUSINESS PLAN (SOURCE OF TRUTH) ===\n"
            "Detected headings (cite these when referencing the plan):\n"
            f"{headings_preview}\n\n"
            "Plan text:\n"
            f"{plan_text}\n"
            "=== END PLAN ===\n"
        )

    return analyst.system_prompt + plan_block


# =========================
# UI Helpers
# =========================

def init_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="AI", layout="wide")
    st.title(APP_TITLE)
    st.caption("Four-analyst council dashboard. GPT-5.2 optimized (Responses API by default for gpt-5* / o* models).")


def sidebar_settings(cfg: LLMConfig) -> LLMConfig:
    st.sidebar.header("Settings")

    api_key = st.sidebar.text_input("OPENAI_API_KEY", value=cfg.api_key, type="password")
    base_url = st.sidebar.text_input("OPENAI_BASE_URL", value=cfg.base_url)
    model = st.sidebar.text_input("Model", value=cfg.model)

    api_mode = st.sidebar.selectbox(
        "API mode",
        options=["auto", "responses", "chat"],
        index=["auto", "responses", "chat"].index(cfg.api_mode) if cfg.api_mode in ["auto", "responses", "chat"] else 0,
        help="auto = Responses for gpt-5* and o* models, else Chat Completions. You can force either.",
    )

    reasoning_effort = st.sidebar.selectbox(
        "Reasoning effort (Responses)",
        options=["none", "low", "medium", "high", "xhigh"],
        index=["none", "low", "medium", "high", "xhigh"].index(cfg.reasoning_effort)
        if cfg.reasoning_effort in ["none", "low", "medium", "high", "xhigh"]
        else 0,
        help="Use none for low latency; higher for complex synthesis.",
    )

    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, float(cfg.temperature), 0.05)
    max_output_tokens = st.sidebar.number_input("Max output tokens", 128, 20000, int(cfg.max_output_tokens), 128)
    timeout_s = st.sidebar.number_input("Timeout (seconds)", 5, 180, int(cfg.timeout_s), 5)

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        reasoning_effort=reasoning_effort,
        api_mode=api_mode,
        timeout_s=int(timeout_s),
    )


def sidebar_plan_vault(state: AppState) -> None:
    st.sidebar.header("Plan Vault (Analyst 2)")
    st.sidebar.caption("Upload a Business Plan (PDF/MD/TXT). Analyst 2 must cite plan headings/sections.")

    upload = st.sidebar.file_uploader("Upload business plan", type=["pdf", "md", "txt"], accept_multiple_files=False)
    if upload is not None:
        plan = parse_plan_upload(upload)
        state.plan = plan
        st.sidebar.success(f"Loaded: {plan.filename} - {len(plan.text)} chars")

    if state.plan is not None:
        with st.sidebar.expander("Plan snapshot", expanded=False):
            st.write({"filename": state.plan.filename, "headings": state.plan.headings[:15]})


def sidebar_memory_notes(state: AppState, analysts: Dict[str, Analyst]) -> None:
    st.sidebar.header("Analyst Memory Notes")
    who = st.sidebar.selectbox("Select analyst", options=list(analysts.keys()), format_func=lambda k: analysts[k].name)
    note = st.sidebar.text_area("Memory notes (session only)", value=state.memory_notes.get(who, ""), height=140)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Save note"):
            state.memory_notes[who] = note
            st.sidebar.success("Saved.")
    with c2:
        if st.button("Clear note"):
            state.memory_notes[who] = ""
            st.sidebar.warning("Cleared.")


# =========================
# Tabs
# =========================

def render_council_tab(llm: LLMClient, cfg: LLMConfig, state: AppState, analysts: Dict[str, Analyst]) -> None:
    st.subheader("Council Mode")

    cols = st.columns([2, 1])
    with cols[0]:
        user_prompt = st.text_area(
            "Your request",
            value=state.last_user_prompt or "",
            height=160,
            placeholder="Describe what you want the council to do...",
        )
    with cols[1]:
        selected = st.multiselect(
            "Analysts",
            options=list(analysts.keys()),
            default=list(analysts.keys()),
            format_func=lambda k: analysts[k].name,
        )
        include_decision_card = st.checkbox("Generate Council Decision Card", value=True)

    run = st.button("Run Council", type="primary", use_container_width=True, disabled=not user_prompt.strip())
    if run:
        state.last_user_prompt = user_prompt
        responses: Dict[str, str] = {}

        plan_text = state.plan.text if state.plan else ""
        plan_headings = state.plan.headings if state.plan else []

        with st.spinner("Running council..."):
            for key in selected:
                analyst = analysts[key]
                sys_prompt = get_analyst_system_prompt(analyst, plan_text=plan_text, plan_headings=plan_headings)
                memory = state.memory_notes.get(key, "")

                packet = build_single_packet(
                    analyst_name=analyst.name,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    memory_notes=memory,
                    extra_mode=state.cso_mode if analyst.key == "a2" else None,
                )

                try:
                    out = llm.complete(packet, cfg=cfg)
                except Exception as e:
                    out = f"LLM error: {e}"
                responses[key] = out

                state.chat_logs[key].append(("user", user_prompt))
                state.chat_logs[key].append(("assistant", out))

        state.last_council = responses

        if include_decision_card:
            with st.spinner("Synthesizing decision card..."):
                synth_packet = build_council_synthesis_packet(user_prompt, responses, analysts)
                try:
                    state.last_decision_card = llm.complete(synth_packet, cfg=cfg)
                except Exception as e:
                    state.last_decision_card = f"LLM error: {e}"
        else:
            state.last_decision_card = ""

    if state.last_council:
        st.divider()

        if state.last_decision_card.strip():
            st.markdown("### Council Decision Card")
            st.markdown(state.last_decision_card)

        st.markdown("### Analyst Responses")
        for key, txt in state.last_council.items():
            with st.expander(analysts[key].name, expanded=True):
                st.markdown(txt)

        st.divider()
        st.markdown("### Export")
        export_md = render_markdown_export(
            user_prompt=state.last_user_prompt or "",
            decision_card=state.last_decision_card or "",
            responses=state.last_council,
            analysts=analysts,
        )
        st.download_button(
            "Download Markdown",
            data=export_md.encode("utf-8"),
            file_name="council_run.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.code(export_md, language="markdown")


def render_single_tab(llm: LLMClient, cfg: LLMConfig, state: AppState, analysts: Dict[str, Analyst]) -> None:
    st.subheader("Single Analyst Chat")

    left, right = st.columns([1, 2])

    with left:
        who = st.selectbox("Analyst", options=list(analysts.keys()), format_func=lambda k: analysts[k].name)
        analyst = analysts[who]

        if analyst.key == "a2":
            st.markdown("CSO Mode")
            state.cso_mode = st.selectbox(
                "Select mode",
                options=[
                    "Critique Mode (The Skeptic)",
                    "Build Mode (The Co-Founder)",
                    "Execution Mode (The Operator)",
                    "Investor Readiness Mode",
                    "Ops Dashboard Mode",
                    "Strategic Sprint Mode",
                    "Scaling & Systems Mode",
                ],
                index=0,
            )
            st.caption("Analyst 2 uses the uploaded plan as source of truth and must cite headings.")

        st.divider()
        if st.button("Clear this chat"):
            state.chat_logs[who] = []
            st.success("Cleared.")

        if st.button("Clear all chats"):
            for k in state.chat_logs:
                state.chat_logs[k] = []
            st.success("Cleared all chats.")

    with right:
        chat = state.chat_logs[who]
        if not chat:
            st.info("Start chatting. The analyst keeps context within this session.")
        else:
            for role, msg in chat:
                st.chat_message("user" if role == "user" else "assistant").markdown(msg)

        user_msg = st.chat_input(f"Message {analyst.name}...")
        if user_msg:
            plan_text = state.plan.text if state.plan else ""
            plan_headings = state.plan.headings if state.plan else []

            sys_prompt = get_analyst_system_prompt(analyst, plan_text=plan_text, plan_headings=plan_headings)
            memory = state.memory_notes.get(who, "")

            packet = build_single_packet(
                analyst_name=analyst.name,
                system_prompt=sys_prompt,
                user_prompt=user_msg,
                memory_notes=memory,
                extra_mode=state.cso_mode if analyst.key == "a2" else None,
                chat_history=chat,
            )

            state.chat_logs[who].append(("user", user_msg))
            with st.spinner("Thinking..."):
                try:
                    out = llm.complete(packet, cfg=cfg)
                except Exception as e:
                    out = f"LLM error: {e}"
            state.chat_logs[who].append(("assistant", out))
            st.rerun()


def render_about_tab(cfg: LLMConfig) -> None:
    st.subheader("About and Debug")
    st.markdown(
        f"- Model: `{cfg.model}`\n"
        f"- API mode: `{cfg.api_mode}`\n"
        f"- Reasoning effort: `{cfg.reasoning_effort}`\n"
        f"- Max output tokens: `{cfg.max_output_tokens}`\n"
        f"- Base URL: `{cfg.base_url}`"
    )
    with st.expander("Current config (raw)", expanded=False):
        st.json(asdict(cfg))


def main() -> None:
    init_page()

    analysts = build_analysts_registry()
    state = get_state()

    cfg = config_from_env()
    cfg = sidebar_settings(cfg)

    if not cfg.api_key:
        st.warning("Set OPENAI_API_KEY in the sidebar or environment to enable completions.")

    sidebar_plan_vault(state)
    sidebar_memory_notes(state, analysts)

    llm = LLMClient()

    tabs = st.tabs(["Council", "Single Analyst", "About"])
    with tabs[0]:
        render_council_tab(llm, cfg, state, analysts)
    with tabs[1]:
        render_single_tab(llm, cfg, state, analysts)
    with tabs[2]:
        render_about_tab(cfg)


if __name__ == "__main__":
    main()
