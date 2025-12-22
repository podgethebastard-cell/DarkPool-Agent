
import streamlit as st
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator

# --- 1. System Configuration & DPC Architecture ---
st.set_page_config(
    page_title="SOLO AI // ARCHITECT",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Resilience Layer: Import Safety ---
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    # Mock classes to prevent runtime crashes before user sees error
    class genai:
        class Client:
            def __init__(self, api_key): pass
    class types:
        class GenerateContentConfig:
            def __init__(self, **kwargs): pass

def _inject_dpc_css():
    st.markdown("""
        <style>
        /* Import Roboto Mono */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap');

        /* Root Variables */
        :root {
            --background-dark: #0e1117;
            --panel-dark: #161b22;
            --text-main: #e0e0e0;
            --neon-green: #00ff41;
            --neon-blue: #00d2ff;
            --neon-purple: #bc13fe;
            --neon-red: #ff0055;
            --glass-panel: rgba(22, 27, 34, 0.9);
        }

        /* Global Resets */
        .stApp {
            background-color: var(--background-dark);
            color: var(--text-main);
            font-family: 'Roboto Mono', monospace;
        }
        
        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            background-color: var(--panel-dark);
            border-left: 3px solid #30363d;
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.2s ease;
        }
        [data-testid="stChatMessage"]:hover {
            border-left-color: var(--neon-blue);
            box-shadow: 0 0 15px rgba(0, 210, 255, 0.1);
        }
        
        /* User Message Distinction */
        [data-testid="stChatMessage"][data-test-user="true"] {
            background-color: rgba(0, 210, 255, 0.05);
            border-left-color: var(--neon-blue);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
            font-family: 'Roboto Mono', monospace !important;
            color: var(--text-main) !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--panel-dark);
            border-right: 1px solid #30363d;
        }
        
        /* Button Architecture */
        div.stButton > button {
            background-color: var(--panel-dark);
            color: var(--neon-blue);
            border: 1px solid #30363d;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            border-color: var(--neon-blue);
            color: var(--neon-blue);
            box-shadow: 0 0 10px rgba(0, 210, 255, 0.2);
        }
        div.stButton > button[kind="primary"] {
            background-color: rgba(0, 255, 65, 0.1);
            color: var(--neon-green);
            border: 1px solid var(--neon-green);
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: var(--neon-green);
            color: var(--background-dark) !important;
            box-shadow: 0 0 20px var(--neon-green);
        }

        /* Input Fields */
        .stTextArea textarea, .stTextInput input, .stChatInput textarea {
            background-color: #0d1117 !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
            border-radius: 4px;
        }
        .stTextArea textarea:focus, .stTextInput input:focus, .stChatInput textarea:focus {
            border-color: var(--neon-purple) !important;
            box-shadow: 0 0 8px rgba(188, 19, 254, 0.3) !important;
        }

        /* Tab Architecture */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: var(--panel-dark);
            border: 1px solid transparent;
            border-radius: 4px 4px 0 0;
            color: #8b949e;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--background-dark);
            color: var(--neon-blue);
            border-top: 2px solid var(--neon-blue);
            border-left: 1px solid #30363d;
            border-right: 1px solid #30363d;
        }
        
        /* Status Badges */
        .badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: bold;
            border: 1px solid;
        }
        .badge-sovereign { color: #ff0055; border-color: #ff0055; background: rgba(255, 0, 85, 0.1); }
        .badge-strategic { color: #00d2ff; border-color: #00d2ff; background: rgba(0, 210, 255, 0.1); }
        .badge-execution { color: #00ff41; border-color: #00ff41; background: rgba(0, 255, 65, 0.1); }
        </style>
    """, unsafe_allow_html=True)

_inject_dpc_css()

# --- 3. Data Structures & Loading ---
@dataclass(frozen=True)
class Role:
    key: str
    id: int
    tier: int
    icon: str
    name: str
    title: str
    focus: str
    capabilities: List[str]

# FALLBACK DATA in case JSON is missing
DEFAULT_ROLES = [
    Role("a2", 2, 2, "💀", "Architect", "Systems Architect", "Structure & Strategy", ["System Design"]),
    Role("a1", 1, 1, "🛡️", "Sentinel", "Security Sentinel", "Risk & Threat", ["Veto Power"]),
    Role("a5", 5, 3, "🚀", "Growth", "Growth Operator", "Scale & GTM", ["Marketing"]),
    Role("a11", 11, 1, "⚖️", "Auditor", "Ethics Compass", "Compliance", ["Oversight"])
]

def load_roles() -> Tuple[str, str, List[Role]]:
    here = os.path.dirname(__file__)
    json_path = os.path.join(here, "roles.json")
    
    if not os.path.exists(json_path):
        # Graceful fallback instead of crash
        return "SYSTEM_META: Default Fallback", "PROTOCOLS: None", DEFAULT_ROLES

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        roles = [
            Role(
                key=r["key"],
                id=int(r["id"]),
                tier=int(r["tier"]),
                icon=r.get("icon", "🔹"),
                name=r["name"],
                title=r["title"],
                focus=r["focus"],
                capabilities=list(r.get("capabilities", [])),
            )
            for r in data["roles"]
        ]
        return data.get("global_meta", ""), data.get("override_protocols", ""), roles
    except Exception as e:
        return f"Error loading roles: {e}", "", DEFAULT_ROLES

GLOBAL_META, OVERRIDE_PROTOCOLS, ROLES = load_roles()
ROLE_BY_KEY = {r.key: r for r in ROLES}
TIER_NAMES = {1: "Sovereign", 2: "Strategic", 3: "Execution", 4: "Support"}

# --- 4. Logic Core ---
def init_state():
    defaults = {
        "selected_analyst_key": "a2", # Architect
        "selected_council": ["a1", "a2", "a5", "a11"],
        "chat_logs": {r.key: [] for r in ROLES},
        "council_log": [], 
        "last_decision": "",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model_name": "gemini-2.0-flash-exp"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def get_client():
    if not HAS_GENAI:
        return None
    if not st.session_state.api_key:
        return None
    return genai.Client(api_key=st.session_state.api_key)

def build_system_prompt(role: Role, context: str = "") -> str:
    caps = "\n".join([f"- {c}" for c in role.capabilities])
    return f"""
    IDENTITY: {role.title} ({role.name})
    FOCUS: {role.focus}
    CAPABILITIES:
    {caps}
    
    SYSTEM CONTEXT:
    {GLOBAL_META}
    
    PROTOCOLS:
    {OVERRIDE_PROTOCOLS}
    
    MISSION:
    You are an elite component of the Solo AI Neural Network. 
    Act strictly according to your role. Be concise, technical, and high-signal.
    {context}
    """

def stream_agent_response(client, role_key: str, user_prompt: str) -> Generator[str, None, None]:
    """Streams the agent's response chunk by chunk."""
    
    # --- SIMULATION MODE (If Lib missing) ---
    if not HAS_GENAI:
        time.sleep(0.5)
        yield "⚠️ **SYSTEM ALERT**: `google-genai` library not found.\n\n"
        yield "Please ensure `requirements.txt` contains `google-genai>=0.3.0` and redeploy."
        return

    if not client:
        yield "⚠️ **ACCESS DENIED**: Missing API Key in Sidebar Config."
        return

    role = ROLE_BY_KEY[role_key]
    sys_prompt = build_system_prompt(role)
    
    try:
        response = client.models.generate_content_stream(
            model=st.session_state.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=0.7,
                max_output_tokens=1000
            )
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"**[SYSTEM FAILURE]**: {str(e)}"

# --- 5. UI Layout ---

# Sidebar
with st.sidebar:
    st.markdown("## 💀 THE ARCHITECT")
    
    # Status Check
    if HAS_GENAI:
        st.caption("Status: **ONLINE** | `v2.4-dpc`")
    else:
        st.error("MISSING LIB: google-genai")
        st.caption("Running in Safe Mode")

    with st.expander("⚙️ System Config"):
        st.session_state.api_key = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
        st.session_state.model_name = st.selectbox("Model", ["gemini-2.0-flash-exp", "gemini-1.5-pro"], index=0)
        
    st.divider()
    
    # Vectorized Role Render
    for tier in sorted(TIER_NAMES.keys()):
        tier_roles = [r for r in ROLES if r.tier == tier]
        if not tier_roles: continue
            
        st.markdown(f"### {TIER_NAMES[tier]}")
        for r in tier_roles:
            kind = "primary" if st.session_state.selected_analyst_key == r.key else "secondary"
            if st.button(f"{r.icon} {r.name}", key=f"btn_{r.key}", use_container_width=True, type=kind):
                st.session_state.selected_analyst_key = r.key

# Main Interface
tab_names = ["⚔️ COUNCIL", "🧠 EXPERT", "📡 LOGS"]
tabs = st.tabs(tab_names)

# --- TAB 1: COUNCIL (Sequential Synthesis) ---
with tabs[0]:
    col_head, col_stat = st.columns([3, 1])
    with col_head:
        st.markdown("## NEURAL COUNCIL GRID")
        st.caption("Decentralized Execution Protocol // Multi-Agent Consensus")
    with col_stat:
        st.markdown(f"<div style='text-align: right; color: #00ff41; font-size: 24px; font-weight: bold;'>{len(st.session_state.selected_council)} ACTIVE</div>", unsafe_allow_html=True)

    # Grid Selection
    with st.expander("Configure Neural Grid", expanded=False):
        cols = st.columns(4)
        for i, r in enumerate(ROLES):
            col = cols[i % 4]
            with col:
                checked = r.key in st.session_state.selected_council
                if st.checkbox(f"{r.icon} {r.name}", value=checked, key=f"chk_{r.key}"):
                    if not checked: st.session_state.selected_council.append(r.key)
                else:
                    if checked: st.session_state.selected_council.remove(r.key)

    st.divider()

    # Council Feed
    feed_container = st.container(height=400)
    with feed_container:
        for entry in st.session_state.council_log:
             with st.chat_message(entry["role"], avatar=entry.get("avatar")):
                 st.markdown(entry["content"])

    # Input
    user_text = st.chat_input("Broadcast strategic parameters to Council...")
    
    if user_text:
        client = get_client()
        
        # 1. Log User Input
        st.session_state.council_log.append({"role": "user", "content": user_text, "avatar": "👤"})
        with feed_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_text)

        # 2. Execute Agents Sequentially
        agent_outputs = {}
        for k in st.session_state.selected_council:
            r = ROLE_BY_KEY[k]
            with feed_container:
                with st.chat_message("assistant", avatar=r.icon):
                    st.markdown(f"**{r.name}** is processing...")
                    response_text = st.write_stream(stream_agent_response(client, k, user_text))
                    agent_outputs[k] = response_text
                    
            st.session_state.council_log.append({"role": "assistant", "content": f"**{r.name}**: {response_text}", "avatar": r.icon})

        # 3. Synthesize
        if HAS_GENAI and client:
            st.markdown("---")
            with st.spinner("Synthesizing Decision Card..."):
                final_prompt = f"Summarize these reports into a decision card: {json.dumps(agent_outputs)}"
                try:
                    resp = client.models.generate_content(
                        model=st.session_state.model_name,
                        contents=final_prompt
                    )
                    st.session_state.last_decision = resp.text
                    st.rerun()
                except Exception as e:
                    st.error(f"Synthesis Error: {e}")
        else:
             st.warning("Synthesis disabled (Library or Key missing).")

    if st.session_state.last_decision:
        st.success("SYNTHESIS COMPLETE")
        st.markdown(st.session_state.last_decision)

# --- TAB 2: EXPERT (Deep Dive) ---
with tabs[1]:
    target = ROLE_BY_KEY[st.session_state.selected_analyst_key]
    
    # Header
    c1, c2 = st.columns([1, 10])
    with c1:
        st.markdown(f"<div style='font-size: 40px;'>{target.icon}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"## {target.name}")
        st.caption(f"**{target.title}** | {target.focus}")
        
    st.divider()

    # Chat Feed
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_logs[target.key]:
            st.info(f"Secure channel established with {target.name}. Awaiting input.")
        
        for role, content in st.session_state.chat_logs[target.key]:
            avatar = "👤" if role == "user" else target.icon
            with st.chat_message(role, avatar=avatar):
                st.markdown(content)

    # Input
    if msg := st.chat_input(f"Direct link to {target.name}..."):
        client = get_client()
        st.session_state.chat_logs[target.key].append(("user", msg))
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg)
            
            with st.chat_message("assistant", avatar=target.icon):
                full_response = st.write_stream(stream_agent_response(client, target.key, msg))
        
        st.session_state.chat_logs[target.key].append(("assistant", full_response))

# --- TAB 3: LOGS ---
with tabs[2]:
    st.markdown("## 📡 SYSTEM LOGS")
    st.json(st.session_state.chat_logs)
