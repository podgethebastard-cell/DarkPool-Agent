
import streamlit as st
import json
import os
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator

# --- 1. System Configuration & DPC Architecture ---
st.set_page_config(
    page_title="SOLO AI // ARCHITECT",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TAILWIND COLOR ENGINE ---
# Maps roles.json classes to specific Hex values for UI branding
TAILWIND_MAP = {
    "bg-rose-500": "#f43f5e",
    "bg-blue-500": "#3b82f6",
    "bg-amber-700": "#b45309",
    "bg-amber-400": "#fbbf24",
    "bg-amber-800": "#92400e",
    "bg-emerald-500": "#10b981",
    "bg-purple-600": "#9333ea",
    "bg-cyan-500": "#06b6d4",
    "bg-slate-600": "#475569",
    "bg-red-600": "#dc2626",
    "bg-indigo-500": "#6366f1",
    "bg-violet-600": "#7c3aed",
    "bg-fuchsia-600": "#c026d3",
    "bg-green-600": "#16a34a",
    # Fallback
    "default": "#3b82f6"
}

def _inject_dpc_css():
    st.markdown("""
        <style>
        /* Import JetBrains Mono */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');

        :root {
            --bg-dark: #0e1117;
            --panel-bg: #131720;
            --border-color: #262730;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --accent-glow: 0 0 10px rgba(59, 130, 246, 0.5);
        }

        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--text-primary);
        }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background-color: var(--panel-bg);
            border-right: 1px solid var(--border-color);
        }

        /* Custom Buttons */
        div.stButton > button {
            background-color: #1f2937;
            border: 1px solid #374151;
            color: #e5e7eb;
            transition: all 0.2s ease;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            font-weight: 700;
        }
        div.stButton > button:hover {
            border-color: #3b82f6;
            color: #3b82f6;
            box-shadow: 0 0 8px rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }

        /* Primary Action Buttons */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(45deg, #1e3a8a, #1d4ed8);
            border: 1px solid #3b82f6;
            color: white;
        }
        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.6);
        }

        /* Input Fields */
        .stTextInput input, .stTextArea textarea {
            background-color: #0d1117 !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }

        /* Chat Message Cards */
        .chat-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            position: relative;
        }
        .chat-card.user {
            border-left: 3px solid #3b82f6;
            background-color: rgba(59, 130, 246, 0.05);
        }
        
        /* Metric Containers */
        [data-testid="stMetric"] {
            background-color: #161b22;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }
        [data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }
        [data-testid="stMetricValue"] { color: #00ff41; font-size: 1.5rem; text-shadow: 0 0 5px rgba(0, 255, 65, 0.3); }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #161b22;
            border: 1px solid transparent;
            color: #8b949e;
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0d1117;
            color: #3b82f6;
            border-top: 2px solid #3b82f6;
        }
        </style>
    """, unsafe_allow_html=True)

_inject_dpc_css()

# --- 2. Resilience Layer & Data Loading ---
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

@dataclass(frozen=True)
class Role:
    key: str
    id: int
    tier: int
    icon: str
    name: str
    title: str
    focus: str
    color_class: str
    capabilities: List[str]
    
    @property
    def hex_color(self):
        return TAILWIND_MAP.get(self.color_class, TAILWIND_MAP["default"])

# Mock Data for Fallback
DEFAULT_ROLES = [
    Role("a2", 2, 2, "💀", "Architect", "Systems Architect", "Structure", "bg-blue-500", ["System Design"]),
    Role("a1", 1, 1, "🛡️", "Sentinel", "Security Sentinel", "Risk", "bg-red-600", ["Veto Power"]),
]

def load_roles() -> Tuple[str, str, List[Role]]:
    here = os.path.dirname(__file__)
    json_path = os.path.join(here, "roles.json")
    
    if not os.path.exists(json_path):
        return "SYSTEM_META: Default", "PROTOCOLS: None", DEFAULT_ROLES

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
                color_class=r.get("color", "bg-slate-600"),
                capabilities=list(r.get("capabilities", [])),
            )
            for r in data["roles"]
        ]
        return data.get("global_meta", ""), data.get("override_protocols", ""), roles
    except Exception:
        return "", "", DEFAULT_ROLES

GLOBAL_META, OVERRIDE_PROTOCOLS, ROLES = load_roles()
ROLE_BY_KEY = {r.key: r for r in ROLES}
TIER_NAMES = {1: "Sovereign", 2: "Strategic", 3: "Execution", 4: "Support"}

# --- 3. Logic Core ---
def init_state():
    defaults = {
        "selected_analyst_key": "a2",
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
    if not HAS_GENAI or not st.session_state.api_key: return None
    return genai.Client(api_key=st.session_state.api_key)

def build_system_prompt(role: Role, context: str = "") -> str:
    caps = "\n".join([f"- {c}" for c in role.capabilities])
    return f"""
    IDENTITY: {role.title} ({role.name})
    FOCUS: {role.focus}
    CAPABILITIES:
    {caps}
    
    SYSTEM CONTEXT: {GLOBAL_META}
    PROTOCOLS: {OVERRIDE_PROTOCOLS}
    
    MISSION: Act strictly according to your role. Be concise, technical, and high-signal.
    {context}
    """

# --- 4. Render Components ---

def render_hud():
    """Renders the top status bar."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("SYSTEM STATUS", "ONLINE", delta="Stable", delta_color="normal")
    with c2:
        st.metric("ACTIVE NODES", f"{len(st.session_state.selected_council)} / {len(ROLES)}")
    with c3:
        st.metric("MODEL", st.session_state.model_name.replace("gemini-", "").upper())
    with c4:
        # Simulate latency variance
        lat = f"{random.randint(12, 45)}ms"
        st.metric("LATENCY", lat, "-2ms")
    st.markdown("---")

def render_chat_message(role_key: str, content: str, is_user: bool = False):
    """Custom HTML render for chat messages to ensure correct coloring."""
    if is_user:
        border_color = "#3b82f6"
        bg_color = "rgba(59, 130, 246, 0.05)"
        icon = "👤"
        name = "COMMAND"
    else:
        role = ROLE_BY_KEY[role_key]
        border_color = role.hex_color
        # Lower opacity hex
        bg_color = f"{border_color}10" 
        icon = role.icon
        name = role.name

    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #e0e0e0;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 8px; color: {border_color}; font-weight: bold;">
            <span style="font-size: 1.2rem; margin-right: 10px;">{icon}</span>
            <span>{name}</span>
        </div>
        <div style="font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. Main Layout ---

# Sidebar
with st.sidebar:
    st.markdown("## 💀 ARCHITECT v2.5")
    
    if not HAS_GENAI:
        st.error("⚠️ LIBRARY MISSING: google-genai")
    
    with st.expander("⚙️ NETWORK CONFIG", expanded=False):
        st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
        st.session_state.model_name = st.selectbox("Model Family", ["gemini-2.0-flash-exp", "gemini-1.5-pro"])
    
    st.divider()
    st.markdown("### 🧬 AGENT REGISTRY")
    
    # Visual Agent Selector
    for tier in sorted(TIER_NAMES.keys()):
        tier_roles = [r for r in ROLES if r.tier == tier]
        if not tier_roles: continue
            
        st.caption(f"// {TIER_NAMES[tier].upper()}")
        for r in tier_roles:
            # Check if active
            is_active = r.key == st.session_state.selected_analyst_key
            label = f"{r.icon} {r.name}"
            
            # Dynamic styling in button label (Streamlit limitation hack)
            if st.button(label, key=f"sel_{r.key}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.selected_analyst_key = r.key

# Main Area
render_hud()

tab_names = ["⚔️ WAR ROOM", "🧠 DIRECT LINK", "📡 DATA STREAMS"]
tabs = st.tabs(tab_names)

# --- TAB 1: WAR ROOM (Council) ---
with tabs[0]:
    c_left, c_right = st.columns([3, 1])
    
    with c_right:
        st.markdown("##### 🕵️‍♂️ COUNCIL GRID")
        # Compact grid selection
        for r in ROLES:
            if r.tier > 3: continue # Skip support roles for council to save space
            checked = r.key in st.session_state.selected_council
            if st.checkbox(f"{r.icon} {r.name}", value=checked, key=f"counc_{r.key}"):
                 if not checked: st.session_state.selected_council.append(r.key)
            else:
                 if checked: st.session_state.selected_council.remove(r.key)
    
    with c_left:
        # Chat Feed
        feed_container = st.container(height=500)
        with feed_container:
            for entry in st.session_state.council_log:
                render_chat_message(entry.get("role_key", "user"), entry["content"], is_user=(entry["role"]=="user"))

        # Input Area
        prompt = st.chat_input("Broadcast strategic directives...")
        
        if prompt:
            client = get_client()
            # 1. Render User Input
            st.session_state.council_log.append({"role": "user", "content": prompt})
            with feed_container:
                render_chat_message("user", prompt, is_user=True)
            
            # 2. Sequential Execution
            agent_outputs = {}
            active_keys = st.session_state.selected_council
            
            if not HAS_GENAI or not client:
                st.error("SYSTEM OFFLINE: Config Required")
            else:
                for k in active_keys:
                    r = ROLE_BY_KEY[k]
                    with feed_container:
                        # Visual Status container
                        with st.status(f"{r.name} Processing...", expanded=True) as status:
                            st.write("Analyzing vector...")
                            try:
                                sys_prompt = build_system_prompt(r)
                                resp = client.models.generate_content(
                                    model=st.session_state.model_name,
                                    contents=prompt,
                                    config=types.GenerateContentConfig(system_instruction=sys_prompt)
                                )
                                text = resp.text
                                status.update(label=f"{r.name} Complete", state="complete", expanded=False)
                            except Exception as e:
                                text = f"Error: {e}"
                                status.update(label="Failed", state="error")
                        
                        # Render final card
                        render_chat_message(k, text)
                    
                    st.session_state.council_log.append({"role": "assistant", "role_key": k, "content": text})
                    agent_outputs[k] = text

                # 3. Synthesis
                if len(active_keys) > 1:
                    with feed_container:
                        with st.status("⚡ SYNTHESIZING CONSENSUS", expanded=True) as status:
                            final_prompt = f"Synthesize these reports into a Decision Card: {json.dumps(agent_outputs)}"
                            resp = client.models.generate_content(
                                model=st.session_state.model_name,
                                contents=final_prompt
                            )
                            st.session_state.last_decision = resp.text
                            status.update(label="Consensus Reached", state="complete")
                    
                    st.markdown(st.session_state.last_decision)

# --- TAB 2: DIRECT LINK (Expert) ---
with tabs[1]:
    target = ROLE_BY_KEY[st.session_state.selected_analyst_key]
    
    # Header Banner
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {target.hex_color}20, transparent); padding: 20px; border-left: 5px solid {target.hex_color}; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
        <h2 style="margin:0; color: white;">{target.icon} {target.name}</h2>
        <p style="margin:0; color: #a0a0a0;">{target.title} | <span style="color: {target.hex_color}">{target.focus}</span></p>
    </div>
    """, unsafe_allow_html=True)

    # Chat
    expert_container = st.container(height=500)
    with expert_container:
        for role, content in st.session_state.chat_logs[target.key]:
            render_chat_message(target.key, content, is_user=(role=="user"))

    if prompt := st.chat_input(f"Secure line to {target.name}..."):
        st.session_state.chat_logs[target.key].append(("user", prompt))
        with expert_container:
            render_chat_message("user", prompt, is_user=True)
            
            client = get_client()
            if client:
                with st.status("Computing...", expanded=True) as status:
                    sys_prompt = build_system_prompt(target)
                    resp = client.models.generate_content(
                        model=st.session_state.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=sys_prompt)
                    )
                    status.update(label="Transmission Received", state="complete", expanded=False)
                
                render_chat_message(target.key, resp.text)
                st.session_state.chat_logs[target.key].append(("assistant", resp.text))
            else:
                st.error("Offline")

# --- TAB 3: LOGS ---
with tabs[2]:
    st.json(st.session_state.chat_logs)
