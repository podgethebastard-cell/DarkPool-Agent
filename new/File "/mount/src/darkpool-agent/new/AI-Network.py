
import streamlit as st
from dataclasses import dataclass
import time
import base64

# --- 1. System Configuration ---
st.set_page_config(
    page_title="SOLO AI // ARCHITECT",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DPC CSS Architecture (Dark Pool Compliant) ---
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
            --border-glow: 0 0 5px rgba(0, 210, 255, 0.3);
        }

        /* Global Resets */
        .stApp {
            background-color: var(--background-dark);
            color: var(--text-main);
            font-family: 'Roboto Mono', monospace;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6, p, div, span {
            font-family: 'Roboto Mono', monospace !important;
            color: var(--text-main) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--panel-dark);
            border-right: 1px solid #30363d;
        }

        /* Custom Buttons */
        div.stButton > button {
            background-color: var(--panel-dark);
            color: var(--neon-blue);
            border: 1px solid var(--neon-blue);
            box-shadow: 0 0 5px rgba(0, 210, 255, 0.1);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: var(--neon-blue);
            color: var(--background-dark) !important;
            box-shadow: 0 0 15px var(--neon-blue);
        }
        
        /* Primary Action Button */
        div.stButton > button[kind="primary"] {
            background-color: rgba(0, 255, 65, 0.1);
            color: var(--neon-green);
            border: 1px solid var(--neon-green);
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: var(--neon-green);
            color: var(--background-dark) !important;
            box-shadow: 0 0 15px var(--neon-green);
        }

        /* Text Areas & Inputs */
        .stTextArea textarea, .stTextInput input {
            background-color: #0d1117;
            border: 1px solid #30363d;
            color: #c9d1d9;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--neon-purple);
            box-shadow: 0 0 5px var(--neon-purple);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: var(--panel-dark);
            border-radius: 4px 4px 0px 0px;
            color: #8b949e;
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--background-dark);
            color: var(--neon-blue);
            border-top: 2px solid var(--neon-blue);
        }

        /* Containers (Cards) */
        [data-testid="stVerticalBlock"] > div > div {
           /* Subtle separation for cards could go here */
        }
        
        </style>
    """, unsafe_allow_html=True)

_inject_dpc_css()

# --- 3. Data Structures ---
@dataclass
class Analyst:
    key: str
    name: str
    tier: int
    title: str
    focus: str
    color: str  # Added for UI accent

ANALYSTS = [
    Analyst("a1", "Sentinel", 1, "Security Sentinel", "Risk & Threat Modeling", "#ff0000"),
    Analyst("a2", "Architect", 2, "Systems Architect", "Architecture & Strategy", "#00d2ff"),
    Analyst("a5", "Growth", 3, "Growth Operator", "Distribution & GTM", "#00ff41"),
    Analyst("a11", "Auditor", 2, "Compliance Auditor", "Controls & Policy", "#bc13fe"),
]

TIER_NAMES = {1: "Sovereign", 2: "Strategic", 3: "Execution", 4: "Support"}
ANALYST_MAP = {a.key: a for a in ANALYSTS}

# --- 4. Logic Core ---
def init_state():
    """Initialize session state with default values."""
    defaults = {
        "active_tab": "Council",
        "selected_analyst": "a2",
        "selected_council": ["a1", "a2", "a5", "a11"],
        "chat_logs": {a.key: [] for a in ANALYSTS},
        "last_decision": "",
        "current_image": None,
        "edited_image": None,
        "processing": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def generate_response(system_prompt: str, user_text: str, agent_key: str):
    """
    Simulated Inference Engine.
    Integration: Connect this to `agentmob.py` or `deepseekagent.py` logic.
    """
    # Simulate processing latency for realism
    time.sleep(0.3) 
    
    # Placeholder logic
    prefix = f"**[{ANALYST_MAP[agent_key].name} Protocol]**"
    return f"{prefix}\n\nAnalyzing parameters: `{user_text}`...\n\nInsight: The architectural pattern requires immediate vectorization. Recommended moving from loop-based execution to matrix operations."

def synthesize_council(user_text: str, responses: dict):
    """Synthesize multiple analyst outputs into a decision card."""
    bullets = "\n".join([f"- **{ANALYST_MAP[k].name}**: {v.splitlines()[-1]}" for k, v in responses.items()])
    
    return f"""
### 💠 Council Consensus
**Directive:** `{user_text}`

---
{bullets}

---
### ⚠️ Executive Action Required
1. **Approve** the vectorized deployment.
2. **Override** safety protocols for `a1`.
3. **Commit** changes to `main`.
"""

# --- 5. UI Layout ---

# Sidebar
with st.sidebar:
    st.markdown("## 💀 THE ARCHITECT")
    st.caption(f"System Status: **ONLINE** | Protocol 2.1")
    st.divider()

    # Tiered Analyst Selection
    for tier in sorted(TIER_NAMES.keys()):
        tier_analysts = [a for a in ANALYSTS if a.tier == tier]
        if not tier_analysts:
            continue
        
        st.markdown(f"### {TIER_NAMES[tier]}")
        for a in tier_analysts:
            # Custom styled button logic via callback
            if st.button(f"{a.name} // {a.title}", key=f"btn_{a.key}", use_container_width=True):
                st.session_state.selected_analyst = a.key
                # Optional: Force switch tab if needed, or just update state
                
    st.divider()
    st.markdown("`v2.1.0-darkpool`")

# Main Interface
tab_names = ["⚔️ Council", "🧠 Expert", "📡 Live", "👁️ Vision"]
tabs = st.tabs(tab_names)

# --- Tab 1: Council (Multi-Agent Synthesis) ---
with tabs[0]:
    col_header, col_status = st.columns([3, 1])
    with col_header:
        st.markdown("## Neural Council Grid")
        st.caption("Decentralized Execution Protocol // Multi-Agent Consensus")
    with col_status:
        st.markdown(f"<div style='text-align: right; color: #00ff41;'>ACTIVE NODES: {len(st.session_state.selected_council)}</div>", unsafe_allow_html=True)

    # Analyst Toggle Grid
    cols = st.columns(len(ANALYSTS))
    for i, a in enumerate(ANALYSTS):
        with cols[i]:
            container = st.container()
            is_selected = a.key in st.session_state.selected_council
            
            # Stylized Card
            st.markdown(f"**{a.name}**", help=a.focus)
            if st.checkbox("Active", value=is_selected, key=f"toggle_{a.key}", label_visibility="collapsed"):
                if a.key not in st.session_state.selected_council:
                    st.session_state.selected_council.append(a.key)
            else:
                if a.key in st.session_state.selected_council:
                    st.session_state.selected_council.remove(a.key)

    st.divider()
    
    # Input & Execution
    user_text = st.text_area("BROADCAST DIRECTIVES", height=120, placeholder="Enter strategic parameters...", key="council_input")

    if st.button("EXECUTE SYNTHESIS", type="primary", use_container_width=True, disabled=not user_text.strip()):
        st.session_state.processing = True
        
        with st.spinner("Encrypting payload..."):
            responses = {}
            progress_bar = st.progress(0)
            
            for idx, k in enumerate(st.session_state.selected_council):
                a = ANALYST_MAP[k]
                resp = generate_response(a.title, user_text, k)
                responses[k] = resp
                
                # Log to individual history
                st.session_state.chat_logs[k].append(("user", user_text))
                st.session_state.chat_logs[k].append(("assistant", resp))
                
                progress_bar.progress((idx + 1) / len(st.session_state.selected_council))

            st.session_state.last_decision = synthesize_council(user_text, responses)
        
        st.session_state.processing = False
        st.rerun()

    # Results Display
    if st.session_state.last_decision:
        st.markdown(st.session_state.last_decision)

# --- Tab 2: Expert (Single Agent Deep Dive) ---
with tabs[1]:
    target = ANALYST_MAP[st.session_state.selected_analyst]
    
    # Header
    c1, c2 = st.columns([1, 6])
    with c1:
        # Simulated Avatar
        st.markdown(f"<div style='background:{target.color}; width:50px; height:50px; border-radius:50%; box-shadow: 0 0 10px {target.color};'></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"## {target.name}")
        st.markdown(f"`{target.title}` | Focus: **{target.focus}**")

    st.divider()

    # Chat Log Container
    chat_container = st.container()
    with chat_container:
        for role, content in st.session_state.chat_logs[target.key]:
            with st.chat_message(role):
                st.markdown(content)

    # Input
    msg = st.chat_input(f"Interrogate {target.name}...")
    if msg:
        st.session_state.chat_logs[target.key].append(("user", msg))
        resp = generate_response(target.title, msg, target.key)
        st.session_state.chat_logs[target.key].append(("assistant", resp))
        st.rerun()

# --- Tab 3: Live (Realtime Data) ---
with tabs[2]:
    st.markdown("## 📡 Voice War Room")
    st.info("Audio uplink requires `streamlit-webrtc` module. Secure channel standby.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latency", "24ms", "-2ms", delta_color="inverse")
    with col2:
        st.metric("Packets", "1.2M", "5%", delta_color="normal")

# --- Tab 4: Vision (Image Analysis) ---
with tabs[3]:
    st.markdown("## 👁️ Neural Vision Lab")
    
    col_upload, col_result = st.columns(2)
    
    with col_upload:
        up = st.file_uploader("Upload Recon Image", type=["png", "jpg", "jpeg"])
        if up:
            st.session_state.current_image = up.getvalue()
            st.image(st.session_state.current_image, caption="Source Input", use_container_width=True)

    with col_result:
        prompt = st.text_area("Vision Directives", height=100, key="img_prompt")
        
        if st.button("DEPLOY VISION MATRIX", use_container_width=True, disabled=not (prompt.strip() and st.session_state.current_image)):
            with st.spinner("Processing visual data..."):
                time.sleep(1) # Sim processing
                st.session_state.edited_image = st.session_state.current_image
                st.success("Analysis Complete")

        if st.session_state.edited_image:
            st.image(st.session_state.edited_image, caption="Analysis Output", use_container_width=True)
