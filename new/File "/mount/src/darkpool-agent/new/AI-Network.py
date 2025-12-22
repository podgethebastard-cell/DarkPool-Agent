
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="SOLO AI", layout="wide")

@dataclass
class Analyst:
    key: str
    name: str
    tier: int
    title: str
    focus: str

ANALYSTS = [
    Analyst("a1","Sentinel",1,"Security Sentinel","Risk & Threat Modeling"),
    Analyst("a2","Architect",2,"Systems Architect","Architecture & Strategy"),
    Analyst("a5","Growth",3,"Growth Operator","Distribution & GTM"),
    Analyst("a11","Auditor",2,"Compliance Auditor","Controls & Policy"),
]

TIER_NAMES = {1:"Sovereign", 2:"Strategic", 3:"Execution", 4:"Support"}

def init_state():
    st.session_state.setdefault("active_tab", "Council")
    st.session_state.setdefault("selected_analyst", "a2")
    st.session_state.setdefault("selected_council", ["a1","a2","a5","a11"])
    st.session_state.setdefault("chat_logs", {a.key: [] for a in ANALYSTS})
    st.session_state.setdefault("last_decision", "")
    st.session_state.setdefault("current_image", None)
    st.session_state.setdefault("edited_image", None)

init_state()

def fake_generate(system_prompt: str, user_text: str):
    # Replace with your real Gemini/OpenAI call
    return f"**[Stub]** {system_prompt}\n\nYou said: {user_text}"

def synthesize(user_text: str, responses: dict):
    # Replace with your real synthesis
    bullets = "\n".join([f"- **{k}**: {v.splitlines()[0][:120]}..." for k,v in responses.items()])
    return f"## Decision Card\n\n**User:** {user_text}\n\n### Council Summary\n{bullets}\n\n### Next Actions\n- Execute plan A\n- Validate assumptions\n- Ship a small test"

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🧠 SOLO AI")
    st.caption("Online • Protocol 2.1")

    # Tier grouped analysts
    for tier in [1,2,3,4]:
        tier_analysts = [a for a in ANALYSTS if a.tier == tier]
        if not tier_analysts:
            continue
        st.markdown(f"### {TIER_NAMES[tier]}")
        for a in tier_analysts:
            if st.button(f"{a.name} — {a.title}", use_container_width=True):
                st.session_state.selected_analyst = a.key
                st.session_state.active_tab = "Expert"

# --- Top tabs ---
tab_names = ["Council", "Expert", "Live", "Media"]
tabs = st.tabs(tab_names)

# --- Council tab ---
with tabs[0]:
    st.markdown("## Neural Council")
    st.caption("Decentralized Execution Protocol")

    cols = st.columns(len(ANALYSTS))
    for i, a in enumerate(ANALYSTS):
        with cols[i]:
            checked = a.key in st.session_state.selected_council
            new = st.toggle(a.name, value=checked, key=f"toggle_{a.key}")
            if new and not checked:
                st.session_state.selected_council.append(a.key)
            if (not new) and checked:
                st.session_state.selected_council = [k for k in st.session_state.selected_council if k != a.key]

    user_text = st.text_area("Broadcast baseline parameters…", height=160, key="council_input")

    if st.button("Execute Synthesis", type="primary", use_container_width=True, disabled=not user_text.strip()):
        responses = {}
        for k in st.session_state.selected_council:
            a = next(x for x in ANALYSTS if x.key == k)
            responses[k] = fake_generate(f"SystemPrompt({a.title})", user_text)

        st.session_state.last_decision = synthesize(user_text, responses)

        # store logs per analyst
        for k, resp in responses.items():
            st.session_state.chat_logs[k].append(("user", user_text))
            st.session_state.chat_logs[k].append(("assistant", resp))

    if st.session_state.last_decision:
        st.markdown("---")
        st.markdown(st.session_state.last_decision)

# --- Expert tab ---
with tabs[1]:
    a = next(x for x in ANALYSTS if x.key == st.session_state.selected_analyst)
    st.markdown(f"## {a.title}")
    st.caption(f"Focus: {a.focus}")

    # render chat
    for role, content in st.session_state.chat_logs[a.key]:
        with st.chat_message(role):
            st.markdown(content)

    msg = st.chat_input(f"Consult {a.name}…")
    if msg:
        st.session_state.chat_logs[a.key].append(("user", msg))
        st.session_state.chat_logs[a.key].append(("assistant", fake_generate(f"SystemPrompt({a.title})", msg)))
        st.rerun()

# --- Live tab (placeholder) ---
with tabs[2]:
    st.markdown("## Voice War Room")
    st.info("Streamlit can do live audio via streamlit-webrtc, or you can run a separate React frontend for realtime voice.")

# --- Media tab ---
with tabs[3]:
    st.markdown("## Neural Lab")
    up = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    if up:
        st.session_state.current_image = up.getvalue()
        st.image(st.session_state.current_image, caption="Injection Point", use_container_width=True)

    prompt = st.text_area("Vision directives…", height=120, key="img_prompt")

    if st.button("Deploy Image Matrix", use_container_width=True, disabled=not (prompt.strip() and st.session_state.current_image)):
        # Replace with real image edit call
        st.session_state.edited_image = st.session_state.current_image

    if st.session_state.edited_image:
        st.image(st.session_state.edited_image, caption="Recon Output", use_container_width=True)
