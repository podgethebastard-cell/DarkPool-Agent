import streamlit as st
from openai import OpenAI

# --- PAGE CONFIG ---
st.set_page_config(page_title="DarkPool Researcher", page_icon="🕵️")

# --- SIDEBAR CONFIG ---
st.sidebar.header("🕵️ Agent Control")
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# --- MAIN INTERFACE ---
st.title("🕵️ DarkPool Research Agent")
st.markdown("##### *Your Dedicated Market Analyst & Strategist*")
st.markdown("---")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I am Agent 2, your DarkPool Researcher. I can help you brainstorm strategies, explain complex macro concepts, or draft trade plans. What are we analyzing?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input & Logic
if prompt := st.chat_input("Ask me about market structure, psychology, or strategy..."):
    if not api_key:
        st.info("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. AI Response
    client = OpenAI(api_key=api_key)
    
    # SYSTEM PROMPT: Defines the agent's specific job
    system_instruction = """
    You are 'Agent 2', a sophisticated Market Research Assistant for the DarkPool Titan Terminal. 
    Your tone is professional, institutional, and concise. 
    You specialize in:
    1. Explaining complex trading concepts (SMC, Wyckoff, Macro).
    2. Brainstorming risk management strategies.
    3. Drafting trade plans based on user input.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_instruction}] + st.session_state.messages
    )
    
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
