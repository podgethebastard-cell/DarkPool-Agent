# --- TAB 3: CONFIG (INPUTS FIRST FOR DATA) ---
with tab3:
    st.header("Titan Configuration")
    
    with st.expander("📡 Market Feed", expanded=True):
        symbol = st.text_input("Symbol (Kraken)", value="BTC/USD") 
        timeframe = st.selectbox("Timeframe", options=['1m', '5m', '15m'], index=1)
        limit = st.slider("Candles", 200, 1500, 500)

    with st.expander("🧠 Logic Engine"):
        amplitude = st.number_input("Sensitivity", min_value=1, value=5)
        channel_dev = st.number_input("Stop Deviation", value=3.0)
        hma_len = st.number_input("HMA Length", value=50)
        mf_len = 14; vol_len = 20
        
    with st.expander("🔑 Keys & Alerts", expanded=True):
        tg_on = st.checkbox("Telegram Active", value=True)
        
        # --- 1. TELEGRAM SECRETS HANDLING ---
        try:
            default_bot = st.secrets["TELEGRAM_TOKEN"]
            default_chat = st.secrets["TELEGRAM_CHAT_ID"]
        except:
            default_bot = ""; default_chat = ""
            
        bot_token = st.text_input("Bot Token", value=default_bot, type="password")
        chat_id = st.text_input("Chat ID", value=default_chat)

        # --- 2. OPENAI SECRETS HANDLING ---
        try:
            default_ai = st.secrets["OPENAI_API_KEY"]
            st.success("OpenAI Key Found in Secrets! ✅")
        except:
            default_ai = ""
            
        # The text input defaults to the secret if it exists, otherwise it's blank
        ai_key = st.text_input("OpenAI Key", value=default_ai, type="password", help="Press ENTER after pasting!")
