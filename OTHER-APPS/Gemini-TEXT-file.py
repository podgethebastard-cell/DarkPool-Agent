import streamlit as st
import io
import json
import zipfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET

# Try importing OpenAI, handle gracefully if missing
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# =============================================================================
# Apex Toolkit v3.1: AI + Mobile Friendly + Partial Reruns (Optimized)
# =============================================================================

st.set_page_config(
    page_title="Apex Toolkit", 
    page_icon="🧰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# 1. Helpers (Preserved & Consolidated)
# ----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def now_stamp_file() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_ext(name: str, ext: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "untitled"
    if not ext.startswith("."):
        ext = "." + ext
    return name if name.lower().endswith(ext.lower()) else f"{name}{ext}"

def safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._ "
    cleaned = "".join(c for c in (name or "") if c in keep).strip()
    return cleaned or "untitled"

def build_zip(files: List[Dict[str, str]]) -> bytes:
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.writestr(f["path"], f["data"])
    buff.seek(0)
    return buff.read()

def init_state():
    if "notes" not in st.session_state:
        st.session_state.notes: List[Dict[str, Any]] = []
    if "snippets" not in st.session_state:
        st.session_state.snippets: List[Dict[str, Any]] = []
    if "active_note_id" not in st.session_state:
        st.session_state.active_note_id = None
    if "active_snippet_id" not in st.session_state:
        st.session_state.active_snippet_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Tool Settings
    if "tool_settings" not in st.session_state:
        st.session_state.tool_settings = {
            "autosave": False,
            "notes_default_export": "txt",
            "snippets_default_export": "native",
            "zip_folder": "apex_export",
            "openai_api_key": ""
        }
    
    # Auto-load API Key from Secrets
    if not st.session_state.tool_settings.get("openai_api_key"):
        try:
            if "OPENAI_API_KEY" in st.secrets:
                st.session_state.tool_settings["openai_api_key"] = st.secrets["OPENAI_API_KEY"]
        except FileNotFoundError:
            pass  # No secrets file found, ignore

def next_id(collection_key: str) -> int:
    items = st.session_state.get(collection_key, [])
    return (max([it["id"] for it in items], default=0) + 1) if items else 1

def get_by_id(items: List[Dict[str, Any]], item_id: Optional[int]):
    if item_id is None:
        return None
    for it in items:
        if it.get("id") == item_id:
            return it
    return None

def delete_by_id(items: List[Dict[str, Any]], item_id: int) -> List[Dict[str, Any]]:
    return [it for it in items if it.get("id") != item_id]

# ----------------------------
# 2. Data Logic: JSON & XML (Preserved)
# ----------------------------
def export_state_as_json() -> str:
    payload = {
        "version": "apex-toolkit-v3",
        "exported_at": now_stamp(),
        "notes": st.session_state.notes,
        "snippets": st.session_state.snippets,
        "settings": st.session_state.tool_settings,
    }
    return json.dumps(payload, indent=2)

def import_state_from_json(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Invalid backup format.")
    return {
        "notes": data.get("notes", []),
        "snippets": data.get("snippets", []),
        "settings": data.get("settings", {})
    }

def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def export_state_as_xml() -> str:
    root = ET.Element("apex_toolkit_export", attrib={"version": "v3", "exported_at": now_stamp()})
    
    settings_el = ET.SubElement(root, "settings")
    for k, v in (st.session_state.tool_settings or {}).items():
        # Avoid exporting sensitive key in plain text if possible, but keeping logic consistent
        item = ET.SubElement(settings_el, "setting", attrib={"key": str(k)})
        item.text = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

    notes_el = ET.SubElement(root, "notes", attrib={"count": str(len(st.session_state.notes))})
    for n in st.session_state.notes:
        note_el = ET.SubElement(notes_el, "note", attrib={"id": str(n.get("id", ""))})
        ET.SubElement(note_el, "title").text = n.get("title", "")
        tags_el = ET.SubElement(note_el, "tags")
        for t in n.get("tags", []) or []:
            ET.SubElement(tags_el, "tag").text = str(t)
        ET.SubElement(note_el, "created_at").text = n.get("created_at", "")
        ET.SubElement(note_el, "updated_at").text = n.get("updated_at", "")
        ET.SubElement(note_el, "body").text = n.get("body", "")

    snippets_el = ET.SubElement(root, "snippets", attrib={"count": str(len(st.session_state.snippets))})
    for s in st.session_state.snippets:
        snip_el = ET.SubElement(snippets_el, "snippet", attrib={"id": str(s.get("id", ""))})
        ET.SubElement(snip_el, "name").text = s.get("name", "")
        ET.SubElement(snip_el, "language").text = s.get("language", "")
        ET.SubElement(snip_el, "ext").text = s.get("ext", "")
        ET.SubElement(snip_el, "created_at").text = s.get("created_at", "")
        ET.SubElement(snip_el, "updated_at").text = s.get("updated_at", "")
        ET.SubElement(snip_el, "content").text = s.get("content", "")

    _indent_xml(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

def import_state_from_xml(xml_text: str) -> Dict[str, Any]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {e}")

    if root.tag not in ("apex_toolkit_export",):
        raise ValueError("Unrecognized XML root element.")

    # Settings
    settings = {}
    settings_el = root.find("settings")
    if settings_el is not None:
        for setting_el in settings_el.findall("setting"):
            key = setting_el.attrib.get("key", "")
            raw = setting_el.text or ""
            try:
                settings[key] = json.loads(raw)
            except Exception:
                settings[key] = raw

    # Notes
    notes = []
    notes_el = root.find("notes")
    if notes_el is not None:
        for note_el in notes_el.findall("note"):
            nid = note_el.attrib.get("id")
            tags = []
            tags_el = note_el.find("tags")
            if tags_el is not None:
                tags = [t.text or "" for t in tags_el.findall("tag")]
            notes.append({
                "id": int(nid) if nid and nid.isdigit() else next_id("notes"),
                "title": (note_el.findtext("title") or ""),
                "body": (note_el.findtext("body") or ""),
                "tags": [t for t in tags if t],
                "created_at": (note_el.findtext("created_at") or now_stamp()),
                "updated_at": (note_el.findtext("updated_at") or now_stamp()),
            })

    # Snippets
    snippets = []
    snippets_el = root.find("snippets")
    if snippets_el is not None:
        for snip_el in snippets_el.findall("snippet"):
            sid = snip_el.attrib.get("id")
            snippets.append({
                "id": int(sid) if sid and sid.isdigit() else next_id("snippets"),
                "name": (snip_el.findtext("name") or ""),
                "language": (snip_el.findtext("language") or "plaintext"),
                "ext": (snip_el.findtext("ext") or ".txt"),
                "content": (snip_el.findtext("content") or ""),
                "created_at": (snip_el.findtext("created_at") or now_stamp()),
                "updated_at": (snip_el.findtext("updated_at") or now_stamp()),
            })

    return {"notes": notes, "snippets": snippets, "settings": settings}

# Note & Snippet Export Helpers
def note_to_txt(n: Dict[str, Any], include_meta: bool = True) -> str:
    if include_meta:
        return (f"{n.get('title','Untitled')}\n\nTags: {', '.join(n.get('tags', []))}\n"
                f"Updated: {n.get('updated_at','')}\n\n{n.get('body','')}")
    return n.get("body", "")

def note_to_md(n: Dict[str, Any], include_meta: bool = True) -> str:
    meta = f"*Tags:* {', '.join(n.get('tags',[]))}  \n*Updated:* {n.get('updated_at','')}\n\n" if include_meta else ""
    return f"# {n.get('title','Untitled')}\n\n{meta}{n.get('body','')}\n"

def note_to_json(n: Dict[str, Any]) -> str: return json.dumps(n, indent=2)

def note_to_xml(n: Dict[str, Any]) -> str:
    root = ET.Element("note", attrib={"id": str(n.get("id", ""))})
    ET.SubElement(root, "title").text = n.get("title", "")
    tags_el = ET.SubElement(root, "tags")
    for t in n.get("tags", []) or []:
        ET.SubElement(tags_el, "tag").text = str(t)
    ET.SubElement(root, "created_at").text = n.get("created_at", "")
    ET.SubElement(root, "updated_at").text = n.get("updated_at", "")
    ET.SubElement(root, "body").text = n.get("body", "")
    _indent_xml(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

def snippet_to_json(s: Dict[str, Any]) -> str: return json.dumps(s, indent=2)

def snippet_to_xml(s: Dict[str, Any]) -> str:
    root = ET.Element("snippet", attrib={"id": str(s.get("id", ""))})
    ET.SubElement(root, "name").text = s.get("name", "")
    ET.SubElement(root, "language").text = s.get("language", "")
    ET.SubElement(root, "ext").text = s.get("ext", "")
    ET.SubElement(root, "created_at").text = s.get("created_at", "")
    ET.SubElement(root, "updated_at").text = s.get("updated_at", "")
    ET.SubElement(root, "content").text = s.get("content", "")
    _indent_xml(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

# ----------------------------
# 3. AI Assistant Logic
# ----------------------------
def run_ai_completion(prompt_text: str, context_text: str = ""):
    """Simple wrapper for OpenAI ChatCompletion."""
    api_key = st.session_state.tool_settings.get("openai_api_key")
    if not HAS_OPENAI:
        return "⚠️ Error: OpenAI library not installed. Please run `pip install openai`."
    if not api_key:
        return "⚠️ Error: No API Key found. Check Sidebar Settings or Streamlit Secrets."

    client = openai.OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant embedded in a note-taking and coding app called Apex Toolkit. Be concise, technical, and helpful."},
    ]
    if context_text:
        messages.append({"role": "system", "content": f"Current User Context (Note/Snippet):\n---\n{context_text}\n---"})
    
    messages.append({"role": "user", "content": prompt_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # or gpt-3.5-turbo
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ----------------------------
# 4. App Fragments (Optimization)
# ----------------------------

# We use @st.fragment to isolate updates. This allows the editor to save
# without reloading the entire page (sidebar, tabs, etc), making it much faster.

@st.fragment
def render_note_editor(note_id: int):
    n = get_by_id(st.session_state.notes, note_id)
    if not n:
        st.info("Note not found (it may have been deleted).")
        return

    # Sub-tabs for Editor vs AI
    edit_tab, ai_tab, prev_tab = st.tabs(["✏️ Edit", "🤖 AI Assistant", "👁️ Preview"])
    
    with edit_tab:
        new_title = st.text_input("Title", value=n.get("title", ""), key=f"n_title_{note_id}")
        new_tags = st.text_input("Tags (comma separated)", value=", ".join(n.get("tags", [])), key=f"n_tags_{note_id}")
        new_body = st.text_area("Content", value=n.get("body", ""), height=400, key=f"n_body_{note_id}")
        
        # Toolbar
        ac1, ac2, ac3 = st.columns([1, 1, 2])
        if ac1.button("💾 Save", use_container_width=True, key=f"n_save_{note_id}"):
            n["title"] = new_title
            n["tags"] = [t.strip() for t in new_tags.split(",") if t.strip()]
            n["body"] = new_body
            n["updated_at"] = now_stamp()
            st.success("Saved!")
            st.rerun() # Rerun fragment to update state visuals
        
        if ac2.button("🗑️ Delete", type="primary", use_container_width=True, key=f"n_del_{note_id}"):
            st.session_state.notes = delete_by_id(st.session_state.notes, n["id"])
            st.session_state.active_note_id = None
            st.rerun() # Must rerun whole app here to update the list outside fragment

        # Autosave logic
        if st.session_state.tool_settings.get("autosave", False):
            if n["title"] != new_title or n["body"] != new_body:
                n["title"] = new_title
                n["tags"] = [t.strip() for t in new_tags.split(",") if t.strip()]
                n["body"] = new_body
                n["updated_at"] = now_stamp()

    with ai_tab:
        st.caption(f"Context: {n.get('title')}")
        ai_input = st.text_area("Ask AI about this note...", height=100, key=f"n_ai_{note_id}")
        if st.button("Generate Answer", key=f"n_ai_btn_{note_id}"):
            with st.spinner("Thinking..."):
                context = f"Title: {n.get('title')}\nBody:\n{n.get('body')}"
                reply = run_ai_completion(ai_input, context)
                st.markdown("### AI Response")
                st.markdown(reply)
                st.info("Tip: Copy useful insights back to your note manually.")

    with prev_tab:
        st.markdown(f"# {n.get('title')}")
        st.markdown(n.get("body"))

@st.fragment
def render_snippet_editor(snip_id: int):
    s = get_by_id(st.session_state.snippets, snip_id)
    if not s:
        st.info("Snippet not found.")
        return

    s_edit, s_ai = st.tabs(["💻 Code", "🤖 AI Debugger"])
    
    with s_edit:
        c_meta1, c_meta2, c_meta3 = st.columns([2, 1, 1])
        s_name = c_meta1.text_input("Filename", value=s.get("name", "untitled"), key=f"s_name_{snip_id}")
        
        lang_opts = ["python", "javascript", "bash", "sql", "html", "css", "json", "markdown"]
        curr_lang = s.get("language","python")
        idx = lang_opts.index(curr_lang) if curr_lang in lang_opts else 0
        s_lang = c_meta2.selectbox("Lang", lang_opts, index=idx, key=f"s_lang_{snip_id}")
        
        s_ext = c_meta3.text_input("Ext", value=s.get("ext", ".txt"), key=f"s_ext_{snip_id}")
        
        s_content = st.text_area("Code", value=s.get("content",""), height=350, label_visibility="collapsed", key=f"s_code_{snip_id}")
        
        sc1, sc2 = st.columns([1, 4])
        if sc1.button("💾 Save Snippet", use_container_width=True, key=f"s_save_{snip_id}"):
            s["name"] = s_name
            s["language"] = s_lang
            s["ext"] = s_ext
            s["content"] = s_content
            s["updated_at"] = now_stamp()
            st.success("Saved")
            st.rerun()
        
        if st.session_state.tool_settings.get("autosave", False):
            s["name"] = s_name
            s["language"] = s_lang
            s["ext"] = s_ext
            s["content"] = s_content
            s["updated_at"] = now_stamp()

    with s_ai:
        st.info("AI Context: Current code snippet")
        ai_code_ask = st.text_area("Ask AI to refactor, debug, or explain...", height=100, key=f"s_ai_in_{snip_id}")
        if st.button("Run AI Analysis", key=f"s_ai_btn_{snip_id}"):
            with st.spinner("Analyzing code..."):
                context = f"Filename: {s.get('name')}\nLang: {s.get('language')}\nCode:\n{s.get('content')}"
                resp = run_ai_completion(ai_code_ask, context)
                st.markdown(resp)

# ----------------------------
# 5. App Initialization & Main Layout
# ----------------------------
init_state()

# =============================================================================
# UI Structure: Sidebar
# =============================================================================
with st.sidebar:
    st.title("🧰 Apex Toolkit")
    
    # Navigation
    page = st.radio("Navigate", [
        "Workstation", # Combined Notes & Snippets for better flow
        "Export & Backup",
        "Settings",
        "About"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # AI Quick Status
    if st.session_state.tool_settings.get("openai_api_key"):
        st.caption("🟢 AI Assistant Active")
    else:
        st.caption("⚪ AI Assistant Inactive (No Key)")

# =============================================================================
# Page: Workstation (Notes + Snippets + AI)
# =============================================================================
if page == "Workstation":
    # Tabbed Interface for Mobile Friendliness
    tab_notes, tab_snippets = st.tabs(["🗒️ Notes", "🧩 Snippets"])
    
    # --- NOTES TAB ---
    with tab_notes:
        col_list, col_editor = st.columns([1, 2], gap="medium")
        
        # NOTE LIST
        with col_list:
            st.subheader("My Notes")
            
            # Toolbar
            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ New Note", use_container_width=True):
                    new_id = next_id("notes")
                    st.session_state.notes.append({
                        "id": new_id,
                        "title": f"Note {new_id}",
                        "body": "",
                        "tags": [],
                        "created_at": now_stamp(),
                        "updated_at": now_stamp(),
                    })
                    st.session_state.active_note_id = new_id
                    st.rerun()
            with c2:
                search_q = st.text_input("Search", placeholder="Keywords...", label_visibility="collapsed", key="search_notes")

            # Filter Logic
            filtered_notes = st.session_state.notes
            if search_q:
                q = search_q.lower()
                filtered_notes = [n for n in filtered_notes if q in n.get("title","").lower() or q in n.get("body","").lower()]
            filtered_notes = sorted(filtered_notes, key=lambda x: x.get("updated_at", ""), reverse=True)

            # Selection List
            if not filtered_notes:
                st.info("No notes found.")
            else:
                note_map = {f"{n['title']} (ID:{n['id']})": n['id'] for n in filtered_notes}
                
                # Restore selection safely
                current_id = st.session_state.active_note_id
                current_idx = 0
                if current_id:
                    valid_ids = [n['id'] for n in filtered_notes]
                    if current_id in valid_ids:
                        current_idx = valid_ids.index(current_id)

                selected_label = st.selectbox(
                    "Select Note", 
                    options=list(note_map.keys()), 
                    index=current_idx,
                    label_visibility="collapsed",
                    key="note_selector"
                )
                if selected_label:
                    st.session_state.active_note_id = note_map[selected_label]

        # NOTE EDITOR (OPTIMIZED FRAGMENT)
        with col_editor:
            if st.session_state.active_note_id:
                render_note_editor(st.session_state.active_note_id)
            else:
                st.info("Create or select a note to begin.")

    # --- SNIPPETS TAB ---
    with tab_snippets:
        col_slist, col_seditor = st.columns([1, 2], gap="medium")
        
        with col_slist:
            st.subheader("Code Snippets")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ New Snippet", use_container_width=True):
                    new_id = next_id("snippets")
                    st.session_state.snippets.append({
                        "id": new_id,
                        "name": f"script_{new_id}",
                        "language": "python",
                        "ext": ".py",
                        "content": "",
                        "created_at": now_stamp(),
                        "updated_at": now_stamp(),
                    })
                    st.session_state.active_snippet_id = new_id
                    st.rerun()
            
            # Filter
            s_query = st.text_input("Find snippet", placeholder="Name or code...", label_visibility="collapsed", key="search_snips")
            filtered_snips = st.session_state.snippets
            if s_query:
                q = s_query.lower()
                filtered_snips = [s for s in filtered_snips if q in s.get("name","").lower() or q in s.get("content","").lower()]
            
            # List
            if not filtered_snips:
                st.info("No snippets found.")
            else:
                snip_map = {f"{s['name']} ({s.get('language')})": s['id'] for s in filtered_snips}
                current_s_id = st.session_state.active_snippet_id
                curr_s_idx = 0
                if current_s_id and current_s_id in snip_map.values():
                    curr_s_idx = list(snip_map.values()).index(current_s_id)
                
                sel_snip = st.selectbox("Select Snippet", list(snip_map.keys()), index=curr_s_idx, label_visibility="collapsed", key="snip_selector")
                if sel_snip:
                    st.session_state.active_snippet_id = snip_map[sel_snip]

        # SNIPPET EDITOR (OPTIMIZED FRAGMENT)
        with col_seditor:
            if st.session_state.active_snippet_id:
                render_snippet_editor(st.session_state.active_snippet_id)
            else:
                st.info("Select a snippet.")

# =============================================================================
# Page: Export & Backup (Consolidated)
# =============================================================================
elif page == "Export & Backup":
    st.title("📦 Export & Backup Operations")
    
    tab_zip, tab_json, tab_xml = st.tabs(["🗜️ Multi-File ZIP", "📄 JSON Backup", "🧾 XML Tools"])
    
    # --- ZIP EXPORTER ---
    with tab_zip:
        st.caption("Select items to bundle into a single ZIP file.")
        
        c_ex1, c_ex2 = st.columns(2)
        with c_ex1:
            st.markdown("**Notes to include**")
            all_n_titles = [n["title"] for n in st.session_state.notes]
            sel_n_titles = st.multiselect("Select Notes", all_n_titles, default=all_n_titles)
            fmt_n = st.selectbox("Format", ["txt", "md", "json", "xml"], index=0)
        
        with c_ex2:
            st.markdown("**Snippets to include**")
            all_s_names = [s["name"] for s in st.session_state.snippets]
            sel_s_names = st.multiselect("Select Snippets", all_s_names, default=all_s_names)
            fmt_s = st.selectbox("Format", ["native", "json", "xml"], index=0)

        st.markdown("---")
        if st.button("⬇️ Generate & Download ZIP", type="primary"):
            files_to_zip = []
            
            # Process Notes
            for n in st.session_state.notes:
                if n["title"] in sel_n_titles:
                    base = safe_filename(n["title"])
                    if fmt_n == "txt":
                        d, e = note_to_txt(n), ".txt"
                    elif fmt_n == "md":
                        d, e = note_to_md(n), ".md"
                    elif fmt_n == "json":
                        d, e = note_to_json(n), ".json"
                    else:
                        d, e = note_to_xml(n), ".xml"
                    files_to_zip.append({"path": f"notes/{base}{e}", "data": d})
            
            # Process Snippets
            for s in st.session_state.snippets:
                if s["name"] in sel_s_names:
                    base = safe_filename(s["name"])
                    if fmt_s == "native":
                        d, e = s["content"], s.get("ext", ".txt")
                    elif fmt_s == "json":
                        d, e = snippet_to_json(s), ".json"
                    else:
                        d, e = snippet_to_xml(s), ".xml"
                    files_to_zip.append({"path": f"snippets/{base}{e}", "data": d})
            
            if files_to_zip:
                zip_data = build_zip(files_to_zip)
                st.download_button(
                    "Download ZIP", 
                    data=zip_data, 
                    file_name=f"apex_export_{now_stamp_file()}.zip", 
                    mime="application/zip"
                )
            else:
                st.warning("No items selected.")

    # --- JSON BACKUP ---
    with tab_json:
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.subheader("Backup")
            st.caption("Full state snapshot.")
            bkp_json = export_state_as_json()
            st.download_button("⬇️ Download JSON Backup", data=bkp_json, file_name=f"backup_{now_stamp_file()}.json", mime="application/json")
        
        with col_j2:
            st.subheader("Restore")
            up_j = st.file_uploader("Upload JSON", type=["json"])
            if up_j and st.button("Restore from JSON"):
                try:
                    data = import_state_from_json(up_j.read().decode("utf-8"))
                    st.session_state.notes = data["notes"]
                    st.session_state.snippets = data["snippets"]
                    st.session_state.tool_settings.update(data["settings"])
                    st.success("Restored successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- XML TOOLS ---
    with tab_xml:
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.subheader("Export XML")
            xml_out = export_state_as_xml()
            st.download_button("⬇️ Download Workspace XML", data=xml_out, file_name=f"workspace_{now_stamp_file()}.xml", mime="application/xml")
        
        with col_x2:
            st.subheader("Restore XML")
            up_x = st.file_uploader("Upload XML", type=["xml"])
            if up_x and st.button("Restore from XML"):
                try:
                    data = import_state_from_xml(up_x.read().decode("utf-8"))
                    st.session_state.notes = data["notes"]
                    st.session_state.snippets = data["snippets"]
                    st.session_state.tool_settings.update(data["settings"])
                    st.success("Restored from XML.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================================
# Page: Settings
# =============================================================================
elif page == "Settings":
    st.title("🔧 Settings")
    
    st.subheader("General")
    st.session_state.tool_settings["autosave"] = st.checkbox(
        "Enable Autosave", 
        value=st.session_state.tool_settings.get("autosave", False),
        help="Saves changes to memory as you type. Explicit save is still recommended for important edits."
    )

    st.markdown("---")
    st.subheader("AI Configuration")
    
    # API Key Input
    # 1. Check secrets first (masked in UI if present)
    # 2. Allow manual override
    
    secrets_key = ""
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", "")
    except FileNotFoundError:
        pass
        
    current_key = st.session_state.tool_settings.get("openai_api_key", "")
    
    if secrets_key and current_key == secrets_key:
        st.success("✅ OpenAI API Key loaded from Streamlit Secrets.")
        if st.checkbox("Override Secrets Key"):
            new_key = st.text_input("Enter new OpenAI API Key", type="password")
            if new_key:
                st.session_state.tool_settings["openai_api_key"] = new_key
    else:
        st.warning("⚠️ No API Key found in Secrets.")
        new_key = st.text_input("Enter OpenAI API Key", value=current_key, type="password")
        if st.button("Save Key"):
            st.session_state.tool_settings["openai_api_key"] = new_key
            st.success("Key saved to session.")

    st.markdown("---")
    st.subheader("Danger Zone")
    if st.button("🧨 Wipe All Data"):
        st.session_state.notes = []
        st.session_state.snippets = []
        st.success("Workspace cleared.")
        st.rerun()

# =============================================================================
# Page: About
# =============================================================================
else:
    st.title("ℹ️ About Apex Toolkit v3.1")
    st.markdown("""
    **Apex Toolkit v3.1** is a powerful, single-file workspace for developers and writers.
    
    **Optimizations:**
    * **Fragments:** Editors run in isolated scopes for fast updates.
    * **Mobile:** Responsive tabs.
    * **Auto-Config:** Streamlit secrets support.
    
    **Core Features:**
    * **Notes:** Markdown support, tagging, and search.
    * **Snippets:** Syntax highlighting for multiple languages.
    * **Export:** ZIP, JSON, and XML standards-compliant export/import.
    """)
