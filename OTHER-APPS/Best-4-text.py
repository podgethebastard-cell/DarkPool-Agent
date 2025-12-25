import streamlit as st
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Optional

# =============================================================================
# Simplified Apex Toolkit: Notes + Multi-format Export
# =============================================================================

st.set_page_config(page_title="Simple Notes", page_icon="📝", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._ "
    cleaned = "".join(c for c in (name or "") if c in keep).strip()
    return cleaned or "untitled"

def ensure_ext(name: str, ext: str) -> str:
    if not ext.startswith("."):
        ext = "." + ext
    return name if name.lower().endswith(ext.lower()) else f"{name}{ext}"

# ----------------------------
# XML Logic
# ----------------------------
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

def note_to_xml(n: Dict[str, Any]) -> str:
    root = ET.Element("note", attrib={"id": str(n.get("id", ""))})
    ET.SubElement(root, "title").text = n.get("title", "")
    ET.SubElement(root, "created_at").text = n.get("created_at", "")
    ET.SubElement(root, "body").text = n.get("body", "")
    _indent_xml(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

# ----------------------------
# State Management
# ----------------------------
if "notes" not in st.session_state:
    st.session_state.notes = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None

def get_note(nid):
    return next((n for n in st.session_state.notes if n["id"] == nid), None)

def create_note():
    new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
    st.session_state.notes.append({
        "id": new_id,
        "title": f"Note {new_id}",
        "body": "",
        "created_at": now_stamp()
    })
    st.session_state.active_id = new_id

def delete_note(nid):
    st.session_state.notes = [n for n in st.session_state.notes if n["id"] != nid]
    st.session_state.active_id = None
    st.rerun()

# =============================================================================
# UI Layout
# =============================================================================

# --- Sidebar: Note List ---
with st.sidebar:
    st.title("📝 Notes")
    if st.button("➕ New Note", use_container_width=True):
        create_note()
        st.rerun()
    
    st.markdown("---")
    
    if not st.session_state.notes:
        st.info("No notes yet.")
    else:
        # Sort by ID descending (newest first)
        sorted_notes = sorted(st.session_state.notes, key=lambda x: x["id"], reverse=True)
        
        # Create a dictionary for the radio button labels
        note_map = {f"#{n['id']} {n['title']}": n["id"] for n in sorted_notes}
        
        # Find index of currently active note for the UI selector
        current_idx = 0
        if st.session_state.active_id:
            current_ids = list(note_map.values())
            if st.session_state.active_id in current_ids:
                current_idx = current_ids.index(st.session_state.active_id)

        selection = st.radio(
            "Select Note", 
            options=note_map.keys(), 
            index=current_idx,
            label_visibility="collapsed"
        )
        st.session_state.active_id = note_map[selection]

# --- Main Area: Editor & Export ---
active_note = get_note(st.session_state.active_id)

if active_note:
    st.subheader(f"Editing: {active_note['title']}")
    
    # Inputs
    new_title = st.text_input("Title", value=active_note["title"])
    new_body = st.text_area("Content", value=active_note["body"], height=300)

    # Save logic (Updates state immediately on interaction)
    if new_title != active_note["title"] or new_body != active_note["body"]:
        active_note["title"] = new_title
        active_note["body"] = new_body

    st.markdown("---")
    
    # Export Section
    col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
    
    with col1:
        fmt = st.selectbox("Format", ["txt", "xml", "json"])
    
    with col2:
        # Prepare Download Data
        file_data = ""
        mime_type = "text/plain"
        ext = ".txt"

        if fmt == "txt":
            file_data = f"{active_note['title']}\n{'-'*20}\n{active_note['body']}"
            mime_type = "text/plain"
            ext = ".txt"
        elif fmt == "xml":
            file_data = note_to_xml(active_note)
            mime_type = "application/xml"
            ext = ".xml"
        elif fmt == "json":
            file_data = json.dumps(active_note, indent=2)
            mime_type = "application/json"
            ext = ".json"

        st.download_button(
            label="⬇️ Download",
            data=file_data,
            file_name=safe_filename(ensure_ext(active_note["title"], ext)),
            mime=mime_type,
            use_container_width=True
        )

    with col3:
        if st.button("🗑️ Delete Note", type="primary"):
            delete_note(active_note["id"])

else:
    st.markdown("### 👋 Welcome")
    st.write("Select a note from the sidebar or click **New Note** to get started.")
