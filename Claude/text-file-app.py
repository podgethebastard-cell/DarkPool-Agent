import streamlit as st
import io
import json
import zipfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET

# =============================================================================
# Apex Toolkit: Notes + Snippets + Multi-file Exporter + XML + Upgrades
# Single-file Streamlit app
# =============================================================================

st.set_page_config(page_title="Apex Toolkit", page_icon="🧰", layout="wide")

# ----------------------------
# Helpers
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
    if "tool_settings" not in st.session_state:
        st.session_state.tool_settings = {
            "autosave": False,
            "autosave_seconds": 0,  # (kept for future extension)
            "notes_default_export": "txt",
            "snippets_default_export": "native",
            "zip_folder": "apex_export",
        }

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
# JSON Backup / Restore
# ----------------------------
def export_state_as_json() -> str:
    payload = {
        "version": "apex-toolkit-v2",
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
    notes = data.get("notes", [])
    snippets = data.get("snippets", [])
    settings = data.get("settings", {})
    if not isinstance(notes, list) or not isinstance(snippets, list):
        raise ValueError("Invalid backup format (notes/snippets).")
    if settings and not isinstance(settings, dict):
        raise ValueError("Invalid backup format (settings).")
    return {"notes": notes, "snippets": snippets, "settings": settings}

# ----------------------------
# XML Export / Import (NEW)
# ----------------------------
def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    # Pretty print indenting for ElementTree
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
    root = ET.Element("apex_toolkit_export", attrib={"version": "v2", "exported_at": now_stamp()})

    settings_el = ET.SubElement(root, "settings")
    for k, v in (st.session_state.tool_settings or {}).items():
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
        # Put body in text node; XML escaping will be handled automatically.
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
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")

def import_state_from_xml(xml_text: str) -> Dict[str, Any]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {e}")

    if root.tag not in ("apex_toolkit_export",):
        raise ValueError("Unrecognized XML root element.")

    # settings
    settings = {}
    settings_el = root.find("settings")
    if settings_el is not None:
        for setting_el in settings_el.findall("setting"):
            key = setting_el.attrib.get("key", "")
            raw = setting_el.text or ""
            # Try to parse JSON for richer types; fallback to string
            try:
                settings[key] = json.loads(raw)
            except Exception:
                settings[key] = raw

    # notes
    notes = []
    notes_el = root.find("notes")
    if notes_el is not None:
        for note_el in notes_el.findall("note"):
            nid = note_el.attrib.get("id")
            title = (note_el.findtext("title") or "")
            created_at = (note_el.findtext("created_at") or "")
            updated_at = (note_el.findtext("updated_at") or "")
            body = (note_el.findtext("body") or "")
            tags = []
            tags_el = note_el.find("tags")
            if tags_el is not None:
                tags = [t.text or "" for t in tags_el.findall("tag")]
            notes.append({
                "id": int(nid) if nid and nid.isdigit() else next_id("notes"),
                "title": title,
                "body": body,
                "tags": [t for t in tags if t],
                "created_at": created_at or now_stamp(),
                "updated_at": updated_at or now_stamp(),
            })

    # snippets
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

# ----------------------------
# Note export helpers (txt/md/json/xml)
# ----------------------------
def note_to_txt(n: Dict[str, Any], include_meta: bool = True) -> str:
    title = n.get("title", "Untitled")
    if include_meta:
        return (
            f"{title}\n\n"
            f"Tags: {', '.join(n.get('tags', []))}\n"
            f"Created: {n.get('created_at','')}\n"
            f"Updated: {n.get('updated_at','')}\n\n"
            f"{n.get('body','')}"
        )
    return n.get("body", "")

def note_to_md(n: Dict[str, Any], include_meta: bool = True) -> str:
    title = n.get("title", "Untitled")
    tags = n.get("tags", [])
    meta = ""
    if include_meta:
        meta = f"*Tags:* {', '.join(tags)}  \n*Updated:* {n.get('updated_at','')}\n\n"
    return f"# {title}\n\n{meta}{n.get('body','')}\n"

def note_to_json(n: Dict[str, Any]) -> str:
    return json.dumps(n, indent=2)

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

# ----------------------------
# Snippet export helpers (native/json/xml)
# ----------------------------
def snippet_to_json(s: Dict[str, Any]) -> str:
    return json.dumps(s, indent=2)

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
# State init
# ----------------------------
init_state()

# =============================================================================
# Sidebar: Toolkit wrapper + settings upgrades
# =============================================================================
st.sidebar.title("🧰 Apex Toolkit")
page = st.sidebar.radio(
    "Navigate",
    [
        "🗒️ Notes",
        "🧩 Snippets (syntax highlighting)",
        "📦 Multi-file Exporter",
        "⚙️ Backup / Restore",
        "🧾 XML Tools",
        "🔧 Settings",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Data is stored in-session. Use Backup/Restore or XML export for persistence.")

# =============================================================================
# Page: Notes
# =============================================================================
if page == "🗒️ Notes":
    st.title("🗒️ Notes")

    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        st.subheader("Your notes")
        query = st.text_input("Search (title/body/tags)", placeholder="e.g., meeting, TODO, idea…")
        tag_filter = st.text_input("Filter by tag (optional)", placeholder="e.g., research")

        filtered = st.session_state.notes
        if query.strip():
            q = query.strip().lower()
            filtered = [
                n for n in filtered
                if q in (n.get("title", "").lower())
                or q in (n.get("body", "").lower())
                or any(q in t.lower() for t in n.get("tags", []))
            ]
        if tag_filter.strip():
            tf = tag_filter.strip().lower()
            filtered = [n for n in filtered if any(tf == t.lower() for t in n.get("tags", []))]

        filtered = sorted(filtered, key=lambda x: x.get("updated_at", ""), reverse=True)

        colA, colB = st.columns(2)
        with colA:
            if st.button("➕ New", use_container_width=True):
                new_id = next_id("notes")
                st.session_state.notes.append({
                    "id": new_id,
                    "title": f"New note {new_id}",
                    "body": "",
                    "tags": [],
                    "created_at": now_stamp(),
                    "updated_at": now_stamp(),
                })
                st.session_state.active_note_id = new_id
                st.rerun()
        with colB:
            if st.button("📌 Pin active", use_container_width=True):
                # Minimal upgrade: pin just moves note to top by bumping updated_at
                n = get_by_id(st.session_state.notes, st.session_state.active_note_id)
                if n:
                    n["updated_at"] = now_stamp()
                    st.rerun()

        st.markdown("")
        if not filtered:
            st.info("No notes yet. Create one with **New**.")
        else:
            labels = [f"#{n['id']} — {n.get('title','(untitled)')}" for n in filtered]
            ids = [n["id"] for n in filtered]
            current = st.session_state.active_note_id
            default_index = ids.index(current) if current in ids else 0

            selected_label = st.selectbox("Select a note", labels, index=default_index)
            selected_id = ids[labels.index(selected_label)]
            st.session_state.active_note_id = selected_id

            note = get_by_id(st.session_state.notes, selected_id)
            if note:
                st.caption(f"Created: {note.get('created_at','')} · Updated: {note.get('updated_at','')}")
                if st.button("🗑️ Delete", use_container_width=True):
                    st.session_state.notes = delete_by_id(st.session_state.notes, selected_id)
                    st.session_state.active_note_id = st.session_state.notes[0]["id"] if st.session_state.notes else None
                    st.rerun()

    with right:
        note = get_by_id(st.session_state.notes, st.session_state.active_note_id)
        if not note:
            st.subheader("Editor")
            st.info("Select a note on the left or create a new one.")
        else:
            st.subheader("Editor")

            title = st.text_input("Title", value=note.get("title", ""))
            tags_raw = st.text_input("Tags (comma-separated)", value=", ".join(note.get("tags", [])))
            body = st.text_area("Body", value=note.get("body", ""), height=320, placeholder="Write your note…")

            c1, c2, c3, c4 = st.columns([0.18, 0.22, 0.22, 0.38])
            with c1:
                save = st.button("💾 Save", use_container_width=True)
            with c2:
                export_fmt = st.selectbox("Export", ["txt", "md", "json", "xml"], index=["txt","md","json","xml"].index(st.session_state.tool_settings.get("notes_default_export","txt")))
            with c3:
                dl = st.button("⬇️ Download", use_container_width=True)
            with c4:
                preview = st.checkbox("Preview markdown", value=False)

            # Autosave upgrade (simple): commit whenever inputs change and autosave enabled
            if st.session_state.tool_settings.get("autosave", False):
                note["title"] = title.strip() or "Untitled"
                note["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]
                note["body"] = body
                note["updated_at"] = now_stamp()

            if save:
                note["title"] = title.strip() or "Untitled"
                note["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]
                note["body"] = body
                note["updated_at"] = now_stamp()
                st.success("Saved.")

            if dl:
                note_obj = {
                    **note,
                    "title": title.strip() or "Untitled",
                    "tags": [t.strip() for t in tags_raw.split(",") if t.strip()],
                    "body": body,
                    "updated_at": now_stamp(),
                }
                if export_fmt == "txt":
                    data = note_to_txt(note_obj, include_meta=True)
                    fname = safe_filename(ensure_ext(note_obj["title"], ".txt"))
                    mime = "text/plain"
                elif export_fmt == "md":
                    data = note_to_md(note_obj, include_meta=True)
                    fname = safe_filename(ensure_ext(note_obj["title"], ".md"))
                    mime = "text/markdown"
                elif export_fmt == "json":
                    data = note_to_json(note_obj)
                    fname = safe_filename(ensure_ext(note_obj["title"], ".json"))
                    mime = "application/json"
                else:
                    data = note_to_xml(note_obj)
                    fname = safe_filename(ensure_ext(note_obj["title"], ".xml"))
                    mime = "application/xml"

                st.download_button(
                    "Click to download",
                    data=data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                )

            st.markdown("---")
            if preview:
                st.subheader("Preview")
                st.markdown(body if body.strip() else "_(empty)_")
            else:
                st.caption("Tip: enable **Preview markdown** to render note content.")

# =============================================================================
# Page: Snippets
# =============================================================================
elif page == "🧩 Snippets (syntax highlighting)":
    st.title("🧩 Snippets (syntax highlighting)")

    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        st.subheader("Your snippets")

        q = st.text_input("Search (name/content)", placeholder="e.g., pine, python, api…")
        filtered = st.session_state.snippets
        if q.strip():
            qq = q.strip().lower()
            filtered = [s for s in filtered if qq in (s.get("name", "").lower()) or qq in (s.get("content", "").lower())]
        filtered = sorted(filtered, key=lambda x: x.get("updated_at", ""), reverse=True)

        if st.button("➕ New snippet", use_container_width=True):
            new_id = next_id("snippets")
            st.session_state.snippets.append({
                "id": new_id,
                "name": f"snippet_{new_id}",
                "language": "python",
                "ext": ".py",
                "content": "",
                "created_at": now_stamp(),
                "updated_at": now_stamp(),
            })
            st.session_state.active_snippet_id = new_id
            st.rerun()

        st.markdown("")
        if not filtered:
            st.info("No snippets yet. Create one with **New snippet**.")
        else:
            labels = [f"#{s['id']} — {s.get('name','(unnamed)')} [{s.get('language','')}]"
                      for s in filtered]
            ids = [s["id"] for s in filtered]
            current = st.session_state.active_snippet_id
            default_index = ids.index(current) if current in ids else 0

            selected_label = st.selectbox("Select a snippet", labels, index=default_index)
            selected_id = ids[labels.index(selected_label)]
            st.session_state.active_snippet_id = selected_id

            snip = get_by_id(st.session_state.snippets, selected_id)
            if snip:
                st.caption(f"Created: {snip.get('created_at','')} · Updated: {snip.get('updated_at','')}")
                if st.button("🗑️ Delete snippet", use_container_width=True):
                    st.session_state.snippets = delete_by_id(st.session_state.snippets, selected_id)
                    st.session_state.active_snippet_id = st.session_state.snippets[0]["id"] if st.session_state.snippets else None
                    st.rerun()

    with right:
        snip = get_by_id(st.session_state.snippets, st.session_state.active_snippet_id)
        if not snip:
            st.subheader("Editor")
            st.info("Select a snippet on the left or create a new one.")
        else:
            st.subheader("Editor")
            name = st.text_input("Name", value=snip.get("name", "snippet"))

            colA, colB, colC = st.columns([0.4, 0.3, 0.3])
            with colA:
                language = st.selectbox(
                    "Language (preview)",
                    ["python", "javascript", "typescript", "json", "yaml", "bash", "html", "css", "sql", "markdown", "plaintext", "pine"],
                    index=(["python", "javascript", "typescript", "json", "yaml", "bash", "html", "css", "sql", "markdown", "plaintext", "pine"]
                           .index(snip.get("language", "python"))
                           if snip.get("language", "python") in ["python", "javascript", "typescript", "json", "yaml", "bash", "html", "css", "sql", "markdown", "plaintext", "pine"] else 0)
                )
            with colB:
                ext = st.text_input("File extension", value=snip.get("ext", ".txt"))
            with colC:
                export_mode = st.selectbox("Export mode", ["native", "json", "xml"],
                                           index=["native", "json", "xml"].index(st.session_state.tool_settings.get("snippets_default_export","native")))

            content = st.text_area("Code / text", value=snip.get("content", ""), height=280)

            c1, c2, c3 = st.columns([0.22, 0.22, 0.56])
            with c1:
                save = st.button("💾 Save", use_container_width=True)
            with c2:
                download = st.button("⬇️ Download", use_container_width=True)
            with c3:
                st.caption("Preview uses Streamlit’s renderer (st.code).")

            if st.session_state.tool_settings.get("autosave", False):
                snip["name"] = name.strip() or "snippet"
                snip["language"] = language
                snip["ext"] = ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
                snip["content"] = content
                snip["updated_at"] = now_stamp()

            if save:
                snip["name"] = name.strip() or "snippet"
                snip["language"] = language
                snip["ext"] = ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
                snip["content"] = content
                snip["updated_at"] = now_stamp()
                st.success("Saved.")

            if download:
                if export_mode == "native":
                    data = content
                    fname = safe_filename(ensure_ext(snip["name"], snip.get("ext", ".txt")))
                    mime = "text/plain"
                elif export_mode == "json":
                    data = snippet_to_json(snip)
                    fname = safe_filename(ensure_ext(snip["name"], ".json"))
                    mime = "application/json"
                else:
                    data = snippet_to_xml(snip)
                    fname = safe_filename(ensure_ext(snip["name"], ".xml"))
                    mime = "application/xml"

                st.download_button(
                    "Click to download",
                    data=data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                )

            st.markdown("---")
            st.subheader("Preview (syntax highlighted)")
            preview_lang = language if language != "plaintext" else None
            st.code(content, language=preview_lang)

# =============================================================================
# Page: Multi-file Exporter (ZIP) + XML options (UPGRADED)
# =============================================================================
elif page == "📦 Multi-file Exporter":
    st.title("📦 Multi-file Exporter")

    st.markdown("Export any selection of notes/snippets as files and download as a ZIP.")

    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.subheader("Notes")
        if not st.session_state.notes:
            st.info("No notes available.")
            selected_notes = []
        else:
            note_options = {f"#{n['id']} — {n.get('title','Untitled')}": n["id"] for n in st.session_state.notes}
            selected_note_labels = st.multiselect("Select notes", list(note_options.keys()))
            selected_notes = [note_options[lbl] for lbl in selected_note_labels]

        note_export_format = st.selectbox("Notes export format", ["txt", "md", "json", "xml"],
                                          index=["txt","md","json","xml"].index(st.session_state.tool_settings.get("notes_default_export","txt")))
        include_note_meta = st.checkbox("Include note metadata (for txt/md)", value=True)

    with col2:
        st.subheader("Snippets")
        if not st.session_state.snippets:
            st.info("No snippets available.")
            selected_snips = []
        else:
            snip_options = {f"#{s['id']} — {s.get('name','snippet')} ({s.get('ext','.txt')})": s["id"] for s in st.session_state.snippets}
            selected_snip_labels = st.multiselect("Select snippets", list(snip_options.keys()))
            selected_snips = [snip_options[lbl] for lbl in selected_snip_labels]

        snippet_export_mode = st.selectbox("Snippets export mode", ["native", "json", "xml"],
                                           index=["native","json","xml"].index(st.session_state.tool_settings.get("snippets_default_export","native")))

    st.markdown("---")
    export_folder = st.text_input("ZIP folder name", value=st.session_state.tool_settings.get("zip_folder", "apex_export"))

    # New: include whole-workspace exports
    st.subheader("Workspace bundle (optional)")
    include_workspace_json = st.checkbox("Include workspace backup JSON inside ZIP", value=True)
    include_workspace_xml = st.checkbox("Include workspace export XML inside ZIP", value=True)

    files_to_zip: List[Dict[str, str]] = []

    # Notes
    for nid in selected_notes:
        n = get_by_id(st.session_state.notes, nid)
        if not n:
            continue
        title = n.get("title", "Untitled")
        base = safe_filename(title)

        if note_export_format == "txt":
            data = note_to_txt(n, include_meta=include_note_meta)
            fname = ensure_ext(base, ".txt")
        elif note_export_format == "md":
            data = note_to_md(n, include_meta=include_note_meta)
            fname = ensure_ext(base, ".md")
        elif note_export_format == "json":
            data = note_to_json(n)
            fname = ensure_ext(base, ".json")
        else:
            data = note_to_xml(n)
            fname = ensure_ext(base, ".xml")

        path = f"{export_folder}/notes/{fname}" if export_folder.strip() else f"notes/{fname}"
        files_to_zip.append({"path": path, "data": data})

    # Snippets
    for sid in selected_snips:
        s = get_by_id(st.session_state.snippets, sid)
        if not s:
            continue
        name = s.get("name", "snippet")
        base = safe_filename(name)

        if snippet_export_mode == "native":
            data = s.get("content", "")
            fname = ensure_ext(base, s.get("ext", ".txt"))
        elif snippet_export_mode == "json":
            data = snippet_to_json(s)
            fname = ensure_ext(base, ".json")
        else:
            data = snippet_to_xml(s)
            fname = ensure_ext(base, ".xml")

        path = f"{export_folder}/snippets/{fname}" if export_folder.strip() else f"snippets/{fname}"
        files_to_zip.append({"path": path, "data": data})

    # Workspace bundle files
    if include_workspace_json:
        files_to_zip.append({
            "path": f"{export_folder}/workspace/apex_toolkit_backup.json" if export_folder.strip() else "workspace/apex_toolkit_backup.json",
            "data": export_state_as_json(),
        })
    if include_workspace_xml:
        files_to_zip.append({
            "path": f"{export_folder}/workspace/apex_toolkit_export.xml" if export_folder.strip() else "workspace/apex_toolkit_export.xml",
            "data": export_state_as_xml(),
        })

    if files_to_zip:
        zip_bytes = build_zip(files_to_zip)
        zip_name = safe_filename(ensure_ext(export_folder or "apex_export", ".zip"))
        st.download_button(
            "⬇️ Download ZIP",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )
        st.caption(f"Files in ZIP: {len(files_to_zip)}")
        with st.expander("View ZIP manifest"):
            for f in files_to_zip:
                st.write(f"- {f['path']}")
    else:
        st.info("Select at least one note or snippet, or include a workspace bundle.")

# =============================================================================
# Page: Backup / Restore (JSON)
# =============================================================================
elif page == "⚙️ Backup / Restore":
    st.title("⚙️ Backup / Restore (JSON)")

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.subheader("Backup JSON")
        backup_json = export_state_as_json()
        st.download_button(
            "⬇️ Download backup JSON",
            data=backup_json,
            file_name=safe_filename(ensure_ext(f"apex_toolkit_backup_{now_stamp_file()}", ".json")),
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("Preview JSON"):
            st.code(backup_json, language="json")

    with colB:
        st.subheader("Restore JSON")
        uploaded = st.file_uploader("Upload backup JSON", type=["json"])
        if uploaded is not None:
            try:
                text = uploaded.read().decode("utf-8")
                restored = import_state_from_json(text)
                if st.button("✅ Restore JSON into session", use_container_width=True):
                    st.session_state.notes = restored["notes"]
                    st.session_state.snippets = restored["snippets"]
                    if restored.get("settings"):
                        st.session_state.tool_settings.update(restored["settings"])
                    st.session_state.active_note_id = st.session_state.notes[0]["id"] if st.session_state.notes else None
                    st.session_state.active_snippet_id = st.session_state.snippets[0]["id"] if st.session_state.snippets else None
                    st.success("Restore complete.")
                    st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")

    st.markdown("---")
    st.subheader("Danger zone")
    if st.button("🧨 Clear all notes & snippets", use_container_width=True):
        st.session_state.notes = []
        st.session_state.snippets = []
        st.session_state.active_note_id = None
        st.session_state.active_snippet_id = None
        st.success("Cleared.")
        st.rerun()

# =============================================================================
# Page: XML Tools (NEW)
# =============================================================================
elif page == "🧾 XML Tools":
    st.title("🧾 XML Tools")

    st.markdown(
        "Export or restore your entire workspace using XML. "
        "This is useful when you want a structured, human-auditable format."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Export XML")
        xml_text = export_state_as_xml()
        st.download_button(
            "⬇️ Download workspace XML",
            data=xml_text,
            file_name=safe_filename(ensure_ext(f"apex_toolkit_export_{now_stamp_file()}", ".xml")),
            mime="application/xml",
            use_container_width=True,
        )
        with st.expander("Preview XML"):
            st.code(xml_text, language="xml")

    with col2:
        st.subheader("Restore XML")
        uploaded = st.file_uploader("Upload workspace XML", type=["xml"], key="xml_uploader")
        if uploaded is not None:
            try:
                text = uploaded.read().decode("utf-8")
                restored = import_state_from_xml(text)
                if st.button("✅ Restore XML into session", use_container_width=True):
                    st.session_state.notes = restored["notes"]
                    st.session_state.snippets = restored["snippets"]
                    if restored.get("settings"):
                        st.session_state.tool_settings.update(restored["settings"])
                    st.session_state.active_note_id = st.session_state.notes[0]["id"] if st.session_state.notes else None
                    st.session_state.active_snippet_id = st.session_state.snippets[0]["id"] if st.session_state.snippets else None
                    st.success("XML restore complete.")
                    st.rerun()
            except Exception as e:
                st.error(f"XML restore failed: {e}")

    st.markdown("---")
    st.subheader("Per-item XML exports")
    colA, colB = st.columns(2, gap="large")
    with colA:
        st.caption("Export a single note to XML")
        if st.session_state.notes:
            note_id = st.selectbox("Note", [n["id"] for n in st.session_state.notes], format_func=lambda i: f"#{i} — {get_by_id(st.session_state.notes, i).get('title','Untitled')}")
            n = get_by_id(st.session_state.notes, note_id)
            if n:
                st.download_button(
                    "⬇️ Download note.xml",
                    data=note_to_xml(n),
                    file_name=safe_filename(ensure_ext(n.get("title","note"), ".xml")),
                    mime="application/xml",
                    use_container_width=True,
                )
        else:
            st.info("No notes available.")
    with colB:
        st.caption("Export a single snippet to XML")
        if st.session_state.snippets:
            snip_id = st.selectbox("Snippet", [s["id"] for s in st.session_state.snippets], format_func=lambda i: f"#{i} — {get_by_id(st.session_state.snippets, i).get('name','snippet')}")
            s = get_by_id(st.session_state.snippets, snip_id)
            if s:
                st.download_button(
                    "⬇️ Download snippet.xml",
                    data=snippet_to_xml(s),
                    file_name=safe_filename(ensure_ext(s.get("name","snippet"), ".xml")),
                    mime="application/xml",
                    use_container_width=True,
                )
        else:
            st.info("No snippets available.")

# =============================================================================
# Page: Settings (UPGRADES)
# =============================================================================
elif page == "🔧 Settings":
    st.title("🔧 Settings")

    st.subheader("Behavior")
    st.session_state.tool_settings["autosave"] = st.checkbox(
        "Enable autosave (updates session data as you type)",
        value=st.session_state.tool_settings.get("autosave", False),
        help="This is an in-session autosave. Use Backup/Restore or XML export for persistence."
    )

    st.subheader("Defaults")
    st.session_state.tool_settings["notes_default_export"] = st.selectbox(
        "Default note export format",
        ["txt", "md", "json", "xml"],
        index=["txt","md","json","xml"].index(st.session_state.tool_settings.get("notes_default_export","txt")),
    )
    st.session_state.tool_settings["snippets_default_export"] = st.selectbox(
        "Default snippet export mode",
        ["native", "json", "xml"],
        index=["native","json","xml"].index(st.session_state.tool_settings.get("snippets_default_export","native")),
    )
    st.session_state.tool_settings["zip_folder"] = st.text_input(
        "Default ZIP folder name",
        value=st.session_state.tool_settings.get("zip_folder", "apex_export"),
    )

    st.markdown("---")
    st.subheader("Quick actions")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🔁 Reset settings to defaults", use_container_width=True):
            st.session_state.tool_settings = {
                "autosave": False,
                "autosave_seconds": 0,
                "notes_default_export": "txt",
                "snippets_default_export": "native",
                "zip_folder": "apex_export",
            }
            st.success("Settings reset.")
            st.rerun()
    with colB:
        if st.button("✅ Validate session data", use_container_width=True):
            # Minimal validation upgrade
            issues = []
            for n in st.session_state.notes:
                if "id" not in n: issues.append("A note is missing 'id'.")
                if "title" not in n: issues.append(f"Note #{n.get('id','?')} missing 'title'.")
            for s in st.session_state.snippets:
                if "id" not in s: issues.append("A snippet is missing 'id'.")
                if "name" not in s: issues.append(f"Snippet #{s.get('id','?')} missing 'name'.")
            if issues:
                st.error("Validation issues:\n- " + "\n- ".join(issues))
            else:
                st.success("Looks good.")

# =============================================================================
# Page: About
# =============================================================================
else:
    st.title("ℹ️ About")
    st.markdown(
        """
**Apex Toolkit** is a single-file Streamlit app that includes:

- **Notes**: note-taking micro-app (search, tags, markdown preview, downloads)
- **Snippets**: store snippets with **syntax-highlighted previews**
- **Multi-file Exporter**: export selected items to a **ZIP**
- **Backup/Restore**: full workspace persistence via **JSON**
- **XML Tools**: full workspace export/restore via **XML** (plus per-item XML)

**Run locally**
1. Save as `app.py`
2. Install: `pip install streamlit`
3. Run: `streamlit run app.py`
        """
    )
