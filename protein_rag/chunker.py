"""
STAGES 1 + 2 of RAG: LOAD the knowledge files and CHUNK them.

Why chunk?  An LLM prompt has limited room, and a search that returns a whole
file is useless.  We cut each markdown file into small passages ("chunks")
so retrieval can return exactly the paragraph that answers the question.

Strategy: split on "## " headings first (each heading is one idea), then, if a
section is still long, pack paragraphs up to MAX_CHARS with a small overlap so
a sentence cut at a boundary still appears whole in one of the two chunks.
"""
import os
import re

MAX_CHARS = 900     # ~150 words. Small enough to be focused, big enough to be self-contained.
OVERLAP_CHARS = 150


def load_documents(folder):
    """Return [(filename, text), ...] for every .md/.txt file in `folder`."""
    docs = []
    if not os.path.isdir(folder):
        return docs
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".md", ".txt")):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                docs.append((name, f.read().replace("\r\n", "\n")))
    return docs


def split_markdown(text):
    """-> (doc_title, [(section_title, body), ...]) using '# ' and '## ' headings."""
    doc_title = ""
    sections, cur_title, cur_lines = [], "Overview", []
    for line in text.split("\n"):
        if line.startswith("# ") and not doc_title:
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            if "".join(cur_lines).strip():
                sections.append((cur_title, "\n".join(cur_lines).strip()))
            cur_title, cur_lines = line[3:].strip(), []
        else:
            cur_lines.append(line)
    if "".join(cur_lines).strip():
        sections.append((cur_title, "\n".join(cur_lines).strip()))
    return doc_title, sections


def _sentences(paragraph):
    return [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s]


def chunk_section(body, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    """Pack paragraphs/sentences into pieces <= max_chars, with overlap."""
    if len(body) <= max_chars:
        return [body]
    units = []
    for para in [p for p in body.split("\n\n") if p.strip()]:
        units.extend(_sentences(para) if len(para) > max_chars else [para])
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = cur[-overlap:].split(" ", 1)[-1] + " " + u   # carry a tail of the previous chunk
        else:
            cur = (cur + " " + u).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def load_chunks(folder):
    """The whole LOAD + CHUNK stage. Returns a list of chunk dicts."""
    chunks = []
    for fname, text in load_documents(folder):
        doc_title, sections = split_markdown(text)
        stem = os.path.splitext(fname)[0]
        n = 0
        for sec_title, body in sections:
            for piece in chunk_section(body):
                n += 1
                chunks.append({
                    "id": f"{stem}#{n}",
                    "source": fname,
                    "title": f"{doc_title} > {sec_title}" if doc_title else sec_title,
                    "text": piece,
                })
    return chunks
