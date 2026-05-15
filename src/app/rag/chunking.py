"""
Markdown-aware chunking for the NileTel knowledge base.

Splits each .md file along header boundaries, prepending a
breadcrumb like `[file.md > Section > Subsection]` to every chunk
so the embedding model sees the full hierarchical context.
"""
from __future__ import annotations 

import re
from dataclasses import dataclass , field 
from pathlib import Path 

@dataclass 
class Chunk:
    text : str 
    raw_text : str 
    source : str 
    heading_path : list[str] = field(default_factory=list)
    chunk_index: int = 0
    
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CODE_FENCE_RE = re.compile(r"^```")

def _parse_blocks(text: str ) -> list[tuple[list[str],str]]:
    """Yield (heading_path, content) tuples, in document order."""
    blocks : list[tuple[list[str],str]] = []
    heading_stack : list[tuple[int,str]] = []
    buffer : list[str]= []
    in_code = False 
    
    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            path = [title for _,title in heading_stack]
            blocks.append((path, content))
        buffer.clear()
    
    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_code = not in_code
            buffer.append(line)
            continue
        
        if not in_code:
            m = _HEADER_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                continue

        buffer.append(line)

    flush()
    return blocks
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _pack(parts: list[str], max_chars: int) -> list[str]:
    """Greedily concatenate parts (with separators) into <=max_chars groups."""
    out: list[str] = []
    cur = ""
    for p in parts:
        if not p:
            continue
        sep = "\n\n" if cur else ""
        if len(cur) + len(sep) + len(p) <= max_chars:
            cur = cur + sep + p
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out

def _split_long_block(content: str, max_chars: int) -> list[str]:
    if len(content) <= max_chars:
        return [content]

    paragraphs = re.split(r"\n\s*\n", content.strip())
    packed = _pack(paragraphs, max_chars)

    # If any single packed piece is still too long, drop to sentence level
    final: list[str] = []
    for piece in packed:
        if len(piece) <= max_chars:
            final.append(piece)
            continue
        sentences = _SENTENCE_RE.split(piece)
        sub_packed = _pack(sentences, max_chars)
        # Last resort: hard split
        for sp in sub_packed:
            if len(sp) <= max_chars:
                final.append(sp)
            else:
                for i in range(0, len(sp), max_chars):
                    final.append(sp[i : i + max_chars])
    return final


def _format_with_breadcrumb(source: str, heading_path: list[str], raw: str) -> str:
    crumbs = " > ".join([source] + heading_path) if heading_path else source
    return f"[{crumbs}]\n\n{raw}"



def chunk_markdown_file(path: Path, max_chars: int = 700) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    source = path.name
    blocks = _parse_blocks(text)

    # Fallback: file with no headers at all
    if not blocks:
        raw_paragraphs = re.split(r"\n\s*\n", text.strip())
        blocks = [([], "\n\n".join(raw_paragraphs).strip())] if raw_paragraphs else []

    # Merge tiny adjacent siblings
    merged: list[tuple[list[str], str]] = []
    i = 0
    while i < len(blocks):
        path_i, content_i = blocks[i]
        j = i + 1
        while j < len(blocks):
            path_j, content_j = blocks[j]
            same_parent = (
                len(path_i) == len(path_j)
                and path_i[:-1] == path_j[:-1]
            )
            combined_len = len(content_i) + 2 + len(content_j)
            if same_parent and combined_len <= max_chars:
                content_i = content_i + "\n\n" + content_j
                j += 1
            else:
                break
        merged_path = path_i[:-1] if j > i + 1 else path_i
        merged.append((merged_path, content_i))
        i = j

    # Split + emit
    chunks: list[Chunk] = []
    idx = 0
    for path_, content in merged:
        for piece in _split_long_block(content, max_chars):
            chunks.append(
                Chunk(
                    text=_format_with_breadcrumb(source, path_, piece),
                    raw_text=piece,
                    source=source,
                    heading_path=path_,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks

def chunk_directory(data_dir: Path, max_chars: int = 700) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        all_chunks.extend(chunk_markdown_file(path, max_chars=max_chars))
    return all_chunks

