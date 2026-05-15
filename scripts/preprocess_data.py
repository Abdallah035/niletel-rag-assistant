"""
Promote NileTel knowledge-base files from plain-text section labels into
proper markdown headers, so the chunker can split on them.

Promotion rules (conservative, idempotent):
  1. Title block at the top of file:
       - First non-empty line  -> "# <line>"
       - Second non-empty line (if it looks like an Arabic subtitle) -> "## <line>"
  2. Section labels (a line on its own that ends with ":"):
       - The text before ":" must be <= 80 chars
       - Promote to "## <line without trailing colon>"
  3. Footer metadata lines starting with "Last Updated:" / "Version:" /
     containing "|" pipe-separated fields are left untouched.
  4. Already-headered lines (starting with "#") are skipped.
  5. Lines inside ``` code fences ``` are skipped.

Usage:
    python scripts/preprocess_data.py --dry-run   (default — prints diff, writes nothing)
    python scripts/preprocess_data.py --apply     (overwrites files in data/)
    python scripts/preprocess_data.py --apply --backup  (also writes data/<file>.bak)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from difflib import unified_diff

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CODE_FENCE_RE = re.compile(r"^```")
ARABIC_RE = re.compile(r"[؀-ۿ]")

# Lines that look like footer metadata — skip these
FOOTER_PREFIXES = ("Last Updated", "Version", "Department")


def looks_like_footer(line: str) -> bool:
    s = line.strip()
    if any(s.startswith(p) for p in FOOTER_PREFIXES):
        return True
    if "|" in s and any(p + ":" in s for p in FOOTER_PREFIXES):
        return True
    return False


def is_section_label(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if s.startswith(("-", "*", "•")):
        return False
    if re.match(r"^\d+[.)]\s", s):
        return False
    if not s.endswith(":"):
        return False
    if looks_like_footer(s):
        return False
    body = s[:-1].strip()
    if not body or len(body) > 80:
        return False
    if not (re.search(r"[A-Za-z]", body) or ARABIC_RE.search(body)):
        return False
    return True


def promote_file(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_code = False
    title_promoted = False
    subtitle_promoted = False
    seen_non_empty = 0

    for line in lines:
        if CODE_FENCE_RE.match(line):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        stripped = line.strip()

        # Skip blank lines but track them
        if not stripped:
            out.append(line)
            continue

        seen_non_empty += 1

        # Already a header — leave it
        if stripped.startswith("#"):
            out.append(line)
            if seen_non_empty == 1:
                title_promoted = True
            continue

        # First non-empty line -> "# title"
        if seen_non_empty == 1 and not title_promoted:
            out.append("# " + stripped)
            title_promoted = True
            continue

        # Second non-empty line -> "## subtitle" if it contains Arabic
        # (typical NileTel pattern: English title, Arabic subtitle)
        if (
            seen_non_empty == 2
            and title_promoted
            and not subtitle_promoted
            and ARABIC_RE.search(stripped)
            and not stripped.endswith(":")  # don't double-promote a section label
        ):
            out.append("## " + stripped)
            subtitle_promoted = True
            continue

        # Section labels ending in ":"
        if is_section_label(line):
            body = stripped.rstrip(":").rstrip()
            out.append("## " + body)
            continue

        out.append(line)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def show_diff(name: str, before: str, after: str) -> None:
    diff = list(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
            n=2,
        )
    )
    if not diff:
        return
    print(f"\n----- {name} -----")
    for d in diff:
        sys.stdout.write(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes to disk")
    ap.add_argument("--backup", action="store_true", help="Write .bak files alongside originals")
    ap.add_argument("--dry-run", action="store_true", help="Print diff only (default)")
    args = ap.parse_args()

    files = sorted(DATA_DIR.glob("*.md"))
    changed = 0
    total = 0

    for path in files:
        before = path.read_text(encoding="utf-8")
        after = promote_file(before)
        if before == after:
            continue
        changed += 1
        total += 1
        show_diff(path.name, before, after)
        if args.apply:
            if args.backup:
                bak = path.with_suffix(path.suffix + ".bak")
                bak.write_text(before, encoding="utf-8")
            path.write_text(after, encoding="utf-8")

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {changed}/{len(files)} files would change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
