#!/usr/bin/env python
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "overleaf_microplastic_project" / "main.tex"
PDF = ROOT / "overleaf_microplastic_project" / "main.pdf"
REQUIRED_SECTIONS = [
    "Introduction",
    "Materials and Methods",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    tex = PAPER.read_text(encoding="utf-8")
    include = PAPER.parent / "c2_current_results.tex"
    if include.exists():
        include_tex = include.read_text(encoding="utf-8")
        tex = tex.replace(r"\input{c2_current_results}", include_tex)
    body_tex = tex.split(r"\appendix", 1)[0]
    top_sections = re.findall(r"^\\section\{([^}]+)\}", body_tex, flags=re.MULTILINE)
    if top_sections != REQUIRED_SECTIONS:
        fail(f"unexpected top-level section order: {top_sections}")
    for section in REQUIRED_SECTIONS:
        if f"\\section{{{section}}}" not in tex:
            fail(f"missing section: {section}")
    if len(re.findall(r"\\bibitem\{", tex)) < 15:
        fail("expected at least 15 references")
    if len(re.findall(r"\\begin\{figure\}", tex)) < 3:
        fail("expected at least 3 figures")
    if len(re.findall(r"\\begin\{table\}", tex)) < 3:
        fail("expected at least 3 tables")
    forbidden = ["PLACEHOLDER", "XX.X", "0.XX"]
    for token in forbidden:
        if token in tex:
            fail(f"found unresolved placeholder token: {token}")

    if not PDF.exists():
        fail("paper.pdf does not exist")
    pdfinfo = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
    match = re.search(r"^Pages:\s+(\d+)", pdfinfo, re.MULTILINE)
    if not match:
        fail("could not read page count")
    pages = int(match.group(1))
    if not 10 <= pages <= 15:
        fail(f"page count outside requested range: {pages}")

    text = subprocess.check_output(["pdftotext", str(PDF), "-"], text=True)
    for expected in [
        "Results",
        "Held-Out Ecological Test Performance",
        "Real + inpainting",
        "U-Net++",
        "SegFormer-B2",
    ]:
        if expected not in text:
            fail(f"expected rendered text not visible: {expected}")
    words = len(text.split())
    references = len(re.findall(r"\\bibitem\{", tex))
    print(f"OK: {pages} pages, {words} rendered words, {references} references")


if __name__ == "__main__":
    main()
