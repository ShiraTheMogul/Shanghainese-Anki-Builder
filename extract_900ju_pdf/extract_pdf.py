#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_pdf_sentences.py

Extract (Shanghainese, Romanisation, Mandarin) triples from the provided PDF
and output a CSV (utf-8-sig) ready for the TTS->Anki pipeline.

Fixes the common “double-layer PDF text” issue (我我也也…),
by de-duplicating pairwise-duplicated glyph streams.

Usage:
  pip install pdfplumber pandas
  python extract_pdf_sentences.py --pdf "/path/to/900-sentences.pdf" --out "sentences.csv"
  # optionally:
  python extract_pdf_sentences.py --pdf "/path/to/900-sentences.pdf" --out "sentences.csv" --max-pages 3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pdfplumber


# =========================
# CJK regex (drop-in, per your ranges)
# =========================
_CJK_RE = re.compile(
    r"["
    r"\u3400-\u4DBF"          # Extension A
    r"\u4E00-\u9FFF"          # Unified Ideographs
    r"\U00020000-\U0002A6DF"  # Extension B
    r"\U0002A700-\U0002B73F"  # Extension C
    r"\U0002B740-\U0002B81D"  # Extension D
    r"\U0002B820-\U0002CEAD"  # Extension E
    r"\U0002CEB0-\U0002EBE0"  # Extension F
    r"\U00031350-\U000323AF"  # Extension H
    r"\U0002EBF0-\U0002EE5D"  # Extension I
    r"\U000323B0-\U00033479"  # Extension J
    r"\U0002F800-\U0002FA1F"  # Compatibility Supplement
    r"]"
)


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for c in chars if _CJK_RE.match(c))
    return cjk / len(chars)


# =========================
# Line normalisation + de-dup
# =========================
_PAGE_FRAC_RE = re.compile(r"\b\d+\s*/\s*\d+\b")

def normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", str(line)).strip()
    line = _PAGE_FRAC_RE.sub("", line).strip()
    return line

def undouble_text(s: str) -> str:
    """
    Fix pairwise duplicate glyph extraction:
      我我也也常常常常 -> 我也常常
    Also handles whole-string duplication:
      ABCABC -> ABC
    Removes internal whitespace.
    """
    s = normalize_line(s)
    compact = re.sub(r"\s+", "", s)
    n = len(compact)
    if n >= 2 and n % 2 == 0:
        half = n // 2
        # pairwise: chars at even == chars at odd
        if compact[0::2] == compact[1::2]:
            return compact[0::2]
        # whole-string doubled
        if compact[:half] == compact[half:]:
            return compact[:half]
    return compact


# =========================
# Romanisation detection (Latin + tone digits, low CJK)
# =========================
_ROMA_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")

def is_romanisation_line(line: str) -> bool:
    line = normalize_line(line)
    if len(line) < 6:
        return False
    if not _ROMA_RE.search(line):
        return False
    if not _DIGIT_RE.search(line):
        return False
    # should have almost no CJK
    if cjk_ratio(line) > 0.05:
        return False
    # avoid headings by requiring some alphanum bulk
    if sum(ch.isalnum() for ch in line) < 6:
        return False
    return True

def is_candidate_chinese(line: str) -> bool:
    line = undouble_text(line)
    if not line:
        return False
    return cjk_ratio(line) >= 0.45


@dataclass
class Triple:
    shanghainese: str
    romanisation: str
    mandarin: str
    page: int


def extract_triples_from_page(lines: List[str], page_no: int) -> List[Triple]:
    triples: List[Triple] = []
    norm = [normalize_line(x) for x in lines]

    for i, line in enumerate(norm):
        if not line:
            continue
        if not is_romanisation_line(line):
            continue

        # nearest chinese above = Shanghainese
        sh: Optional[str] = None
        for j in range(i - 1, max(-1, i - 12), -1):
            if j < 0:
                break
            if is_candidate_chinese(norm[j]):
                sh = undouble_text(norm[j])
                break

        # nearest chinese below = Mandarin
        mn: Optional[str] = None
        for j in range(i + 1, min(len(norm), i + 12)):
            if is_candidate_chinese(norm[j]):
                mn = undouble_text(norm[j])
                break

        if sh and mn and sh != mn:
            triples.append(Triple(shanghainese=sh, romanisation=normalize_line(line), mandarin=mn, page=page_no))

    # de-dup within page
    seen = set()
    out: List[Triple] = []
    for t in triples:
        k = (t.shanghainese, t.romanisation, t.mandarin)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def extract_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> List[Triple]:
    triples: List[Triple] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        end = len(pdf.pages) if max_pages is None else min(len(pdf.pages), max_pages)
        for idx in range(end):
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            lines = [x for x in text.splitlines() if x.strip()]
            triples.extend(extract_triples_from_page(lines, page_no=idx + 1))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--out", default="sentences.csv", help="Output CSV (utf-8-sig)")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit pages for testing")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    triples = extract_pdf(pdf_path, max_pages=args.max_pages)

    df = pd.DataFrame([{
        "shanghainese": t.shanghainese,
        "romanisation_pdf": t.romanisation,
        "mandarin": t.mandarin,
        "page": t.page,
    } for t in triples])

    df = df[df["shanghainese"].str.len() >= 2]
    df = df[df["mandarin"].str.len() >= 2]
    df = df.drop_duplicates(subset=["shanghainese", "romanisation_pdf", "mandarin"])

    out_path = Path(args.out)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Extracted {len(df)} triples -> {out_path}")
    if len(df) > 0:
        print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
