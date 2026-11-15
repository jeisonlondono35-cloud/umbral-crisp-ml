"""English / Spanish for the whole site (the button EN | ES at the top right).

The pages are written in English (the language of the course deliverables). When the visitor chooses Spanish, every
HTML response is translated on the server, text segment by text segment, with the table in app/i18n/es.json:
  * a segment is the text between two tags (whitespace collapsed); <script>, <style>, <pre>, <code> and <textarea>
    are never touched, and neither are the values typed by users (they are not in the table);
  * numbers inside a segment are replaced by {n} in the key, so "20 of 39 next actions" and "7 of 40 next actions"
    use the same entry, and the numbers are put back in the same order;
  * placeholder="..." and title="..." attributes are translated the same way.
To add texts: python tools/i18n_extract.py (lists the segments of every page that are not in the table yet).
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).resolve().parent / "i18n" / "es.json"
LANGS = ("en", "es")
COOKIE = "lang"

_SKIP = re.compile(r"<(script|style|pre|code|textarea)\b.*?</\1\s*>", re.S | re.I)
_TEXT = re.compile(r">([^<>]+)<")
_ATTR = re.compile(r'\b(placeholder|title|aria-label)="([^"<>]+)"')
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[A-Za-z]{2,}")


def key_of(text: str) -> tuple[str, list[str]]:
    """Normalized key (spaces collapsed, numbers -> {n}) and the numbers that were taken out."""
    t = " ".join(html.unescape(text).split())
    nums = _NUM.findall(t)
    return _NUM.sub("{n}", t), nums


@lru_cache(maxsize=1)
def table() -> dict:
    try:
        return json.loads(TABLE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _tr(text: str, tab: dict, attr: bool = False) -> str | None:
    if not _WORD.search(text):
        return None
    k, nums = key_of(text)
    es = tab.get(k)
    if es is None:
        return None
    it = iter(nums)
    es = html.escape(re.sub(r"\{n\}", lambda m: next(it, "{n}"), es), quote=attr)
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    return f"{lead}{es}{trail}"


def translate_html(html: str, tab: dict | None = None) -> str:
    tab = table() if tab is None else tab
    if not tab:
        return html
    out, pos = [], 0
    for m in _SKIP.finditer(html):                       # translate only outside the protected blocks
        out.append(_translate_part(html[pos:m.start()], tab))
        out.append(m.group(0))
        pos = m.end()
    out.append(_translate_part(html[pos:], tab))
    return "".join(out).replace('<html lang="en"', '<html lang="es"', 1)


def _translate_part(part: str, tab: dict) -> str:
    # the edges of a part touch a protected block (<code>...): close them with > and < so that text is a segment too
    part = _TEXT.sub(lambda m: ">" + (_tr(m.group(1), tab) or m.group(1)) + "<", ">" + part + "<")[1:-1]
    return _ATTR.sub(lambda m: f'{m.group(1)}="{_tr(m.group(2), tab, attr=True) or m.group(2)}"', part)


def segments(html: str) -> list[str]:
    """Every translatable segment of a page (for tools/i18n_extract.py)."""
    html = _SKIP.sub("<x>", html)                     # a protected block splits the text, as in translate_html
    found = [m.group(1) for m in _TEXT.finditer(html)] + [m.group(2) for m in _ATTR.finditer(html)]
    return [key_of(t)[0] for t in found if _WORD.search(t)]
