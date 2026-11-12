"""Visual theme: the approved design tokens, applied as CSS on top of the Streamlit light theme."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGE_ICON = Path(__file__).resolve().parent / "assets" / "app_icon.png"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ground: #F4EFE6; --paper: #FBF8F2; --ink: #1F1B16; --ink-muted: #6B6358;
  --rule: #D9D0C1; --accent: #B45309; --accent-soft: #F3E4CC;
  --ok: #4D6B3A; --danger: #9B2C2C;
}
.stApp { background: var(--ground); color: var(--ink); font-family: 'Source Sans 3', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; font-weight: 400; color: var(--ink); }
[data-testid="stSidebar"] { background: var(--ground); border-right: 1px solid var(--rule); }
.lrag-brand { font-family: 'Fraunces', Georgia, serif; font-size: 20px; font-weight: 600; margin-bottom: 8px; }
.lrag-brand span { color: var(--accent); }
.lrag-label { font-size: 13px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: .08em; }
.lrag-muted { color: var(--ink-muted); font-size: 14px; }
.lrag-mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 13px; color: var(--ink-muted); }
.lrag-chip {
  display: inline-block; font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12px;
  border: 1px solid var(--rule); background: var(--paper); border-radius: 4px;
  padding: 2px 8px; margin: 0 6px 6px 0; color: var(--ink-muted);
}
.lrag-chip b { color: var(--accent); font-weight: 500; }
.lrag-error {
  border: 1px solid var(--danger); border-left: 4px solid var(--danger);
  background: var(--paper); border-radius: 10px; padding: 16px 20px; margin: 8px 0;
}
.lrag-error h4 { font-family: 'Fraunces', Georgia, serif; font-weight: 400; color: var(--danger); margin: 0 0 6px; }
.lrag-key-set { color: var(--ok); }
.lrag-key-missing { color: var(--danger); }
"""


def apply_theme() -> None:
    """Inject the design tokens as CSS for the current page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
