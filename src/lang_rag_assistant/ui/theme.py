"""Theme assets: the app icon and the stylesheet rules config cannot express.

Colours, fonts and the page background live in `.streamlit/config.toml`. The active
sidebar item takes the wireframe's accent, which no config option sets, so it is styled
here. The sidebar collapse toggle is hidden because the app has four menu items and the
wireframes show no toggle.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGE_ICON = Path(__file__).resolve().parent / "assets" / "app_icon.png"
# Sidebar logo: app_icon.png with the title drawn beside it in Inter Bold 16px, #1F1B16, 10px gap.
SIDEBAR_LOGO = Path(__file__).resolve().parent / "assets" / "app_logo.png"

CSS = """
/* Sidebar: the four menu items only; no collapse or expand toggle. */
[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {
    display: none;
}

/* Sidebar footer: pinned to the bottom of the sidebar on every page. */
.lrag-footer {
    position: fixed;
    bottom: 16px;
    left: 30px;
    font-family: var(--font-body, "Inter", system-ui, sans-serif);
    font-size: 13px;
    color: #6B6158;
}

/* Active menu item takes the wireframe's accent. */
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: #B45309 !important;
    font-weight: 600;
}
"""


FOOTER_TEXT = "Powered by Vidhaata Labs"


def footer_html() -> str:
    """The sidebar footer line."""
    return f'<div class="lrag-footer">{FOOTER_TEXT}</div>'


def apply_theme() -> None:
    """Inject the stylesheet for the current page."""
    st.html(f"<style>{CSS}</style>")


def render_footer() -> None:
    """Draw the footer at the bottom of the sidebar."""
    with st.sidebar:
        st.html(footer_html())
