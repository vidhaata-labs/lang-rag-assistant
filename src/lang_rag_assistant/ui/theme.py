"""Visual theme: the approved Doop wireframe tokens, applied as config plus CSS.

Colours, fonts, radius and top padding live in `.streamlit/config.toml`. The CSS
below adds what config cannot express: the 272px sidebar, the 32px page gutter,
and the layout of the wireframe components. Each rule targets a `.st-key-<key>`
class generated from an element's `key=`; the sidebar and gutter rules are the
only selectors that target Streamlit's own containers, because no config option
sets them.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGE_ICON = Path(__file__).resolve().parent / "assets" / "app_icon.png"

CSS = """
/* Layout: sidebar width and page gutters from the wireframes. */
section[data-testid="stSidebar"] { width: 272px !important; min-width: 272px !important; }
section[data-testid="stSidebar"] > div:first-child { padding: 0 16px 24px; }
[data-testid="stSidebarHeader"] { display: none; }
.block-container { padding-left: 32px !important; padding-right: 32px !important; max-width: none !important; }

/* Keyed buttons: labels align to the left, as in the wireframes. */
[class*="st-key-starter-"] button > div, [class*="st-key-recent-"] button > div, [class*="st-key-nav-"] button > div, .st-key-sidebar-history button > div, .st-key-sidebar-new button > div { width: 100% !important; justify-content: flex-start !important; }
[class*="st-key-starter-"] button > div > span, [class*="st-key-recent-"] button > div > span, [class*="st-key-nav-"] button > div > span, .st-key-sidebar-history button > div > span, .st-key-sidebar-new button > div > span { display: block !important; width: 100% !important; text-align: left !important; }
[class*="st-key-starter-"] button p, [class*="st-key-recent-"] button p, [class*="st-key-nav-"] button p, .st-key-sidebar-history button p, .st-key-sidebar-new button p { text-align: left !important; margin: 0 !important; }

[class*="st-key-starter-"] button > div > span > span > div, [class*="st-key-recent-"] button > div > span > span > div, [class*="st-key-nav-"] button > div > span > span > div, .st-key-sidebar-history button > div > span > span > div, .st-key-sidebar-new button > div > span > span > div { display: block !important; width: 100% !important; justify-content: flex-start !important; text-align: left !important; }

/* Shared text helpers. */
.lrag-brand { font-family: 'Fraunces', Georgia, serif; font-size: 20px; font-weight: 600; margin: 0 0 24px; color: #1F1B16; }
.lrag-brand span { color: #B45309; }
.lrag-label { font-size: 13px; color: #6B6358; text-transform: uppercase; letter-spacing: .08em; margin: 16px 0 8px; }
.lrag-mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11px; color: #6B6358; }
.lrag-muted { color: #6B6358; font-size: 14px; }

/* Page header: title with a rule underneath. */
.lrag-header { border-bottom: 1px solid #D9D0C1; padding-bottom: 16px; margin-bottom: 24px; }
.lrag-header h1 { font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 400; margin: 0; color: #1F1B16; }
.lrag-header p { color: #6B6358; font-size: 15px; margin: 4px 0 0; }

/* Sidebar: bordered new-conversation button, list items, active item, nav. */
.st-key-sidebar-new button { width: 100%; justify-content: flex-start; background: transparent; border: 1px solid #1F1B16; color: #1F1B16; font-weight: 600; font-size: 14px; border-radius: 4px; padding: 10px 14px; }
[class*="st-key-recent-"] button { width: 100%; justify-content: flex-start; background: transparent; border: none; border-bottom: 1px solid #D9D0C1; border-radius: 0; color: #6B6358; font-size: 15px; font-weight: 400; padding: 8px 0; }
.st-key-recent-active button { color: #1F1B16; font-weight: 600; }
.st-key-sidebar-history button { justify-content: flex-start; background: transparent; border: none; color: #B45309; font-weight: 600; font-size: 14px; padding: 8px 0; }
[class*="st-key-nav-"] button { width: 100%; justify-content: flex-start; background: transparent; border: none; color: #6B6358; font-size: 16px; padding: 10px 12px; border-radius: 4px; }
.st-key-nav-active button { background: #F3E4CC; color: #1F1B16; font-weight: 600; }

/* Chat empty state: hero headline, lede, 2x2 starter cards. */
.lrag-hero { font-family: 'Fraunces', Georgia, serif; font-size: 44px; line-height: 1.1; font-weight: 400; letter-spacing: -0.02em; margin: 0; max-width: 720px; color: #1F1B16; }
.lrag-hero em { color: #B45309; font-style: italic; }
.lrag-lede { color: #6B6358; font-size: 18px; margin: 16px 0 32px; }
[class*="st-key-starter-"] { background: #FBF8F2; border-radius: 10px; }
[class*="st-key-starter-"] button { width: 100%; justify-content: flex-start; background: transparent; border: none; color: #1F1B16; font-size: 16px; font-weight: 400; line-height: 1.4; padding: 0; text-align: left; }

/* Conversation thread: 820px column, user bubble right, assistant text 17px. */
.st-key-thread { max-width: 820px; width: 100%; margin: 0 auto; }
[class*="st-key-usermsg-"] { margin-left: auto; background: #F3E4CC; border-radius: 10px; padding: 12px 16px; }
[class*="st-key-usermsg-"] p { font-size: 17px; margin: 0; }
[class*="st-key-botmsg-"] p { font-size: 17px; line-height: 1.6; }
.st-key-composer { max-width: 820px; width: 100%; margin: 0 auto; }

/* Related questions under answers: bordered chips, as in the Chat frames. */
[class*="st-key-hist-"] button, [class*="st-key-new-"] button { background: #FBF8F2; border: 1px solid #D9D0C1; border-radius: 10px; color: #1F1B16; font-size: 15px; font-weight: 400; justify-content: flex-start; text-align: left; }

/* Citation chips and error cards. */
.lrag-chip { display: inline-block; font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12px; border: 1px solid #D9D0C1; background: #FBF8F2; border-radius: 4px; padding: 2px 8px; margin: 0 6px 6px 0; color: #6B6358; }
.lrag-chip b { color: #B45309; font-weight: 500; }
.lrag-error { border: 1px solid #9B2C2C; border-left: 4px solid #9B2C2C; background: #FBF8F2; border-radius: 10px; padding: 16px 20px; margin: 8px 0; max-width: 640px; }
.lrag-error h4 { font-family: 'Fraunces', Georgia, serif; font-weight: 400; color: #9B2C2C; margin: 0 0 6px; }
.lrag-key-set { color: #4D6B3A; }
.lrag-key-missing { color: #9B2C2C; }

/* Documents: dashed drop zone and row cards. */
.st-key-pdf-upload section { border: 1.5px dashed #6B6358; background: #FBF8F2; border-radius: 10px; }
[class*="st-key-doc-"], [class*="st-key-history-"] { background: #FBF8F2; border-radius: 10px; }

/* Model and keys: the settings panel card. */
.st-key-model-panel { background: #FBF8F2; border-radius: 10px; max-width: 720px; }
"""


def apply_theme() -> None:
    """Inject the wireframe layout CSS for the current page."""
    st.html(f"<style>{CSS}</style>")


def page_header(title: str, subtitle: str | None = None) -> None:
    """Render a page title with the wireframe's underline rule."""
    lede = f"<p>{subtitle}</p>" if subtitle else ""
    st.html(f'<div class="lrag-header"><h1>{title}</h1>{lede}</div>')
