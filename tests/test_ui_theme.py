from lang_rag_assistant.ui import theme


def test_app_icon_file_exists():
    assert theme.PAGE_ICON.is_file()


def test_sidebar_logo_carries_the_title_and_icon():
    from PIL import Image

    assert theme.SIDEBAR_LOGO.is_file()
    with Image.open(theme.SIDEBAR_LOGO) as logo:
        width, height = logo.size
        # st.logo size="large" fixes the image height at 32px. Vertical padding
        # would shrink the 32px icon and the 16px title, so the artwork is cropped
        # to the icon's bounds.
        bbox = logo.getchannel("A").getbbox()
    assert width > height  # icon followed by the title, so wider than tall
    assert bbox == (0, 0, width, height)


def test_active_menu_item_uses_the_wireframe_accent():
    assert "#B45309" in theme.CSS


def test_footer_says_powered_by_and_is_pinned_to_the_sidebar_bottom():
    html = theme.footer_html()
    assert "Powered by Vidhaata Labs" in html
    assert "lrag-footer" in theme.CSS
    assert "position: fixed" in theme.CSS


def test_sidebar_collapse_toggle_is_hidden():
    assert "stSidebarCollapseButton" in theme.CSS
    assert "display: none" in theme.CSS
