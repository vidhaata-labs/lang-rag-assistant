"""Model page: choose the chat model and the embedding model, and see which keys are set."""

from __future__ import annotations

import streamlit as st

from lang_rag_assistant.config.model_config import (
    EMBEDDING_MODELS,
    PROVIDER_KEY_ENV,
    create_embeddings,
)
from lang_rag_assistant.config.settings import SUPPORTED_PROVIDERS
from lang_rag_assistant.ingest import reembed_all
from lang_rag_assistant.ui.components import show_alert
from lang_rag_assistant.ui.services import (
    EMBEDDING_SETTING,
    PROVIDER_LABELS,
    Services,
    option_label,
    save_selection,
    saved_option_key,
    selected_option,
)


def render(services: Services) -> None:
    st.title("Model")
    st.caption("Choose the chat model and the embedding model. Only models whose provider has a configured key are listed.")
    picker, keys = st.columns([3, 2], vertical_alignment="top")
    with picker, st.container(border=True):
        _choose(services)
    with keys, st.container(border=True):
        _key_table(services)


def _choose(services: Services) -> None:
    enabled = services.enabled
    current = selected_option(services)
    saved = saved_option_key(services)

    chat_label = st.selectbox(
        "Chat model",
        [option_label(o) for o in enabled],
        index=enabled.index(current) if current in enabled else None,
        placeholder="No models available",
        disabled=not enabled,
        key="model_chat_pick",
    )
    embedding_choice = st.selectbox(
        "Embedding model",
        list(EMBEDDING_MODELS),
        index=list(EMBEDDING_MODELS).index(services.embedding_model),
        key="model_embedding_pick",
    )
    if saved is not None and current is None:
        provider = saved.split(":", 1)[0]
        show_alert(
            "warning",
            f"No API key for {PROVIDER_LABELS.get(provider, provider)}",
            f"Add {PROVIDER_KEY_ENV[provider]} to .env and restart the app, "
            "or choose a model whose provider has a key.",
        )
    if not enabled:
        show_alert(
            "info",
            "No model configured",
            "Add an API key for Google GenAI, Groq, OpenAI, Requesty or NVIDIA in .env and restart the app.",
        )

    if st.button("Use this model", type="primary", disabled=chat_label is None, key="use_model"):
        chosen = next(o for o in enabled if option_label(o) == chat_label)
        if embedding_choice != services.embedding_model:
            _confirm_reembed(services, chosen, embedding_choice)
        else:
            save_selection(services, chosen)
            st.toast("Model saved")


@st.dialog("Re-embed documents?")
def _confirm_reembed(services: Services, chosen, new_embedding: str) -> None:
    st.write(
        f"Switching to {new_embedding} re-embeds every stored passage. "
        "This can take a few minutes."
    )
    if st.button("Re-embed and switch", type="primary", key="confirm_reembed"):
        reembed_all(
            repo=services.repo,
            embeddings=create_embeddings(new_embedding),
            embedding_model=new_embedding,
        )
        services.repo.set_setting(EMBEDDING_SETTING, new_embedding)
        save_selection(services, chosen)
        st.cache_resource.clear()
        st.rerun()
    if st.button("Cancel", key="cancel_reembed"):
        st.rerun()


def _key_table(services: Services) -> None:
    st.markdown("**Provider keys**")
    by_provider = {o.provider: o for o in services.options}
    st.table(
        {
            "Provider": [PROVIDER_LABELS[p] for p in SUPPORTED_PROVIDERS],
            "Default model": [by_provider[p].model for p in SUPPORTED_PROVIDERS],
            "Key": ["Configured" if by_provider[p].has_key else "Missing key" for p in SUPPORTED_PROVIDERS],
        }
    )
    st.caption("Keys are read from .env. A provider with a missing key is listed but cannot be selected.")
