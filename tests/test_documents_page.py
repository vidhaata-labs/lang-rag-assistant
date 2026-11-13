from lang_rag_assistant.errors import DocumentError
from lang_rag_assistant.ui.pages.documents import failure_message


def test_document_error_message_is_shown_as_is():
    assert failure_message(DocumentError("notes.pdf is password-protected.")) == "notes.pdf is password-protected."


def test_unexpected_failure_gets_a_generic_message_not_the_raw_text():
    message = failure_message(RuntimeError("embedding model unavailable: /secret/path"))
    assert "/secret/path" not in message
    assert message.startswith("Something went wrong")


def test_dismissing_a_failed_upload_clears_it_and_resets_the_uploader():
    from lang_rag_assistant.ui.pages.documents import dismiss_failed_upload

    state = {"upload_error": {"title": "Couldn't read this file"}, "upload_version": 2}
    dismiss_failed_upload(state)
    assert "upload_error" not in state
    assert state["upload_version"] == 3


def test_uploaded_documents_are_stored_with_a_kind_the_database_accepts(tmp_path):
    from lang_rag_assistant.database import Repository
    from lang_rag_assistant.ui.pages.documents import UPLOAD_SOURCE_KIND

    repo = Repository(tmp_path / "u.sqlite")
    doc_id = repo.add_document(name="a.pdf", source_kind=UPLOAD_SOURCE_KIND, sha256="u1", chunks=[], vectors=[], embedding_model="m")
    assert repo.list_documents()[0].id == doc_id
