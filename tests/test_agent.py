from langchain_core.messages import AIMessage, ToolMessage

from lang_rag_assistant.agent import StepEvent, extract_final_answer, map_stream_update


def test_todo_call_maps_to_plan_step():
    update = {"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "1"}])]}}
    assert map_stream_update(update) == [StepEvent(key="plan", label="Planned search queries", detail="")]


def test_search_call_maps_to_search_step():
    update = {"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "search_documentation", "args": {"query": "x"}, "id": "2"}])]}}
    assert map_stream_update(update)[0].key == "search"


def test_task_call_maps_to_read_step_with_count():
    calls = [{"name": "task", "args": {"description": "a"}, "id": str(i)} for i in range(2)]
    events = map_stream_update({"model": {"messages": [AIMessage(content="", tool_calls=calls)]}})
    assert [e.key for e in events] == ["read", "read"]


def test_tool_result_maps_to_done_detail():
    update = {"tools": {"messages": [ToolMessage(content="Saved 4 documentation chunks:\n/retrieved/a/chunk_1.md",
                                                 tool_call_id="2", name="search_documentation")]}}
    assert map_stream_update(update)[0].detail == "4 passages found"


def test_non_tool_update_yields_nothing():
    assert map_stream_update({"model": {"messages": [AIMessage(content="hi")]}}) == []


def test_final_answer_is_last_ai_text():
    messages = [AIMessage(content="draft"), AIMessage(content="Final [1].")]
    assert extract_final_answer(messages) == "Final [1]."


def test_build_agent_compiles_with_search_tool_and_subagent(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from lang_rag_assistant.agent import build_agent
    from lang_rag_assistant.database import Repository
    from lang_rag_assistant.retrieval import KnowledgeBase
    from lang_rag_assistant.tools import make_search_tool

    repo = Repository(tmp_path / "a.sqlite")
    kb = KnowledgeBase(repo, DeterministicFakeEmbedding(size=8), "all-MiniLM-L6-v2")
    tool = make_search_tool(kb, tmp_path / "ws", [])
    agent = build_agent(FakeListChatModel(responses=["hi"]), tool, tmp_path / "ws", 3)
    assert agent is not None
    repo.close()


def test_answers_do_not_list_internal_file_paths():
    from lang_rag_assistant.prompts import RAG_WORKFLOW_INSTRUCTIONS

    assert "do not list file paths" in RAG_WORKFLOW_INSTRUCTIONS.lower()
