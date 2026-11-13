from deepagents import create_deep_agent
from rich.console import Console
from rich.markdown import Markdown
from lang_rag_assistant.config.logging_config import configure_logging, get_logger
from lang_rag_assistant.config.model_config import create_llm
from lang_rag_assistant.config.generic_config import get_max_concurrent_analysts
from lang_rag_assistant.prompts import RAG_WORKFLOW_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS, CHUNK_ANALYST_INSTRUCTIONS
from lang_rag_assistant.tools import search_documentation, backend
from langchain.messages import HumanMessage

from lang_rag_assistant.utils.text_utils import extract_text_content

configure_logging()
logger = get_logger(__name__)

console = Console()

max_concurrent_analysts = get_max_concurrent_analysts()

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts
    )
)

logger.info("Building RAG agent with max_concurrent_analysts=%d", max_concurrent_analysts)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyzse one retrieved documentation chunk file."
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}
model = create_llm()
logger.info("LLM model created: %s", type(model).__name__)

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)
logger.info("Deep agent created successfully")

example_query = "What are the causes of climate change?"
logger.info("Invoking agent with example query: %s", example_query)

result = agent.invoke(
    {"messages": [HumanMessage(content=example_query)]}
)

content = extract_text_content(result["messages"][-1].content)
logger.info("Agent response received")
console.print(Markdown(content))
