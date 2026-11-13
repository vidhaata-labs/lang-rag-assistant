import os
from lang_rag_assistant.config.logging_config import get_logger

logger = get_logger(__name__)

MAX_CONCURRENT_ANALYSTS_MIN = 2
MAX_CONCURRENT_ANALYSTS_MAX = 5


def get_max_concurrent_analysts() -> int:
    """Return the max concurrent analysts from the MAX_CONCURRENT_ANALYSTS env var.

    Value must be between 2 and 5 inclusive. Raises ValueError otherwise.
    """
    raw = os.getenv("MAX_CONCURRENT_ANALYSTS", "3")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"MAX_CONCURRENT_ANALYSTS must be an integer, got '{raw}'"
        )
    if value < MAX_CONCURRENT_ANALYSTS_MIN or value > MAX_CONCURRENT_ANALYSTS_MAX:
        raise ValueError(
            f"MAX_CONCURRENT_ANALYSTS must be between "
            f"{MAX_CONCURRENT_ANALYSTS_MIN} and {MAX_CONCURRENT_ANALYSTS_MAX}, "
            f"got {value}"
        )
    logger.info("Using max_concurrent_analysts: %d", value)
    return value
