from loguru import logger
from langchain.schema import Runnable
from tenacity import retry, stop_after_attempt, wait_fixed


def log_retry(retry_state):
    """Callback function to log retry attempts"""
    logger.warning(
        f"Retrying LLM call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}"
    )


async def invoke_with_retry(
    chain: Runnable,
    args: dict,
    config: dict = None,
    max_attempts: int = 10,
    retry_delay: float = 0.1,
    supress_retry_logs: bool = False,
) -> str:
    """Helper function that wraps the LLM call with retry logic

    Args:
        chain: The runnable chain to invoke
        args: Dictionary of input arguments
        config: Optional configuration dictionary for the chain
        max_attempts: Maximum number of retry attempts (default: 10)
    """
    retry_decorator = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(retry_delay),
        after=log_retry if not supress_retry_logs else None,
    )

    @retry_decorator
    async def _invoke():
        if config:
            return await chain.ainvoke(args, config=config)
        return await chain.ainvoke(args)

    return await _invoke()
