import functools
from app.logger import get_logger

logger = get_logger(__name__)


class NodeExecutionError(Exception):
    """
    Raised when a LangGraph node fails, tagged with the EXACT node name
    that was running — not guessed afterward from the exception type.
    This replaces the unreliable module-name guessing in run_pipeline_safely.
    """
    def __init__(self, node_name: str, original_exception: Exception):
        self.node_name = node_name
        self.original_exception = original_exception
        super().__init__(f"[{node_name}] {type(original_exception).__name__}: {original_exception}")


def node_error_boundary(node_name: str):
    """
    Decorator for LangGraph node functions. Catches ANY exception raised
    inside the node and re-raises it as NodeExecutionError, tagged with
    the node's real name — captured at the exact point of failure, so
    there's no guessing later.

    Usage: @node_error_boundary("statistics")
           def statistics_node(state): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except NodeExecutionError:
                raise  # already tagged by a nested call — don't double-wrap
            except Exception as e:
                logger.error(f"Node '{node_name}' failed: {type(e).__name__}: {e}")
                raise NodeExecutionError(node_name, e)
        return wrapper
    return decorator