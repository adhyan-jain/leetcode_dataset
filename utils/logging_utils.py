import functools
import logging
from typing import Any, Callable

def setup_global_logger() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True
    )

def log_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__} with args={args} kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise
    return wrapper
