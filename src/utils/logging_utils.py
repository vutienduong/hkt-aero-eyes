from loguru import logger
import sys

def get_logger(name=None, level="INFO"):
    """
    Get a configured logger instance using loguru.

    Args:
        name: Optional logger name (for filtering)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        logger: Configured loguru logger
    """
    # Remove default handler
    logger.remove()

    # Add custom handler with formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )

    return logger
