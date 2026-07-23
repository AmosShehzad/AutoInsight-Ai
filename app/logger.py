import logging
import os

os.makedirs("outputs/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("outputs/logs/pipeline.log"),
        logging.StreamHandler(),  # also prints to terminal
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Call this at the top of every node file: logger = get_logger(__name__)"""
    return logging.getLogger(name)