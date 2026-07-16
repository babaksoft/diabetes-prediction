import logging

from diabetes_prediction.config import settings


def configure_logging() -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
    )

    logging.basicConfig(
        level=settings.LOG_LEVEL,
        handlers=[console_handler],
    )
