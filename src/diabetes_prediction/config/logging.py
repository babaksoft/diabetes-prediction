import logging

from diabetes_prediction.config import settings


def configure_logging() -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] "
            "[%(name)s] "
            "[%(levelname)s] "
            "[%(module)s:%(lineno)d] "
            "%(message)s"
        )
    )

    logging.basicConfig(
        level=settings.LOG_LEVEL,
        handlers=[console_handler],
    )
