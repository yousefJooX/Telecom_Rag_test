# for system  logs  ->  matrix   
import sys
import logging
from logging.handlers import RotatingFileHandler # ->  size  limit  for  log  files

def get_logger(name: str = "RAG") -> logging.Logger: 
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
        )

        # Stream Handler (Console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Rotating File Handler (Logs to app.log)
        file_handler = RotatingFileHandler("app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = get_logger()