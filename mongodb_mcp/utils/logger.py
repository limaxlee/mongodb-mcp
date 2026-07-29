import os
import asyncio
import logging
import logging.handlers
import multiprocessing
from io import BytesIO
from zipfile import ZipFile
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from common.constants import ROOT_DIR

spawn_ctx = multiprocessing.get_context("spawn")
executor = ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=spawn_ctx)


class LogConfig:
    LEVEL = logging.INFO
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    MAX_BYTES = 10 * 1024 * 1024
    BACKUP_COUNT = 20
    ENCODING = "UTF-8"
    FORMAT = "%(asctime)s %(levelname)-8s %(process)-5s --- [%(threadName)s] [%(filename)s:%(lineno)d] : %(message)s"


def initialize_logger(filename, logger=None):
    if not os.path.exists(LogConfig.LOG_DIR):
        os.makedirs(LogConfig.LOG_DIR, exist_ok=True)

    if not logger:
        logger = logging.getLogger()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LogConfig.LEVEL)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LogConfig.LOG_DIR, filename),
        maxBytes=LogConfig.MAX_BYTES,
        backupCount=LogConfig.BACKUP_COUNT,
        encoding=LogConfig.ENCODING
    )
    formatter = logging.Formatter(LogConfig.FORMAT)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(LogConfig.LEVEL)

    return logger


async def get_logs_zip_file() -> None | bytes:
    log_files = [os.path.join(LogConfig.LOG_DIR, log) for log in os.listdir(LogConfig.LOG_DIR)]
    if not log_files:
        return

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(_create_logs_zip_file, log_files))


def _create_logs_zip_file(log_files: list[str]) -> bytes:
    logs = {}
    for log_path in log_files:
        with open(log_path, "rb") as f:
            logs[os.path.basename(log_path)] = f.read()

    with BytesIO() as buffer:
        with ZipFile(buffer, "w") as z:
            for file_name, data in logs.items():
                z.writestr(file_name, data)

        return buffer.getvalue()
