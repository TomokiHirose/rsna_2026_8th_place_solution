import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """標準の Python ロガーを作って返す。ハンドラが無ければ 1 つ足す。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
