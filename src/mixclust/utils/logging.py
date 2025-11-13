# src/mixclust/utils/logging.py
import logging, sys

def get_logger(name="mixclust", level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        f = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        h.setFormatter(f); lg.addHandler(h)
    lg.setLevel(level)
    return lg
