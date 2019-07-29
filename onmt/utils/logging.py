# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import logging
from datetime import datetime

logger = logging.getLogger()


def init_logger(log_file):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def try_debugger(opt=None, addr='localhost', port=3000):
    if opt:
        if opt.debug is not None and not opt.debug:
            return

    try:
        import ptvsd
        ptvsd.enable_attach(address=(addr, port), redirect_output=True)
        print("Waiting for VS debugger to attach... press Ctrl+C to continue")
        ptvsd.wait_for_attach()
    except:
        pass
