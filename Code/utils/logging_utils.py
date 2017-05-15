# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for logging

"""

import os
import logging
import logging.handlers


def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
                    filename=os.path.join(logdir, logname),
                    maxBytes=10*1024*1024, 
                    backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger("")
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)        # remove all old handlers
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger
