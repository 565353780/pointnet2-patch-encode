#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()
