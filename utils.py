#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import tracemalloc
import time

import ipdb
import memory_profiler

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

@contextmanager
def memory_time_moniter():
	"""Helper for measuring runtime and memory cost"""
	m0 = memory_profiler.memory_usage()
	# tracemalloc.start()
	time0 = time.perf_counter()
	yield
	print('[elapsed time: %.7f s]' % (time.perf_counter() - time0))
	print('[took memory: %.7f Mb]' % (memory_profiler.memory_usage()[0] - m0[0]))
	# current, peak = tracemalloc.get_traced_memory()
	# print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")