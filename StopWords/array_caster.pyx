import numpy as np


def to_array(list inp):
    cdef long[:] arr = np.zeros(len(inp), dtype=long)
    cdef Py_ssize_t idx
    for idx in range(len(inp)):
        arr[idx] = inp[idx]
    return np.asarray(arr)