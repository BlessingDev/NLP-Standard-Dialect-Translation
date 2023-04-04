from libcpp.vector import vector
from libcpp.map import map
from libcpp.string import string

cdef cppclass vocab(object):
    cdef map[string, int] _token_to_idx
    cdef map[int, string] _idx_to_token

    cdef __init__(self, token_to_idx=None):
        

cdef sentence_from_indices(vector[int] indices, map[int, string] vocab, bool strict):
    cdef vector[string] out

    for index in indices:
        if index == vocab