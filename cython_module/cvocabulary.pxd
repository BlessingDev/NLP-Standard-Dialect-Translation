
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef class Vocabulary():
    cdef dict _token_to_idx
    cdef dict _idx_to_token

    cpdef to_serializable(self)

    @staticmethod
    cpdef from_serializable(cls, dict contents)

    cpdef add_token(self, str token)
    cpdef add_many(self, list[str] tokens)
    cpdef lookup_token(self, str token)
    cpdef lookup_index(self, int index)

cdef class SequenceVocabulary(Vocabulary):
    cdef str _mask_token
    cdef str _unk_token
    cdef str _begin_seq_token
    cdef str _end_seq_token

    cdef public int mask_index, unk_index, begin_seq_index, end_seq_index
    
    cpdef to_serializable(self)
    cpdef lookup_token(self, str token)