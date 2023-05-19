# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

from cvocabulary cimport SequenceVocabulary

import numpy as np
cimport numpy as np

np.import_array()

cdef list c_batch_sentence_mt(SequenceVocabulary source_vocab, SequenceVocabulary target_vocab, 
                    np.ndarray x_sources, np.ndarray y_targets, np.ndarray preds, int batch_size)

cdef list c_batch_sentence_tl(SequenceVocabulary vocab, np.ndarray x_sources, np.ndarray y_labels, np.ndarray preds, int batch_size)

cdef string c_sentence_from_indices(vector[int] indices, SequenceVocabulary vocab, bool strict=*)