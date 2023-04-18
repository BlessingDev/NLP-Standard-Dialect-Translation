# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

from cvocabulary cimport SequenceVocabulary

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "string_util.cpp":
    string join(vector[string], string)
    string replace_all(string, string, string)

def batch_sentence(SequenceVocabulary source_vocab, SequenceVocabulary target_vocab, 
                    np.ndarray x_sources, np.ndarray y_targets, np.ndarray preds, int batch_size):
    cdef string source_sentence
    cdef string true_sentence
    cdef string pred_sentence
    cdef np.ndarray pred_idx
    cdef dict m = dict()
    cdef list result_list = list()

    for i in range(batch_size):
        source_sentence = get_source_sentence(source_vocab, x_sources[i])
        true_sentence = get_true_sentence(target_vocab, y_targets[i])
        pred_idx = np.argmax(preds[i], axis=1)
        pred_sentence = sentence_from_indices(pred_idx, target_vocab)

        m.clear()
        m["source"] = source_sentence.decode("UTF-8")
        m["truth"] = true_sentence.decode("UTF-8")
        m["pred"] = pred_sentence.decode("UTF-8")
        result_list.append(m)
    
    return result_list

def sentence_from_indices(vector[int] indices, SequenceVocabulary vocab, bool strict=True):
    cdef vector[string] out
    cdef string.iterator str_iter
    cdef string out_sentence
    cdef int index

    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.push_back(vocab.lookup_index(index).encode())
    
    out_sentence = join(out, " ")

    out_sentence = replace_all(out_sentence, " ##", "")
    
    return out_sentence

def get_source_sentence(SequenceVocabulary source_vocab, np.ndarray indices):
    return sentence_from_indices(indices, source_vocab)

def get_true_sentence(SequenceVocabulary target_vocab, np.ndarray indices):
    return sentence_from_indices(indices, target_vocab)