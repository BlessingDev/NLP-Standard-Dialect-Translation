# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

from cvocabulary cimport SequenceVocabulary

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "cpp_source/string_util.cpp":
    string join(vector[string], string)
    string replace_all(string, string, string)

cdef list c_batch_sentence_mt(SequenceVocabulary source_vocab, SequenceVocabulary target_vocab, 
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

cdef list c_batch_sentence_tl(SequenceVocabulary vocab, np.ndarray x_sources, np.ndarray y_labels, np.ndarray preds, int batch_size):
    cdef string source_sentence
    cdef string true_sentence
    cdef string pred_sentence
    cdef np.ndarray pred_idx
    cdef np.ndarray label_idx
    cdef dict m
    cdef list result_list

    result_list = []
    for i in range(batch_size):
        source_sentence = sentence_from_indices(x_sources[i], vocab)
        label_idx = np.where(y_labels[i] == 1)[0]
        true_sentence = sentence_from_indices(x_sources[i][label_idx], vocab, connect_enable=False)
        pred_idx = np.where(preds[i] == 1)[0]
        pred_sentence = sentence_from_indices(x_sources[i][pred_idx], vocab, connect_enable=False)

        m = dict()
        m["input"] = source_sentence.decode("UTF-8")
        m["label"] = true_sentence.decode("UTF-8")
        m["pred"] = pred_sentence.decode("UTF-8")
        result_list.append(m)
    
    return result_list

cdef string c_sentence_from_indices(vector[int] indices, SequenceVocabulary vocab, bool strict=True, bool connect_enable=True):
    cdef vector[string] out
    cdef string out_sentence
    cdef int index

    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        elif index == vocab.mask_index and strict:
            break
        else:
            out.push_back(vocab.lookup_index(index).encode("UTF-8"))
    
    out_sentence = join(out, " ")

    if connect_enable :
        out_sentence = replace_all(out_sentence, " ##", "")
    
    return out_sentence

def batch_sentence_mt(SequenceVocabulary source_vocab, SequenceVocabulary target_vocab, 
                    np.ndarray x_sources, np.ndarray y_targets, np.ndarray preds, int batch_size):
    return c_batch_sentence_mt(source_vocab, target_vocab, x_sources, y_targets, preds, batch_size)

def batch_sentence_tl(SequenceVocabulary vocab, np.ndarray x_sources, np.ndarray y_labels, np.ndarray preds, int batch_size):
    return c_batch_sentence_tl(vocab, x_sources, y_labels, preds, batch_size)

def sentence_from_indices(vector[int] indices, SequenceVocabulary vocab, bool strict=True, bool connect_enable=True):
    return c_sentence_from_indices(indices, vocab, strict, connect_enable)

def get_source_sentence(SequenceVocabulary source_vocab, np.ndarray indices):
    return sentence_from_indices(indices, source_vocab)

def get_true_sentence(SequenceVocabulary target_vocab, np.ndarray indices):
    return sentence_from_indices(indices, target_vocab)