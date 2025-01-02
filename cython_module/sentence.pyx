# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

from cvocabulary cimport SequenceVocabulary
from cjamo import jamo_to_hangeul

import re

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

        m = dict()
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

cdef list c_batch_sentence_to_result_dict(list sentence_lists, list keys, int batch_size):
    cdef dict t_dict
    cdef list out_list
    cdef int key_num

    out_list = list()
    key_num = len(keys)
    for i in range(batch_size):
        t_dict = dict()
        for j in range(key_num):
            t_dict[keys[j]] = sentence_lists[j][i]
        
        out_list.append(t_dict)
    
    return out_list


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

def sentence_from_jamo_indices(vector[int] indices, SequenceVocabulary vocab, bool strict=True):
    cdef list jamo_list = []
    cdef list out = []
    cdef str cur_c
    cdef int jamo_len
    cdef str hangeul_pattern = "[ㄱ-ㅎ]|[ㅏ-ㅣ]"
    cdef str out_sentence = ""

    for c_idx in indices:
        if c_idx == vocab.end_seq_index and strict:
            break
        elif c_idx == vocab.mask_index and strict:
            break
        elif c_idx == vocab.begin_seq_index and strict:
            continue
        
        cur_c = vocab.lookup_index(c_idx)

        한글여부 = re.match(hangeul_pattern, cur_c)

        if cur_c == "<SEP>":
            jamo_len = len(jamo_list)

            if jamo_len >= 2 and jamo_len <= 3:
                # 자모 2~3개로 구성된 일반적인 한글
                out.append(jamo_to_hangeul(*jamo_list))
                jamo_list.clear()
            elif jamo_len > 0:
                # 자모가 4개 이상으로 구성된 문자는 없다.
                # 이런 문자가 나왔을 경우 오류이므로 특수 토큰으로 처리
                out.append('<잘못된 시퀸스 {0}>'.format(jamo_list))
        elif cur_c == "<SPC>":
            # 공백 문자
            out.append(' ')
        elif 한글여부 is not None:
            jamo_list.append(cur_c)
        else:
            out.append(cur_c)
    
    out_sentence = ''.join(out)

    return out_sentence

def batch_sentence_jamo(SequenceVocabulary source_vocab, SequenceVocabulary target_vocab, 
                    np.ndarray x_sources, np.ndarray y_targets, np.ndarray preds, int batch_size):
    cdef str source_sentence
    cdef str true_sentence
    cdef str pred_sentence
    cdef np.ndarray pred_idx
    cdef dict m = dict()
    cdef list result_list = list()

    for i in range(batch_size):
        source_sentence = sentence_from_jamo_indices(x_sources[i], source_vocab)
        true_sentence = sentence_from_jamo_indices(y_targets[i], target_vocab)
        pred_idx = np.argmax(preds[i], axis=1)
        pred_sentence = sentence_from_jamo_indices(pred_idx, target_vocab)
    
        m = {
            "source": source_sentence,
            "truth": true_sentence,
            "pred": pred_sentence
        }

        result_list.append(m)

    return result_list

def batch_sentence_to_result_dict(list sentence_lists, list keys, int batch_size):
    return c_batch_sentence_to_result_dict(sentence_lists, keys, batch_size)

def batch_process_exa_output(list sta_sentence_list, list dia_sentence_list, int batch_size):
    cdef str sta_target
    cdef str dia_target

    for idx in range(batch_size):
        sta_target = sta_sentence_list[idx]
        dia_target = dia_sentence_list[idx]

        sta_target = sta_target.split('\n')[-1]
        dia_target = dia_target.split('\n')[-1]
        
        sta_target = sta_target[13:]
        dia_target = dia_target[13:]

        sta_sentence_list[idx] = sta_target
        dia_sentence_list[idx] = dia_target

def batch_process_exa_templates(list sta_source_sentences, list dia_source_sentences, str prompt_template, int batch_size):
    cdef list messages
    cdef list sta_messages
    cdef list dia_messages
    cdef str prompt

    sta_messages = list()
    dia_messages = list()
    messages = [
        {"role": "system", 
        "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
        {"role": "user", "content": ""}
    ]

    for idx in range(batch_size):
        prompt = prompt_template.format(sta_source_sentences[idx])
        messages[1]["content"] = prompt
        sta_messages.append(messages)

        prompt = prompt_template.format(dia_source_sentences[idx])
        messages[1]["content"] = prompt
        dia_messages.append(messages)
    
    return (sta_messages, dia_messages)