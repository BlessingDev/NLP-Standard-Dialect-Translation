# distutils: language = c++

import numpy as np
cimport numpy as np

np.import_array()

def batch_f1(np.ndarray y_preds, np.ndarray y_targets):
    cdef int batch_size
    cdef double tp
    cdef double model_true
    cdef double y_true
    cdef double precision
    cdef double recall
    cdef double running_f1
    cdef double running_pre
    cdef double running_rec
    cdef double f1_t
    cdef int valid_idx
    cdef dict met_dict

    met_dict = dict()

    valid_idx = 0
    running_f1 = 0
    running_pre = 0
    running_rec = 0
    batch_size = y_preds.shape[0]
    for i in range(batch_size):
        f1_t = 0

        tp = len(np.intersect1d(np.where(y_preds[i] == 1)[0], np.where(y_targets[i] == 1)[0]))
        model_true = np.sum(y_preds[i] == 1)
        y_true = np.sum(y_targets[i] == 1)

        if y_true == 0:
            f1_t = 0
        else:
            if model_true == 0:
                f1_t = 0
                precision = 0
                recall = 0
            else:
                precision = tp / model_true
                recall = tp / y_true
                
                if tp == 0:
                    f1_t = 0
                else:
                    f1_t = 2 * (precision * recall) / (precision + recall)
        
            running_f1 += (f1_t - running_f1) / (valid_idx + 1)
            running_pre += (precision - running_pre) / (valid_idx + 1)
            running_rec += (recall - running_rec) / (valid_idx + 1)

            valid_idx += 1
    
    met_dict["precision"] = running_pre
    met_dict["recall"] = running_rec
    met_dict["f1"] = running_f1
    return met_dict