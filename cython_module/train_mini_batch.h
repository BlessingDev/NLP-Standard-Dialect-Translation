#ifndef __TRAIN_MINIBATCH_H__
#define __TRAIN_MINIBATCH_H__
#include <map>
#include <vector>
#include <string>
#include <torch/torch.h>
#include <boost/python.hpp>

void TrainMiniBatch(PyObject* train_state, void* model_void_pointer, std::vector<std::map<std::string, at::Tensor>>& batch_list);


void EvaluateMiniBatch(PyObject* train_state, void* model_void_pointer, std::vector<std::map<std::string, at::Tensor>>& batch_list);

#endif