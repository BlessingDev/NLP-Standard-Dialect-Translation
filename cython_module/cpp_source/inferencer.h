#pragma once
#include <map>
#include <string>
#include <cstdio>
#include <boost/python.hpp>
#include <torch/torch.h>
#include "train_util.h"

namespace p = boost::python;

template <typename T>
class Inferencer
{
protected:
    p::dict args_dict;
    p::dict eval_dict;

    std::string model_path;
    torch::Device cur_device;

    T model = nullptr;
public:
    Inferencer(PyObject* args, PyObject* eval_dict);

    virtual ~Inferencer();

    virtual void InitModel() = 0;

    virtual void LoadModel();

    virtual at::Tensor InferenceSingleBatch(std::map<std::string, at::Tensor>& batch_tensor, int batch_idx) = 0;
};

template <typename T>
Inferencer<T>::Inferencer(PyObject* args, PyObject* eval_dict)
    : cur_device(torch::kCPU)
{
    InitBoostPython();

    args_dict = ObjectToDict(args);
    this->eval_dict = ObjectToDict(eval_dict);

    model_path = p::extract<std::string>(args_dict["model_state_file"]);

    if (torch::cuda::is_available())
        cur_device = torch::Device(torch::kCUDA);
}

template <typename T>
Inferencer<T>::~Inferencer()
{
}

template <typename T>
void Inferencer<T>::LoadModel()
{
    if(ExistFile(model_path))
    {
        std::cout << "load model from file" << std::endl;
        torch::load(model, model_path);
    }
    else
    {
        std::cout << "can't load model from " << model_path << std::endl;
    }
}