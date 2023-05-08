#ifndef __TRAINER_H__
#define __TRAINER_H__
#include <map>
#include <string>
#include <cstdio>
#include <boost/python.hpp>
#include <torch/torch.h>
#include "train_util.h"

namespace p = boost::python;

template <typename T>
class Trainer
{
protected:
    p::dict args_dict;
    p::dict train_state;

    p::list t_loss_list;
    p::list v_loss_list;

    int current_epoch;
    std::string model_path;
    torch::Device cur_device;

    T model = nullptr;
private:
    virtual void InitModel() = 0;

public:
    Trainer(PyObject* args, PyObject* py_train_state);

    virtual ~Trainer();

    virtual void InitTrain() = 0;

    virtual void StepEpoch();

    virtual void SaveModel();

    virtual void LoadModel();

    virtual void TrainBatch(std::map<std::string, at::Tensor>& batch_tensor) = 0;

    virtual void ValidateBatch(std::map<std::string, at::Tensor>& batch_tensor) = 0;
};


template <typename T>
Trainer<T>::Trainer(PyObject* args, PyObject* py_train_state)
    : cur_device(torch::kCPU), current_epoch(0)
{
    InitBoostPython();

    args_dict = ObjectToDict(args);
    train_state = ObjectToDict(py_train_state);

    v_loss_list = p::extract<p::list>(train_state["val_loss"]);
    t_loss_list = p::extract<p::list>(train_state["train_loss"]);
    model_path = p::extract<std::string>(args_dict["model_state_file"]);

    if (torch::cuda::is_available())
        cur_device = torch::Device(torch::kCUDA);
}

template <typename T>
Trainer<T>::~Trainer()
{
}

template <typename T>
void Trainer<T>::StepEpoch()
{
    current_epoch += 1;
}

template <typename T>
void Trainer<T>::SaveModel()
{
    if(ExistFile(model_path))
    {
        //std::cout << "model overwrite" << std::endl;
        std::remove(model_path.c_str());
    }

    torch::save(model, model_path);
}

template <typename T>
void Trainer<T>::LoadModel()
{
    if(ExistFile(model_path))
    {
        std::cout << "load model from file" << std::endl;
        torch::load(model, model_path);
    }
    else
    {
        std::cout << "can't load model at " << model_path << std::endl;
    }
}

#endif