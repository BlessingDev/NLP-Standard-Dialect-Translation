#ifndef __TRAIN_UTIL_H__
#define __TRAIN_UTIL_H__
#include <torch/torch.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <tuple>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <torch/torch.h>
#include "labeling_model.h"
#include "npy.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

p::dict ObjectToDict(PyObject*);

void InitBoostPython()
{
    //std::cout << "Init boost python" << std::endl;

    Py_Initialize();
    np::initialize();
}

void SetSeed(int s)
{
    torch::manual_seed(s);
    torch::cuda::manual_seed(s);
}

at::Tensor LoadNpyToTensor(std::string& file_path)
{
    torch::Device dev(torch::kCPU);
    if (torch::cuda::is_available())
        dev = torch::Device(torch::kCUDA);

    auto opts = torch::TensorOptions().dtype(torch::kDouble);

    std::vector<unsigned long> shape;
    std::vector<double> data;

    npy::LoadArrayFromNumpy(file_path, shape, data);

    std::vector<int64_t> shape_64(shape.begin(), shape.end());
    
    auto out_tensor = torch::from_blob(data.data(), shape_64, opts).clone();
    out_tensor.to(torch::kCPU, torch::kDouble);

    //out_tensor = out_tensor.to(dev);

    //std::cout << "npy loaded from " << file_path << " with size " << out_tensor.sizes() << std::endl;
    return out_tensor;
}

void TensorMapToBatch(std::map<std::string, at::Tensor>& tensor_map, std::map<std::string, at::Tensor>& batch_tensor_map, std::vector<std::string>& keys, int batch_size, bool shuffle=true)
{
    batch_tensor_map.clear();

    int set_size = tensor_map[keys[0]] .sizes()[0];
    int batch_num = set_size / batch_size;
    int batch_set_size = batch_num * batch_size;

    if(shuffle)
    {
        auto rand_idx = torch::randperm(set_size);
        
        // 텐서를 셔플함
        for (std::string k : keys)
        {
            batch_tensor_map[k] = tensor_map[k].index({rand_idx});
        }
    }

    for (std::string k : keys)
    {
        batch_tensor_map[k] = batch_tensor_map[k].slice(0, 0, batch_set_size); // batch 크기에 맞게 drop_last
    }

    for (std::string k : keys)
    {
        auto data_tensor = batch_tensor_map[k];
        std::vector<int64_t> shape({batch_num, batch_size});
        for (int idx = 1; idx < data_tensor.dim(); idx += 1)
        {
            shape.push_back(data_tensor.sizes()[idx]);
        }
        batch_tensor_map[k] = data_tensor.reshape(shape);

        //std::cout << "key: " << k << std::endl; 
        //std::cout << "data (" << tensor_map[k].sizes() << ") to batch data (" << batch_tensor_map[k].sizes() << ")" << std::endl;
    }
}

bool IsCudaAvailable()
{
    return torch::cuda::is_available();
}

bool ExistFile(std::string& name)
{
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

std::vector<int> GetTensorSize(const at::Tensor& t)
{
    auto t_size = t.sizes().vec();

    return std::vector<int>(t_size.begin(), t_size.end());
}

template <typename T>
p::list toPythonList(std::vector<T>& v)
{
    p::list l;
    for (T v_val : v)
    {
        l.append(v_val);
    }

    return l;
}

PyObject* GetNdarrayFromTensor(at::Tensor& t)
{
    t = t.to(c10::DeviceType::CPU, c10::ScalarType::Double);
    double* tensor_data = t.data_ptr<double>();

    // 1차원 텐서로 리쉐이프
    // vector 만들기
    // 출력
    
    std::vector<double> dobule_arr(tensor_data, tensor_data + t.ndimension());
    
    p::tuple shape(toPythonList(t.sizes().vec()));
    p::tuple stride = p::make_tuple(sizeof(double));
    np::dtype double_type = np::dtype::get_builtin<double>();

    p::object own;
    np::ndarray np_arr = np::from_data(tensor_data, double_type, shape, stride, own);

    return own.ptr();
}

at::Tensor MakeRandomTensor()
{
    at::Tensor t =  torch::rand({2, 3});

    std::cout << t << std::endl;

    return t;
}

/*
at::Tensor NdarrayToTensor(np::ndarray& array)
{
    //std::cout << "ndarr to tensor" << std::endl;
    std::vector<int> ori_shape = GetNparrSize(array);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
        device = torch::Device(torch::kCUDA);
    auto opts = torch::TensorOptions().dtype(torch::kDouble);

    at::Tensor out_tensor;
    switch (ori_shape.size())
    {
    case 1:
        out_tensor = torch::from_blob(array.get_data(), {1, ori_shape[0]}, opts);
        break;
    case 2:
        out_tensor = torch::from_blob(array.get_data(), {ori_shape[0], ori_shape[1]}, opts);
        break;
    case 3:
        out_tensor = torch::from_blob(array.get_data(), {ori_shape[0], ori_shape[1], ori_shape[2]}, opts);
        break;
    default:
        std::cout << "3차원 이상 텐서입니다." << std::endl;
        break;
    }
    out_tensor = out_tensor.clone();
    out_tensor = out_tensor.to(device);

    //std::cout << "tensor from ndarray" << std::endl;
    //std::cout << out_tensor.sizes() << std::endl;
    //std::cout << out_tensor[0] << std::endl;
    return out_tensor;
}

np::ndarray ObjectToNdarray(PyObject* obj_pointer)
{
    //std::cout << "object to ndarray" << std::endl;
    p::handle<> handle(p::borrowed(obj_pointer));
    p::object arr_obj(handle);

    np::ndarray np_arr = np::from_object(arr_obj);

    //std::cout << "conversion complete" << std::endl;
    return np_arr;
}
*/

p::list ObjectToList(PyObject* obj_pointer)
{
    //std::cout << "object to list" << std::endl;
    p::handle<> handle(p::borrowed(obj_pointer));
    p::object arr_obj(handle);

    p::list py_list = p::extract<p::list>(arr_obj);

    return py_list;
}

p::dict ObjectToDict(PyObject* obj_pointer)
{
    //std::cout << "object to dict" << std::endl;
    
    p::handle<> handle(p::borrowed(obj_pointer));
    p::object arr_obj(handle);

    p::dict py_dict = p::extract<p::dict>(arr_obj);

    return py_dict;
}

#endif