#include "train_mini_batch.h"
#include "labeling_model.h"
#include "train_util.h"
#include "tqdm/tqdm.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

void TrainMiniBatch(PyObject* train_state, void* model_void_pointer, std::vector<std::map<std::string, at::Tensor>>& batch_list)
{
    TokenLabelingModel* model_pointer = (TokenLabelingModel*)model_void_pointer;
    model_pointer->get()->train();

    p::dict args_dict = ObjectToDict(train_state);
    double lr = p::extract<double>(args_dict["learning_rate"]);
    p::list t_loss_list = p::extract<p::list>(args_dict["train_loss"]);

    std::string optimizer_file = p::extract<std::string>(args_dict["optimizer_file"]);

    auto optimizer = torch::optim::Adam(model_pointer->get()->parameters(), torch::optim::AdamOptions(lr));
    
    if(ExistFile(optimizer_file))
    {
        torch::load(optimizer, optimizer_file);
    }

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
        device = torch::Device(torch::kCUDA);
    //std::cout << "current device: " << device << std::endl;
    model_pointer->get()->to(device);

    double running_loss = 0.0f;
    int batch_idx = 0;
    int barWidth = 70;
    int batch_size = 0;
    float batch_len = batch_list.size();
    
    float progress = (batch_idx + 1) / batch_len;
    for(auto i : tqdm::range(int(batch_len)))
    {
        auto batch_dict = batch_list[i];
        optimizer.zero_grad();

        auto x = batch_dict["x"];
        x = x.to(device, c10::ScalarType::Int);
        auto y_true = batch_dict["y_target"];
        y_true = y_true.to(device, c10::ScalarType::Long);

        batch_size = x.sizes()[0];

        //std::cout << "prediction"<< std::endl;
        auto y_pred = model_pointer->get()->forward(x);
        y_pred = y_pred.permute({0, 2, 1});
        //std::cout << "pred size: ";
        //std::cout << y_pred.sizes() << std::endl;

        //std::cout << "true size: " << y_true.sizes() << std::endl;
        auto loss_function = nn::CrossEntropyLoss();
        auto loss = loss_function(y_pred, y_true);
        loss.backward();
        
        //std::cout << "gradient"<< std::endl;
        optimizer.step();

        //std::cout << "running loss" << std::endl;
        running_loss += (loss.item().toDouble() - running_loss) / (batch_idx + 1);

        batch_idx += 1;
        progress = (batch_idx + 1) / batch_len;
    }

    t_loss_list.append(running_loss);

    std::cout << "current_train_loss: " << running_loss << std::endl;

    torch::save(optimizer, optimizer_file);
}

void EvaluateMiniBatch(PyObject* train_state, void* model_void_pointer, std::vector<std::map<std::string, at::Tensor>>& batch_list)
{
    TokenLabelingModel* model_pointer = (TokenLabelingModel*)model_void_pointer;
    model_pointer->get()->eval();

    p::dict args_dict = ObjectToDict(train_state);
    p::list v_loss_list = p::extract<p::list>(args_dict["val_loss"]);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
        device = torch::Device(torch::kCUDA);
    model_pointer->get()->to(device);

    double running_loss = 0;
    int batch_idx = 0;
    int barWidth = 70;
    int batch_size = 0;
    float batch_len = batch_list.size();
    float progress = (batch_idx + 1) / batch_len;
    for(auto i : tqdm::range(int(batch_len)))
    {
        auto batch_dict = batch_list[i];

        auto x = batch_dict["x"];
        x = x.to(device, c10::ScalarType::Int);
        auto y_true = batch_dict["y_target"];
        y_true = y_true.to(device, c10::ScalarType::Long);

        batch_size = x.sizes()[0];

        auto y_pred = model_pointer->get()->forward(x);
        y_pred = y_pred.permute({0, 2, 1});

        auto loss = F::cross_entropy(y_pred, y_true);

        running_loss += (loss.item().toDouble() - running_loss) / (batch_idx + 1);

        //std::cout.flush();
        batch_idx += 1;
        progress = (batch_idx + 1) / batch_len;
    }

    v_loss_list.append(running_loss);

    std::cout << "current_val_loss: " << running_loss << std::endl;
}