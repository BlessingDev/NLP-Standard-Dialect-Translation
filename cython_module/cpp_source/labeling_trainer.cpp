#include "labeling_trainer.h"
#include <tuple>
#include <cstring>
#include <atlstr.h>
#include "metric.h"
#include "tqdm/tqdm.h"

LabelingTrainer::LabelingTrainer(PyObject* args, PyObject* train_state)
    : Trainer<TokenLabelingModel>(args, train_state)
{
    mask_index = p::extract<int>(args_dict["mask_index"]);

    InitModel();
}

LabelingTrainer::~LabelingTrainer()
{
    if (optimizer != nullptr)
        delete optimizer;
    if (scheduler != nullptr)
        delete scheduler;
}

void LabelingTrainer::InitModel()
{
    //std::cout << "labeling model init" << std::endl;

    int num_embedding = p::extract<int>(args_dict["num_embedding"]);

    int embedding_size = 0;
    embedding_size = p::extract<int>(args_dict["embedding_size"]);

    int rnn_hidden_size = 0;
    rnn_hidden_size = p::extract<int>(args_dict["rnn_hidden_size"]);

    int class_num = 0;
    class_num = p::extract<int>(args_dict["class_num"]);

    model = TokenLabelingModel(
        num_embedding, 
        embedding_size, 
        rnn_hidden_size, 
        class_num);

    model->to(cur_device);

    int64_t p_num = 0;
    for(auto p : model->parameters())
    {
        p_num += p.numel();
    }

    /*
    std::wstring uni(L"모델 파라미터 수: ");
    char str_utf8[256] = {0, };
    int nLen = WideCharToMultiByte(CP_UTF8, 0, uni.c_str(), lstrlenW(uni.c_str()), NULL, 0, NULL, NULL);
    WideCharToMultiByte(CP_UTF8, 0, uni.c_str(), lstrlenW(uni.c_str()), str_utf8, nLen, NULL, NULL);
    */

    std::cout << "num of parameters: " << p_num << std::endl;
}

void LabelingTrainer::InitTrain()
{
    double lr = p::extract<double>(args_dict["learning_rate"]);
    double w_decay = p::extract<double>(args_dict["opt_weight_decay"]);

    auto opt_option = torch::optim::AdamOptions(lr).weight_decay(w_decay);
    optimizer = new torch::optim::Adam(model->parameters(), opt_option);

    int step_size = p::extract<int>(args_dict["sch_step_size"]);
    double gamma = p::extract<double>(args_dict["sch_gamma"]);
    scheduler = new torch::optim::StepLR((*optimizer), step_size, gamma);
}

void LabelingTrainer::StepEpoch()
{
    Trainer<TokenLabelingModel>::StepEpoch();
    scheduler->step();
}

void LabelingTrainer::TrainBatch(std::map<std::string, at::Tensor>& batch_tensor)
{
    // batch_tensor (batch_len, batch_size, data_size)
    model->train();

    double running_loss = 0.0f;
    double running_acc = 0.0f;

    auto x_batch = batch_tensor["x"];
    auto target_batch = batch_tensor["y_target"];
    int batch_idx = 0;
    int batch_len = x_batch.sizes()[0];

    for (auto i : tqdm::range(int(batch_len)))
    {
        optimizer->zero_grad();

        auto x = x_batch[i];
        x = x.to(cur_device, c10::ScalarType::Int);
        auto y_true = target_batch[i];
        y_true = y_true.to(cur_device, c10::ScalarType::Long);

        auto y_pred = model->forward(x);
        y_pred = y_pred.permute({0, 2, 1});

        std::tuple<at::Tensor, at::Tensor> max_res = at::_ops::max_dim::call(y_pred, 1, false);
        auto pred_idx = std::get<1>(max_res);

        double acc_t = ComputeAccuracy(pred_idx, x, y_true, mask_index);

        // std::cout << "true size: " << y_true.sizes() << std::endl;
        auto loss_function = nn::CrossEntropyLoss();
        auto loss = loss_function(y_pred, y_true);
        loss.backward();

        // std::cout << "gradient"<< std::endl;
        optimizer->step();

        // std::cout << "running loss" << std::endl;
        running_loss += (loss.item().toDouble() - running_loss) / (batch_idx + 1);
        running_acc += (acc_t - running_acc) / (batch_idx + 1);

        batch_idx += 1;
    }

    t_loss_list.append(running_loss);
    t_acc_list.append(running_acc);

    std::cout << "current_train_loss: " << running_loss << " current_train_acc: " << running_acc << std::endl;
}

void LabelingTrainer::ValidateBatch(std::map<std::string, at::Tensor>& batch_tensor)
{
    model->eval();

    double running_loss = 0;
    double running_acc = 0.0f;

    auto x_batch = batch_tensor["x"];
    auto target_batch = batch_tensor["y_target"];
    int batch_idx = 0;
    int batch_len = x_batch.sizes()[0];
    for (auto i : tqdm::range(int(batch_len)))
    {
        auto x = x_batch[i];
        x = x.to(cur_device, c10::ScalarType::Int);
        auto y_true = target_batch[i];
        y_true = y_true.to(cur_device, c10::ScalarType::Long);

        auto y_pred = model->forward(x);
        y_pred = y_pred.permute({0, 2, 1});

        auto loss = F::cross_entropy(y_pred, y_true);

        std::tuple<at::Tensor, at::Tensor> max_res = at::_ops::max_dim::call(y_pred, 1, false);
        auto pred_idx = std::get<1>(max_res);

        double acc_t = ComputeAccuracy(pred_idx, x, y_true, mask_index);

        running_loss += (loss.item().toDouble() - running_loss) / (batch_idx + 1);
        running_acc += (acc_t - running_acc) / (batch_idx + 1);

        std::cout.flush();
        batch_idx += 1;
    }
    
    v_loss_list.append(running_loss);
    v_acc_list.append(running_acc);

    std::cout << "current_val_loss: " << running_loss << " current_val_acc: " << running_acc << std::endl;
}