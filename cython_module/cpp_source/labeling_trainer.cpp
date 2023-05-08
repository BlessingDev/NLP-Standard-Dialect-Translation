#include "labeling_trainer.h"
#include "tqdm/tqdm.h"

LabelingTrainer::LabelingTrainer(PyObject* args, PyObject* train_state)
    : Trainer<TokenLabelingModel>(args, train_state)
{
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

        // std::cout << "prediction"<< std::endl;
        auto y_pred = model->forward(x);
        y_pred = y_pred.permute({0, 2, 1});
        // std::cout << "pred size: ";
        // std::cout << y_pred.sizes() << std::endl;

        // std::cout << "true size: " << y_true.sizes() << std::endl;
        auto loss_function = nn::CrossEntropyLoss();
        auto loss = loss_function(y_pred, y_true);
        loss.backward();

        // std::cout << "gradient"<< std::endl;
        optimizer->step();

        // std::cout << "running loss" << std::endl;
        running_loss += (loss.item().toDouble() - running_loss) / (batch_idx + 1);

        batch_idx += 1;
    }

    t_loss_list.append(running_loss);

    std::cout << "current_train_loss: " << running_loss << std::endl;
}

void LabelingTrainer::ValidateBatch(std::map<std::string, at::Tensor>& batch_tensor)
{
    model->eval();

    double running_loss = 0;

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

        running_loss += (loss.item().toDouble() - running_loss) / (batch_idx + 1);

        // std::cout.flush();
        batch_idx += 1;
    }

    v_loss_list.append(running_loss);

    std::cout << "current_val_loss: " << running_loss << std::endl;
}