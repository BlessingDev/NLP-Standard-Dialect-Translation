#include "labeling_inferencer.h"
#include "metric.h"

LabelingInferencer::LabelingInferencer(PyObject* args, PyObject* eval_dict)
    : Inferencer<TokenLabelingModel>(args, eval_dict)
{
    batch_size = p::extract<int>(args_dict["batch_size"]);
    threshold = p::extract<double>(args_dict.get<std::string, double>("threshold", 0.5f));
    mask_index = p::extract<int>(args_dict["mask_index"]);

    InitModel();
}

LabelingInferencer::~LabelingInferencer()
{

}

void LabelingInferencer::InitModel()
{
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

at::Tensor LabelingInferencer::InferenceSingleBatch(std::map<std::string, at::Tensor>& batch_tensor, int batch_idx)
{
    auto x_batch = batch_tensor["x"];
    
    auto x = x_batch[batch_idx];

    x = x.to(cur_device, c10::ScalarType::Int);

    auto y_pred = model->forward(x);

    // 성능 평가
    auto target_batch = batch_tensor["y_target"];
    double batch_accuracy = 0;
    double batch_f1 = 0;

    for (int i = 0; i < batch_size; i += 1)
    {
        std::tuple<at::Tensor, at::Tensor> max_res = at::_ops::max_dim::call(y_pred[i], 1, false);
        auto pred_idx = std::get<1>(max_res);

        double cur_acc = ComputeAccuracy(pred_idx, x, target_batch[i], mask_index);
        double cur_f1 = ComputeF1Binary(pred_idx, target_batch[i], threshold);

        batch_accuracy += (cur_acc - batch_accuracy) / (i + 1);
        batch_f1 += (cur_f1 - batch_f1) / (i + 1);
    }
    
    accuracy += (batch_accuracy - accuracy) / (batch_idx + 1);
    f1 += (f1 - batch_f1) / (batch_idx + 1);

    eval_dict["acc."] = accuracy;
    eval_dict["f1_score"] = f1;

    y_pred = y_pred.detach().cpu();

    return y_pred;
}