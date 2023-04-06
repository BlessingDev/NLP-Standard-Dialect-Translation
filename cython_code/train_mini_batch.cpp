#include "labeling_model.h"
#include <map>
#include <string>
#include <sstream>

void* TrainMiniBatch(std::map<std::string, std::string> args, void* model_void_pointer, std::vector<std::map<std::string, at::Tensor>> batch_list)
{
    TokenLabelingModel* model_pointer = (TokenLabelingModel*)model_void_pointer;

    double lr = 0.001f;
    std::stringstream lrstream(args["learning_rate"]);
    lrstream >> lr;

    auto optimizer = torch::optim::Adam(model_pointer->parameters(), torch::optim::AdamOptions(lr));

    for(auto batch_dict : batch_list)
    {
        auto y_pred = model_pointer->forward(batch_dict["x"], batch_dict["x_length"]);

        
    }
}