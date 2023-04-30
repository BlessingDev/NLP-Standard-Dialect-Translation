#ifndef __LABELING_TRAINER_HPP__
#define __LABELING_TRAINER_HPP__
#include <map>
#include <string>
#include <boost/python.hpp>
#include <torch/torch.h>
#include "trainer.h"
#include "labeling_model.h"

class LabelingTrainer : public Trainer<TokenLabelingModel>
{
private:
    torch::optim::Adam* optimizer = nullptr;
    torch::optim::StepLR* scheduler = nullptr;

private:
    virtual void InitModel() override;

public:
    LabelingTrainer(PyObject* args, PyObject* train_state);

    virtual ~LabelingTrainer();

    virtual void InitTrain() override;

    virtual void StepEpoch() override;

    virtual void TrainBatch(std::map<std::string, at::Tensor>& batch_tensor) override;

    virtual void ValidateBatch(std::map<std::string, at::Tensor>& batch_tensor) override;
};

#endif