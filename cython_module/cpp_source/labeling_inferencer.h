#pragma once
#include "inferencer.h"
#include "labeling_model.h"

class LabelingInferencer 
    : public Inferencer<TokenLabelingModel>
{
private:
    double accuracy = 0;
    double f1 = 0;
    int batch_size = 0;
    int mask_index = 0;
    double threshold = 0.5f;

public:
    LabelingInferencer(PyObject* args, PyObject* eval_dict);

    virtual ~LabelingInferencer();

    virtual void InitModel() override;

    virtual at::Tensor InferenceSingleBatch(std::map<std::string, at::Tensor>& batch_tensor, int batch_idx) override;
};