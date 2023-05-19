#pragma once
#include <torch/torch.h>
#include <tuple>

double ComputeAccuracy(at::Tensor y_pred, at::Tensor x_source, at::Tensor y_true, int mask_index)
{
    auto correct_indices = torch::eq(y_pred, y_true).to(c10::ScalarType::Float);
    auto valid_indices = torch::ne(x_source, mask_index).to(c10::ScalarType::Float);

    double n_correnct = (correct_indices * valid_indices).sum().item().toDouble();
    double n_valid = valid_indices.sum().item().toDouble();

    return n_correnct / n_valid * 100;
}

double ComputeF1Binary(at::Tensor y_pred, at::Tensor y_true, double thereshold = 0.5f)
{
    double precision = 0;
    double recall = 0;

    auto pred_result = torch::where(y_pred >= thereshold, 1.0, 0.0);
    double model_true = pred_result.argwhere().sizes()[0];
    auto true_positive_tensor = torch::where((pred_result * y_true) == 1, 1.0, 0.0);
    double true_positive = true_positive_tensor.argwhere().sizes()[0];
    precision = true_positive / model_true;

    double real_true = y_true.argwhere().sizes()[0];
    recall = true_positive / real_true;

    return 2 * (precision * recall) / (precision + recall);
}