#ifndef __LABELING_MODEL_H__
#define __LABELING_MODEL_H__

#include <torch/torch.h>

namespace F = torch::nn::functional;
namespace nn = torch::nn;

struct TokenLabelingModelImpl : public nn::Module
{
    torch::nn::Embedding source_embedding = nullptr;
    torch::nn::GRU rnn = nullptr;
    torch::nn::Linear fc1 = nullptr;
    torch::nn::Linear fc2 = nullptr;
    torch::nn::Linear out_layer = nullptr;
    float dropout_prob;

    TokenLabelingModelImpl(int num_embeddings, int embedding_size, int rnn_hidden_size, int class_num, double dropout_prob=0.5f)
        : dropout_prob(dropout_prob), source_embedding(torch::nn::EmbeddingOptions(num_embeddings, embedding_size).padding_idx(0)),
        rnn(torch::nn::GRUOptions(embedding_size, rnn_hidden_size).batch_first(true)),
        fc1(torch::nn::LinearOptions(rnn_hidden_size, rnn_hidden_size)),
        fc2(torch::nn::LinearOptions(rnn_hidden_size, rnn_hidden_size)),
        out_layer(torch::nn::LinearOptions(rnn_hidden_size, class_num))
    {
        register_module("embedding", source_embedding);
        register_module("rnn", rnn);
        register_module("linear1", fc1);
        register_module("linear2", fc2);
        register_module("out_layer", out_layer);
    }

    torch::Tensor forward(torch::Tensor x_source, bool apply_softmax=false)
    {
        auto x_embedded = source_embedding(x_source);

        auto rnn_out_tuple = rnn(x_embedded);
        // (batch_size, seq_length, hidden_size)
        auto rnn_out = std::get<0>(rnn_out_tuple);

        auto intermediate = F::relu(fc1(rnn_out));
        intermediate = F::dropout(intermediate, F::DropoutFuncOptions().p(this->dropout_prob));
        intermediate = fc2(intermediate);
        intermediate = F::dropout(intermediate, F::DropoutFuncOptions().p(this->dropout_prob));

        auto y_out = out_layer(intermediate);

        // y_out shape: (batch_size, seq_length, class_num)
        if (apply_softmax)
        {
            auto softOpt = F::SoftmaxFuncOptions(2);
            y_out = F::softmax(y_out, softOpt);
        }

        return y_out;
    }
};

TORCH_MODULE(TokenLabelingModel);

#endif