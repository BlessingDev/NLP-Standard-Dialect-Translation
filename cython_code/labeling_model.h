#include <torch/torch.h>

namespace F = torch::nn::functional;
namespace nn = torch::nn;

struct TokenLabelingModel : torch::nn::Module
{
    torch::nn::Embedding source_embedding;
    torch::nn::GRU rnn;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    float dropout_prob;

    TokenLabelingModel(int num_embeddings, int embedding_size, int rnn_hidden_size, int class_num, double dropout_prob=0.5f)
        : torch::nn::Module(), dropout_prob(dropout_prob)
    {
        // embedding 층
        auto embedding_opt = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_size);
        embedding_opt.padding_idx(0);
        source_embedding = register_module("embedding", torch::nn::Embedding(embedding_opt));

        auto rnn_opt = torch::nn::GRUOptions(embedding_size, rnn_hidden_size);
        rnn_opt.batch_first(true);
        rnn = register_module("GRU", torch::nn::GRU(rnn_opt));

        auto linear_opt = torch::nn::LinearOptions(rnn_hidden_size, rnn_hidden_size);
        fc1 = register_module("Linear1", torch::nn::Linear(linear_opt));

        linear_opt = torch::nn::LinearOptions(rnn_hidden_size, class_num);
        fc2 = register_module("Linear2", torch::nn::Linear(linear_opt));
    }

    torch::Tensor forward(torch::Tensor x_source, torch::Tensor x_lengths, bool apply_softmax=false)
    {
        auto x_embedded = source_embedding(x_source);

        auto rnn_out_tuple = rnn(x_embedded);
        auto rnn_out = std::get<0>(rnn_out_tuple);

        auto y_out = F::dropout(rnn_out, F::DropoutFuncOptions().p(this->dropout_prob));
        y_out = F::relu(fc1(y_out));
        y_out = F::dropout(y_out, F::DropoutFuncOptions().p(this->dropout_prob));
        y_out = F::relu(fc2(y_out));

        if (apply_softmax)
        {
            y_out = F::softmax(y_out, F::SoftmaxFuncOptions(y_out.sizes()[1]));
        }

        return y_out;
    }
};