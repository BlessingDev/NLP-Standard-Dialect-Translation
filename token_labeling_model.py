from torch.nn import functional as F

import torch
import torch.nn as nn

class TokenLabelingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, class_num, dropout_prob=0.5):
        """
        매개변수:
            num_embeddings (int): 임베딩 개수는 소스 어휘 사전의 크기입니다
            embedding_size (int): 임베딩 벡터의 크기
            rnn_hidden_size (int): RNN 은닉 상태 벡터의 크기
        """
        super(TokenLabelingModel, self).__init__()
    
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.rnn = nn.GRU(embedding_size, rnn_hidden_size, batch_first=True)
        self.linear1 = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.linear2 = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.out_layer = nn.Linear(rnn_hidden_size, class_num)

    def forward(self, x_source, apply_softmax=False):
        x_embedded = self.embedding(x_source)

        rnn_out, _ = self.rnn(x_embedded)

        intermediate = F.relu(self.linear1(rnn_out))
        intermediate = F.dropout(intermediate, p=self.dropout_prob)
        intermediate = F.relu(self.linear2(rnn_out))
        intermediate = F.dropout(intermediate, p=self.dropout_prob)

        y_out = self.out_layer(intermediate)

        if apply_softmax:
            y_out = F.softmax(y_out, dim=2)
        
        return y_out