from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

import torch
import torch.nn as nn


class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """
        매개변수:
            num_embeddings (int): 임베딩 개수는 소스 어휘 사전의 크기입니다
            embedding_size (int): 임베딩 벡터의 크기
            rnn_hidden_size (int): RNN 은닉 상태 벡터의 크기
        """
        super(NMTEncoder, self).__init__()
    
        self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)
    
    def forward(self, x_source, x_lengths):
        """ 모델의 정방향 계산
        
        매개변수:
            x_source (torch.Tensor): 입력 데이터 텐서
                x_source.shape는 (batch, seq_size)이다.
            x_lengths (torch.Tensor): 배치에 있는 아이템의 길이 벡터
        반환값:
            튜플: x_unpacked (torch.Tensor), x_birnn_h (torch.Tensor)
                x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
                x_birnn_h.shape = (batch, rnn_hidden_size * 2)
        """
        x_embedded = self.source_embedding(x_source)
        # PackedSequence 생성; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(x_embedded, x_lengths.detach().cpu().numpy(), 
                                        batch_first=True)
        
        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        x_birnn_out, x_birnn_h  = self.birnn(x_packed)
        # (batch_size, num_rnn, feature_size)로 변환
        x_birnn_h = x_birnn_h.permute(1, 0, 2)
        
        # 특성 펼침; (batch_size, num_rnn * feature_size)로 바꾸기
        # (참고: -1은 남은 차원에 해당합니다, 
        #       두 개의 RNN 은닉 벡터를 1로 펼칩니다)
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)
        
        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)
        
        return x_unpacked, x_birnn_h

def verbose_attention(encoder_state_vectors, query_vector):
    """ 원소별 연산을 사용하는 어텐션 메커니즘 버전
    
    매개변수:
        encoder_state_vectors (torch.Tensor): 인코더의 양방향 GRU에서 출력된 3차원 텐서
        query_vector (torch.Tensor): 디코더 GRU의 은닉 상태
    """
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), 
                              dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores

def terse_attention(encoder_state_vectors, query_vector):
    """ 점곱을 사용하는 어텐션 메커니즘 버전
    
    매개변수:
        encoder_state_vectors (torch.Tensor): 인코더의 양방향 GRU에서 출력된 3차원 텐서
        query_vector (torch.Tensor): 디코더 GRU의 은닉 상태
    """
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1), 
                                   vector_probabilities.unsqueeze(dim=2)).squeeze()
    return context_vectors, vector_probabilities


class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index, eos_index, max_seq_length):
        """
        매개변수:
            num_embeddings (int): 임베딩 개수는 타깃 어휘 사전에 있는 고유한 단어의 개수이다
            embedding_size (int): 임베딩 벡터 크기
            rnn_hidden_size (int): RNN 은닉 상태 크기
            bos_index(int): begin-of-sequence 인덱스
        """
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                             embedding_dim=embedding_size, 
                                             padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size, 
                                   rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.max_seq_length = max_seq_length
    
    def _init_indices(self, batch_size):
        """ BEGIN-OF-SEQUENCE 인덱스 벡터를 반환합니다 """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    def _init_context_vectors(self, batch_size):
        """ 문맥 벡터를 초기화하기 위한 0 벡터를 반환합니다 """
        return torch.zeros(batch_size, self._rnn_hidden_size)
            
    def forward(self, encoder_state, initial_hidden_state, target_sequence):
        """ 모델의 정방향 계산
        
        매개변수:
            encoder_state (torch.Tensor): NMTEncoder의 출력
            initial_hidden_state (torch.Tensor): NMTEncoder의 마지막 은닉 상태
            target_sequence (torch.Tensor): 타깃 텍스트 데이터 텐서
        반환값:
            output_vectors (torch.Tensor): 각 타임 스텝의 예측 벡터
        """    
        # 가정: 첫 번째 차원은 배치 차원입니다
        # 즉 입력은 (Batch, Seq)
        # 시퀀스에 대해 반복해야 하므로 (Seq, Batch)로 차원을 바꿉니다
        target_sequence = target_sequence.permute(1, 0)
        output_sequence_size = target_sequence.size(0)

        # 주어진 인코더의 은닉 상태를 초기 은닉 상태로 사용합니다
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)
        # 문맥 벡터를 0으로 초기화합니다
        context_vectors = self._init_context_vectors(batch_size)
        # 첫 단어 y_t를 BOS로 초기화합니다
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
        for i in range(output_sequence_size):
            y_t_index = target_sequence[i]
                
            # 단계 1: 단어를 임베딩하고 이전 문맥과 연결합니다
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
            
            # 단계 2: GRU를 적용하고 새로운 은닉 벡터를 얻습니다
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())
            
            # 단계 3: 현재 은닉 상태를 사용해 인코더의 상태를 주목합니다
            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state, 
                                                           query_vector=h_t)
            
            # 부가 작업: 시각화를 위해 어텐션 확률을 저장합니다
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            # 단게 4: 현재 은닉 상태와 문맥 벡터를 사용해 다음 단어를 예측합니다
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))
            
            # 부가 작업: 예측 성능 점수를 기록합니다
            output_vectors.append(score_for_y_t_index)
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors
    
    def inference(self, encoder_state, initial_hidden_state):

        # 주어진 인코더의 은닉 상태를 초기 은닉 상태로 사용합니다
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)
        # 문맥 벡터를 0으로 초기화합니다
        context_vectors = self._init_context_vectors(batch_size)
        # 첫 단어 y_t를 BOS로 초기화합니다
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
        idx = 0
        generation_finished = False
        while idx < self.max_seq_length and not generation_finished:
            if len(output_vectors) > 0:
                y_t_index = output_vectors[-1].max(dim=1).indices
            
            # 단계 1: 단어를 임베딩하고 이전 문맥과 연결합니다
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
            
            # 단계 2: GRU를 적용하고 새로운 은닉 벡터를 얻습니다
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())
            
            # 단계 3: 현재 은닉 상태를 사용해 인코더의 상태를 주목합니다
            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state, 
                                                           query_vector=h_t)
            
            # 부가 작업: 시각화를 위해 어텐션 확률을 저장합니다
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            # 단게 4: 현재 은닉 상태와 문맥 벡터를 사용해 다음 단어를 예측합니다
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))
            
            # 부가 작업: 예측 성능 점수를 기록합니다
            output_vectors.append(score_for_y_t_index)

            idx += 1
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors


class NMTModel(nn.Module):
    """ 신경망 기계 번역 모델 """
    def __init__(self, source_vocab_size, source_embedding_size, 
                 target_vocab_size, target_embedding_size, encoding_size, 
                 target_bos_index, target_eos_index, 
                 max_gen_length=256):
        """
        매개변수:
            source_vocab_size (int): 소스 언어에 있는 고유한 단어 개수
            source_embedding_size (int): 소스 임베딩 벡터의 크기
            target_vocab_size (int): 타깃 언어에 있는 고유한 단어 개수
            target_embedding_size (int): 타깃 임베딩 벡터의 크기
            encoding_size (int): 인코더 RNN의 크기
            target_bos_index (int): BEGIN-OF-SEQUENCE 토큰 인덱스
        """
        super(NMTModel, self).__init__()
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size, 
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(num_embeddings=target_vocab_size, 
                                  embedding_size=target_embedding_size, 
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index,
                                  eos_index=target_eos_index,
                                  max_seq_length=max_gen_length)
    
    def forward(self, x_source, x_source_lengths, target_sequence=None):
        """ 모델의 정방향 계산
        
        매개변수:
            x_source (torch.Tensor): 소스 텍스트 데이터 텐서
                x_source.shape는 (batch, vectorizer.max_source_length)입니다.
            x_source_lengths torch.Tensor): x_source에 있는 시퀀스 길이
            target_sequence (torch.Tensor): 타깃 텍스트 데이터 텐서
        반환값:
            decoded_states (torch.Tensor): 각 출력 타임 스텝의 예측 벡터
        """
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        if target_sequence:
            decoded_states = self.decoder(encoder_state=encoder_state, 
                                        initial_hidden_state=final_hidden_states, 
                                        target_sequence=target_sequence)
        else:
            decoded_states = self.decoder.inference(encoder_state=encoder_state, 
                                        initial_hidden_state=final_hidden_states)
        return decoded_states