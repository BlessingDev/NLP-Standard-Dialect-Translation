import torch
import numpy as np
from argparse import Namespace

default_args = Namespace(
                seed=1337,
                early_stopping_criteria=3,
                source_embedding_size=64, 
                target_embedding_size=64,
                encoding_size=64)

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

# cython으로 옮긴 코드
'''def get_source_sentence(source_vocab, batch_dict, index):
    indices = batch_dict['x_source'][index].cpu().data.numpy()
    return sentence.sentence_from_indices(indices, source_vocab)

def get_true_sentence(target_vocab, batch_dict, index):
    return sentence.sentence_from_indices(batch_dict['y_target'].cpu().data.numpy()[index], target_vocab)
    
def get_sampled_sentence(model, vectorizer, batch_dict, index):
    y_pred = model(x_source=batch_dict['x_source'], 
                   x_source_lengths=batch_dict['x_source_length'], 
                   target_sequence=batch_dict['x_target'])
    return sentence_from_indices(torch.max(y_pred, dim=2)[1].cpu().data.numpy()[index], vectorizer.target_vocab)

def get_all_sentences(vocab_source, vocab_target, batch_dict, pred, index):
    #vocab_target = vectorizer.target_vocab
    #vocab_sorce = vectorizer.source_vocab
    pred_idx = np.argmax(pred, axis=1)
    return {"source": get_source_sentence(vocab_source, batch_dict, index), 
            "truth": get_true_sentence(vocab_target, batch_dict, index), 
            "pred": sentence.sentence_from_indices(pred_idx, vocab_target)}
    
def sentence_from_indices(indices, vocab, strict=True):
    ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index))
    
    out_sentence = " ".join(out).replace(" ##", "")
    return out_sentence'''