import pickle
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
import numpy as np
import torch
import logging

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
outputs = pickle.load(open('/mnt/nfs/work1/mfiterau/brawat/bionlp/sparsification/sparsification/outputs_b.pkl', 'rb'))

input = torch.tensor(outputs[0])
labels = outputs[1]
sparse_embeds = torch.tensor(outputs[2])


logger = logging.getLogger(__name__)

for ind, inp in enumerate(input):
    tmp_tokens = [tokenizer._convert_id_to_token(x) for x in inp.tolist()]
    tmp_tokens = [x for x in tmp_tokens if(x!='[PAD]')]
    actual_str = tokenizer.convert_tokens_to_string(tmp_tokens)
    tmp_embeds = sparse_embeds[ind]
    tmp_embeds = tmp_embeds[:len(tmp_tokens)]
    cls_embed = tmp_embeds[0]

    # import ipdb; ipdb.set_trace()
    max_idx = np.argmax(cls_embed.tolist())
    cnt = [1 for x in cls_embed.tolist() if(x == cls_embed[max_idx])]
    logger.info('=='*10)
    print('=='*10)
    logger.info('Actual Datapoint: %s', actual_str)
    print('Actual Datapoint: ', actual_str)
    print('Gold Label: ', labels[ind])
    print('Max Index: ', max_idx)
    print('Max Val: ', cls_embed[max_idx])
    print('Dimensions with same val: ', len(cnt))
    # print('Dimensions with same val: ', np.sum(cnt)/len(cls_embed.tolist()))

    closest_words = []
    cos_ = cosine_similarity(tmp_embeds)
    cos_ = cos_[0, :].tolist()
    for idx, token in enumerate(tmp_tokens):
        # closest_words.append((token, tmp_embeds[idx].tolist()[max_idx]))
        if idx == 0:
            continue
        closest_words.append((token, cos_[idx]))
    print('Closest words: ')
    # print('=='*10)
    closest_list = sorted(closest_words, key=lambda x: x[1], reverse=True)
    for i in closest_list:
        print(i)
