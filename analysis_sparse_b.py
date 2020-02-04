import pickle
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
outputs = pickle.load(open('/mnt/nfs/work1/mfiterau/brawat/bionlp/sparsification/sparsification/outputs_b.pkl', 'rb'))

input = torch.tensor(outputs[0])
labels = outputs[1]
sparse_embeds = torch.tensor(outputs[2])


for ind, inp in enumerate(input):
    tmp_tokens = [tokenizer._convert_id_to_token(x) for x in input[0].tolist()]
    tmp_tokens = [x for x in tmp_tokens if(x!='[PAD]')]
    actual_str = tokenizer.convert_tokens_to_string(tmp_tokens)
    tmp_embeds = sparse_embeds[ind]
    tmp_embeds = tmp_embeds[:len(tmp_tokens)]
    cls_embed = tmp_embeds[0]
    max_idx = np.argmax(cls_embed.tolist())
    print('=='*10)
    print('Actual Datapoint: ', actual_str)
    print('Max Index: ', max_idx)
    print('Max Val: ', cls_embed[max_idx])

    closest_words = []
    for idx, token in tmp_tokens:
        closest_words.append((token, tmp_embeds[idx].tolist()[max_idx]))
    print('Closest words: ')
    # print('=='*10)
    closest_list = sorted(closest_words, key=lambda x: x[1], reverse=True)
    for i in closest_list:
        print(i)
