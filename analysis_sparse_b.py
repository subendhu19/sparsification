import pickle
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
import numpy as np
import torch
import logging

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

logger = logging.getLogger(__name__)


def main():
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO)

    logging.basicConfig(format='%(message)s',
                        level=logging.INFO)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    outputs = pickle.load(
        open('/mnt/nfs/work1/mfiterau/brawat/bionlp/sparsification/sparsification/outputs_b.pkl', 'rb'))

    input = torch.tensor(outputs[0])
    labels = ['entailment' for x in range(8)]+['contradiction' for x in range(8)]+['neutral' for x in range(7)]
    sparse_embeds = torch.tensor(outputs[2])

    # import ipdb; ipdb.set_trace()

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
        logger.info('Actual Datapoint: %s', actual_str)
        logger.info('Gold Label: %s', labels[ind])
        logger.info('Max Index: %s', max_idx)
        logger.info('Max Val: %s', cls_embed[max_idx])
        logger.info('Dimensions with same val: %s', len(cnt))
        # logger.info('Dimensions with same val: ', np.sum(cnt)/len(cls_embed.tolist()))

        closest_words = []
        cos_all = cosine_similarity(tmp_embeds)
        for ind_token, curr_token in enumerate(tmp_tokens):
            logger.info('---')
            logger.info('Inspected Token: %s', curr_token)
            cos_ = cos_all[ind_token, :].tolist()
            for idx, token in enumerate(tmp_tokens):
                # closest_words.append((token, tmp_embeds[idx].tolist()[max_idx]))
                if idx == ind_token:
                    continue
                closest_words.append((token, cos_[idx]))
            logger.info('Closest words: ')
            closest_list = sorted(closest_words, key=lambda x: x[1], reverse=True)[:5]
            for i in closest_list:
                if i != '[SEP]' or i != 'a' or i != 'the':
                    logger.info(i)


if __name__ == "__main__":
    main()