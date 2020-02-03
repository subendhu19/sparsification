import torch
import json
from run_snli import BertForSequenceClassificationWithSparsity, SparseBertForSequenceClassification
from transformers import BertTokenizer


def main():
    config = 1

    global sparse_config
    sparse_config = json.load(open('configs/config{}.json'.format(str(config))))
    pretrained_folder = '/mnt/nfs/scratch1/srongali/sparsification/exps/config_{}/checkpoint-15000'.format(str(config))
    if config <= 32:
        model = BertForSequenceClassificationWithSparsity.from_pretrained(pretrained_folder)
    else:
        model = SparseBertForSequenceClassification.from_pretrained(pretrained_folder)

    tokenizer = BertTokenizer.from_pretrained(pretrained_folder)
    input_ids = torch.tensor([tokenizer.encode('this is it'), tokenizer.encode('this is not')])

    print(model.get_sparse_embeddings(input_ids))


if __name__ == "__main__":
    main()
