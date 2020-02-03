import torch
from torch import nn
import json
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import argparse
from run_snli import load_and_cache_examples
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)


class DenoisingAutoencoder(nn.Module):

    def __init__(self, input_dim, sparse_dim=1000, noise_std=0.4):
        super(DenoisingAutoencoder, self).__init__()
        self.hidden = nn.Linear(input_dim, sparse_dim)
        self.out = nn.Linear(sparse_dim, input_dim)
        self.noise_std = noise_std

    def forward(self, x):
        x = x + torch.normal(0, self.noise_std, size=x.size()).to(x.device)
        h = torch.clamp(self.hidden(x), min=0, max=1)
        o = self.out(h)
        return o, h


class BertForSequenceClassificationWithSparsity(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.sparse_net = DenoisingAutoencoder(config.hidden_size, sparse_config['sparse_size'],
                                               sparse_config['sparse_noise_std'])
        if sparse_config['sparse_net_params']:
            self.sparse_net.load_state_dict(torch.load(sparse_config['sparse_net_params']))
        self.sparsity_frac = sparse_config['sparse_frac']
        self.sparsity_imp = sparse_config['sparse_imp']

        self.init_weights()

    def get_sparse_embeddings(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        osize = outputs[0].size()
        all_outputs = outputs[0].reshape(osize[0] * osize[1], self.hidden_size)
        _, sparse_outputs = self.sparse_net(all_outputs)
        return sparse_outputs.reshape(osize[0], osize[1], -1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        osize = outputs[0].size()
        all_outputs = outputs[0].reshape(osize[0] * osize[1], self.hidden_size)
        rec_outputs, sparse_outputs = self.sparse_net(all_outputs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_recon = MSELoss()
            target_sf = sparse_outputs.new_full(sparse_outputs[0].size(), fill_value=self.sparsity_frac)
            loss = self.sparsity_imp * (loss_recon(rec_outputs, all_outputs) +
                                        torch.sum(torch.clamp((sparse_outputs.mean(axis=0) - target_sf), min=0)
                                                  ** 2) + torch.mean(sparse_outputs * (1 - sparse_outputs)))
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss += loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class SparseBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.sparse_size = sparse_config['sparse_size']

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sparse_net = DenoisingAutoencoder(config.hidden_size, sparse_config['sparse_size'],
                                               sparse_config['sparse_noise_std'])
        if sparse_config['sparse_net_params']:
            self.sparse_net.load_state_dict(torch.load(sparse_config['sparse_net_params']))
        self.sparsity_frac = sparse_config['sparse_frac']
        self.sparsity_imp = sparse_config['sparse_imp']

        self.sparse_dense = nn.Linear(sparse_config['sparse_size'], sparse_config['sparse_size'])
        self.sparse_activation = nn.Tanh()
        self.sparse_classifier = nn.Linear(sparse_config['sparse_size'], config.num_labels)

        self.init_weights()

    def get_sparse_embeddings(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        osize = outputs[0].size()
        all_outputs = outputs[0].reshape(osize[0] * osize[1], self.hidden_size)
        _, sparse_outputs = self.sparse_net(all_outputs)
        return sparse_outputs.reshape(osize[0], osize[1], -1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        osize = outputs[0].size()
        all_outputs = outputs[0].reshape(osize[0] * osize[1], self.hidden_size)
        rec_outputs, sparse_outputs = self.sparse_net(all_outputs)

        loss_recon = MSELoss()
        target_sf = sparse_outputs.new_full(sparse_outputs[0].size(), fill_value=self.sparsity_frac)
        loss = self.sparsity_imp * (loss_recon(rec_outputs, all_outputs) + torch.sum(
            torch.clamp((sparse_outputs.mean(axis=0) - target_sf), min=0) ** 2) + torch.mean(
            sparse_outputs * (1 - sparse_outputs)))

        sparse_outputs = sparse_outputs.reshape(osize[0], osize[1], -1)
        sp_first_token_tensor = sparse_outputs[:, 0]
        sp_pooled_output = self.sparse_dense(sp_first_token_tensor)
        sp_pooled_output = self.sparse_activation(sp_pooled_output)
        sp_pooled_output = self.dropout(sp_pooled_output)
        sp_logits = self.sparse_classifier(sp_pooled_output)

        outputs = (sp_logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss += loss_fct(sp_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(sp_logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/mnt/nfs/work1/mfiterau/brawat/snli/data/snli_1.0', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--per_gpu_train_batch_size", default=20, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=20, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    args = parser.parse_args()
    args.task_name = 'xnli'
    args.do_lower_case = True
    args.n_gpu = torch.cuda.device_count()
    args.language = 'english'
    args.local_rank = -1
    args.model_name_or_path = 'bert-base-uncased'
    args.overwrite_cache = False

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Best performing config on test: acc 91.05
    config = 50

    global sparse_config
    sparse_config = json.load(open('configs/config{}.json'.format(str(config))))
    pretrained_folder = '/mnt/nfs/scratch1/srongali/sparsification/snli_exps/config_{}/checkpoint-15000'.format(str(config))
    if config <= 32:
        model = BertForSequenceClassificationWithSparsity.from_pretrained(pretrained_folder)
    else:
        model = SparseBertForSequenceClassification.from_pretrained(pretrained_folder)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Get the actual dataset or examples
    # input_ids = torch.tensor([tokenizer.encode('this is it'), tokenizer.encode('this is not')])

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples(args, 'xnli', tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3],
                      'token_type_ids': batch[2]}
            outputs = model.get_sparse_embeddings(**inputs)
            print(outputs.size())

    # print(model.get_sparse_embeddings(input_ids))


if __name__ == "__main__":
    main()
