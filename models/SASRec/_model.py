import torch
import torch.nn as nn
import numpy as np
from base import AbstractModel
from modules import TransformerEncoder, in_batch_negative_sampling, extract_axis_1




class SASRec(AbstractModel):
    def __init__(self, config: dict):
        super(AbstractModel, self).__init__()
        self.config = config
        self.load_item_embeddings()

        # Initialize embeddings
        self.positional_embeddings = nn.Embedding(
            num_embeddings=config['max_seq_length'],
            embedding_dim=config['hidden_size']
        )

        self.emb_dropout = nn.Dropout(config['dropout'])

        # Initialize Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoder(config)
            for _ in range(config['layer_num'])
        ])

        # Initialize loss function
        if config['loss_type'] == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == "ce":
            self.loss_func = nn.CrossEntropyLoss()

    def load_item_embeddings(self):
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.config['item_num'] + 1,
            embedding_dim=self.config['hidden_size'],
            padding_idx=0
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)

    def get_embeddings(self, items):
        return self.item_embeddings(items)

    def get_all_embeddings(self, device=None):
        return self.item_embeddings.weight.data

    def get_current_embeddings(self, device=None):
        return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]


    def get_representation(self, batch):
        inputs_emb = self.get_embeddings(batch['item_seqs'])
        inputs_emb += self.positional_embeddings(
            torch.arange(self.config['max_seq_length']).to(inputs_emb.device)
        )
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(batch['item_seqs'], 0).float().unsqueeze(-1).to(inputs_emb.device)

        for layer in self.layers:
            seq = layer(seq, mask=mask)

        state_hidden = extract_axis_1(seq, batch['seq_lengths'] - 1).squeeze()
        return state_hidden

    def forward(self, batch):
        state_hidden = self.get_representation(batch)
        labels_neg = self._generate_negative_samples(batch)
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        if self.config['loss_type'] == 'bce':
            labels_neg = labels_neg.view(-1, 1)
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            pos_scores = torch.gather(logits, 1, batch['labels'].view(-1, 1))
            neg_scores = torch.gather(logits, 1, labels_neg)
            pos_labels = torch.ones((batch['labels'].view(-1).shape[0], 1))
            neg_labels = torch.zeros((batch['labels'].view(-1).shape[0], 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(state_hidden.device)
            loss = self.loss_func(scores, labels)

        elif self.config['loss_type'] == 'ce':
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            loss = self.loss_func(logits, batch['labels'].view(-1))
        return {'loss': loss}

    def predict(self, batch, n_return_sequences=1):
        state_hidden = self.get_representation(batch).view(-1, self.config['hidden_size'])
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        scores = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))[:,
                 self.config['select_pool'][0]: self.config['select_pool'][1]]
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        return preds

    def _generate_negative_samples(self, batch):
        if self.config['sample_func'] == 'batch':
            return in_batch_negative_sampling(batch['labels'])

        labels_neg = []
        for index in range(len(batch['labels'])):
            import numpy as np
            neg_samples = np.random.choice(range(self.config['select_pool'][0], self.config['select_pool'][1]), size=1,
                                           replace=False)
            neg_samples = neg_samples[neg_samples != batch['labels'][index]]
            labels_neg.append(neg_samples.tolist())
        return torch.LongTensor(labels_neg).to(batch['labels'].device).reshape(-1, 1)
