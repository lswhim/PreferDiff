import torch
import torch.nn as nn
import tqdm

from modules import SinusoidalPositionEmbeddings, diagonalize_and_scale, in_batch_negative_sampling_sample, in_batch_negative_sampling
from diffusion import PreferenceDiffusion
from models.SASRec._model import SASRec

class PDSRec(SASRec):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=config['hidden_size'],
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.diff = PreferenceDiffusion(config=config)
        self.step_nn = nn.Sequential(
            SinusoidalPositionEmbeddings(config['hidden_size']),
            nn.Linear(config['hidden_size'], config['hidden_size'] * 2),
            nn.GELU(),
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
        )
        self.denoise_nn = nn.Sequential(
            nn.Linear(config['hidden_size'] * 3, config['hidden_size'])
        )
    def get_embeddings(self, items):
        if self.config.get('ab', None) == 'iids':
            return self.item_embeddings(items)
        else:
            return self.item_embeddings[items].to(items.device)

    def get_all_embeddings(self, device=None):
        if self.config.get('ab', None) == 'iids':
            return self.item_embeddings.weight.data
        else:
            return self.item_embeddings.to(device)

    def get_current_embeddings(self, device=None):
        if self.config.get('ab', None) == 'iids':
            return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]
        else:
            return self.item_embeddings.to(device)

    def load_item_embeddings(self):
        import pickle
        if self.config.get('ab', None) == 'iids':
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.config['item_num'] + 1,
                embedding_dim=self.config['hidden_size'],
                padding_idx=0
            )
            nn.init.normal_(self.item_embeddings.weight, 0, 1)
        elif self.config.get('ab', None) == 'single':
            single_domain = self.config['sd']
            self.item_embeddings = diagonalize_and_scale(diagonalize_and_scale(torch.tensor(
                pickle.load(
                    open(
                        f"./data/{self.config['source_dict'][single_domain]}/{self.config['embedding']}_item_embedding.pkl",
                        'rb'))
            ).float()))
            random_embedding = torch.randn_like(self.item_embeddings)[0, :].reshape(1, -1)
            self.item_embeddings = torch.cat([random_embedding, self.item_embeddings], dim=0)

        else:
            self.item_embeddings = diagonalize_and_scale(torch.cat([
                diagonalize_and_scale(torch.tensor(
                    pickle.load(
                        open(
                            f"./data/{self.config['source_dict'][domain]}/{self.config['embedding']}_item_embedding.pkl",
                            'rb'))
                ).float())
                for domain in self.config['source_dict']
            ], dim=0))


            random_embedding = torch.randn_like(self.item_embeddings)[0, :].reshape(1, -1)
            self.item_embeddings = torch.cat([random_embedding, self.item_embeddings], dim=0)
    def calcu_h(self, logits, p):
        B, D = logits.shape[0], logits.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(logits.device)
        h = logits * mask + self.none_embedding(torch.tensor([0]).to(logits.device)) * (1 - mask)
        return h

    def denoise_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(x.device))
        h = torch.cat([h.view(1, x.shape[1])] * x.shape[0], dim=0)
        return self.denoise(x, h, step)

    def denoise(self, x, h, step):
        t = self.step_nn(step)
        if len(x.shape) < 3:
            return self.denoise_nn(torch.cat((x, h, t), dim=1))
        else:
            B, N, D = x.shape
            x = x.view(-1, D)
            h_expanded = h.unsqueeze(1).repeat(1, N, 1).view(-1, D)
            t_expanded = t.unsqueeze(1).repeat(1, N, 1).view(-1, D)
            input = torch.cat((x, h_expanded, t_expanded), dim=1)
            return self.denoise_nn(input).view(B, N, D)

    def forward(self, batch):
        state_hidden = self.get_representation(batch)
        labels_neg = self._generate_negative_samples(batch)
        labels = batch['labels'].view(-1)
        x_start = self.get_embeddings(labels)
        x_start_neg = self.get_embeddings(labels_neg)
        h = self.calcu_h(state_hidden, self.config['p'])
        n = torch.randint(0, self.config['timesteps'], (labels.shape[0],), device=h.device).long()
        loss, _ = self.diff.p_losses(self, x_start, x_start_neg, h, n, loss_type=self.config['loss_type'])
        return {'loss': loss}

    def predict(self, batch, n_return_sequences=1):
        state_hidden = self.get_representation(batch)
        x = self.diff.sample(self, state_hidden)
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))[:,
                 self.config['select_pool'][0]: self.config['select_pool'][1]]
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        if self.config['exp_type'] == 'check':
            self.samples = self.get_samples(batch)
            self.target_embedding = self.get_embeddings(torch.tensor([30724]).to(x.device))
            self.predict_embeddings = self.get_embeddings(preds)
        return preds

    def get_samples(self, batch):
        state_hidden = self.get_representation(batch)
        samples = []
        for i in tqdm.tqdm(range(1000)):
            x = self.diff.sample(self, state_hidden)
            samples.append(x.detach().cpu().numpy())
        return samples

    def _generate_negative_samples(self, batch):
        if self.config['sample_func'] == 'batch':
            return in_batch_negative_sampling(batch['labels'])
        elif self.config['sample_func'] == 'random':
            return in_batch_negative_sampling_sample(batch['labels'], self.config['neg_samples'])
        labels_neg = []
        for index in range(len(batch['labels'])):
            import numpy as np
            neg_samples = np.random.choice(range(self.config['select_pool'][0], self.config['select_pool'][1]), size=1,
                                           replace=False)
            neg_samples = neg_samples[neg_samples != batch['labels'][index]]
            labels_neg.append(neg_samples.tolist())
        return torch.LongTensor(labels_neg).to(batch['labels'].device).reshape(-1, 1)

