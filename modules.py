import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.bidirectional = bidirectional

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)

        if not self.bidirectional:
            # Causality - Future Blinding
            diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
            causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
            matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                           matmul_output_m1)  # (h*N, T_q, T_k)

            # Activation
            matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        else:
            matmul_output_sm = self.softmax(matmul_output_m1)  # (h*N, T_q, T_k)
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.ln_1 = nn.LayerNorm(config['hidden_size'])
        self.ln_2 = nn.LayerNorm(config['hidden_size'])
        self.ln_3 = nn.LayerNorm(config['hidden_size'])
        self.mh_attn = MultiHeadAttention(config['hidden_size'], config['hidden_size'], config['num_heads'], config['dropout'], config.get('bidirectional', False))
        self.feed_forward = PositionwiseFeedForward(config['hidden_size'], config['hidden_size'], config['dropout'])


        self._reset_parameters()

    def _reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, seq, mask):
        seq = seq * mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        seq = seq + mh_attn_out  # Residual connection
        ff_out = self.feed_forward(self.ln_2(seq))
        ff_out *= mask
        seq = seq + ff_out
        ff_out = self.ln_3(seq)
        return ff_out

def in_batch_negative_sampling(batch_items):
    B = batch_items.size(0)
    expanded_items = batch_items.unsqueeze(0).repeat(B, 1)
    mask = ~torch.eye(B, dtype=torch.bool, device=batch_items.device)
    neg_items = expanded_items[mask].view(B, B - 1)
    return neg_items


import torch


def in_batch_negative_sampling_sample(batch_items, num_neg=16):
    B = batch_items.size(0)
    expanded_items = batch_items.unsqueeze(0).repeat(B, 1)
    mask = ~torch.eye(B, dtype=torch.bool, device=batch_items.device)
    neg_items = expanded_items.masked_select(mask).view(B, B - 1)
    weights = torch.ones_like(neg_items, dtype=torch.float)
    sample_indices = torch.multinomial(weights, num_neg, replacement=False)
    neg_sampled = neg_items.gather(1, sample_indices)
    return neg_sampled


def extract_axis_1(data, indices):
    """
    Extracts elements from axis 1 based on the provided indices.
    """
    return torch.stack([data[i, indices[i], :] for i in range(data.shape[0])], dim=0).unsqueeze(1)

def diagonalize_and_scale(e, epsilon=1e-7):
    var_e = torch.cov(e.T)
    mean_e = torch.mean(e, axis=0)
    eigvals, eigvecs = torch.linalg.eigh(var_e)
    eigvals = eigvals + epsilon
    D = torch.diag(1.0 / torch.sqrt(eigvals))
    O = eigvecs
    transformed_e = (e - mean_e) @ O @ D

    return transformed_e

# Diffusion
import torch

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


# UniSRec
class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class PLMEmb:
    def __init__(self, config, plm_embeddings):
        self.item_drop_ratio = config['item_drop_ratio']
        self.plm_embeddings = plm_embeddings

    def __call__(self, interaction):
        '''Sequence augmentation and PLM embedding fetching
        '''
        item_seq_len = interaction['item_length']
        item_seq = interaction['item_id_list']

        item_emb_seq = self.plm_embeddings[item_seq]
        pos_item_id = interaction['item_id']
        pos_item_emb = self.plm_embeddings[pos_item_id]

        mask_p = torch.full_like(item_seq, 1 - self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)

        # Augmentation
        seq_mask = item_seq.eq(0).to(torch.bool)
        mask = torch.logical_or(mask, seq_mask)
        mask[:, 0] = True
        drop_index = torch.cumsum(mask, dim=1) - 1

        item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
        item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
        item_emb_seq_aug = self.plm_embeddings[item_seq_aug]

        interaction.update({
            'item_emb_list': item_emb_seq,
            'pos_item_emb': pos_item_emb,
            'item_id_list_aug': item_seq_aug,
            'item_length_aug': item_seq_len_aug,
            'item_emb_list_aug': item_emb_seq_aug,
        })

        return interaction


class PointwiseAggregatedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TODO: add relative attention bias based on time
        self.rab_p = RelativeAttentionBias(num_heads, relative_attention_num_buckets=32,
                                           relative_attention_max_distance=128)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None):
        batch_size = q.shape[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        # attention_scores=torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        rab = self.rab_p(q.shape[2], k.shape[2], device=q.device)

        att_w_bias = attention_scores + rab

        av = (F.silu(att_w_bias) @ v)
        return av.transpose(1, 2).flatten(2)


class RelativeAttentionBias(nn.Module):
    def __init__(self, num_heads, relative_attention_num_buckets, relative_attention_max_distance=128):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    def forward(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    # https://github.com/huggingface/transformers/blob/6cdbd73e01a9719bfaec07d91fd108e8d932bbbb/src/transformers/models/t5/modeling_t5.py#L384
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


class HSTUBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_model * 4)  # Transform and split
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, num_heads)
        self.f2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def split(self, x):
        u, v, q, k = x.chunk(4, dim=-1)
        return u, v, q, k

    def forward(self, x):
        # Pointwise Projection
        x_proj = F.silu(self.f1(x))
        u, v, q, k = self.split(x_proj)

        # Spatial Aggregation
        av = self.pointwise_attn(v, k, q)

        # Pointwise Transformation
        y = self.f2(self.norm(av * u))

        return y