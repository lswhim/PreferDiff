import torch
import numpy as np
import torch.nn.functional as F
from modules import extract, betas_for_alpha_bar, linear_beta_schedule, exp_beta_schedule, cosine_beta_schedule

class BaseDiffusion():
    def __init__(self, config):
        self.config = config
        self.timesteps = config['timesteps']
        self.device = config['device']
        self.w = config['w']

        self.betas = self.get_beta_schedule(config['beta_sche'], config['timesteps'], config['beta_start'],
                                            config['beta_end'])
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.init_ddim_variables()

    def get_beta_schedule(self, beta_sche, timesteps, beta_start, beta_end):
        if beta_sche == 'linear':
            return linear_beta_schedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_sche == 'exp':
            return exp_beta_schedule(timesteps=timesteps)
        elif beta_sche == 'cosine':
            return cosine_beta_schedule(timesteps=timesteps)
        elif beta_sche == 'sqrt':
            return torch.tensor(betas_for_alpha_bar(timesteps, lambda t: 1 - np.sqrt(t + 0.0001))).float()
        else:
            raise ValueError("Invalid beta schedule")

    def init_ddim_variables(self):
        indices = list(range(0, self.timesteps + 1, self.config['ddim_step']))
        self.sub_timesteps = len(indices)
        indices_now = [indices[i] - 1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)
        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(
            1. - self.alphas_cumprod_ddim_prev) / self.sqrt_recipm1_alphas_cumprod_ddim
        self.posterior_ddim_coef2 = torch.sqrt(1. - self.alphas_cumprod_ddim_prev) / torch.sqrt(
            1. - self.alphas_cumprod_ddim)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def i_sample(self, model_forward, model_forward_uncon, x, y, t, t_index):
        x_start = (1 + self.w) * model_forward(x, y, t) - self.w * model_forward_uncon(x, t)
        x_t = x
        model_mean = (
                self.posterior_ddim_coef1[t_index] * x_start +
                self.posterior_ddim_coef2[t_index] * x_t
        )
        return model_mean

    @torch.no_grad()
    def sample_from_noise(self, model_forward, model_forward_uncon, h):
        x = torch.randn_like(h).to(h.device)
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0],), n * self.config['ddim_step'], device=h.device, dtype=torch.long)
            x = self.i_sample(model_forward, model_forward_uncon, x, h, step, n)
        return x

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        def cosine_loss(pred_matrix, gt_matrix):
            pred_norm = F.normalize(pred_matrix, p=2, dim=1)
            gt_norm = F.normalize(gt_matrix, p=2, dim=1)
            dot_product = torch.sum(pred_norm * gt_norm, dim=1)
            loss = torch.mean((dot_product - 1) ** 2)
            return loss

        def sce_loss(x, y, alpha=0.2):
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

            loss = loss.mean()
            return loss

        predicted_x = denoise_model.denoise(x_noisy, h, t)
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        elif loss_type == "cosine":
            loss = cosine_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()
        return loss, {'main': loss}

    def get_x(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model.denoise(x_noisy, h, t)
        return predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, h, mode='ddim'):
        if mode=='ddpm':
            x = torch.randn_like(h)
            for n in reversed(range(0, self.timesteps)):
                step = torch.full((h.shape[0],), n, device=self.device, dtype=torch.long)
                x = self.p_sample(model.denoise, model.denoise_uncon, x, h, step, n)
        else:
            x = self.sample_from_noise(model.denoise, model.denoise_uncon, h)
        return x

    @torch.no_grad()
    def ddim_sample(self, model_forward, model_forward_uncon, h, ddim_timesteps=100, ddim_discr_method='uniform'):
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.arange(0, self.timesteps, c)
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        x_t = torch.randn_like(h)
        ddim_timestep_seq += 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((h.shape[0],), ddim_timestep_seq[i], device=self.device, dtype=torch.long)
            prev_t = torch.full((h.shape[0],), ddim_timestep_prev_seq[i], device=self.device, dtype=torch.long)
            predicted_x_0 = (1 + self.w) * model_forward(x_t, h, t) - self.w * model_forward_uncon(x_t, t)
            alpha_t = extract(self.alphas, t, x_t.shape)
            alpha_prev_t = extract(self.alphas, prev_t, x_t.shape)
            x_t = torch.sqrt(alpha_prev_t) * predicted_x_0 + torch.sqrt((1 - alpha_t) / alpha_prev_t) * (
                    x_t - torch.sqrt(alpha_prev_t) * predicted_x_0)
        return x_t


class PreferenceDiffusion(BaseDiffusion):
    def __init__(self, config):
        super().__init__(config=config)
        self.gamma = config['gamma']
        self.lamda = config['lamda']
        self.mode = config['mode']
        self.ab = config.get('ab', None)
        self.losses = []
        self.loss_poss = []
        self.loss_negs = []

    def p_losses(self, denoise_model, x_start_pos, x_start_neg, h, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start_pos)

        if loss_type != 'sdpo':
            x_start_neg = torch.mean(x_start_neg, dim=1)

        def cosine_loss(pred_matrix, gt_matrix):
            pred_norm = F.normalize(pred_matrix, p=2, dim=1)
            gt_norm = F.normalize(gt_matrix, p=2, dim=1)

            # 计算点积
            dot_product = torch.sum(pred_norm * gt_norm, dim=1)

            # 计算与1之间的均方误差损失
            loss = torch.mean((dot_product - 1) ** 2)
            # x = F.normalize(pred_matrix, p=2, dim=-1)
            # y = F.normalize(gt_matrix, p=2, dim=-1)
            # loss = (1 - (x * y).sum(dim=-1)).pow_(self.config['tau'])
            #
            # loss = loss.mean()
            return loss


        loss_func = {
            'l1': F.l1_loss,
            'l2': F.mse_loss,
            'huber': F.smooth_l1_loss,
            'sdpo': F.mse_loss,
            'cosine': cosine_loss,
        }.get(loss_type)

        if loss_func is None:
            raise NotImplementedError()

        x_noisy_pos = self.q_sample(x_start=x_start_pos, t=t, noise=noise)
        x_noisy_neg = self.q_sample(x_start=x_start_neg, t=t)

        predicted_x_pos = denoise_model.denoise(x_noisy_pos, h, t)
        predicted_x_neg = denoise_model.denoise(x_noisy_neg, h, t)

        neg_indices = torch.randint(0, h.shape[0], (h.shape[0],))
        neg_h = h[neg_indices]

        # 负样本去噪
        predicted_x_neg_h = denoise_model.denoise(x_noisy_pos, neg_h, t)

        loss_pos = loss_func(x_start_pos, predicted_x_pos)

        if loss_type != 'sdpo':
            loss_neg = loss_func(x_start_neg, predicted_x_neg)
            loss_neg_h = loss_func(x_start_pos, predicted_x_neg_h)
        else:
            loss_neg = F.mse_loss(x_start_neg, predicted_x_neg, reduction='none').sum(dim=2)
            B, N = loss_neg.shape
            loss_pos_expand = loss_pos.expand(B, N)
            loss_diff = torch.log(loss_neg - loss_pos_expand)
            loss = -F.logsigmoid(
                -torch.log(torch.mean(torch.sum(torch.exp(self.gamma * loss_diff), dim=1)) + 1e-8) + 1e-8) + loss_pos
            return loss, {'main': loss.detach().cpu()}

        model_diff = loss_pos - loss_neg

        if self.mode == 'neg':
            # loss = -1 * F.logsigmoid(-self.gamma * model_diff + 1e-8) + loss_pos
            loss = -(1 - self.lamda) * F.logsigmoid(-self.gamma * model_diff + 1e-8) + loss_pos * self.lamda
        elif self.mode == 'negh':
            h_loss = - F.logsigmoid(-0.01 * (loss_pos - loss_neg_h) + 1e-8)
            loss = loss_pos + h_loss + -F.logsigmoid(-self.gamma * model_diff + 1e-8)
        elif self.mode == 'dpo':
            loss = -F.logsigmoid(0 + 1e-8) + loss_pos
        elif self.mode == 'neg_sample':
            loss = -F.logsigmoid(-self.gamma * model_diff + 1e-8)
        else:
            raise ValueError('Unsupported mode')

        if self.ab == 'to':
           loss = loss_func(x_start_pos, predicted_x_pos)
        self.losses.append(loss.detach().cpu())
        self.loss_poss.append(loss_pos.detach().cpu())
        self.loss_negs.append(loss_neg.detach().cpu())

        loss_dict = {'main': loss.detach().cpu(), 'pos': loss_pos.detach().cpu(), 'neg': loss_neg.detach().cpu()}
        return loss, loss_dict
