import torch
from torch import nn
import torch.nn.functional as F
from methods.template import LDLTemplate, kaiming_normal_init_net


class GDMAM(LDLTemplate):
    def __init__(self, num_feature, num_classes, loss_func, hidden_dim=100,
                 lambda1=0.1, lambda2=0.1, lambda3=0.1,
                 lr=1e-3, weight_decay=1e-4, adjust_lr=False, gradient_clip_value=5.0,
                 max_epoch=None, verbose=False, device='cuda:0'):
        super(GDMAM, self).__init__(num_feature,
                                  num_classes,
                                  adjust_lr=adjust_lr,
                                  gradient_clip_value=gradient_clip_value,
                                  max_epoch=max_epoch,
                                  verbose=verbose,
                                  device=device)
        self.loss_func = loss_func
        self.hidden_dim = hidden_dim
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3

        # enc/dec
        self.encoder_x = nn.Linear(num_feature, hidden_dim)
        self.encoder_x_mu = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_x_log_var = nn.Linear(hidden_dim, hidden_dim)

        self.encoder_y = nn.Linear(num_classes, hidden_dim)
        self.encoder_y_mu = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_y_log_var = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, num_classes)

        kaiming_normal_init_net(self)

        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        self.optimizer = torch.optim.AdamW(
            [{'params': params_decay, 'lr': lr, 'weight_decay': weight_decay},
             {'params': params_no_decay, 'lr': lr}], amsgrad=True)
        self.to(self.device)

        # ---------- 自适应动量与权重 ----------
        self.beta_min, self.beta_max = 0.7, 0.999
        self.beta_pre = 0.95
        self.beta_par = 0.95
        self.beta_cz  = 0.95
        self.k = 0.05
        self._ema = {
            'pre_mean': 0.0, 'pre_sq': 0.0,
            'par_mean': 0.0, 'par_sq': 0.0,
            'cz_mean':  0.0, 'cz_sq':  0.0,
        }
        self._ema_momentum = 0.9
        self.eps = 1e-12

    # ----------------- utils -----------------
    @staticmethod
    def _clip_beta(x, bmin, bmax):
        return float(max(bmin, min(bmax, x)))

    def _ema_update(self, key_mean, key_sq, value: float):
        m = self._ema_momentum
        self._ema[key_mean] = m * self._ema[key_mean] + (1 - m) * value
        self._ema[key_sq]   = m * self._ema[key_sq]   + (1 - m) * (value * value)

    def _ema_std(self, key_mean, key_sq):
        mean = self._ema[key_mean]
        sq   = self._ema[key_sq]
        var = max(0.0, sq - mean * mean)
        return var ** 0.5

    @staticmethod
    def _flatten_pair(grads_a, grads_b, params):
        """
        将两个梯度列表按同一组 params 对齐，None 用 0 填充，返回两个等长的一维向量。
        """
        fa, fb = [], []
        for ga, gb, p in zip(grads_a, grads_b, params):
            if ga is None:
                za = torch.zeros_like(p, memory_format=torch.contiguous_format)
            else:
                za = ga
            if gb is None:
                zb = torch.zeros_like(p, memory_format=torch.contiguous_format)
            else:
                zb = gb
            fa.append(za.reshape(-1))
            fb.append(zb.reshape(-1))
        return torch.cat(fa) if fa else torch.tensor([], device=params[0].device), \
               torch.cat(fb) if fb else torch.tensor([], device=params[0].device)

    # ----------------- forward -----------------
    def set_forward(self, x):
        x = self.encoder_x(x)
        x = self.decoder(torch.relu(x))
        return F.softmax(x, dim=1)

    # -------------- forward + loss --------------
    def set_forward_loss(self, x, y):
        # 编码/采样
        xx = self.encoder_x(x)
        xx_mu = self.encoder_x_mu(xx)
        xx_log_var = self.encoder_x_log_var(xx)
        rand = torch.normal(mean=0., std=1., size=xx_mu.shape, device=self.device)
        dx = xx_mu + rand * torch.exp(xx_log_var).sqrt()

        yy = self.encoder_y(y.float())
        yy_mu = self.encoder_y_mu(yy)
        yy_log_var = self.encoder_y_log_var(yy)
        rand = torch.normal(mean=0., std=1., size=yy_mu.shape, device=self.device)
        dy = yy_mu + rand * torch.exp(yy_log_var).sqrt()

        # 各项损失
        sigma2 = yy_log_var.detach()
        loss_target = -0.5 * torch.mean(torch.sum(
            xx_log_var - sigma2 - torch.exp(xx_log_var) / torch.exp(sigma2)
            - (xx_mu - yy_mu.detach()) ** 2 / torch.exp(sigma2) + 1, dim=1))

        loss_recovery = F.cross_entropy(self.decoder(dy), y)

        sx = self.cosine_similarity(dx, dx).reshape(-1)
        sy = self.cosine_similarity(dy, dy).reshape(-1)
        loss_similarity = F.mse_loss(sx, sy, reduction='sum')

        pred_logits = self.decoder(torch.relu(xx))
        loss_pred = self.loss_func(pred_logits, y)

        # 对齐任务的聚合
        loss_align = (self.lambda1 * loss_recovery
                      + self.lambda2 * loss_target
                      + self.lambda3 * loss_similarity)

        # ================== 动态加权与分解 ==================
        params = [p for p in self.parameters() if p.requires_grad]

        # 单独求梯度（不创建高阶图），并与参数对齐后展平
        g_pred  = torch.autograd.grad(loss_pred,  params, retain_graph=True,  create_graph=False, allow_unused=True)
        g_align = torch.autograd.grad(loss_align, params, retain_graph=True,  create_graph=False, allow_unused=True)
        g_pred_f, g_align_f = self._flatten_pair(g_pred, g_align, params)

        # 范数
        n_pred  = torch.linalg.norm(g_pred_f)  + self.eps
        n_align = torch.linalg.norm(g_align_f) + self.eps

        # 任务级动态权重
        w_pred  = (n_pred / (n_pred + n_align)).item()
        w_align = (n_align / (n_pred + n_align)).item()

        # 平行/正交分解（在参数空间）
        if n_pred.item() <= 1e-12:
            g_par_f = torch.zeros_like(g_pred_f)
        else:
            g_par_f = (torch.dot(g_align_f, g_pred_f) / (n_pred ** 2)) * g_pred_f
        g_cz_f = g_align_f - g_par_f

        n_par = torch.linalg.norm(g_par_f) + self.eps
        n_cz  = torch.linalg.norm(g_cz_f)  + self.eps

        # gate：用余弦相似度截断到 [0,1]
        cos_ap = (torch.dot(g_align_f, g_pred_f) / (n_align * n_pred)).clamp(min=-1.0, max=1.0)
        gate = float(torch.clamp(cos_ap, 0.0, 1.0).item())

        # 自适应动量更新
        self._ema_update('pre_mean', 'pre_sq', float(n_pred.item()))
        self._ema_update('par_mean', 'par_sq',  float(n_par.item()))
        self._ema_update('cz_mean',  'cz_sq',   float(n_cz.item()))

        sigma_pre = self._ema_std('pre_mean', 'pre_sq')
        sigma_par = self._ema_std('par_mean', 'par_sq')
        sigma_cz  = self._ema_std('cz_mean',  'cz_sq')

        self.beta_pre = self._clip_beta(self.beta_pre - self.k * sigma_pre, self.beta_min, self.beta_max)
        self.beta_par = self._clip_beta(self.beta_par - 0.1 * sigma_par,   self.beta_min, self.beta_max)
        self.beta_cz  = self._clip_beta(self.beta_cz  - 0.1 * sigma_cz,    self.beta_min, self.beta_max)

        beta1_eff = w_pred * self.beta_pre + gate * self.beta_par + (1.0 - gate) * self.beta_cz
        for g in self.optimizer.param_groups:
            b1, b2 = g.get('betas', (0.9, 0.999))
            g['betas'] = (self._clip_beta(beta1_eff, self.beta_min, self.beta_max), b2)

        # ================== 最终总损失 ==================
        total_loss = w_pred * loss_pred + w_align * loss_align
        return total_loss
