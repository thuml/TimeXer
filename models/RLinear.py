import torch
import torch.nn as nn
from layers.StandardNorm import Normalize


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2305.10721
    RLinear: Linear mapping with RevIN
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.rev = Normalize(self.channels, affine=True, non_norm=False)

    def forecast(self, x_enc):
        # x_enc: [Batch, Length, Channel]
        x = self.rev(x_enc, 'norm')
        x = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        dec_out = self.rev(x, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
