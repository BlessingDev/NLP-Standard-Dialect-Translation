# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = torch.sub(log_values, a_star).logcumsumexp(dim = 1)
    log_h = torch.add(a_star, log_h0_plus_b_star)
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (torch.add(F.relu(x), 0.5)).log(), torch.mul(F.softplus(torch.mul(x, -1)), -1))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False) if proj_out else Identity()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            z_log = torch.sigmoid(gate).log()
            
            log_coeffs = torch.mul(F.softplus(gate), -1)
            hidden_log = torch.add(F.relu(hidden), 0.5).log()
            log_values = torch.add(hidden_log, z_log)

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
