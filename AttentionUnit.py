import jittor as jt
import numpy as np

class AttentionWrapper(jt.Module):
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.Wh = jt.rand([input_size, hidden_size])
        self.bh = jt.zeros([hidden_size])
        self.Ws = jt.rand([input_size, hidden_size])
        self.bs = jt.zeros([hidden_size])
        self.Wo = jt.rand([2 * input_size, hidden_size])
        self.bo = jt.zeros([hidden_size])
    
    def execute(self, x, hs, fds, finished = None):
        self.hs = jt.transpose(hs, [1, 0, 2])
        hs2d = jt.reshape(self.hs, [-1, self.input_size])
        phi_hs2d = jt.tanh(jt.nn.matmul(hs2d, self.Wh) + self.bh)
        self.phi_hs = jt.reshape(phi_hs2d, self.hs.shape)

        gamma_h = jt.tanh(jt.nn.matmul(x, self.Ws) + self.bs)
        weights = jt.sum(self.phi_hs * gamma_h, dim=2, keepdims=True)
        weights = jt.exp(weights - jt.max(weights, dim=0, keepdims=True))
        weights = jt.divide(weights, (1e-6 + jt.sum(weights, dim=0, keepdims=True)))
        context = jt.sum(self.hs * weights, dim=0)
        out = jt.tanh(jt.nn.matmul(jt.contrib.concat([context, x], dim=-1), self.Wo) + self.bo)
        if finished is not None:
            finished = finished.unsqueeze(1)
            out = jt.array(np.where(finished, np.zeros_like(out), np.array(out)))
        return out, weights
