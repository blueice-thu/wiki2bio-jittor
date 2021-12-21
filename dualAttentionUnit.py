import numpy as np
import jittor as jt

class dualAttentionWrapper(jt.Module):
    def __init__(self, hidden_size, input_size, field_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size

        self.Wh = jt.rand([input_size, hidden_size])
        self.bh = jt.zeros([hidden_size])
        self.Ws = jt.rand([input_size, hidden_size])
        self.bs = jt.zeros([hidden_size])
        self.Wo = jt.rand([2 * input_size, hidden_size])
        self.bo = jt.zeros([hidden_size])
        self.Wf = jt.rand([field_size, hidden_size])
        self.bf = jt.zeros([hidden_size])
        self.Wr = jt.rand([input_size, hidden_size])
        self.br = jt.zeros([hidden_size])

        self.params = {'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
            'bh': self.bh, 'bs': self.bs, 'bo': self.bo,
            'Wf': self.Wf, 'Wr': self.Wr, 
            'bf': self.bf, 'br': self.br}
    
    def execute(self, x, hs, fds, finished = None):
        self.hs = jt.transpose(hs, [1,0,2]) # input_len * batch * input_size
        self.fds = jt.transpose(fds, [1,0,2])
        hs2d = jt.reshape(self.hs, [-1, self.input_size])
        phi_hs2d = jt.tanh(jt.nn.matmul(hs2d, self.Wh) + self.bh)
        self.phi_hs = jt.reshape(phi_hs2d, self.hs.shape)
        fds2d = jt.reshape(self.fds, [-1, self.field_size])
        phi_fds2d = jt.tanh(jt.nn.matmul(fds2d, self.Wf) + self.bf)
        self.phi_fds = jt.reshape(phi_fds2d, self.hs.shape)

        gamma_h = jt.tanh(jt.nn.matmul(x, self.Ws) + self.bs)  # batch * hidden_size
        alpha_h = jt.tanh(jt.nn.matmul(x, self.Wr) + self.br)
        fd_weights = jt.sum(self.phi_fds * alpha_h, dim=2, keepdims=True)
        fd_weights = jt.exp(fd_weights - jt.max(fd_weights, dim=0, keepdims=True))
        fd_weights = jt.divide(fd_weights, (1e-6 + jt.sum(fd_weights, dim=0, keepdims=True)))
        
        
        weights = jt.sum(self.phi_hs * gamma_h, dim=2, keepdims=True)  # input_len * batch
        weights = jt.exp(weights - jt.max(weights, dim=0, keepdims=True))
        weights = jt.divide(weights, (1e-6 + jt.sum(weights, dim=0, keepdims=True)))
        weights = jt.divide(weights * fd_weights, (1e-6 + jt.sum(weights * fd_weights, dim=0, keepdims=True)))
        
        context = jt.sum(self.hs * weights, dim=0)  # batch * input_size
        out = jt.tanh(jt.nn.matmul(jt.contrib.concat([context, x], -1), self.Wo) + self.bo)

        if finished is not None:
            finished = finished.unsqueeze(1)
            out = jt.array(np.where(finished, np.zeros_like(out), np.array(out)))
        return out, weights
    
    def save(self, path):
        jt.save(self.params, path)
    
    def load(self, path):
        params = jt.load(path)
        for param in params:
            self.params[param].assign(params[param])
