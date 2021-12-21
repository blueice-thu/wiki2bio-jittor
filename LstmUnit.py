import jittor as jt
import numpy as np

class LstmUnit(jt.Module):
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.linear = jt.nn.Linear(self.input_size+self.hidden_size, 4*self.hidden_size)
        self.params = {}
    
    def execute(self, x, s, finished = None):
        h_prev, c_prev = s

        x = jt.contrib.concat([x, h_prev], 1)
        i, j, f, o = jt.chunk(self.linear(x), 4, 1)

        # Final Memory cell
        c = jt.sigmoid(f+1.0) * c_prev + jt.sigmoid(i) * jt.tanh(j)
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            finished = finished.unsqueeze(1)
            inds = jt.where(finished)[0]
            out[inds] = 0
            state[0][inds] = h_prev[inds]
            state[1][inds] = c_prev[inds]

        return out, state
        
    def save(self, path):
        jt.save(self.params, path)
    
    def load(self, path):
        params = jt.load(path)
        for param in params:
            self.params[param].assign(params[param])
