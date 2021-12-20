import jittor as jt
import numpy as np

class LstmUnit(jt.Module):
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W = jt.rand([self.input_size+self.hidden_size, 4*self.hidden_size])
        self.b = jt.zeros([4*self.hidden_size])

        self.params = {'W':self.W, 'b':self.b}
    
    def execute(self, x, s, finished = None):
        h_prev, c_prev = s

        x = jt.contrib.concat([x, h_prev], 1)
        i, j, f, o = np.split(jt.nn.matmul(x, self.W) + self.b, 4, 1)

        # Final Memory cell
        c = jt.sigmoid(f+1.0) * c_prev + jt.sigmoid(i) * jt.tanh(j)
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            finished = finished.unsqueeze(1)
            out = jt.array(np.where(finished, np.zeros_like(h), h))
            state = jt.array(np.where(finished, h_prev, h), np.where(finished, c_prev, c))
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state
        
    def save(self, path):
        jt.save(self.params, path)
    
    def load(self, path):
        params = jt.load(path)
        for param in params:
            self.params[param].assign(params[param])
