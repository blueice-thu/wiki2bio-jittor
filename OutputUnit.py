import jittor as jt
import numpy as np

class OutputUnit(jt.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = jt.rand([input_size, output_size])
        self.b = jt.zeros([output_size])

    def execute(self, x, finished = None):
        out = jt.nn.matmul(x, self.W) + self.b

        if finished is not None:
            finished = finished.unsqueeze(1)
            inds = jt.where(finished)[0]
            out[inds] = 0
            #out = tf.multiply(1 - finished, out)
        return out
