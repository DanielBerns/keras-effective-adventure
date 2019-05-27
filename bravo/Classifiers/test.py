import numpy as np

def build(initial_alpha, factor):
    def step_decay(epoch):
        # initialize the base initial learning rate, drop factor, and
        # epochs to drop every
        _initial_alpha = initial_alpha
        _factor = factor
        drop_every = 4
        step = np.floor((1 + epoch) / drop_every)
        # compute learning rate for the current epoch
        alpha = float(_initial_alpha * (_factor ** step))
        print('{0:f} {1:f} {2:d}'.format(alpha, step, drop_every))
        return alpha  # learning rate

    return step_decay


fun = build(0.001, 0.2)

for e in range(20):
    a = fun(e)

