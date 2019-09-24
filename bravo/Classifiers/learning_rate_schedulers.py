import matplotlib.pyplot as plt
import numpy as np


class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]

        # the learning rate schedule
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):
    def __init__(self, alpha_zero=0.01, factor=0.25, drop_every=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self._alpha_zero = alpha_zero
        self._factor = factor
        self._drop_every = drop_every

    @property
    def alpha_zero(self):
        return self._alpha_zero
    
    @property
    def factor(self):
        return self._factor
    
    @property
    def drop_every(self):
        return self._drop_every
    
    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exponent = np.floor((1 + epoch) / self.drop_every)
        alpha = self.alpha_zero * (self.factor ** exp)

        # return the learning rate
        return float(alpha)


class PolynomialDecay(LearningRateDecay):
    def __init__(self, max_epochs=100, alpha_zero=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self._max_epochs = max_epochs
        self._alpha_zero = alpha_zero
        self._power = power

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def alpha_zero(self):
        return self._alpha_zero
  
    @property
    def power(self):
        return self._power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / self.max_epochs)) ** self.power
        alpha = self.alpha_zero * decay

        # return the new learning rate
        return float(alpha)

