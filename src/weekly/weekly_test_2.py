#F0D5F1
import math
import random

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        exponent = -abs(x - self.loc) / self.scale
        probability = (1 / (2 * self.scale)) * math.exp(exponent)
        if x < self.loc:
            probability *= 1
        else:
            probability *= 1
        return probability

    def cdf(self, x):
        if x < self.loc:
            cdf_value = 0.5 *  math.exp((x - self.loc) / self.scale)
        else:
            cdf_value = 1 - 0.5* math.exp(-(x - self.loc) / self.scale)
        return cdf_value

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("A valószínűség értéke 0 és 1 között kell legyen.")
        if p < 0.5:
            ppf_value = self.loc + self.scale * math.log(2 * p )
        else:
            ppf_value = self.loc - self.scale * math.log(2 * (1 - p) )
        return ppf_value

    def gen_rand(self):
        u = random.uniform(0, 1)
        if u < 0.5:
            random_value = self.loc + self.scale * math.log(2 * u)
        else:
            random_value = self.loc - self.scale * math.log(2 * (1 - u))
        return random_value

    def mean(self):
       return self.loc


    def variance(self):
       return 2 * self.scale ** 2


    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 3

    def mvsk(self):
            mean_value = self.loc
            variance_value = 2 * self.scale ** 2
            skewness = 0
            kurtosis = 3
            return [mean_value, variance_value, skewness, kurtosis]

class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.scale:
            return 0.0
        else:
            return self.shape * self.scale ** self.shape / (x ** (self.shape + 1))
    def cdf(self, x):
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape
    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        if p == 0:
            return self.scale
        elif p == 1:
            return float("inf")
        else:
            return self.scale / (1.0 - p) ** (1.0 / self.shape)

    def gen_rand(self):
        u = random.uniform(0, 1)
        if u < 0.5:
            x = self.shape - self.scale * math.log(1 - 2 * u)
        else:
            x = self.shape + self.scale * math.log(2 * u - 1)
        return float(x)
    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        else:
            return (self.shape * self.scale) / (self.shape - 1)
    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        else:
            return (self.scale ** 2) * (self.shape / ((self.shape - 1) ** 2) * (self.shape - 2))
    def skewness(self):
        if self.shape <= 3:
            raise Exception("Moment undefined")
        else:
            return (2 * (1 + self.shape)) / (self.shape - 3) * ((self.shape - 2) / self.shape) ** 0.5
    def ex_kurtosis(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        else:
            return (6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2)) / \
                (self.shape * (self.shape - 3) * (self.shape - 4))

    def mvsk(self):
        first = self.scale
        second = ((self.shape * self.scale) ** 2) / ((self.shape - 2) * (self.shape - 1) ** 2)
        third = (self.shape * self.scale) ** 3 * ((self.shape - 3) * (self.shape - 2)) / \
                    ((self.shape - 1) ** 3 * (self.shape - 4))
        return [first, second, third]










