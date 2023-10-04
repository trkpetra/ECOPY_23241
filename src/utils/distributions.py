#F0D5F1

import random
import math
import scipy.special

class LogisticDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        coefficient = 1 / (4 * self.scale)
        exponent = -abs((x - self.location) / (2 * self.scale))
        pdf_value = coefficient * math.pow(math.cosh(exponent), -2)
        return pdf_value

    def cdf(self, x):
        cdf_value = 1 / (1 + math.exp(-(x - self.location) / self.scale))
        return cdf_value

    def ppf(self, p):
        ppf_value = self.location + self.scale * math.log(p / (1 - p))
        return ppf_value

    def gen_rand(self):
        u = random.random()
        rand_value = self.location + self.scale * math.log(u / (1 - u))
        return rand_value

    def mean(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return self.location


    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        var = (math.pi ** 2) * (self.scale ** 2) / 3
        return var


    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 6/5

    def mvsk(self):
        mean = self.location
        var = (math.pi ** 2) * (self.scale ** 2) / 3
        skewness = 0
        excess_kurtosis = 1.2

        return [mean, var, skewness, excess_kurtosis]
class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        if x < 0:
            return 0
        coefficient = (1 / (2 ** (self.dof / 2))) * (1 / math.gamma(self.dof / 2))
        pdf_value = coefficient * (x ** ((self.dof / 2) - 1)) * math.exp(-x / 2)
        return pdf_value

    def cdf(self, x):
        cdf = scipy.special.gammainc((self.dof) / 2.0, x / 2.0)
        return cdf

    def ppf(self, p):
        ppf = 2.0 * scipy.special.gammaincinv((self.dof) / 2.0, p)
        return ppf

    def gen_rand(self):
        u = self.rand.uniform(0,1)
        rand_value = self.ppf(u)
        return rand_value
    def mean(self):
        if self.dof <= 1:
            raise Exception("Moment undefined")
        return self.dof

    def variance(self):
        return 2 * self.dof

    def skewness(self):
        return (8 / self.dof)**0.5

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]