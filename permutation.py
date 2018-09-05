import numpy as np


class Permutation(object):

    @classmethod
    def compose(cls, *values):
        x = values[-1]
        for value in reversed(values[:-1]):
            x = value(x)
        return x

    def __init__(self, *values):
        self.values = values

    def __call__(self, i):
        if type(i) is Permutation:
            return Permutation(*[self(elem) for elem in i.values])

        return self.values[i]

    def __str__(self):
        return self.__class__.__name__ + str(tuple(self.values))

    def __pow__(self, power, modulo=None):
        if power < -1:
            return Permutation(*[i[1] for i in sorted(zip(self.values, range(len(self.values))))]) ** (power + 1)
        if power == -1:
            return Permutation(*[i[1] for i in sorted(zip(self.values, range(len(self.values))))])
        if power == 0:
            return Permutation(*range(len(self.values)))
        if power == 1:
            return self
        if power > 1:
            return Permutation(*[self(i) for i in self.values]) ** (power - 1)

    def asarray(self):
        return sum([np.einsum('i,j->ij', np.eye(len(self.values))[i], np.eye(len(self.values))[self(i)]) for i in range(len(self.values))])


sigma = Permutation(2, 0, 1)
gamma = Permutation(0, 2, 1)
tau = Permutation(0, 2, 1)

print("\nPermutation composition same as functional-style permutation composition")
print(np.allclose(sigma(tau(gamma)).asarray(), Permutation.compose(sigma, tau, gamma).asarray()))

print("\nP(σ⁻¹(x)) == P(σ(x))⁻¹")
print(np.allclose((sigma**-1).asarray(), np.linalg.inv(sigma.asarray())))

print("\nσ(σ⁻¹) == id")
print(sigma(sigma**-1))

print("\nPermutation raised to cardinality of orbit is equal to itself")
print(np.allclose((sigma**3).asarray(), sigma.asarray()))  # orbit of length 3
