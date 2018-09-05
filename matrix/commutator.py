import numpy as np


# Lie group has the property that, for any A, B ∈ Λ, where Λ is an arbitrary Lie group, AB - BA ∈ Λ
# Typically [A, B] denotes the commutator of A and B, evaluated as AB - BA
def commutator(A, B):
    return A @ B - B @ A


X = np.triu(np.random.uniform(size=(5, 5)), k=0)
Y = np.triu(np.random.uniform(size=(5, 5)), k=0)
Z = np.triu(np.random.uniform(size=(5, 5)), k=0)

# The set of all strictly upper diagonal matrices forms a lie group,
# commutators of any two elements in the group of upper triangular matrices produces an element in the same group
print(commutator(commutator(X, Y), commutator(Y, Z)))
