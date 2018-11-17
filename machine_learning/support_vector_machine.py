# WARNING: This is not fully implemented

# First find the optimal hyperplane, then generalize for non-linearly separable data (SVM), then apply kernels.
#
# The optimal hyperplane maximizes margin M when ||β|| = 1 (any hyperplane x.T@β may be expressed when ||β|| = 1),
# and y(x.T@β - β₀) ≥ M (confidence ≥ margin). x ∈ dataset with p attributes.
#
# Equivalently, maximize M when confidence ≥ M ||β||, where ||β|| can be anything
# Equivalently, minimize ||β||²/2 when y*(x.T@β + β₀) ≥ 1 (we let M = 1/||β|| WLOG).
# You could minimize ||β||, but quadratic programming on linear constraints is convex (global min ftw).

# To make this hyperplane-search an SVM, just add C ∑ε,
#   a misclassification allowance with weight C to the objective function.
# Notice ε ≥ 0, which is another constraint. Also we permit a margin of 1 - ε for each data point now.

# Find the primal function by solving constraints for zero and adding to the objective function (lagrange multiplier).
# After optimization the lagrange multiplier vectors α and μ are are always zero except when x is a support vector.
# primal = ||β||²/2 + C∑ε - ∑α[y*(x.T@β + β₀) - (1 - ε)] - ∑με

# The optimal value will be when the gradient wrt to the parameters is zero,
#   so compute the partials wrt β, β₀ and ε, to get the constraints:
# β = ∑αyx, 0 = ∑yx and α = C - μ.

# new primal to maximize = ∑α - ∑∑αα' yy' x.T@x' = ∑α - ∑∑αα' yy' <x, x'>
# constraints:
# ∑με = 0 (pulled out from primal)
# ∑α[y(x.T@β + β₀) - (1 - ε)] = 0 (from the hyperplane algorithm)

# The primal and constraints may now be provided to a python package like CVXOPT,
#   a dynamic programming wrapper for constrained optimization solvers.
# Take note that the primal has <x, x'> but still CVXOPT still provides a solution.
# Prediction in the form x.T@β + β₀.

# Unfortunately, this is all still in search of a linear decision boundary.
# Instead let <x, x'> be <φ(x), φ(x')>, where φ is a mapping to a higher dimensional space.
# So the prediction is φ(x).T@β + β₀ = ∑α'y'<φ(x), φ(x')>.
# To evaluate this:
#   for each support vector you pass the unseen data point through φ,
#   take the inner product with said support vector,
#   then multiply by the class label of that support vector (α' filters x' to only support vectors).

# Now the kernel trick is straightforward, since <φ(x), φ(x')> is a kernel K(x, x').
# Alternatively let K(x, x') = (1 + <x, x'>)² polynomial, or exp(-γ||x - x'||²) RBF or tanh(k₁<x, x'> + k₂) etc.
# Pick the kernel, adjust the primal form for CVXOPT, and evaluate unseen data with ∑α'y'K(x, x').


import cvxopt
import numpy as np


class SVM(object):
    def __init__(self, kernel, cost):
        self.kernel = kernel
        self.cost = cost

    def fit(self, X, Y):
        # The α in my derivations is respectively x in the cvxopt docs.
        # X has shape [n, p], where n = number of observations/rows, p = number of attributes/columns

        # https://en.wikipedia.org/wiki/Gramian_matrix
        gramian = np.zeros([X.shape[0]] * 2)
        for i, j in np.array(np.tril_indices(X.shape[0])).T:
            gramian[i, j] = self.kernel(X[i, :, None], X[j, None, :])
        # Gramian matrix is PSD, implies hermitian. Assuming x real, then gramian is symmetric
        gramian += np.tril(gramian, k=1).T

        # objective to minimize:
        # ∑∑αα' yy' <x, x'> - ∑α
        confidence = cvxopt.matrix(Y @ Y.T * gramian / 2)  # hadamard product of PSD matrices is PSD
        allowance_used = -cvxopt.matrix(np.ones(X.shape[0]))

        # under the inequality constraints:
        # 0 ≤ α ≤ C → -α ≤ 0 ∧ α ≤ C
        G = cvxopt.matrix(np.vstack([-np.eye(X.shape[0]), np.eye(X.shape[0])]))
        h = cvxopt.matrix(np.hstack([np.zeros(X.shape[0]), np.full(X.shape[0], self.cost)]))

        # under the equality constraint:
        # ∑ αy = 0
        A = cvxopt.matrix(np.diag(Y))
        b = cvxopt.matrix(np.zeros(Y.shape[0]))

        # minimize xPx + qx constrained by Gx ≤ h; Ax = b
        alpha = cvxopt.solvers.qp(confidence, allowance_used, G, h, A, b)

        # TODO derive β, β₀
        print(alpha)
        print(alpha['x'])


samples = np.array(np.random.normal(size=[10, 2]))
labels = 2 * (samples.sum(axis=1) > 0) - 1.0


model = SVM(lambda a, b: b @ a, 1)
model.fit(samples, labels)
