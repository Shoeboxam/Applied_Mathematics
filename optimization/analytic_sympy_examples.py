# finding the analytic critical points of vector functions via sympy
# uncomment lines to view hessians/gradients/critical points/plots/etc.

from sympy import *
from sympy.plotting import plot3d

x, y = symbols('x y')


# This is my own helper function to evaluate a multivariate expression. Could be made more generic...
def evaluate(expression):
    return expression.subs(x, center[0]).subs(y, center[1])


# EXAMPLE 1
center = (1, 0)

# define the vector function
f = exp((x - 1)**2)*cos(y) + 2*x**2*y

# take gradient
gradient = Matrix([diff(f, var) for var in [x, y]])

# take second derivatives
hessian = Matrix([[diff(f1, var) for var in [x, y]] for f1 in gradient])

origin = Matrix([[x], [y]]) - Matrix(center)

term1 = Matrix([[evaluate(f)]])
term2 = evaluate(gradient).T @ origin
term3 = origin.T @ evaluate(hessian) @ origin / 2

# print(simplify(term1 + term2 + term3))


# EXAMPLE 2
f = (x**2 + 3*y**2) * exp(1 - x**2 - y**2)

# plot3d(f, (x, -2, 2), (y, -2, 2))

# take gradient
gradient = Matrix([diff(f, var) for var in [x, y]])

# take second derivatives
hessian = Matrix([[diff(f1, var) for var in [x, y]] for f1 in gradient])

# Tests to find critical points. Plug these into gradient to check zeros
# print(solve(diff(f, x), x))
# print(solve(diff(f, y), y))

# center = (0, -1)  # (0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)

# print(evaluate(hessian))


# EXAMPLE 3
f = 2 * x**2 + x + y**2 - 2
constraint = x**2 + y**2 - 4

# take gradient
# gradient = Matrix([diff(f, var) for var in [x, y]])

# take second derivatives
# hessian = Matrix([[diff(f1, var) for var in [x, y]] for f1 in gradient])

# print(solve(diff(f, x), x))
# print(solve(diff(f, y), y))

center = (-1/2, sqrt(15/4))
# print(evaluate(f))

# plot3d(f, constraint, (x, -2, 2), (y, -2, 2))


# EXAMPLE 4
# function is parameterized by a, which changes the properties of the hessian based on the region of a.
a = symbols('a')
f = x*y + exp(a * (x**2 + y**2))

# take gradient
gradient = Matrix([diff(f, var) for var in [x, y]])

# take second derivatives
hessian = Matrix([[diff(f1, var) for var in [x, y]] for f1 in gradient])

# Vary among ℝ, partitioned by -sqrt(1/4) and sqrt(1/4)
a_value = -sqrt(1/4)

plot3d(f.subs(a, a_value), (x, -.5, .5), (y, -.5, .5))
# print(gradient)

center = (0, 0)
# when a is on a boundary partition, the hessian evaluated at (0, 0) is zero
print(evaluate(det(hessian.subs(a, a_value))))
