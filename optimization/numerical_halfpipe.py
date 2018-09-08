import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True)
newaxisEnd = (..., np.newaxis)

fig = plt.figure(figsize=(10, 5))
fig.suptitle('(x*cos(θ) + y*sin(θ))^2' + ' ' * 30 + '2 * sigmoid((x*cos(θ) + y*sin(θ))^2) - 1')
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

gamma = .2
epsilon = 1e-5

domain = np.array([[-5, 5], [-5, 5]])


# get the corners of the domain
def corners(domain):
    if len(domain) == 1:
        return [[i] for i in domain[0]]
    return [[i, *j] for i in domain[0] for j in corners(domain[1:])]


# mask points that are not within the domain (used before plotting)
def domain_mask(points):
    return points[:, np.all((domain[:, 0][newaxisEnd] < points) & (points < domain[:, 1][newaxisEnd]), axis=0)]


theta = np.random.uniform(-np.pi, np.pi)

# returns a rotation matrix from the special orthogonal group on the plane
SO2 = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# logistic sigmoid function
sig = lambda x: 1 / (1 + np.exp(np.negative(x)))

# objective functions
obj = lambda x, y: (x * np.cos(theta) + y * np.sin(theta)) ** 2
obj_sig = lambda x, y: 2 * sig(obj(x, y)) - 1
obj_red = lambda x: np.array([1, 1])

# first derivatives
obj_d1 = lambda x, y: 2 * (x * np.cos(theta) + y * np.sin(theta)) * np.array([np.cos(theta), np.sin(theta)])
obj_sig_d1 = lambda x, y: 2 * sig(obj(x, y)) * (1 - sig(obj(x, y))) * obj_d1(x, y)

# second derivatives
obj_d2 = lambda x, y: 2 * np.array([[np.cos(theta), np.sin(theta)]]).T @ np.array([[np.cos(theta), np.sin(theta)]])
obj_sig_d2 = lambda x, y: 2 * sig(obj(x, y)) * (1 - sig(obj(x, y))) * (1 - 2 * sig(obj(x, y))) * obj_d2(x, y)


def mesh(func, axis_length=30):
    axes = [np.linspace(start=axis[0], stop=axis[1], num=axis_length) for axis in domain]
    plane = np.meshgrid(*axes)
    return [*plane, func(*plane)]


def replot(ax, func, point, reduction=None):
    ax.clear()
    ax.plot_surface(*mesh(func), zorder=3)
    if reduction is not None: ax.plot(*reduction, color='limegreen')

    heights = [func(*corner) for corner in corners(domain)]
    ax.plot(*zip(*[point, [*point[:2], np.max(heights) + .5]]), color='red')


# Optimize both functions separately
minima = [np.random.uniform(*axis) for axis in domain]  # ball starts at a random spot on the domain
minima_sig = np.array(minima)

while True:
    update = obj_d1(*minima)
    minima -= update * gamma

    update_sig = obj_sig_d1(*minima_sig)
    minima_sig -= update_sig * gamma

    reduction = np.einsum('i,j->ij', SO2(theta)[:, 0], np.linspace(np.min(domain[:, 0]), np.max(domain[:, 1]), 100)) * 2

    reduction_obj = domain_mask(reduction + minima[newaxisEnd])
    replot(ax1, obj, [*minima, obj(*minima)], np.array([*reduction_obj, obj(*reduction_obj)]))

    reduction_obj_sig = domain_mask(reduction + minima_sig[newaxisEnd])
    replot(ax2, obj_sig, [*minima_sig, obj_sig(*minima_sig)], np.array([*reduction_obj_sig, obj_sig(*reduction_obj_sig)]))
    plt.pause(.00001)

    if np.sum(np.abs(update)) < epsilon and np.sum(np.sum(update_sig)) < epsilon:
        break

# hessian analysis

print('objective function hessian (constant over the domain)')
hessian = obj_d2(*minima)
print(hessian)

eigvals, eigvecs = np.linalg.eig(hessian)
for eigval, eigvec in zip(eigvals, eigvecs.T):
    if abs(eigval) < epsilon: continue
    rotation = np.arctan2(*reversed(eigvec))
    ax1.plot(*zip(*[
        [*minima, obj(*minima)],
        [*minima + eigvec, obj(*minima)]]))

print('sigmoid objective function hessian at ' + str(minima_sig))
hessian_sig = obj_sig_d2(*minima_sig)
print(hessian_sig)

eigvals_sig, eigvecs_sig = np.linalg.eig(hessian_sig)
for eigval, eigvec in zip(eigvals_sig, eigvecs_sig.T):
    if abs(eigval) < epsilon: continue
    rotation_sig = np.arctan2(*reversed(eigvec))
    ax2.plot(*zip(*[
        [*minima_sig, obj_sig(*minima_sig)],
        [*eigvec + minima_sig, obj_sig(*minima_sig)]]))

plt.title('all functions are done optimizing')
plt.show()

fig = plt.figure(figsize=(10, 5))
fig.suptitle('rotated objective functions via SO2 rotations from the arctan of the hessian eigenvectors')
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

try:
    obj_rot = lambda *point: obj(*np.einsum('i...,ji->j...', np.array(point), SO2(rotation)))
    minima_rot = minima
    replot(ax1, obj_rot, [*minima_rot, obj_rot(*minima_rot)])
except NameError as e:
    print('No rotation eigenvector to rotate the objective function.')

try:
    obj_sig_rot = lambda *point: obj_sig(*np.einsum('i...,ji->j...', np.array(point), SO2(rotation_sig)))
    minima_sig_rot = minima_sig
    replot(ax2, obj_sig_rot, [*minima_sig_rot, obj_sig_rot(*minima_sig_rot)])
except NameError as e:
    print('No rotation eigenvector to rotate the sigmoidal objective function.')

plt.show()
