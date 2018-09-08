import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True)
newaxisEnd = (..., np.newaxis)

fig = plt.figure(figsize=(5, 5))
fig.suptitle('r^2 - r, for -pi/2 < theta < pi/2')
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

gamma = .2
epsilon = 1e-5

domain = np.array([[-1, 1], [-1, 1]])


# get the corners of the domain
def corners(domain):
    if len(domain) == 1:
        return [[i] for i in domain[0]]
    return [[i, *j] for i in domain[0] for j in corners(domain[1:])]


# mask points that are not within the domain (used before plotting)
def domain_mask(points):
    return points[:, np.all((domain[:, 0][newaxisEnd] < points) & (points < domain[:, 1][newaxisEnd]), axis=0)]


# objective function
polar = lambda x, y: np.array([np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)])
polar_d1 = lambda x, y: np.array([
    [x / np.sqrt(x ** 2 + y ** 2), y / np.sqrt(x ** 2 + y ** 2)],
    [-y / (x ** 2 + y ** 2), x / (x ** 2 + y ** 2)]])


def obj(x, y):
    r, theta = polar(x, y)
    return r ** 2 - r


# first derivative
def obj_d1(x, y):
    r, theta = polar(x, y)
    return np.squeeze(np.array([[2 * r - 1, 0]]) @ polar_d1(x, y))


def mesh(func, axis_length=30):
    axes = [np.linspace(start=axis[0], stop=axis[1], num=axis_length) for axis in domain]
    plane = np.meshgrid(*axes)
    return [*plane, func(*plane)]


def replot(ax, func, point, reduction=None):
    ax.clear()
    ax.plot_surface(*mesh(func), zorder=3)
    if reduction is not None: ax.plot(*reduction, color='limegreen')

    heights = [func(*corner) for corner in corners(domain)]
    ax.plot(*zip(*[point, [*point[:2], np.max(heights)]]), color='red')


minima = [np.random.uniform(*axis) for axis in domain]  # ball starts at a random spot on the domain

while True:
    update = obj_d1(*minima)
    minima -= update * gamma

    replot(ax1, obj, [*minima, obj(*minima)])
    plt.pause(.00001)

    if np.sum(np.abs(update)) < epsilon:
        break
