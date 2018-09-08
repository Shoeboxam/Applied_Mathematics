# proof of concept for forcing Tensorflow to derive global hessian for the entire network
# typically tf.hessians returns block diagonal submatrices on the global hessian

# all weight matrices are slices into a global parameter vector
# so calling tf.hessians with the global parameter vector returns
#   second partials for combinations of derivatives from different weight matrices

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
# because I don't want the hardware specs logged in red every time I run
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fig = plt.figure(figsize=(6, 2.5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

shapes = [[2, 3], [3, 4]]

with tf.Session() as sess:
    # initialize all weight matrices under one global weight matrix
    global_weights = tf.get_variable("global_weights",
                              shape=sum([np.prod(mat) for mat in shapes]),
                              initializer=tf.initializers.random_normal)

    # extract weight matrices local to a layer by slicing into the weights and reshaping
    W1 = tf.reshape(global_weights[:np.prod(shapes[0])], shape=shapes[0])
    W2 = tf.reshape(global_weights[np.prod(shapes[0]):], shape=shapes[1])

    # simple proof of concept network (that does nothing)
    out = tf.nn.relu((W1 ** 2) @ (W2 **2))

    sess.run(tf.global_variables_initializer())

    # generate total hessian for all combinations of all variables from all weight matrices
    total_hessian = sess.run(tf.hessians(out, global_weights))[0]
    ax1.matshow(total_hessian)

    # generate hessians restricted to the subset of variables within a layer
    hess1, hess2 = sess.run(tf.hessians(out, [W1, W2]))
    ax2.matshow(np.reshape(hess1, newshape=[6, 6]))
    ax3.matshow(np.reshape(hess2, newshape=[12, 12]))

    plt.title("global hessian, then block diagonal submatrix hessians for W1 and W2", loc='right', y=1.2)
    plt.show()
