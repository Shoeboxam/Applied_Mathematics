import tensorflow as tf
import numpy as np

import os
# because I don't want the hardware specs logged in red every time I run
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

iterations = 10
learn_rate = .1
data = np.array([
    [3, 2],
    [1, 2],
    [0, 1],
    [4, 3]
])

# in larger networks its natural to think of each element of X and Y as feature vectors
# each column is a feature vector X or expectation Y
X = data[:, 0][None]
Y = data[:, 1][None]

# using Tensorflow is overkill, but just for fun

with tf.Session() as sess:

    stimulus = tf.placeholder(tf.float32, name='stimulus', shape=[1, None])
    expected = tf.placeholder(tf.float32, name='expected', shape=[1, None])
    W_tf = tf.get_variable('weights', shape=[1, 1], initializer=tf.initializers.random_normal)
    b_tf = tf.get_variable('bias', shape=[1, 1], initializer=tf.initializers.zeros)

    predict = W_tf @ stimulus + b_tf
    cost = (predict - expected) ** 2 / (2 * X.shape[1])
    loss = tf.reduce_sum(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)

    # create an op and immediately run it in the default session (set by the context handler)
    tf.global_variables_initializer().run()

    # optionally, use this to set the initial value of the homemade optimization network, for comparison
    # W = sess.run(W_tf)

    for i in range(iterations):
        sampleID = np.random.randint(X.shape[1])
        sess.run(optimizer, feed_dict={
            'stimulus:0': X[:, sampleID][None],
            'expected:0': Y[:, sampleID][None]
        })

        print("Error %s: %s" % (i, str(sess.run(loss, feed_dict={
            'stimulus:0': X[:, sampleID][None],
            'expected:0': Y[:, sampleID][None]
        }))))

    print("Tensorflow prediction")
    print(sess.run(predict, feed_dict={'stimulus:0': X}))


# homegrown function optimization
W = np.array([[1.]])
b = np.array([[0.]])

for i in range(iterations):
    # I'm using an implicit broadcast on last dimension of b_home
    # 2/2 = 1 is simplified away
    # not the exact same results as TF, but relatively close

    dl_dW = ((W @ X + b) - Y) @ X.T / X.shape[1]
    dl_db = np.sum(((W @ X + b) - Y), axis=1) / X.shape[1]  # db/db == np.eye

    W -= learn_rate * dl_dW
    b -= learn_rate * dl_db
    print("Error %s: %s" % (i, str(np.sum(((W @ X + b) - Y) ** 2 / (2 * X.shape[1])))))

print("homegrown prediction")
print(W @ X + b)
