# I wrote my own neural network library here:
# https://github.com/Shoeboxam/Neural_Network

# Tested with Tensorflow 1.12

import os
import tensorflow as tf

import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
predictors = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predicted = ['Species']

# ~~~~ Data Preparation ~~~~
dataframe = pd.read_csv(data_url, header=None, names=[*predictors, *predicted])
# Add a numeric expectation to dataframe
dataframe['SpeciesEncoded'] = pd.factorize(dataframe['Species'])[0]

# Save the encoding for easy interpretation of predictions later
encoding = pd.unique(dataframe['Species'])

samples = np.array(dataframe[predictors])
samples_expected = np.array(dataframe['SpeciesEncoded'])

# Convert into onehot format
samples_expected_onehot = np.zeros((samples_expected.shape[0], 3))
samples_expected_onehot[range(samples_expected.shape[0]), samples_expected] = 1


# ~~~~ Define Model ~~~~
# Input variables
source = tf.placeholder(tf.float32, [None, 4])  # Four inputs, arbitrary batch size
expect = tf.placeholder(tf.float32, [None, 3])  # Three outputs in onehot class encoding

# Layer one
weight_1 = tf.Variable(tf.zeros([4, 10]))
biases_1 = tf.Variable(tf.zeros([10]))
hidden_1 = tf.nn.softplus(source @ weight_1 + biases_1)

# Layer two
weight_2 = tf.Variable(tf.zeros([10, 3]))
biases_2 = tf.Variable(tf.zeros([3]))
prediction = tf.nn.softmax(hidden_1 @ weight_2 + biases_2)

# Cost (used in training)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=hidden_1 @ weight_2 + biases_2,
            labels=expect
        ))

# Define ops
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# ~~~~ Train Model ~~~~
batch_size = 10
for iteration in range(500):
    ids = np.random.randint(0, samples.shape[0] - 1, size=batch_size)

    sess.run(train_step, feed_dict={
        source: samples[ids],
        expect: samples_expected_onehot[ids]
    })

    # Print predictions (debug)
    # if iteration % 100 == 0:
    #     print(sess.run(prediction, feed_dict={source: samples}))


# ~~~~ Test Model ~~~~
# Evaluate accuracy over entire dataset
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(expect, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Training set classification accuracy:")
print(sess.run(accuracy, feed_dict={
    source: samples,
    expect: samples_expected_onehot
}))

# Generate measurements for a new plant and predict its species:
new_plant = np.array([[5.1, 3.7, 6.2, 2.5]])
probabilities = sess.run(prediction, feed_dict={source: new_plant})

print("\n\nPercent probability that a new plant with measurements [5.1, 3.7, 6.2, 2.5] is each plant type:")
print(np.round(probabilities * 100))

print("\nPredicted plant type:")
print(encoding[np.argmax(probabilities)])

# Notice: this trains over the entire dataset, so there are really no test or validation training sets
