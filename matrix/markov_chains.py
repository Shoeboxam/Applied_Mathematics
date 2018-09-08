import numpy as np

states = 4


def uniform_mc():
    probs = np.random.uniform(size=[states] * 2)
    return (probs / np.sum(probs, axis=0)).T


def random_walk(p):
    markov = np.diag([p] * states, k=1) + np.diag([1 - p] * states, k=-1)
    markov[0, 1] = 1
    markov[states, -2] = 1
    return markov


def renewal_process(p):
    markov = np.zeros([states] * 2)
    for i in range(states):
        for j in range(states):

            if i == 0 and j == 0:
                markov[i, j] = p + (1 - p)**states
            elif i <= j:
                markov[i, j] = p * (1 - p)**(j - i)
            elif 1 <= i <= states - 1 and j == 0:
                markov[i, j] = (1 - p)**(states - i)

    return markov


# Create a random initial state
initial_state = np.random.uniform(0, 1, size=states)
initial_state /= initial_state.sum()

# Create an initial state where state is known
# initial_state = np.zeros(states)
# initial_state[np.random.randint(states)] = 1
print("π₀: " + str(initial_state))

# Get the stochastic matrix (change the function to use a different matrix)
P = renewal_process(.7)

# simulate n steps
n = 23
print("πₙ: " + str(initial_state @ np.linalg.matrix_power(P, n)))