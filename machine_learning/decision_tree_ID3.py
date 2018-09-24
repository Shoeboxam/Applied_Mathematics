import numpy as np
import pandas as pd

space = '\t'

predictors, predicted = ['x1', 'x2', 'x3'], ['x4']

data = pd.DataFrame([[1, 0, 0, 1],
                     [0, 1, 0, 1],
                     [0, 0, 0, 0],
                     [1, 0, 1, 0],
                     [0, 0, 0, 0],
                     [1, 1, 0, 1],
                     [0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 0, 0, 0],
                     [1, 0, 0, 1]],
                    columns=[*predictors, *predicted])


class Node(object):
    def __init__(self, attribute, child_true, child_false):

        self.attribute = attribute
        self.child_true = child_true
        self.child_false = child_false

    def __str__(self, depth=0):

        child_true = self.child_true.__str__(depth + 1) if type(self.child_true) is Node else str(self.child_true)
        child_false = self.child_false.__str__(depth + 1) if type(self.child_false) is Node else str(self.child_false)

        return f'{self.attribute}\n' \
               f'{space * (depth + 1)}T: {child_true}\n' \
               f'{space * (depth + 1)}F: {child_false}'

    def __call__(self, data):

        def get_child(child, row):
            return child(row)[0] if type(child) is Node else child

        def get(row):
            return get_child(self.child_true, row) if bool(row[self.attribute][row.index[0]]) else get_child(self.child_false, row)

        return np.array([get(data[row.astype(bool)]) for row in np.eye(data.shape[0])])


def get_entropy(data):
    probabilities = np.bincount(data) / float(data.size)

    entropy = -np.sum(probabilities * np.nan_to_num(np.log2(probabilities)))

    print(f'{round(entropy, 2)} Probabilities of predicted values: {list(probabilities)}')
    return entropy


def make(data, predictors, predicted, depth=0):
    if np.unique(data[predicted]).size == 1:
        return data[predicted].values[0]

    best_predictor = predictors[0]
    best_gain = 0

    print(f'{space * depth}Current Entropy: ', end='')
    current_entropy = get_entropy(data[predicted])

    for predictor in predictors:
        print(f'{space * depth}Analyzing {predictor}')
        mask = data[predictor].values.astype(bool)

        print(f'{space * depth} Entropy when 1: ', end='')
        true_branch = get_entropy(data[predicted][mask])
        print(f'{space * depth} Entropy when 0: ', end='')
        false_branch = get_entropy(data[predicted][~mask])

        p = sum(mask) / len(mask)
        gain = current_entropy - p * true_branch - (1 - p) * false_branch
        print(f'{space * depth} Gain for {predictor}: {round(gain, 2)}')

        if gain > best_gain:
            best_predictor, best_gain = predictor, gain

    print(f'{space * depth}Best gain from {best_predictor} with information gain {round(best_gain, 2)}.')
    print()

    mask = data[best_predictor].values.astype(bool)
    other_predictors = [i for i in predictors if i != best_predictor]

    true_child = make(data[mask], other_predictors, predicted, depth + 1)
    false_child = make(data[~mask], other_predictors, predicted, depth + 1)
    return Node(best_predictor, true_child, false_child)


tree = make(data, predictors, predicted[0])

print('Final Decision Tree')
print(tree)

print('\nPredicted values:')
print(tree(data[predictors]))
print('\nExpected Values:')
print(np.array(data[predicted]).T[0])
