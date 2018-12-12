from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

from collections import Counter
import warnings
import json

import requests
import zipfile
import io

np.set_printoptions(suppress=True, precision=4)

visualize = False

# ~~~~ DATA PREPROCESSING ~~~~
remote_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
remote = io.BytesIO(requests.get(remote_url, stream=True).content)
# local = '../datasets/classification/Online News Popularity/OnlineNewsPopularity.zip'

with zipfile.ZipFile(remote, 'r') as zipped:
    dataframe = pd.read_csv(zipped.open('OnlineNewsPopularity/OnlineNewsPopularity.csv'))

print('creating categorical shares variable')
dataframe['dummy_shares'] = dataframe.apply(lambda row: int(row[' shares'] > 1400), axis=1)

X_labels = [' timedelta', ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words',
            ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
            ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle', ' data_channel_is_entertainment',
            ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world',
            ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
            ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares', ' self_reference_max_shares',
            ' self_reference_avg_sharess', ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
            ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday', ' weekday_is_sunday', ' is_weekend',
            ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity', ' global_sentiment_polarity',
            ' global_rate_positive_words', ' global_rate_negative_words', ' rate_positive_words',
            ' rate_negative_words', ' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity',
            ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity', ' title_subjectivity',
            ' title_sentiment_polarity', ' abs_title_subjectivity', ' abs_title_sentiment_polarity']
y_labels = ['dummy_shares']

print('partitioning data')
train, test = train_test_split(dataframe, test_size=0.2, random_state=0)

print('standard scaling data')
X_scaler, y_scaler = StandardScaler(), StandardScaler()
X_scaler.fit(train[X_labels])
y_scaler.fit(train[y_labels])

X_train, y_train = X_scaler.transform(train[X_labels]), np.squeeze(train[y_labels].T)
X_test, y_test = X_scaler.transform(test[X_labels]), np.squeeze(test[y_labels].T)

if visualize:
    import seaborn as sns
    import matplotlib.pyplot as plt

    # covariance of standardized data is just correlation
    sns.heatmap(np.cov(X_train.T),
                xticklabels=X_labels,
                yticklabels=X_labels)
    plt.show()
    # input()


# avoid sklearn's fragile floating point check
# https://github.com/scikit-learn/scikit-learn/issues/9633
def normalize_fp(x):
    x /= x.sum(axis=1)[..., None]
    x[:, -1] = 1 - x[:, :-1].sum(axis=1)
    return x


# ~~~~ MODEL / HYPERPARAMETER SPACE ~~~~
models = [
    {
        'name': 'Decision Tree',
        'model': DecisionTreeClassifier(),
        'parameters': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 5]
        }
    },
    {
        'name': 'Neural Network',
        'model': MLPClassifier(),
        'parameters': {
            'hidden_layer_sizes': [[50, 10], [100, 20]],
            'activation': ['relu', 'tanh'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [0, .0001],
            'max_iter': [40],
        }
    },
    {
        'name': 'SVM',
        'model': SVC(),
        'parameters': {
            'kernel': ['rbf', 'linear'],
            'gamma': [1e-3, 1e-4],
            'degree': [2, 3, 5],
            'C': [1, 10, 100, 1000],
            'max_iter': [40],
        }
    },
    {
        'name': 'Gaussian Naive Bayes',
        'model': GaussianNB(),
        'parameters': {
            'priors': [
                None,
                # take 10 samples from dirichlet parameterized by y_train frequencies, zeros ignored
                *normalize_fp(np.random.dirichlet(np.bincount(y_train)[np.nonzero(np.bincount(y_train))], 5))
            ]
        }
    },
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(),
        'parameters': {
            'penalty': ['l1', 'l2'],
            'C': [.1, 1, 10],
            'fit_intercept': [True, False],
            'class_weight': [None, 'balanced'],
            'max_iter': [40]
        }
    },
    {
        'name': 'K Nearest Neighbors',
        'model': KNeighborsClassifier(),
        'parameters': {
            'n_neighbors': [1, 5, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, np.inf]
        }
    },
    {
        'name': 'Bagging Classifier',
        'model': BaggingClassifier(),
        'parameters': {
            'n_estimators': [5, 10, 20],
            'max_samples': [.5, 1, 2],
            'max_features': [1, 2, 3],
            'random_state': range(3)
        }
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(),
        'parameters': {
            'n_estimators': [5, 10, 20],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 5]
        }
    },
    {
        'name': 'AdaBoost',
        'model': AdaBoostClassifier(),
        'parameters': {
            # 'base_estimator': [
            #     DecisionTreeClassifier(max_depth=1),
            #     RandomForestClassifier(n_estimators=3, max_depth=1),
            #     SVC()
            # ],
            'n_estimators': [10, 40],
            'learning_rate': [.5, .75, 1.],
            'algorithm': ['SAMME', 'SAMME.R'],
            'random_state': range(3)
        }
    },
    {
        'name': 'Gradient Boosting Classifier',
        'model': GradientBoostingClassifier(),
        'parameters': {
            'learning_rate': [.02, .1, .5],
            'n_estimators': [20, 50],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 5]
        }
    },
    {
        'name': 'XGBoost',
        'model': XGBClassifier(),
        'parameters': {
            'learning_rate': [.02, .1, .5],
            'n_estimators': [20, 50],
            'min_child_weight': [1, 3],
            'booster': ['gbtree', 'gblinear', 'dart']
        }
    }
]


def stringify(tree):
    return ' '.join([f'{key}={value}' for key, value in tree.items()])


def evaluate(name, model, parameters=None, print_results=False):
    parameters = parameters or {}

    print(f'{name}: Tuning hyper-parameters')

    # catch warnings in bulk, show frequencies for each after grid search
    with warnings.catch_warnings(record=True) as warns:

        clf = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        if print_results:
            warning_counts = dict(Counter([str(warn.category) for warn in warns]))
            if warning_counts:
                print('Warnings during grid search:')
                print(json.dumps(warning_counts, indent=4))

            print("Best parameters set found on development set:")
            print(clf.best_params_)

            print("Grid scores on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            params = clf.cv_results_['params']

            for mean, std, params in zip(means, stds, params):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            print("\nDetailed classification report:")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print(classification_report(y_true, y_pred))

            print("\nDetailed confusion matrix:")
            print(confusion_matrix(y_true, y_pred))

            print("Accuracy Score:")
            print(accuracy_score(y_true, y_pred))

        return [
            name,
            clf.best_params_,
            precision_score(y_true, y_pred, average='micro'),
            recall_score(y_true, y_pred, average='micro'),
            f1_score(y_true, y_pred, average='micro'),
            accuracy_score(y_true, y_pred),
        ]


all_scores = []
all_parameters = []
for model in models:
    name, params, *scores = evaluate(**model, print_results=True)
    scores = [round(score, 4) for score in scores]
    all_parameters.append([name, *[f'{key}={value}' for key, value in params.items()]])
    all_scores.append([name, *scores])

with open('./scores.csv', 'w') as file:
    file.write(', '.join([
        'Algorithm',
        'Avg Precision',
        'Avg Recall',
        'Avg F1',
        'Accuracy'
    ]) + '\n')
    file.writelines([', '.join([str(j) for j in i]) + '\n' for i in all_scores])

with open('./parameters.csv', 'w') as file:
    file.writelines([', '.join(i) + '\n' for i in all_parameters])
