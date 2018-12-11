import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# part A
from sklearn.impute import SimpleImputer
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# part B
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, AgglomerativeClustering

import matplotlib.pyplot as plt


np.set_printoptions(precision=4, suppress=True)


data_url = 'https://www.google.org/flutrends/about/data/dengue/mx/data.txt'
data = pd.read_csv(data_url, skiprows=11, parse_dates=['Date'])

coordinates = {
    "Baja California": [30.76405113, -116.0092603],
    "Chiapas": [16.74999697, -92.63337447],
    "Colima": [18.92038129, -103.8799748],
    "Distrito Federal": [19.44244244, -99.1309882],
    "Estado de México": [19.4969, -99.7233],
    "Jalisco": [19.77001935, -104.3699966],
    "Morelos": [18.92110476, -99.23999964],
    "Nayarit": [21.81999758, -105.2200481],
    "Nuevo León": [25.1899986, -99.83998885],
    "Oaxaca": [16.42999066, -95.01999882],
    "Quintana Roo": [21.20839057, -86.7114549],
    "Sinaloa": [23.19999086, -106.2300381],
    "Sonora": [27.58000775, -109.9299931],
    "Tabasco": [18.40002545, -93.22997888],
    "Tamaulipas": [22.73335268, -98.95001734],
    "Veracruz": [17.93997601, -94.73999007],
    "Yucatán": [21.09998985, -89.27998743]
}

features_GDT = [
    "Baja California", "Distrito Federal", "Jalisco", "Estado de México", "Morelos", "Nuevo León", "Oaxaca",
    "Quintana Roo", "Sonora", "Tamaulipas", "Veracruz", "Yucatán", "Chiapas", "Colima", "Nayarit", "Sinaloa", "Tabasco"
]


# adapted from: https://en.wikipedia.org/wiki/Talk:Varimax_rotation, same variable names used from Lecture 9 notes
def varimax(L, gamma=1, lim=20, tol=1e-6):
    p, m = L.shape

    # rotation matrix that is iteratively updated
    T = np.eye(m)

    d = 0
    for i in range(lim):
        LT = L @ T
        temp = LT ** 3 - (gamma / p) * LT @ np.diag(np.diag(LT.T @ LT))
        U, S, Vh = np.linalg.svd(L.T @ temp)

        T = U @ Vh
        d_old, d = d, np.sum(S)

        if d / d_old < tol:
            break

    return L @ T


def analyze_index(data):

    data = data[["Mexico", *features_GDT]].dropna()

    Y = data['Mexico'].apply(lambda x: x > data['Mexico'].mean())
    # X = SimpleImputer(strategy='mean').fit_transform(data[features_GDT])
    X = np.hstack([StandardScaler().fit_transform(feature[:, None]) for feature in np.array(data[features_GDT]).T])

    stepwise = RFE(estimator=LogisticRegression(solver='lbfgs'))
    stepwise.fit(X, Y)

    print('ranking')
    print(list(sorted(zip(stepwise.ranking_, features_GDT))))

    Y_pred = stepwise.predict(X)
    print(accuracy_score(Y, Y_pred))
    print(confusion_matrix(Y_pred, Y))

    model_fa = FactorAnalysis()
    model_fa.fit_transform(X)
    loadings = varimax(model_fa.components_[:6].T)
    print(loadings)

    factors = []
    for loading in (loadings > .7).T:
        factors.append([feature for feature, included in zip(features_GDT, loading) if included])
    print(factors)


def cluster_months(data):
    data = data[data['Mexico'] > data['Mexico'].quantile(.8)]
    month = data['Date'].apply(lambda x: x.month)

    def inverse_time(features):
        return np.block([
            np.arctan2(*features[:, :2].T)[:, None] * 6 / np.pi + 7
        ])

    data = np.block([
        np.sin((month - 7) / 6 * np.pi)[:, None],
        np.cos((month - 7) / 6 * np.pi)[:, None]
    ])

    print('Month modular mean')
    print(inverse_time(data.mean(axis=0)[None])[0, 0])

    plt.scatter(*data.T, s=np.bincount(month)**1.5)
    for label, location in zip(month, data):
        plt.text(*location + .1, label)

    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.show()


def cluster_time(data, max_clusters, selected_clusters):
    # data = data[data['Mexico'] > data['Mexico'].quantile(.8)].dropna()

    scaler_year = MinMaxScaler()
    scaler_dengue = MinMaxScaler()

    def inverse_time(features):
        return np.block([
            np.array([[datetime.fromtimestamp(i).year] for i in scaler_year.inverse_transform(features[:, 0, None])]),
            np.array([[datetime.fromtimestamp(i).month] for i in scaler_year.inverse_transform(features[:, 0, None])]),
            # scaler_dengue.inverse_transform(features[:, 1, None])
        ])

    weights = scaler_dengue.fit_transform(data['Mexico'][:, None])[:, 0]
    data = scaler_year.fit_transform(data['Date'].apply(lambda x: x.timestamp())[:, None])

    totals = []
    for k in range(1, max_clusters):
        model = KMeans(n_clusters=k)
        model.fit(data, sample_weight=weights)
        totals.append(sum(np.min(pairwise_distances(data, model.cluster_centers_, 'euclidean'), axis=1)))
    plt.plot(range(1, max_clusters), [i / data.shape[0] for i in totals])
    plt.show()

    model = KMeans(n_clusters=selected_clusters)
    model.fit(data, sample_weight=weights)

    events = inverse_time(model.cluster_centers_)
    events.sort(axis=0)
    print('Clustered dates')
    print(events)


def cluster_space(data, max_clusters, selected_clusters):
    data = data[features_GDT].dropna(thresh=12).reset_index(drop=True)
    data = SimpleImputer(strategy='mean').fit_transform(data[features_GDT]).T

    r = 3  # number of weeks to permit warping

    # http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
    def LB_Keogh(s1, s2):
        LB_sum = 0
        for ind, i in enumerate(s1):

            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound) ** 2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound) ** 2

        return np.sqrt(LB_sum)

    model = AgglomerativeClustering(
        affinity='precomputed',
        linkage='complete',
        n_clusters=selected_clusters)
    distances = pairwise_distances(data, data, metric=LB_Keogh)
    model.fit(distances.max() - distances)

    print('Agglomerative state clustering with DTW')
    print(sorted(zip(model.labels_, features_GDT)))


def cluster_space_time(data, max_clusters, selected_clusters):
    features = ['Date', 'Latitude', 'Longitude']

    data_melted = data.melt(id_vars=['Date', 'Mexico'], var_name='Location').dropna()

    # data_melted = data_melted[data_melted['value'] > data_melted['value'].quantile(.8)].dropna().reset_index(drop=True)
    data_melted['Date'] = data_melted['Date'].apply(lambda x: x.timestamp())

    data_melted['Longitude'] = data_melted['Location'].apply(lambda x: coordinates[x][0])
    data_melted['Latitude'] = data_melted['Location'].apply(lambda x: coordinates[x][1])

    scaler_date = MinMaxScaler()
    scaler_lat = MinMaxScaler()
    scaler_lon = MinMaxScaler()

    transformed = np.array(data_melted[features])
    transformed = np.hstack([
        scaler_date.fit_transform(transformed[:, 0, None]),
        scaler_lat.fit_transform(transformed[:, 1, None]),
        scaler_lon.fit_transform(transformed[:, 2, None])
    ])

    weights = data_melted['value'] ** 1.5

    totals = []
    for k in range(1, max_clusters):
        model = KMeans(n_clusters=k)
        model.fit(transformed, sample_weight=weights)
        totals.append(sum(np.min(pairwise_distances(transformed, model.cluster_centers_, 'euclidean'), axis=1)))
    plt.plot(range(1, max_clusters), [i / transformed.shape[0] for i in totals])
    plt.show()

    model = KMeans(n_clusters=selected_clusters)
    model.fit(transformed, sample_weight=weights)

    datestamps = scaler_date.inverse_transform(model.cluster_centers_[:, 0, None])
    dates = [datetime.utcfromtimestamp(int(stamp)).strftime('%Y-%m-%d') for stamp in datestamps]

    locations = np.hstack([
        np.array(list(coordinates.keys()))[:, None],
        scaler_lat.transform(np.array([i[1] for i in coordinates.values()])[:, None]),
        scaler_lon.transform(np.array([i[0] for i in coordinates.values()])[:, None])
    ])

    fig, ax = plt.subplots()
    ax.imshow(plt.imread("Municipalities_of_Mexico_(equirectangular_projection).png"))

    for group in (locations, model.cluster_centers_):
        plt.scatter(group[:, 1].astype(float) * 1800 + 150, (1 - group[:, 2].astype(float)) * 950 + 250)

    texts = []
    for location in locations:
        texts.append(plt.text(float(location[1]) * 1800 + 150, (1 - float(location[2])) * 950 + 250, location[0]))

    for date, point in zip(dates, model.cluster_centers_):
        texts.append(plt.text(point[1] * 1800 + 150, (1 - point[2]) * 950 + 250, date))

    try:
        from adjustText import adjust_text
        adjust_text(texts, force_points=0.15, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    except ImportError:
        print('adjustText is not installed. The labels on the plot will overlap.')

    plt.show()


analyze_index(data)
cluster_months(data)
cluster_time(data, max_clusters=20, selected_clusters=5)
cluster_space(data, max_clusters=10, selected_clusters=6)
cluster_space_time(data, max_clusters=20, selected_clusters=10)
