# WARNING: This is not fully implemented

import numpy as np
import pandas as pd
from os import path

import seaborn as sns

import matplotlib.pyplot as plt
sns.set()

np.set_printoptions(suppress=True, precision=4)

datasets_folder = '..', 'datasets', 'regression', 'GPS Trajectories'
dataset = 'go_track_trackspoints.csv',

df = pd.read_csv(path.join(*datasets_folder, *dataset), parse_dates=['time'])
df = df.where(df['latitude'] > -12).where(df['latitude'] < -10.8).where(df['longitude'] > -37.2).dropna()

df.sort_values(['track_id', 'time'], inplace=True)

# sns.lineplot(x='latitude', y='longitude', hue='track_id', sort=None, data=df)
# plt.show()


# given a point at time t, predict time t + 1
class PhysicsModel(object):
    def __init__(self, state, time):
        self.state = state
        self.time = time

        self.state_old = None
        self.time_old = None

        self.transition = np.eye(state.size)

    def __call__(self, new_state, time):
        delta = time - self.time
        self.transition[((0, 1), (2, 3))] = delta

        transition_accel = np.block([[np.diag([delta ** 2 / 2] * 2)], [np.diag([delta] * 2)]])
        second_order = (self.state[:2] - self.state_old[:2]) / (self.time - self.time_old) if self.time_old else 0
        acceleration = ((new_state[:2] - self.state[:2]) / delta + second_order) / delta

        self.state_old = self.state
        self.time_old = self.time

        self.state = new_state
        self.time = time

        return self.transition @ new_state + transition_accel @ acceleration


class KalmanFilter(object):
    def __init__(self, model, measurement, variance):
        self.model = model
        self.state = measurement
        self.covariance = np.diag(variance)

    def __call__(self, measurement, variance, time):
        predicted = self.model(measurement, time)
        # derived from cov(predicted)
        covariance_predicted = self.model.transition @ self.covariance @ self.model.transition.T

        gain = self.covariance @ np.linalg.inv(covariance_predicted + np.diag(variance))

        self.state = predicted + gain @ (measurement - predicted)
        self.covariance = (np.eye(predicted.size) - gain) @ covariance_predicted

        return self.state


for track_id in np.unique(df['track_id']):
    print(track_id)
    track_df = df.where(df['track_id'] == track_id).dropna()

    diff = np.max(track_df, axis=0)[['latitude', 'longitude']] - np.min(track_df, axis=0)[['latitude', 'longitude']]
    variances = [min(diff) / 40] * 2

    # simulate sensor noise
    track_df[['longitude']] += np.random.normal(scale=variances[0], size=track_df[['longitude']].shape)
    track_df[['latitude']] += np.random.normal(scale=variances[1], size=track_df[['latitude']].shape)

    sns.lineplot(x='latitude', y='longitude', sort=None, data=track_df)
    sns.scatterplot(x='latitude', y='longitude', data=track_df)

    # data normalization
    statistics = {
        'longitude': {
            'mean': float(np.mean(track_df[['longitude']])),
            'std': float(np.std(track_df[['longitude']]))
        },
        'latitude': {
            'mean': float(np.mean(track_df[['latitude']])),
            'std': float(np.std(track_df[['latitude']]))
        }
    }

    track_df[['longitude']] = (track_df[['longitude']] - statistics['longitude']['mean']) / statistics['longitude']['std']
    track_df[['latitude']] = (track_df[['latitude']] - statistics['latitude']['mean']) / statistics['latitude']['std']
    variances = [var / std**2 for var, std in zip(variances, [statistics['longitude']['std'], statistics['latitude']['std']])]

    track_df_iter = track_df.iterrows()
    previous = next(track_df_iter)[1]

    position_previous = previous[['latitude', 'longitude']].astype(float).values[..., None]
    time_previous = previous['time'].timestamp()

    # initialize the filter with location and zero velocity
    initial_state = np.vstack([position_previous, [0], [0]])
    model = PhysicsModel(initial_state, time_previous)
    kalman_filter = KalmanFilter(model, initial_state, [*variances, 1, 1])

    predictions = []
    for i, point in track_df_iter:
        position = np.array(point[['latitude', 'longitude']].astype(float).values[..., None])
        time = point['time'].timestamp()

        velocity = (position - position_previous) / (time - time_previous)
        velocity_variance = [sum(variances) / (time - time_previous)**2] * 2
        prediction = kalman_filter(np.vstack([position, velocity]), [*variances, *velocity_variance], time)

        predictions.append(np.squeeze(prediction))

        position_previous = position
        time_previous = time

    predictions = pd.DataFrame(predictions, columns=['latitude', 'longitude', 'velocity_x', 'velocity_y'])

    # denormalize
    predictions[['longitude']] = predictions[['longitude']] * statistics['longitude']['std'] + statistics['longitude']['mean']
    predictions[['latitude']] = predictions[['latitude']] * statistics['latitude']['std'] + statistics['latitude']['mean']
    predictions[['velocity_x']] *= statistics['longitude']['std']
    predictions[['velocity_y']] *= statistics['latitude']['std']

    sns.lineplot(x='latitude', y='longitude', sort=None, data=predictions)

    plt.show()
