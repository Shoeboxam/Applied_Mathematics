import requests
import re
import random
from collections import defaultdict

# Uses KMeans to cluster sentences in a random book from project Gutenberg. Very little preprocessing done
book_url = 'http://www.gutenberg.org/cache/epub/32971/pg32971.txt'
book_text = requests.get(book_url).text

num_clusters = 25
output_path = 'book-k-means-output.txt'


# preprocess to a set of unique alphanumeric-only words
def preprocess(text):
    return set(re.sub('[^0-9a-zA-Z ]+', '', text).split(' '))


sentence_dict = {i: preprocess(sentence) for i, sentence in enumerate(book_text.split('.'))}
seeds = random.sample(range(len(sentence_dict)), num_clusters)


def distance_jaccard(a, b):
    return 1 - len(a & b) / len(a | b)


def distance_sentences(id1, id2):
    return distance_jaccard(sentence_dict[id1], sentence_dict[id2])


class KMeans(object):
    def __init__(self, distance, seeds):
        self.distance = distance
        self.centroids = seeds

    def fit(self, data):

        while True:
            print(f'Sum Squared Error: {self.sse(data)}')
            voronoi = defaultdict(list)

            # label each point
            for observation in data:
                voronoi[self.classify(observation)].append(observation)

            # recompute centroids
            centroids = {min(
                cluster,
                key=lambda point: sum(self.distance(point, other)**2 for other in cluster)
            ) for cluster in voronoi.values()}

            if self.centroids == centroids:
                break

            self.centroids = centroids

    def classify(self, point):
        return min(self.centroids, key=lambda c: self.distance(point, c))

    def sse(self, data):
        return sum(self.distance(self.classify(point), point)**2 for point in data)


model = KMeans(distance_sentences, seeds)
model.fit(range(len(sentence_dict)))

labels = defaultdict(list)
for id_sentence in sentence_dict:
    labels[model.classify(id_sentence)].append(id_sentence)

with open(output_path, 'w') as resultfile:
    for i, label in enumerate(labels):
        resultfile.write(f'{i}\t{", ".join([str(j) for j in [label, *labels[label]]])}\n')
