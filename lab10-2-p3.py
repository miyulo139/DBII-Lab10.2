import time
import random
import numpy as np
from rtree import index
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from queue import PriorityQueue


# Se generaron encodings aleatorios para los N solicitados
def create_face_encodings(N, dim=128):
    return [np.random.rand(dim) for i in range(N)]


def knn_rtree(face_encodings, K):
    # Se uso PCA para reducir la dimensión de los encodings
    pca = PCA(n_components=2)
    reduced_encodings = pca.fit_transform(face_encodings)

    rtree_index = index.Index()
    for idx, encoding in enumerate(reduced_encodings):
        rtree_index.insert(idx, list(encoding) * 2)

    total_time = 0
    for encoding in reduced_encodings:
        start = time.time()
        list(rtree_index.nearest(list(encoding) * 2, K))
        end = time.time()
        total_time += (end - start)

    return total_time * 1000  # ms


def knn_sequential(face_encodings, K):
    total_time = 0
    for encoding in face_encodings:
        start = time.time()
        distances = euclidean_distances(face_encodings, [encoding])
        queue = PriorityQueue()
        for idx, distance in enumerate(distances):
            queue.put((distance, idx))
            if queue.qsize() > K:
                queue.get()
        end = time.time()
        total_time += (end - start)

    return total_time * 1000  # ms


def performance_analysis():
    K = 1 # Top-k=1
    N_values = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
    results = []

    for N in N_values:
        face_encodings = create_face_encodings(N)
        rtree_time = knn_rtree(face_encodings, K)
        seq_time = knn_sequential(face_encodings, K)
        results.append({"N": N, "RTree": rtree_time, "Sequential": seq_time})

    df = pd.DataFrame(results),
    print(df)


performance_analysis()
