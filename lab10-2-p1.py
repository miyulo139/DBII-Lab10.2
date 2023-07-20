import time
import random
from rtree import index
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def create_points(N):
    return [(random.random(), random.random()) for _ in range(N)]


def knn_rtree(points, K_values):
    rtree_index = index.Index()
    for idx, point in enumerate(points):
        rtree_index.insert(idx, point)

    # Se suma el tiempo que toma para todos los Top-K solicitados
    total_time = 0
    for K in K_values:
        start = time.time()
        for point in points:
            list(rtree_index.nearest(point, K))
        end = time.time()
        total_time += (end - start)

    return total_time * 1000  # ms


def knn_sequential(points, K_values):
    total_time = 0

    # Se suma el tiempo que toma para todos los Top-K solicitados
    for K in K_values:
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute').fit(points)
        start = time.time()
        for point in points:
            nbrs.kneighbors([point])
        end = time.time()
        total_time += (end - start)

    return total_time * 1000  # ms


def performance_analysis():
    K_values = [3, 6, 9]
    N_values = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
    results = []

    for N in N_values:
        points = create_points(N)
        rtree_time = knn_rtree(points, K_values)
        seq_time = knn_sequential(points, K_values)
        results.append({"N": N, "RTree": rtree_time, "Lineal Scan": seq_time})

    df = pd.DataFrame(results)
    print(df)


performance_analysis()
