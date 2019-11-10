#!/usr/bin/python3
import numpy as np

import config

def read_data(path: str) -> dict:
    A = []
    B = []

    with open(path) as file:
        data = file.read().split('\n')

    for line in data:
        line = line.split(',')
        if line[0] == 'A':
            A.append(line[1:])
        else:
            B.append(line[1:])

    input_data = {'A': np.array(A, np.dtype("float32")), 'B': np.array(B, np.dtype("float32"))}
    return input_data


def calculate_distance(feature: np.ndarray, punkt: np.ndarray) -> float:
    if len(feature) != len(punkt):
        raise Exception('Liczba cech punktu jest inna niż klasy.')

    distance = 0.0
    for i in range(len(feature)):
        distance += (feature[i] - punkt[i])**2 
    return np.sqrt(distance)

def calculate_average(features: np.ndarray) -> np.ndarray:
    return np.average(features, axis=0)

def knn(k: int, dane: dict, punkts: list) -> None:
    ost_vote = {'A': 0, 'B': 0}

    for punkt in punkts:
        distances_A = []
        distances_B = []

        for nn_class, features in dane.items():
            for feature in features:
                try:
                    if nn_class == 'A':
                        distances_A.append(calculate_distance(feature, punkt))    
                    else:
                        distances_B.append(calculate_distance(feature, punkt))
                except Exception as error:
                    print(error)
                    return (2)

        distances_A.sort()
        distances_B.sort()

        vote = {'A': 0, 'B': 0}

        for i in range(k):
            for j in range(k):
                if distances_A[i] == distances_B[j]:
                    continue
                elif distances_A[i] < distances_B[j]:
                    vote['A'] += 1
                else:
                    vote['B'] += 1

        if vote['A'] > vote['B']:
            ost_vote['A'] += 1
        else:
            ost_vote['B'] += 1
    print(ost_vote)

def mn(dane: dict, punkts: list) -> None:
    ost_vote = {'A': 0, 'B': 0}

    average_A = calculate_average(dane['A'])
    average_B = calculate_average(dane['B'])

    for punkt in punkts:
        distance_a = calculate_distance(average_A, punkt)
        distance_b = calculate_distance(average_B, punkt)

        if distance_a < distance_b:
            ost_vote['A'] += 1
        else:
            ost_vote['B'] += 1

    print(ost_vote)


if __name__ == "__main__":
    input_data = read_data(config.path_data)
    knn(config.k, input_data, config.punct)
    mn(input_data, config.punct)