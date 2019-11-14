#!/usr/bin/python3
import numpy as np
import config


def read_data(path: str) -> dict:
    with open(path) as file:
        data = file.read().split('\n')

    A = [line[2:].split(',') for line in data if line[0] == 'A']
    B = [line[2:].split(',') for line in data if line[0] == 'B']

    input_data = {'A': np.array(A, np.dtype("float32")), 
        'B': np.array(B, np.dtype("float32"))}
    return input_data


def calculate_distance(feature: np.ndarray, punct: np.ndarray) -> float:
    if len(feature) != len(punct):
        raise Exception('''The number of features of point is diffrent   
                than the number of features of class.''')

    distance = 0.0
    for i in range(len(feature)):
        distance += (feature[i] - punct[i])**2 
    return np.sqrt(distance)


def calculate_average(features: np.ndarray) -> np.ndarray:
    return np.average(features, axis=0)


def knn(k: int, dane: dict, puncts: list) -> None:
    ost_vote = {'A': 0, 'B': 0}

    for punct in puncts:
        distances_A = []
        distances_B = []

        for nn_class, features in dane.items():
            for feature in features:
                try:
                    if nn_class == 'A':
                        distances_A.append(calculate_distance(feature, punct))    
                    else:
                        distances_B.append(calculate_distance(feature, punct))
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
    print("knn: ", ost_vote)


def mn(dane: dict, puncts: list) -> None:
    ost_vote = {'A': 0, 'B': 0}

    average_A = calculate_average(dane['A'])
    average_B = calculate_average(dane['B'])

    for punct in puncts:
        distance_a = calculate_distance(average_A, punct)
        distance_b = calculate_distance(average_B, punct)

        if distance_a < distance_b:
            ost_vote['A'] += 1
        else:
            ost_vote['B'] += 1

    print("mn: ", ost_vote)


if __name__ == "__main__":
    input_data = read_data(config.class_data)
    knn(config.k, input_data, config.punct_data)
    mn(input_data, config.punct_data)