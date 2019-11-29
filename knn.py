#!/usr/bin/python3
import numpy as np
import config
import random

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


def randomize_centres(k: int, puncts: list) -> list:
    k_array = []
    while len(k_array) != 2:
        val = random.choice(puncts)
        if list(val) not in k_array:
            k_array.append(list(val))
    return k_array


def calculate_new_classes(k: int, puncts: list, centres: list) -> list:
    test = [[] for i in range(k)]

    for feature in puncts:
        smallest_distance = calculate_distance(centres[0], feature)
        flag = 0
        for i in range(k):
            if smallest_distance > calculate_distance(centres[i], feature):
                smallest_distance = calculate_distance(centres[i], feature)
                flag = i
        test[flag].append(feature)
    return test


def clusterise(k: int, puncts: list, centres: list) -> list:
    flag = True
    error = 0.001
    while flag:
        k_class = calculate_new_classes(k, puncts, centres)
        
        old_centr = centres.copy()

        for i in range(k):
            centres[i] = calculate_average(k_class[i])

        flag = False
        for i in range(k):
            if calculate_distance(old_centr[i], centres[i]) > error:
                flag = True
                break
    return k_class, centres

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
      
        all_distances = list(set().union(distances_A, distances_B))
        all_distances.sort()
        
        vote = {'A': 0, 'B': 0}
        for i in range(k):
            if all_distances[i] in distances_A:
                vote['A'] += 1  
            else:
                vote['B'] += 1

        if vote['A'] > vote['B']:
            ost_vote['A'] += 1
        elif vote['A'] < vote['B']:
            ost_vote['B'] += 1
        else:
            ost_vote['A'] += 1
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


def kmn(k: int, dane: dict, puncts: list) -> None:
    ost_vote = {'A': 0, 'B': 0}
    
    k_A = randomize_centres(k, dane['A'])
    k_B = randomize_centres(k, dane['B'])

    class_A, k_A = clusterise(k, dane['A'], k_A)
    class_B, k_B = clusterise(k, dane['B'], k_B)
   
    for punct in puncts:
        distances_A = []
        distances_B = []
        for i in range(k):
            distances_A.append(calculate_distance(k_A[i], punct))
            distances_B.append(calculate_distance(k_B[i], punct))

        all_distances = list(set().union(distances_A, distances_B))
        all_distances.sort()

        vote = {'A': 0, 'B': 0}
        for i in range(k):
            if all_distances[i] in distances_A:
                vote['A'] += 1  
            else:
                vote['B'] += 1

        if vote['A'] > vote['B']:
            ost_vote['A'] += 1
        elif vote['A'] < vote['B']:
            ost_vote['B'] += 1
        else:
            ost_vote['A'] += 1
    print("knm: ", ost_vote)


def fischer(dane: dict) -> None:
    
    feature_A_avg = np.average(dane['A'], axis = 0)
    feature_B_avg = np.average(dane['B'], axis = 0)

    std_dev_A = np.std(dane['A'], axis = 0)
    std_dev_B = np.std(dane['B'], axis = 0)

    fischer_feature = []

    for i in range(len(feature_A_avg)):
        value = (abs(feature_A_avg[i] - feature_B_avg[i]) / (std_dev_A[i] + std_dev_B[i]))
        fischer_feature.append(value)
    print(fischer_feature)

    print("The most significant feature is:  ", fischer_feature.index(max(fischer_feature)))


if __name__ == "__main__":
    input_data = read_data(config.class_data)
    # knn(config.k, input_data, config.punct_data)
    # mn(input_data, config.punct_data)
    kmn(config.k, input_data, config.punct_data)
    fischer(input_data)