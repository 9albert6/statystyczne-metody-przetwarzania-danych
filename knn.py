#!/usr/bin/python3
import numpy as np
import random
import itertools
from typing import Tuple
from sklearn.model_selection import train_test_split
import sys

def read_data(path: str) -> dict:
    with open(path) as file:
        data = file.read().replace(' ', '').split('\n')

    # dla testowych danych
    A = [line[line.find(',')+1:].split(',') for line in data if line[0] == 'A']
    B = [line[line.find(',')+1:].split(',') for line in data if line[0] == 'Q']

    input_data = {'A': np.array(A, np.dtype("float32")), 
        'B': np.array(B, np.dtype("float32"))}
    print(input_data['B'].shape)
    return input_data


def split_data_crossvalidation(dane: dict, num_of_pieces: int) -> Tuple[dict, dict]:
    for single_class in dane.values():
        np.random.shuffle(single_class)

    pieces_A = np.array_split(dane['A'], num_of_pieces)
    pieces_B = np.array_split(dane['B'], num_of_pieces)

    train = {'A': np.concatenate(pieces_A[1:]), 'B': np.concatenate(pieces_B[1:])}
    test = {'A': pieces_A[0], 'B': pieces_B[0]}

    return train, test

def split_data_bootstrap(dane: dict, test_train_percentage: float) -> Tuple[dict, dict]:
    
    X_train, X_test, y_train, y_test = train_test_split(dane['A'], dane['B'],
                                                        test_size=test_train_percentage, random_state=42)

    train = {'A': X_train, 'B': y_train}
    test = {'A': X_test, 'B': y_test}
    return train, test

def calculate_distance(feature: np.ndarray, punct: np.ndarray) -> float:
    if len(feature) != len(punct):
        raise Exception('''The number of features of point is diffrent   
                than the number of features of class.''')

    # distance = 0.0
    # for i in range(len(feature)):
    #     distance += (feature[i] - punct[i])**2 
    # return np.sqrt(distance)
    return np.linalg.norm(feature - punct)


def calculate_average(features: np.ndarray) -> np.ndarray:
    return np.average(features, axis=0)


def randomize_centres(k: int, puncts: list) -> list:
    k_array = []
    while len(k_array) != k:
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
            val = calculate_distance(centres[i], feature)
            if smallest_distance > val:
                smallest_distance = val
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


def knn(k: int, dane: dict, dict_puncts: dict) -> float:

    features_A = dane["A"]
    features_B = dane["B"]
    correct = 0
    for classes, puncts in dict_puncts.items():
        for punct in puncts:
            vote = {'A': 0, 'B': 0}
            distances_A = []
            distances_B = []

            for feature in features_A:
                distances_A.append(calculate_distance(feature, punct))    
            for feature in features_B:
                distances_B.append(calculate_distance(feature, punct))
               
            distances_A.sort()
            distances_B.sort()
        
            all_distances = list(set().union(distances_A, distances_B))
            all_distances.sort()
        
            for i in range(k):
                if all_distances[i] in distances_A:
                    vote['A'] += 1
                elif all_distances[i] in distances_B:
                    vote['B'] += 1
            
            if vote['A'] > vote['B'] and classes == 'A':
                correct += 1
            elif vote['A'] < vote['B'] and classes == 'B':
                correct += 1
            elif vote['A'] == vote['B'] and classes == 'B':
                correct += 1

    return correct/(len(dict_puncts['A'])+len(dict_puncts['B']))


def mn(dane: dict, dict_puncts: dict) -> float:
    ost_vote = {'A': 0, 'B': 0}

    average_A = calculate_average(dane['A'])
    average_B = calculate_average(dane['B'])

    correct = 0
    for classes, puncts in dict_puncts.items():
        for punct in puncts:
            distance_a = calculate_distance(average_A, punct)
            distance_b = calculate_distance(average_B, punct)

            if distance_a < distance_b and classes == 'A':
                correct += 1
            elif distance_a > distance_b and classes == 'B':
                correct += 1

    return correct/(len(dict_puncts['A'])+len(dict_puncts['B']))


def kmn(k: int, dane: dict, dict_puncts: dict) -> float:
    correct = 0    
    k_A = randomize_centres(k, dane['A'])
    k_B = randomize_centres(k, dane['B'])

    class_A, k_A = clusterise(k, dane['A'], k_A)
    class_B, k_B = clusterise(k, dane['B'], k_B)
   
    for classes, puncts in dict_puncts.items():
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

            if vote['A'] > vote['B'] and classes == 'A':
                correct += 1
            elif vote['A'] < vote['B'] and classes == 'B':
                correct += 1
            elif vote['A'] == vote['B'] and classes == 'B':
                correct += 1
    return correct/(len(dict_puncts['A'])+len(dict_puncts['B']))


def fisher(dane: dict) -> int:
    feature_A_avg = np.average(dane['A'], axis = 0)
    feature_B_avg = np.average(dane['B'], axis = 0)

    std_dev_A = np.std(dane['A'], axis = 0)
    std_dev_B = np.std(dane['B'], axis = 0)

    fisher_feature = []

    for i in range(len(feature_A_avg)):
        value = (abs(feature_A_avg[i] - feature_B_avg[i])) / (std_dev_A[i] + std_dev_B[i])
        fisher_feature.append(value)
    return fisher_feature.index(max(fisher_feature))


def fisher_for_multiple_features(dane: dict, combinations: int) -> float:
    features_A = []
    features_B = []

    for i in range(len(dane['A'])):
        features_A.append(dane['A'][i][combinations])
    for i in range(len(dane['B'])):
        features_B.append(dane['B'][i][combinations])
    # features_A = [[1,-1],[1,0],[2,-1],[1,-1]]
    # features_B = [[1,1], [1,1], [2,2], [2,1]]
    # combinations = [20, 0]
    features_A_avg = np.average(features_A, axis=0)
    features_B_avg = np.average(features_B, axis=0)
    means_vectors_distance = np.linalg.norm(features_A_avg - features_B_avg)

    ones_A = np.ones((len(features_A), len(combinations)))
    ones_B = np.ones((len(features_B), len(combinations)))
    
    features_A_avg_ext = ones_A * features_A_avg
    features_B_avg_ext = ones_B * features_B_avg

    A_minus_mean = (features_A - features_A_avg_ext)
    B_minus_mean = (features_B - features_B_avg_ext)

    covariance_matrix_A = np.dot(A_minus_mean.T, A_minus_mean) / len(features_A)
    covariance_matrix_B = np.dot(B_minus_mean.T, B_minus_mean) / len(features_B)

    covariance_matrices_det_sum = np.linalg.det(covariance_matrix_A + covariance_matrix_B) 
    if covariance_matrices_det_sum == 0.0:
        return sys.float_info.min
    result = means_vectors_distance / covariance_matrices_det_sum
    return result


def n_fisher(dane: dict, n: int) -> list:
    number_of_features = len(dane['A'][0])
    combinations = list(itertools.combinations(range(0, number_of_features), n))
    
    fisher_result = 0.0
    feature_combination = None

    for combination in combinations:
        result = fisher_for_multiple_features(dane, list(combination))
        print("Combination:", combination, "result:", result)
        if result > fisher_result:
            fisher_result = result
            feature_combination = combination

    feature_combination = list(feature_combination)
    return feature_combination


def sfs_fisher(dane: dict, n: int) -> list:
    number_of_features = len(dane['A'][0])
    best_fisher = [fisher(dane)]
    
    for i in range(1, n):
        fisher_result = 0.0
        for j in range(0, number_of_features):
            if j in best_fisher:
                continue
            next_combination = best_fisher.copy()
            next_combination.append(j)
            result = fisher_for_multiple_features(dane, next_combination)
            # print("Combination:", next_combination, "result:", result)
            if result > fisher_result:
                fisher_result = result
                feature_num = j
        best_fisher.append(feature_num)
    
    return best_fisher


def exclude_data(dane: dict, exclude_features: list) -> dict:
    A = []
    B = []

    for i in range(len(dane['A'])):
        A.append(dane['A'][i][exclude_features])
    for i in range(len(dane['B'])):
        B.append(dane['B'][i][exclude_features])

    excluded_dane =  {'A': np.array(A, np.dtype("float32")), 
        'B': np.array(B, np.dtype("float32"))}
    return excluded_dane

if __name__ == "__main__":
    class_data = "rece.txt"
    ####VARIABLES
    ###SPLIT
    split = "crossvalidation" #crossvalidation/bootstup
    number_of_pieces = 4
    number_of_repetition = 4
    test_train_ratio = 0.2
    ###FISHER
    fisher_func = "sfs" #sfs/n_fisher
    number_of_features = 10
    ###ALGORITHM
    algorithm = "kmn" #knn/mn/kmn
    k = 7

    input_data = read_data(class_data)
    result = []
    if fisher_func is "sfs":
        main_features = sfs_fisher(input_data, number_of_features)
    else:
        main_features = n_fisher(input_data, number_of_features)
    print("Cechy", main_features)
    # main_features = [30, 15, 7, 33, 60, 55, 39, 21, 45, 22]
    data = exclude_data(input_data, main_features)
    for i in range(number_of_repetition):
        if split is "crossvalidation":
            train, test = split_data_crossvalidation(data, number_of_pieces)
        else:
            train, test = split_data_bootstrap(data, test_train_ratio)

        accuracy = 0
        if algorithm is "knn":
            accuracy = knn(k, train, test)
        elif algorithm is "mn":
            accuracy = mn(train, test)
        else:
            accuracy = kmn(k, train, test)
        result.append(accuracy)
    
    print(sum(result)/number_of_repetition)