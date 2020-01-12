#!/usr/bin/python3
import numpy as np
import random
import itertools

def read_data(path: str) -> dict:
    with open(path) as file:
        data = file.read().split('\n')

    A = [line[line.find(',')+1:].split(',') for line in data if line[0] == 'A']
    B = [line[line.find(',')+1:].split(',') for line in data if line[0] == 'Q']

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
    features_A = dane['A'][combinations]
    features_B = dane['B'][combinations]

    features_A_avg = calculate_average(features_A)
    features_B_avg = calculate_average(features_B)

    a_sample_minus_mean = np.copy(features_A)
    b_sample_minus_mean = np.copy(features_B)
    for i in range(0, len(combinations)):
        a_sample_minus_mean[i] = a_sample_minus_mean[i] - features_A_avg[i]
        b_sample_minus_mean[i] = b_sample_minus_mean[i] - features_B_avg[i]

    covariance_matrix_A = np.cov(features_A)
    covariance_matrix_B = np.cov(features_B)

    covariance_matrices_det_sum = np.linalg.det(covariance_matrix_A + covariance_matrix_B)
    means_vectors_distance = np.linalg.norm(features_A_avg - features_B_avg)

    result = means_vectors_distance / covariance_matrices_det_sum
    return result


def n_fisher(dane: dict, n: int) -> list:
    number_of_features = len(dane['A'][0])
    combinations = list(itertools.combinations(range(0, number_of_features), n))

    fisher_result = 0.0
    feature_combination = None

    for combination in combinations:
        result = fisher_for_multiple_features(dane, list(combination))
        if result > fisher_result:
            fisher_result = result
            feature_combination = combination

    feature_combination = list(feature_combination)
    return feature_combination


def sts_fisher(dane: dict, n: int) -> list:
    number_of_features = len(dane['A'][0])
    best_fisher = [fisher(dane)]
    
    fisher_result = 0.0

    for i in range(1, n):
        for j in range(0, number_of_features):
            if j in best_fisher:
                continue
            next_combination = best_fisher.copy()
            next_combination.append(j)
            result = fisher_for_multiple_features(dane, next_combination)
            if result > fisher_result:
                fisher_result = result
                feature_num = j
        best_fisher.append(feature_num)
    
    return best_fisher


def exclude_data(dane: dict, features: list) -> dict:
    A = []
    B = []
    for i in features:
        A.append(dane['A'][i])
        B.append(dane['B'][i])

    excluded_dane =  {'A': np.array(A, np.dtype("float32")), 
        'B': np.array(B, np.dtype("float32"))}
    return excluded_dane

if __name__ == "__main__":
    class_data = "Maple_Oak.txt"
    input_data = read_data(class_data)
    
    fisher_1d = fisher(input_data)
    print("The most significant feature is:", fisher_1d)
    # fisher_3d = n_fisher(input_data, 3)
    # print("3 the most significant features are:", fisher_3d)
    fisher_4d_sfs = sts_fisher(input_data, 4)
    print("4 best features according to sts:", fisher_4d_sfs)
    excluded_data = exclude_data(input_data, fisher_4d_sfs)
    # k = 2
    # punct_data = [[1, 7, 3]]
    # number_of_features = 3
    
    # knn(config.k, input_data, config.punct_data)
    # mn(input_data, config.punct_data)
    # kmn(config.k, input_data, config.punct_data)
    # sts_fisher(input_data, 3)