#!/usr/bin/python3
import os
import math
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

    input_data = {'A': A, 'B': B}
    return input_data


def calculate_distance(feature: list, punkt: list) -> int:
    if len(feature) != len(punkt):
        raise Exception('Liczba cech punktu jest inna niż klasy.')

    distance = 0.0
    feature = list(map(int, feature))
    for i in range(len(feature)):
        distance += (feature[i] - punkt[i])**2 
    return math.sqrt(distance)


def knn(k: int, dane: dict, punkt: list) -> None:
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
    print(distances_A)
    distances_B.sort()
    print(distances_B)
    vote = {'A': 0, 'B': 0}
    for i in range(k):
        if distances_A[i] == distances_B[i]:
            continue
        elif distances_A[i] < distances_B[i]:
            vote['A'] += 1
        else:
            vote['B'] += 1

    if vote['A'] > vote['B']:
        print('Punkt nalezy do klasy A')
    else:
        print('Punkt nalezy do klasy B')


if __name__ == "__main__":
    input_data = read_data(config.path_data)
    knn(config.k, input_data, config.punct)