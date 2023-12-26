import csv
import random
import math


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def split_dataset(dataset, split_ratio=0.67):
    train_size = int(len(dataset) * split_ratio)
    train_set = random.sample(dataset, train_size)
    test_set = [data for data in dataset if data not in train_set]
    return train_set, test_set


def separate_by_class(dataset):
    separated = {}
    for vector in dataset:
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated


def summarize_dataset(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(instances)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label = max(probabilities, key=probabilities.get)
    return best_label


def get_predictions(summaries, test_set):
    predictions = [predict(summaries, data) for data in test_set]
    return predictions


def get_accuracy(test_set, predictions):
    correct = sum(1 for i in range(len(test_set)) if test_set[i][-1] == predictions[i])
    return (correct / float(len(test_set))) * 100.0


with open('6.csv') as f:
    data = [list(map(float, row)) for row in csv.reader(f)]
    training_set, test_set = split_dataset(data)
    print(f'Split {len(data)} rows into train={len(training_set)} and test={len(test_set)} rows')
    summaries = summarize_by_class(training_set)
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print(f'Accuracy of the classifier is: {accuracy}%')
