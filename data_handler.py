import torch
import torch.utils.data as data
import pandas

def GetTensors(sample, train_split = 0.05):
    size = sample.shape[0]  #  10000

    def to_int(s):
        result = torch.zeros((1, 81), dtype = torch.float)
        for i in range(81):
            result[0, i] = int(s[i])
        return result

    quizzes = sample.quizzes.apply(to_int)
    solutions = sample.solutions.apply(to_int)
    quizzes = torch.cat(quizzes.values.tolist())
    solutions = torch.cat(solutions.values.tolist())

    rand = torch.randperm(size)
    training_set = rand[:int(train_split * size)]
    testing_set = rand[int(train_split * size):]

    return data.TensorDataset(quizzes[training_set], solutions[training_set]),\
        data.TensorDataset(quizzes[testing_set], solutions[testing_set])


def LoadData(test_count = 1000000):
    dataset = pandas.read_csv("sudoku.csv", sep = ',')
    sample = dataset.sample(test_count)
    training_set, testing_set = GetTensors(sample)
    return training_set, testing_set
