import torch
import torch.nn as nn
import torch.nn.functional as F


def CreateConstraintMask():
#  constraintmask[i][zone][j] = 1 <-> j influences i in the current zone; 0 otherwise
#  zone = 0 -> rows; zone = 1 -> columns; zone = 2 -> boxes
    constraint_mask = torch.zeros((81, 3, 81), dtype = torch.float)

    #  row -> zone = 0
    for i in range(81):
        start_index = 9 * (i // 9)
        for j in range(9):
            constraint_mask[i, 0, start_index + j] = 1

    #  column -> zone = 1
    for i in range(81):
        start_index = i % 9
        for j in range(9):
            constraint_mask[i, 1, start_index + 9 * j] = 1

    #  box -> zone = 2
    for i in range(81):
        row = i // 9
        col = i % 9
        box_row = 27 * (row // 3)
        box_col = 3 * (col // 3)

        for j in range(9):
            row = 9 * (j // 3)
            col = j % 3
            constraint_mask[i, 2, box_row + row + box_col + col] = 1

    return constraint_mask


class SudokuNetwork(nn.Module):
    def __init__(self, constraint_mask, n = 9, hidden_size = 100):
        super(SudokuNetwork, self).__init__()

        self.constraint_mask = constraint_mask.view(1, n * n, 3, n * n, 1)
        self.n = n
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(3 * n, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, n)
        self.softmax = nn.Softmax(dim = 1)


    #  x -> (batch_size x n*n x n)
    def forward(self, x):
        n = self.n
        batch_size = x.shape[0]
        constraint_mask = self.constraint_mask

        max_empty = (x.sum(dim = 2) == 0).sum(dim = 1).max() #  nr maxim de casute goale dintr-unul batch

        x_pred = x.clone()
        for i in range(max_empty):
            constraints = (x.view(batch_size, 1, 1, n * n, n) * constraint_mask).sum(dim = 3)
            #  (batch_size x 81 x 3 x 9)
            #  81 = numarul la care ne referim in constraint_mask
            #  3 = tipul de constraint
            #  9 = vector de frecventa -> cate valori de 1, 2, .., 9 influenteaza nr in cauza

            empty_cell_mask = (x.sum(dim = 2) == 0)
            constraints = constraints.reshape(batch_size, n * n, 3 * n)

            layer = self.linear1(constraints[empty_cell_mask])
            layer = F.relu(layer)
            layer = self.linear2(layer)
            layer = self.softmax(layer)

            x_pred[empty_cell_mask] = layer

            predictions = torch.zeros_like(x_pred)
            predictions[empty_cell_mask] = layer

            #  gasim predictia cea mai "buna"
            values, indices = predictions.max(dim = 2)
            best_batch_predictions = values.max(dim = 1)[1]  # indici

            non_zero_batches = empty_cell_mask.sum(dim = 1).nonzero().view(-1)  # indicii patratelor de sudoku care au macar un 0

            best_batch_predictions = best_batch_predictions[non_zero_batches]  # practic are aceeasi semnificatie ca inainte, doar ca eliminam patratele de sudoku care nu mai au casute goale

            for sudoku_index, value in zip(non_zero_batches, best_batch_predictions):
                x[sudoku_index, value, indices[sudoku_index, value]] = 1

        return x_pred, x
