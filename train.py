import data_handler as d
import network as n
import torch
import torch.nn as nn

BATCH_SIZE = 100

def one_hot_encode(s):
    result = torch.zeros((81, 9), dtype = torch.float)
    for i in range(81):
        if int(s[i]) > 0:
            result[i, int(s[i]) - 1] = 1
    return result

training_set, testing_set = d.LoadData()

training_set = torch.utils.data.DataLoader(training_set, batch_size = BATCH_SIZE, shuffle = True)

testing_set = torch.utils.data.DataLoader(testing_set, batch_size = BATCH_SIZE, shuffle = False)

print('Finished loading data...')

constraint_mask = n.CreateConstraintMask()
net =  n.SudokuNetwork(constraint_mask)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)


print('Started training...')
for epoch in range(5):
    print('Epoch %d' % (epoch))
    running_loss = 0.0

    for i, data in enumerate(training_set, 0):
        input_raw, correct_output_raw = data

        input = torch.zeros((BATCH_SIZE, 81, 9), dtype = torch.float)
        correct_output = torch.zeros((BATCH_SIZE, 81, 9), dtype = torch.float)


        for (j, test) in enumerate(input_raw, 0):
            input[j] = one_hot_encode(test)

        for (j, test) in enumerate(correct_output_raw, 0):
            correct_output[j] = one_hot_encode(test)

        optimizer.zero_grad()
        x_pred, x_fill = net(input)
        loss = criterion(x_pred, correct_output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 9:
            print('[%d %d] loss %.5f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished training\n')

total_correct = 0
total_cells = 0
with torch.no_grad():
    for (i, data) in enumerate(testing_set, 0):
        input_raw, correct_output_raw = data

        input = torch.zeros((BATCH_SIZE, 81, 9), dtype = torch.float)
        correct_output = torch.zeros((BATCH_SIZE, 81, 9), dtype = torch.float)

        for (j, test) in enumerate(input_raw, 0):
            input[j] = one_hot_encode(test)

        for (j, test) in enumerate(correct_output_raw, 0):
            correct_output[j] = one_hot_encode(test)

        x_pred, x_fill = net(input)
        errors = x_fill.max(dim = 2)[1] != correct_output.max(dim = 2)[1]

        count = errors.sum().item()
        print('Errors in the current batch (%d): %d. Accuracy %.5f%c' % (i, count, 100 * (8100 - count) / 8100, '%'))

        total_cells += 8100
        total_correct += (8100 - count)

print('Total accuracy: %.5f%c' % (100 * total_correct / total_cells, '%'))


#  99.109%
