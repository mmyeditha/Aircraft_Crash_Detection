from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchviz import make_dot
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

def train_model(net, learning_rate, num_epochs):
    loss, optimizer = net.get_loss(learning_rate)
    print_every = 20
    idx = 0
    train_hist_x = []
    train_loss_hist = []
    test_hist_x = []
    test_loss_hist = []
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.BCELoss()
    grad_magnitudes = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            input = data['image']
            label = data['label']
            # print(f'input: {type(input)}\nlabel: {type(label)}')
            input, label = Variable(input).to(device), Variable(label).to(device)

            # Reset gradient and compute loss
            optimizer.zero_grad()
            output = net(input)
            #print(output.size(), label.size())
            
            loss_size = loss(output, label)
            # backpropagate and compute gradient
            loss_size.backward()
            # update model weights
            optimizer.step()
            # compute stats
            running_loss += loss_size.data.item()
            
            # print stats
            if i%print_every == print_every-1:
                print(f"Epoch {epoch+1}, Iteration {i+1} train_loss: {running_loss/print_every}")
                train_loss_hist.append(running_loss / print_every)
                train_hist_x.append(idx)
                running_loss = 0.0
            idx += 1

        total_test_loss = 0
        for i, data in enumerate(test_loader):
            inputs = data['image']
            labels = data['label']
            # Wrap tensors in Variables
            try:
              inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            except:
              print(inputs, labels)
            # Forward pass
            test_outputs = net(inputs)
            test_loss_size = loss(test_outputs, labels)
            total_test_loss += test_loss_size.data.item()
        test_loss_hist.append(total_test_loss / len(test_loader))
        test_hist_x.append(idx)

        print("Validation loss = {:.4f}".format(
            total_test_loss / len(test_loader)))
    return train_hist_x, test_hist_x, train_loss_hist, test_loss_hist

def eval_model(net)
    net.eval()
    tot_correct = 0
    tot_incorrect = 0
    tot_false_pos = 0
    tot_false_neg = 0
    tot_true_pos = 0
    tot_true_neg = 0
    tot = 0
    for i, data in enumerate(test_loader):
        inputs = data['image']
        labels = data['label']
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        outputs = net(inputs)

        for i in range(32):
            tot += 1
            img_tensor = outputs[i, :].cpu().detach().numpy()
            label = labels[i].cpu().detach().numpy()
            pred = np.where(img_tensor == (max(img_tensor)))[0][0]
            if pred == label:
                tot_correct += 1
                if pred == 1:
                    tot_true_pos += 1
                else:
                    tot_true_neg += 1
            else:
                tot_incorrect += 1
                if pred == 1:
                    tot_false_pos += 1
                else:
                    tot_false_neg += 1
    return tot_correct, tot_incorrect, tot_false_neg, tot_false_pos, tot_true_pos,
            tot_true_neg, tot