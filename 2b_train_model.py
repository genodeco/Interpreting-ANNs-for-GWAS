import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import random
from sklearn import metrics
from model import *


inpt = "mock_genomes.npy" #input genotype file in numpy npy format
epochs = 50 #number of epochs for training
batch_size = 64 #batch size
lr = 0.0001 #learning rate
init_dropout = 0.50 #amount initial random masking of input data
save_that = 1000 #model saving frequency - at each xth epoch, model will be outputted along with evaluation metrics
out_dir = "mock_output" #output directory
status_ = "mock_case_control_status.npy" #input case/control status file for each corresponding genotype in genotype file
gpu = 1 #number of gpus - only 1 gpu scenario was tested

##Input data preperation and train/test split
df = np.load(inpt, allow_pickle=True)
status = np.load(status_, allow_pickle=True)
status = status.reshape(-1,1)
df = np.concatenate((status, df), axis=1)
df = df.astype(int)
df_case = df[df[:,0]==1].copy()
df_control = df[df[:,0]==0].copy()
df_control_test = df_control[:150,:].copy()
df_control = df_control[150:,:]
df_case_test = df_case[:150,:].copy()
df_case = df_case[150:,:]
df = np.concatenate((df_control, df_case))
df_test = np.concatenate((df_control_test, df_case_test))
df_labels = df[:,0]
df_test_labels = df_test[:,0]
df_labels = df_labels.astype(float)
df_test_labels = df_test_labels.astype(float)
df = df[:,1:]
df_test = df_test[:,1:]

#Train models with different seeds
seeds = list(range(0,10))
for seed in seeds:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataloader = torch.utils.data.DataLoader([ [df[i], df_labels[i]] for i in range(len(df_labels))], shuffle=True, batch_size=batch_size, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader([ [df_test[i], df_test_labels[i]] for i in range(len(df_test))], shuffle=True, batch_size=batch_size, pin_memory=True)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")
    
    classifi = Classifier(data_shape=df.shape[1], init_dropout=init_dropout)

    if (device.type == 'cuda') and (gpu > 1):
        classifi = nn.DataParallel(classifi, list(range(gpu)))
    classifi.to(device)
    optimizer = torch.optim.Adam(classifi.parameters(), lr=lr, weight_decay=1e-3)

    losses = []
    test_losses = []
    accuracies = []
    test_accuracies = []

    print(f"Training loop for seed {seed} is starting...")
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        train_acc = 0.0
        for data in dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y = y - torch.normal(mean=0, std=0.1, size=(1, y.shape[0]), device=device).flatten() #adding noise to the labels
            y = torch.clamp(y, min=0, max=1) #clamping label values for cross-entropy loss
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1])).float()
            x_inpt = x.clone()
            x_inpt = x_inpt - torch.normal(mean=0, std=0.1, size=(x.shape[0], 1, x.shape[2]), device=device)

            optimizer.zero_grad()

            y_hat = classifi(x_inpt)
            loss_pred = F.binary_cross_entropy(y_hat.flatten(), y.float(), reduction="mean")
            loss_pred.backward()
            optimizer.step()

            temp = y_hat.detach().flatten().round()
            train_acc_batch = (temp == y.round()).float().mean()

            x_test, y_test = next(iter(dataloader_test))
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            x_test = torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])).float()
            x_test_inpt = x_test.clone()
            classifi.eval()
            with torch.no_grad():
                y_hat_test = classifi(x_test_inpt)
                loss_pred_test = F.binary_cross_entropy(y_hat_test.flatten(), y_test.float(), reduction="mean")

            test_acc_batch = (y_hat_test.detach().flatten().round() == y_test).float().mean()

            train_loss += loss_pred.item()
            test_loss += loss_pred_test.item()
            test_acc += test_acc_batch.item()
            train_acc += train_acc_batch.item()

            classifi.train()

        threshold = 0.5
        y_pred_class = (y_hat_test > threshold).float().flatten()
        tn, fp, fn, tp = metrics.confusion_matrix(y_test.detach().cpu().numpy(), y_pred_class.detach().cpu().numpy()).ravel()

        if fp != 0 and tp != 0:
            fdr = fp / (fp + tp)
        else:
            fdr = 999
        train_loss = train_loss/len(dataloader)
        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)
        train_acc = train_acc/len(dataloader)

        print('Epoch: {}  Loss: {:.4f}  Test_loss: {:.4f} Train_acc: {:.4f} Test_acc: {:.4f} Test_FDR: {:.4f}'.format(epoch, train_loss, test_loss, train_acc, test_acc, fdr))
        losses.append(round(train_loss, 3))
        test_losses.append(round(test_loss, 3))
        accuracies.append(round(train_acc, 3))
        test_accuracies.append(round(test_acc, 3))

        #At each save_that epoch and last epoch, output trained models and evaluation plots
        if epoch == epochs-1:
        #if epoch%save_that == 0 or epoch == epochs-1:
            torch.save({
            'classifi': classifi.state_dict(),
            'optimizer': optimizer.state_dict()},
            f'{out_dir}/s{seed}_e{epoch}.model')

            fig = plt.figure(figsize=(12,6))
            plt.plot(losses, label="Train")
            plt.plot(test_losses, label="Test")
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc=1)
            fig.savefig(out_dir + '/s' + str(seed) + '_e' + str(epoch) + 'loss.pdf', format='pdf')
            fig = plt.figure(figsize=(12,6))
            plt.plot(accuracies, label="Train")
            plt.plot(test_accuracies, label="Test")
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc=1)
            fig.savefig(out_dir + '/s' + str(seed) + '_e' + str(epoch) + 'accuracy.pdf', format='pdf')

            fpr, tpr, _ = metrics.roc_curve(y_test.flatten().detach().cpu().numpy(), y_hat_test.flatten().detach().cpu().numpy())
            auc = metrics.roc_auc_score(y_test.flatten().detach().cpu().numpy(), y_hat_test.flatten().detach().cpu().numpy())

            fig = plt.figure(figsize=(6,6))
            plt.plot(fpr,tpr, label="auc="+str(auc))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc=4)
            fig.savefig(out_dir + '/s' + str(seed) + '_e' + str(epoch) + 'auc.pdf', format='pdf')
            
    training_time = time.time() - start_time
    print(f"Seed {seed} training lasted {training_time} seconds.\n")

