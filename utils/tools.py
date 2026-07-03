import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 2 else args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate * 1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def save_loss_history(history, csv_path):
    pd.DataFrame(history).to_csv(csv_path, index=False)


def save_learning_curve(history, image_path):
    epochs = history.get('epoch', [])
    train_loss = history.get('train_loss', [])
    vali_loss = history.get('vali_loss', [])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('TimeBase: Training and Validation Loss', fontsize=16, y=0.97)

    if len(train_loss) > 0:
        axes[0].plot(epochs, train_loss, marker='o', color='tab:blue', linewidth=2)
        axes[0].set_title('Training Loss')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        for x, y in zip(epochs, train_loss):
            axes[0].text(x, y, f'{y:.3f}', fontsize=8, ha='center', va='bottom')

    if len(vali_loss) > 0:
        axes[1].plot(epochs, vali_loss, marker='o', color='tab:red', linewidth=2)
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        for x, y in zip(epochs, vali_loss):
            axes[1].text(x, y, f'{y:.3f}', fontsize=8, ha='center', va='bottom')

    plt.xticks(epochs)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
