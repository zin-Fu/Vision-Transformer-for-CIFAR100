import pandas as pd
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def compute_mean_std():
    # 加载CIFAR-100数据集
    dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=None)

    # 将数据集转换为PyTorch张量
    data_tensor = torch.stack([torch.Tensor(i) for i in dataset.data])

    # 计算均值和标准差
    mean = torch.mean(data_tensor, dim=(0, 1, 2)) / 255.0
    std = torch.std(data_tensor, dim=(0, 1, 2)) / 255.0

    # print(f"mean: {mean}")
    # print(f"std: {std}")

    return mean, std


def get_and_show_pictures(dataset, std, mean):
    # 获取数据集中前10张图像和对应标签
    images, labels = [], []
    for i in range(10):
        image, label = dataset[i]
        images.append(image)
        labels.append(label)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()

    # 获取数据集中所有类别的名称
    classes = dataset.classes

    for i in range(10):
        image, label = dataset[i]
        image = image * std[:, None, None] + mean[:, None, None]
        image = torch.clamp(image, 0, 1)  # 将像素值限制在[0,1]的范围内
        axs[i].imshow(np.transpose(image, (1, 2, 0)))
        axs[i].set_title("Label: {}".format(classes[label]))  # 使用类别索引从classes属性中获取类别名称
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.show()


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def print_train_status(train_acc, valid_acc, train_loss, valid_loss):
    train_acc = torch.tensor(train_acc)
    valid_acc = torch.tensor(valid_acc)
    train_loss = torch.tensor(train_loss)
    valid_loss = torch.tensor(valid_loss)

    train_loss = train_loss.cpu().detach().numpy()
    valid_loss = valid_loss.cpu().detach().numpy()
    train_acc = train_acc.cpu().detach().numpy()
    valid_acc = valid_acc.cpu().detach().numpy()
    # Creating DataFrames for train and validation results
    data_train = {'Train Loss': train_loss, 'Train Accuracy': train_acc}
    df_train = pd.DataFrame(data=data_train)
    df_train.index += 1

    data_valid = {'Valid Loss': valid_loss, 'Valid Accuracy': valid_acc}
    df_valid = pd.DataFrame(data=data_valid)
    df_valid.index += 1

    # Plotting train and validation results in subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].plot(df_train.index, df_train['Train Loss'], label='Train Loss')
    ax[0].plot(df_valid.index, df_valid['Valid Loss'], label='Valid Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training/Validation Loss')
    ax[0].legend()

    ax[1].plot(df_train.index, df_train['Train Accuracy'], label='Train Accuracy')
    ax[1].plot(df_valid.index, df_valid['Valid Accuracy'], label='Valid Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training/Validation Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
