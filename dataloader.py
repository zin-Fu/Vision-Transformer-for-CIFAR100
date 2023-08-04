import numpy as np
from utils import *
from config import *

mean, std = compute_mean_std()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量
        transforms.Normalize(mean, std),  # ToTensor要放在Normalize之前不然后面加载图像会报错
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量
        transforms.Normalize(mean, std)
    ])
}

train_dataset = datasets.CIFAR100(root='data',
                                  train=True,
                                  transform=data_transforms['train'],
                                  download=True)
test_dataset = datasets.CIFAR100(root='data',
                                 train=False,
                                 transform=data_transforms['test'])

# create valid dataset
num_train = int(len(train_dataset) * TRAIN_SIZE)  # 设置训练集容量
indices = list(range(num_train))  # 表示训练集中样本的索引列表
split = int(np.floor(VALID_SIZE * num_train))  # 创建训练集和验证集的分界点
np.random.shuffle(indices)  # 随机打乱indices列表中的元素顺序，这样可以在训练集和验证集中都包含来自不同类别的样本
train_idx, valid_idx = indices[split:], indices[:split]  # 将indices列表分为训练索引和验证索引两部分
# print("Total train dataset: "+str(len(train_idx)), "Total valid dataset: "+str(len(valid_idx)))


# 将训练集和验证集中的样本按照给定索引列表采样,train_sampler和valid_sampler表示训练集和验证集的采样器，可以通过它们创建相应的数据加载器
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           sampler=train_sampler,
                                           num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           sampler=valid_sampler,
                                           num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=NUM_WORKERS)





