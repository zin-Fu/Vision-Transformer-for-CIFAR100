import torch

VALID_SIZE = 0.2  # 验证集切片
TRAIN_SIZE = 0.1
BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

