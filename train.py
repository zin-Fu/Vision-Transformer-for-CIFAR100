import time
from model import *
from dataloader import *
from utils import *
from tqdm import tqdm


def train(model, optimizer):
    best_acc = 0.0
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            epoch_train_loss = 0.0
            for batch_idx, (features, targets) in enumerate(train_loader):
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                logits, probas = model(features)
                cost = F.cross_entropy(logits, targets)
                optimizer.zero_grad()

                cost.backward()
                optimizer.step()

                # 更新进度条和训练损失
                epoch_train_loss += cost.item()
                pbar.update(1)
                pbar.set_description('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                                     % (epoch + 1, NUM_EPOCHS, batch_idx + 1,
                                        len(train_loader), cost))

            # 计算训练损失并将其存储在列表中
            epoch_train_loss /= len(train_loader)
            train_loss.append(epoch_train_loss)

        # 在验证集上测试模型
        model.eval()
        with torch.set_grad_enabled(False):  # 评估时节约内存
            train_acc.append(compute_accuracy(model, train_loader, device=DEVICE))
            valid_acc.append(compute_accuracy(model, valid_loader, device=DEVICE))
            valid_epoch_loss = 0.0
            for batch_idx, (features, targets) in enumerate(valid_loader):
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                logits, probas = model(features)
                cost = F.cross_entropy(logits, targets)

                # 更新验证损失
                valid_epoch_loss += cost.item()

            valid_epoch_loss /= len(valid_loader)
            valid_loss.append(valid_epoch_loss)
            print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%% | Train Loss: %.4f | Valid Loss: %.4f' % (
            epoch + 1, NUM_EPOCHS, train_acc[-1], valid_acc[-1], train_loss[-1], valid_loss[-1]))

            # 根据验证集准确率更新最佳模型参数
            if valid_acc[-1] > best_acc:
                best_acc = valid_acc[-1]
                torch.save(model.state_dict(), 'best.pt')
                print('Best validation accuracy updated! Saving model parameters to best.pt')

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    print_train_status(train_acc, valid_acc, train_loss, valid_loss)

