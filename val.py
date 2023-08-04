from dataloader import *
from utils import *

def evaluation_and_show(model):
    with torch.set_grad_enabled(False):
        test_acc = compute_accuracy(model, test_loader, device=DEVICE)
        print('Test accuracy: %.2f%%' % (test_acc))
        # 获取标签名称
        label_names = datasets.CIFAR100(root='data', train=False).classes

        # 显示前10张测试图像
        dataiter = iter(test_loader)  # 获取测试集数据的迭代器
        images, labels = next(iter(dataiter))  # 使用 next 函数获取数据集中的一个 batch
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)  # 移动到指定设备上
        logits, probas = model(images)
        _, predicted_labels = torch.max(probas, 1)  # 获取预测标签

        # 将图像从张量转换为NumPy数组，并还原归一化
        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # 将通道维度调整到最后一个维度
        '''
        在 PyTorch 中，数据的维度顺序通常是 (batch_size, channel, height, width)，
        其中 batch_size 表示数据批次大小，channel 表示数据的通道数，height 和 width 分别表示数据的高度和宽度。
        而在 NumPy 中，通常是 (batch_size, height, width, channel) 的顺序
        '''
        # 在训练过程中，我们将图像进行了归一化，即将图像像素值减去均值，然后除以标准差,这里×标准差再+均值
        images = images * np.array(std) + np.array(mean)  # 还原图像
        images = np.clip(images, 0, 1)  # 将图像中的像素值限制在 [0, 1] 范围内
        # 绘制图像和标注
        plt.figure(figsize=(10, 10))  # 创建一个宽度为 10 英寸、高度为 10 英寸的图像画布
        for i in range(10):  # 遍历前 10 张图像
            plt.subplot(5, 2, i+1)  # 创建一个大小为 5x2 的子图，并将第 i+1 个子图作为当前子图, 这样可以将 10 张图像排列成 5 行 2 列的形式，方便可视化。
            plt.imshow(images[i])
            plt.title(f"Predicted: {label_names[predicted_labels[i]]}, True: {label_names[labels[i]]}")
            plt.axis('off')  # 关闭坐标轴显示
        plt.show()
