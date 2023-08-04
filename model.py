import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=2048, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)  # 初始化一个多头注意力机制
        self.norm1 = nn.LayerNorm(embed_dim)  # 初始化一个 LayerNorm 层，用于标准化输入 x
        self.norm2 = nn.LayerNorm(embed_dim)  # 标准化前向神经网络的输出
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # 处理注意力机制的输出
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # 将前向神经网络的输出映射回嵌入维度
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)  # dropout加强鲁棒性，减少过拟合

    def forward(self, x):
        residual = x  # 残差链接
        x = self.norm1(x)  # 标准化
        x, _ = self.attn(x, x, x)  # 传给注意力机制，计算qkv
        x = self.dropout1(x)
        x = x + residual  # 将多头注意力输出与残差链接

        residual = x
        x = self.norm2(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + residual

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim  # 输出特征向量的维数
        self.num_patches = (image_size // patch_size) ** 2  # 图像被切分成的 patch 的数量
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # 输入图像->特征向量

    def forward(self, x):
        x = self.projection(x)  # (batch_size, embed_dim, num_patches)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)展平成2维
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)使得特征向量在第二个维度，方便后续处理
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, hidden_dim,
                 dropout):
        super(VisionTransformer, self).__init__()
        # 将输入图像数据转化成特征向量序列
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        # 表示整个图像序列的特征向量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 为每个特征向量添加位置信息
        self.positional_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim))
        # 对特征向量序列进行多头自注意力计算和前馈神经网络计算
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        # 将特征向量序列转化为分类结果
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b = x.size(0) # 获取batch_size

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = self.patch_embedding(x)

        # 在特征向量序列的开头添加 CLS Token，并与位置编码相加
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding

        # 对特征向量序列进行多个注意力块的计算
        for attn_block in self.attention_blocks:
            x = attn_block(x)

        # 获取 CLS Token 的输出，并通过全连接层将其转换为分类结果
        x = x[:, 0]  # 获取 cls token 的输出
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas


def get_model():
    image_size = 32
    patch_size = 4
    in_channels = 3
    num_classes = 100
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    hidden_dim = 512
    dropout = 0.1

    model = VisionTransformer(image_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers,
                              hidden_dim, dropout)
    return model
