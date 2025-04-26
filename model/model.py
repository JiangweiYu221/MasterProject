import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
import seaborn as sns
import os



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "best_model_selfattention.pth"
train_img = "training_metrics_selfattention.png"
confusion_matrix_path = "confusion_matrix_selfattention.png"
report_path = 'classification_report_selfattention.txt'
confusion_matrix_npy_path = "confusion_matrix_selfattention.npy"

# 加载数据
train_data = np.load('/home/mnt_disk1/model_result/data_train.npy')
train_labels = np.load('/home/mnt_disk1/model_result/label_train.npy')
val_data = np.load('/home/mnt_disk1/model_result/data_val.npy')
val_labels = np.load('/home/mnt_disk1/model_result/label_val.npy')
test_data = np.load('/home/mnt_disk1/model_result/data_test.npy')
test_labels = np.load('/home/mnt_disk1/model_result/label_test.npy')

train_data = train_data*1e6
val_data = val_data*1e6
test_data = test_data*1e6

def trainlossweight(labels):
    # 统计每个类别的出现次数
    unique_elements, counts = np.unique(labels, return_counts=True)

    # 创建一个字典来存储类别及其对应的频率
    class_frequencies = dict(zip(unique_elements, counts))

    # 计算权重
    # 权重可以是频率的倒数，也可以是其他策略
    total_samples = labels.size
    class_weights = {class_label: total_samples / (frequency * len(unique_elements)) for class_label, frequency in
                     class_frequencies.items()}

    # 将权重转换为 PyTorch 张量
    class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)
    return class_weights_tensor

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # (N, 34, 375)
        self.labels = torch.tensor(labels, dtype=torch.long)  # 修改为long类型

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TransformerEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, in_feature):
        super(TransformerEncoder, self).__init__()

        self.in_c = in_channel
        self.out_c = out_channel
        self.in_feature = in_feature

        self.convert_block = nn.Sequential(
            nn.Linear(self.in_feature, 32),
            nn.Conv2d(self.in_c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
            )
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 312)
        )
        self.encoder = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=2)
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.convert_block(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        encoded = self.transformer(x)
        encoded = encoded.transpose(1, 2)
        out = self.mlp(encoded)
        out = torch.reshape(out, [B,self.out_c,H,W])
        return out

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UNetEEG(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):  # 修改输出通道
        super(UNetEEG, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d((2, 2))
        self.bottleneck = conv_block(256, 512)
        # self.trans = TransformerEncoder(512, 32, 78)
        # self.up3 = nn.ConvTranspose2d(32, 256, kernel_size=(2, 2), stride=(2, 2))
        self.up3 = up_conv(512, 256)
        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.decoder3 = conv_block(512, 256)
        # self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
        self.up2 = up_conv(256, 128)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.decoder2 = conv_block(256, 128)
        # self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
        self.up1 = up_conv(128, 64)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.decoder1 = conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # 输出通道

    def forward(self, x):
        x = x.unsqueeze(1)
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        b = self.bottleneck(p3)

        #b = self.trans(b)

        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode='nearest')
        e3 = self.Att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode='nearest')
        e2 = self.Att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode='nearest')
        e1 = self.Att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        return self.final_conv(d1)


train_losses = []
loss_weight = trainlossweight(train_labels)

def train_model(model, train_loader, val_loader, optimizer, num_epochs=256):
    model.to(device)
    dice_loss = DiceLoss(softmax=True, to_onehot_y=True, squared_pred=True)
    ce_loss = nn.CrossEntropyLoss(weight=loss_weight.to(device))  # 增加类别权重
    # 初始化记录器
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)

            # 组合损失函数
            loss_dice = dice_loss(outputs, labels.unsqueeze(1))
            loss_ce = ce_loss(outputs, labels)
            loss = 0.6 * loss_dice + 0.4 * loss_ce  # 调整权重组合

            loss.backward()
            optimizer.step()
            # 累计指标
            epoch_train_loss += loss.item()

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                # 计算损失
                loss_dice = dice_loss(outputs, labels.unsqueeze(1))
                loss_ce = ce_loss(outputs, labels)
                loss = 0.6 * loss_dice + 0.4 * loss_ce
                epoch_val_loss += loss.item()

                # 计算准确率
                predicted = torch.argmax(outputs, dim=1)
                total_val = labels.numel()
                correct_val = (predicted == labels).sum().item()

            # 记录指标
            val_loss = epoch_val_loss
            val_acc = correct_val / total_val
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # 每 64 个 epoch 保存一次 checkpoint
            if (epoch + 1) % 64 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
                print(f"Checkpoint saved at epoch {epoch + 1}")


            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}\n")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.savefig(train_img)
    plt.close()


def evaluate(model, test_loader):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    test_data_npy = []
    test_label_npy = []
    test_preds_npy = []
    class_names = ['Background', 'Resting', "ver_eyem", "hor_eyem", "blink",
                   "hor_headm", "ver_headm", "tongue","chew", "swallow",
                   "eyebrow", "blink_hor_headm", "blink_ver_headm","blink_eyebrow",
                   "tongue_eyebrow", "swallow_eyebrow"]

    confusion_matrix = np.zeros((16, 16), dtype=int)

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            # 保存到列表中
            test_data_npy.append(data.cpu().numpy())  # shape: [64, 34, 625]
            test_label_npy.append(labels.cpu().numpy())  # shape: [64, 34, 625] (假设是逐点标签)
            test_preds_npy.append(preds.cpu().numpy())  # shape: [64, 34, 625]

            for true, pred in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[true.item(), pred.item()] += 1   #横轴为真实标签，纵轴为预测标签

    # 保存为 .npy 文件
    np.save('check_data.npy', test_data_npy)
    np.save('check_labels.npy', test_label_npy)
    np.save('check_preds.npy', test_preds_npy)
    np.save(confusion_matrix_npy_path, confusion_matrix)

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(15, 12))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='plasma',
                xticklabels=class_names, yticklabels=class_names
                )# vmin=0, vmax=np.max(confusion_matrix) / 500
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    # 打印分类报告并保存到文件
    with open(report_path, 'w') as file:
        # 控制台和文件同时输出标题
        print("\nClassification Report:")
        file.write("Classification Report:\n")

        for i in range(16):
            # 计算指标
            precision = confusion_matrix[i, i] / (confusion_matrix[:, i].sum() + 1e-8)
            recall = confusion_matrix[i, i] / (confusion_matrix[i, :].sum() + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            # 生成报告行
            report_line = f"{class_names[i]:<10} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"

            # 输出到控制台和文件
            print(report_line)
            file.write(report_line + '\n')

# 创建 DataLoader（添加pin_memory优化）
dataset_train = EEGDataset(train_data, train_labels)
dataset_val = EEGDataset(val_data, val_labels)
dataset_test = EEGDataset(test_data, test_labels)

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset_val, batch_size=64, pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=64, pin_memory=True)

# 初始化模型
model = UNetEEG().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 添加正则化

# 训练和评估
# train_model(model, train_loader, val_loader, optimizer)
evaluate(model, test_loader)