import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from glob import glob
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


# 解析点云数据
def parse_point_cloud(bin_file):
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape(-1, 4)  # 假设每个点包含 x, y, z 和强度
    return points


# 解析标签文件
def parse_labels(label_file):
    labels = np.fromfile(label_file, dtype=np.uint32)
    instance_ids = labels >> 16
    class_labels = labels & 0xFFFF
    class_labels = class_labels.astype(np.int64)
    return instance_ids, class_labels


# 预处理数据
def preprocess_point_cloud(point_cloud, instance_ids, class_labels):
    data_frame = pd.DataFrame(point_cloud, columns=['x', 'y', 'z', 'intensity'])
    data_frame['instance_id'] = instance_ids
    data_frame['class_label'] = class_labels
    data_frame['distance'] = np.linalg.norm(data_frame[['x', 'y', 'z']], axis=1)
    return data_frame


# 定义 PointNet 模型
class PointCloudNet(nn.Module):
    def __init__(self, num_classes):
        super(PointCloudNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


# 训练模型
def train_point_cloud_net(data_loader, num_classes, num_epochs, device, learning_rate=0.005):
    model = PointCloudNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/training')

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(
                tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log loss every 10 batches
            if batch_idx % 10 == 0:
                writer.add_scalar('Loss/Batch', loss.item(), epoch * len(data_loader) + batch_idx)

        # Log average loss for this epoch
        avg_loss = running_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    writer.close()
    print("Training completed.")

    # Plotting the loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig('training_loss.png')  # Save the figure
    plt.show()  # Display the figure

    return model


# 主函数
def main_process(bin_directory, label_directory, epochs, batch_size=4096, num_workers=96, output_directory='predict_frames'):
    bin_files = sorted(glob(os.path.join(bin_directory, '*.bin')))
    label_files = sorted(glob(os.path.join(label_directory, '*.label')))

    assert len(bin_files) == len(label_files), "The number of bin files and label files must be the same."

    data_files = list(zip(bin_files, label_files))
    train_size = int(0.8 * len(data_files))
    train_files = data_files[:train_size]
    test_files = data_files[train_size:]

    all_train_data = []
    for bin_file, label_file in train_files:
        point_cloud = parse_point_cloud(bin_file)
        instance_ids, class_labels = parse_labels(label_file)
        data_frame = preprocess_point_cloud(point_cloud, instance_ids, class_labels)
        all_train_data.append(data_frame)

    all_train_data = pd.concat(all_train_data, ignore_index=True)

    unique_labels = np.unique(all_train_data['class_label'])
    print(f"Unique class labels: {unique_labels}")
    print(f"Max class label: {unique_labels.max()}")

    num_classes = unique_labels.max() + 1
    print(f"Number of classes: {num_classes}")

    X = all_train_data[['x', 'y', 'z', 'intensity', 'distance']].values
    y = all_train_data['class_label'].values

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = train_point_cloud_net(train_data_loader, num_classes, epochs, device)

    print("Saving trained model...")
    torch.save(model, 'trained_model.pth')

    return model


if __name__ == '__main__':
    model = main_process('./sequences/00/velodyne', './sequences/00/labels', epochs=20, batch_size=64,
                         num_workers=16, output_directory='predict_frames')
