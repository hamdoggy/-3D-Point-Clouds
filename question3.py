import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from glob import glob
import os


# 解析点云数据
def load_point_cloud(bin_file):
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape(-1, 4)  # 假设每个点包含 x, y, z 和强度
    return points


# 解析标签文件
def load_labels(label_file):
    labels = np.fromfile(label_file, dtype=np.uint32)
    instance_ids = labels >> 16
    class_labels = labels & 0xFFFF
    class_labels = class_labels.astype(np.int64)
    return instance_ids, class_labels


# 预处理数据
def preprocess_data(point_cloud, instance_ids, class_labels):
    data = pd.DataFrame(point_cloud, columns=['x', 'y', 'z', 'intensity'])
    data['instance_id'] = instance_ids
    data['class_label'] = class_labels
    data['distance'] = np.linalg.norm(data[['x', 'y', 'z']], axis=1)
    return data


# 定义 PointNet 模型（保留到 fc3）
class PointNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained_mlp_path=None):
        super(PointNetFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)

        if pretrained_mlp_path:
            self.load_pretrained_mlp(pretrained_mlp_path)

    def load_pretrained_mlp(self, path):
        pretrained_mlp = torch.load(path)
        self.fc1.weight.data = pretrained_mlp['fc1.weight']
        self.fc1.bias.data = pretrained_mlp['fc1.bias']
        self.fc2.weight.data = pretrained_mlp['fc2.weight']
        self.fc2.bias.data = pretrained_mlp['fc2.bias']
        self.fc3.weight.data = pretrained_mlp['fc3.weight']
        self.fc3.bias.data = pretrained_mlp['fc3.bias']

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return x


# 屏蔽部分标签，模拟未标注数据
def mask_labels(class_labels, mask_ratio=0.5):
    masked_labels = class_labels.copy()
    num_masked = int(len(class_labels) * mask_ratio)
    mask_indices = np.random.choice(len(class_labels), num_masked, replace=False)
    masked_labels[mask_indices] = -1  # 使用-1作为未标注的标签
    return masked_labels, mask_indices


# 使用特征余弦距离进行标签传播
def label_propagation(features, masked_labels, num_classes, max_iterations=100, tolerance=1e-3):
    prev_labels = np.copy(masked_labels)
    for iteration in range(max_iterations):
        avg_features = []
        for cls in range(num_classes):
            class_features = features[masked_labels == cls]
            if len(class_features) > 0:
                avg_features.append(class_features.mean(axis=0))
            else:
                avg_features.append(np.zeros(features.shape[1]))  # 若无数据点则用零向量代替

        avg_features = np.array(avg_features)
        similarities = cosine_similarity(features, avg_features)
        propagated_labels = np.argmax(similarities, axis=1)

        # 更新标签并计算变化比例
        masked_labels = np.where(masked_labels == -1, propagated_labels, masked_labels)
        label_change = np.mean(masked_labels != prev_labels)
        prev_labels = np.copy(masked_labels)

        print(f"Iteration {iteration + 1}, Label Change: {label_change:.6f}")

        if label_change < tolerance:
            print("Convergence reached.")
            break

    return masked_labels


def main(base_dir, sequences, batch_size=2048, num_workers=16, mask_ratio=0.5, max_iterations=100, tolerance=1e-3, pretrained_mlp_path=None):
    data_files = load_all_files(base_dir, sequences)

    all_train_data = []
    for bin_file, label_file in data_files:
        point_cloud = load_point_cloud(bin_file)
        instance_ids, class_labels = load_labels(label_file)
        data = preprocess_data(point_cloud, instance_ids, class_labels)
        all_train_data.append(data)

    all_train_data = pd.concat(all_train_data, ignore_index=True)

    unique_labels = np.unique(all_train_data['class_label'])
    print(f"Unique class labels: {unique_labels}")
    print(f"Max class label: {unique_labels.max()}")

    num_classes = unique_labels.max() + 1
    print(f"Number of classes: {num_classes}")

    X = all_train_data[['x', 'y', 'z', 'intensity', 'distance']].values
    y = all_train_data['class_label'].values

    # 屏蔽部分标签
    masked_labels, mask_indices = mask_labels(y, mask_ratio)

    # 创建数据集和数据加载器
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(masked_labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化特征提取模型
    feature_extractor = PointNetFeatureExtractor(pretrained_mlp_path=pretrained_mlp_path).to(device)
    feature_extractor.eval()  # 因为只是提取特征，不需要训练

    with torch.no_grad():
        features = []
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            features.append(outputs.cpu().numpy())

    features = np.concatenate(features, axis=0)

    # 特征归一化
    features = normalize(features)

    # 标签传播
    propagated_labels = label_propagation(features, masked_labels, num_classes, max_iterations, tolerance)

    # 计算标签传播的准确率
    accuracy = accuracy_score(y[mask_indices], propagated_labels[mask_indices])
    print(f"Label Propagation Accuracy: {accuracy:.4f}")

    return feature_extractor


def load_all_files(base_dir, sequences):
    bin_files = []
    label_files = []
    for seq in sequences:
        seq_bin_files = sorted(glob(os.path.join(base_dir, f'{seq}/velodyne/*.bin')))
        seq_label_files = sorted(glob(os.path.join(base_dir, f'{seq}/labels/*.label')))
        bin_files.extend(seq_bin_files)
        label_files.extend(seq_label_files)

    assert len(bin_files) == len(label_files), "The number of bin files and label files must be the same."

    return list(zip(bin_files, label_files))


if __name__ == '__main__':
    sequences = ['00', '01', '02', '03', '04', '05']
    pretrained_mlp_path = 'mlp_FeatureExtractor.pth'  # 预训练模型路径
    feature_extractor = main('./sequences', sequences, batch_size=2048, num_workers=16, mask_ratio=0.5,
                             max_iterations=100, tolerance=1e-3, pretrained_mlp_path=pretrained_mlp_path)
