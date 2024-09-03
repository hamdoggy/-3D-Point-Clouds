import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

label_descriptions = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",  # standing sign
    11: "traffic sign 2",  # hanging sign
    12: "traffic sign 3",  # high/big hanging sign
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground"  # class definition
}

# 定义 PointNet 类
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
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

# 加载模型的函数
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

# 定义加载点云数据的函数
def load_point_cloud(bin_file):
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape(-1, 4)  # 假设每个点包含 x, y, z 和强度
    return points

# 可视化点云和预测的分类
def visualize_point_cloud_with_labels(point_cloud, class_labels, output_file):
    df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z', 'intensity'])
    df['class'] = class_labels  # 保持类别为数字
    df['class_description'] = df['class'].map(label_descriptions)  # 映射数字到描述

    # 使用 Plotly Express 创建 3D 散点图
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='class_description', opacity=0.7)
    fig.update_traces(marker=dict(size=2))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        legend=dict(
            title=dict(font=dict(size=20)),
            font=dict(size=15),
            itemsizing='constant'
        )
    )

    # 保存 HTML 文件
    fig.write_html(output_file)

# 主执行部分
if __name__ == '__main__':
    model_path = 'trained_model.pth'
    bin_file = 'sequences/03/velodyne/000002.bin'
    output_html = '000004_segmented.html'

    model = load_model(model_path)  # 加载模型
    model.eval()

    point_cloud = load_point_cloud(bin_file)  # 加载点云数据
    data = pd.DataFrame(point_cloud, columns=['x', 'y', 'z', 'intensity'])
    data['distance'] = np.linalg.norm(data[['x', 'y', 'z']], axis=1)

    # 准备数据，进行预测
    inputs = torch.tensor(data[['x', 'y', 'z', 'intensity', 'distance']].values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)

    # 可视化并保存HTML文件
    visualize_point_cloud_with_labels(point_cloud, predicted_labels.numpy(), output_html)
