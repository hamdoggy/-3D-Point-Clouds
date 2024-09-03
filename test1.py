import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

label_descriptions = {
    0: "unground",
    22: "ground"  # class definition
}

# 定义 PointNet 类
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

# 加载模型的函数
def load_model(model_path):
    model = PointNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
    output_html = '000003_segmented.html'

    # 加载模型
    model = load_model(model_path)
    model.eval()

    # 加载点云数据
    point_cloud = load_point_cloud(bin_file)
    data = pd.DataFrame(point_cloud, columns=['x', 'y', 'z', 'intensity'])
    data['distance'] = np.linalg.norm(data[['x', 'y', 'z']], axis=1)

    # 加载cross_points.txt中的阈值
    cross_points = np.loadtxt("cross_points.txt")
    H = cross_points[0]  # 假设cross_points.txt中有一个值-1.41

    # 准备数据，进行预测
    inputs = torch.tensor(data[['x', 'y', 'z', 'intensity', 'distance']].values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        softmax_outputs = torch.softmax(outputs, dim=1)
        _, predicted_labels = torch.max(softmax_outputs, 1)
        prob_diff = torch.abs(softmax_outputs[:, 0] - softmax_outputs[:, 1])

    # 根据概率差和阈值标签进行最终预测
    final_labels = []
    for i in range(len(data)):
        if prob_diff[i] > 0.2:
            final_labels.append(predicted_labels[i].item())
        else:
            if data.iloc[i]['z'] > H:
                final_labels.append(1)  # z轴坐标大于H的阈值标签付标签1
            else:
                final_labels.append(22)  # z轴坐标小于H的阈值标签付标签22

    # 可视化并保存HTML文件
    visualize_point_cloud_with_labels(point_cloud, final_labels, output_html)
